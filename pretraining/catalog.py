import random
from pathlib import Path

from .io import file_hash, object_hash, read_json, write_json


def scan(root, output, *, seed=1337):
    root = Path(root).resolve()
    previous = read_json(output) if Path(output).is_file() else {}
    if previous and (previous.get('root')!=str(root) or previous.get('seed')!=seed):
        raise ValueError('Catalog root/seed is frozen; choose a new output for a different split')
    cached = {r['path']: r for r in previous.get('demos', [])}
    records, seen = [], set()
    for path in sorted(root.rglob('*.dem')):
        stat = path.stat()
        old = cached.get(str(path), {})
        digest = old.get('sha256') if old.get('bytes') == stat.st_size and old.get('mtime_ns') == stat.st_mtime_ns else file_hash(path)
        after=path.stat()
        if (after.st_size,after.st_mtime_ns)!=(stat.st_size,stat.st_mtime_ns):
            raise ValueError(f'Demo changed while hashing; finish downloading before scan: {path}')
        if digest in seen:
            continue
        seen.add(digest)
        records.append({'path': str(path), 'sha256': digest, 'bytes': stat.st_size,
                        'mtime_ns': stat.st_mtime_ns, 'group': str(path.relative_to(root).parent)})
    groups = sorted({r['group'] for r in records})
    if len(groups) < 3:
        raise ValueError('At least three independent match groups are needed for train/val/test splitting')
    random.Random(seed).shuffle(groups)
    n = max(1, round(len(groups) * .1))
    splits = {group: 'test' if i < n else 'val' if i < 2*n else 'train' for i, group in enumerate(groups)}
    existing={r['group']:r['split'] for r in previous.get('demos',[])}
    if existing:
        for group in groups:
            if group in existing:
                splits[group]=existing[group]
            else:
                bucket=int(object_hash([seed,group])[:8],16)%10
                splits[group]='test' if bucket==0 else 'val' if bucket==1 else 'train'
    for row in records:
        row['split'] = splits[row['group']]
    result = {'schema': 'cs2-catalog-v1', 'root': str(root), 'seed': seed, 'demos': records}
    result['hash'] = object_hash(result)
    write_json(output, result)
    return result
