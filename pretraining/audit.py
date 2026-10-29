"""Bounded raw-tick extraction and fail-closed user-command quality checks."""

from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics

from .io import file_hash, object_hash, read_json, write_json
from .parser import query

FIELDS = ['X','Y','yaw','pitch','game_time','health','is_alive','team_num','total_rounds_played',
          'armor_value','balance','has_helmet','has_defuser','active_weapon_name',
          'active_weapon_ammo','total_ammo_left','is_freeze_period','is_warmup_period',
          'is_bomb_planted','is_walking','usercmd_viewangle_x','usercmd_viewangle_y',
          'usercmd_buttonstate_1','usercmd_forward_move','usercmd_left_move',
          'usercmd_mouse_dx','usercmd_mouse_dy','usercmd_weapon_select']


def finite(value):
    return isinstance(value, (int,float)) and math.isfinite(value)


def quality(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row['steamid'])].append(row)
    report = {}
    for player, values in groups.items():
        values.sort(key=lambda r:r['tick'])
        alive = [r for r in values if r.get('is_alive') and not r.get('is_freeze_period') and not r.get('is_warmup_period')]
        fields = {}
        for name in ['usercmd_forward_move','usercmd_left_move','usercmd_buttonstate_1','usercmd_viewangle_x','usercmd_viewangle_y']:
            present = [r[name] for r in alive if r.get(name) is not None]
            fields[name] = {'coverage': len(present)/max(1,len(alive)), 'changes': sum(a!=b for a,b in zip(present,present[1:]))}
        has_motion = sum(abs(b.get('X',0)-a.get('X',0))+abs(b.get('Y',0)-a.get('Y',0)) > 1 for a,b in zip(alive,alive[1:]))
        moves = fields['usercmd_forward_move']['changes']+fields['usercmd_left_move']['changes']
        command_good = len(alive)>=32 and all(v['coverage']>=.98 for v in fields.values()) and moves>=4 and fields['usercmd_buttonstate_1']['changes']>=4
        discrepancies = [abs((r['usercmd_viewangle_y']-r['yaw']+180)%360-180) for r in alive if finite(r.get('usercmd_viewangle_y')) and finite(r.get('yaw'))]
        angle_good = command_good and bool(discrepancies) and statistics.median(discrepancies) < 5
        walk_pairs = [(bool(int(r['usercmd_buttonstate_1']) & ((1<<16)|(1<<17))),r['is_walking']) for r in alive if r.get('usercmd_buttonstate_1') is not None and isinstance(r.get('is_walking'),bool)]
        walk_good = command_good and len({p[0] for p in walk_pairs})==2 and sum(a==b for a,b in walk_pairs)/max(1,len(walk_pairs))>=.98
        report[player] = {'rows': len(values), 'alive_rows':len(alive),'position_changes':has_motion,'fields':fields,
                          'commands_usable':command_good,'angles_usable':angle_good,'walk_usable':walk_good,
                          'view_disagreement_median_deg':statistics.median(discrepancies) if discrepancies else None,
                          'reason':None if command_good else 'Insufficient or suspicious command coverage/variation; do not fabricate no-op labels'}
    return report


def load_rows(directory):
    with gzip.open(Path(directory)/'ticks.jsonl.gz','rt',encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def audit(demo, output, *, players=2, rounds=3, catalog=None, max_ticks=30000):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    demo = Path(demo).resolve()
    source_hash = file_hash(demo)
    request={'demo_sha256':source_hash,'players':players,'rounds':rounds,'max_ticks':max_ticks,'fields':FIELDS,'parser':'@laihoe/demoparser2@0.42.0'}
    if (output/'audit.json').exists():
        previous=read_json(output/'audit.json')
        if previous.get('request')==request and file_hash(output/'ticks.jsonl.gz')==previous['ticks_sha256']:
            return previous
        if previous.get('request') is not None:
            raise ValueError('Audit directory belongs to different inputs/settings; use a new output directory')
    info = query(demo,'header')
    events = query(demo,'events',events=['round_start','round_freeze_end','round_end','weapon_fire','player_death'])
    ends = sorted(e['tick'] for e in events if e['event_name']=='round_end')
    starts = sorted({e['tick'] for e in events if e['event_name']=='round_freeze_end'})
    intervals=[]
    for start in starts:
        end=next((x for x in ends if x>start),None)
        if end is not None:
            intervals.append({'start_tick':start,'end_tick':end})
    intervals=intervals[:rounds]
    if not intervals:
        raise ValueError('No complete live rounds found')
    ticks=[t for interval in intervals for t in range(interval['start_tick'],interval['end_tick']+1)]
    if len(ticks)>max_ticks:
        raise ValueError(f'Selected {len(ticks)} ticks exceeds --max-ticks={max_ticks}; reduce rounds or raise explicit budget')
    selected=[str(p['steamid']) for p in info['players'][:players]]
    if not selected:
        raise ValueError('No identifiable players found')
    rows=query(demo,'ticks',fields=FIELDS,ticks=ticks,players=selected)
    if not rows:
        raise ValueError('No player ticks returned')
    raw=output/'ticks.jsonl.gz'
    temporary=output/'ticks.jsonl.gz.tmp'
    with gzip.open(temporary,'wt',encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
    temporary.replace(raw)
    record=next((r for r in (catalog or {}).get('demos',[]) if r['sha256']==source_hash),None)
    if record is None:
        raise ValueError('Demo must be registered in catalog before audit')
    report={'schema':'cs2-audit-v1','request':request,'parser':'@laihoe/demoparser2@0.42.0','demo':str(demo),
            'demo_sha256':source_hash,'group':record['group'],'split':record['split'],
            'catalog_hash':catalog['hash'],'header':info['header'],'players':info['players'],
            'selected_players':selected,'intervals':intervals,'ticks_sha256':file_hash(raw),
            'events':events,'quality':quality(rows),
            'limitations':['Per-tick exposed usercmd values are not proof of subtick-perfect input timing.',
                           'No POV frames have been rendered by this audit.']}
    report['usable']=any(p['commands_usable'] for p in report['quality'].values())
    report['hash']=object_hash(report)
    write_json(output/'audit.json',report)
    return report
