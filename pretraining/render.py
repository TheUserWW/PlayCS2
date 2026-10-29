"""Explicit POV import and human-assisted capture. Never fabricate frames/ticks."""

import csv
import math
from pathlib import Path

from .io import file_hash, read_json, write_json


def plan(audit_dir, output):
    audit=read_json(Path(audit_dir)/'audit.json')
    if not audit['usable']:
        raise ValueError('Audit has no usable action labels')
    jobs=[{'player':p,'interval':interval,'status':'awaiting_pov_frames'}
          for p,q in audit['quality'].items() if q['commands_usable'] for interval in audit['intervals']]
    result={'schema':'cs2-render-plan-v1','audit':str(Path(audit_dir).resolve()),'audit_hash':audit['hash'],
            'demo':audit['demo'],'demo_sha256':audit['demo_sha256'],'map':audit['header']['map_name'],
            'jobs':jobs,'instructions':[
                'Play the demo in CS2, select a fixed first-person player, disable x-ray and observer panels.',
                'Export RGB frames and a frames.csv with player,tick,image columns; ticks must be measured, not inferred from wall time.',
                'Use import-frames with provenance.json after visually verifying POV/overlays/alignment.',
                'capture-frame can save one manually paused/verified demo tick; it never advances or controls the game.']}
    write_json(output,result)
    return result


def import_frames(audit_dir, csv_path, provenance_path, output):
    import cv2
    audit=read_json(Path(audit_dir)/'audit.json')
    provenance=read_json(provenance_path)
    required={'demo_sha256':audit['demo_sha256'],'xray':False,'observer_panel':False,'pov_verified':True,'alignment_verified':True}
    for key,value in required.items():
        if provenance.get(key)!=value:
            raise ValueError(f'POV provenance requires {key}={value!r}')
    kind=provenance.get('kind')
    if kind not in ('cs2_pov','synthetic_self_test'):
        raise ValueError('kind must be cs2_pov or explicitly synthetic_self_test')
    if provenance.get('tick_source') not in ('engine_export','verified_demo_overlay','manual_paused_tick'):
        raise ValueError('A measured demo tick source is required; wall-clock estimates are rejected')
    csv_path=Path(csv_path).resolve()
    frames=[]
    seen=set()
    with csv_path.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            player=str(row['player'])
            tick=float(row['tick'])
            if not math.isfinite(tick) or player not in audit['selected_players']:
                raise ValueError('Invalid player or tick in frame manifest')
            if not any(i['start_tick']<=tick<=i['end_tick'] for i in audit['intervals']):
                raise ValueError(f'Frame tick {tick} is outside audited live intervals')
            if (player,tick) in seen:
                raise ValueError('Duplicate player/tick frame')
            seen.add((player,tick))
            path=(csv_path.parent/row['image']).resolve()
            image=cv2.imread(str(path),cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f'Unreadable POV image: {path}')
            frames.append({'player':player,'tick':tick,'image':str(path),'sha256':file_hash(path)})
    if not frames:
        raise ValueError('No frames imported')
    result={'schema':'cs2-frames-v1','audit_hash':audit['hash'],'provenance':provenance,
            'frames':sorted(frames,key=lambda r:(r['player'],r['tick']))}
    write_json(output,result)
    return result


def capture_frame(output, *, region):
    """Capture the user-verified paused POV. No input injection or fake timestamps."""
    import cv2
    import mss
    import numpy as np
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with mss.mss() as screen:
        image=np.asarray(screen.grab(region))
    if not cv2.imwrite(str(output),image[:,:,:3]):
        raise OSError(f'Could not write {output}')
    return {'image':str(output.resolve()),'sha256':file_hash(output)}


def import_video(audit_dir,video,csv_path,provenance,output_dir):
    """Extract frames using measured frame_index/tick pairs, never a wall-time guess."""
    import cv2
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    with Path(csv_path).open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    requested={}
    for row in rows:
        index=int(row['frame_index'])
        if index<0 or index in requested:
            raise ValueError('Video frame indices must be unique nonnegative integers')
        requested[index]=row
    if not requested: raise ValueError('No measured video frame/tick pairs supplied')
    capture=cv2.VideoCapture(str(video))
    if not capture.isOpened(): raise ValueError(f'Cannot open POV video {video}')
    converted=[];index=0
    try:
        while requested:
            ok,frame=capture.read()
            if not ok: raise ValueError('POV video ended before all timestamped frames were found')
            if index in requested:
                row=requested.pop(index);filename=f'frame-{index:08d}.jpg'
                if not cv2.imwrite(str(output_dir/filename),frame): raise OSError('Frame extraction failed')
                converted.append({'player':row['player'],'tick':row['tick'],'image':filename})
            index+=1
    finally:
        capture.release()
    with (output_dir/'frames.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['player','tick','image']);writer.writeheader();writer.writerows(converted)
    return import_frames(audit_dir,output_dir/'frames.csv',provenance,output_dir/'frames.json')
