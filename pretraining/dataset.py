"""Verified POV frames -> masked numeric episodes -> bounded training windows."""

from bisect import bisect_left, bisect_right
from collections import defaultdict, OrderedDict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cs2.actions import ACTION_SCHEMA_VERSION, BRANCHES, YAW_BINS, PITCH_BINS, angle_target, encode_previous_action
from cs2.state import SCHEMA_VERSION, STATE_DIM
from .audit import load_rows
from .io import file_hash, object_hash, read_json, write_json
from .labels import command_action, local_state


def build(audit_dir, frames_path, output, *, hz=20., gsi_delay=.1):
    if not 1<=hz<=128 or not 0<=gsi_delay<=2:
        raise ValueError('hz must be 1..128 and GSI delay 0..2 seconds')
    audit_dir,output=Path(audit_dir),Path(output)
    audit=read_json(audit_dir/'audit.json')
    frames=read_json(frames_path)
    if frames['audit_hash']!=audit['hash'] or file_hash(audit_dir/'ticks.jsonl.gz')!=audit['ticks_sha256']:
        raise ValueError('Audit or raw ticks changed; regenerate the frame manifest')
    if not audit['usable']:
        raise ValueError('No usable command stream')
    raw=defaultdict(list)
    for row in load_rows(audit_dir):
        raw[str(row['steamid'])].append(row)
    grouped=defaultdict(list)
    image_records=[]
    for frame in frames['frames']:
        path=Path(frame['image'])
        if file_hash(path)!=frame['sha256']:
            raise ValueError(f'POV image changed: {path}')
        image_records.append({'path':str(path),'sha256':frame['sha256'],'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns})
        grouped[frame['player']].append(frame)
    output.mkdir(parents=True,exist_ok=True)
    artifacts=[]
    coverage=np.zeros(len(BRANCHES),dtype=np.int64)
    rejected=0
    for player,images in grouped.items():
        rows=sorted(raw[player],key=lambda r:r['tick'])
        ticks=[r['tick'] for r in rows]
        images.sort(key=lambda r:r['tick'])
        chunks=[]
        chunk=[]
        previous=None
        last_tick=None
        last_round=None

        def flush():
            nonlocal chunk,previous,last_tick,last_round
            if len(chunk)>=4:
                chunks.append(chunk)
            chunk=[];previous=None;last_tick=None;last_round=None

        for a,b in zip(images,images[1:]):
            start=bisect_right(ticks,a['tick'])-1
            end=bisect_right(ticks,b['tick'])-1
            if start<0 or end<=start:
                flush();rejected+=1;continue
            before,after=rows[start],rows[end]
            t0,t1=before.get('game_time'),after.get('game_time')
            dt=t1-t0 if isinstance(t0,(int,float)) and isinstance(t1,(int,float)) else -1
            live=all(r.get('is_alive') and not r.get('is_freeze_period') and not r.get('is_warmup_period') for r in rows[start:end+1])
            round_id=before.get('total_rounds_played')
            if not live or not .5/hz<=dt<=1.5/hz or round_id!=after.get('total_rounds_played'):
                flush();rejected+=1;continue
            if last_tick!=a['tick'] or last_round!=round_id:
                flush()
            action=command_action(before,rows[start+1:end+1],audit['quality'][player])
            if not any(action.available):
                flush();rejected+=1;continue
            # Simulated 10 Hz GSI receive stream, with explicit causal delay.
            observed_time=np.floor((t0-gsi_delay)*10)/10
            times=[r['game_time'] for r in rows[max(0,start-64):start+1]]
            candidate=bisect_right(times,observed_time)-1
            state=None
            if candidate>=0:
                sr=rows[max(0,start-64)+candidate]
                if sr.get('total_rounds_played')==round_id:
                    state=local_state(sr)
            state_values=state.values if state else (0.,)*STATE_DIM
            state_mask=state.available if state else (False,)*STATE_DIM
            yi,yr=angle_target(action.delta_yaw,YAW_BINS)
            pi,pr=angle_target(action.delta_pitch,PITCH_BINS)
            chunk.append({'frame':a['image'],'time':t0,'previous':encode_previous_action(previous),
                          'state':state_values,'state_mask':state_mask,'age':max(0.,t0-observed_time),
                          'fresh':state is not None,'start':previous is None,
                          'forward':action.forward,'strafe':action.strafe,'buttons':action.buttons,
                          'weapon':action.weapon_slot,'angles':(action.delta_yaw,action.delta_pitch),
                          'angle_bins':(yi,pi),'residual':(yr,pr),'label_mask':action.available})
            previous=action;last_tick=b['tick'];last_round=round_id
        flush()
        for index,chunk in enumerate(chunks):
            filename=f'{player}-{index:04d}.npz'
            arrays={key:np.asarray([row[key] for row in chunk]) for key in chunk[0]}
            coverage+=arrays['label_mask'].sum(axis=0).astype(np.int64)
            # Strings/numeric arrays only; never pickle data in a training artifact.
            np.savez_compressed(output/filename,**arrays)
            artifacts.append({'path':filename,'sha256':file_hash(output/filename),'steps':len(chunk),
                              'split':audit['split'],'group':audit['group'],'player':player,'demo_sha256':audit['demo_sha256']})
    if not artifacts:
        raise ValueError('No continuous verified POV episodes. Check tick alignment, player, capture rate, and live intervals.')
    result={'schema':'cs2-dataset-v2','action_schema':ACTION_SCHEMA_VERSION,'state_schema':SCHEMA_VERSION,
            'hz':hz,'gsi_delay':gsi_delay,'audit_hash':audit['hash'],'provenance':frames['provenance'],
            'diagnostic_only':frames['provenance']['kind']!='cs2_pov','episodes':artifacts,
            'images':image_records,'coverage':dict(zip(BRANCHES,coverage.tolist())),'rejected_intervals':rejected}
    result['hash']=object_hash(result)
    write_json(output/'manifest.json',result)
    return result


def combine(manifests, output):
    episodes=[];images=[];groups={};coverage=defaultdict(int);settings=None;diagnostic=False
    for source in manifests:
        source=Path(source).resolve()
        data=read_json(source)
        current=(data['schema'],data['action_schema'],data['state_schema'],data['hz'],data['gsi_delay'])
        if settings is not None and current!=settings:
            raise ValueError('Cannot combine differing dataset schemas or sample/observation rates')
        settings=current
        diagnostic|=data['diagnostic_only']
        for row in data['episodes']:
            if row['group'] in groups and groups[row['group']]!=row['split']:
                raise ValueError('Match-group leakage between splits')
            groups[row['group']]=row['split']
            episodes.append({**row,'path':str((source.parent/row['path']).resolve())})
        images.extend(data['images'])
        for k,v in data['coverage'].items(): coverage[k]+=v
    if not episodes:
        raise ValueError('No manifests to combine')
    keys=[(r['demo_sha256'],r['player'],r['sha256']) for r in episodes]
    if len(set(keys))!=len(keys):
        raise ValueError('Duplicate episodes supplied')
    result={'schema':settings[0],'action_schema':settings[1],'state_schema':settings[2],
            'hz':settings[3],'gsi_delay':settings[4],'diagnostic_only':diagnostic,
            'episodes':episodes,'images':images,'coverage':dict(coverage)}
    result['hash']=object_hash(result)
    write_json(output,result)
    return result


class DemoWindows:
    """Worker-free, finite windows with explicit padding and context-only prefix."""

    def __init__(self, manifest, *, split, length=32, training=False, allow_diagnostic=False, verify_images=True):
        import torch
        self.torch=torch
        self.path=Path(manifest).resolve();self.manifest=read_json(self.path)
        data=self.manifest
        if data['action_schema']!=ACTION_SCHEMA_VERSION or data['state_schema']!=SCHEMA_VERSION:
            raise ValueError('Dataset schema does not match the CS2 model')
        expected=data.copy();expected.pop('hash',None)
        if object_hash(expected)!=data['hash']:
            raise ValueError('Manifest fingerprint mismatch')
        if data['diagnostic_only'] and not allow_diagnostic:
            raise ValueError('Synthetic self-test data cannot be used as professional demo pretraining')
        if not 4<=length<=128:
            raise ValueError('Sequence length must be 4..128')
        self.length=length;self.warmup=max(1,length//4);self.cache=OrderedDict();self.index=[]
        self.episodes=[r for r in data['episodes'] if r['split']==split]
        for i,row in enumerate(self.episodes):
            path=self.path.parent/row['path']
            if file_hash(path)!=row['sha256']:
                raise ValueError(f'Dataset episode changed: {path}')
            stride=length//2 if training else length-self.warmup
            self.index.extend((i,start) for start in range(0,row['steps']-self.warmup,stride))
        if not self.index:
            raise ValueError(f'No usable {split} windows (length={length}); prepare a separate match for this split')
        if verify_images:
            for row in data['images']:
                path=Path(row['path']);stat=path.stat()
                if stat.st_size!=row['bytes'] or stat.st_mtime_ns!=row['mtime_ns'] or file_hash(path)!=row['sha256']:
                    raise ValueError(f'POV image changed: {path}')

    def __len__(self): return len(self.index)

    def __getitem__(self,index):
        import cv2
        episode,start=self.index[index]
        if episode not in self.cache:
            with np.load(self.path.parent/self.episodes[episode]['path'],allow_pickle=False) as data:
                self.cache[episode]={k:data[k] for k in data.files}
            while len(self.cache)>2: self.cache.popitem(last=False)
        data=self.cache[episode]
        self.cache.move_to_end(episode)
        n=min(self.length,len(data['time'])-start)
        arrays={}
        for key,values in data.items():
            if key=='frame': continue
            a=np.zeros((self.length,*values.shape[1:]),dtype=values.dtype)
            a[:n]=values[start:start+n]
            arrays[key]=a
        images=np.zeros((self.length,3,128,224),dtype=np.uint8)
        for i,path in enumerate(data['frame'][start:start+n]):
            frame=cv2.imread(str(path),cv2.IMREAD_COLOR)
            if frame is None: raise ValueError(f'Unreadable image {path}')
            images[i]=cv2.cvtColor(cv2.resize(frame,(224,128)),cv2.COLOR_BGR2RGB).transpose(2,0,1)
        valid=np.arange(self.length)<n
        loss_mask=valid & (np.arange(self.length)>=self.warmup)
        # Padded timestamps are ignored by attention, but remain finite.
        arrays['time'][n:]=arrays['time'][n-1]
        result={k:self.torch.from_numpy(v.copy()) for k,v in arrays.items()}
        result.update(images=self.torch.from_numpy(images),valid=self.torch.from_numpy(valid),loss_mask=self.torch.from_numpy(loss_mask))
        return result
