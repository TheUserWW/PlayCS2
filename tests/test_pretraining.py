import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from Transformer import CS2PolicyConfig
from cs2.actions import ACTION_DIM, CS2Action, YAW_BINS, PITCH_BINS, angle_target, decode_angle, encode_previous_action
from pretraining.catalog import scan
from pretraining.dataset import build, combine, DemoWindows
from pretraining.io import file_hash, object_hash, read_json, write_json
from pretraining.labels import command_action
from pretraining.render import import_frames
from pretraining.training import train, evaluate, export, load_export


def fixture(base,split,*,group=None):
    """Explicitly synthetic fixtures. Never treated as recorded professional images."""
    directory=Path(base)/split;directory.mkdir()
    source=directory/'fixture.dem';source.write_bytes(('not-a-demo:'+split).encode())
    audit_dir=directory/'audit';audit_dir.mkdir()
    rows=[]
    for tick in range(161):
        rows.append({'tick':tick,'steamid':'fixture-player','game_time':1+tick/64,
                     'is_alive':True,'is_freeze_period':False,'is_warmup_period':False,
                     'health':100,'team_num':2,'total_rounds_played':0,
                     'usercmd_forward_move':1.,'usercmd_left_move':0.,
                     'usercmd_buttonstate_1':'8','usercmd_viewangle_x':0.,'usercmd_viewangle_y':tick*.1})
    with gzip.open(audit_dir/'ticks.jsonl.gz','wt',encoding='utf-8') as stream:
        for row in rows: stream.write(json.dumps(row)+'\n')
    audit={'usable':True,'demo':str(source),'demo_sha256':file_hash(source),'group':group or 'fixture-'+split,'split':split,
           'selected_players':['fixture-player'],'intervals':[{'start_tick':0,'end_tick':160}],
           'ticks_sha256':file_hash(audit_dir/'ticks.jsonl.gz'),
           'quality':{'fixture-player':{'commands_usable':True,'angles_usable':True,'walk_usable':False}}}
    audit['hash']=object_hash(audit);write_json(audit_dir/'audit.json',audit)
    frames=directory/'frames';frames.mkdir()
    with (directory/'frames.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['player','tick','image']);writer.writeheader()
        for i in range(48):
            image=np.full((48,80,3),80+i,dtype=np.uint8)
            path=frames/f'{i:04d}.jpg';cv2.imwrite(str(path),image)
            writer.writerow({'player':'fixture-player','tick':i*3.2,'image':f'frames/{i:04d}.jpg'})
    provenance={'kind':'synthetic_self_test','demo_sha256':audit['demo_sha256'],'xray':False,
                'observer_panel':False,'pov_verified':True,'alignment_verified':True,'tick_source':'manual_paused_tick'}
    write_json(directory/'provenance.json',provenance)
    import_frames(audit_dir,directory/'frames.csv',directory/'provenance.json',directory/'frames.json')
    build(audit_dir,directory/'frames.json',directory/'dataset')
    return directory/'dataset/manifest.json'


class ActionTests(unittest.TestCase):
    def test_angular_units_and_unknown_history(self):
        for bins in (YAW_BINS,PITCH_BINS):
            for value in (bins[0],-1.234,-.012,0.,.041,17.3,bins[-1]):
                index,residual=angle_target(value,bins)
                self.assertAlmostEqual(value,decode_angle(index,residual,bins),places=6)
        self.assertEqual(ACTION_DIM,35)
        unknown=CS2Action(forward=1,delta_yaw=20,available=(False,)*13)
        self.assertFalse(any(encode_previous_action(unknown)))

    def test_missing_movement_is_masked_and_reload_is_an_edge(self):
        before={'usercmd_buttonstate_1':'0','usercmd_viewangle_x':0.,'usercmd_viewangle_y':0.}
        row={'usercmd_buttonstate_1':str(1<<13),'usercmd_forward_move':None,'usercmd_left_move':-1.,
             'usercmd_viewangle_x':0.,'usercmd_viewangle_y':1.}
        q={'commands_usable':True,'angles_usable':True}
        action=command_action(before,[row],q)
        self.assertFalse(action.available[0]);self.assertTrue(action.available[1])
        self.assertEqual(action.strafe,2);self.assertTrue(action.buttons[3])
        self.assertFalse(command_action(row,[row],q).buttons[3])


class PipelineTests(unittest.TestCase):
    def test_catalog_splits_are_frozen_when_new_matches_arrive(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'demos'
            for i in range(5):
                p=root/f'match{i}'/'map.dem';p.parent.mkdir(parents=True);p.write_bytes(bytes([i]))
            path=Path(d)/'catalog.json'
            old=scan(root,path)
            p=root/'new-match'/'new.dem';p.parent.mkdir();p.write_bytes(b'new')
            new=scan(root,path)
            mapping={r['group']:r['split'] for r in new['demos']}
            self.assertTrue(all(mapping[r['group']]==r['split'] for r in old['demos']))

    def test_build_padding_integrity_and_diagnostic_gate(self):
        with tempfile.TemporaryDirectory() as d:
            path=fixture(d,'train')
            with self.assertRaises(ValueError): DemoWindows(path,split='train')
            dataset=DemoWindows(path,split='train',length=32,allow_diagnostic=True)
            batch=dataset[0]
            self.assertFalse(batch['previous'][0].any())
            self.assertEqual(batch['previous'].shape,(32,35))
            self.assertFalse(batch['label_mask'][...,9].any())  # drop unavailable
            self.assertFalse(batch['label_mask'][...,10].any())  # slot unavailable
            image=Path(read_json(path)['images'][0]['path']);image.write_bytes(b'corrupt')
            with self.assertRaises(ValueError): DemoWindows(path,split='train',allow_diagnostic=True)

    def test_match_group_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a=fixture(d,'train',group='same-match');b=fixture(d,'val',group='same-match')
            with self.assertRaises(ValueError): combine([a,b],Path(d)/'combined.json')

    def test_train_resume_evaluate_export_and_true_gradients(self):
        with tempfile.TemporaryDirectory() as d:
            manifest=Path(d)/'combined.json'
            combine([fixture(d,s) for s in ('train','val','test')],manifest)
            cfg={'steps':4,'length':16,'accumulate':1,'lr':.001,'cnn_lr':.001,'eval_every':2,'save_every':2,'device':'cpu','precision':'fp32'}
            mc=CS2PolicyConfig(n_embed=24,n_head=3,n_layer=1,vision_dim=16,block_size=32,dropout=0.,image_height=32,image_width=48)
            full=Path(d)/'full';resumed=Path(d)/'resumed'
            a=train(manifest,full,config=cfg,model_config=mc,smoke=True)
            train(manifest,resumed,config=cfg,model_config=mc,smoke=True,stop_after=2)
            b=train(manifest,resumed,config=cfg,model_config=mc,smoke=True,resume=resumed/'last.pt')
            self.assertEqual((a['step'],b['step']),(4,4))
            wa=torch.load(full/'last.pt',weights_only=True);wb=torch.load(resumed/'last.pt',weights_only=True)
            for key in wa['model']: torch.testing.assert_close(wa['model'][key],wb['model'][key],atol=0,rtol=0)
            self.assertLess(a['history'][-1]['train_loss'],a['history'][0]['train_loss'])
            metrics=evaluate(full/'best.pt',manifest,device='cpu',allow_diagnostic=True)
            self.assertTrue(np.isfinite(metrics['loss']))
            with self.assertRaises(ValueError): export(full/'best.pt',Path(d)/'unsafe.pt')
            target=Path(d)/'diagnostic.pt';export(full/'best.pt',target,allow_diagnostic=True)
            with self.assertRaises(ValueError): load_export(target)
            model,meta=load_export(target,allow_diagnostic=True)
            self.assertEqual(meta['kind'],'pipeline_smoke')
            self.assertFalse(model.action_availability[9]);self.assertFalse(model.action_availability[10])


if __name__=='__main__': unittest.main()
