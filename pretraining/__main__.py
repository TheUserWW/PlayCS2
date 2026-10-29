"""Run `python -m pretraining --help` for the implemented pipeline stages."""

import argparse
import json
from pathlib import Path
import sys

from .io import read_json, write_json


def arguments():
    parser=argparse.ArgumentParser(description='CS2 demo audit, POV datasets and supervised pretraining')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('scan',help='Hash demos and freeze match-group splits')
    p.add_argument('--root',default='demos');p.add_argument('--out',default='datasets/cs2/catalog.json');p.add_argument('--seed',type=int,default=1337)
    p=sub.add_parser('audit',help='Extract and audit two players / three complete live rounds')
    p.add_argument('--demo',required=True);p.add_argument('--catalog',default='datasets/cs2/catalog.json');p.add_argument('--out',default='datasets/cs2/pilot')
    p.add_argument('--players',type=int,default=2);p.add_argument('--rounds',type=int,default=3);p.add_argument('--max-ticks',type=int,default=30000)
    p=sub.add_parser('render-plan',help='Create fixed POV tasks; does not invent RGB frames')
    p.add_argument('--audit',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('import-frames',help='Validate measured POV frame ticks and provenance')
    p.add_argument('--audit',required=True);p.add_argument('--csv',required=True);p.add_argument('--provenance',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('import-video',help='Extract POV frames using a measured frame_index/tick CSV')
    p.add_argument('--audit',required=True);p.add_argument('--video',required=True);p.add_argument('--csv',required=True);p.add_argument('--provenance',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('capture-frame',help='Capture one user-verified paused demo frame; no game input')
    p.add_argument('--out',required=True);p.add_argument('--left',type=int,default=0);p.add_argument('--top',type=int,default=0)
    p.add_argument('--width',type=int,required=True);p.add_argument('--height',type=int,required=True)
    p=sub.add_parser('build',help='Create masked episodes from audited commands and verified POV frames')
    p.add_argument('--audit',required=True);p.add_argument('--frames',required=True);p.add_argument('--out',required=True)
    p.add_argument('--hz',type=float,default=20);p.add_argument('--gsi-delay',type=float,default=.1)
    p=sub.add_parser('combine',help='Combine episodes while forbidding match-group leakage')
    p.add_argument('--manifest',action='append',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('prepare',help='Scan/audit/plan; build if measured POV frames have been supplied')
    p.add_argument('--demo',required=True);p.add_argument('--root',default='demos');p.add_argument('--workspace',default='datasets/cs2')
    p.add_argument('--players',type=int,default=2);p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--frame-csv');p.add_argument('--provenance')
    p=sub.add_parser('train',help='Train with strict resume; requires separate validation match unless --smoke')
    p.add_argument('--manifest',required=True);p.add_argument('--run',required=True);p.add_argument('--config',default='configs/pretraining/short.json')
    p.add_argument('--device',choices=['cpu','cuda']);p.add_argument('--resume');p.add_argument('--init-from');p.add_argument('--smoke',action='store_true')
    p.add_argument('--stop-after',type=int,help='Stop after this many complete updates; preserve total schedule for resume')
    p=sub.add_parser('evaluate',help='Evaluate immutable dataset split')
    p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--split',choices=['train','val','test'],default='test')
    p.add_argument('--device',default='cuda');p.add_argument('--out',required=True);p.add_argument('--allow-diagnostic',action='store_true')
    p=sub.add_parser('export',help='Export inference package; reject smoke weights by default')
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--allow-diagnostic',action='store_true')
    p=sub.add_parser('status',help='Show pipeline state without starting training or the game')
    p.add_argument('--workspace',default='datasets/cs2')
    return parser.parse_args()


def execute(a):
    if a.command=='scan':
        from .catalog import scan
        d=scan(a.root,a.out,seed=a.seed)
        return {'catalog':str(Path(a.out).resolve()),'demos':len(d['demos']),'groups':len({r['group'] for r in d['demos']}),'hash':d['hash']}
    if a.command=='audit':
        from .audit import audit
        d=audit(a.demo,a.out,players=a.players,rounds=a.rounds,catalog=read_json(a.catalog),max_ticks=a.max_ticks)
        if not d['usable']: raise ValueError('Usercmd quality checks failed; inspect audit.json')
        return {'audit':str(Path(a.out).resolve()),'quality':d['quality']}
    if a.command=='render-plan':
        from .render import plan
        d=plan(a.audit,a.out);return {'output':a.out,'jobs':len(d['jobs']),'status':'awaiting_verified_pov_frames'}
    if a.command=='capture-frame':
        from .render import capture_frame
        return capture_frame(a.out,region={'left':a.left,'top':a.top,'width':a.width,'height':a.height})
    if a.command=='import-frames':
        from .render import import_frames
        d=import_frames(a.audit,a.csv,a.provenance,a.out);return {'output':a.out,'frames':len(d['frames']),'kind':d['provenance']['kind']}
    if a.command=='import-video':
        from .render import import_video
        d=import_video(a.audit,a.video,a.csv,a.provenance,a.out);return {'output':a.out,'frames':len(d['frames']),'kind':d['provenance']['kind']}
    if a.command=='build':
        from .dataset import build
        d=build(a.audit,a.frames,a.out,hz=a.hz,gsi_delay=a.gsi_delay)
        return {'manifest':str((Path(a.out)/'manifest.json').resolve()),'episodes':len(d['episodes']),'coverage':d['coverage'],'diagnostic_only':d['diagnostic_only']}
    if a.command=='combine':
        from .dataset import combine
        d=combine(a.manifest,a.out);return {'manifest':a.out,'episodes':len(d['episodes']),'hash':d['hash']}
    if a.command=='prepare':
        from .catalog import scan
        from .audit import audit
        from .render import plan,import_frames
        from .dataset import build
        workspace=Path(a.workspace);workspace.mkdir(parents=True,exist_ok=True)
        catalog=scan(a.root,workspace/'catalog.json')
        pilot=workspace/'pilot'
        audit(a.demo,pilot,players=a.players,rounds=a.rounds,catalog=catalog)
        plan(pilot,workspace/'render-plan.json')
        if not a.frame_csv or not a.provenance:
            write_json(workspace/'status.json',{'stage':'awaiting_pov_frames','audit':str(pilot.resolve()),'plan':str((workspace/'render-plan.json').resolve())})
            print('Commands audited. Supply measured POV frames with --frame-csv and --provenance; training has NOT started.',file=sys.stderr)
            return {'status':'awaiting_pov_frames','exit_code':4,'plan':str((workspace/'render-plan.json').resolve())}
        import_frames(pilot,a.frame_csv,a.provenance,workspace/'frames.json')
        d=build(pilot,workspace/'frames.json',workspace/'pilot-dataset')
        write_json(workspace/'status.json',{'stage':'dataset_ready','hash':d['hash']})
        return {'status':'dataset_ready','manifest':str((workspace/'pilot-dataset/manifest.json').resolve())}
    if a.command=='train':
        from Transformer import CS2PolicyConfig
        from .training import train
        cfg=read_json(a.config)
        if a.device: cfg.setdefault('training',{})['device']=a.device
        d=train(a.manifest,a.run,config=cfg.get('training'),model_config=CS2PolicyConfig(**cfg.get('model',{})),resume=a.resume,init_from=a.init_from,smoke=a.smoke,stop_after=a.stop_after)
        return {k:v for k,v in d.items() if k!='history'}
    if a.command=='evaluate':
        from .training import evaluate
        d=evaluate(a.checkpoint,a.manifest,split=a.split,device=a.device,allow_diagnostic=a.allow_diagnostic);write_json(a.out,d);return d
    if a.command=='export':
        from .training import export
        return export(a.checkpoint,a.out,allow_diagnostic=a.allow_diagnostic)
    if a.command=='status':
        root=Path(a.workspace)
        return {'workspace':str(root.resolve()),'status':read_json(root/'status.json') if (root/'status.json').exists() else None,
                'audits':[str(p.resolve()) for p in root.rglob('audit.json')],
                'datasets':[str(p.resolve()) for p in root.rglob('manifest.json')]}


def main():
    try:
        result=execute(arguments())
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
        return result.get('exit_code',130 if result.get('interrupted') else 0)
    except KeyboardInterrupt:
        print('Interrupted; resume the last completed atomic checkpoint.',file=sys.stderr);return 130
    except (ValueError,FileNotFoundError) as error:
        print(f'Data/configuration error: {error}',file=sys.stderr);return 3
    except (RuntimeError,OSError) as error:
        print(f'Pipeline error: {error}',file=sys.stderr);return 5


if __name__=='__main__':
    sys.exit(main())
