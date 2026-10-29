"""Supervised CNN/Transformer optimization, evaluation, strict resume and export."""

from dataclasses import asdict
import math
import os
from pathlib import Path
import random
import signal
import threading

import torch
from torch.nn import functional as F
from torch.utils.data import default_collate

from Transformer import CS2Policy, CS2PolicyConfig
from cs2.actions import ACTION_SCHEMA_VERSION, BRANCHES, YAW_BINS, PITCH_BINS, decode_angle
from cs2.state import SCHEMA_VERSION
from .dataset import DemoWindows
from .io import object_hash, read_json, write_json

CHECKPOINT_SCHEMA='cs2-bc-checkpoint-v2'
DEFAULTS={'steps':200,'length':32,'accumulate':8,'lr':3e-4,'cnn_lr':1e-4,'weight_decay':.01,
          'seed':1337,'eval_every':50,'save_every':50,'patience':5,'warmup_fraction':.05,'device':'cuda','precision':'auto'}


def inputs(batch,device):
    names={'images':'images','previous':'previous_actions','state':'state_values','state_mask':'state_available',
           'age':'gsi_age','fresh':'gsi_fresh','time':'timestamps','valid':'valid','start':'episode_starts'}
    result={target:batch[source].to(device) for source,target in names.items()}
    for name in ('previous_actions','state_values','gsi_age'):
        result[name]=result[name].float()
    return result


def loss_parts(out,batch,device):
    mask=batch['label_mask'].to(device).bool() & batch['loss_mask'].to(device)[...,None] & batch['valid'].to(device)[...,None]
    parts={}; terms=[]
    def add(name,values,valid,weight):
        count=valid.sum()
        if count.item():
            total=(values.float()*valid).sum()
            parts[name]=(total,count)
            terms.append((total/count,weight))
    for name,index,key in [('forward',0,'forward'),('strafe',1,'strafe'),('weapon',10,'weapon')]:
        target=batch[key].to(device).long()
        ce=F.cross_entropy(out[name+'_logits'].flatten(0,1).float(),target.flatten(),reduction='none').reshape(target.shape)
        add(name,ce,mask[...,index],.5 if name=='weapon' else .5)
    button_loss=F.binary_cross_entropy_with_logits(out['button_logits'].float(),batch['buttons'].to(device).float(),reduction='none')
    active_buttons=max(1,int(mask[...,2:10].any(dim=(0,1)).sum()))
    for i,name in enumerate(BRANCHES[2:10]):
        add(name,button_loss[...,i],mask[...,2+i],1/active_buttons)
    for i,name in enumerate(('yaw','pitch')):
        target=batch['angle_bins'][...,i].to(device).long()
        ce=F.cross_entropy(out[name+'_logits'].flatten(0,1).float(),target.flatten(),reduction='none').reshape(target.shape)
        add(name,ce,mask[...,11+i],.5)
        residual=F.huber_loss(out['angle_residual'][...,i].float().tanh(),batch['residual'][...,i].to(device).float(),reduction='none')
        add(name+'_residual',residual,mask[...,11+i],.125)
    if not terms:
        raise ValueError('Batch has no valid supervised actions')
    return sum(value*weight for value,weight in terms)/sum(weight for _,weight in terms),parts


def _precision(device,precision):
    if device.type!='cuda': return False,torch.float32
    if precision=='fp32': return False,torch.float32
    dtype=torch.bfloat16 if precision in ('auto','bf16') and torch.cuda.is_bf16_supported() else torch.float16
    if precision=='bf16' and dtype!=torch.bfloat16: raise ValueError('BF16 is unsupported on this device')
    return True,dtype


@torch.no_grad()
def evaluate_model(model,dataset,device,*,precision='auto'):
    model.eval(); totals={}; counts={}; confusion={}; angle_errors={'yaw':[],'pitch':[]}
    enabled,dtype=_precision(device,precision)
    for index in range(len(dataset)):
        batch=default_collate([dataset[index]])
        with torch.autocast(device.type,dtype=dtype,enabled=enabled):
            out=model(**inputs(batch,device))
            _,parts=loss_parts(out,batch,device)
        for key,(total,count) in parts.items():
            totals[key]=totals.get(key,0)+float(total)
            counts[key]=counts.get(key,0)+int(count)
        mask=batch['label_mask'].bool() & batch['loss_mask'][...,None] & batch['valid'][...,None]
        predictions=(out['button_logits'].float().cpu()>=0)
        for i,name in enumerate(BRANCHES[2:10]):
            m=mask[...,i+2]; target=batch['buttons'][...,i].bool(); pred=predictions[...,i]
            tp=int((pred & target & m).sum()); fp=int((pred & ~target & m).sum()); fn=int((~pred & target & m).sum())
            old=confusion.get(name,[0,0,0]);confusion[name]=[old[0]+tp,old[1]+fp,old[2]+fn]
        residual=out['angle_residual'].float().tanh().cpu()
        for i,(name,bins) in enumerate((('yaw',YAW_BINS),('pitch',PITCH_BINS))):
            selected=out[name+'_logits'].argmax(-1).cpu()
            for b,t in mask[...,11+i].nonzero().tolist():
                guess=decode_angle(int(selected[b,t]),float(residual[b,t,i]),bins)
                angle_errors[name].append(abs(guess-float(batch['angles'][b,t,i])))
    losses={k:totals[k]/counts[k] for k in totals}
    # Fixed branch-average criterion, independent of window padding length.
    buttons=[losses[n] for n in BRANCHES[2:10] if n in losses]
    weighted=[(losses[k],w) for k,w in [('forward',.5),('strafe',.5),('weapon',.5),('yaw',.5),('pitch',.5),('yaw_residual',.125),('pitch_residual',.125)] if k in losses]
    if buttons: weighted.append((sum(buttons)/len(buttons),1.))
    score=sum(x*w for x,w in weighted)/sum(w for _,w in weighted)
    metrics={'loss':score,'branch_loss':losses,'valid_labels':counts,'windows':len(dataset),
             'angle_mae_deg':{k:sum(v)/len(v) if v else None for k,v in angle_errors.items()},
             'buttons':{k:{'tp':v[0],'fp':v[1],'fn':v[2],
                           'precision':v[0]/max(1,v[0]+v[1]),'recall':v[0]/max(1,v[0]+v[2])} for k,v in confusion.items()}}
    return metrics


def atomic_checkpoint(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(data,temporary)
    os.replace(temporary,path)


def _contract(config,model_config,manifest_hash,smoke):
    return {'settings':config,'model':asdict(model_config),'manifest_hash':manifest_hash,'smoke':smoke,
            'action_schema':ACTION_SCHEMA_VERSION,'state_schema':SCHEMA_VERSION}


def train(manifest,run_dir,*,config=None,model_config=None,resume=None,init_from=None,smoke=False,stop_after=None):
    cfg={**DEFAULTS,**(config or {})}
    if set(cfg)-set(DEFAULTS): raise ValueError(f'Unknown settings: {set(cfg)-set(DEFAULTS)}')
    for key in ('steps','length','accumulate','eval_every','save_every','patience'):
        if type(cfg[key]) is not int or cfg[key]<=0: raise ValueError(f'{key} must be a positive integer')
    if cfg['lr']<=0 or cfg['cnn_lr']<=0 or not 0<=cfg['warmup_fraction']<1: raise ValueError('Invalid optimizer settings')
    device=torch.device(cfg['device'])
    if device.type=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA is unavailable; explicitly select CPU')
    torch.set_num_threads(2);random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg['seed'])
    data=read_json(manifest)
    split='train' if any(e['split']=='train' for e in data['episodes']) else data['episodes'][0]['split']
    if not smoke and split!='train': raise ValueError('Production training requires a train split')
    trainset=DemoWindows(manifest,split=split,length=cfg['length'],training=True,allow_diagnostic=smoke)
    valset=DemoWindows(manifest,split=split if smoke else 'val',length=cfg['length'],allow_diagnostic=smoke)
    mc=model_config or CS2PolicyConfig()
    if cfg['length']>mc.block_size: raise ValueError('Training sequence exceeds model context')
    model=CS2Policy(mc).to(device)
    if resume and init_from: raise ValueError('Choose either exact resume or stage initialization')
    if init_from:
        initial=torch.load(init_from,map_location='cpu',weights_only=True)
        if initial.get('schema')!=CHECKPOINT_SCHEMA or initial['contract']['action_schema']!=ACTION_SCHEMA_VERSION:
            raise ValueError('Stage initialization requires a matching CS2 checkpoint')
        if object_hash(initial['contract']['model'])!=object_hash(asdict(mc)):
            raise ValueError('Model configuration must match for stage initialization')
        if initial['kind']=='pipeline_smoke' and not smoke:
            raise ValueError('Diagnostic weights cannot initialize professional demo training')
        model.load_state_dict(initial['model'])
    cnn_ids={id(p) for p in model.frame_encoder.parameters()}
    optimizer=torch.optim.AdamW([{'params':model.frame_encoder.parameters(),'lr':cfg['cnn_lr']},
                                 {'params':[p for p in model.parameters() if id(p) not in cnn_ids],'lr':cfg['lr']}],weight_decay=cfg['weight_decay'])
    warmup=max(1,round(cfg['steps']*cfg['warmup_fraction']))
    def schedule(step):
        if step<warmup: return max(1e-3,(step+1)/warmup)
        return .1+.9*.5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,cfg['steps']-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    enabled,dtype=_precision(device,cfg['precision'])
    scaler=torch.amp.GradScaler('cuda',enabled=enabled and dtype==torch.float16)
    contract=_contract(cfg,mc,data['hash'],smoke)
    contract['observation']={'sample_hz':data['hz'],'gsi_delay':data['gsi_delay']}
    step=cursor=bad=0;best=float('inf');history=[]
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    if resume:
        checkpoint=torch.load(resume,map_location='cpu',weights_only=True)
        if checkpoint.get('schema')!=CHECKPOINT_SCHEMA or object_hash(checkpoint['contract'])!=object_hash(contract):
            raise ValueError('Resume requires identical model, data, schema and training settings')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler']);scaler.load_state_dict(checkpoint['scaler'])
        step,cursor,best,bad=checkpoint['step'],checkpoint['cursor'],checkpoint['best'],checkpoint['bad']
        history=checkpoint['history'];torch.set_rng_state(checkpoint['rng'])
        if torch.cuda.is_available(): torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        random.setstate(checkpoint['python_rng'])
    elif (run_dir/'last.pt').exists():
        raise ValueError('Run directory already has a checkpoint; use --resume or a new directory')
    write_json(run_dir/'config.json',contract)
    order=[];order_epoch=None
    last_boundary_cursor=cursor
    def save(name):
        atomic_checkpoint(run_dir/name,{'schema':CHECKPOINT_SCHEMA,'contract':contract,
             'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
             'scaler':scaler.state_dict(),'step':step,'cursor':cursor,'best':best,'bad':bad,'history':history,
             'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             'python_rng':random.getstate(),'coverage':data['coverage'],
             'kind':'pipeline_smoke' if smoke else 'professional_demo_bc','value_trained':False})
    budget=cfg['steps'] if stop_after is None else min(cfg['steps'],step+stop_after)
    stopping=False
    def request_stop(*args):
        nonlocal stopping
        stopping=True
    old_handler=signal.signal(signal.SIGINT,request_stop) if threading.current_thread() is threading.main_thread() else None
    try:
        while step<budget:
            model.train();optimizer.zero_grad(set_to_none=True);losses=[]
            for _ in range(cfg['accumulate']):
                epoch=cursor//len(trainset)
                if epoch!=order_epoch:
                    order=list(range(len(trainset)));random.Random(cfg['seed']+epoch).shuffle(order);order_epoch=epoch
                batch=default_collate([trainset[order[cursor%len(trainset)]]]);cursor+=1
                with torch.autocast(device.type,dtype=dtype,enabled=enabled):
                    out=model(**inputs(batch,device));loss,_=loss_parts(out,batch,device)
                if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
                scaler.scale(loss/cfg['accumulate']).backward();losses.append(float(loss.detach()))
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            scaler.step(optimizer);scaler.update();scheduler.step();step+=1
            last_boundary_cursor=cursor
            record={'step':step,'train_loss':sum(losses)/len(losses),'grad_norm':float(norm)}
            if step%cfg['eval_every']==0 or step==cfg['steps']:
                metrics=evaluate_model(model,valset,device,precision=cfg['precision']);record['validation']=metrics
                if metrics['loss']<best:
                    best=metrics['loss'];bad=0;history.append(record);save('best.pt');history.pop()
                else: bad+=1
            history.append(record)
            with (run_dir/'metrics.jsonl').open('a',encoding='utf-8') as stream:
                import json
                stream.write(json.dumps(record,allow_nan=False)+'\n')
            if step==1 or step%10==0: print(f"update {step}/{cfg['steps']} loss={record['train_loss']:.5f}",flush=True)
            if step%cfg['save_every']==0: save('last.pt')
            if bad>=cfg['patience'] or stopping: break
    except KeyboardInterrupt:
        # Never serialize half-accumulated gradients/updated RNG as an exact boundary.
        raise KeyboardInterrupt('Training interrupted; resume the last completed atomic checkpoint')
    except BaseException as error:
        write_json(run_dir/'failure.json',{'step':step,'error':str(error),'last_complete_cursor':last_boundary_cursor})
        raise
    finally:
        if old_handler is not None: signal.signal(signal.SIGINT,old_handler)
    save('last.pt')
    if stopping: save('interrupt.pt')
    if not (run_dir/'best.pt').exists() and smoke: save('best.pt')
    result={'step':step,'interrupted':stopping,'best_validation_loss':best if math.isfinite(best) else None,'kind':'pipeline_smoke' if smoke else 'professional_demo_bc',
            'manifest_hash':data['hash'],'last':str((run_dir/'last.pt').resolve()),'history':history,
            'peak_cuda_mib':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None}
    write_json(run_dir/'summary.json',result)
    return result


def evaluate(checkpoint,manifest,*,split='test',device='cuda',allow_diagnostic=False):
    data=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if data.get('schema')!=CHECKPOINT_SCHEMA: raise ValueError('Unknown checkpoint schema')
    if data['kind']=='pipeline_smoke' and not allow_diagnostic: raise ValueError('Smoke weights are not a pretrained professional policy')
    if split=='test' and read_json(manifest)['hash']!=data['contract']['manifest_hash']:
        raise ValueError('Test evaluation must use the checkpoint dataset manifest')
    model=CS2Policy(CS2PolicyConfig(**data['contract']['model'])).to(device)
    model.load_state_dict(data['model'])
    dataset=DemoWindows(manifest,split=split,length=data['contract']['settings']['length'],allow_diagnostic=allow_diagnostic)
    return evaluate_model(model,dataset,torch.device(device),precision=data['contract']['settings']['precision'])


def export(checkpoint,output,*,allow_diagnostic=False):
    data=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if data.get('schema')!=CHECKPOINT_SCHEMA: raise ValueError('Unknown checkpoint schema')
    if data['kind']=='pipeline_smoke' and not allow_diagnostic:
        raise ValueError('Refusing to export self-test weights as a professional pretrained policy')
    capabilities=[data['coverage'].get(name,0)>0 for name in BRANCHES]
    weights=data['model'].copy()
    weights['action_availability']=torch.tensor(capabilities,dtype=torch.bool)
    package={'schema':'cs2-policy-export-v2','model':weights,'config':data['contract']['model'],
             'action_schema':ACTION_SCHEMA_VERSION,'state_schema':SCHEMA_VERSION,
             'yaw_bins':YAW_BINS,'pitch_bins':PITCH_BINS,'observation':data['contract']['observation'],
             'coverage':data['coverage'],'kind':data['kind'],'value_trained':False,'source_manifest_hash':data['contract']['manifest_hash']}
    atomic_checkpoint(output,package)
    write_json(Path(output).with_suffix('.json'),{k:v for k,v in package.items() if k!='model'})
    return {'output':str(Path(output).resolve()),'kind':data['kind']}


def load_export(path,*,device='cpu',allow_diagnostic=False):
    data=torch.load(path,map_location='cpu',weights_only=True)
    if data.get('schema')!='cs2-policy-export-v2' or data.get('action_schema')!=ACTION_SCHEMA_VERSION or data.get('state_schema')!=SCHEMA_VERSION:
        raise ValueError('Export does not match this CS2 policy schema')
    if data['kind']=='pipeline_smoke' and not allow_diagnostic:
        raise ValueError('Diagnostic export is not a professional pretrained policy')
    model=CS2Policy(CS2PolicyConfig(**data['config']))
    model.load_state_dict(data['model'])
    return model.to(device).eval(),{k:v for k,v in data.items() if k!='model'}
