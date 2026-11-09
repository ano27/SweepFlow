"""One training engine, fixed GLOBAL scene batch, resume across GPU counts.

No DataLoader subprocesses. IO threads only read CPU arrays. Rank-aware loss
normalization makes uneven 32/3 partitions valid. Runtime layout is not the
scientific batch size. RNG is indexed by global update and scene slot.
"""
from __future__ import annotations
from ..progress import Progress,activity,reported,detail,event
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
import math,os,time,json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from ..common import *
from ..assets import asset_paths
from ..models.point import ScaLRPoints
from ..models.semantic import SemanticJointFlow
from ..models.semantic import alignment_loss
from ..models.runtime import EMA,lr_at,load_scheduler,draw_training_times,configure_timesteps
from ..models.prompt import EmptyPromptEmbeds
from ..data.dataset import PointScenes,GenerationScenes
from ..data.points import collate_generation,collate_points,projection_loss


class GlobalCursor:
    def __init__(self,size,seed,state=None):
        self.size=int(size);self.seed=int(seed);self.epoch=0;self.offset=0;self.draws=0;self.skipped=0;self._order=None;self._epoch=None
        if state:
            if state['size']!=size or state['seed']!=seed:raise ValueError('Dataset/seed changes on resume')
            for k in ('epoch','offset','draws','skipped'):setattr(self,k,int(state[k]))
    def peek(self,batch):
        if not 0<batch<=self.size:raise ValueError('Global batch exceeds dataset')
        if self.offset+batch>self.size:self.skipped+=self.size-self.offset;self.epoch+=1;self.offset=0
        if self._epoch!=self.epoch:
            self._order=torch.randperm(self.size,generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist();self._epoch=self.epoch
        return self._order[self.offset:self.offset+batch]
    def advance(self,batch):self.offset+=batch;self.draws+=batch
    def state_dict(self):return {k:getattr(self,k) for k in ('size','seed','epoch','offset','draws','skipped')}


def slots(batch,w,r):return list(range(r,batch,w))
def support_mass(c):return torch.nn.functional.avg_pool2d(c['semantic_support'].float().flatten(0,1),2,2).sum()


def build_model(cfg,stage,d):
    if stage=='stage_a':
        s=cfg['semantic'];return ScaLRPoints(asset_paths(cfg['assets_root']),s['dim'],s['projection_seed'],cfg['points']['checkpoint_blocks']).to(d)
    return SemanticJointFlow(cfg).to(d)


def optimizer_for(m,cfg,stage):
    s=cfg[stage]
    if stage=='stage_a':
        body=[p for n,p in m.named_parameters() if p.requires_grad and not n.startswith('native.classif')]
        head=[p for n,p in m.named_parameters() if p.requires_grad and n.startswith('native.classif')]
        groups=[{'params':body,'lr_scale':1.},{'params':head,'lr_scale':s['head_lr']/s['lr']}]
    else:groups=[{'params':[p for p in m.parameters() if p.requires_grad],'lr_scale':1.}]
    return torch.optim.AdamW(groups,lr=s['lr'],weight_decay=s['weight_decay'],foreach=False)


def train_contract(cfg,stage):
    return {k:cfg[k] for k in ('dataset','seed','data','points','semantic','model')}|{'stage':stage,'schedule':cfg[stage],'encoder':cfg.get('encoder','adapted')}


@reported('train')
def train(cfg,stage='stage_b',through=None,*,model_factory=None,dataset=None):
    d=device();r=rank();w=world();torch.set_num_threads(cfg['workers']['cpu_threads']);seed_all(cfg['seed']);s=cfg[stage]
    gb=int(s['global_batch']);micro=int(s['microbatch']);end=min(int(through or s['steps']),s['steps'])
    if min(gb,micro)<1:raise ValueError('Batch sizes must be positive')
    if w>gb:raise ValueError('GPU count may not exceed global batch; no silent extra GPU allocation')
    ds=dataset or (PointScenes(cfg,'train',True) if stage=='stage_a' else GenerationScenes(cfg,'train',True))
    out=dataset_root(cfg)/'stage_a' if stage=='stage_a' else run_root(cfg)/'stage_b';out.mkdir(parents=True,exist_ok=True)
    cp=out/'checkpoint_last.pt';contract=train_contract(cfg,stage)
    # Microbatch is a memory bound, not a scientific training choice.
    contract['schedule']=dict(contract['schedule']);contract['schedule'].pop('microbatch',None)
    with activity('train/load-model-and-optimizer',dataset=cfg['dataset'],arm=cfg['arm'],stage=stage):
        model=(model_factory(cfg,stage,d) if model_factory else build_model(cfg,stage,d));opt=optimizer_for(model,cfg,stage)
    ema=EMA(model,s.get('ema_decay',.9999)) if stage=='stage_b' else None;cursor=GlobalCursor(len(ds),cfg['seed']);step=0
    if cp.is_file():
        with activity('train/read-checkpoint',path=cp):
            v=load_torch(cp,trusted=True)
        if v.get('version')!='sweepflow3_training_v1' or v['contract']!=contract:raise ValueError('Checkpoint belongs to different experiment; use a new workdir')
        if stage=='stage_a':model.load_state_dict(v['network'],strict=True)
        else:restore_parameters(model,v['adapter'])
        opt.load_state_dict(v['optimizer'])
        if ema:ema.load_state_dict(v['ema'])
        cursor=GlobalCursor(len(ds),cfg['seed'],v['cursor']);step=int(v['step']);del v
    if step>=end:
        event('train','REUSED',dataset=cfg['dataset'],arm=cfg['arm'],stage=stage,saved_step=step,requested_through=end)
        return
    if w>1:
        opts={'backend':'nccl' if d.type=='cuda' else 'gloo','timeout':timedelta(minutes=10)}
        if d.type=='cuda':opts['device_id']=d
        with activity('train/DDP-init',world_size=w,stage=stage):
            dist.init_process_group(**opts)
    net=DDP(model,device_ids=[d.index] if d.type=='cuda' else None,find_unused_parameters=False,broadcast_buffers=False) if w>1 else model
    net.train();scheduler=prompt=None
    if stage=='stage_b':
        scheduler=load_scheduler(cfg['model']['sd35_root']);H,W=cfg['data']['image_hw'];configure_timesteps(scheduler,scheduler.config.num_train_timesteps,d,(H//16)*(W//16))
        prompt=EmptyPromptEmbeds(cfg['model']['sd35_root'],d,cache_path=cfg['model']['empty_prompt_cache'])
    sl=slots(gb,w,r);rounds=math.ceil(math.ceil(gb/w)/micro);pool=ThreadPoolExecutor(max_workers=cfg['workers']['io_threads']) if cfg['workers']['io_threads'] else None
    params=[p for p in model.parameters() if p.requires_grad];start=step;t0=time.monotonic();phase='ready'
    if d.type=='cuda':torch.cuda.reset_peak_memory_stats(d)
    if r==0:
        atomic_json(out/'layout.json',{'world_size':w,'gpus':os.environ.get('CUDA_VISIBLE_DEVICES'),'global_batch':gb,'microbatch':micro,'rank_scenes':[len(slots(gb,w,k)) for k in range(w)],'separate_loader_processes':False})
    @reported('checkpoint-save')
    def save(numbered=False):
        detail(step=step,path=str(cp),numbered=numbered)
        value={'version':'sweepflow3_training_v1','stage':stage,'step':step,'contract':contract,'cursor':cursor.state_dict(),'optimizer':opt.state_dict(),'ema':ema.state_dict() if ema else None}
        if stage=='stage_a':value['network']={k:v.detach().cpu() for k,v in model.state_dict().items()}
        else:value['adapter']=parameter_state(model)
        atomic_torch(cp,value)
        if numbered:atomic_torch(out/f'checkpoint_step_{step:06d}.pt',value)
    progress=Progress(f'train/{cfg["dataset"]}/{cfg["arm"]}/{stage}',end,initial=step,unit='updates',visible=r==0,
                      seed=cfg['seed'],target_updates=s['steps'],requested_through=end,global_batch=gb,microbatch=micro,world_size=w)
    progress.__enter__()
    try:
        while step<end:
            progress.detail(phase='read_cpu',current_update=step+1)
            phase='read_cpu';ids=cursor.peek(gb);local=[ids[k] for k in sl];items=list(pool.map(ds.__getitem__,local)) if pool else [ds[i] for i in local]
            denom=torch.tensor(sum(float(support_mass(collate_generation([x])['condition'])) for x in items) if stage=='stage_b' else gb,dtype=torch.float32,device=d)
            if stage=='stage_b' and w>1:dist.all_reduce(denom)
            opt.zero_grad(set_to_none=True);lr=lr_at(step,s['steps'],s['lr'],s['warmup'])
            for group in opt.param_groups:group['lr']=lr*group['lr_scale']
            logs=torch.zeros(3,device=d);lam=cfg['semantic']['alignment_weight']*min(1.,(step+1)/max(1,cfg['semantic']['alignment_warmup'])) if stage=='stage_b' else 0
            for turn in range(rounds):
                small=items[turn*micro:(turn+1)*micro];ss=sl[turn*micro:(turn+1)*micro];valid=len(small)
                if not small:small=[items[0]];ss=[sl[0]]
                sync=net.no_sync() if w>1 and turn<rounds-1 else nullcontext()
                with sync:
                    phase='forward';progress.detail(phase=phase,microbatch_round=turn+1,microbatch_rounds=rounds)
                    if stage=='stage_a':
                        b=collate_points(small);x=move({k:b[k] for k in ('feats','cell_ind','occupied','neighbors')},d)
                        with autocast(d):pred=net(x)
                        loss0,details=projection_loss(pred,b['items'],cfg['data']['image_hw']);loss=loss0*(w*valid/gb)
                        logs[1]+=loss0.detach()*valid/gb
                    else:
                        b=collate_generation(small);c=move(b['condition'],d);tar=move(b['target'],d);z=tar['latent'].float();B,V=z.shape[:2];noise=[];tt=[];sg=[]
                        for j,slot in enumerate(ss):
                            rng=torch.Generator(device=d).manual_seed((cfg['seed']+100000+step*gb+slot)%(2**63-1));t,sigma=draw_training_times(scheduler,1,V,d,rng)
                            tt.append(t);sg.append(sigma);noise.append(torch.randn(z[j:j+1].shape,device=d,generator=rng))
                        noise=torch.cat(noise);t=torch.cat(tt);sigma=torch.cat(sg);z_t=(1-sigma)*z+sigma*noise;p,pp=prompt.get(B*V)
                        with autocast(d):pred,al=net(z_t,t,sigma,c,p,pp,return_alignment=True)
                        fm=(pred.float()-(noise-z)).square().mean();aloss=alignment_loss(al,tar['teacher'],c['semantic_support']);mass=support_mass(c).clamp_min(1) if valid else denom.new_zeros(())
                        loss=w*(fm*(valid/gb)+lam*aloss*(mass/denom.clamp_min(1)))
                        logs[1]+=fm.detach()*valid/gb;logs[2]+=aloss.detach()*mass/denom.clamp_min(1)
                    phase='backward';progress.detail(phase=phase);ensure_finite(loss,'training loss');loss.backward();logs[0]+=loss.detach()/w
            phase='optimizer';progress.detail(phase=phase);norm=torch.nn.utils.clip_grad_norm_(params,s['grad_clip'],error_if_nonfinite=True);opt.step()
            if ema:ema.update(model)
            cursor.advance(gb);step+=1
            if w>1:dist.all_reduce(logs)
            if r==0 and (step==start+1 or step%cfg['log_every']==0):
                record={'stage':stage,'step':step,'loss':float(logs[0]),'fm_or_point':float(logs[1]),'alignment':float(logs[2]),'alignment_lambda':lam,'lr':lr,'grad_norm':float(norm),
                        'global_batch':gb,'world_size':w,'scene_draws':cursor.draws,'seconds':time.monotonic()-t0,'peak_gib':torch.cuda.max_memory_allocated(d)/2**30 if d.type=='cuda' else 0}
                with (out/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
                progress.detail(loss=record['loss'],fm_or_point=record['fm_or_point'],alignment=record['alignment'])
            progress.advance(lr=lr,scene_draws=cursor.draws,epoch=cursor.epoch+1,phase='optimizer_update_completed')
            if r==0 and (step%s['save_every']==0 or step==end):phase='checkpoint';save(step==end)
        if r==0:atomic_json(out/'progress.json',{'state':'completed' if step==s['steps'] else 'milestone_completed','step':step,'cursor':cursor.state_dict()})
        # Wait for rank 0's atomic checkpoint write before successful teardown.
        # Never enter this barrier from an exception handler.
        if dist.is_initialized():
            with activity('train/final-rank-barrier',step=step):
                dist.barrier()
    except BaseException as e:
        progress.close('FAILED',failed_step=step,failed_phase=phase,exception=repr(e))
        atomic_json(out/f'failure_rank_{r}.json',{'step':step,'phase':phase,'exception':repr(e)});raise
    finally:
        progress.close()
        detail(phase='close_IO_threads_and_DDP')
        if pool:pool.shutdown(wait=True,cancel_futures=True)
        if dist.is_initialized():dist.destroy_process_group()
