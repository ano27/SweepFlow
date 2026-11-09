"""Small checkpoint and scheduling helpers shared by training and verification."""
from __future__ import annotations
import math
import torch







def module_of(model):
    return model.module if hasattr(model,'module') else model




def restore_adapter(model,state):
    named={n:p for n,p in module_of(model).named_parameters() if p.requires_grad}
    if set(named)!=set(state):
        raise ValueError('Checkpoint trainable parameter names do not match this runtime.')
    with torch.no_grad():
        for n,p in named.items():
            if p.shape!=state[n].shape:
                raise ValueError(f'Checkpoint shape differs for {n}')
            p.copy_(state[n].to(p))


class EMA:
    def __init__(self,model,decay=.9999):
        self.decay=decay;self.updates=0
        self.shadow={n:p.detach().clone() for n,p in module_of(model).named_parameters() if p.requires_grad}
    @torch.no_grad()
    def update(self,model):
        self.updates+=1
        d=min(self.decay,(1+self.updates)/(10+self.updates))
        named=dict(module_of(model).named_parameters())
        for n,s in self.shadow.items():s.lerp_(named[n].detach(),1-d)
    def state_dict(self):
        return {'decay':self.decay,'updates':self.updates,'shadow':{n:s.cpu() for n,s in self.shadow.items()}}
    def load_state_dict(self,x):
        if set(x['shadow'])!=set(self.shadow):raise ValueError('EMA parameter names differ.')
        self.decay=x['decay'];self.updates=x['updates']
        for n,s in self.shadow.items():s.copy_(x['shadow'][n].to(s))


def lr_at(step,total,peak,warmup):
    warmup=min(warmup,max(0,total-1))
    if step<warmup:return peak*(step+1)/max(1,warmup)
    progress=(step-warmup)/max(1,total-warmup-1)
    return peak*.5*(1+math.cos(math.pi*min(1.,progress)))


def configure_timesteps(scheduler,steps,device,image_token_count):
    kwargs={'device':device}
    cfg=scheduler.config
    if getattr(cfg,'use_dynamic_shifting',False):
        x0=float(getattr(cfg,'base_image_seq_len',256)); x1=float(getattr(cfg,'max_image_seq_len',4096))
        y0=float(getattr(cfg,'base_shift',.5));y1=float(getattr(cfg,'max_shift',1.15))
        kwargs['mu']=y0+(image_token_count-x0)*(y1-y0)/(x1-x0)
    scheduler.set_timesteps(int(steps),**kwargs)
    return scheduler


def draw_training_times(scheduler,B,V,device,generator):
    # Index on the same device as the lookup tables. Views share one scene time.
    table_t=scheduler.timesteps.to(device)
    table_s=scheduler.sigmas[:len(table_t)].to(device)
    ids=torch.randint(len(table_t),(B,),device=device,generator=generator)
    t=table_t[ids,None].expand(B,V)
    sigma=table_s[ids,None,None,None,None].expand(B,V,1,1,1)
    return t,sigma




def load_scheduler(root):
    from diffusers import FlowMatchEulerDiscreteScheduler
    return FlowMatchEulerDiscreteScheduler.from_pretrained(root,subfolder='scheduler',local_files_only=True)
