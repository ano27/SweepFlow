from __future__ import annotations
import json, os, random
from pathlib import Path
import numpy as np
import torch


def read_json(path):
    return json.loads(Path(path).read_text())

def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(x) for x in f if x.strip()]

def atomic_json(path, value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.tmp-{os.getpid()}')
    try:
        t.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(t,p)
    finally:t.unlink(missing_ok=True)

def atomic_torch(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.tmp-{os.getpid()}')
    try:torch.save(value,t);os.replace(t,p)
    finally:t.unlink(missing_ok=True)

def load_torch(path, *, trusted=False):
    return torch.load(path,map_location='cpu',weights_only=not trusted)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed % (2**32));torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def device():
    n=int(os.environ.get('LOCAL_RANK',0))
    if torch.cuda.is_available():
        torch.cuda.set_device(n);d=torch.device('cuda',n);torch.empty(1,device=d);return d
    return torch.device('cpu')

def rank():return int(os.environ.get('RANK',0))
def world():return int(os.environ.get('WORLD_SIZE',1))
def move(x,d):
    if torch.is_tensor(x):return x.to(d,non_blocking=True)
    if isinstance(x,dict):return {k:move(v,d) for k,v in x.items()}
    return x

def autocast(d):return torch.autocast(d.type,dtype=torch.bfloat16,enabled=d.type=='cuda')


def parameter_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}

def restore_parameters(model,state):
    expected={n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(expected)!=set(state):raise ValueError('Trainable parameter names do not match this model.')
    with torch.no_grad():
        for n,p in expected.items():
            if p.shape!=state[n].shape:raise ValueError(f'Parameter shape mismatch: {n}')
            p.copy_(state[n])

def ensure_finite(value,label):
    if not torch.isfinite(value).all():raise FloatingPointError(f'Nonfinite {label}')


def atomic_jsonl(path,rows):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.tmp-{os.getpid()}')
    try:
        with tmp.open('w') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        os.replace(tmp,p)
    finally:tmp.unlink(missing_ok=True)


def dataset_root(cfg):return Path(cfg['workdir'])/'data'/cfg['dataset']
def manifest_path(cfg,split):return dataset_root(cfg)/'manifests'/(split+'.jsonl')
def rows_for(cfg,split):return read_jsonl(manifest_path(cfg,split))
def packet_path(cfg,kind,split,i,encoder='adapted'):
    prefix=cfg.get('packet_override_root') if kind in ('student','geometry') else None
    base=Path(prefix) if prefix else dataset_root(cfg)/'cache'
    if kind=='student':base=base/'student'/encoder
    else:base=base/kind
    return base/split/'items'/f'{i:08d}.pt'
def run_root(cfg):return Path(cfg['workdir'])/'runs'/cfg['dataset']/cfg['arm']/f'seed_{cfg["seed"]}'
def point_checkpoint(cfg):
    return Path(cfg.get('point_checkpoint_override') or dataset_root(cfg)/'stage_a'/'checkpoint_last.pt')
def stats_path(cfg):return Path(cfg.get('stats_override') or dataset_root(cfg)/'intensity_statistics.json')
