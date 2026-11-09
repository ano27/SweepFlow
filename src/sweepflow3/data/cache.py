from __future__ import annotations
from ..progress import Progress,activity,reported,iterate,detail,reused,event
import time,math
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from ..common import *
from .readers import read_points
from .camera import camera_plan,prepare_rectified_image
from .geometry import (spherical_raster,build_exact_return_graph,rasterize_local_pcd,patch_camera_condition)
from .dataset import PointScenes
from .points import collate_points,make_packet
from ..models.vae import SD35Autoencoder
from ..models.point import DinoTeacher,ScaLRPoints
from ..assets import asset_paths


def read_rgb(spec,hw):
    with Image.open(spec['image_path']) as im:
        image,_,_=prepare_rectified_image(im,spec['intrinsics'],hw,spec.get('distortion'),spec.get('image_state'))
    return torch.from_numpy(np.asarray(image,dtype=np.uint8).copy()).permute(2,0,1).float()/255


@reported('intensity-statistics')
def fit_intensity(cfg):
    p=stats_path(cfg)
    if p.exists():
        event('intensity-statistics','REUSED',dataset=cfg['dataset'],path=p)
        return read_json(p)
    rows=rows_for(cfg,'train');idx=np.linspace(0,len(rows)-1,min(len(rows),128),dtype=int);vals=[];missing=0
    for i in iterate(idx,f'intensity/{cfg["dataset"]}',unit='scenes'):
        detail(index=int(i),phase='read_raw_points')
        a,has,_=read_points(rows[i]);missing+=not has
        if has:vals.append(a[::max(1,len(a)//4096),3])
    a=np.concatenate(vals).astype(float) if vals else np.zeros(1)
    lo,hi=np.quantile(a,[.01,.99]);hi=max(hi,lo+1e-6)
    value={'mean':float(a.mean()),'std':float(max(a.std(),1e-6)),'low':float(lo),'high':float(hi),
           'fit_split':'train','sampled_scenes':len(idx),'missing_intensity_scenes':int(missing),'missing_channel_policy':'zero_no_spatial_proxy'}
    atomic_json(p,value);return value


@torch.no_grad()
def geometry_packet(row,cfg,stats,compute_device=None):
    raw,available,ids=read_points(row);points=torch.tensor(raw,device=compute_device);points[:,3]=((points[:,3]-stats['low'])/(stats['high']-stats['low'])).clamp(0,1)
    if not available:points[:,3].zero_()
    hw=cfg['data']['image_hw'];lh=(hw[0]//8,hw[1]//8);th=(hw[0]//16,hw[1]//16);K=[];T=[];local=[];camera=[]
    for spec in row['camera_specs']:
        k,*_=camera_plan((spec['width'],spec['height']),spec['intrinsics'],hw,spec.get('distortion'),spec.get('image_state'))
        k=torch.tensor(k.copy(),device=points.device);t=torch.tensor(spec['lidar_to_camera'],dtype=torch.float32,device=points.device)
        K.append(k);T.append(t);local.append(rasterize_local_pcd(points,k,t,hw,lh)[0]);camera.append(patch_camera_condition(k,t,hw,lh,2,spec['timestamp_delta_sec']))
    # Same original spherical top-2 return selection, but the large spherical
    # tensor is temporary and is not duplicated in the new semantic cache.
    _,idm,_=spherical_raster(points,96,2048,hits=2);hit,y,x=torch.where(idm>=0);retained=idm[hit,y,x].long()
    graph=build_exact_return_graph(points,retained,torch.stack((y,x),1),hit,torch.stack(K),torch.stack(T),torch.ones(len(K),*hw,dtype=torch.bool,device=points.device),
            hw,th,torch.tensor([s['timestamp_delta_sec'] for s in row['camera_specs']],device=points.device),budget=cfg['model']['return_budget'])
    return {'scene_uid':row['scene_uid'],'local_pcd':torch.stack(local).half().cpu(),'camera_condition':torch.stack(camera).float().cpu(),
            'return_features':graph['return_features'].float().cpu(),'edge_index':graph['edge_index'].cpu(),'edge_weight':graph['edge_weight'].cpu(),
            'return_ids':torch.from_numpy(ids)[graph['return_ids'].cpu()],'K':torch.stack(K).cpu(),'T':torch.stack(T).cpu(),
            'intensity_available':available,'raw_point_count':len(raw),'shared_before_budget':graph['shared_return_count_before_budget']}


def base_kinds(split):
    # RGB targets are needed only for the training split.
    return ('geometry','latent','teacher') if split=='train' else ('geometry',)


@reported('base-cache')
def prepare_base(cfg,split,limit=0):
    d=device();torch.set_num_threads(cfg['workers']['cpu_threads']);rows=rows_for(cfg,split);stats=read_json(stats_path(cfg));n=min(limit,len(rows)) if limit else len(rows)
    vae=teacher=None;t0=time.monotonic()
    for j,i in enumerate(iterate(range(rank(),n,world()),f'base/{cfg["dataset"]}/{split}',unit='scenes',dataset_total=n)):
        detail(index=i,phase='geometry_or_target_cache')
        r=rows[i];paths={k:packet_path(cfg,k,split,i) for k in base_kinds(split)}
        did_compute=False
        if not paths['geometry'].is_file():
            atomic_torch(paths['geometry'],geometry_packet(r,cfg,stats,d));did_compute=True
        if any(k in paths and not paths[k].is_file() for k in ('latent','teacher')):
            did_compute=True
            rgb=torch.stack([read_rgb(s,cfg['data']['image_hw']) for s in r['camera_specs']]).to(d)
            if 'latent' in paths and not paths['latent'].is_file():
                if vae is None:
                    with activity('load-native-VAE',dataset=cfg['dataset']):
                        vae=SD35Autoencoder(cfg['model']['sd35_root']).to(d)
                z=vae.encode(rgb).half().cpu();atomic_torch(paths['latent'],{'scene_uid':r['scene_uid'],'latent':z})
            if 'teacher' in paths and not paths['teacher'].is_file():
                if teacher is None:
                    with activity('load-DINO-teacher',dataset=cfg['dataset']):
                        teacher=DinoTeacher(asset_paths(cfg['assets_root'],teacher=True),cfg['semantic']['dim'],cfg['semantic']['projection_seed'],cfg['semantic']['teacher_hw']).to(d)
                with torch.inference_mode(),autocast(d):f=torch.cat([teacher(x).cpu().half() for x in rgb.split(cfg['semantic']['teacher_view_batch'])])
                atomic_torch(paths['teacher'],{'scene_uid':r['scene_uid'],'feature':f,'role':'target_only'})
        if not did_compute:reused()
    atomic_json(dataset_root(cfg)/'cache'/f'base_{split}_rank{rank()}.json',{'state':'completed','limit':limit,'world':world()})


@reported('load-point-model')
def load_point(cfg,encoder,d):
    if encoder != 'adapted':raise ValueError('Only the trained Full point encoder is supported')
    p=point_checkpoint(cfg);v=load_torch(p,trusted=True)
    if v.get('stage')!='stage_a':raise ValueError(f'Not a Stage A checkpoint: {p}')
    if 'network' not in v:raise ValueError(f'Stage A checkpoint is missing its network: {p}')
    if 'contract' in v:
        contract=v['contract']
        if contract['dataset']!=cfg['dataset'] or contract['data']!=cfg['data']:
            raise ValueError('Stage A checkpoint dataset, camera order, or image dimensions do not match')
        for key in ('dim','projection_seed'):
            if contract['semantic'][key]!=cfg['semantic'][key]:
                raise ValueError(f'Stage A semantic setting differs: {key}')
        for key in ('max_points','voxel_size'):
            if contract['points'][key]!=cfg['points'][key]:
                raise ValueError(f'Stage A point preprocessing differs: {key}')
    s=cfg['semantic']
    m=ScaLRPoints(asset_paths(cfg['assets_root'],point_weights=False),s['dim'],s['projection_seed'],0,pretrained=False).to(d)
    m.load_state_dict(v['network'],strict=True)
    return m.eval().requires_grad_(False)


def encode_point_item(model,x,cfg,d):
    """Inference-only point feature computation, including absent FOV support."""
    if x['selected_count']==0:
        # There are still valid RAW points; only the fixed semantic crop is empty.
        return torch.empty((int(cfg['semantic']['dim']),0),dtype=torch.float32)
    if model is None:
        raise RuntimeError('A native point encoder is required for nonempty PCD')
    b=collate_points([x]);inp=move({k:b[k] for k in ('feats','cell_ind','occupied','neighbors')},d)
    with torch.inference_mode(),autocast(d):f=model(inp)[0,:,:x['selected_count']]
    ensure_finite(f,'student features')
    return f






@reported('student-packets')
def export_packets(cfg,split,encoder='adapted',limit=0):
    d=device();torch.set_num_threads(cfg['workers']['cpu_threads']);ds=PointScenes(cfg,split,targets=False);n=min(limit,len(ds)) if limit else len(ds)
    model=None;t0=time.monotonic()
    for j,i in enumerate(iterate(range(rank(),n,world()),f'student/{cfg["dataset"]}/{encoder}/{split}',unit='scenes',dataset_total=n)):
        detail(index=i,phase='read_PCD_encode_and_save')
        p=packet_path(cfg,'student',split,i,encoder)
        if p.is_file():reused();continue
        x=ds[i]
        if model is None and x['selected_count']>0:model=load_point(cfg,encoder,d)
        f=encode_point_item(model,x,cfg,d)
        packet=make_packet(f,x,cfg);packet['encoder']=encoder
        atomic_torch(p,packet)





