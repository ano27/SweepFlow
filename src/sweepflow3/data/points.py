from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree
MIN_WAFFLE_POINTS = 17
SEMANTIC_FOV_LOW = (-64., -64., -8.)
SEMANTIC_FOV_HIGH = (64., 64., 8.)
SPARSE_INPUT_POLICY = 'pcd_sparse_inference_v1_2'


def point_diagnostics(raw_points, selected=None):
    """Counts/extents after read_points' finite/zero-return filter; no RGB access."""
    a = np.asarray(raw_points)
    xyz = a[:, :3]
    finite = np.isfinite(xyz).all(1)
    inside = finite & ((xyz >= SEMANTIC_FOV_LOW) & (xyz < SEMANTIC_FOV_HIGH)).all(1)
    p = xyz[finite]
    count = None if selected is None else len(selected)
    return {
        'policy': SPARSE_INPUT_POLICY,
        'valid_raw_points_after_reader': len(a),
        'finite_model_xyz_points': int(finite.sum()),
        'points_inside_semantic_fov_before_voxel': int(inside.sum()),
        'selected_points_after_voxel_and_cap': count,
        'model_xyz_min': p.min(0).astype(float).tolist() if len(p) else None,
        'model_xyz_max': p.max(0).astype(float).tolist() if len(p) else None,
        'model_range_p50_m': float(np.median(np.linalg.norm(p, axis=1))) if len(p) else None,
        'semantic_fov': [list(SEMANTIC_FOV_LOW), list(SEMANTIC_FOV_HIGH)],
        'neighbor_slots': MIN_WAFFLE_POINTS,
        'mode': ('empty_semantic_context' if count == 0 else
                 'available_neighbor_repeat' if count is not None and count < MIN_WAFFLE_POINTS else
                 'native_17_neighbors'),
    }


class InsufficientSemanticPoints(ValueError):
    def __init__(self, raw_points, selected, where):
        self.diagnostics = point_diagnostics(raw_points, selected)
        super().__init__(f'Fewer than 17 finite points inside WaffleIron pretraining FOV '
                         f'or after voxelization ({where}); '
                         f'counts={self.diagnostics}. '
                         'Stage A target-supervised training remains strict; no frame was skipped.')


def point_selection(raw_points,cap=20000,voxel=.1,seed=0,*,allow_sparse=False):
    """Full-azimuth, depth-stratified deterministic cap AFTER voxelization.

    WI pretraining FOV is a model-input restriction, not an assertion that the
    original scan lacks observations outside it. Existing camera PCD remains.
    """
    if not isinstance(cap, (int, np.integer)) or cap < 17:
        raise ValueError('max_points must be an integer >=17; the sparse fallback is for data, not a reduced cap')
    if not np.isfinite(voxel) or voxel <= 0:
        raise ValueError('voxel_size must be finite and positive')
    if np.asarray(raw_points).ndim != 2 or raw_points.shape[1] < 4:
        raise ValueError('Point selection requires an [N, >=4] XYZI array')
    xyz=raw_points[:,:3];lo=np.array([-64,-64,-8]);hi=-lo
    inside=((xyz>=lo)&(xyz<hi)).all(1) & np.isfinite(xyz).all(1)
    candidates=np.flatnonzero(inside)
    if len(candidates)<17 and not allow_sparse:
        raise InsufficientSemanticPoints(raw_points, candidates, 'before_voxel')
    vox=np.floor(xyz[candidates]/voxel).astype(np.int32)
    _,first=np.unique(vox,axis=0,return_index=True);candidates=candidates[np.sort(first)]
    if len(candidates)>cap:
        voxel_candidates=candidates.copy()
        p=xyz[candidates];angle=(np.arctan2(p[:,1],p[:,0])+np.pi)/(2*np.pi)
        sector=np.minimum((angle*32).astype(int),31)
        radial=np.minimum((np.linalg.norm(p,axis=1)/16).astype(int),5)
        group=sector*6+radial;rng=np.random.default_rng(seed)
        # Proportional strata; every occupied stratum gets at least one point.
        groups=np.unique(group);pieces=[];left=cap-len(groups)
        quotas=np.floor(np.array([(group==g).sum() for g in groups])*left/len(group)).astype(int)+1
        while quotas.sum()<cap:quotas[np.argmax(np.array([(group==g).sum() for g in groups])-quotas)]+=1
        for g,q in zip(groups,quotas):
            ids=candidates[group==g];pieces.append(rng.choice(ids,size=min(int(q),len(ids)),replace=False))
        candidates=np.concatenate(pieces)
        if len(candidates)<cap:
            rest=np.setdiff1d(voxel_candidates,candidates)
            candidates=np.r_[candidates,rng.choice(rest,cap-len(candidates),replace=False)]
        candidates=np.sort(candidates)
    if len(candidates)<17 and not allow_sparse:
        raise InsufficientSemanticPoints(raw_points, candidates, 'after_voxel_and_cap')
    return candidates

def waffle_input(points,mean,std,*,allow_sparse=False):
    n=len(points)
    if n<17 and not allow_sparse:
        raise ValueError(f'Point neighborhood requires >=17 points; got {n}')
    xyz=points[:,:3];radius=np.linalg.norm(xyz,axis=1)
    feats=np.c_[(points[:,3]-mean)/max(std,1e-6),xyz,radius].astype(np.float32).T
    if n == 0:
        # Empty metadata-only item. Never pass it into the native point network.
        return {'feats':torch.from_numpy(feats),'cell_ind':torch.empty((3,0),dtype=torch.long),
                'neighbors':torch.empty((17,0),dtype=torch.long),'occupied':torch.empty(0),
                'xyz':torch.from_numpy(xyz.copy())}
    k=min(17,n)
    _,neighbors=cKDTree(xyz).query(xyz,k=k,workers=1)
    neighbors=np.asarray(neighbors,dtype=np.int64).reshape(n,k)
    if k<17:
        # Reuse a VALID observed neighbor index. No point/XYZ is added.
        # Repeating a valid last neighbor preserves a max-pooled neighborhood;
        # for a singleton the only available index is the center itself.
        neighbors=np.concatenate((neighbors,np.repeat(neighbors[:,-1:],17-k,axis=1)),axis=1)
    cells=[]
    for axes,shape in (((0,1),(256,256)),((0,2),(256,32)),((1,2),(256,32))):
        low=np.array([-64,-64,-8])[list(axes)];high=-low
        ij=np.floor((xyz[:,axes]-low)/(high-low)*np.asarray(shape)).astype(np.int64)
        ij=np.clip(ij,0,np.asarray(shape)-1);cells.append(ij[:,0]*shape[1]+ij[:,1])
    return {'feats':torch.from_numpy(feats),'cell_ind':torch.from_numpy(np.stack(cells)).long(),
            'neighbors':torch.from_numpy(neighbors.T.copy()).long(),
            'occupied':torch.ones(len(points),dtype=torch.float32),'xyz':torch.from_numpy(xyz.copy())}

def projection(raw,selected,spec,image_hw,tolerance_m=.25,tolerance_ratio=.01,subcell=8):
    from .camera import camera_plan
    if int(spec.get('width',0))>0 and int(spec.get('height',0))>0:
        source_wh=(int(spec['width']),int(spec['height']))
    else:
        # Import of an existing RGB dataset may recover size metadata once.
        # Raw deployment manifests must specify width and height.
        with Image.open(spec['image_path']) as im:source_wh=im.size
    K,_,_,_=camera_plan(source_wh,np.asarray(spec['intrinsics'],np.float32),tuple(image_hw),
                         spec.get('distortion'),spec.get('image_state'))
    T=np.asarray(spec['lidar_to_camera'],np.float32)
    cam=raw[:,:3]@T[:3,:3].T+T[:3,3];z=cam[:,2]
    q=cam@np.asarray(K).T
    uv=q[:,:2]/np.maximum(q[:,2:],1e-8)
    H,W=image_hw
    # Integer pixel z-buffer is formed from ALL raw points, including non-WI FOV points.
    valid=(z>0)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)
    pixel=np.floor(np.nan_to_num(uv,nan=-1,posinf=-1,neginf=-1)).astype(np.int64)
    ids=np.flatnonzero(valid);flat=pixel[ids,1]*W+pixel[ids,0]
    zbuf=np.full(H*W,np.inf,dtype=np.float32);np.minimum.at(zbuf,flat,z[ids])
    s=selected;ok=valid[s].copy();where=np.flatnonzero(ok)
    if len(where):
        ref=zbuf[pixel[s[where],1]*W+pixel[s[where],0]]
        ok[where]&=z[s[where]]<=ref+tolerance_m+tolerance_ratio*ref
    good=np.flatnonzero(ok)
    # Nearest measured point per 8x8 output cell. No foreground/background averaging.
    gh,gw=H//subcell,W//subcell
    cell=np.floor(uv[s[good]]/subcell).astype(np.int64)
    cell=np.minimum(cell,[gw-1,gh-1]);idx=cell[:,1]*gw+cell[:,0]
    order=np.lexsort((s[good],z[s[good]],idx));ids_sort=idx[order]
    unique=np.r_[True,ids_sort[1:]!=ids_sort[:-1]] if len(order) else np.zeros(0,bool)
    chosen=good[order[unique]];chosen_cell=idx[order[unique]]
    dt=float(spec.get('timestamp_delta_sec',0.0))
    out={'point_index':torch.from_numpy(good).long(),'uv':torch.from_numpy(uv[s[good]].astype(np.float32)),
         'weight':torch.full((len(good),),float(np.exp(-abs(dt)/.1))),
         'cell':torch.from_numpy(chosen_cell).long(),'chosen':torch.from_numpy(chosen).long(),
         'depth':torch.from_numpy(z[s[chosen]].astype(np.float32)),'out_K':torch.tensor(K,dtype=torch.float32),
         'visible_ratio':len(good)/max(1,len(s))}
    return out

def sample_map(feature,uv,image_hw):
    """Bilinear sample a patch-centered feature field in resized RGB coordinates."""
    import torch.nn.functional as F
    H,W=image_hw
    grid=torch.stack((2*(uv[:,0]+.5)/W-1,2*(uv[:,1]+.5)/H-1),-1).view(1,-1,1,2)
    return F.grid_sample(feature[None].float(),grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,:,0].T

def collate_points(items):
    if not items or any(x['feats'].shape[1] == 0 for x in items):
        raise ValueError('Empty point item must bypass the native point encoder; use encode_point_item')
    n=max(x['feats'].shape[1] for x in items);B=len(items)
    out={'feats':torch.zeros(B,5,n),'cell_ind':torch.zeros(B,3,n,dtype=torch.long),
         'neighbors':torch.zeros(B,17,n,dtype=torch.long),'occupied':torch.zeros(B,n)}
    for i,x in enumerate(items):
        k=x['feats'].shape[1]
        for key in ('feats','cell_ind','neighbors'):out[key][i,:,:k]=x[key]
        out['neighbors'][i,:,k:]=n-1;out['occupied'][i,:k]=1
    # Metadata/targets are intentionally kept out of the point-network input.
    out['items']=items
    return out

def projection_loss(pred,items,image_hw):
    import torch.nn.functional as F
    # pred[B,D,N] in the shared fixed teacher projection; balanced mean over views/scenes.
    losses=[];cosines=[];pairs=0
    for b,x in enumerate(items):
        each=[]
        for v,o in enumerate(x['projections']):
            idx=o['point_index'].to(pred.device)
            if not idx.numel():continue
            target=sample_map(x['teacher'][v].to(pred.device),o['uv'].to(pred.device),image_hw).detach()
            a=F.normalize(pred[b,:,idx].T.float(),dim=-1);t=F.normalize(target,dim=-1)
            w=o['weight'].to(pred.device);c=(a*t).sum(-1)
            each.append(((2-2*c).clamp_min(1e-8).sqrt()*w).sum()/w.sum().clamp_min(1e-6))
            cosines.append(c.detach().mean());pairs+=len(idx)
        if each:losses.append(torch.stack(each).mean())
    if not losses:raise ValueError('Batch has no visible point-teacher correspondences')
    return torch.stack(losses).mean(),{'cosine':float(torch.stack(cosines).mean()),'point_view_pairs':pairs}

def make_packet(pred,x,cfg):
    """Input is PCD student output, never an RGB teacher tensor."""
    import torch.nn.functional as F
    D,N=pred.shape;H,W=cfg['data']['image_hw'];gh,gw=H//8,W//8
    sparse=[]
    points=x['points'].to(pred.device)
    for o in x['projections']:
        ids=o['chosen'].to(pred.device);coords=points[ids]
        geom=torch.cat((coords[:,:3]/64,coords[:,3:4],o['depth'].to(pred.device)[:,None]/80),-1)
        sparse.append({'cell':o['cell'].cpu().int(),'features':F.normalize(pred[:,ids].T.float(),dim=-1).cpu().half(),
                       'geometry':geom.cpu().half()})
    xyz=x['xyz'].to(pred.device);theta=torch.atan2(xyz[:,1],xyz[:,0])
    sect=((theta+math.pi)/(2*math.pi)*32).long().clamp(0,31)
    feat=F.normalize(pred.T.float(),dim=-1)
    tokens=feat.new_zeros(32,D).index_add(0,sect,feat)
    counts=feat.new_zeros(32).index_add(0,sect,torch.ones(N,device=feat.device))
    tokens=tokens/counts.clamp_min(1)[:,None]
    packet={'version':'semantic_packet_v1','scene_uid':x['scene_uid'],'camera_names':x['camera_names'],
            'grid_hw':[gh,gw],'sparse_views':sparse,'scene_semantic':tokens.cpu().half(),
            'scene_valid':(counts>0).cpu(),'raw_points':x['raw_count'],'selected_points':N,
            'condition_source':'PCD_student_only','semantic_fov':x['semantic_fov']}
    if 'semantic_input' in x:
        packet['semantic_input']=dict(x['semantic_input'])
    return packet

def collate_generation(items):
    B=len(items);c0=items[0]['condition'];out={}
    for k in ('local_pcd','camera_condition','semantic','semantic_geometry','semantic_support','scene_semantic','scene_valid'):
        out[k]=torch.stack([x['condition'][k] for x in items])
    R=max(x['condition']['return_features'].shape[0] for x in items)
    dim=c0['return_features'].shape[-1]
    out['return_features']=torch.zeros(B,R,dim);out['return_valid']=torch.zeros(B,R,dtype=torch.bool)
    edges=[];weights=[]
    for b,x in enumerate(items):
        c=x['condition'];n=len(c['return_features']);out['return_features'][b,:n]=c['return_features'].float();out['return_valid'][b,:n]=True
        e=c['edge_index'].long();edges.append(torch.cat((torch.full((len(e),1),b,dtype=torch.long),e),1));weights.append(c['edge_weight'].float())
    out['edge_index']=torch.cat(edges);out['edge_weight']=torch.cat(weights)
    batch={'condition':out,'indices':[x['index'] for x in items],'rows':[x['row'] for x in items]}
    if 'target' in items[0]:batch['target']={k:torch.stack([x['target'][k] for x in items]) for k in ('latent','teacher')}
    return batch
