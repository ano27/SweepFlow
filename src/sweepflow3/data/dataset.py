from __future__ import annotations
from pathlib import Path
import torch
import numpy as np
from ..common import rows_for,packet_path,stats_path,load_torch,read_json
from .readers import read_points
from .points import point_selection,waffle_input,projection,point_diagnostics,InsufficientSemanticPoints


class PointScenes(torch.utils.data.Dataset):
    def __init__(self,cfg,split,targets=True,*,rows=None,stats=None):
        self.cfg,self.split,self.targets=cfg,split,targets;self.rows=rows_for(cfg,split) if rows is None else rows;self.stats=read_json(stats_path(cfg)) if stats is None else stats
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];raw,available,ids=read_points(row);cfg=self.cfg
        try:
            selected=point_selection(raw,cfg['points']['max_points'],cfg['points']['voxel_size'],
                                     cfg['split_seed']+int(row.get('index',i)),allow_sparse=not self.targets)
        except InsufficientSemanticPoints as error:
            raise ValueError(f'{error} dataset={cfg["dataset"]} split={self.split} '
                             f'index={i} scene_uid={row["scene_uid"]} '
                             f'lidar_path={row["lidar_path"]}') from error
        p=raw[selected];x=waffle_input(p,self.stats['mean'],self.stats['std'],allow_sparse=not self.targets)
        if not available:x['feats'][0].zero_()
        x.update(scene_index=i,scene_uid=row['scene_uid'],camera_names=row['camera_names'],points=torch.from_numpy(p),
                 projections=[projection(raw,selected,s,cfg['data']['image_hw']) for s in row['camera_specs']],raw_count=len(raw),
                 selected_count=len(selected),semantic_fov=[[-64,-64,-8],[64,64,8]],intensity_available=available)
        if len(selected)<17:
            info=point_diagnostics(raw,selected)
            info.update(dataset=cfg['dataset'],split=self.split,index=int(i),
                        scene_uid=row['scene_uid'],sequence=row.get('sequence'),
                        lidar_path=row['lidar_path'],intensity_available=bool(available),
                        model_from_lidar=row['model_from_lidar'],
                        raw_geometry_preserved=True,frame_excluded=False)
            x['semantic_input']=info
        if self.targets:
            t=load_torch(packet_path(cfg,'teacher',self.split,i))
            if t['scene_uid']!=row['scene_uid']:raise ValueError('Teacher scene mismatch')
            x['teacher']=t['feature'].float()
        return x


class GenerationScenes(torch.utils.data.Dataset):
    def __init__(self,cfg,split,targets=True,*,rows=None):
        self.cfg,self.split,self.targets=cfg,split,targets;self.rows=rows_for(cfg,split) if rows is None else rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        cfg=self.cfg;r=self.rows[i];encoder=cfg.get('encoder','adapted')
        g=load_torch(packet_path(cfg,'geometry',self.split,i));s=load_torch(packet_path(cfg,'student',self.split,i,encoder))
        if g['scene_uid']!=r['scene_uid'] or s['scene_uid']!=r['scene_uid']:raise ValueError('PCD packet scene mismatch')
        if s['camera_names']!=r['camera_names']:raise ValueError('PCD packet camera order mismatch')
        V=len(r['camera_names']);D=cfg['semantic']['dim'];H,W=s['grid_hw'];feature=torch.zeros(V,D,H*W);geo=torch.zeros(V,5,H*W);mask=torch.zeros(V,1,H*W)
        for v,item in enumerate(s['sparse_views']):
            idx=item['cell'].long();feature[v,:,idx]=item['features'].float().T;geo[v,:,idx]=item['geometry'].float().T;mask[v,0,idx]=1
        c={k:g[k].float() if k not in ('edge_index','return_ids') else g[k].long() for k in ('local_pcd','camera_condition','return_features','edge_index','edge_weight')}
        c.update(semantic=feature.reshape(V,D,H,W),semantic_geometry=geo.reshape(V,5,H,W),semantic_support=mask.reshape(V,1,H,W),
                 scene_semantic=s['scene_semantic'].float(),scene_valid=s['scene_valid'].bool())
        result={'condition':c,'index':i,'row':r}
        if self.targets:
            z=load_torch(packet_path(cfg,'latent',self.split,i));t=load_torch(packet_path(cfg,'teacher',self.split,i))
            if z['scene_uid']!=r['scene_uid'] or t['scene_uid']!=r['scene_uid']:raise ValueError('Target scene mismatch')
            result['target']={'latent':z['latent'].float(),'teacher':t['feature'].float()}
        return result
