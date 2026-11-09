"""Strict native ScaLR WI-48-768 and DINOv2 ViT-L/14 loading.

The 1024-D pretrained features are projected by a FIXED orthonormal 128-D
matrix for storage. Both teachers and students share exactly the same matrix.
Stage A adapts the point encoder separately for each dataset.
"""
from __future__ import annotations
import os,sys
from functools import partial
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from ..common import load_torch,ensure_finite


def projection_matrix(dim=128,seed=7351,source_dim=1024):
    if not 1<=dim<=source_dim:raise ValueError('Projection dimension outside [1,1024]')
    if dim==source_dim:return torch.eye(source_dim)
    g=torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(source_dim,dim,generator=g),mode='reduced').Q.contiguous()


def compact_feature(features,matrix):
    return F.normalize(F.normalize(features.float(),dim=-1)@matrix.float(),dim=-1,eps=1e-6)


def scalr_state(path):
    # Load only a trusted published pretraining checkpoint or the user's own run.
    payload=load_torch(path,trusted=True)
    if isinstance(payload,dict):
        for k in ('model_point','model_points'):
            if k in payload:payload=payload[k];break
    if not isinstance(payload,dict) or not all(torch.is_tensor(v) for v in payload.values()):
        raise ValueError('Expected official ScaLR model_point/model_points state dictionary')
    return {k.removeprefix('module.'):v for k,v in payload.items()}


class ScaLRPoints(nn.Module):
    def __init__(self,assets=None,dim=128,projection_seed=7351,checkpoint_blocks=4,*,native=None,pretrained=True):
        super().__init__()
        if native is None:
            sys.path.insert(0,str(Path(assets['waffle_source']).resolve()))
            from waffleiron.segmenter import Segmenter
            native=Segmenter(5,768,1024,48,[[256,256],[256,32],[256,32]],drop_path_prob=0,layer_norm=True)
            if pretrained:native.load_state_dict(scalr_state(assets['scalr_weights']),strict=True)
        if hasattr(native,'embed') and not hasattr(native.embed,'compressed'):
            # The published uncompressed weights use the ordinary embedding
            # branch; some upstream revisions omit initializing this flag.
            native.embed.compressed=False
        self.native=native
        self.checkpoint_blocks=int(checkpoint_blocks)
        self.register_buffer('projection',projection_matrix(dim,projection_seed))
        self.train(True)
    def train(self,mode=True):
        super().train(mode)
        # Preserve source BN statistics. Do not update them using padded point sets.
        for m in self.native.modules():
            if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()
        return self
    def raw_forward(self,x):
        n=self.native
        if not self.training or self.checkpoint_blocks<=0:
            return n(x['feats'],x['cell_ind'],x['occupied'],x['neighbors'])
        from waffleiron.backbone import get_all_projections
        tokens=checkpoint(n.embed,x['feats'],x['neighbors'],use_reentrant=False)
        B,C,N=tokens.shape;w=n.waffleiron
        mats=get_all_projections(x['cell_ind'],C,B,N,x['occupied'],tokens.device,w.grids_shape,tokens.dtype)
        def group(t,start,end):
            for i in range(start,end):
                t=w.spatial_mix[i](t,mats[i%len(mats)]);t=w.channel_mix[i](t)
            return t
        for start in range(0,w.depth,self.checkpoint_blocks):
            tokens=checkpoint(partial(group,start=start,end=min(start+self.checkpoint_blocks,w.depth)),tokens,use_reentrant=False)
        return n.classif(tokens)
    def forward(self,x):
        # No image, target or teacher fields are passed into the native encoder.
        allowed={k:x[k] for k in ('feats','cell_ind','occupied','neighbors')}
        result=self.raw_forward(allowed)
        if result.ndim!=3 or result.shape[1]!=1024:raise ValueError('Expected ScaLR pretraining 1024-D output, not class logits')
        return compact_feature(result.transpose(1,2),self.projection).transpose(1,2)


class DinoTeacher(nn.Module):
    def __init__(self,assets,dim=128,projection_seed=7351,teacher_hw=(336,644)):
        super().__init__()
        os.environ.setdefault('XFORMERS_DISABLED','1')
        source=Path(assets['dino_source']).resolve();sys.path.insert(0,str(source))
        from dinov2.hub.backbones import dinov2_vitl14
        self.native=dinov2_vitl14(pretrained=False)
        weights=load_torch(assets['dino_weights'])
        self.native.load_state_dict(weights,strict=True)
        if self.native.num_register_tokens!=0:raise ValueError('ScaLR target requires non-register DINOv2 ViT-L/14')
        self.native.eval().requires_grad_(False)
        self.teacher_hw=tuple(teacher_hw)
        if any(s%14 for s in self.teacher_hw):raise ValueError('DINO input H,W must both be multiples of 14')
        self.register_buffer('projection',projection_matrix(dim,projection_seed))
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
    def train(self,mode=True):super().train(False);return self
    @torch.no_grad()
    def forward(self,rgb):
        x=F.interpolate(rgb.float(),size=self.teacher_hw,mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
        out=self.native.forward_features((x-self.mean)/self.std)
        # ScaLR x_pre_norm corresponds to the public DINOv2 x_prenorm before final LayerNorm.
        tokens=out['x_prenorm'][:,1:]
        H,W=self.teacher_hw;expected=(H//14)*(W//14)
        if tokens.shape[1:]!=(expected,1024):raise ValueError(f'Unexpected DINO patch grid: {tokens.shape}')
        f=compact_feature(tokens,self.projection)
        ensure_finite(f,'teacher feature')
        return f.transpose(1,2).reshape(len(rgb),self.projection.shape[1],H//14,W//14)
