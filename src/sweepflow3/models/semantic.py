"""SweepFlow Full: native SD3.5 traversal with semantic control and CSE.

The condition API cannot read RGB teacher targets. Alignment is returned to the
training loss, never fed back into the condition. No GT feature substitution.
"""
from __future__ import annotations
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .blocks import ImageBlockResidual


class SemanticGeometry(nn.Module):
    def __init__(self,dim=128,width=128):
        super().__init__()
        self.width=width
        # Distinct 8-pixel measured subcells are learned BEFORE native 16-pixel token aggregation.
        self.semantic=nn.Sequential(nn.Conv2d(dim+6,width,1),nn.SiLU(),nn.Conv2d(width,width,2,2))
        self.raw_local=nn.Sequential(nn.Conv2d(6,width,2,2),nn.SiLU(),nn.Conv2d(width,width,3,padding=1))
        self.camera=nn.Sequential(nn.Linear(11,width),nn.SiLU(),nn.Linear(width,width))
        self.scene=nn.Sequential(nn.Linear(dim,width),nn.SiLU(),nn.Linear(width,width))
        self.direction=nn.Linear(3,width)
        self.read=nn.MultiheadAttention(width,4,batch_first=True)
        self.norm=nn.LayerNorm(width)
        self.ret=nn.Sequential(nn.Linear(10,width),nn.SiLU(),nn.LayerNorm(width))
    def forward(self,c):
        B,V,_,H,W=c['local_pcd'].shape
        if c['semantic'].shape[:2]!=(B,V) or c['semantic'].shape[-2:]!=(H,W):raise ValueError('Semantic packet and latent geometry grids differ')
        measured=torch.cat((c['semantic'].float(),c['semantic_geometry'].float(),c['semantic_support'].float()),2)
        h=self.semantic(measured.flatten(0,1))+self.raw_local(c['local_pcd'].float().flatten(0,1))
        h=h.flatten(2).transpose(1,2)
        if c['camera_condition'].shape[2]!=h.shape[1]:raise ValueError('Camera tokens not aligned with native grid')
        h=h+self.camera(c['camera_condition'].float().flatten(0,1))
        count=c['scene_semantic'].shape[1]
        theta=(torch.arange(count,device=h.device)+.5)*(2*torch.pi/count)-torch.pi
        pos=torch.stack((torch.sin(theta),torch.cos(theta),torch.ones_like(theta)),1)
        scene=self.scene(c['scene_semantic'].float())+self.direction(pos)[None]
        scene=scene[:,None].expand(-1,V,-1,-1).flatten(0,1)
        mask=~c['scene_valid'].bool()
        empty=mask.all(1)
        # No semantic-crop points is not no raw LiDAR. Keep local geometry/rays/R.
        # Attention needs at least one unmasked slot to avoid all-masked softmax.
        # This private computational slot is NEVER marked observed in the packet;
        # its complete attention output (including bias) is zeroed for empty scenes.
        if empty.any():
            mask=mask.clone();mask[empty,0]=False
        mask=mask[:,None].expand(-1,V,-1).flatten(0,1)
        message=self.read(self.norm(h),scene,scene,key_padding_mask=mask,need_weights=False)[0]
        if empty.any():
            message=message.masked_fill(empty[:,None].expand(-1,V).reshape(-1,1,1),0.)
        h=self.norm(h+message)
        ret=self.ret(c['return_features'].float())*c['return_valid'][...,None]
        return h.view(B,V,h.shape[1],self.width),ret


class SemanticJointFlow(nn.Module):
    def __init__(self,cfg,*,transformer=None):
        super().__init__();m=cfg['model'];s=cfg['semantic']
        if transformer is None:
            from diffusers import SD3Transformer2DModel
            transformer=SD3Transformer2DModel.from_pretrained(m['sd35_root'],subfolder='transformer',local_files_only=True,torch_dtype=torch.bfloat16)
        self.transformer=transformer.eval().requires_grad_(False)
        self.patch=int(transformer.config.patch_size);self.channels=int(transformer.config.in_channels)
        if self.patch!=2:raise ValueError('Native patch2 required by saved geometry')
        self.blocks=tuple(m['internal_blocks']);D=int(transformer.inner_dim)
        if any(i<0 or i>=len(transformer.transformer_blocks) for i in self.blocks):raise ValueError('Invalid native insertion block')
        self.geometry=SemanticGeometry(s['dim'],m['internal_width'])
        self.adapters=nn.ModuleDict({str(i):ImageBlockResidual(D,m['internal_width']) for i in self.blocks})
        self.alignment_block=int(s['alignment_block'])
        if self.alignment_block not in self.blocks:raise ValueError('Alignment must read an adapted internal block')
        self.alignment_head=nn.Sequential(nn.LayerNorm(D,elementwise_affine=False),nn.Linear(D,s['dim']))
        self.use_r=True;self.r_scale=1.;self.geometry_scale=1.
        self.activation_checkpointing=bool(m.get('activation_checkpointing',True))
    def train(self,mode=True):super().train(mode);self.transformer.eval();return self
    def trainable_parameter_count(self):return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def _block(self,i,h,c,t):
        out=self.transformer.transformer_blocks[i](hidden_states=h,encoder_hidden_states=c,temb=t)
        if not isinstance(out,(tuple,list)) or len(out)!=2:raise TypeError('Unsupported SD3 block return contract')
        return out
    def forward(self,noisy,times,sigma,condition,prompt,pooled,*,return_alignment=False):
        B,V,C,H,W=noisy.shape
        if (C!=self.channels) or H%2 or W%2:raise ValueError('Native latent shape mismatch')
        times=times.reshape(B,V)
        if not torch.equal(times,times[:,:1].expand_as(times)):raise ValueError('Multi-view flow requires scene-shared time')
        if not torch.equal(sigma.reshape(B,V),sigma.reshape(B,V)[:,:1].expand(B,V)):raise ValueError('Multi-view sigma mismatch')
        geo,ret=self.geometry(condition);n=self.transformer;dtype=next(n.parameters()).dtype
        h=n.pos_embed(noisy.flatten(0,1).to(dtype));t=n.time_text_embed(times.flatten().float(),pooled.to(dtype))
        context=n.context_embedder(prompt.to(dtype));align=None
        if h.shape[1]!=geo.shape[2]:raise ValueError('Native spatial token order mismatch')
        for i in range(len(n.transformer_blocks)):
            fn=partial(self._block,i)
            context,h=checkpoint(fn,h,context,t,use_reentrant=False) if self.training and self.activation_checkpointing else fn(h,context,t)
            if str(i) in self.adapters:
                h,_=self.adapters[str(i)](h,geo,ret,condition['edge_index'],condition['edge_weight'],sigma,
                                           self.geometry_scale,self.r_scale if self.use_r else 0.,False)
            if i==self.alignment_block and return_alignment:
                align=self.alignment_head(h.float()).reshape(B,V,H//2,W//2,-1).permute(0,1,4,2,3)
        h=n.proj_out(n.norm_out(h,t)).reshape(B*V,H//2,W//2,2,2,C)
        v=h.permute(0,5,1,3,2,4).reshape(B,V,C,H,W)+ret.sum().to(h.dtype)*0
        if return_alignment:return v,align
        return v


def alignment_loss(pred,teacher,support):
    """Teacher supervises a returned feature only; unsupported RGB still has FM loss."""
    B,V,D,H,W=pred.shape
    target=F.interpolate(teacher.detach().float().flatten(0,1),size=(H,W),mode='bilinear',align_corners=False)
    a=F.normalize(pred.float().flatten(0,1),dim=1);t=F.normalize(target,dim=1)
    mask=F.avg_pool2d(support.float().flatten(0,1),2,2).squeeze(1)
    value=(1-(a*t).sum(1))*mask
    return value.sum()/mask.sum().clamp_min(1)
