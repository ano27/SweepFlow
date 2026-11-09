from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
def rms(x):
    return x.detach().float().square().mean().sqrt()

def return_view_features(tokens, edges, weights, returns):
    """Gather bilinear corners per (scene, return, VIEW), then exclude own view.

    All accumulation is FP32. No cross-scene edge and no within-view bilinear
    corner can become peer evidence. Output vectors are aligned with edge rows.
    """
    B, V, N, D = tokens.shape
    if not edges.numel() or returns == 0:
        empty = tokens.new_zeros((0, D))
        return empty, empty, tokens.new_zeros((0, 1))
    e = edges.long()
    source = tokens[e[:, 0], e[:, 1], e[:, 2]].float()
    w = weights.float().view(-1, 1)
    group = (e[:, 0] * returns + e[:, 3]) * V + e[:, 1]
    sums = source.new_zeros(B * returns * V, D).index_add(0, group, source * w)
    mass = source.new_zeros(B * returns * V, 1).index_add(0, group, w)
    sums, mass = sums.view(B, returns, V, D), mass.view(B, returns, V, 1)
    per_view = sums / mass.clamp_min(1e-8)
    visible = (mass > 0).float()
    own = per_view[e[:, 0], e[:, 3], e[:, 1]]
    total = (per_view * visible).sum(2)[e[:, 0], e[:, 3]]
    count = visible.sum(2)[e[:, 0], e[:, 3]] - visible[e[:, 0], e[:, 3], e[:, 1]]
    peer = (total - own) / count.clamp_min(1)
    valid = (count > 0).to(tokens.dtype)
    return own.to(tokens.dtype), (peer * valid).to(tokens.dtype), valid

def splat_messages(message, valid, edges, weights, shape):
    B, V, N, D = shape
    if not edges.numel():
        return message.new_zeros(shape)
    e = edges.long()
    index = (e[:, 0] * V + e[:, 1]) * N + e[:, 2]
    w = weights.float().view(-1, 1) * valid.float()
    accum = torch.zeros(B*V*N, D, device=message.device, dtype=torch.float32)
    accum = accum.index_add(0, index, message.float() * w)
    mass = torch.zeros(B*V*N, 1, device=message.device, dtype=torch.float32).index_add(0, index, w)
    return (accum / mass.clamp_min(1e-8)).view(shape).to(message.dtype)

class ImageBlockResidual(nn.Module):
    def __init__(self, hidden_dim, width):
        super().__init__()
        self.width = width
        self.image_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.image_down = nn.Linear(hidden_dim, width)
        self.time = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width,width))
        self.geo_fuse = nn.Sequential(nn.Linear(3*width,width), nn.SiLU(), nn.Linear(width,width), nn.SiLU())
        self.geo_out = nn.Linear(width, hidden_dim)
        self.peer_gate = nn.Sequential(nn.Linear(4*width+1,width),nn.SiLU(),nn.Linear(width,1),nn.Sigmoid())
        self.peer_value = nn.Linear(width,width, bias=False)
        self.peer_out = nn.Linear(width,hidden_dim, bias=False)
        for head in (self.geo_out,self.peer_out):
            nn.init.zeros_(head.weight)
            if head.bias is not None: nn.init.zeros_(head.bias)

    def forward(self, hidden, geometry, returns, edges, weights, sigma,
                geometry_scale=1., r_scale=1., collect=False):
        B,V,N,D = geometry.shape
        h = hidden.view(B,V,N,-1)
        state = self.image_down(self.image_norm(h.float()))
        s = sigma.reshape(B,-1)[:,0].float()
        sf = torch.stack((s, torch.sin(math.pi*s),torch.cos(math.pi*s)), -1)
        time = self.time(sf)[:,None,None].expand(B,V,N,-1)
        geo_delta = self.geo_out(self.geo_fuse(torch.cat((state,geometry,time),-1)))
        own,peer,valid = return_view_features(state,edges,weights,returns.shape[1])
        if edges.numel() and returns.shape[1]:
            e = edges.long()
            rg = returns[e[:,0],e[:,3]]
            ray_local = geometry[e[:,0],e[:,1],e[:,2]]
            gate = self.peer_gate(torch.cat((own,peer,rg,ray_local,s[e[:,0],None].to(own.dtype)),-1))
            # Only OTHER-view values are transmitted. A bias cannot manufacture
            # a cross-view value when there is no peer.
            messages = self.peer_value(peer) * gate
            update = splat_messages(messages,valid,edges,weights,state.shape)
            peer_delta = self.peer_out(update)
        else:
            gate = state.new_zeros((0,1))
            zero = sum(p.sum()*0 for m in (self.peer_gate,self.peer_value,self.peer_out) for p in m.parameters())
            peer_delta = h.new_zeros(h.shape) + zero
        delta = float(geometry_scale)*geo_delta + float(r_scale)*peer_delta
        stats = None
        if collect:
            scale = rms(h).clamp_min(1e-8)
            stats = {'hidden_rms':float(scale),'geometry_delta_rms':float(rms(geo_delta)),
                     'return_delta_rms':float(rms(peer_delta)),
                     'geometry_delta_relative':float(rms(geo_delta)/scale),
                     'return_delta_relative':float(rms(peer_delta)/scale),
                     'peer_gate_mean':float(gate.detach().float().mean()) if gate.numel() else None,
                     'peer_edges':int(valid.sum())}
        return (hidden + delta.flatten(0,1).to(hidden.dtype)), stats
