from __future__ import annotations
import torch
from .runtime import configure_timesteps


@torch.no_grad()
def sample_sd35_fm(model,scheduler,batch,prompt_embeds,pooled_prompt_embeds,*,latent_shape,
                   generator=None,device,steps):
    B,V,C,H,W=latent_shape
    latents=torch.randn(latent_shape,generator=generator,device=device,dtype=torch.float32)
    configure_timesteps(scheduler,steps,device,(H//2)*(W//2))
    for i,t in enumerate(scheduler.timesteps):
        ts=t.to(device).expand(B,V)
        sigma=scheduler.sigmas[i].to(device).reshape(1,1,1,1,1).expand(B,V,1,1,1)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            v=model(latents,ts,sigma,batch,prompt_embeds,pooled_prompt_embeds)
        latents=scheduler.step(v.float().flatten(0,1),t,latents.flatten(0,1),return_dict=True).prev_sample.reshape(latent_shape)
    return latents
