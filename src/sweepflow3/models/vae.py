"""Native SD3.5 VAE, frozen FP32, deterministic posterior mode."""
from pathlib import Path
import torch
from torch import nn


class SD35Autoencoder(nn.Module):
    def __init__(self, root):
        super().__init__()
        from diffusers import AutoencoderKL
        root = Path(root).expanduser().resolve()
        self.vae = AutoencoderKL.from_pretrained(str(root), subfolder='vae',
                       local_files_only=True, torch_dtype=torch.float32).eval().requires_grad_(False)
        c = self.vae.config
        self.latent_channels = int(c.latent_channels)
        self.scaling_factor = float(c.scaling_factor)
        self.shift_factor = float(c.shift_factor or 0.)
        self.spatial_compression = 2 ** (len(c.block_out_channels)-1)
        if self.latent_channels != 16 or self.spatial_compression != 8:
            raise ValueError('Expected the native F8/C16 SD3.5 VAE.')
        # Bound memory across views. This does not pad or crop images.
        if hasattr(self.vae, 'enable_slicing'):
            self.vae.enable_slicing()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def encode(self, image):
        if image.ndim != 4 or image.shape[1] != 3 or not torch.isfinite(image).all():
            raise ValueError('Expected finite BCHW RGB [0,1].')
        if any(x % 8 for x in image.shape[-2:]):
            raise ValueError('Native VAE image size must be divisible by 8.')
        with torch.autocast(device_type=image.device.type, enabled=False):
            z = self.vae.encode(image.float()*2-1).latent_dist.mode()
        return (z-self.shift_factor)*self.scaling_factor

    @torch.no_grad()
    def decode(self, z):
        if z.ndim != 4 or z.shape[1] != 16 or not torch.isfinite(z).all():
            raise ValueError('Expected finite BCHW C16 native SD3.5 latent.')
        with torch.autocast(device_type=z.device.type, enabled=False):
            image = self.vae.decode(z.float()/self.scaling_factor+self.shift_factor).sample
        return (image*.5+.5).clamp(0,1)
