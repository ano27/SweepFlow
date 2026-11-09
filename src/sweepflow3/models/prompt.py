"""Encode the native empty text once, before loading GPU denoisers."""
from pathlib import Path
import os
import torch


def prepare_empty_prompt(model_root, output):
    output = Path(output)
    root = str(Path(model_root).expanduser().resolve())
    if output.is_file():
        value = torch.load(output, map_location='cpu', weights_only=True)
        if value.get('model_root') != root:
            raise ValueError('Existing empty-prompt cache belongs to a different model root.')
        return
    from diffusers import StableDiffusion3Pipeline
    # Text encoders are CPU-only and destroyed before GPU cache/training begins.
    pipe = StableDiffusion3Pipeline.from_pretrained(root, transformer=None, vae=None,
                    torch_dtype=torch.float32, local_files_only=True)
    with torch.inference_mode():
        p, _, pooled, _ = pipe.encode_prompt(prompt='', prompt_2='', prompt_3='',
                    device=torch.device('cpu'), do_classifier_free_guidance=False,
                    num_images_per_prompt=1, max_sequence_length=256)
    value = {'model_root': root, 'prompt': p[:1].float().cpu().clone(),
             'pooled': pooled[:1].float().cpu().clone(), 'text': '', 'max_sequence_length':256}
    if not torch.isfinite(value['prompt']).all() or not torch.isfinite(value['pooled']).all():
        raise ValueError('Native empty-text embeddings are not finite.')
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name+f'.tmp-{os.getpid()}')
    torch.save(value,tmp); os.replace(tmp,output)
    del pipe


class EmptyPromptEmbeds:
    def __init__(self, model_root, device, *, cache_path):
        value = torch.load(cache_path, map_location='cpu', weights_only=True)
        if value['model_root'] != str(Path(model_root).expanduser().resolve()):
            raise ValueError('Empty-prompt cache model root mismatch.')
        self.prompt = value['prompt'].to(device)
        self.pooled = value['pooled'].to(device)

    def get(self, batch):
        return self.prompt.expand(batch,-1,-1), self.pooled.expand(batch,-1)
