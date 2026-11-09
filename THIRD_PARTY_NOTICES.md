# Third-party sources and assets

SweepFlow loads the following components from their official distributions. Their code, weights, and data retain the terms set by the respective providers; this package does not redistribute those weights or datasets.

| Component | Source and terms |
| --- | --- |
| Stable Diffusion 3.5 Medium | [Model card, access conditions, and license](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) |
| Diffusers | [Source and license](https://github.com/huggingface/diffusers) |
| WaffleIron | [Source and license](https://github.com/valeoai/WaffleIron) |
| ScaLR | [Pretraining method, checkpoint links, and license](https://github.com/valeoai/ScaLR) |
| DINOv2 | [Source, checkpoints, and license](https://github.com/facebookresearch/dinov2) |
| nuScenes | [Dataset terms](https://www.nuscenes.org/terms-of-use); [official split definitions](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/splits.py) |
| ONCE | [Dataset website and access terms](https://once-for-auto-driving.github.io/) |
| DSERT-RoLL | Obtain the dataset and comply with the terms accompanying its official distribution. |

The bundled `src/sweepflow3/data/nuscenes_val_scenes.json` contains the official validation scene names used by the source implementation. Downloaded source archives retain their upstream license files. No new license grant for SweepFlow-specific code is introduced by this packaging revision.
