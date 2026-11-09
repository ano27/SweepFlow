# SweepFlow

SweepFlow generates a complete set of RGB camera images from a single LiDAR sweep and the camera calibration. Its Cross-View State Exchange (CSE) module passes evolving image features between views through the projections of shared measured points. RGB images are used only as training supervision.

This repository contains the full SweepFlow model and its two-stage training pipeline for three datasets:

| Dataset | Cameras | Config |
| --- | --- | --- |
| ONCE | 7 | `configs/once.yaml` |
| nuScenes | 6 | `configs/nuscenes.yaml` |
| DSERT-RoLL | 2 | `configs/dsert_roll.yaml` |

Datasets and model weights are not included.

## 1. Installation

Use Python 3.10 or later in a dedicated environment. Install a CUDA build of PyTorch and torchvision by following the [official PyTorch instructions](https://pytorch.org/get-started/locally/). Then install this package from the repository root:

```bash
python -m pip install -e .
sweepflow --help
```

`python -m sweepflow3` runs the same commands as `sweepflow`. Diffusers is pinned to 0.35.2 because the model works directly on the native SD3 transformer blocks. Training requires CUDA and enough GPU memory for a complete camera set.

## 2. Configuration

Pick one of the three configs and edit these paths:

| Key | Meaning | Default (ONCE) |
| --- | --- | --- |
| `dataset_config.root` | Dataset root | `./data/once` |
| `workdir` | Location for caches, checkpoints, and logs | `./outputs/once` |
| `assets_root` | Location for downloaded auxiliary assets | `./assets` |
| `model.sd35_root` | SD3.5 Medium Diffusers directory | `./pretrained/stable-diffusion-3.5-medium` |

Relative paths are resolved from the directory you run the command in. The default configs use a 336 × 640 output resolution, 128-dimensional semantic features, and CSE inserted at transformer blocks 2, 6, 10, 14, 18, and 22.

## 3. Pretrained assets

**Stable Diffusion 3.5 Medium.** First obtain access under the model terms on [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium). Then download the **complete Diffusers repository**. The directory must contain `model_index.json`, `transformer/`, `vae/`, `scheduler/`, and all three text encoder and tokenizer directories. A single `.safetensors` file will not work.

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="stabilityai/stable-diffusion-3.5-medium", local_dir="pretrained/stable-diffusion-3.5-medium")'
```

**Auxiliary assets.** Run:

```bash
sweepflow assets --config configs/once.yaml
```

This command downloads:

- the WaffleIron source code,
- the annotation-free ScaLR WI-48-768 pretraining checkpoint,
- the DINOv2 source code and the non-register DINOv2 ViT-L/14 teacher weights.

It also encodes the fixed empty prompt once on the CPU. It does not download datasets or SD3.5 weights. Valid entries already listed in `assets/assets.json` are reused. If you move the SD3.5 directory, run the command again to regenerate the empty-prompt cache.

**Offline setup.** You can write `assets/assets.json` yourself. Relative entries are resolved against the directory that contains the file:

```json
{
  "waffle_source": "WaffleIron",
  "scalr_weights": "scalr_wi48_768.pth",
  "dino_source": "dinov2",
  "dino_weights": "dinov2_vitl14_pretrain.pth"
}
```

The WaffleIron source directory must contain its `waffleiron/` package. Each downloaded source records its upstream revision in `UPSTREAM.json`.

## 4. Dataset preparation

Expected layouts:

| Config | Expected input layout | Camera order |
| --- | --- | --- |
| `once.yaml` | `ImageSets/train.txt`, `raw_small.txt`, `val.txt`; the sequence JSON, `lidar_roof/`, and camera directories under `data/<sequence>/` | cam01, cam03, cam05, cam06, cam07, cam08, cam09 |
| `nuscenes.yaml` | The full `v1.0-trainval/` JSON metadata and the sensor files it references | CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_RIGHT, CAM_BACK, CAM_BACK_LEFT |
| `dsert_roll.yaml` | `split/train.txt`, `split/val.txt`, and `<condition>/<sequence>/label.pkl` with the LiDAR and RGB files it references | RGB_L, RGB_R |

**Splits.** The official validation set becomes `test`. About 10% of the training sequences are held out as `dev`. No sequence appears in more than one split. For nuScenes, the official validation scene names bundled with the package are used. For ONCE, the training split includes both the `train` and `raw_small` lists. Object labels are never used.

Run the following before training:

```bash
# 1) Build the manifests (single process)
sweepflow prepare --config configs/once.yaml --kind manifests

# 2) Cache LiDAR geometry, VAE latents, and teacher targets for the training split
sweepflow prepare --config configs/once.yaml --kind base --split train
```

- Building the manifests reads image headers to record each camera's native resolution.
- Intensity statistics are fitted on the training LiDAR only.
- All generated files are written under `workdir`, and the original dataset is left untouched. Cache items that already exist are reused.
- If you change the data or conditioning settings, use a new `workdir`.
- Once the manifests are built, you can run step 2 with `torchrun` to shard the work across GPUs.

## 5. Training

Training has two stages.

| Stage | What is trained | Default schedule |
| --- | --- | --- |
| **Stage A** | The point encoder is adapted to the frozen DINOv2 image teacher | 2 epochs, global batch of 8 scenes |
| **Stage B** | The point encoder and the SD3.5 backbone are frozen. The conditioning modules, CSE adapters, and visual alignment head are trained | 20,000 updates, global batch of 32 scenes |

Each scene always contains every camera in the config.

```bash
# Stage A: train the point representation
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m sweepflow3 train --config configs/once.yaml --stage a

# Export the Stage A point features used as Stage B conditions
sweepflow prepare --config configs/once.yaml --kind features --split train

# Stage B: train the full SweepFlow model
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m sweepflow3 train --config configs/once.yaml --stage b
```

**Single GPU.** Run `sweepflow train --config configs/once.yaml --stage a` (or `--stage b`) directly.

**Running out of memory.** Lower `--microbatch`, for example `--microbatch 1`. Gradients are accumulated, so the global batch does not change, even when the number of GPUs does not divide it evenly.

**Resuming.** A run automatically resumes from `checkpoint_last.pt`. The optimizer state, EMA, data cursor, and learning-rate schedule are all restored. `--through 1000` stops after update 1,000 but keeps the full schedule. To continue, run the same command again without `--through`.

**Notes.**

- Finish Stage A before you export point features.
- If you retrain Stage A, use a new `workdir` or rebuild the feature cache before Stage B.

With the default ONCE config, the outputs are:

| Output | Path |
| --- | --- |
| Stage A checkpoint | `outputs/once/data/once/stage_a/checkpoint_last.pt` |
| Stage B checkpoint | `outputs/once/runs/once/full/seed_0/stage_b/checkpoint_last.pt` |
| Intensity statistics | `outputs/once/data/once/intensity_statistics.json` |

## Third-party components

External weights and datasets are distributed by their original providers. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for sources and licenses.
