"""RGB-free condition preparation and joint camera-set sampling."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .common import (atomic_json, atomic_jsonl, atomic_torch, device, load_torch,
                     move, packet_path, point_checkpoint as default_point_checkpoint,
                     rank, read_json, read_jsonl, rows_for, stats_path, world)
from .data.cache import geometry_packet, load_point, encode_point_item
from .data.dataset import PointScenes, GenerationScenes
from .data.points import make_packet, collate_generation
from .data.readers import rigid
from .models.semantic import SemanticJointFlow
from .models.runtime import restore_adapter, load_scheduler
from .models.prompt import EmptyPromptEmbeds
from .models.vae import SD35Autoencoder
from .models.sampler import sample_sd35_fm
from .progress import iterate


def inference_rows(rows, cfg, base):
    """Keep only PCD/calibration fields, resolving paths without opening RGB."""
    expected = cfg["data"]["camera_names"]
    result = []
    for i, row in enumerate(rows):
        if row["camera_names"] != expected:
            raise ValueError(f"Row {i}: camera order differs from the configured rig")
        if row.get("dataset", cfg["dataset"]) != cfg["dataset"]:
            raise ValueError(f"Row {i}: dataset differs from the trained configuration")
        lidar = Path(row["lidar_path"]).expanduser()
        lidar = lidar if lidar.is_absolute() else Path(base) / lidar
        if not lidar.is_file():
            raise FileNotFoundError(lidar)
        if len(row["camera_specs"]) != len(expected):
            raise ValueError(f"Row {i}: missing camera calibration")
        specs = []
        for name, spec in zip(expected, row["camera_specs"]):
            if spec["name"] != name or min(int(spec["width"]), int(spec["height"])) <= 0:
                raise ValueError(f"Row {i}: invalid camera name or native image dimensions")
            k = np.asarray(spec["intrinsics"], dtype=float)
            if k.shape != (3, 3) or not np.isfinite(k).all():
                raise ValueError(f"Row {i}: invalid intrinsic matrix")
            dt = float(spec["timestamp_delta_sec"])
            if not np.isfinite(dt):
                raise ValueError(f"Row {i}: nonfinite camera time offset")
            specs.append({"name": name, "width": int(spec["width"]), "height": int(spec["height"]),
                          "intrinsics": k.tolist(), "lidar_to_camera": rigid(spec["lidar_to_camera"]).tolist(),
                          "image_state": spec["image_state"], "distortion": spec.get("distortion"),
                          "timestamp_delta_sec": dt})
        index = int(row.get("index", i))
        if index < 0:
            raise ValueError("Manifest indices must be nonnegative")
        result.append({"scene_uid": str(row["scene_uid"]), "sequence": str(row.get("sequence", "deployment")),
                       "dataset": cfg["dataset"], "index": index,
                       "lidar_path": str(lidar.resolve()), "point_dim": int(row.get("point_dim", 4)),
                       "model_from_lidar": rigid(row["model_from_lidar"]).tolist(),
                       "camera_names": expected.copy(), "camera_specs": specs})
    if not result:
        raise ValueError("Inference manifest is empty")
    if len({row["scene_uid"] for row in result}) != len(result):
        raise ValueError("Inference scene_uid values must be unique")
    return result


def checkpoint_state(path, cfg):
    """Accept the original Full training checkpoint; retain strict tensor checks."""
    value = load_torch(path, trusted=True)
    if value.get("stage") != "stage_b" or value.get("version") != "sweepflow3_training_v1":
        raise ValueError("Expected a SweepFlow Full Stage B training checkpoint")
    contract = value["contract"]
    model = contract["model"]
    if (model.get("use_r", True) is not True or model.get("paper_cross_view", "return") != "return"
            or contract.get("encoder", "adapted") != "adapted"
            or contract["semantic"]["alignment_weight"] <= 0):
        raise ValueError("This package supports the Full checkpoint only")
    if contract["dataset"] != cfg["dataset"] or contract["data"] != cfg["data"]:
        raise ValueError("Checkpoint dataset, camera order, or image dimensions do not match")
    for key in ("internal_width", "internal_blocks", "return_budget"):
        if model[key] != cfg["model"][key]:
            raise ValueError(f"Checkpoint model setting differs: {key}")
    for key in ("dim", "projection_seed", "alignment_block"):
        if contract["semantic"][key] != cfg["semantic"][key]:
            raise ValueError(f"Checkpoint semantic setting differs: {key}")
    if contract["points"] != cfg["points"]:
        raise ValueError("Checkpoint point preprocessing settings do not match")
    state = value["ema"]["shadow"]
    step = int(value["step"])
    return state, step


def save_image(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    temporary = path.with_suffix(".tmp.png")
    Image.fromarray(array).save(temporary)
    temporary.replace(path)


@torch.inference_mode()
def infer(cfg, *, checkpoint, output, split="test", manifest=None, point_checkpoint=None,
          stats=None, steps=30, seed=91357, limit=None):
    if world() != 1 or rank() != 0:
        raise ValueError("Inference uses one device and generates the complete camera set jointly")
    cfg = copy.deepcopy(cfg)
    out = Path(output).expanduser().resolve()
    if point_checkpoint:
        cfg["point_checkpoint_override"] = str(Path(point_checkpoint).expanduser().resolve())
    if stats:
        cfg["stats_override"] = str(Path(stats).expanduser().resolve())
    checkpoint = Path(checkpoint).expanduser().resolve()
    state, step = checkpoint_state(checkpoint, cfg)
    if manifest:
        source = Path(manifest).expanduser().resolve()
        rows = read_jsonl(source)
        base = source.parent
    else:
        rows = rows_for(cfg, split)
        base = Path.cwd()
    rows = inference_rows(rows[:limit] if limit else rows, cfg, base)
    statistics = read_json(stats_path(cfg))
    if (statistics["std"] <= 0 or statistics["high"] <= statistics["low"]
            or not all(np.isfinite(statistics[k]) for k in ("mean", "std", "high", "low"))):
        raise ValueError("Invalid training intensity statistics")
    cfg["packet_override_root"] = str(out / "conditions")
    contract = {"checkpoint": str(checkpoint), "checkpoint_step": step,
                "point_checkpoint": str(default_point_checkpoint(cfg)), "intensity_statistics": statistics,
                "data": cfg["data"], "model": cfg["model"], "points": cfg["points"],
                "semantic": cfg["semantic"], "split_seed": cfg["split_seed"],
                "steps": steps, "seed": seed, "rows": rows}
    run = out / "inference_config.json"
    if run.is_file() and read_json(run) != contract:
        raise ValueError("Inference configuration changed. Choose a new output directory.")
    if not run.exists():
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Output directory is nonempty and has no SweepFlow inference configuration")
        atomic_json(run, contract)
        atomic_jsonl(out / "input_manifest.jsonl", rows)

    d = device()
    torch.set_num_threads(cfg["workers"]["cpu_threads"])
    point_data = PointScenes(cfg, split, targets=False, rows=rows, stats=statistics)
    point_model = None
    for i in iterate(range(len(rows)), "inference/conditions", unit="scenes"):
        gp = packet_path(cfg, "geometry", split, i)
        sp = packet_path(cfg, "student", split, i)
        if not gp.is_file():
            atomic_torch(gp, geometry_packet(rows[i], cfg, statistics, d))
        if not sp.is_file():
            item = point_data[i]
            if point_model is None and item["selected_count"] > 0:
                point_model = load_point(cfg, "adapted", d)
            features = encode_point_item(point_model, item, cfg, d)
            packet = make_packet(features, item, cfg)
            packet["encoder"] = "adapted"
            atomic_torch(sp, packet)
    del point_model
    if d.type == "cuda":
        torch.cuda.empty_cache()

    data = GenerationScenes(cfg, split, targets=False, rows=rows)
    model = vae = prompt = scheduler = None
    h, w = cfg["data"]["image_hw"]
    for i in iterate(range(len(rows)), "inference/generate", unit="camera_sets"):
        row = rows[i]
        folder = out / "images" / f"{i:08d}"
        paths = [folder / f"{name}.png" for name in row["camera_names"]]
        marker = folder / "generation.json"
        if marker.is_file() and all(path.is_file() for path in paths):
            continue
        if model is None:
            model = SemanticJointFlow(cfg).to(d).eval()
            restore_adapter(model, state)
            del state
            vae = SD35Autoencoder(cfg["model"]["sd35_root"]).to(d)
            prompt = EmptyPromptEmbeds(cfg["model"]["sd35_root"], d, cache_path=cfg["model"]["empty_prompt_cache"])
            scheduler = load_scheduler(cfg["model"]["sd35_root"])
        batch = collate_generation([data[i]])
        condition = move(batch["condition"], d)
        views = len(row["camera_names"])
        p, pooled = prompt.get(views)
        noise_seed = (seed + row["index"] * 104729) % (2**63 - 1)
        generator = torch.Generator(device=d).manual_seed(noise_seed)
        latent = sample_sd35_fm(model, scheduler, condition, p, pooled,
                               latent_shape=(1, views, 16, h // 8, w // 8),
                               generator=generator, device=d, steps=steps)
        rgb = vae.decode(latent.flatten(0, 1)).cpu()
        for image, path in zip(rgb, paths):
            save_image(image, path)
        atomic_json(marker, {"scene_uid": row["scene_uid"], "noise_seed": noise_seed,
                             "steps": steps, "camera_names": row["camera_names"]})
    print(f"Generated camera sets in {out / 'images'}", flush=True)
