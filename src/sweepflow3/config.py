"""One dataset, one Full model, explicit local paths."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml


CAMERAS = {
    "once": ["cam01", "cam03", "cam05", "cam06", "cam07", "cam08", "cam09"],
    "nuscenes": ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"],
    "dsert_roll": ["RGB_L", "RGB_R"],
}


def absolute_path(value):
    return str(Path(value).expanduser().resolve())


def load_config(path):
    """Resolve relative paths against the current working directory."""
    cfg = yaml.safe_load(Path(path).expanduser().read_text())
    dataset = cfg["dataset"]
    if dataset not in CAMERAS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    cfg["workdir"] = absolute_path(cfg["workdir"])
    cfg["assets_root"] = absolute_path(cfg["assets_root"])
    for name in ("sd35_root", "empty_prompt_cache"):
        cfg["model"][name] = absolute_path(cfg["model"][name])
    for name in ("point_checkpoint_override", "stats_override"):
        if cfg.get(name):
            cfg[name] = absolute_path(cfg[name])
    spec = cfg.pop("dataset_config")
    spec["root"] = absolute_path(spec["root"])
    spec.setdefault("cameras", CAMERAS[dataset].copy())
    if spec["cameras"] != CAMERAS[dataset]:
        raise ValueError(f"Expected the released {dataset} camera order: {CAMERAS[dataset]}")
    if spec.get("splits_file", "auto") != "auto":
        spec["splits_file"] = absolute_path(spec["splits_file"])
    cfg["datasets"] = {dataset: spec}
    cfg["data"]["camera_names"] = spec["cameras"].copy()
    cfg["arm"], cfg["encoder"] = "full", "adapted"
    cfg["seed"] = int(cfg["seed"])
    cfg["model"]["use_r"] = True
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    h, w = cfg["data"]["image_hw"]
    if min(h, w) <= 0 or h % 16 or w % 16:
        raise ValueError("image_hw must contain positive multiples of 16")
    if any(x <= 0 or x % 14 for x in cfg["semantic"]["teacher_hw"]):
        raise ValueError("teacher_hw must contain positive multiples of 14")
    if cfg["model"]["internal_width"] % 4:
        raise ValueError("internal_width must be divisible by 4")
    if cfg["semantic"]["alignment_block"] not in cfg["model"]["internal_blocks"]:
        raise ValueError("The alignment block must contain an adapter")
    if cfg["semantic"]["alignment_weight"] <= 0:
        raise ValueError("SweepFlow Full requires Stage B visual alignment")
    if cfg["points"]["max_points"] < 17 or cfg["points"]["voxel_size"] <= 0:
        raise ValueError("Invalid point selection settings")
    for stage in ("stage_a", "stage_b"):
        s = cfg[stage]
        if min(s["global_batch"], s["microbatch"], s["save_every"]) < 1:
            raise ValueError(f"{stage}: batch sizes and save_every must be positive")
    if cfg["stage_a"]["epochs"] < 1 or cfg["stage_b"]["steps"] < 1:
        raise ValueError("Training budgets must be positive")


def resolve_stage_a_steps(cfg):
    path = Path(cfg["workdir"]) / "data" / cfg["dataset"] / "manifests" / "complete.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run prepare --kind manifests first: {path}")
    count = int(json.loads(path.read_text())["counts"]["train"])
    cfg["stage_a"]["steps"] = count // cfg["stage_a"]["global_batch"] * cfg["stage_a"]["epochs"]
    if cfg["stage_a"]["steps"] < 1:
        raise ValueError("Training split is smaller than the Stage A global batch")
    return cfg


def make_nuscenes_splits(cfg):
    if cfg["dataset"] != "nuscenes":
        return
    spec = cfg["datasets"]["nuscenes"]
    if spec.get("splits_file", "auto") != "auto":
        return
    root = Path(spec["root"])
    rows = json.loads((root / spec["version"] / "scene.json").read_text())
    present = {r["name"] for r in rows}
    reference = json.loads((Path(__file__).parent / "data" / "nuscenes_val_scenes.json").read_text())
    val = set(reference["val"])
    if len(present) != 850 or not val <= present:
        raise ValueError("Default nuScenes preparation requires complete v1.0-trainval metadata")
    path = Path(cfg["workdir"]) / "dataset_protocols" / "nuscenes_splits.json"
    from .common import atomic_json
    atomic_json(path, {"train": sorted(present - val), "val": sorted(val)})
    spec["splits_file"] = str(path)


def lock_preparation(cfg):
    """Prevent accidental reuse of caches after changing the input/model setup."""
    from .common import atomic_json
    keys = ("dataset", "datasets", "split_seed", "seed", "data", "points", "semantic", "model", "assets_root")
    contract = {key: copy.deepcopy(cfg[key]) for key in keys}
    # Activation checkpointing affects memory, not the cached conditions.
    contract["model"].pop("activation_checkpointing", None)
    path = Path(cfg["workdir"]) / "preparation_config.json"
    if path.is_file() and json.loads(path.read_text()) != contract:
        raise ValueError("Input/model configuration changed. Select a new workdir for fresh caches.")
    if not path.exists():
        atomic_json(path, contract)
