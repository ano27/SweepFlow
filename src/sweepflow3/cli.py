"""Training and inference entry points for SweepFlow Full."""
from __future__ import annotations

import argparse
from pathlib import Path


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser():
    p = argparse.ArgumentParser(description="SweepFlow Full: multi-view LiDAR-to-RGB generation")
    sub = p.add_subparsers(dest="command", required=True)
    assets = sub.add_parser("assets", help="download point/teacher assets and encode the fixed empty prompt")
    assets.add_argument("--config", required=True)
    assets.add_argument("--inference-only", action="store_true", help="download WaffleIron source and prepare the empty prompt; no teacher or point pretraining weights")
    prep = sub.add_parser("prepare", help="prepare manifests, training targets, or trained point features")
    prep.add_argument("--config", required=True)
    prep.add_argument("--kind", choices=("manifests", "base", "features"), default="base")
    prep.add_argument("--split", choices=("train", "dev", "test"), default="train")
    train = sub.add_parser("train", help="train Stage A or Stage B, resuming the last checkpoint")
    train.add_argument("--config", required=True)
    train.add_argument("--stage", choices=("a", "b"), required=True)
    train.add_argument("--through", type=positive, help="stop after this update without changing the full schedule")
    train.add_argument("--microbatch", type=positive, help="per-device memory bound; global batch stays fixed")
    infer = sub.add_parser("infer", help="generate RGB camera sets from LiDAR and calibration")
    infer.add_argument("--config", required=True)
    infer.add_argument("--checkpoint", required=True, help="Full Stage B training checkpoint")
    source = infer.add_mutually_exclusive_group()
    source.add_argument("--split", choices=("train", "dev", "test"), default="test")
    source.add_argument("--manifest", help="JSONL with LiDAR paths, camera sizes and calibration; no RGB required")
    infer.add_argument("--point-checkpoint", help="Stage A checkpoint; defaults to the configured run")
    infer.add_argument("--stats", help="training intensity_statistics.json; defaults to the configured run")
    infer.add_argument("--output", required=True, help="new output directory; one PNG per camera")
    infer.add_argument("--steps", type=positive, default=30)
    infer.add_argument("--seed", type=int, default=91357)
    infer.add_argument("--limit", type=positive, help="maximum number of manifest rows")
    return p


def prepare(cfg, kind, split):
    from .common import dataset_root, rank, world
    from .config import make_nuscenes_splits, lock_preparation
    from .data.readers import build_manifests
    from .data.cache import fit_intensity, prepare_base, export_packets

    manifest = dataset_root(cfg) / "manifests" / "complete.json"
    if world() > 1:
        if kind == "manifests" or not manifest.is_file():
            raise ValueError("Run prepare --kind manifests on one process before distributed cache preparation")
        # Metadata and training statistics are already immutable at this point.
        if cfg["dataset"] == "nuscenes" and cfg["datasets"]["nuscenes"].get("splits_file") == "auto":
            cfg["datasets"]["nuscenes"]["splits_file"] = str(Path(cfg["workdir"]) / "dataset_protocols" / "nuscenes_splits.json")
    else:
        make_nuscenes_splits(cfg)
        lock_preparation(cfg)
        build_manifests(cfg)
        fit_intensity(cfg)
    lock_preparation(cfg)
    if kind == "base":
        prepare_base(cfg, split)
    elif kind == "features":
        export_packets(cfg, split)
    if rank() == 0:
        print(f"Prepared {kind} for {cfg['dataset']}/{split}", flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    from .config import load_config
    cfg = load_config(args.config)
    if args.command == "assets":
        from .assets import prepare_assets
        from .models.prompt import prepare_empty_prompt
        prepare_assets(cfg["assets_root"], teacher=not args.inference_only, point_weights=not args.inference_only)
        prepare_empty_prompt(cfg["model"]["sd35_root"], cfg["model"]["empty_prompt_cache"])
    elif args.command == "prepare":
        prepare(cfg, args.kind, args.split)
    elif args.command == "train":
        from .config import resolve_stage_a_steps, lock_preparation
        from .training.engine import train
        resolve_stage_a_steps(cfg)
        if cfg["dataset"] == "nuscenes" and cfg["datasets"]["nuscenes"].get("splits_file") == "auto":
            cfg["datasets"]["nuscenes"]["splits_file"] = str(Path(cfg["workdir"]) / "dataset_protocols" / "nuscenes_splits.json")
        lock_preparation(cfg)
        stage = "stage_a" if args.stage == "a" else "stage_b"
        if args.microbatch:
            cfg[stage]["microbatch"] = args.microbatch
        train(cfg, stage, args.through)
    else:
        from .inference import infer
        infer(cfg, checkpoint=args.checkpoint, output=args.output, split=args.split,
              manifest=args.manifest, point_checkpoint=args.point_checkpoint,
              stats=args.stats, steps=args.steps, seed=args.seed, limit=args.limit)


if __name__ == "__main__":
    main()
