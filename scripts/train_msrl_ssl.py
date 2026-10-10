"""Train one global MSRL self-supervised model on all available images.

This entry point deliberately does not accept a test fold.  The source images
are used without their SSC labels, shuffled deterministically, and split into
one global training set and one global validation set.  The best weights are
written to::

    <output.root>/msrl_pipeline/01_ssl/best.pt

Run from any working directory, for example::

    python scripts/train_msrl_ssl.py --device cuda
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msrl4ssc.config import load_config, resolve_path, save_yaml
from msrl4ssc.msrl_pipeline import msrl_pipeline_paths, training_completion_reason
from msrl4ssc.msrl import build_msrl_from_config
from msrl4ssc.data import (
    FoldStats,
    build_ssl_dataset,
    fit_fold_stats,
)
from msrl4ssc.engine import (
    autocast_context,
    make_grad_scaler,
    make_loader,
    save_checkpoint,
    save_fp16_model_weights,
)
from msrl4ssc.pretrained import maybe_initialize_croma
from msrl4ssc.utils import (
    choose_device,
    ensure_dir,
    load_json,
    move_batch,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one global MSRL SSL model on all available S1/S2 images."
    )
    parser.add_argument(
        "--config",
        default="configs/five_index_full.yaml",
        help="YAML configuration path (default: configs/five_index_full.yaml).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device, for example cuda, cuda:0, or cpu (default: cuda).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last.pt when it exists.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override pretraining.epochs from the YAML file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override pretraining.batch_size from the YAML file.",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=None,
        help="Override pretraining.gradient_accumulation from the YAML file.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override pretraining.num_workers from the YAML file.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=None,
        help="Override the random global validation fraction.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate paths/data and print the resolved training plan without training.",
    )
    return parser.parse_args()


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    stage_cfg = cfg["pretraining"]
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "num_workers": args.num_workers,
        "validation_fraction": args.validation_fraction,
    }
    for key, value in overrides.items():
        if value is not None:
            stage_cfg[key] = value
    if args.no_amp:
        stage_cfg["amp"] = False


def validate_settings(cfg: dict[str, Any], device: torch.device) -> None:
    data_cfg = cfg["data"]
    stage_cfg = cfg["pretraining"]

    required_paths = {
        "source_index": resolve_path(
            cfg, data_cfg.get("pretrain_index") or data_cfg.get("source_index")
        ),
        "source_npz_root": resolve_path(
            cfg, data_cfg.get("pretrain_npz_root") or data_cfg.get("source_npz_root")
        ),
    }
    missing = [f"{name}: {path}" for name, path in required_paths.items() if not path or not path.exists()]
    if missing:
        raise FileNotFoundError("Missing configured input paths:\n  " + "\n  ".join(missing))

    positive_integer_keys = ("epochs", "batch_size", "gradient_accumulation")
    for key in positive_integer_keys:
        if int(stage_cfg[key]) <= 0:
            raise ValueError(f"pretraining.{key} must be positive, got {stage_cfg[key]!r}.")
    if int(stage_cfg.get("num_workers", 0)) < 0:
        raise ValueError("pretraining.num_workers cannot be negative.")

    val_fraction = float(stage_cfg.get("validation_fraction", 0.1))
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            "pretraining.validation_fraction must be between 0 and 1, "
            f"got {val_fraction}."
        )
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) exist."
            )


def combined_loss(parts: dict[str, torch.Tensor], weights: dict[str, Any]) -> torch.Tensor:
    return (
        float(weights.get("contrastive", 1.0)) * parts["contrastive"]
        + float(weights.get("mae_s2", 1.0)) * parts["mae_s2"]
        + float(weights.get("mae_s1", 1.0)) * parts["mae_s1"]
        + float(weights.get("alignment", 0.0)) * parts["alignment"]
    )


def configure_ssl_trainability(
    model: torch.nn.Module, mode: str, last_n_blocks: int
) -> None:
    """Protect the official CROMA initialization on a very small local corpus."""
    mode = str(mode).lower()
    if mode == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return
    if mode not in {"frozen_backbone", "last_blocks"}:
        raise ValueError(
            "pretraining.finetune_mode must be full, last_blocks, or frozen_backbone"
        )
    for parameter in model.parameters():
        parameter.requires_grad = False

    # The local objectives have task-specific projection/reconstruction heads;
    # these always adapt even when the official encoders are protected.
    for module in [
        model.contrast,
        model.s2_decoder,
        model.s1_decoder,
        model.s2_gap_head,
        model.s1_gap_head,
    ]:
        for parameter in module.parameters():
            parameter.requires_grad = True

    if mode == "frozen_backbone":
        return
    n = max(1, int(last_n_blocks))
    for blocks in [
        model.s2_encoder.blocks,
        model.s1_encoder.blocks,
        model.s1_query_s2,
        model.s2_query_s1,
    ]:
        for block in list(blocks)[-n:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    for module in [
        model.s2_encoder.norm,
        model.s1_encoder.norm,
        model.s1_joint_norm,
        model.s2_joint_norm,
        model.joint_norm,
    ]:
        for parameter in module.parameters():
            parameter.requires_grad = True


def run_epoch(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    cfg: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "contrastive": 0.0,
        "mae_s2": 0.0,
        "mae_s1": 0.0,
        "alignment": 0.0,
    }
    sample_count = 0
    stage_cfg = cfg["pretraining"]
    accumulation = max(1, int(stage_cfg.get("gradient_accumulation", 1)))

    if training:
        # Do not retain repeated examples as false negatives across epochs on
        # this 221-image dataset.  Within an epoch, shuffling visits each image
        # once and the queue still supplies negatives for batch_size=2.
        if hasattr(model, "contrast") and hasattr(model.contrast, "reset_queue"):
            model.contrast.reset_queue()
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)
        with autocast_context(device, bool(stage_cfg.get("amp", True))):
            parts = model.pretrain_forward(
                batch["s2"],
                batch["s1"],
                float(stage_cfg["s2_mask_ratio"]),
                float(stage_cfg["s1_mask_ratio"]),
                batch.get("has_s2"),
                batch.get("has_s1"),
            )
            loss = combined_loss(parts, stage_cfg["loss_weights"])

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite SSL loss at step {step + 1}: {float(loss.detach())}"
            )

        if training:
            if scaler is None:
                raise RuntimeError("A gradient scaler is required during training.")
            scaler.scale(loss / accumulation).backward()
            update_now = (step + 1) % accumulation == 0 or (step + 1) == len(loader)
            if update_now:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(stage_cfg.get("grad_clip", 1.0))
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        batch_size = int(batch["s2"].shape[0])
        totals["loss"] += float(loss.detach()) * batch_size
        for key in ("contrastive", "mae_s2", "mae_s1", "alignment"):
            totals[key] += float(parts[key].detach()) * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("The SSL data loader is empty.")
    return {key: value / sample_count for key, value in totals.items()}


def build_global_datasets(
    cfg: dict[str, Any],
    stats: FoldStats,
    seed: int,
):
    """Use every image, then make a deterministic label-free train/val split."""
    # test_fold=0 is only metadata here.  split="all" never consults fold columns.
    train_full = build_ssl_dataset(
        cfg, stats, seed, test_fold=0, val_fold=0, split="all", augment=True
    )
    eval_full = build_ssl_dataset(
        cfg, stats, seed, test_fold=0, val_fold=0, split="all", augment=False
    )
    if len(train_full) < 3:
        raise ValueError(
            f"Global SSL needs at least three images, but found {len(train_full)}."
        )

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train_full), generator=generator).tolist()
    val_fraction = float(cfg["pretraining"].get("validation_fraction", 0.1))
    val_count = max(1, int(round(len(order) * val_fraction)))
    val_count = min(val_count, len(order) - 2)
    val_indices = order[:val_count]
    train_indices = order[val_count:]
    return Subset(train_full, train_indices), Subset(eval_full, val_indices)


def load_history(path: Path, resume: bool) -> list[dict[str, float]]:
    if not resume or not path.exists():
        return []
    payload = load_json(path)
    if not isinstance(payload, list):
        raise TypeError(f"Training history must be a JSON list: {path}")
    return payload


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    apply_cli_overrides(cfg, args)

    seed_everything(args.seed)
    device = choose_device(args.device)
    validate_settings(cfg, device)

    output_root = resolve_path(cfg, cfg["output"]["root"])
    if output_root is None:
        raise ValueError("output.root must be configured.")
    paths = msrl_pipeline_paths(cfg)
    ensure_dir(paths.root)
    stage_dir = ensure_dir(paths.ssl_dir)
    stats_path = paths.stats_path
    history_path = stage_dir / "history.json"
    checkpoint_last = stage_dir / "last.pt"
    checkpoint_best = stage_dir / "best.pt"

    # Global statistics are fitted on every source image.  SSC labels are never
    # loaded by the SSL dataset and never enter the loss.
    if stats_path.exists():
        stats = FoldStats.load(stats_path)
        expected_bands = list(
            dict.fromkeys(list(data_cfg["s2_bands"]) + list(data_cfg["s1_bands"]))
        )
        incompatible = []
        if list(stats.band_names) != expected_bands:
            incompatible.append(
                f"image bands expected={expected_bands}, found={list(stats.band_names)}"
            )
        if incompatible:
            raise ValueError(
                f"Existing {stats_path} is incompatible with the current config: "
                + "; ".join(incompatible)
                + ". Remove this stale statistics file or select a new output path."
            )
    else:
        stats = fit_fold_stats(
            cfg,
            fold_seed=args.seed,
            test_fold=0,
            val_fold=0,
            output_path=stats_path,
            fit_split="all",
        )

    train_dataset, val_dataset = build_global_datasets(cfg, stats, args.seed)
    stage_cfg = cfg["pretraining"]
    batch_size = int(stage_cfg["batch_size"])
    workers = int(stage_cfg.get("num_workers", 0))
    epochs = int(stage_cfg["epochs"])
    amp_enabled = bool(stage_cfg.get("amp", True) and device.type == "cuda")
    stage_cfg["amp"] = amp_enabled

    history = load_history(history_path, args.resume)
    if args.resume and checkpoint_last.exists() and checkpoint_best.exists():
        completion_reason = training_completion_reason(
            history,
            epochs=epochs,
            patience=int(stage_cfg.get("patience", 25)),
            min_delta=float(stage_cfg.get("min_delta", 1e-4)),
            score_keys="val_loss",
        )
        if completion_reason is not None:
            print(f"Training is already complete: {completion_reason}.")
            print(f"Best SSL checkpoint: {checkpoint_best}")
            return

    print(f"Project root : {ROOT}")
    print(f"Config       : {config_path}")
    print(f"Device       : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(device)}")
    print(f"Train images : {len(train_dataset)}")
    print(f"Val images   : {len(val_dataset)}")
    print(f"Epochs       : {epochs}")
    print(f"Batch size   : {batch_size}")
    print(f"Grad accum   : {int(stage_cfg.get('gradient_accumulation', 1))}")
    print(f"AMP          : {amp_enabled}")
    print(f"Output       : {stage_dir}")

    save_yaml(cfg, stage_dir / "resolved_config.yaml")
    save_json(
        {
            "mode": "global_ssl",
            "seed": args.seed,
            "train_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "stats_path": str(stats_path),
        },
        stage_dir / "split_metadata.json",
    )
    if args.check_only:
        print("Configuration and global SSL dataset validation passed (--check-only).")
        return

    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=len(train_dataset) >= batch_size and batch_size > 1,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )

    model = build_msrl_from_config(cfg)
    if not (args.resume and checkpoint_last.exists()):
        report = maybe_initialize_croma(model, cfg)
        if report is not None:
            save_json(report, stage_dir / "official_initialization_report.json")
            print(
                "Official initialization: "
                f"loaded={report.get('loaded_count', len(report.get('loaded', [])))} "
                f"skipped={report.get('skipped_count', len(report.get('skipped', [])))}"
            )
    configure_ssl_trainability(
        model,
        str(stage_cfg.get("finetune_mode", "last_blocks")),
        int(stage_cfg.get("last_n_blocks", 1)),
    )
    model = model.to(device)
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100.0 * trainable / total:.1f}%)"
    )

    head_prefixes = (
        "contrast.", "s2_decoder.", "s1_decoder.",
        "s2_gap_head.", "s1_gap_head.",
    )
    backbone_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(head_prefixes):
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)
    parameter_groups = []
    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": float(stage_cfg.get("backbone_lr", stage_cfg["lr"])),
            }
        )
    if head_parameters:
        parameter_groups.append(
            {
                "params": head_parameters,
                "lr": float(stage_cfg.get("head_lr", stage_cfg["lr"])),
            }
        )
    if not parameter_groups:
        raise RuntimeError("Global SSL has no trainable parameters.")
    optimizer = AdamW(
        parameter_groups,
        weight_decay=float(stage_cfg["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(stage_cfg.get("min_lr", 1e-6)),
    )
    scaler = make_grad_scaler(amp_enabled)

    start_epoch = 1
    best_val = math.inf
    no_improve = 0
    if args.resume:
        if checkpoint_last.exists():
            checkpoint = torch.load(
                checkpoint_last, map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            if "scaler_state" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_val = float(checkpoint.get("best_val_loss", math.inf))
            no_improve = int(checkpoint.get("no_improve", 0))
            print(f"Resuming from epoch {start_epoch}.")
        else:
            print(f"--resume specified, but no checkpoint exists at {checkpoint_last}; starting fresh.")

    patience = int(stage_cfg.get("patience", 25))
    if start_epoch > epochs or no_improve >= patience:
        print(
            f"Training is already complete ({start_epoch - 1}/{epochs} epochs, "
            f"no_improve={no_improve}, patience={patience})."
        )
        print(f"Best SSL checkpoint: {checkpoint_best}")
        return

    try:
        for epoch in range(start_epoch, epochs + 1):
            train_log = run_epoch(
                model, train_loader, device, cfg, optimizer=optimizer, scaler=scaler
            )
            # Compare checkpoints with the same validation masks while leaving
            # the training RNG stream untouched.  Previously every epoch saw a
            # different random mask, which added substantial noise for 22
            # validation images.
            validation_devices = []
            if device.type == "cuda":
                validation_devices = [
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                ]
            with torch.random.fork_rng(devices=validation_devices):
                torch.manual_seed(
                    int(stage_cfg.get("validation_mask_seed", args.seed + 10_000))
                )
                with torch.no_grad():
                    val_log = run_epoch(model, val_loader, device, cfg)
            scheduler.step()

            row = {
                "epoch": epoch,
                "lr": [float(group["lr"]) for group in optimizer.param_groups],
                **{f"train_{key}": value for key, value in train_log.items()},
                **{f"val_{key}": value for key, value in val_log.items()},
            }
            history.append(row)
            print(
                f"[Global SSL] epoch={epoch:03d}/{epochs:03d} "
                f"train={train_log['loss']:.5f} val={val_log['loss']:.5f} "
                f"contrast={val_log['contrastive']:.4f} "
                f"maeS2={val_log['mae_s2']:.4f} "
                f"maeS1={val_log['mae_s1']:.4f}",
                flush=True,
            )

            improved = val_log["loss"] < (
                best_val - float(stage_cfg.get("min_delta", 1e-4))
            )
            if improved:
                best_val = val_log["loss"]
                no_improve = 0
            else:
                no_improve += 1

            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_val_loss": best_val,
                "no_improve": no_improve,
                "seed": args.seed,
                "global_ssl": True,
                "train_samples": len(train_dataset),
                "validation_samples": len(val_dataset),
                "stats_path": str(stats_path),
            }
            save_checkpoint(checkpoint_last, state)
            if improved:
                save_fp16_model_weights(
                    checkpoint_best,
                    model,
                    {
                        "format": "fp16_state_dict_v1",
                        "stage": "global_ssl",
                        "epoch": epoch,
                        "best_val_loss": best_val,
                        "seed": args.seed,
                        "global_ssl": True,
                        "train_samples": len(train_dataset),
                        "validation_samples": len(val_dataset),
                        "stats_path": str(stats_path),
                    },
                )
            save_json(history, history_path)

            if no_improve >= patience:
                print(f"Early stopping after epoch {epoch}.")
                break
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA ran out of memory. Re-run with a smaller --batch-size (for "
            "example 1) and a larger --gradient-accumulation value."
        ) from exc

    print(f"Best SSL checkpoint: {checkpoint_best}")


if __name__ == "__main__":
    main()
