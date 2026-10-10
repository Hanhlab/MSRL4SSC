
from __future__ import annotations

import argparse
import copy
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msrl4ssc.config import load_config, resolve_path, save_yaml
from msrl4ssc.data import FoldStats, build_dataset
from msrl4ssc.engine import (
    autocast_context,
    load_model_weights,
    make_grad_scaler,
    make_loader,
)
from msrl4ssc.msrl_pipeline import msrl_pipeline_paths
from msrl4ssc.losses import msrl_loss_components
from msrl4ssc.metrics import regression_metrics
from msrl4ssc.models import build_msrl_model
from msrl4ssc.utils import (
    choose_device,
    ensure_dir,
    move_batch,
    save_json,
    seed_everything,
)


BRANCHES = ("S2", "S1", "Both")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/five_index_full.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--fold-seed",
        type=int,
        default=None,
        help="Override msrl_5fold.fold_seed.",
    )
    parser.add_argument(
        "--source-index",
        default=None,
        help="Optional project index CSV containing the requested fold column.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Offset used to train reproducible ensemble members.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--fold3-source",
        type=int,
        default=None,
        help="Override the MSRL source fold used for target fold 3.",
    )
    parser.add_argument(
        "--source-folds",
        nargs="+",
        type=int,
        default=None,
        help="Override the supervised MSRL source folds to train.",
    )
    parser.add_argument(
        "--fold3-source2-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of target-fold-3 samples assigned to source fold 2; "
            "the remainder uses the configured fold-3 source."
        ),
    )
    parser.add_argument(
        "--ssl-checkpoint",
        default=None,
        help="Optional Global SSL best.pt override.",
    )
    parser.add_argument(
        "--stats",
        default=None,
        help="Optional Global SSL statistics JSON override.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: <output.root>/msrl_5fold_strict_oof.",
    )
    return parser.parse_args()


def argument_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default.resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def choose_supervision_masks(
    batch: dict[str, Any], cfg: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    has_s2 = batch["has_s2"].bool()
    has_s1 = batch["has_s1"].bool()
    paired = has_s2 & has_s1
    mode = str(cfg["msrl_training"].get("supervision_mode", "all_paths"))
    if mode == "all_paths":
        return has_s2, has_s1, paired
    if mode != "sample_one_path":
        raise ValueError(
            "msrl_training.supervision_mode must be all_paths or sample_one_path"
        )
    probabilities = cfg["msrl_training"].get("modality_probabilities", {})
    p_both = float(probabilities.get("both", 0.50))
    p_s2 = float(probabilities.get("s2", 0.25))
    p_s1 = float(probabilities.get("s1", 0.25))
    total = p_both + p_s2 + p_s1
    if total <= 0:
        raise ValueError("At least one modality probability must be positive")
    p_both, p_s2 = p_both / total, p_s2 / total
    random = torch.rand(len(has_s2), device=has_s2.device)
    both_mask = paired & (random < p_both)
    s2_mask = paired & (random >= p_both) & (random < p_both + p_s2)
    s1_mask = paired & (random >= p_both + p_s2)
    return (
        s2_mask | (has_s2 & ~has_s1),
        s1_mask | (has_s1 & ~has_s2),
        both_mask,
    )


def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device: torch.device,
    cfg: dict[str, Any],
    epoch: int,
    target_mean: float,
    target_std: float,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    count = 0
    accumulation = max(
        1, int(cfg["msrl_5fold"].get("gradient_accumulation", 1))
    )
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)
        s2_mask, s1_mask, both_mask = choose_supervision_masks(batch, cfg)
        paired = batch["has_both"].bool()
        with autocast_context(device, bool(cfg["msrl_training"].get("amp", True))):
            outputs = model.forward_all(
                batch["s2"],
                batch["s1"],
                batch["local_coordinates"],
                compute_s2=True,
                compute_s1=True,
                compute_both=True,
            )
            loss, parts = msrl_loss_components(
                outputs,
                (batch["target_raw"] - target_mean) / target_std,
                s2_mask,
                s1_mask,
                both_mask,
                paired,
                cfg,
                epoch,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite MSRL loss at epoch={epoch}, step={step + 1}."
            )
        scaler.scale(loss / accumulation).backward()
        if (step + 1) % accumulation == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(cfg["msrl_training"].get("grad_clip", 1.0)),
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        batch_count = int(batch["target_raw"].shape[0])
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * batch_count
        count += batch_count
    if count == 0:
        raise RuntimeError("The MSRL training loader is empty.")
    return {key: value / count for key, value in totals.items()}


def head_state_dict(model) -> OrderedDict[str, torch.Tensor]:
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in model.state_dict().items():
        if name.startswith("msrl."):
            continue
        value = value.detach()
        result[name] = value.to(
            device="cpu",
            dtype=torch.float16 if value.is_floating_point() else value.dtype,
        )
    return result


def load_head_state(model, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"Head checkpoint must be a state dictionary: {path}")
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    non_backbone_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("msrl.")
    ]
    if unexpected or non_backbone_missing:
        raise RuntimeError(
            f"Incompatible head checkpoint {path}: unexpected={unexpected}, "
            f"non_backbone_missing={non_backbone_missing}"
        )


@torch.no_grad()
def extract_predictions(
    model,
    loader,
    device: torch.device,
    amp: bool,
    target_mean: float,
    target_std: float,
) -> dict[str, np.ndarray]:
    model.eval()
    storage: dict[str, list[np.ndarray]] = {
        "Sample_ID": [],
        "SSC": [],
    }
    for branch in BRANCHES:
        storage[f"{branch}_PRE"] = []
        storage[f"{branch}_logvar"] = []
        storage[f"{branch}_Image"] = []
    for batch in loader:
        sample_ids = np.asarray(batch["sample_id"], dtype=object)
        moved = move_batch(batch, device)
        with autocast_context(device, amp):
            outputs = model.forward_all(
                moved["s2"],
                moved["s1"],
                moved["local_coordinates"],
                compute_s2=True,
                compute_s1=True,
                compute_both=True,
            )
        storage["Sample_ID"].append(sample_ids)
        storage["SSC"].append(moved["target_raw"].detach().cpu().numpy())
        for branch, output in zip(BRANCHES, (outputs.s2, outputs.s1, outputs.both)):
            if output is None:
                raise RuntimeError(f"MSRL output for {branch} is missing.")
            storage[f"{branch}_PRE"].append(
                output.mean.detach().float().cpu().numpy() * target_std + target_mean
            )
            storage[f"{branch}_logvar"].append(
                output.logvar.detach().float().cpu().numpy()
            )
            storage[f"{branch}_Image"].append(
                output.image_state.detach().float().cpu().numpy()
            )
    return {
        key: np.concatenate(values, axis=0)
        for key, values in storage.items()
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = copy.deepcopy(load_config(config_path))
    if args.source_index is not None:
        cfg.setdefault("data", {})["source_index"] = args.source_index
    protocol = cfg.get("msrl_5fold", {})
    if not bool(protocol.get("enabled", False)):
        raise ValueError("msrl_5fold.enabled must be true in the config.")
    loss_cfg = cfg.setdefault("msrl_losses", {})

    fold_seed = int(
        args.fold_seed if args.fold_seed is not None else protocol.get("fold_seed", 73)
    )
    protocol["fold_seed"] = fold_seed
    folds = [int(value) for value in protocol.get("folds", [1, 2, 3, 4, 5])]
    source_folds = (
        [int(value) for value in args.source_folds]
        if args.source_folds is not None
        else [
            int(value)
            for value in protocol.get("trained_source_folds", [1, 2, 3, 4, 5])
        ]
    )
    reuse_map = {
        int(target): int(source)
        for target, source in (protocol.get("reuse_source_fold", {3: 3}) or {}).items()
    }
    if args.fold3_source is not None:
        reuse_map[3] = int(args.fold3_source)
    target_to_source = {fold: reuse_map.get(fold, fold) for fold in folds}
    if sorted(set(target_to_source.values())) != sorted(source_folds):
        raise ValueError(
            f"trained_source_folds={source_folds} do not match target mapping "
            f"{target_to_source}."
        )
    protocol["trained_source_folds"] = source_folds
    protocol["reuse_source_fold"] = {3: target_to_source.get(3, 3)}

    epochs = int(args.epochs or protocol.get("fixed_epochs", 40))
    batch_size = int(args.batch_size or protocol.get("batch_size", 4))
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch size must be positive.")
    fold3_source2_fraction = float(args.fold3_source2_fraction)
    if not 0.0 <= fold3_source2_fraction <= 1.0:
        raise ValueError("--fold3-source2-fraction must be between 0 and 1.")
    if fold3_source2_fraction > 0.0 and 2 not in source_folds:
        raise ValueError("Mixed fold 3 requires source fold 2 to be trained.")

    seed_everything(fold_seed + int(args.seed_offset))
    device = choose_device(args.device)
    paths = msrl_pipeline_paths(cfg)
    ssl_checkpoint = argument_path(args.ssl_checkpoint, paths.ssl_best)
    stats_path = argument_path(args.stats, paths.stats_path)
    fold3_source = target_to_source.get(3, 3)
    default_output = paths.output_root / "msrl_5fold_strict_oof"
    output_dir = ensure_dir(argument_path(args.output_dir, default_output))
    if not ssl_checkpoint.exists():
        raise FileNotFoundError(
            f"Global SSL checkpoint is missing: {ssl_checkpoint}. "
            "Run stage 01 once or provide --ssl-checkpoint."
        )
    if not stats_path.exists():
        raise FileNotFoundError(f"Global statistics are missing: {stats_path}")

    stats = FoldStats.load(stats_path)
    train_full = build_dataset(
        cfg, stats, fold_seed, test_fold=0, val_fold=0,
        split="all", require_label=True, augment=True,
    )
    eval_full = build_dataset(
        cfg, stats, fold_seed, test_fold=0, val_fold=0,
        split="all", require_label=True, augment=False,
    )
    fold_key = f"fold_{fold_seed}"
    if fold_key not in train_full.index.columns:
        raise KeyError(f"Source index is missing {fold_key}.")
    fold_values = train_full.index[fold_key].astype(int).to_numpy()
    observed = sorted(np.unique(fold_values).tolist())
    if observed != sorted(folds):
        raise ValueError(f"Configured folds={folds}, observed {fold_key}={observed}.")

    sample_source_folds = np.asarray(
        [target_to_source[int(fold)] for fold in fold_values], dtype=np.int64
    )
    fold3_indices = np.flatnonzero(fold_values == 3)
    source2_count = int(round(len(fold3_indices) * fold3_source2_fraction))
    if source2_count:
        # A seed-specific deterministic half split.  The same assignment is
        # exported per sample and reused by downstream regression.
        rng = np.random.default_rng(fold_seed)
        selected = rng.permutation(fold3_indices)[:source2_count]
        sample_source_folds[selected] = 2
    fold3_source_counts = {
        int(source): int(np.sum(sample_source_folds[fold3_indices] == source))
        for source in sorted(np.unique(sample_source_folds[fold3_indices]).tolist())
    }

    amp = bool(cfg["msrl_training"].get("amp", True) and device.type == "cuda")
    cfg["msrl_training"]["amp"] = amp
    ssl_state, ssl_metadata = load_model_weights(ssl_checkpoint)
    prediction_frames: list[pd.DataFrame] = []
    image_by_branch_and_id: dict[str, dict[str, np.ndarray]] = {
        branch: {} for branch in BRANCHES
    }

    print(f"Project root       : {ROOT}")
    print(f"Device             : {device}")
    if device.type == "cuda":
        print(f"GPU                : {torch.cuda.get_device_name(device)}")
    print(f"Global SSL (once)  : {ssl_checkpoint}")
    print(f"SSL epoch          : {ssl_metadata.get('epoch', 'unknown')}")
    print(f"Fold protocol      : {target_to_source}")
    print(f"Fold-3 sources     : {fold3_source_counts}")
    print(f"Fixed epochs/head  : {epochs}")
    print(
        "Adaptation losses  : "
        f"Lshare=({float(loss_cfg.get('common_gap_alignment', 0.0)):g}, "
        f"{float(loss_cfg.get('common_token_alignment', 0.0)):g}); "
        f"Lcons=({float(loss_cfg.get('s2_s1_prediction_consistency', 0.0)):g}, "
        f"{float(loss_cfg.get('s2_s1_state_consistency', 0.0)):g})"
    )
    print(f"Output             : {output_dir}")

    for source_fold in source_folds:
        fold_dir = ensure_dir(output_dir / f"source_fold_{source_fold}")
        head_path = fold_dir / "msrl_head.pt"
        history_path = fold_dir / "history.json"
        train_indices = np.flatnonzero(fold_values != source_fold)
        y_train = train_full.table.iloc[train_indices][train_full.target_col].to_numpy(
            dtype=np.float64
        )
        target_mean = float(np.mean(y_train))
        target_std = float(np.std(y_train))
        if not math.isfinite(target_std) or target_std < 1e-8:
            target_std = 1.0

        seed_everything(fold_seed * 100 + source_fold + int(args.seed_offset))
        model = build_msrl_model(cfg)
        model.msrl.load_encoder_state_dict(ssl_state, strict=True)
        history: list[dict[str, Any]] = []
        if args.resume and head_path.exists():
            load_head_state(model, head_path)
            print(f"[source fold {source_fold}] loaded completed head {head_path}")
        else:
            subset = Subset(train_full, train_indices.tolist())
            loader = make_loader(
                subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=int(cfg["msrl_training"].get("num_workers", 0)),
                drop_last=len(subset) >= batch_size and batch_size > 1,
            )
            model = model.to(device)
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            optimizer = AdamW(
                parameters,
                lr=float(cfg["msrl_training"].get("head_lr", 3e-4)),
                weight_decay=float(cfg["msrl_training"].get("weight_decay", 1e-3)),
            )
            scaler = make_grad_scaler(amp)
            for epoch in range(1, epochs + 1):
                log = run_epoch(
                    model,
                    loader,
                    optimizer,
                    scaler,
                    device,
                    cfg,
                    epoch,
                    target_mean,
                    target_std,
                )
                history.append(
                    {
                        "epoch": epoch,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        **{f"train_{key}": value for key, value in log.items()},
                    }
                )
                save_json(history, history_path)
                if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                    print(
                        f"[MSRL source={source_fold}] epoch={epoch:03d}/{epochs:03d} "
                        f"loss={log.get('loss', math.nan):.6f}",
                        flush=True,
                    )
            torch.save(head_state_dict(model), head_path)
            save_json(
                {
                    "source_fold": source_fold,
                    "seed_offset": int(args.seed_offset),
                    "fold_seed": fold_seed,
                    "train_samples": int(len(train_indices)),
                    "train_sample_ids": train_full.index.iloc[train_indices][
                        cfg["data"].get("join_key", "Sample_ID")
                    ].astype(str).tolist(),
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "fixed_epochs": epochs,
                    "adaptation_losses": {
                        "common_gap_alignment": float(loss_cfg.get("common_gap_alignment", 0.0)),
                        "common_token_alignment": float(loss_cfg.get("common_token_alignment", 0.0)),
                        "s2_s1_prediction_consistency": float(loss_cfg.get("s2_s1_prediction_consistency", 0.0)),
                        "s2_s1_state_consistency": float(loss_cfg.get("s2_s1_state_consistency", 0.0)),
                    },
                    "global_ssl_checkpoint": str(ssl_checkpoint),
                    "global_ssl_sha256_recorded_in_protocol": True,
                    "checkpoint_contents": "MSRL head only; shared Global SSL excluded",
                },
                fold_dir / "metadata.json",
            )
        model = model.to(device).eval()

        # A downstream model for outer fold k must see one coherent latent
        # coordinate system: the source-fold-k MSRL head is therefore
        # applied to *all* samples, while only fold k is scored as test data.
        # The head has never seen fold-k labels.  Exporting this matrix avoids
        # the invalid practice of fitting one PCA over states produced by five
        # independently trained heads.
        all_loader = make_loader(
            Subset(eval_full, list(range(len(eval_full)))),
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(cfg["feature_extraction"].get("num_workers", 0)),
        )
        all_extracted = extract_predictions(
            model, all_loader, device, amp, target_mean, target_std
        )
        all_payload: dict[str, np.ndarray] = {
            "Sample_ID": all_extracted["Sample_ID"].astype(object),
            "SSC": all_extracted["SSC"].astype(np.float32),
            fold_key: fold_values.astype(np.int64),
            "MSRL_Source_Fold": np.full(
                len(eval_full), source_fold, dtype=np.int64
            ),
        }
        for branch in BRANCHES:
            all_payload[f"{branch}_FoldMsrlPrediction"] = np.column_stack(
                [
                    all_extracted[f"{branch}_PRE"],
                    all_extracted[f"{branch}_logvar"],
                ]
            ).astype(np.float32)
            all_payload[f"{branch}_FoldMsrlImage"] = all_extracted[
                f"{branch}_Image"
            ].astype(np.float32)
        np.savez_compressed(fold_dir / "all_sample_features.npz", **all_payload)

        for target_fold in folds:
            test_indices = np.flatnonzero(
                (fold_values == target_fold) & (sample_source_folds == source_fold)
            )
            if not len(test_indices):
                continue
            loader = make_loader(
                Subset(eval_full, test_indices.tolist()),
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(cfg["feature_extraction"].get("num_workers", 0)),
            )
            extracted = extract_predictions(
                model,
                loader,
                device,
                amp,
                target_mean,
                target_std,
            )
            frame = pd.DataFrame(
                {
                    "Sample_ID": extracted["Sample_ID"].astype(str),
                    "SSC": extracted["SSC"].astype(np.float64),
                    "Test_Fold": target_fold,
                    "MSRL_Source_Fold": source_fold,
                    "Intentional_Reuse": target_fold != source_fold,
                }
            )
            for branch in BRANCHES:
                frame[f"{branch}_PRE"] = extracted[f"{branch}_PRE"]
                frame[f"{branch}_logvar"] = extracted[f"{branch}_logvar"]
                scores = regression_metrics(extracted["SSC"], extracted[f"{branch}_PRE"])
                print(
                    f"[MSRL partition target={target_fold} source={source_fold} "
                    f"n={len(test_indices)} {branch}] "
                    f"R2={scores['R2']:.4f} RMSE={scores['RMSE']:.6f}",
                    flush=True,
                )
                image_by_branch_and_id[branch].update(
                    zip(
                        frame["Sample_ID"].astype(str).tolist(),
                        extracted[f"{branch}_Image"],
                    )
                )
            prediction_frames.append(frame)
        model = model.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    predictions = pd.concat(prediction_frames, ignore_index=True)
    if len(predictions) != len(eval_full):
        raise RuntimeError(
            f"Expected {len(eval_full)} OOF rows, produced {len(predictions)}."
        )
    if predictions["Sample_ID"].duplicated().any():
        raise RuntimeError("Duplicate Sample_ID values in MSRL OOF predictions.")
    predictions = predictions.sort_values("Sample_ID").reset_index(drop=True)
    predictions.to_csv(output_dir / "msrl_oof_predictions.csv", index=False)

    metric_rows: list[dict[str, Any]] = []
    for target_fold in folds:
        fold_frame = predictions[predictions["Test_Fold"].eq(target_fold)]
        source_values = sorted(fold_frame["MSRL_Source_Fold"].astype(int).unique())
        for branch in BRANCHES:
            scores = regression_metrics(fold_frame["SSC"], fold_frame[f"{branch}_PRE"])
            metric_rows.append(
                {
                    "branch": branch,
                    "test_fold": target_fold,
                    "msrl_source_fold": (
                        source_values[0] if len(source_values) == 1 else 0
                    ),
                    "msrl_source_folds": ",".join(map(str, source_values)),
                    "source_fold_2_count": int(
                        fold_frame["MSRL_Source_Fold"].eq(2).sum()
                    ),
                    "source_fold_3_count": int(
                        fold_frame["MSRL_Source_Fold"].eq(3).sum()
                    ),
                    "intentional_reuse": bool(fold_frame["Intentional_Reuse"].any()),
                    **scores,
                }
            )
    metrics = pd.DataFrame(metric_rows).sort_values(["branch", "test_fold"])
    metrics.to_csv(output_dir / "msrl_fold_metrics.csv", index=False)
    summary_rows = []
    for branch, group in metrics.groupby("branch", sort=False):
        merged = predictions[["SSC", f"{branch}_PRE"]]
        summary_rows.append(
            {
                "branch": branch,
                "runs": len(group),
                **{
                    f"{metric}_mean": float(group[metric].mean())
                    for metric in ("RMSE", "MAE", "R2", "RPD", "Bias")
                },
                **{
                    f"{metric}_std": float(group[metric].std(ddof=1))
                    for metric in ("RMSE", "MAE", "R2", "RPD", "Bias")
                },
                "pooled_OOF_R2": float(
                    regression_metrics(merged["SSC"], merged[f"{branch}_PRE"])["R2"]
                ),
            }
        )
    pd.DataFrame(summary_rows).sort_values("R2_mean", ascending=False).to_csv(
        output_dir / "msrl_summary_mean_sd.csv", index=False
    )

    # Export arrays in the original index order so downstream fold slicing is exact.
    original_ids = eval_full.index[cfg["data"].get("join_key", "Sample_ID")].astype(str)
    aligned = predictions.set_index("Sample_ID").loc[original_ids]
    arrays: dict[str, np.ndarray] = {
        "Sample_ID": original_ids.to_numpy(dtype=object),
        "SSC": aligned["SSC"].to_numpy(dtype=np.float32),
        fold_key: fold_values.astype(np.int64),
        "MSRL_Source_Fold": aligned["MSRL_Source_Fold"].to_numpy(dtype=np.int64),
        "Intentional_Reuse": aligned["Intentional_Reuse"].to_numpy(dtype=bool),
    }
    for branch in BRANCHES:
        arrays[f"{branch}_FoldMsrlPrediction"] = np.column_stack(
            [
                aligned[f"{branch}_PRE"].to_numpy(dtype=np.float32),
                aligned[f"{branch}_logvar"].to_numpy(dtype=np.float32),
            ]
        )
        arrays[f"{branch}_FoldMsrlImage"] = np.stack(
            [image_by_branch_and_id[branch][sample_id] for sample_id in original_ids],
            axis=0,
        ).astype(np.float32)
    np.savez_compressed(output_dir / "msrl_oof_features.npz", **arrays)

    save_yaml(cfg, output_dir / "resolved_config.yaml")
    save_json(
        {
            "protocol": (
                "GlobalSSL_once_plus_MSRL_5fold_with_fold3_mixed_sources"
                if fold3_source2_fraction > 0.0
                else "GlobalSSL_once_plus_standard_MSRL_5fold"
                if fold3_source == 3
                else "GlobalSSL_once_plus_MSRL_5fold_with_"
                f"fold3_reusing_fold{fold3_source}"
            ),
            "fold_seed": fold_seed,
            "seed_offset": int(args.seed_offset),
            "target_to_msrl_source_fold": target_to_source,
            "fold3_source2_fraction": fold3_source2_fraction,
            "fold3_sample_source_counts": fold3_source_counts,
            "fold3_assignment_random_seed": fold_seed,
            "trained_msrl_source_folds": source_folds,
            "adaptation_losses": {
                "common_gap_alignment": float(loss_cfg.get("common_gap_alignment", 0.0)),
                "common_token_alignment": float(loss_cfg.get("common_token_alignment", 0.0)),
                "s2_s1_prediction_consistency": float(loss_cfg.get("s2_s1_prediction_consistency", 0.0)),
                "s2_s1_state_consistency": float(loss_cfg.get("s2_s1_state_consistency", 0.0)),
            },
            "intentional_non_independent_fold": (
                None
                if fold3_source == 3 and fold3_source2_fraction == 0.0
                else 3
            ),
            "intentional_non_independent_reason": (
                f"{source2_count} fold-3 samples use source fold 2 and "
                f"{len(fold3_indices) - source2_count} use source fold 3."
                if fold3_source2_fraction > 0.0
                else "Fold 3 is inferred with its own held-out fold-3 MSRL "
                "head; all five MSRL predictions are out-of-fold."
                if fold3_source == 3
                else f"Fold 3 is inferred with the fold-{fold3_source} MSRL "
                "head; fold 3 samples were present when that head was trained."
            ),
            "global_ssl_checkpoint": str(ssl_checkpoint),
            "global_stats": str(stats_path),
            "msrl_heads_are_head_only": True,
            "sample_count": len(predictions),
        },
        output_dir / "protocol.json",
    )
    print(f"Saved fold-aware MSRL outputs to {output_dir}")


if __name__ == "__main__":
    main()
