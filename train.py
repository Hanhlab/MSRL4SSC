"""Retrain Stage 1/2 with strict outer-fold isolation and inner-fold selection.

The script uses the bundled 768-d MSRL backbone initialized from official CROMA weights. All generated
configs, indices, checkpoints, selections and exported features are written
under this project's ``outputs/nested_strict5`` directory by default.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.utils.data import Subset


HERE = Path(__file__).resolve().parent
CORE_ROOT = HERE
BASE_CONFIG = HERE / "configs/five_index_full.yaml"
SSL_SCRIPT = CORE_ROOT / "scripts/train_msrl_ssl.py"
PYTHON = Path(sys.executable)
DATA_ROOT = HERE / "data/example_dataset"
RUN_ROOT = HERE / "outputs/nested_strict5"
PRETRAINED_CHECKPOINT = HERE / "checkpoints/CROMA_base.pt"
FOLDS = (1, 2, 3, 4, 5)
S2_NAMES = ["SI3", "SI5", "SI8", "kNDMI", "TCARI"]
S2_CHANNELS = [9, 11, 14, 18, 27]
S1_NAMES = ["VV", "VH"]
S1_CHANNELS = [30, 31]
ENV_NAMES = ["DEM", "Soil_Clay", "Soil_Sand", "Pb_030"]
ENV_CHANNELS = [33, 34, 35, 40]

STAGE1_CANDIDATES = {
    "base": {"backbone_lr": 1.0e-5, "head_lr": 1.0e-4, "mask": 0.65},
    "low_lr": {"backbone_lr": 5.0e-6, "head_lr": 5.0e-5, "mask": 0.60},
}
STAGE2_CANDIDATES = {
    "lr_5e-4": {"head_lr": 5.0e-4},
    "lr_1e-3": {"head_lr": 1.0e-3},
    "lr_2e-3": {"head_lr": 2.0e-3},
}


def make_spatial_folds(frame: pd.DataFrame, seed: int = 74, n_folds: int = 5, n_blocks: int = 25) -> np.ndarray:
    """Allocate intact coordinate clusters to balanced folds without using SSC."""
    xy = StandardScaler().fit_transform(frame[["X", "Y"]].to_numpy())
    n_blocks = min(int(n_blocks), len(frame))
    blocks = KMeans(n_clusters=n_blocks, n_init=20, random_state=seed).fit_predict(xy)
    rng = np.random.default_rng(seed)
    block_ids = np.arange(n_blocks)
    rng.shuffle(block_ids)
    sizes = {int(block): int(np.sum(blocks == block)) for block in block_ids}
    loads = [0] * int(n_folds)
    assignment: dict[int, int] = {}
    for block in sorted(block_ids, key=lambda value: sizes[int(value)], reverse=True):
        fold = int(np.argmin(loads))
        assignment[int(block)] = fold + 1
        loads[fold] += sizes[int(block)]
    return np.asarray([assignment[int(block)] for block in blocks], dtype=np.int64)


def load_msrl_module():
    spec = importlib.util.spec_from_file_location(
        "verified_train_msrl", CORE_ROOT / "scripts/train_msrl_5fold.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the verified MSRL trainer.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from msrl4ssc.data import FoldStats, build_dataset  # noqa: E402
from msrl4ssc.engine import load_model_weights, make_grad_scaler, make_loader  # noqa: E402
from msrl4ssc.models import build_msrl_model  # noqa: E402
from msrl4ssc.utils import seed_everything  # noqa: E402


MSRL_TRAINER = load_msrl_module()


def save_yaml(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def save_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_master_index() -> Path:
    destination = RUN_ROOT / "master_index_seed74.csv"
    if destination.exists():
        return destination
    frame = pd.read_csv(DATA_ROOT / "sample_index.csv")
    if "fold_74" not in frame.columns:
        frame["fold_74"] = make_spatial_folds(frame, seed=74)
    vh = []
    for relpath in frame["NPZ_Path"]:
        payload = np.load(DATA_ROOT / relpath)
        row, col = payload.get("center_pixel", np.asarray([64, 64])).astype(int)
        vh.append(float(payload["image"][31, row, col]))
    frame["VH"] = np.asarray(vh, dtype=np.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def base_config(master_index: Path) -> dict:
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    band_names = [f"unused_{index:02d}" for index in range(44)]
    for name, index in zip(
        S2_NAMES + S1_NAMES + ENV_NAMES,
        S2_CHANNELS + S1_CHANNELS + ENV_CHANNELS,
    ):
        band_names[index] = name
    data = cfg["data"]
    data.update({
        "source_index": str(master_index),
        "source_npz_root": str(DATA_ROOT),
        "band_names": band_names,
        "s2_bands": S2_NAMES,
        "s1_bands": S1_NAMES,
        "s2_intensity_augment_bands": S2_NAMES,
    })
    data.pop("pretrain_index", None)
    data.pop("pretrain_npz_root", None)
    cfg["croma_initialization"]["path"] = str(PRETRAINED_CHECKPOINT)
    cfg["pretraining"].update({"epochs": 20, "patience": 6, "validation_fraction": 0.1})
    cfg["msrl_model"]["finetune_mode"] = "last_blocks"
    cfg["msrl_model"]["last_n_blocks"] = 1
    cfg["msrl_5fold"].update({
        "fold_seed": 74, "folds": list(FOLDS), "trained_source_folds": list(FOLDS),
        "reuse_source_fold": {}, "fixed_epochs": 40, "batch_size": 2,
        "gradient_accumulation": 2,
    })
    return cfg


def prepare_stage1_config(base: dict, master: pd.DataFrame, outer_fold: int, name: str, params: dict) -> Path:
    fold_root = RUN_ROOT / f"outer_fold_{outer_fold}"
    fold_root.mkdir(parents=True, exist_ok=True)
    dev_index = fold_root / "development_index.csv"
    if not dev_index.exists():
        master.loc[master["fold_74"].ne(outer_fold)].to_csv(dev_index, index=False)
    cfg = copy.deepcopy(base)
    cfg["data"]["source_index"] = str(dev_index)
    cfg["cache"]["enabled"] = True
    cfg["cache"]["path"] = str(fold_root / "cache/development.h5")
    cfg["output"]["root"] = str(fold_root / "stage1_candidates" / name)
    cfg["pretraining"].update({
        "backbone_lr": params["backbone_lr"], "head_lr": params["head_lr"],
        "s2_mask_ratio": params["mask"], "s1_mask_ratio": params["mask"],
    })
    path = fold_root / "stage1_candidates" / name / "config.yaml"
    save_yaml(cfg, path)
    return path


def run_stage1_candidates(base: dict, master: pd.DataFrame, outer_fold: int, resume: bool) -> tuple[Path, Path, str]:
    records = []
    for name, params in STAGE1_CANDIDATES.items():
        config_path = prepare_stage1_config(base, master, outer_fold, name, params)
        command = [
            str(PYTHON), str(SSL_SCRIPT), "--config", str(config_path),
            "--device", "cuda", "--seed", str(7400 + outer_fold),
        ]
        if resume:
            command.append("--resume")
        subprocess.run(command, cwd=CORE_ROOT, check=True)
        root = config_path.parent / "msrl_pipeline"
        history = json.loads((root / "01_ssl/history.json").read_text(encoding="utf-8"))
        score = min(float(row["val_loss"]) for row in history)
        records.append({"candidate": name, "inner_ssl_val_loss": score})
    selection = pd.DataFrame(records).sort_values("inner_ssl_val_loss").reset_index(drop=True)
    selection.to_csv(RUN_ROOT / f"outer_fold_{outer_fold}/stage1_selection.csv", index=False)
    winner = str(selection.iloc[0]["candidate"])
    root = RUN_ROOT / f"outer_fold_{outer_fold}/stage1_candidates" / winner / "msrl_pipeline"
    return root / "01_ssl/best.pt", root / "global_stats.json", winner


def full_stage2_config(base: dict, master_index: Path, outer_fold: int, name: str, params: dict) -> dict:
    cfg = copy.deepcopy(base)
    cfg["data"]["source_index"] = str(master_index)
    cfg["cache"]["enabled"] = True
    cfg["cache"]["path"] = str(RUN_ROOT / "cache/all_samples.h5")
    cfg["output"]["root"] = str(RUN_ROOT / f"outer_fold_{outer_fold}/stage2_candidates/{name}")
    cfg["msrl_training"]["head_lr"] = params["head_lr"]
    return cfg


def train_stage2_model(
    cfg: dict, stats: FoldStats, ssl_checkpoint: Path, train_full,
    train_indices: np.ndarray, seed: int, epochs: int = 40,
):
    seed_everything(seed)
    ssl_state, _ = load_model_weights(ssl_checkpoint)
    model = build_msrl_model(cfg)
    model.msrl.load_encoder_state_dict(ssl_state, strict=True)
    y_train = train_full.table.iloc[train_indices][train_full.target_col].to_numpy(np.float64)
    target_mean, target_std = float(y_train.mean()), max(float(y_train.std()), 1e-8)
    model = model.cuda()
    loader = make_loader(
        Subset(train_full, train_indices.tolist()), batch_size=2, shuffle=True,
        num_workers=0, drop_last=True,
    )
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(cfg["msrl_training"]["head_lr"]),
        weight_decay=float(cfg["msrl_training"]["weight_decay"]),
    )
    scaler = make_grad_scaler(True)
    history = []
    for epoch in range(1, epochs + 1):
        log = MSRL_TRAINER.run_epoch(model, loader, optimizer, scaler, torch.device("cuda"), cfg, epoch, target_mean, target_std)
        history.append({"epoch": epoch, **{key: float(value) for key, value in log.items()}})
        if epoch == 1 or epoch % 10 == 0:
            print(f"[outer={seed // 1000} stage2] epoch={epoch:03d} loss={log['loss']:.6f}", flush=True)
    return model, target_mean, target_std, history


def extract(model, dataset, indices: np.ndarray, target_mean: float, target_std: float) -> dict[str, np.ndarray]:
    loader = make_loader(Subset(dataset, indices.tolist()), batch_size=4, shuffle=False, num_workers=0)
    return MSRL_TRAINER.extract_predictions(model, loader, torch.device("cuda"), True, target_mean, target_std)


def fixed_stage3_score(master: pd.DataFrame, train_indices: np.ndarray, val_indices: np.ndarray, train_states: np.ndarray, val_states: np.ndarray) -> float:
    columns = ["SI3", "SI5", "SI8", "kNDMI", "TCARI", "DEM", "Soil_Clay", "Soil_Sand", "Pb_030", "VH"]
    raw = master[columns].to_numpy(np.float32)
    x_train = np.column_stack([raw[train_indices], train_states])
    x_val = np.column_stack([raw[val_indices], val_states])
    transformer = ColumnTransformer([
        ("conventional", "passthrough", list(range(10))),
        ("point_pca", PCA(n_components=4, random_state=74, svd_solver="randomized"), list(range(10, 138))),
    ])
    regressor = CatBoostRegressor(
        iterations=700, depth=3, learning_rate=0.04, l2_leaf_reg=5.0,
        loss_function="RMSE", random_seed=74, verbose=False,
        allow_writing_files=False, thread_count=4,
    )
    estimator = Pipeline([("features", transformer), ("regressor", regressor)])
    target = master["SSC"].to_numpy(np.float32)
    estimator.fit(x_train, target[train_indices])
    return float(r2_score(target[val_indices], estimator.predict(x_val)))


def run_stage2_selection(base: dict, master: pd.DataFrame, master_index: Path, outer_fold: int, ssl: Path, stats_path: Path) -> None:
    fold_values = master["fold_74"].to_numpy(np.int64)
    inner_val_fold = FOLDS[outer_fold % len(FOLDS)]
    inner_train = np.flatnonzero((fold_values != outer_fold) & (fold_values != inner_val_fold))
    inner_val = np.flatnonzero(fold_values == inner_val_fold)
    stats = FoldStats.load(stats_path)
    records = []
    for candidate_index, (name, params) in enumerate(STAGE2_CANDIDATES.items()):
        cfg = full_stage2_config(base, master_index, outer_fold, name, params)
        config_path = RUN_ROOT / f"outer_fold_{outer_fold}/stage2_candidates/{name}/config.yaml"
        save_yaml(cfg, config_path)
        train_full = build_dataset(cfg, stats, 74, 0, 0, "all", True, True)
        eval_full = build_dataset(cfg, stats, 74, 0, 0, "all", True, False)
        model, mean, std, history = train_stage2_model(
            cfg, stats, ssl, train_full, inner_train,
            seed=outer_fold * 1000 + candidate_index, epochs=20,
        )
        train_output = extract(model, eval_full, inner_train, mean, std)
        val_output = extract(model, eval_full, inner_val, mean, std)
        score = fixed_stage3_score(
            master, inner_train, inner_val,
            train_output["Both_Image"], val_output["Both_Image"],
        )
        save_json(history, config_path.parent / "history.json")
        torch.save(MSRL_TRAINER.head_state_dict(model), config_path.parent / "inner_head.pt")
        records.append({"candidate": name, "inner_val_fold": inner_val_fold, "inner_stage3_r2": score})
        del model
        torch.cuda.empty_cache()
    selection = pd.DataFrame(records).sort_values("inner_stage3_r2", ascending=False).reset_index(drop=True)
    selection.to_csv(RUN_ROOT / f"outer_fold_{outer_fold}/stage2_selection.csv", index=False)
    winner = str(selection.iloc[0]["candidate"])
    params = STAGE2_CANDIDATES[winner]
    cfg = full_stage2_config(base, master_index, outer_fold, winner, params)
    train_full = build_dataset(cfg, stats, 74, 0, 0, "all", True, True)
    eval_full = build_dataset(cfg, stats, 74, 0, 0, "all", True, False)
    outer_train = np.flatnonzero(fold_values != outer_fold)
    model, mean, std, history = train_stage2_model(
        cfg, stats, ssl, train_full, outer_train, seed=74000 + outer_fold,
    )
    all_indices = np.arange(len(master), dtype=np.int64)
    output = extract(model, eval_full, all_indices, mean, std)
    fold_dir = RUN_ROOT / "selected_features" / f"source_fold_{outer_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "Sample_ID": output["Sample_ID"].astype(object),
        "SSC": output["SSC"].astype(np.float32),
        "fold_74": fold_values,
        "MSRL_Source_Fold": np.full(len(master), outer_fold, np.int64),
    }
    for branch in ("S1", "S2", "Both"):
        payload[f"{branch}_FoldMsrlImage"] = output[f"{branch}_Image"].astype(np.float32)
        payload[f"{branch}_FoldMsrlPrediction"] = np.column_stack([
            output[f"{branch}_PRE"], output[f"{branch}_logvar"],
        ]).astype(np.float32)
    np.savez_compressed(fold_dir / "all_sample_features.npz", **payload)
    torch.save(MSRL_TRAINER.head_state_dict(model), fold_dir / "msrl_head.pt")
    save_json({
        "outer_fold": outer_fold, "inner_val_fold": inner_val_fold,
        "selected_candidate": winner, "selected_inner_stage3_r2": float(selection.iloc[0]["inner_stage3_r2"]),
        "stage1_checkpoint": str(ssl), "train_sample_ids": master.iloc[outer_train]["Sample_ID"].tolist(),
        "test_sample_ids": master.loc[master["fold_74"].eq(outer_fold), "Sample_ID"].tolist(),
        "history": history,
    }, fold_dir / "metadata.json")
    del model
    torch.cuda.empty_cache()


def finalize_protocol(master: pd.DataFrame, base: dict) -> None:
    root = RUN_ROOT / "selected_features"
    folds = master["fold_74"].to_numpy(np.int64)
    np.savez_compressed(
        root / "msrl_oof_features.npz",
        Sample_ID=master["Sample_ID"].to_numpy(object),
        SSC=master["SSC"].to_numpy(np.float32),
        fold_74=folds,
        MSRL_Source_Fold=folds,
    )
    config = copy.deepcopy(base)
    config["data"]["s2_bands"] = S2_NAMES
    config["croma_initialization"]["path"] = str(PRETRAINED_CHECKPOINT)
    save_yaml(config, root / "resolved_config.yaml")
    audit_rows = []
    for fold in FOLDS:
        test_ids = set(master.loc[master["fold_74"].eq(fold), "Sample_ID"].astype(str))
        stage1_ids = set(pd.read_csv(RUN_ROOT / f"outer_fold_{fold}/development_index.csv")["Sample_ID"].astype(str))
        metadata = json.loads((root / f"source_fold_{fold}/metadata.json").read_text(encoding="utf-8"))
        stage2_ids = set(map(str, metadata["train_sample_ids"]))
        audit_rows.append({
            "outer_fold": fold,
            "n_test": len(test_ids),
            "stage1_development_n": len(stage1_ids),
            "stage1_test_overlap": len(test_ids & stage1_ids),
            "stage2_train_n": len(stage2_ids),
            "stage2_test_overlap": len(test_ids & stage2_ids),
        })
    pd.DataFrame(audit_rows).to_csv(RUN_ROOT / "protocol_audit.csv", index=False)
    selection_rows = []
    for fold in FOLDS:
        stage1 = pd.read_csv(RUN_ROOT / f"outer_fold_{fold}/stage1_selection.csv").iloc[0]
        stage2 = pd.read_csv(RUN_ROOT / f"outer_fold_{fold}/stage2_selection.csv").iloc[0]
        selection_rows.append({
            "outer_fold": fold,
            "stage1_candidate": stage1["candidate"],
            "stage1_inner_ssl_val_loss": float(stage1["inner_ssl_val_loss"]),
            "stage2_candidate": stage2["candidate"],
            "stage2_inner_val_fold": int(stage2["inner_val_fold"]),
            "stage2_inner_r2": float(stage2["inner_stage3_r2"]),
        })
    pd.DataFrame(selection_rows).to_csv(RUN_ROOT / "hyperparameter_selection.csv", index=False)


def main() -> None:
    global DATA_ROOT, RUN_ROOT, BASE_CONFIG, PRETRAINED_CHECKPOINT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--outer-fold", type=int, choices=FOLDS, default=None)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT,
                        help="Directory containing sample_index.csv and the patch NPZ files.")
    parser.add_argument("--output-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=PRETRAINED_CHECKPOINT)
    args = parser.parse_args()
    DATA_ROOT = args.data_root.expanduser().resolve()
    RUN_ROOT = args.output_root.expanduser().resolve()
    BASE_CONFIG = args.base_config.expanduser().resolve()
    PRETRAINED_CHECKPOINT = args.pretrained_checkpoint.expanduser().resolve()
    for required in (DATA_ROOT / "sample_index.csv", BASE_CONFIG, PRETRAINED_CHECKPOINT):
        if not required.exists():
            raise FileNotFoundError(f"Required input not found: {required}")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    master_index = prepare_master_index()
    master = pd.read_csv(master_index)
    base = base_config(master_index)
    selected_folds = [args.outer_fold] if args.outer_fold else list(FOLDS)
    for outer_fold in selected_folds:
        print(f"\n===== strict outer fold {outer_fold} =====", flush=True)
        completed = RUN_ROOT / "selected_features" / f"source_fold_{outer_fold}/all_sample_features.npz"
        if args.resume and completed.exists():
            print(f"[outer {outer_fold}] completed artifacts reused", flush=True)
            continue
        ssl, stats, stage1_winner = run_stage1_candidates(base, master, outer_fold, args.resume)
        print(f"[outer {outer_fold}] selected Stage 1: {stage1_winner}", flush=True)
        run_stage2_selection(base, master, master_index, outer_fold, ssl, stats)
    if all((RUN_ROOT / "selected_features" / f"source_fold_{fold}/all_sample_features.npz").exists() for fold in FOLDS):
        finalize_protocol(master, base)
        print(f"Strict selected features: {RUN_ROOT / 'selected_features'}", flush=True)


if __name__ == "__main__":
    main()
