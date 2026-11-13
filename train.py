"""Train the three-stage MSRL4SSC pipeline from GeoTIFFs and an SSC table."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

from model import (CONVENTIONAL_BANDS, S1_BANDS, S2_BANDS, CrossModalSSL,
                   TaskAdapter, band_indexes, fit_stage3, predict_stage3,
                   save_stage3)

ROOT = Path(__file__).resolve().parent


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", help="CSV table containing raster_path, SSC, and coordinates")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "training"))
    parser.add_argument("--pretrained", help="Optional compatible Stage-1 state_dict (.pt)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-column", default="SSC")
    parser.add_argument("--raster-column", default="raster_path")
    parser.add_argument("--fold-column", default="fold")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--stage1-epochs", type=int, default=120)
    parser.add_argument("--stage2-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=74)
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Use bundled synthetic inputs, one epoch, and Ridge in Stage 3")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_raster(value: str, table_path: Path) -> Path:
    path = Path(value).expanduser()
    return (table_path.parent / path).resolve() if not path.is_absolute() else path.resolve()


def sample_pixel(row: pd.Series, source: rasterio.io.DatasetReader) -> tuple[int, int]:
    if "row" in row and "col" in row and pd.notna(row["row"]) and pd.notna(row["col"]):
        return int(row["row"]), int(row["col"])
    if "x" in row and "y" in row and pd.notna(row["x"]) and pd.notna(row["y"]):
        return source.index(float(row["x"]), float(row["y"]))
    raise KeyError("Each sample needs row/col or x/y coordinates")


class GeoTIFFPatchDataset(Dataset):
    """Read 128x128 paired radar-optical patches around field samples."""
    def __init__(self, table: pd.DataFrame, table_path: Path, image_size: int,
                 means: dict[str, np.ndarray] | None = None,
                 stds: dict[str, np.ndarray] | None = None) -> None:
        self.table = table.reset_index(drop=True).copy()
        self.table_path, self.image_size = table_path, image_size
        self.means, self.stds = means, stds

    def __len__(self) -> int:
        return len(self.table)

    def _read(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        path = resolve_raster(str(row["raster_path"]), self.table_path)
        with rasterio.open(path) as src:
            center_row, center_col = sample_pixel(row, src)
            half = self.image_size // 2
            window = Window(center_col - half, center_row - half, self.image_size, self.image_size)
            s2_idx, s1_idx = band_indexes(src.descriptions, S2_BANDS), band_indexes(src.descriptions, S1_BANDS)
            all_names = list(dict.fromkeys(CONVENTIONAL_BANDS["both"]))
            conv_idx = band_indexes(src.descriptions, all_names)
            s2 = src.read(s2_idx, window=window, boundless=True, fill_value=np.nan).astype(np.float32)
            s1 = src.read(s1_idx, window=window, boundless=True, fill_value=np.nan).astype(np.float32)
            conventional = np.asarray(list(src.sample([src.xy(center_row, center_col)], indexes=conv_idx))[0], np.float32)
        return s2, s1, conventional

    def raw(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._read(self.table.iloc[index])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        s2, s1, conventional = self.raw(index)
        if self.means is None or self.stds is None:
            raise RuntimeError("Dataset normalization statistics have not been fitted")
        s2 = np.nan_to_num((s2 - self.means["s2"][:, None, None]) / self.stds["s2"][:, None, None], nan=0).clip(-8, 8)
        s1 = np.nan_to_num((s1 - self.means["s1"][:, None, None]) / self.stds["s1"][:, None, None], nan=0).clip(-8, 8)
        return {"s2": torch.from_numpy(s2.astype(np.float32)),
                "s1": torch.from_numpy(s1.astype(np.float32)),
                "local": torch.zeros(2, dtype=torch.float32),
                "target": torch.tensor(float(self.table.iloc[index]["SSC"]), dtype=torch.float32),
                "conventional": torch.from_numpy(conventional),
                "index": torch.tensor(index)}


def fit_stats(dataset: GeoTIFFPatchDataset) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    values = {"s2": [], "s1": []}
    for index in range(len(dataset)):
        s2, s1, _ = dataset.raw(index); values["s2"].append(s2); values["s1"].append(s1)
    means, stds = {}, {}
    for key, arrays in values.items():
        stack = np.concatenate([a.reshape(a.shape[0], -1) for a in arrays], axis=1)
        means[key] = np.nanmean(stack, axis=1).astype(np.float32)
        stds[key] = np.nanstd(stack, axis=1).astype(np.float32)
        stds[key][stds[key] < 1e-6] = 1.0
    return means, stds


def spatial_folds(table: pd.DataFrame, fold_column: str, seed: int) -> np.ndarray:
    if fold_column in table and table[fold_column].notna().all():
        labels = table[fold_column].astype(int).to_numpy()
        _, labels = np.unique(labels, return_inverse=True)
        return labels
    if {"x", "y"}.issubset(table.columns):
        xy = table[["x", "y"]].to_numpy(float)
    elif {"row", "col"}.issubset(table.columns):
        xy = table[["row", "col"]].to_numpy(float)
    else:
        raise KeyError("Spatial folds require fold, x/y, or row/col columns")
    xy = StandardScaler().fit_transform(xy)
    return KMeans(n_clusters=min(5, len(table)), n_init=20, random_state=seed).fit_predict(xy)


def make_ssl(config: dict) -> CrossModalSSL:
    return CrossModalSSL(**config)


def train_stage1(model: CrossModalSSL, dataset: Dataset, device: torch.device,
                 epochs: int, batch_size: int) -> None:
    model.to(device).train(); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    for epoch in range(epochs):
        total = 0.0
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            losses = model(batch["s2"].to(device), batch["s1"].to(device), 0.65)
            loss = losses["contrast"] + losses["s2_rec"] + losses["s1_rec"] + 0.05 * losses["align"]
            loss.backward(); optimizer.step(); total += float(loss) * len(batch["target"])
        print(f"Stage 1 epoch {epoch + 1:03d}: loss={total / len(dataset):.6f}")


def adaptation_loss(output: dict, target: torch.Tensor) -> torch.Tensor:
    loss = 1.5 * F.smooth_l1_loss(output["both"]["prediction"], target)
    loss += 0.35 * F.smooth_l1_loss(output["s1"]["prediction"], target)
    loss += 0.35 * F.smooth_l1_loss(output["s2"]["prediction"], target)
    loss += 0.02 * (1 - F.cosine_similarity(output["shared"]["s1"], output["shared"]["s2"])).mean()
    loss += 0.01 * F.smooth_l1_loss(output["s1"]["prediction"], output["s2"]["prediction"])
    return loss


def train_stage2(model: TaskAdapter, dataset: Dataset, indices: np.ndarray,
                 device: torch.device, epochs: int, batch_size: int) -> None:
    for parameter in model.ssl.parameters(): parameter.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-3)
    targets = np.asarray([float(dataset[int(i)]["target"]) for i in indices])
    mean, std = float(targets.mean()), max(float(targets.std()), 1e-6)
    model.to(device).train()
    for epoch in range(epochs):
        total = 0.0
        for batch in DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            target = (batch["target"].to(device) - mean) / std
            output = model(batch["s2"].to(device), batch["s1"].to(device), batch["local"].to(device))
            loss = adaptation_loss(output, target); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step()
            total += float(loss) * len(target)
        print(f"Stage 2 epoch {epoch + 1:03d}: loss={total / len(indices):.6f}")


@torch.inference_mode()
def extract(model: TaskAdapter, dataset: Dataset, device: torch.device,
            batch_size: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    model.eval(); states = {p: [] for p in ("s1", "s2", "both")}; raw_both = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        output = model(batch["s2"].to(device), batch["s1"].to(device), batch["local"].to(device))
        for path in states: states[path].append(output[path]["state"].cpu().numpy())
        raw_both.append(batch["conventional"].numpy())
    raw = np.concatenate(raw_both)
    names = list(dict.fromkeys(CONVENTIONAL_BANDS["both"]))
    columns = {name: raw[:, names.index(name)] for name in names}
    conventional = {path: np.column_stack([columns[name] for name in CONVENTIONAL_BANDS[path]])
                    for path in states}
    return {p: np.concatenate(v) for p, v in states.items()}, conventional


def main() -> None:
    args = arguments(); seed_all(args.seed)
    if args.smoke_test:
        args.samples = args.samples or str(ROOT / "data" / "smoke_test" / "train_samples.csv")
        args.output_dir = str(ROOT / "outputs" / "smoke_training")
        args.stage1_epochs = args.stage2_epochs = 1
        args.dim, args.depth, args.heads, args.batch_size = 32, 1, 4, 4
    if not args.samples:
        raise SystemExit("--samples is required unless --smoke-test is used")
    table_path = Path(args.samples).expanduser().resolve(); table = pd.read_csv(table_path)
    required = {args.raster_column, args.target_column}
    if not required.issubset(table): raise KeyError(f"Sample table is missing {sorted(required - set(table))}")
    table = table.rename(columns={args.raster_column: "raster_path", args.target_column: "SSC"})
    if not np.isfinite(pd.to_numeric(table["SSC"], errors="coerce")).all():
        raise ValueError("SSC contains missing or non-numeric values")
    raw_dataset = GeoTIFFPatchDataset(table, table_path, args.image_size)
    means, stds = fit_stats(raw_dataset)
    dataset = GeoTIFFPatchDataset(table, table_path, args.image_size, means, stds)
    folds = spatial_folds(table, args.fold_column, args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    config = {"s2_channels": 5, "s1_channels": 2, "dim": args.dim, "depth": args.depth,
              "heads": args.heads, "patch_size": args.patch_size, "dropout": args.dropout,
              "temperature": 0.1, "image_size": args.image_size}
    ssl = make_ssl(config)
    if args.pretrained:
        payload = torch.load(args.pretrained, map_location="cpu", weights_only=True)
        ssl.load_state_dict(payload.get("model", payload), strict=False)
    train_stage1(ssl, dataset, device, args.stage1_epochs, args.batch_size)
    torch.save({"model": ssl.state_dict(), "config": config}, output / "stage1_cross_modal_ssl.pt")
    target = table["SSC"].to_numpy(np.float32); fold_records = []
    if not args.skip_cv:
        for fold in sorted(np.unique(folds)):
            train_idx, test_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            fold_ssl = make_ssl(config); fold_ssl.load_state_dict(ssl.state_dict())
            adapter = TaskAdapter(fold_ssl, args.dim, args.heads, args.dropout)
            train_stage2(adapter, dataset, train_idx, device, args.stage2_epochs, args.batch_size)
            states, conventional = extract(adapter, dataset, device, args.batch_size)
            for path in ("s1", "s2", "both"):
                bundle = fit_stage3(states[path][train_idx], conventional[path][train_idx], target[train_idx],
                                    args.seed + int(fold), args.smoke_test)
                prediction = predict_stage3(bundle, states[path][test_idx], conventional[path][test_idx])
                fold_records.append({"fold": int(fold) + 1, "path": path, "n": len(test_idx),
                                     "R2": r2_score(target[test_idx], prediction) if len(test_idx) > 1 else np.nan,
                                     "RMSE": mean_squared_error(target[test_idx], prediction) ** 0.5})
    final_ssl = make_ssl(config); final_ssl.load_state_dict(ssl.state_dict())
    final = TaskAdapter(final_ssl, args.dim, args.heads, args.dropout)
    train_stage2(final, dataset, np.arange(len(dataset)), device, args.stage2_epochs, args.batch_size)
    states, conventional = extract(final, dataset, device, args.batch_size)
    torch.save({"model": final.state_dict(), "config": config, "means": means, "stds": stds},
               output / "stage2_final.pt")
    for path in ("s1", "s2", "both"):
        bundle = fit_stage3(states[path], conventional[path], target, args.seed, args.smoke_test)
        bundle.update({"path": path, "conventional_bands": CONVENTIONAL_BANDS[path],
                       "target_min": float(target.min()), "target_max": float(target.max()),
                       "smoke_test": bool(args.smoke_test)})
        save_stage3(output / f"stage3_{path}.joblib", bundle)
    pd.DataFrame(fold_records).to_csv(output / "fold_metrics.csv", index=False)
    metadata = {"seed": args.seed, "sample_count": len(table), "device": str(device),
                "smoke_test": args.smoke_test, "model_config": config,
                "input": str(table_path), "folds": folds.tolist()}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved three-stage model to {output}")


if __name__ == "__main__":
    main()
