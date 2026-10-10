from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .config import resolve_path
def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}. Use CSV/XLSX/XLS.")


def split_mask(
    frame: pd.DataFrame,
    fold_seed: int,
    split: str,
    val_fold: int,
    test_fold: int,
) -> np.ndarray:
    if split == "all":
        return np.ones(len(frame), dtype=bool)
    fold_col = f"fold_{fold_seed}"
    if fold_col not in frame.columns:
        raise KeyError(f"Missing {fold_col} in index table.")
    fold = frame[fold_col].astype(int).to_numpy()
    if split == "train":
        return (fold != val_fold) & (fold != test_fold)
    if split == "val":
        return fold == val_fold
    if split == "test":
        return fold == test_fold
    if split in {"development", "non_test", "outer_development"}:
        return fold != test_fold
    raise ValueError(f"Unknown split: {split}")


def resolve_val_fold(all_folds: list[int], test_fold: int, configured: int | None) -> int:
    candidates = [int(v) for v in all_folds if int(v) != int(test_fold)]
    if not candidates:
        raise ValueError("At least two folds are required.")
    if configured is not None and int(configured) != int(test_fold):
        return int(configured)
    position = all_folds.index(test_fold)
    return int(all_folds[(position + 1) % len(all_folds)])


def align_index_and_table(
    index_frame: pd.DataFrame,
    table: pd.DataFrame,
    join_key: str | None,
) -> pd.DataFrame:
    index_frame = index_frame.reset_index(drop=True).copy()
    table = table.reset_index(drop=True).copy()
    if join_key and join_key in index_frame.columns and join_key in table.columns:
        order_col = "__index_order__"
        left = index_frame.copy()
        left[order_col] = np.arange(len(left))
        merged = left[[join_key, order_col]].merge(
            table,
            how="left",
            on=join_key,
            validate="one_to_one",
            sort=False,
            indicator=True,
        ).sort_values(order_col)
        if (merged["_merge"] != "both").any():
            missing_keys = merged.loc[merged["_merge"] != "both", join_key].astype(str).tolist()[:10]
            raise ValueError(f"Some index rows could not be joined by {join_key}: {missing_keys}")
        return merged.drop(columns=[order_col, "_merge"]).reset_index(drop=True)
    if "Src_Row" in index_frame.columns:
        positions = index_frame["Src_Row"].astype(int).to_numpy() - 1
        if len(positions) and (positions.min() < 0 or positions.max() >= len(table)):
            raise IndexError("Src_Row points outside the Excel/CSV table.")
        return table.iloc[positions].reset_index(drop=True)
    if len(index_frame) != len(table):
        raise ValueError(
            "Cannot align image index and Excel/CSV table. Add a shared Sample_ID, "
            "or a one-based Src_Row column to the image index."
        )
    return table.reset_index(drop=True)


def build_aligned_sample_table(
    cfg: dict[str, Any],
    index_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build sample metadata plus predictors taken only from the NPZ/raster source.

    The source index supplies identifiers, SSC, coordinates, folds and NPZ paths.
    Every configured downstream predictor is sampled from the image payload; no
    external feature table participates in this data path.
    """
    data_cfg = cfg["data"]
    table = index_frame.reset_index(drop=True).copy()
    # Stage 1/2 never collapse raster channels into conventional point variables.
    # The Stage-3 evaluator joins its predictors after fold-specific feature export.
    return table.reset_index(drop=True)


def load_npz_image(path: str | Path) -> tuple[np.ndarray, float | None, tuple[float, float] | None]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        if "image" not in payload:
            raise KeyError(f"{path} does not contain key 'image'.")
        image = np.asarray(payload["image"], dtype=np.float32)
        label = float(np.asarray(payload["ssc"]).item()) if "ssc" in payload else None
        center = None
        if "center_pixel" in payload:
            raw_center = np.asarray(payload["center_pixel"], dtype=np.float32).reshape(-1)
            if len(raw_center) >= 2 and np.isfinite(raw_center[:2]).all():
                center = (float(raw_center[0]), float(raw_center[1]))
    if image.ndim != 3:
        raise ValueError(f"Expected CxHxW image, got {image.shape} in {path}")
    return image, label, center


def resize_image_preserve_nan(
    image: np.ndarray,
    target_size: int,
    center_pixel: tuple[float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    """Resize CxHxW data with validity-weighted interpolation and rescale its center."""
    image = np.asarray(image, dtype=np.float32)
    source_h, source_w = image.shape[-2:]
    target_h = target_w = int(target_size)
    if (source_h, source_w) == (target_h, target_w):
        return image, center_pixel
    if target_h <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")

    valid = np.isfinite(image)
    values = torch.from_numpy(np.where(valid, image, 0.0))[None]
    weights = torch.from_numpy(valid.astype(np.float32))[None]
    mode = "area" if target_h <= source_h and target_w <= source_w else "bilinear"
    if mode == "area":
        resized_values = F.interpolate(values, size=(target_h, target_w), mode=mode)
        resized_weights = F.interpolate(weights, size=(target_h, target_w), mode=mode)
    else:
        resized_values = F.interpolate(
            values, size=(target_h, target_w), mode=mode, align_corners=False
        )
        resized_weights = F.interpolate(
            weights, size=(target_h, target_w), mode=mode, align_corners=False
        )
    resized = resized_values / resized_weights.clamp_min(1e-8)
    resized = resized.squeeze(0).numpy()
    resized[resized_weights.squeeze(0).numpy() <= 1e-8] = np.nan

    if center_pixel is not None:
        row_scale = (target_h - 1) / max(source_h - 1, 1)
        col_scale = (target_w - 1) / max(source_w - 1, 1)
        center_pixel = (
            float(center_pixel[0]) * row_scale,
            float(center_pixel[1]) * col_scale,
        )
    return resized.astype(np.float32), center_pixel


def prepare_image_size(
    image: np.ndarray,
    center_pixel: tuple[float, float] | None,
    expected_size: int,
    allow_resize: bool,
    source: str,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    if image.shape[-2:] == (expected_size, expected_size):
        return image, center_pixel
    if not allow_resize:
        raise ValueError(
            f"{source} is {image.shape[-2:]}; data.image_size requires "
            f"{(expected_size, expected_size)}. Set data.resize_to_image_size=true "
            "to enable explicit validity-weighted resizing."
        )
    return resize_image_preserve_nan(image, expected_size, center_pixel)


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Install h5py or set cache.enabled=false.") from exc
    return h5py


def cache_fingerprint(
    index_path: Path,
    npz_root: Path,
    band_names: list[str],
    selected_bands: list[str],
    image_size: int,
    resize_to_image_size: bool,
) -> str:
    frame = pd.read_csv(index_path)
    inventory = []
    for rel in frame["NPZ_Path"].astype(str):
        path = npz_root / rel
        stat = path.stat()
        inventory.append((rel, int(stat.st_size), int(stat.st_mtime_ns)))
    payload = {
        "version": 3,
        "index": str(index_path.resolve()),
        "index_mtime": int(index_path.stat().st_mtime_ns),
        "inventory": inventory,
        "bands": band_names,
        "selected_bands": selected_bands,
        "image_size": int(image_size),
        "resize_to_image_size": bool(resize_to_image_size),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_hdf5_cache(cfg: dict[str, Any], force: bool = False) -> Path | None:
    cache_cfg = cfg.get("cache", {})
    if not bool(cache_cfg.get("enabled", False)):
        return None
    data_cfg = cfg["data"]
    index_path = resolve_path(cfg, data_cfg["source_index"])
    npz_root = resolve_path(cfg, data_cfg["source_npz_root"])
    cache_path = resolve_path(cfg, cache_cfg["path"])
    assert index_path and npz_root and cache_path
    frame = pd.read_csv(index_path)
    bands = list(data_cfg["band_names"])
    selected_bands = list(dict.fromkeys(list(data_cfg["s2_bands"]) + list(data_cfg["s1_bands"])))
    selected_indices = [bands.index(name) for name in selected_bands]
    expected_size = int(data_cfg["image_size"])
    allow_resize = bool(data_cfg.get("resize_to_image_size", False))
    fingerprint = cache_fingerprint(
        index_path, npz_root, bands, selected_bands, expected_size, allow_resize
    )
    h5py = _require_h5py()
    if cache_path.exists() and not force:
        try:
            with h5py.File(cache_path, "r") as handle:
                if handle.attrs.get("fingerprint", "") == fingerprint:
                    return cache_path
        except OSError:
            pass
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    first, _, first_center = load_npz_image(npz_root / str(frame.iloc[0]["NPZ_Path"]))
    if first.shape[0] != len(bands):
        raise ValueError(f"First NPZ has {first.shape[0]} channels, config has {len(bands)} bands.")
    selected_first, _ = prepare_image_size(
        first[selected_indices], first_center, expected_size, allow_resize, "First NPZ"
    )
    shape = (len(frame),) + tuple(selected_first.shape)
    compression = cache_cfg.get("compression", "lzf")
    compression = None if compression in {None, "", "none"} else compression
    string_dtype = h5py.string_dtype(encoding="utf-8")
    try:
        with h5py.File(tmp, "w") as handle:
            images = handle.create_dataset(
                "images",
                shape=shape,
                dtype="float32",
                chunks=(1,) + tuple(selected_first.shape),
                compression=compression,
                shuffle=True,
            )
            sample_ids = handle.create_dataset("sample_ids", (len(frame),), dtype=string_dtype)
            labels = handle.create_dataset("npz_labels", (len(frame),), dtype="float32")
            centers = handle.create_dataset("center_pixels", (len(frame), 2), dtype="float32")
            for i, row in frame.iterrows():
                image, label, center = load_npz_image(npz_root / str(row["NPZ_Path"]))
                if image.shape != first.shape:
                    raise ValueError(f"All NPZ images must share one shape; row {i} has {image.shape}, expected {first.shape}.")
                selected, center = prepare_image_size(
                    image[selected_indices],
                    center,
                    expected_size,
                    allow_resize,
                    str(row["NPZ_Path"]),
                )
                images[i] = selected
                sample_ids[i] = str(row.get("Sample_ID", i))
                labels[i] = np.nan if label is None else label
                centers[i] = (np.nan, np.nan) if center is None else center
            handle.attrs["fingerprint"] = fingerprint
            handle.attrs["bands_json"] = json.dumps(selected_bands)
            handle.attrs["source_bands_json"] = json.dumps(bands)
            handle.attrs["source_image_shape_json"] = json.dumps(list(first.shape[-2:]))
            handle.attrs["target_image_size"] = expected_size
            handle.attrs["resize_to_image_size"] = allow_resize
        os.replace(tmp, cache_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return cache_path


class HDF5Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle = None
        self._pid: int | None = None
        self._positions: dict[str, int] | None = None

    def _open(self):
        h5py = _require_h5py()
        pid = os.getpid()
        if self._handle is None or self._pid != pid:
            self.close()
            self._handle = h5py.File(self.path, "r")
            self._pid = pid
            ids = self._handle["sample_ids"].asstr()[:]
            self._positions = {str(v): i for i, v in enumerate(ids)}
        return self._handle

    def get(self, sample_id: str) -> tuple[np.ndarray, float | None, tuple[float, float] | None]:
        handle = self._open()
        assert self._positions is not None
        if sample_id not in self._positions:
            raise KeyError(f"{sample_id} is missing from cache {self.path}")
        pos = self._positions[sample_id]
        image = np.asarray(handle["images"][pos], dtype=np.float32)
        label = float(handle["npz_labels"][pos])
        raw_center = np.asarray(handle["center_pixels"][pos], dtype=np.float32) if "center_pixels" in handle else np.asarray([np.nan, np.nan])
        center = (float(raw_center[0]), float(raw_center[1])) if np.isfinite(raw_center).all() else None
        return image, label if np.isfinite(label) else None, center

    def close(self) -> None:
        handle = self._handle
        self._handle = None
        self._pid = None
        self._positions = None
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    def __getstate__(self):
        return {"path": self.path, "_handle": None, "_pid": None, "_positions": None}

    def __del__(self) -> None:
        self.close()


@dataclass
class FoldStats:
    band_names: list[str]
    image_center: list[float]
    image_scale: list[float]
    image_lower: list[float]
    image_upper: list[float]
    target_mean: float
    target_std: float
    fit_seed: int
    fit_test_fold: int
    fit_val_fold: int
    fit_split: str
    image_stats_source: str = ""

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "FoldStats":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(**json.load(f))

    def transform_image(self, image: np.ndarray, clip_z: float | None) -> np.ndarray:
        center = np.asarray(self.image_center, dtype=np.float32)[:, None, None]
        scale = np.asarray(self.image_scale, dtype=np.float32)[:, None, None]
        lower = np.asarray(self.image_lower, dtype=np.float32)[:, None, None]
        upper = np.asarray(self.image_upper, dtype=np.float32)[:, None, None]
        x = np.where(np.isfinite(image), image, center)
        x = np.clip(x, lower, upper)
        x = (x - center) / scale
        if clip_z is not None:
            x = np.clip(x, -clip_z, clip_z)
        return x.astype(np.float32)

    def transform_target(self, value: float | np.ndarray) -> np.ndarray:
        return (np.asarray(value, dtype=np.float32) - self.target_mean) / self.target_std

    def inverse_target(self, value: float | np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float32) * self.target_std + self.target_mean


def with_shared_image_stats(
    local_stats: FoldStats,
    shared_stats: FoldStats,
    source: str | Path,
) -> FoldStats:
    """Use GlobalSSL image normalization while retaining fold-local target stats."""
    if list(local_stats.band_names) != list(shared_stats.band_names):
        raise ValueError(
            "GlobalSSL and fold-local image bands differ: "
            f"global={shared_stats.band_names}, local={local_stats.band_names}."
        )
    return replace(
        local_stats,
        band_names=list(shared_stats.band_names),
        image_center=list(shared_stats.image_center),
        image_scale=list(shared_stats.image_scale),
        image_lower=list(shared_stats.image_lower),
        image_upper=list(shared_stats.image_upper),
        image_stats_source=str(Path(source).resolve()),
    )


def _safe_scale(values: np.ndarray, robust: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if robust:
        center = np.nanmedian(values, axis=0)
        q25 = np.nanquantile(values, 0.25, axis=0)
        q75 = np.nanquantile(values, 0.75, axis=0)
        scale = (q75 - q25) / 1.349
    else:
        center = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    scale = np.where(np.isfinite(scale) & (np.abs(scale) > 1e-8), scale, 1.0)
    return center.astype(np.float64), scale.astype(np.float64)


def fit_fold_stats(
    cfg: dict[str, Any],
    fold_seed: int,
    test_fold: int,
    val_fold: int,
    output_path: str | Path,
    fit_split: str = "development",
) -> FoldStats:
    data_cfg = cfg["data"]
    index_path = resolve_path(cfg, data_cfg["source_index"])
    npz_root = resolve_path(cfg, data_cfg["source_npz_root"])
    assert index_path and npz_root
    full_index = pd.read_csv(index_path)
    mask = split_mask(full_index, fold_seed, fit_split, val_fold, test_fold)
    index = full_index.loc[mask].reset_index(drop=True)
    table = build_aligned_sample_table(cfg, index)
    if len(index) == 0:
        raise ValueError(f"Statistics split {fit_split} is empty.")

    bands = list(data_cfg["band_names"])
    image_bands = list(dict.fromkeys(list(data_cfg["s2_bands"]) + list(data_cfg["s1_bands"])))
    image_indices = [bands.index(v) for v in image_bands]
    cache_path = build_hdf5_cache(cfg)
    store = HDF5Store(cache_path) if cache_path else None
    rng = np.random.default_rng(int(fold_seed) + int(test_fold) * 1000)
    pixels_per_patch = int(data_cfg.get("stats_pixels_per_patch", 4096))
    sampled: list[list[np.ndarray]] = [[] for _ in image_bands]
    for row_idx, row in index.iterrows():
        sample_id = str(row.get("Sample_ID", row_idx))
        image, _, center_pixel = store.get(sample_id) if store else load_npz_image(npz_root / str(row["NPZ_Path"]))
        if image.shape[0] == len(bands):
            selected = image[image_indices]
        elif image.shape[0] == len(image_bands):
            selected = image
        else:
            raise ValueError(
                f"{row['NPZ_Path']} or cache has {image.shape[0]} channels; expected "
                f"{len(bands)} raw bands or {len(image_bands)} selected S1/S2 bands."
            )
        selected, _ = prepare_image_size(
            selected,
            center_pixel,
            int(data_cfg["image_size"]),
            bool(data_cfg.get("resize_to_image_size", False)),
            str(row["NPZ_Path"]),
        )
        for channel in range(len(image_bands)):
            values = selected[channel].reshape(-1)
            values = values[np.isfinite(values)]
            if len(values) > pixels_per_patch:
                values = values[rng.choice(len(values), pixels_per_patch, replace=False)]
            if len(values):
                sampled[channel].append(values.astype(np.float32))
    if store:
        store.close()
    lower_q, upper_q = data_cfg.get("image_winsor_quantiles", [0.005, 0.995])
    centers, scales, lowers, uppers = [], [], [], []
    for chunks in sampled:
        if not chunks:
            centers.append(0.0); scales.append(1.0); lowers.append(-np.inf); uppers.append(np.inf)
            continue
        values = np.concatenate(chunks)
        lower = float(np.quantile(values, lower_q))
        upper = float(np.quantile(values, upper_q))
        clipped = np.clip(values, lower, upper)
        center, scale = _safe_scale(clipped[:, None], robust=True)
        centers.append(float(center[0])); scales.append(float(scale[0])); lowers.append(lower); uppers.append(upper)

    target_col = str(data_cfg["target_column"])
    required = [target_col]
    missing = [c for c in required if c not in table.columns]
    if missing:
        raise KeyError(
            f"Source index/raster predictors are missing required columns: {missing}."
        )
    target = table[target_col].to_numpy(dtype=np.float64)
    target = target[np.isfinite(target)]
    if len(target) == 0:
        target_mean, target_std = 0.0, 1.0
    else:
        target_mean = float(np.mean(target))
        target_std = float(np.std(target))
        if target_std < 1e-8:
            target_std = 1.0
    stats = FoldStats(
        band_names=image_bands,
        image_center=centers,
        image_scale=scales,
        image_lower=lowers,
        image_upper=uppers,
        target_mean=target_mean,
        target_std=target_std,
        fit_seed=int(fold_seed),
        fit_test_fold=int(test_fold),
        fit_val_fold=int(val_fold),
        fit_split=fit_split,
    )
    stats.save(output_path)
    return stats


class MsrlPatchDataset(Dataset):
    def __init__(
        self,
        cfg: dict[str, Any],
        stats: FoldStats,
        fold_seed: int,
        test_fold: int,
        val_fold: int,
        split: str,
        require_label: bool,
        augment: bool,
    ) -> None:
        data_cfg = cfg["data"]
        index_path = resolve_path(cfg, data_cfg["source_index"])
        npz_root = resolve_path(cfg, data_cfg["source_npz_root"])
        assert index_path and npz_root
        full_index = pd.read_csv(index_path)
        mask = split_mask(full_index, fold_seed, split, val_fold, test_fold)
        self.index = full_index.loc[mask].reset_index(drop=True)
        self.table = build_aligned_sample_table(cfg, self.index)
        self.npz_root = npz_root
        self.stats = stats
        self.all_bands = list(data_cfg["band_names"])
        self.selected_bands = list(stats.band_names)
        self.selected_idx = [self.all_bands.index(v) for v in self.selected_bands]
        self.s2_idx = [self.selected_bands.index(v) for v in data_cfg["s2_bands"]]
        self.s1_idx = [self.selected_bands.index(v) for v in data_cfg["s1_bands"]]
        s2_names = list(data_cfg["s2_bands"])
        intensity_names = list(data_cfg.get("s2_intensity_augment_bands", s2_names))
        unknown_intensity = [name for name in intensity_names if name not in s2_names]
        if unknown_intensity:
            raise KeyError(f"s2_intensity_augment_bands are not S2-side bands: {unknown_intensity}")
        self.s2_intensity_idx = [s2_names.index(name) for name in intensity_names]
        self.target_col = str(data_cfg["target_column"])
        self.s2_available_column = data_cfg.get("s2_available_column")
        self.s1_available_column = data_cfg.get("s1_available_column")
        for column in [self.s2_available_column, self.s1_available_column]:
            if column and column not in self.index.columns:
                raise KeyError(f"Configured availability column is missing from index: {column}")
        self.require_label = bool(require_label)
        self.augment = bool(augment)
        self.image_clip_z = data_cfg.get("image_clip_z", 8.0)
        self.expected_size = int(data_cfg["image_size"])
        self.allow_resize = bool(data_cfg.get("resize_to_image_size", False))
        cache_path = build_hdf5_cache(cfg)
        self.store = HDF5Store(cache_path) if cache_path else None
        required = [self.target_col] if require_label else []
        missing = [c for c in required if c not in self.table.columns]
        if missing:
            raise KeyError(f"Missing aligned table columns: {missing}")

    def __len__(self) -> int:
        return len(self.index)

    def _augment(
        self, s2: np.ndarray, s1: np.ndarray, local_coordinates: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        local = local_coordinates.astype(np.float32).copy()
        k = np.random.randint(0, 4)
        s2 = np.rot90(s2, k, axes=(1, 2)).copy()
        s1 = np.rot90(s1, k, axes=(1, 2)).copy()
        y, x = float(local[0]), float(local[1])
        if k == 1:
            local[:] = (-x, y)
        elif k == 2:
            local[:] = (-y, -x)
        elif k == 3:
            local[:] = (x, -y)
        if np.random.rand() < 0.5:
            s2 = s2[:, :, ::-1].copy(); s1 = s1[:, :, ::-1].copy()
            local[1] *= -1.0
        if np.random.rand() < 0.5:
            s2 = s2[:, ::-1, :].copy(); s1 = s1[:, ::-1, :].copy()
            local[0] *= -1.0
        # Modality-specific weak radiometric jitter after shared geometry.
        if np.random.rand() < 0.8 and self.s2_intensity_idx:
            scales = np.random.uniform(0.95, 1.05, size=(len(self.s2_intensity_idx), 1, 1)).astype(np.float32)
            s2[self.s2_intensity_idx] = s2[self.s2_intensity_idx] * scales
        if np.random.rand() < 0.8:
            s1 = s1 + np.random.normal(0.0, 0.03, size=(s1.shape[0], 1, 1)).astype(np.float32)
        return s2, s1, np.clip(local, -1.0, 1.0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.index.iloc[idx]
        sample_id = str(row.get("Sample_ID", idx))
        image, npz_label, center_pixel = self.store.get(sample_id) if self.store else load_npz_image(self.npz_root / str(row["NPZ_Path"]))
        if image.shape[0] == len(self.all_bands):
            selected_raw = image[self.selected_idx]
        elif image.shape[0] == len(self.selected_bands):
            selected_raw = image
        else:
            raise ValueError(
                f"{row['NPZ_Path']} or cache has {image.shape[0]} channels; expected "
                f"{len(self.all_bands)} raw bands or {len(self.selected_bands)} selected S1/S2 bands."
            )
        selected_raw, center_pixel = prepare_image_size(
            selected_raw,
            center_pixel,
            self.expected_size,
            self.allow_resize,
            str(row["NPZ_Path"]),
        )
        selected = self.stats.transform_image(selected_raw, self.image_clip_z)
        s2 = selected[self.s2_idx]
        s1 = selected[self.s1_idx]
        if center_pixel is None:
            center_pixel = ((image.shape[-2] - 1) / 2.0, (image.shape[-1] - 1) / 2.0)
        local_coordinates = np.asarray([
            2.0 * center_pixel[0] / max(image.shape[-2] - 1, 1) - 1.0,
            2.0 * center_pixel[1] / max(image.shape[-1] - 1, 1) - 1.0,
        ], dtype=np.float32)
        local_coordinates = np.clip(local_coordinates, -1.0, 1.0)
        if self.augment:
            s2, s1, local_coordinates = self._augment(s2, s1, local_coordinates)
        has_s2 = bool(row[self.s2_available_column]) if self.s2_available_column else True
        has_s1 = bool(row[self.s1_available_column]) if self.s1_available_column else True
        if not has_s2:
            s2 = np.zeros_like(s2)
        if not has_s1:
            s1 = np.zeros_like(s1)
        target_raw = float(self.table.iloc[idx][self.target_col]) if self.require_label else float("nan")
        if self.require_label and not np.isfinite(target_raw):
            raise ValueError(f"Missing {self.target_col} for sample {sample_id}")
        if self.require_label and npz_label is not None and not np.isclose(target_raw, npz_label, atol=1e-5, rtol=1e-4):
            raise ValueError(f"Table/NPZ target mismatch for {sample_id}: {target_raw} vs {npz_label}")
        target = float(self.stats.transform_target(target_raw)) if np.isfinite(target_raw) else 0.0
        return {
            "s2": torch.from_numpy(s2),
            "s1": torch.from_numpy(s1),
            "local_coordinates": torch.from_numpy(local_coordinates),
            "target": torch.tensor(target, dtype=torch.float32),
            "target_raw": torch.tensor(target_raw if np.isfinite(target_raw) else 0.0, dtype=torch.float32),
            "has_label": torch.tensor(np.isfinite(target_raw), dtype=torch.bool),
            "has_s2": torch.tensor(has_s2, dtype=torch.bool),
            "has_s1": torch.tensor(has_s1, dtype=torch.bool),
            "has_both": torch.tensor(has_s2 and has_s1, dtype=torch.bool),
            "sample_id": sample_id,
            "longitude": torch.tensor(float(row.get("Longitude", np.nan)), dtype=torch.float32),
            "latitude": torch.tensor(float(row.get("Latitude", np.nan)), dtype=torch.float32),
        }


def build_dataset(
    cfg: dict[str, Any],
    stats: FoldStats,
    fold_seed: int,
    test_fold: int,
    val_fold: int,
    split: str,
    require_label: bool,
    augment: bool,
) -> MsrlPatchDataset:
    return MsrlPatchDataset(cfg, stats, fold_seed, test_fold, val_fold, split, require_label, augment)


class MsrlUnlabeledDataset(Dataset):
    """Image-only S1/S2 dataset for optional large unlabeled pretraining pools."""

    def __init__(
        self,
        cfg: dict[str, Any],
        stats: FoldStats,
        fold_seed: int,
        test_fold: int,
        val_fold: int,
        split: str,
        augment: bool,
    ) -> None:
        data_cfg = cfg["data"]
        using_external = bool(data_cfg.get("pretrain_index"))
        index_path = resolve_path(cfg, data_cfg.get("pretrain_index") or data_cfg["source_index"])
        npz_root = resolve_path(cfg, data_cfg.get("pretrain_npz_root") or data_cfg["source_npz_root"])
        assert index_path and npz_root
        frame = pd.read_csv(index_path)
        if split != "all":
            fold_col = f"fold_{fold_seed}"
            if fold_col not in frame.columns:
                raise KeyError(
                    f"Unlabeled pretraining index {index_path} lacks {fold_col}; "
                    "use pretraining.scope=all_unlabeled."
                )
            frame = frame.loc[split_mask(frame, fold_seed, split, val_fold, test_fold)]
        self.index = frame.reset_index(drop=True)
        self.npz_root = npz_root
        self.stats = stats
        self.all_bands = list(data_cfg["band_names"])
        self.selected_bands = list(stats.band_names)
        self.selected_idx = [self.all_bands.index(v) for v in self.selected_bands]
        self.s2_idx = [self.selected_bands.index(v) for v in data_cfg["s2_bands"]]
        self.s1_idx = [self.selected_bands.index(v) for v in data_cfg["s1_bands"]]
        s2_names = list(data_cfg["s2_bands"])
        intensity_names = list(data_cfg.get("s2_intensity_augment_bands", s2_names))
        unknown_intensity = [name for name in intensity_names if name not in s2_names]
        if unknown_intensity:
            raise KeyError(f"s2_intensity_augment_bands are not S2-side bands: {unknown_intensity}")
        self.s2_intensity_idx = [s2_names.index(name) for name in intensity_names]
        self.expected_size = int(data_cfg["image_size"])
        self.image_clip_z = data_cfg.get("image_clip_z", 8.0)
        self.s2_available_column = data_cfg.get("s2_available_column")
        self.s1_available_column = data_cfg.get("s1_available_column")
        for column in [self.s2_available_column, self.s1_available_column]:
            if column and column not in self.index.columns:
                raise KeyError(f"Configured availability column is missing from pretraining index: {column}")
        self.augment = bool(augment)
        # The source cache cannot be reused for an external pool with a different index.
        self.store = None
        if not using_external:
            cache_path = build_hdf5_cache(cfg)
            self.store = HDF5Store(cache_path) if cache_path else None

    def __len__(self) -> int:
        return len(self.index)

    def _augment_pair(self, s2: np.ndarray, s1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = np.random.randint(0, 4)
        s2 = np.rot90(s2, k, axes=(1, 2)).copy()
        s1 = np.rot90(s1, k, axes=(1, 2)).copy()
        if np.random.rand() < 0.5:
            s2 = s2[:, :, ::-1].copy(); s1 = s1[:, :, ::-1].copy()
        if np.random.rand() < 0.5:
            s2 = s2[:, ::-1, :].copy(); s1 = s1[:, ::-1, :].copy()
        if np.random.rand() < 0.8 and self.s2_intensity_idx:
            scales = np.random.uniform(0.95, 1.05, size=(len(self.s2_intensity_idx), 1, 1)).astype(np.float32)
            s2[self.s2_intensity_idx] = s2[self.s2_intensity_idx] * scales
        if np.random.rand() < 0.8:
            s1 = s1 + np.random.normal(0.0, 0.03, size=(s1.shape[0], 1, 1)).astype(np.float32)
        return s2, s1

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.index.iloc[idx]
        sample_id = str(row.get("Sample_ID", idx))
        if self.store is not None:
            image, _, _ = self.store.get(sample_id)
        else:
            image, _, _ = load_npz_image(self.npz_root / str(row["NPZ_Path"]))
        if image.shape[-2:] != (self.expected_size, self.expected_size):
            raise ValueError(
                f"Unlabeled image {row['NPZ_Path']} has spatial shape {image.shape[-2:]}; "
                f"expected {(self.expected_size, self.expected_size)}."
            )
        if image.shape[0] == len(self.all_bands):
            selected_raw = image[self.selected_idx]
        elif image.shape[0] == len(self.selected_bands):
            selected_raw = image
        else:
            raise ValueError(
                f"Unlabeled image {row['NPZ_Path']} has {image.shape[0]} channels; expected "
                f"{len(self.all_bands)} raw bands or {len(self.selected_bands)} selected S1/S2 bands."
            )
        selected = self.stats.transform_image(selected_raw, self.image_clip_z)
        s2, s1 = selected[self.s2_idx], selected[self.s1_idx]
        has_s2 = bool(row[self.s2_available_column]) if self.s2_available_column else True
        has_s1 = bool(row[self.s1_available_column]) if self.s1_available_column else True
        if not has_s2:
            s2 = np.zeros_like(s2)
        if not has_s1:
            s1 = np.zeros_like(s1)
        if self.augment:
            s2, s1 = self._augment_pair(s2, s1)
        return {
            "s2": torch.from_numpy(s2),
            "s1": torch.from_numpy(s1),
            "has_s2": torch.tensor(has_s2, dtype=torch.bool),
            "has_s1": torch.tensor(has_s1, dtype=torch.bool),
            "has_both": torch.tensor(has_s2 and has_s1, dtype=torch.bool),
            "sample_id": sample_id,
        }


def build_ssl_dataset(
    cfg: dict[str, Any],
    stats: FoldStats,
    fold_seed: int,
    test_fold: int,
    val_fold: int,
    split: str,
    augment: bool,
) -> MsrlUnlabeledDataset:
    return MsrlUnlabeledDataset(cfg, stats, fold_seed, test_fold, val_fold, split, augment)
