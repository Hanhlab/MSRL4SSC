from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import regression_metrics
from .utils import load_json, move_batch, save_json


Branch = Literal["s2", "s1", "both"]


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
    )


def autocast_context(device: torch.device, enabled: bool):
    active = bool(enabled and device.type == "cuda")
    if not active:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(enabled: bool):
    active = bool(enabled and torch.cuda.is_available())
    try:
        return torch.amp.GradScaler("cuda", enabled=active)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=active)


def forward_msrl(model, branch: Branch, batch: dict[str, Any]):
    common = {"local_coordinates": batch["local_coordinates"]}
    if branch == "s2":
        return model.forward_s2(batch["s2"], **common)
    if branch == "s1":
        return model.forward_s1(batch["s1"], **common)
    if branch == "both":
        return model.forward_both(batch["s2"], batch["s1"], **common)
    raise ValueError(f"Unknown branch: {branch}")


@torch.no_grad()
def evaluate_branch(
    model,
    branch: Branch,
    loader: DataLoader,
    stats,
    device: torch.device,
    amp: bool,
    target_mean: float | None = None,
    target_std: float | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    y_true, y_pred, logvars, sample_ids = [], [], [], []
    availability_key = {"s2": "has_s2", "s1": "has_s1", "both": "has_both"}[branch]
    for batch in loader:
        batch = move_batch(batch, device)
        valid = batch.get(availability_key)
        if valid is None:
            valid = torch.ones(len(batch["sample_id"]), dtype=torch.bool, device=device)
        valid = valid.bool()
        if not valid.any():
            continue
        with autocast_context(device, amp):
            output = forward_msrl(model, branch, batch)
        mean = float(stats.target_mean if target_mean is None else target_mean)
        std = float(stats.target_std if target_std is None else target_std)
        prediction = output.mean.detach().cpu().numpy() * std + mean
        valid_cpu = valid.detach().cpu().numpy().astype(bool)
        y_pred.append(prediction[valid_cpu])
        y_true.append(batch["target_raw"].detach().cpu().numpy()[valid_cpu])
        logvars.append(output.logvar.detach().cpu().numpy()[valid_cpu])
        sample_ids.extend(np.asarray(batch["sample_id"], dtype=object)[valid_cpu].tolist())
    y_true_array = np.concatenate(y_true) if y_true else np.zeros((0,), dtype=np.float32)
    y_pred_array = np.concatenate(y_pred) if y_pred else np.zeros((0,), dtype=np.float32)
    payload = {
        "sample_id": np.asarray(sample_ids, dtype=object),
        "y_true": y_true_array,
        "y_pred": y_pred_array,
        "logvar": np.concatenate(logvars) if logvars else np.zeros((0,), dtype=np.float32),
    }
    return regression_metrics(y_true_array, y_pred_array), payload


@torch.no_grad()
def evaluate_all_paths(
    model,
    loader: DataLoader,
    stats,
    device: torch.device,
    amp: bool,
    target_mean: float | None = None,
    target_std: float | None = None,
) -> dict[str, tuple[dict[str, float], dict[str, np.ndarray]]]:
    return {
        branch: evaluate_branch(model, branch, loader, stats, device, amp, target_mean, target_std)
        for branch in ("s2", "s1", "both")
    }


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def checkpoint_metadata_path(path: str | Path) -> Path:
    """Return the JSON sidecar path for a model-only checkpoint."""
    path = Path(path)
    return path.with_name(f"{path.stem}_metadata.json")


def fp16_model_state_dict(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    """Copy a model state dict to CPU, storing floating tensors as FP16."""
    source = model.state_dict()
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in source.items():
        if not torch.is_tensor(value):
            raise TypeError(f"Model state entry {name!r} is not a tensor: {type(value)}")
        value = value.detach()
        if value.is_floating_point():
            value = value.to(device="cpu", dtype=torch.float16)
        else:
            value = value.to(device="cpu")
        result[name] = value
    if hasattr(source, "_metadata"):
        result._metadata = source._metadata  # type: ignore[attr-defined]
    return result


def save_fp16_model_weights(
    path: str | Path,
    model: torch.nn.Module,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save a pure FP16 model state dict plus optional lightweight JSON metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fp16_model_state_dict(model), path)
    if metadata is not None:
        save_json(dict(metadata), checkpoint_metadata_path(path))


def load_model_weights(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[Mapping[str, torch.Tensor], dict[str, Any]]:
    """
    Load new pure-state-dict checkpoints and legacy full training checkpoints.

    Lightweight metadata is read from ``<stem>_metadata.json``. For legacy
    checkpoints, scalar top-level fields are exposed as metadata as well.
    """
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    metadata: dict[str, Any] = {}
    if isinstance(payload, Mapping) and "model_state" in payload:
        state_dict = payload["model_state"]
        for key, value in payload.items():
            if key.endswith("_state") or key in {"model_state", "encoder_state"}:
                continue
            if value is None or isinstance(value, (str, int, float, bool)):
                metadata[key] = value
    else:
        state_dict = payload
    if not isinstance(state_dict, Mapping) or not all(
        isinstance(name, str) and torch.is_tensor(value)
        for name, value in state_dict.items()
    ):
        raise TypeError(f"Checkpoint does not contain a model state dict: {path}")
    sidecar = checkpoint_metadata_path(path)
    if sidecar.exists():
        loaded_metadata = load_json(sidecar)
        if not isinstance(loaded_metadata, dict):
            raise TypeError(f"Checkpoint metadata must be a JSON object: {sidecar}")
        metadata.update(loaded_metadata)
    return state_dict, metadata
