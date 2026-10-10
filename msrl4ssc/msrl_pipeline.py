"""Minimal path resolver required by the clean MSRL trainer."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

from .config import resolve_path


@dataclass(frozen=True)
class MsrlPipelinePaths:
    output_root: Path
    root: Path
    ssl_dir: Path
    stats_path: Path
    ssl_best: Path


def msrl_pipeline_paths(cfg: dict) -> MsrlPipelinePaths:
    output_root = resolve_path(cfg, cfg["output"]["root"])
    if output_root is None:
        raise ValueError("output.root must be configured")
    root = output_root / cfg["output"]["msrl_pipeline"]
    ssl_dir = root / "01_ssl"
    return MsrlPipelinePaths(
        output_root=output_root,
        root=root,
        ssl_dir=ssl_dir,
        stats_path=root / "global_stats.json",
        ssl_best=ssl_dir / "best.pt",
    )


def training_completion_reason(
    history: Sequence[dict[str, Any]],
    *,
    epochs: int,
    patience: int,
    min_delta: float,
    score_keys: str | Sequence[str],
    minimum_epochs: int = 1,
) -> str | None:
    """Return why a saved training history is complete, or ``None`` to resume."""
    if not history:
        return None
    last_epoch = int(history[-1].get("epoch", 0))
    if last_epoch >= int(epochs):
        return f"maximum epoch {last_epoch}/{int(epochs)} reached"
    keys = [score_keys] if isinstance(score_keys, str) else list(score_keys)
    best_score = math.inf
    no_improve = 0
    for row in history:
        score = math.nan
        for key in keys:
            try:
                candidate = float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(candidate):
                score = candidate
                break
        if not math.isfinite(score):
            return None
        if score < best_score - float(min_delta):
            best_score = score
            no_improve = 0
        else:
            no_improve += 1
    if last_epoch >= int(minimum_epochs) and no_improve >= int(patience):
        return (
            f"early stopping already reached at epoch {last_epoch} "
            f"({no_improve} non-improving epochs, patience={int(patience)})"
        )
    return None
