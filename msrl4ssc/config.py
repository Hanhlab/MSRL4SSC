from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    cfg = deepcopy(cfg)
    embedded_project_root = cfg.get("_project_root")
    if embedded_project_root is None:
        project_root = path.parent.parent
    else:
        project_root = Path(embedded_project_root).expanduser()
        if not project_root.is_absolute():
            project_root = path.parent / project_root
    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(project_root.resolve())
    return cfg


def resolve_path(cfg: dict[str, Any], value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(cfg["_project_root"]) / path).resolve()


def save_yaml(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in payload.items() if not str(k).startswith("_")}
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(clean, f, allow_unicode=True, sort_keys=False)


def save_config_snapshot(payload: dict[str, Any], path: str | Path) -> None:
    """Save an immutable run config while retaining its relative-path base."""
    if "_project_root" not in payload:
        raise ValueError("Config snapshots require a loaded config with _project_root.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        key: deepcopy(value)
        for key, value in payload.items()
        if key != "_config_path"
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot, f, allow_unicode=True, sort_keys=False)
