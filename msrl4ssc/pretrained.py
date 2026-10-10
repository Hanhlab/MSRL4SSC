from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .msrl import FlexibleMsrl, ModalityEncoder, CrossBlock
from .config import resolve_path


def _copy(target: torch.Tensor, source: torch.Tensor, name: str, report: dict[str, Any]) -> bool:
    if tuple(target.shape) != tuple(source.shape):
        report["skipped"].append({"name": name, "target": list(target.shape), "source": list(source.shape)})
        return False
    with torch.no_grad():
        target.copy_(source.to(dtype=target.dtype))
    report["loaded"].append(name)
    return True


def _adapt_patch_weight(
    linear_weight: torch.Tensor,
    source_channels: int,
    source_patch: int,
    target_channels: int,
    target_patch: int,
    selected_source_channels: list[int],
    extra_channel_init: str,
    selected_source_channel_groups: list[list[int]] | None = None,
) -> torch.Tensor:
    dim = linear_weight.shape[0]
    expected = source_channels * source_patch * source_patch
    if linear_weight.shape[1] != expected:
        raise ValueError(
            f"Official patch weight has flattened size {linear_weight.shape[1]}, expected {expected}."
        )
    weight = linear_weight.reshape(dim, source_channels, source_patch, source_patch)
    if selected_source_channel_groups is not None:
        if len(selected_source_channel_groups) != target_channels:
            raise ValueError(
                "selected_source_channel_groups must provide exactly one group "
                f"per target channel ({target_channels}), got "
                f"{len(selected_source_channel_groups)}"
            )
        if any(not group for group in selected_source_channel_groups):
            raise ValueError("Source channel groups cannot be empty")
        flat = [index for group in selected_source_channel_groups for index in group]
        if any(i < 0 or i >= source_channels for i in flat):
            raise IndexError(
                f"Invalid source channel groups {selected_source_channel_groups} "
                f"for {source_channels} channels"
            )
        # Derived-index channels have no one-to-one counterpart in the official
        # CROMA checkpoint.  Initialise each target channel from the mean kernel
        # of the raw Sentinel-2 bands entering that index, then let local SSL
        # adapt the representation.
        chosen = torch.cat(
            [weight[:, group].mean(dim=1, keepdim=True) for group in selected_source_channel_groups],
            dim=1,
        )
    else:
        if any(i < 0 or i >= source_channels for i in selected_source_channels):
            raise IndexError(f"Invalid source channel indices {selected_source_channels} for {source_channels} channels")
        chosen = weight[:, selected_source_channels]
    if chosen.shape[1] > target_channels:
        chosen = chosen[:, :target_channels]
    if chosen.shape[1] < target_channels:
        missing = target_channels - chosen.shape[1]
        if extra_channel_init == "zero":
            extra = torch.zeros(dim, missing, source_patch, source_patch, dtype=chosen.dtype)
        elif extra_channel_init == "first":
            extra = chosen[:, :1].repeat(1, missing, 1, 1)
        else:
            extra = chosen.mean(dim=1, keepdim=True).repeat(1, missing, 1, 1)
        chosen = torch.cat([chosen, extra], dim=1)
    if source_patch != target_patch:
        reshaped = chosen.reshape(dim * target_channels, 1, source_patch, source_patch)
        resized = F.interpolate(reshaped, size=(target_patch, target_patch), mode="bicubic", align_corners=False)
        chosen = resized.reshape(dim, target_channels, target_patch, target_patch)
        # Preserve approximate response magnitude after changing kernel area.
        chosen = chosen * (source_patch / target_patch) ** 2
    return chosen


def _load_encoder(
    target: ModalityEncoder,
    source: dict[str, torch.Tensor],
    source_channels: int,
    selected_channels: list[int],
    extra_channel_init: str,
    prefix: str,
    report: dict[str, Any],
    selected_channel_groups: list[list[int]] | None = None,
) -> None:
    if "linear_input.weight" not in source:
        raise KeyError(f"{prefix}: official encoder lacks linear_input.weight")
    source_patch = int(round((source["linear_input.weight"].shape[1] / source_channels) ** 0.5))
    patch_weight = _adapt_patch_weight(
        source["linear_input.weight"], source_channels, source_patch,
        target.in_channels, target.patch_size, selected_channels, extra_channel_init,
        selected_channel_groups,
    )
    _copy(target.patch_embed.weight, patch_weight, f"{prefix}.patch_embed.weight", report)
    if "linear_input.bias" in source:
        _copy(target.patch_embed.bias, source["linear_input.bias"], f"{prefix}.patch_embed.bias", report)

    depth = min(len(target.blocks), sum(k.startswith("transformer.layers.") for k in source) + 1)
    for i, block in enumerate(target.blocks):
        base = f"transformer.layers.{i}"
        if f"{base}.0.to_qkv.weight" not in source:
            report["notes"].append(f"{prefix}: source contains only {i} transformer blocks")
            break
        pairs = [
            (block.attn.qkv.weight, source[f"{base}.0.to_qkv.weight"], "attn.qkv.weight"),
            (block.attn.proj.weight, source[f"{base}.0.to_out.weight"], "attn.proj.weight"),
            (block.attn.proj.bias, source[f"{base}.0.to_out.bias"], "attn.proj.bias"),
            (block.attn.norm.weight, source[f"{base}.0.input_norm.weight"], "attn.norm.weight"),
            (block.attn.norm.bias, source[f"{base}.0.input_norm.bias"], "attn.norm.bias"),
            (block.ffn.net[0].weight, source[f"{base}.1.net.0.weight"], "ffn.fc1.weight"),
            (block.ffn.net[0].bias, source[f"{base}.1.net.0.bias"], "ffn.fc1.bias"),
            (block.ffn.net[3].weight, source[f"{base}.1.net.3.weight"], "ffn.fc2.weight"),
            (block.ffn.net[3].bias, source[f"{base}.1.net.3.bias"], "ffn.fc2.bias"),
            (block.ffn.norm.weight, source[f"{base}.1.input_norm.weight"], "ffn.norm.weight"),
            (block.ffn.norm.bias, source[f"{base}.1.input_norm.bias"], "ffn.norm.bias"),
        ]
        for target_tensor, source_tensor, suffix in pairs:
            _copy(target_tensor, source_tensor, f"{prefix}.blocks.{i}.{suffix}", report)
    if "transformer.norm_out.weight" in source:
        _copy(target.norm.weight, source["transformer.norm_out.weight"], f"{prefix}.norm.weight", report)
        _copy(target.norm.bias, source["transformer.norm_out.bias"], f"{prefix}.norm.bias", report)


def _load_cross_blocks(
    blocks: torch.nn.ModuleList,
    source: dict[str, torch.Tensor],
    prefix: str,
    report: dict[str, Any],
) -> None:
    for i, block in enumerate(blocks):
        assert isinstance(block, CrossBlock)
        base = f"layers.{i}"
        if f"{base}.0.to_qkv.weight" not in source:
            report["notes"].append(f"{prefix}: source contains only {i} joint blocks")
            break
        pairs = [
            (block.self_attn.qkv.weight, source[f"{base}.0.to_qkv.weight"], "self_attn.qkv.weight"),
            (block.self_attn.proj.weight, source[f"{base}.0.to_out.weight"], "self_attn.proj.weight"),
            (block.self_attn.proj.bias, source[f"{base}.0.to_out.bias"], "self_attn.proj.bias"),
            (block.self_attn.norm.weight, source[f"{base}.0.input_norm.weight"], "self_attn.norm.weight"),
            (block.self_attn.norm.bias, source[f"{base}.0.input_norm.bias"], "self_attn.norm.bias"),
            (block.cross.q.weight, source[f"{base}.1.to_q.weight"], "cross.q.weight"),
            (block.cross.k.weight, source[f"{base}.1.to_k.weight"], "cross.k.weight"),
            (block.cross.v.weight, source[f"{base}.1.to_v.weight"], "cross.v.weight"),
            (block.cross.proj.weight, source[f"{base}.1.to_out.weight"], "cross.proj.weight"),
            (block.cross.proj.bias, source[f"{base}.1.to_out.bias"], "cross.proj.bias"),
            (block.cross.q_norm.weight, source[f"{base}.1.input_norm.weight"], "cross.q_norm.weight"),
            (block.cross.q_norm.bias, source[f"{base}.1.input_norm.bias"], "cross.q_norm.bias"),
            (block.cross.kv_norm.weight, source[f"{base}.1.input_norm.weight"], "cross.kv_norm.weight"),
            (block.cross.kv_norm.bias, source[f"{base}.1.input_norm.bias"], "cross.kv_norm.bias"),
            (block.ffn.net[0].weight, source[f"{base}.2.net.0.weight"], "ffn.fc1.weight"),
            (block.ffn.net[0].bias, source[f"{base}.2.net.0.bias"], "ffn.fc1.bias"),
            (block.ffn.net[3].weight, source[f"{base}.2.net.3.weight"], "ffn.fc2.weight"),
            (block.ffn.net[3].bias, source[f"{base}.2.net.3.bias"], "ffn.fc2.bias"),
            (block.ffn.norm.weight, source[f"{base}.2.input_norm.weight"], "ffn.norm.weight"),
            (block.ffn.norm.bias, source[f"{base}.2.input_norm.bias"], "ffn.norm.bias"),
        ]
        for target_tensor, source_tensor, suffix in pairs:
            _copy(target_tensor, source_tensor, f"{prefix}.{i}.{suffix}", report)


def _load_gap_head(target: torch.nn.Sequential, source: dict[str, torch.Tensor], prefix: str, report: dict[str, Any]) -> None:
    # Official GAP_FFN: LayerNorm, Linear(4D), GELU, Linear(D)
    pairs = [
        (target[0].weight, source.get("0.weight"), "0.weight"),
        (target[0].bias, source.get("0.bias"), "0.bias"),
        (target[1].weight, source.get("1.weight"), "1.weight"),
        (target[1].bias, source.get("1.bias"), "1.bias"),
        (target[3].weight, source.get("3.weight"), "3.weight"),
        (target[3].bias, source.get("3.bias"), "3.bias"),
    ]
    for target_tensor, source_tensor, suffix in pairs:
        if source_tensor is not None:
            _copy(target_tensor, source_tensor, f"{prefix}.{suffix}", report)


def initialize_from_official_croma(
    model: FlexibleMsrl,
    checkpoint_path: str | Path,
    s2_source_channel_indices: list[int],
    s1_source_channel_indices: list[int],
    extra_s1_channel_init: str = "mean",
    s2_source_channel_groups: list[list[int]] | None = None,
) -> dict[str, Any]:
    """Adapt an official CROMA base/large checkpoint to the configured image-only S2/S1 channel counts.

    Official optical order is assumed to be the 12 Sentinel-2 bands after removing
    cirrus B10: [B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12]. The recommended map
    for [Blue,Green,Red,RE,NIR,SWIR1,SWIR2] is [1,2,3,4,7,10,11].
    """
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Official CROMA checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = ["s2_encoder", "s1_encoder", "joint_encoder"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise KeyError(f"Checkpoint {path} is not an official CROMA checkpoint; missing {missing}")
    source_dim = int(payload["s2_encoder"]["linear_input.weight"].shape[0])
    if source_dim != model.encoder_dim:
        raise ValueError(
            f"Official checkpoint encoder_dim={source_dim}, but config msrl_backbone.encoder_dim="
            f"{model.encoder_dim}. Use 768 for CROMA_base or 1024 for CROMA_large."
        )
    report: dict[str, Any] = {"checkpoint": str(path), "loaded": [], "skipped": [], "notes": []}
    _load_encoder(
        model.s2_encoder,
        payload["s2_encoder"],
        12,
        s2_source_channel_indices,
        "mean",
        "s2_encoder",
        report,
        s2_source_channel_groups,
    )
    _load_encoder(model.s1_encoder, payload["s1_encoder"], 2, s1_source_channel_indices, extra_s1_channel_init, "s1_encoder", report)
    _load_cross_blocks(model.s1_query_s2, payload["joint_encoder"], "s1_query_s2", report)
    # The official model has radar queries optical. Copying it into the reverse
    # direction gives a stable symmetric initialization; local SSL then adapts it.
    _load_cross_blocks(model.s2_query_s1, payload["joint_encoder"], "s2_query_s1", report)
    if "norm_out.weight" in payload["joint_encoder"]:
        for name, norm in [("s1_joint_norm", model.s1_joint_norm), ("s2_joint_norm", model.s2_joint_norm), ("joint_norm", model.joint_norm)]:
            _copy(norm.weight, payload["joint_encoder"]["norm_out.weight"], f"{name}.weight", report)
            _copy(norm.bias, payload["joint_encoder"]["norm_out.bias"], f"{name}.bias", report)
    if "s2_GAP_FFN" in payload:
        _load_gap_head(model.s2_gap_head, payload["s2_GAP_FFN"], "s2_gap_head", report)
    if "s1_GAP_FFN" in payload:
        _load_gap_head(model.s1_gap_head, payload["s1_GAP_FFN"], "s1_gap_head", report)
    report["loaded_count"] = len(report["loaded"])
    report["skipped_count"] = len(report["skipped"])
    return report


def maybe_initialize_croma(model: FlexibleMsrl, cfg: dict[str, Any]) -> dict[str, Any] | None:
    init_cfg = cfg.get("croma_initialization", {})
    if not bool(init_cfg.get("enabled", False)):
        return None
    checkpoint = resolve_path(cfg, init_cfg.get("path"))
    if checkpoint is None or not checkpoint.exists():
        message = f"Official CROMA checkpoint is missing: {checkpoint}"
        if bool(init_cfg.get("required", True)):
            raise FileNotFoundError(message)
        return {"checkpoint": str(checkpoint), "loaded": [], "skipped": [], "notes": [message]}
    groups = init_cfg.get("s2_source_channel_groups")
    parsed_groups = (
        [[int(index) for index in group] for group in groups]
        if groups is not None
        else None
    )
    return initialize_from_official_croma(
        model,
        checkpoint,
        [int(v) for v in init_cfg.get("s2_source_channel_indices", [1, 2, 3, 4, 7, 10, 11])],
        [int(v) for v in init_cfg.get("s1_source_channel_indices", [0, 1])],
        str(init_cfg.get("extra_s1_channel_init", "zero")),
        parsed_groups,
    )
