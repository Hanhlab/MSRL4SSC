from __future__ import annotations

import torch
import torch.nn.functional as F

from .models import MultiPathOutput, MsrlRepresentation


def linear_ramp(epoch: int, warmup: int, ramp: int, maximum: float) -> float:
    if epoch <= int(warmup):
        return 0.0
    return float(maximum) * min(1.0, (epoch - int(warmup)) / max(1, int(ramp)))


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def supervised_components(
    output: MsrlRepresentation,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    nll_weight: float,
    rank_weight: float,
    rank_min_gap: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid_mask = valid_mask.bool()
    if not valid_mask.any():
        zero = output.mean.sum() * 0.0
        return zero, {"huber": zero, "nll": zero, "rank": zero}
    error = F.smooth_l1_loss(output.mean, target, reduction="none")
    huber = masked_mean(error, valid_mask)
    nll_each = 0.5 * (
        torch.exp(-output.logvar) * (target - output.mean).pow(2) + output.logvar
    )
    nll = masked_mean(nll_each, valid_mask)
    rank = output.mean.new_tensor(0.0)
    indices = torch.where(valid_mask)[0]
    if rank_weight > 0 and len(indices) > 1:
        y = target[indices]
        p = output.mean[indices]
        target_gap = y[:, None] - y[None, :]
        pred_gap = p[:, None] - p[None, :]
        pair_mask = torch.triu(target_gap.abs() >= float(rank_min_gap), diagonal=1)
        if pair_mask.any():
            rank = F.softplus(-torch.sign(target_gap[pair_mask]) * pred_gap[pair_mask]).mean()
    total = huber + float(nll_weight) * nll + float(rank_weight) * rank
    return total, {"huber": huber, "nll": nll, "rank": rank}


def cosine_alignment(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return a.sum() * 0.0
    values = 1.0 - F.cosine_similarity(a, b, dim=-1)
    if values.ndim > 1:
        values = values.mean(dim=tuple(range(1, values.ndim)))
    return masked_mean(values, mask)


def prediction_consistency(
    a: MsrlRepresentation,
    b: MsrlRepresentation,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not mask.any():
        return a.mean.sum() * 0.0
    return masked_mean(F.smooth_l1_loss(a.mean, b.mean, reduction="none"), mask)


def state_consistency(
    a: MsrlRepresentation,
    b: MsrlRepresentation,
    mask: torch.Tensor,
) -> torch.Tensor:
    return cosine_alignment(a.state, b.state, mask)


def shared_private_orthogonality(output: MsrlRepresentation, mask: torch.Tensor) -> torch.Tensor:
    """Penalize batch cross-covariance between shared and private pooled features."""
    indices = torch.where(mask.bool())[0]
    if len(indices) < 2:
        return output.common_gap.sum() * 0.0
    common = output.common_gap[indices]
    private = output.private_gap[indices]
    common = common - common.mean(dim=0, keepdim=True)
    private = private - private.mean(dim=0, keepdim=True)
    common = F.normalize(common, dim=0)
    private = F.normalize(private, dim=0)
    covariance = common.transpose(0, 1) @ private / max(1, len(indices) - 1)
    return covariance.pow(2).mean()


def msrl_loss_components(
    outputs: MultiPathOutput,
    target: torch.Tensor,
    s2_supervised_mask: torch.Tensor,
    s1_supervised_mask: torch.Tensor,
    both_supervised_mask: torch.Tensor,
    paired_mask: torch.Tensor,
    cfg: dict,
    epoch: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    train_cfg = cfg["msrl_training"]
    loss_cfg = cfg.get("msrl_losses", {})
    nll_weight = linear_ramp(
        epoch,
        int(train_cfg.get("nll_warmup_epochs", 5)),
        int(train_cfg.get("nll_ramp_epochs", 15)),
        float(train_cfg.get("nll_weight", 0.05)),
    )
    rank_weight = float(train_cfg.get("rank_weight", 0.0))
    rank_gap = float(train_cfg.get("rank_min_gap", 0.2))
    zero = target.sum() * 0.0
    parts: dict[str, torch.Tensor] = {}
    total = zero

    path_specs = [
        ("s2", outputs.s2, s2_supervised_mask),
        ("s1", outputs.s1, s1_supervised_mask),
        ("both", outputs.both, both_supervised_mask),
    ]
    for name, output, mask in path_specs:
        if output is None:
            parts[f"{name}_supervised"] = zero
            parts[f"{name}_huber"] = zero
            parts[f"{name}_nll"] = zero
            parts[f"{name}_rank"] = zero
            continue
        supervised, subparts = supervised_components(
            output, target, mask, nll_weight, rank_weight, rank_gap
        )
        parts[f"{name}_supervised"] = supervised
        for key, value in subparts.items():
            parts[f"{name}_{key}"] = value
        total = total + float(loss_cfg.get(f"supervised_{name}", 1.0)) * supervised

    if outputs.s2 is not None and outputs.s1 is not None:
        parts["common_gap_alignment"] = cosine_alignment(
            outputs.s2.common_gap, outputs.s1.common_gap, paired_mask
        )
        parts["common_token_alignment"] = cosine_alignment(
            outputs.s2.common_tokens, outputs.s1.common_tokens, paired_mask
        )
        parts["s2_s1_prediction_consistency"] = prediction_consistency(
            outputs.s2, outputs.s1, paired_mask
        )
        parts["s2_s1_state_consistency"] = state_consistency(
            outputs.s2, outputs.s1, paired_mask
        )
        parts["orthogonal_s2"] = shared_private_orthogonality(outputs.s2, s2_supervised_mask)
        parts["orthogonal_s1"] = shared_private_orthogonality(outputs.s1, s1_supervised_mask)
    else:
        for key in [
            "common_gap_alignment", "common_token_alignment",
            "s2_s1_prediction_consistency", "s2_s1_state_consistency",
            "orthogonal_s2", "orthogonal_s1",
        ]:
            parts[key] = zero

    # MSRL4SSC deliberately has no Both-to-Single transfer objective.
    # ``Full`` below means supervised losses + Lshare + Lcons.
    if outputs.both is not None:
        parts["orthogonal_both"] = shared_private_orthogonality(outputs.both, both_supervised_mask)
    else:
        parts["orthogonal_both"] = zero

    weighted_specs = {
        "common_gap_alignment": ("common_gap_alignment", None),
        "common_token_alignment": ("common_token_alignment", None),
        "s2_s1_prediction_consistency": (
            "s2_s1_prediction_consistency", "prediction_consistency"
        ),
        "s2_s1_state_consistency": (
            "s2_s1_state_consistency", "state_consistency"
        ),
        "orthogonal_s2": ("shared_private_orthogonal", None),
        "orthogonal_s1": ("shared_private_orthogonal", None),
        "orthogonal_both": ("shared_private_orthogonal", None),
    }
    for part_name, (weight_name, legacy_name) in weighted_specs.items():
        if weight_name in loss_cfg:
            weight = float(loss_cfg[weight_name])
        elif legacy_name is not None:
            weight = float(loss_cfg.get(legacy_name, 0.0))
        else:
            weight = 0.0
        total = total + weight * parts[part_name]

    parts["nll_weight"] = target.new_tensor(nll_weight)
    parts["loss"] = total
    return total, parts
