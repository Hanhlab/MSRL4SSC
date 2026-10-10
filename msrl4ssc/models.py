from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn
import torch.nn.functional as F

from .msrl import FlexibleMsrl, build_msrl_from_config


Modality = Literal["s2", "s1", "both"]


class FourierCoordinateEncoder(nn.Module):
    """Encode the within-patch point location with Fourier features."""

    def __init__(
        self,
        output_dim: int,
        num_frequencies: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_frequencies = max(0, int(num_frequencies))
        raw_dim = 2
        encoded_dim = raw_dim * (1 + 2 * self.num_frequencies)
        self.net = nn.Sequential(
            nn.Linear(encoded_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, local_coordinates: torch.Tensor) -> torch.Tensor:
        raw = local_coordinates
        features = [raw]
        for k in range(self.num_frequencies):
            frequency = float(2**k) * torch.pi
            features.extend([torch.sin(frequency * raw), torch.cos(frequency * raw)])
        return self.net(torch.cat(features, dim=1))


class PointTokenAttention(nn.Module):
    """Coordinate query -> image tokens with a learnable spatial distance penalty."""

    def __init__(
        self,
        token_dim: int,
        state_dim: int,
        heads: int,
        grid_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if state_dim % heads:
            raise ValueError(f"state_dim={state_dim} must be divisible by heads={heads}")
        self.heads = int(heads)
        self.head_dim = int(state_dim // heads)
        self.scale = self.head_dim**-0.5
        self.grid_size = int(grid_size)
        self.query_norm = nn.LayerNorm(state_dim)
        self.token_norm = nn.LayerNorm(token_dim)
        self.q = nn.Linear(state_dim, state_dim, bias=False)
        self.k = nn.Linear(token_dim, state_dim, bias=False)
        self.v = nn.Linear(token_dim, state_dim, bias=False)
        self.out = nn.Linear(state_dim, state_dim)
        self.dropout = nn.Dropout(dropout)
        self.log_distance_slopes = nn.Parameter(torch.zeros(heads))

    def forward(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        local_coordinates: torch.Tensor,
        use_distance_bias: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, _ = tokens.shape
        q = self.q(self.query_norm(query)).view(b, self.heads, 1, self.head_dim)
        normalized = self.token_norm(tokens)
        k = self.k(normalized).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(normalized).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if use_distance_bias:
            axis = torch.linspace(-1.0, 1.0, self.grid_size, device=tokens.device, dtype=tokens.dtype)
            yy, xx = torch.meshgrid(axis, axis, indexing="ij")
            grid = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)
            if grid.shape[0] != n:
                raise ValueError(
                    f"Token count {n} does not match configured grid "
                    f"{self.grid_size}x{self.grid_size}."
                )
            distance = torch.linalg.vector_norm(
                grid.unsqueeze(0) - local_coordinates.to(tokens.dtype).unsqueeze(1), dim=-1
            ).view(b, 1, 1, n)
            slopes = F.softplus(self.log_distance_slopes).view(1, self.heads, 1, 1)
            logits = logits - slopes * distance
        weights = torch.softmax(logits, dim=-1)
        attended = torch.matmul(self.dropout(weights), v).transpose(1, 2).reshape(b, 1, -1)
        return self.out(attended).squeeze(1), weights.mean(dim=1).squeeze(1)


class RegressionHead(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        hidden = max(32, dim // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        return out[:, 0], out[:, 1].clamp(-8.0, 4.0)


@dataclass
class MsrlRepresentation:
    mean: torch.Tensor
    logvar: torch.Tensor
    state: torch.Tensor
    image_state: torch.Tensor
    coordinate_state: torch.Tensor
    token_state: torch.Tensor
    image_attention: torch.Tensor
    common_tokens: torch.Tensor
    private_tokens: torch.Tensor
    common_gap: torch.Tensor
    private_gap: torch.Tensor
    modality: str


@dataclass
class MultiPathOutput:
    s2: MsrlRepresentation | None
    s1: MsrlRepresentation | None
    both: MsrlRepresentation | None


class Msrl4SscRegressor(nn.Module):
    """
    One MSRL model with three deployment paths: S2-only, S1-only and paired S1+S2.

    There is no separate teacher network. Sensor-specific encoders feed a shared
    semantic space plus modality-private features; on paired samples the Both path
    can act as a stop-gradient teacher for the two single-modality paths. Every path
    uses the same coordinate-query image state. Stage 2 is image-only; conventional
    predictors are introduced only by the downstream Stage-3 regressor.
    """

    def __init__(
        self,
        msrl: FlexibleMsrl,
        common_dim: int,
        private_dim: int,
        state_dim: int,
        attention_heads: int,
        coordinate_frequencies: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.msrl = msrl
        self.common_dim = int(common_dim)
        self.private_dim = int(private_dim)
        self.state_dim = int(state_dim)
        d = msrl.encoder_dim

        self.s2_common = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, common_dim), nn.LayerNorm(common_dim))
        self.s1_common = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, common_dim), nn.LayerNorm(common_dim))
        self.s2_private = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, private_dim), nn.LayerNorm(private_dim))
        self.s1_private = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, private_dim), nn.LayerNorm(private_dim))

        if common_dim % attention_heads:
            raise ValueError("msrl.common_dim must be divisible by msrl.attention_heads")
        self.s2_query_s1 = nn.MultiheadAttention(common_dim, attention_heads, dropout=dropout, batch_first=True)
        self.s1_query_s2 = nn.MultiheadAttention(common_dim, attention_heads, dropout=dropout, batch_first=True)
        self.common_fusion_gate = nn.Sequential(
            nn.Linear(common_dim * 2, common_dim), nn.GELU(), nn.Linear(common_dim, common_dim), nn.Sigmoid()
        )
        self.common_fusion_norm = nn.LayerNorm(common_dim)

        # The two modality-private streams should not be averaged: they encode
        # intentionally different S2 spectral and S1 scattering information.
        # Use explicit interaction terms to learn both a per-channel modality
        # gate and a nonlinear residual proposal.
        private_interaction_dim = private_dim * 4
        private_hidden = max(64, private_dim * 2)
        self.private_fusion_gate = nn.Sequential(
            nn.LayerNorm(private_interaction_dim),
            nn.Linear(private_interaction_dim, private_dim),
            nn.Sigmoid(),
        )
        self.private_fusion_proposal = nn.Sequential(
            nn.LayerNorm(private_interaction_dim),
            nn.Linear(private_interaction_dim, private_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(private_hidden, private_dim),
        )
        self.private_fusion_norm = nn.LayerNorm(private_dim)

        token_input_dim = common_dim + private_dim
        self.s2_token_adapter = nn.Sequential(nn.LayerNorm(token_input_dim), nn.Linear(token_input_dim, state_dim), nn.LayerNorm(state_dim))
        self.s1_token_adapter = nn.Sequential(nn.LayerNorm(token_input_dim), nn.Linear(token_input_dim, state_dim), nn.LayerNorm(state_dim))
        self.both_token_adapter = nn.Sequential(nn.LayerNorm(token_input_dim), nn.Linear(token_input_dim, state_dim), nn.LayerNorm(state_dim))

        self.coordinate_encoder = FourierCoordinateEncoder(
            state_dim, coordinate_frequencies, dropout
        )
        self.point_attention = PointTokenAttention(
            state_dim, state_dim, attention_heads, msrl.grid_size, dropout
        )
        # Learnable GAP-to-state projections.  Earlier versions used
        # adaptive_avg_pool1d to resize the common/private GAP vectors to
        # ``state_dim``.  Pooling across the feature axis is parameter-free and
        # treats latent feature coordinates as if they were an ordered signal.
        # A learned Linear projection is both semantically cleaner and trainable.
        self.common_gap_projector = nn.Sequential(
            nn.LayerNorm(common_dim),
            nn.Linear(common_dim, state_dim),
            nn.LayerNorm(state_dim),
        )
        self.private_gap_projector = nn.Sequential(
            nn.LayerNorm(private_dim),
            nn.Linear(private_dim, state_dim),
            nn.LayerNorm(state_dim),
        )
        self.image_state_projector = nn.Sequential(
            nn.LayerNorm(state_dim * 3),
            nn.Linear(state_dim * 3, state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(state_dim, state_dim),
            nn.LayerNorm(state_dim),
        )
        self.regression_head = RegressionHead(state_dim, dropout)

    def train(self, mode: bool = True):
        """Keep a fully frozen MSRL backbone deterministic during head training.

        ``nn.Module.train()`` normally re-enables dropout everywhere, including
        modules whose parameters have ``requires_grad=False``.  For the supervised
        frozen-backbone setting we explicitly keep MSRL in eval mode so the
        supervised MSRL heads see a fixed feature extractor.
        """
        super().train(mode)
        if mode and not any(parameter.requires_grad for parameter in self.msrl.parameters()):
            self.msrl.eval()
        return self

    def _encode_s2_latent(self, s2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.msrl.encode_s2(s2)
        return self.s2_common(encoded.s2_tokens), self.s2_private(encoded.s2_tokens)

    def _encode_s1_latent(self, s1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.msrl.encode_s1(s1)
        return self.s1_common(encoded.s1_tokens), self.s1_private(encoded.s1_tokens)

    def _fuse_common(self, s2_common: torch.Tensor, s1_common: torch.Tensor) -> torch.Tensor:
        s2_cross, _ = self.s2_query_s1(s2_common, s1_common, s1_common, need_weights=False)
        s1_cross, _ = self.s1_query_s2(s1_common, s2_common, s2_common, need_weights=False)
        a = s2_common + s2_cross
        b = s1_common + s1_cross
        gate = self.common_fusion_gate(torch.cat([a, b], dim=-1))
        return self.common_fusion_norm(gate * a + (1.0 - gate) * b)

    def _fuse_private(self, s2_private: torch.Tensor, s1_private: torch.Tensor) -> torch.Tensor:
        """Gated interaction fusion for modality-private token streams.

        Interaction = [S2, S1, |S2-S1|, S2*S1].  The interaction predicts a
        per-channel S2-vs-S1 gate plus a nonlinear complementary residual.
        This preserves modality-specific evidence instead of cancelling it by
        arithmetic averaging.
        """
        interaction = torch.cat(
            [
                s2_private,
                s1_private,
                torch.abs(s2_private - s1_private),
                s2_private * s1_private,
            ],
            dim=-1,
        )
        gate = self.private_fusion_gate(interaction)
        gated_base = gate * s2_private + (1.0 - gate) * s1_private
        proposal = self.private_fusion_proposal(interaction)
        return self.private_fusion_norm(gated_base + proposal)

    def _decode_path(
        self,
        common_tokens: torch.Tensor,
        private_tokens: torch.Tensor,
        modality: Modality,
        local_coordinates: torch.Tensor,
    ) -> MsrlRepresentation:
        adapter = {
            "s2": self.s2_token_adapter,
            "s1": self.s1_token_adapter,
            "both": self.both_token_adapter,
        }[modality]
        token_state = adapter(torch.cat([common_tokens, private_tokens], dim=-1))
        coordinate_state = self.coordinate_encoder(local_coordinates)
        point_state, image_attention = self.point_attention(
            coordinate_state, token_state, local_coordinates, use_distance_bias=True
        )
        common_gap = common_tokens.mean(dim=1)
        private_gap = private_tokens.mean(dim=1)
        common_state = self.common_gap_projector(common_gap)
        private_state = self.private_gap_projector(private_gap)
        image_state = self.image_state_projector(
            torch.cat([point_state, common_state, private_state], dim=1)
        )
        state = image_state
        mean, logvar = self.regression_head(state)
        return MsrlRepresentation(
            mean=mean,
            logvar=logvar,
            state=state,
            image_state=image_state,
            coordinate_state=coordinate_state,
            token_state=token_state,
            image_attention=image_attention,
            common_tokens=common_tokens,
            private_tokens=private_tokens,
            common_gap=common_gap,
            private_gap=private_gap,
            modality=modality,
        )

    def forward_s2(
        self,
        s2: torch.Tensor,
        local_coordinates: torch.Tensor,
    ) -> MsrlRepresentation:
        common, private = self._encode_s2_latent(s2)
        return self._decode_path(common, private, "s2", local_coordinates)

    def forward_s1(
        self,
        s1: torch.Tensor,
        local_coordinates: torch.Tensor,
    ) -> MsrlRepresentation:
        common, private = self._encode_s1_latent(s1)
        return self._decode_path(common, private, "s1", local_coordinates)

    def forward_both(
        self,
        s2: torch.Tensor,
        s1: torch.Tensor,
        local_coordinates: torch.Tensor,
    ) -> MsrlRepresentation:
        s2_common, s2_private = self._encode_s2_latent(s2)
        s1_common, s1_private = self._encode_s1_latent(s1)
        common = self._fuse_common(s2_common, s1_common)
        private = self._fuse_private(s2_private, s1_private)
        return self._decode_path(common, private, "both", local_coordinates)

    def forward(
        self,
        s2: torch.Tensor | None,
        s1: torch.Tensor | None,
        local_coordinates: torch.Tensor,
        mode: Modality | Literal["auto"] = "auto",
    ) -> MsrlRepresentation:
        if mode == "auto":
            mode = "both" if s2 is not None and s1 is not None else "s2" if s2 is not None else "s1"
        if mode == "both":
            if s2 is None or s1 is None:
                raise ValueError("mode='both' requires S1 and S2")
            return self.forward_both(s2, s1, local_coordinates)
        if mode == "s2":
            if s2 is None:
                raise ValueError("mode='s2' requires S2")
            return self.forward_s2(s2, local_coordinates)
        if mode == "s1":
            if s1 is None:
                raise ValueError("mode='s1' requires S1")
            return self.forward_s1(s1, local_coordinates)
        raise ValueError(f"Unknown mode: {mode}")

    def forward_all(
        self,
        s2: torch.Tensor,
        s1: torch.Tensor,
        local_coordinates: torch.Tensor,
        compute_s2: bool = True,
        compute_s1: bool = True,
        compute_both: bool = True,
    ) -> MultiPathOutput:
        # Reuse the expensive modality encoders once.
        s2_common = s2_private = s1_common = s1_private = None
        if compute_s2 or compute_both:
            s2_common, s2_private = self._encode_s2_latent(s2)
        if compute_s1 or compute_both:
            s1_common, s1_private = self._encode_s1_latent(s1)
        out_s2 = self._decode_path(
            s2_common, s2_private, "s2", local_coordinates
        ) if compute_s2 else None
        out_s1 = self._decode_path(
            s1_common, s1_private, "s1", local_coordinates
        ) if compute_s1 else None
        out_both = None
        if compute_both:
            assert s2_common is not None and s1_common is not None
            assert s2_private is not None and s1_private is not None
            both_common = self._fuse_common(s2_common, s1_common)
            both_private = self._fuse_private(s2_private, s1_private)
            out_both = self._decode_path(
                both_common,
                both_private,
                "both",
                local_coordinates,
            )
        return MultiPathOutput(s2=out_s2, s1=out_s1, both=out_both)


def configure_msrl_trainability(model: Msrl4SscRegressor, mode: str, last_n_blocks: int) -> None:
    for parameter in model.msrl.parameters():
        parameter.requires_grad = False
    if mode == "frozen":
        return
    if mode == "full":
        for parameter in model.msrl.parameters():
            parameter.requires_grad = True
        return
    if mode != "last_blocks":
        raise ValueError("msrl.finetune_mode must be frozen, last_blocks, or full")
    n = max(1, int(last_n_blocks))
    block_groups = [
        model.msrl.s2_encoder.blocks,
        model.msrl.s1_encoder.blocks,
        model.msrl.s1_query_s2,
        model.msrl.s2_query_s1,
    ]
    for blocks in block_groups:
        for block in list(blocks)[-n:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    for module in [
        model.msrl.s2_encoder.norm,
        model.msrl.s1_encoder.norm,
        model.msrl.s1_joint_norm,
        model.msrl.s2_joint_norm,
        model.msrl.joint_norm,
        model.msrl.s1_gap_head,
        model.msrl.s2_gap_head,
    ]:
        for parameter in module.parameters():
            parameter.requires_grad = True


def build_msrl_model(cfg: dict[str, Any]) -> Msrl4SscRegressor:
    model_cfg = cfg["msrl_model"]
    model = Msrl4SscRegressor(
        msrl=build_msrl_from_config(cfg),
        common_dim=int(model_cfg.get("common_dim", 128)),
        private_dim=int(model_cfg.get("private_dim", 64)),
        state_dim=int(model_cfg.get("state_dim", 256)),
        attention_heads=int(model_cfg.get("attention_heads", 8)),
        coordinate_frequencies=int(model_cfg.get("coordinate_frequencies", 4)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )
    configure_msrl_trainability(
        model,
        str(model_cfg.get("finetune_mode", "last_blocks")),
        int(model_cfg.get("last_n_blocks", 1)),
    )
    return model
