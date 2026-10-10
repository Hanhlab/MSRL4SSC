from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class MaskInfo:
    ids_keep: torch.Tensor
    ids_restore: torch.Tensor
    mask: torch.Tensor


def random_mask(batch_size: int, length: int, mask_ratio: float, device: torch.device) -> MaskInfo:
    if not 0.0 <= float(mask_ratio) < 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1), got {mask_ratio}")
    keep = max(1, int(round(length * (1.0 - float(mask_ratio)))))
    noise = torch.rand(batch_size, length, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :keep]
    mask = torch.ones(batch_size, length, device=device)
    mask[:, :keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return MaskInfo(ids_keep=ids_keep, ids_restore=ids_restore, mask=mask)


def gather_tokens(tokens: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return torch.gather(tokens, 1, ids.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    b, c, h, w = images.shape
    if h % patch_size or w % patch_size:
        raise ValueError(f"Image size {(h, w)} must be divisible by patch_size={patch_size}")
    gh, gw = h // patch_size, w // patch_size
    x = images.reshape(b, c, gh, patch_size, gw, patch_size)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, c * patch_size * patch_size)


def get_2d_sincos_pos_embed(grid_h: int, grid_w: int, dim: int, device=None) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError("Embedding dimension must be divisible by 4 for 2D sin/cos positions.")
    y = torch.arange(grid_h, dtype=torch.float32, device=device)
    x = torch.arange(grid_w, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    quarter = dim // 4
    omega = torch.arange(quarter, dtype=torch.float32, device=device) / max(quarter, 1)
    omega = 1.0 / (10000**omega)
    out_y = yy.reshape(-1, 1) * omega.reshape(1, -1)
    out_x = xx.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat([out_y.sin(), out_y.cos(), out_x.sin(), out_x.cos()], dim=1).unsqueeze(0)


def alibi_slopes(num_heads: int, device=None) -> torch.Tensor:
    def power_of_two(n: int) -> list[float]:
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(num_heads).is_integer():
        values = power_of_two(num_heads)
    else:
        closest = 2 ** math.floor(math.log2(num_heads))
        values = power_of_two(closest) + power_of_two(2 * closest)[0::2][: num_heads - closest]
    return torch.tensor(values, dtype=torch.float32, device=device)


def full_2d_alibi(grid_h: int, grid_w: int, num_heads: int, device=None) -> torch.Tensor:
    y = torch.arange(grid_h, dtype=torch.float32, device=device)
    x = torch.arange(grid_w, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)
    distance = torch.cdist(coords, coords, p=2)
    slopes = alibi_slopes(num_heads, device=device).view(1, num_heads, 1, 1)
    return -slopes * distance.view(1, 1, grid_h * grid_w, grid_h * grid_w)


def gather_self_bias(bias: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    b, k = ids.shape
    h, length = bias.shape[1], bias.shape[-1]
    expanded = bias.expand(b, -1, -1, -1)
    rows = torch.gather(expanded, 2, ids[:, None, :, None].expand(-1, h, -1, length))
    return torch.gather(rows, 3, ids[:, None, None, :].expand(-1, h, k, -1))


def gather_cross_bias(bias: torch.Tensor, query_ids: torch.Tensor, context_ids: torch.Tensor) -> torch.Tensor:
    b, q = query_ids.shape
    k = context_ids.shape[1]
    h, length = bias.shape[1], bias.shape[-1]
    expanded = bias.expand(b, -1, -1, -1)
    rows = torch.gather(expanded, 2, query_ids[:, None, :, None].expand(-1, h, -1, length))
    return torch.gather(rows, 3, context_ids[:, None, None, :].expand(-1, h, q, k))


class FeedForward(nn.Module):
    def __init__(self, dim: int, ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(round(dim * ratio))
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class BiasedAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, n, 3, self.heads, self.head_dim)
        q, k, v = [t.transpose(1, 2) for t in qkv.unbind(dim=2)]
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if bias is not None:
            logits = logits + bias.to(dtype=logits.dtype)
        weights = torch.softmax(logits, dim=-1)
        out = torch.matmul(self.dropout(weights), v).transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        b, nq, d = x.shape
        nk = context.shape[1]
        q = self.q(self.q_norm(x)).reshape(b, nq, self.heads, self.head_dim).transpose(1, 2)
        ctx = self.kv_norm(context)
        k = self.k(ctx).reshape(b, nk, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(ctx).reshape(b, nk, self.heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if bias is not None:
            logits = logits + bias.to(dtype=logits.dtype)
        weights = torch.softmax(logits, dim=-1)
        out = torch.matmul(self.dropout(weights), v).transpose(1, 2).reshape(b, nq, d)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_ratio: float, dropout: float) -> None:
        super().__init__()
        self.attn = BiasedAttention(dim, heads, dropout)
        self.ffn = FeedForward(dim, ffn_ratio, dropout)

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        x = x + self.attn(x, bias)
        return x + self.ffn(x)


class CrossBlock(nn.Module):
    """self-attention + cross-attention + FFN block."""

    def __init__(self, dim: int, heads: int, ffn_ratio: float, dropout: float) -> None:
        super().__init__()
        self.self_attn = BiasedAttention(dim, heads, dropout)
        self.cross = CrossAttention(dim, heads, dropout)
        self.ffn = FeedForward(dim, ffn_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        self_bias: torch.Tensor | None,
        cross_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + self.self_attn(x, self_bias)
        x = x + self.cross(x, context, cross_bias)
        return x + self.ffn(x)


class ModalityEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        image_size: int,
        patch_size: int,
        dim: int,
        depth: int,
        heads: int,
        ffn_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.patch_embed = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, ffn_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        return self.patch_embed(images).flatten(2).transpose(1, 2)

    def forward(
        self,
        images: torch.Tensor,
        bias: torch.Tensor,
        mask: MaskInfo | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tokens = self.embed(images)
        local_bias = bias
        ids = None
        if mask is not None:
            ids = mask.ids_keep
            tokens = gather_tokens(tokens, ids)
            local_bias = gather_self_bias(bias, ids)
        for block in self.blocks:
            tokens = block(tokens, local_bias)
        return self.norm(tokens), ids


class MAEDecoder(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        depth: int,
        heads: int,
        ffn_ratio: float,
        dropout: float,
        num_patches: int,
        patch_dim: int,
        grid_h: int,
        grid_w: int,
    ) -> None:
        super().__init__()
        self.project = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer("pos_embed", get_2d_sincos_pos_embed(grid_h, grid_w, decoder_dim), persistent=False)
        self.register_buffer("alibi", full_2d_alibi(grid_h, grid_w, heads), persistent=False)
        self.blocks = nn.ModuleList([TransformerBlock(decoder_dim, heads, ffn_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(decoder_dim)
        self.output = nn.Linear(decoder_dim, patch_dim)
        self.num_patches = int(num_patches)

    def forward(self, encoded: torch.Tensor, mask: MaskInfo) -> torch.Tensor:
        x = self.project(encoded)
        b, keep, d = x.shape
        missing = self.num_patches - keep
        x = torch.cat([x, self.mask_token.expand(b, missing, d)], dim=1)
        x = torch.gather(x, 1, mask.ids_restore.unsqueeze(-1).expand(-1, -1, d))
        x = x + self.pos_embed.to(device=x.device, dtype=x.dtype)
        bias = self.alibi.to(device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x = block(x, bias)
        return self.output(self.norm(x))


class SymmetricContrastiveLoss(nn.Module):
    """Symmetric cross-modal InfoNCE with an optional FIFO memory queue."""

    def __init__(self, dim: int, projection_dim: int, temperature: float, queue_size: int = 0) -> None:
        super().__init__()
        self.s1_proj = nn.Linear(dim, projection_dim)
        self.s2_proj = nn.Linear(dim, projection_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
        self.queue_size = max(0, int(queue_size))
        self.register_buffer("s1_queue", torch.zeros(self.queue_size, projection_dim), persistent=True)
        self.register_buffer("s2_queue", torch.zeros(self.queue_size, projection_dim), persistent=True)
        self.register_buffer("queue_ptr", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("queue_filled", torch.zeros((), dtype=torch.long), persistent=True)

    @torch.no_grad()
    def reset_queue(self) -> None:
        """Start a fresh negative bank without reallocating its buffers.

        The local dataset is much smaller than the configured memory bank.  If
        the bank is carried across epochs, the same image eventually appears as
        both the positive pair and a stale negative.  Clearing it once per
        epoch keeps the queue useful for physical batches of two while avoiding
        those repeated-sample false negatives.
        """
        self.queue_ptr.zero_()
        self.queue_filled.zero_()

    @torch.no_grad()
    def _enqueue(self, a: torch.Tensor, b: torch.Tensor) -> None:
        if self.queue_size == 0 or len(a) == 0:
            return
        a = a.detach()
        b = b.detach()
        if len(a) >= self.queue_size:
            self.s1_queue.copy_(a[-self.queue_size :])
            self.s2_queue.copy_(b[-self.queue_size :])
            self.queue_ptr.zero_()
            self.queue_filled.fill_(self.queue_size)
            return
        ptr = int(self.queue_ptr.item())
        first = min(len(a), self.queue_size - ptr)
        self.s1_queue[ptr : ptr + first].copy_(a[:first])
        self.s2_queue[ptr : ptr + first].copy_(b[:first])
        remaining = len(a) - first
        if remaining:
            self.s1_queue[:remaining].copy_(a[first:])
            self.s2_queue[:remaining].copy_(b[first:])
        self.queue_ptr.fill_((ptr + len(a)) % self.queue_size)
        self.queue_filled.fill_(min(self.queue_size, int(self.queue_filled.item()) + len(a)))

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        if s1.shape[0] == 0:
            return s1.new_tensor(0.0)
        a = F.normalize(self.s1_proj(s1), dim=-1)
        b = F.normalize(self.s2_proj(s2), dim=-1)
        # Validation must be comparable from epoch to epoch.  Using the training
        # queue here made the validation candidate count grow from 0 to 1024,
        # so the absolute loss rose even when the representation improved and
        # epoch 1 was almost always selected.  Evaluate with in-batch negatives
        # only; the queue remains a training-only optimization.
        filled = int(self.queue_filled.item()) if self.training else 0
        if len(a) < 2 and filled == 0:
            if self.training:
                self._enqueue(a, b)
            return a.sum() * 0.0
        b_candidates = torch.cat([b, self.s2_queue[:filled].detach()], dim=0)
        a_candidates = torch.cat([a, self.s1_queue[:filled].detach()], dim=0)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits_ab = scale * (a @ b_candidates.T)
        logits_ba = scale * (b @ a_candidates.T)
        labels = torch.arange(len(a), device=a.device)
        loss = 0.5 * (F.cross_entropy(logits_ab, labels) + F.cross_entropy(logits_ba, labels))
        if self.training:
            self._enqueue(a, b)
        return loss


@dataclass
class MsrlOutput:
    s2_tokens: torch.Tensor
    s1_tokens: torch.Tensor
    joint_tokens: torch.Tensor
    s2_gap: torch.Tensor
    s1_gap: torch.Tensor
    joint_gap: torch.Tensor


@dataclass
class S2Output:
    s2_tokens: torch.Tensor
    s2_gap: torch.Tensor


@dataclass
class S1Output:
    s1_tokens: torch.Tensor
    s1_gap: torch.Tensor


class FlexibleMsrl(nn.Module):
    """Msrl-style radar-optical SSL backbone with configurable channels and size."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        s2_channels: int,
        s1_channels: int,
        encoder_dim: int = 256,
        optical_depth: int = 6,
        radar_depth: int = 3,
        joint_depth: int = 3,
        heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
        decoder_dim: int = 192,
        decoder_depth: int = 2,
        decoder_heads: int = 6,
        projection_dim: int = 128,
        contrastive_temperature: float = 0.10,
        contrastive_queue_size: int = 0,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        grid = image_size // patch_size
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid_size = int(grid)
        self.num_patches = int(grid * grid)
        self.encoder_dim = int(encoder_dim)
        self.s2_channels = int(s2_channels)
        self.s1_channels = int(s1_channels)
        self.heads = int(heads)
        self.ffn_ratio = float(ffn_ratio)
        self.s2_encoder = ModalityEncoder(s2_channels, image_size, patch_size, encoder_dim, optical_depth, heads, ffn_ratio, dropout)
        self.s1_encoder = ModalityEncoder(s1_channels, image_size, patch_size, encoder_dim, radar_depth, heads, ffn_ratio, dropout)
        self.s1_query_s2 = nn.ModuleList([CrossBlock(encoder_dim, heads, ffn_ratio, dropout) for _ in range(joint_depth)])
        self.s2_query_s1 = nn.ModuleList([CrossBlock(encoder_dim, heads, ffn_ratio, dropout) for _ in range(joint_depth)])
        self.s1_joint_norm = nn.LayerNorm(encoder_dim)
        self.s2_joint_norm = nn.LayerNorm(encoder_dim)
        self.joint_norm = nn.LayerNorm(encoder_dim)
        self.s1_gap_head = nn.Sequential(
            nn.LayerNorm(encoder_dim), nn.Linear(encoder_dim, encoder_dim * 4),
            nn.GELU(), nn.Linear(encoder_dim * 4, encoder_dim),
        )
        self.s2_gap_head = nn.Sequential(
            nn.LayerNorm(encoder_dim), nn.Linear(encoder_dim, encoder_dim * 4),
            nn.GELU(), nn.Linear(encoder_dim * 4, encoder_dim),
        )
        self.contrast = SymmetricContrastiveLoss(
            encoder_dim, projection_dim, contrastive_temperature, contrastive_queue_size
        )
        self.register_buffer("alibi", full_2d_alibi(grid, grid, heads), persistent=False)
        self.s2_decoder = MAEDecoder(
            encoder_dim, decoder_dim, decoder_depth, decoder_heads, ffn_ratio, dropout,
            self.num_patches, s2_channels * patch_size * patch_size, grid, grid,
        )
        self.s1_decoder = MAEDecoder(
            encoder_dim, decoder_dim, decoder_depth, decoder_heads, ffn_ratio, dropout,
            self.num_patches, s1_channels * patch_size * patch_size, grid, grid,
        )

    def _cross_fuse(
        self,
        s2_tokens: torch.Tensor,
        s1_tokens: torch.Tensor,
        s2_ids: torch.Tensor | None,
        s1_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bias = self.alibi.to(device=s2_tokens.device, dtype=s2_tokens.dtype)
        if s2_ids is None or s1_ids is None:
            s2_self_bias = s1_self_bias = s2_s1_bias = s1_s2_bias = bias
        else:
            s2_self_bias = gather_self_bias(bias, s2_ids)
            s1_self_bias = gather_self_bias(bias, s1_ids)
            s2_s1_bias = gather_cross_bias(bias, s2_ids, s1_ids)
            s1_s2_bias = gather_cross_bias(bias, s1_ids, s2_ids)
        s1_joint, s2_joint = s1_tokens, s2_tokens
        for block in self.s1_query_s2:
            s1_joint = block(s1_joint, s2_tokens, s1_self_bias, s1_s2_bias)
        for block in self.s2_query_s1:
            s2_joint = block(s2_joint, s1_tokens, s2_self_bias, s2_s1_bias)
        return self.s2_joint_norm(s2_joint), self.s1_joint_norm(s1_joint)

    @staticmethod
    def _masked_normalized_mse_per_item(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        target_mean = target.mean(dim=-1, keepdim=True)
        target_var = target.var(dim=-1, keepdim=True, unbiased=False)
        target = (target - target_mean) / torch.sqrt(target_var + 1e-6)
        per_patch = (pred - target).pow(2).mean(dim=-1)
        return (per_patch * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _availability_mean(values: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        weights = available.to(values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1.0)

    def encode_s2(self, s2: torch.Tensor) -> S2Output:
        bias = self.alibi.to(device=s2.device, dtype=s2.dtype)
        tokens, _ = self.s2_encoder(s2, bias, None)
        return S2Output(s2_tokens=tokens, s2_gap=self.s2_gap_head(tokens.mean(dim=1)))

    def encode_s1(self, s1: torch.Tensor) -> S1Output:
        bias = self.alibi.to(device=s1.device, dtype=s1.dtype)
        tokens, _ = self.s1_encoder(s1, bias, None)
        return S1Output(s1_tokens=tokens, s1_gap=self.s1_gap_head(tokens.mean(dim=1)))

    def project_ssl_shared_s2(self, s2_gap: torch.Tensor) -> torch.Tensor:
        """Project an S2 GAP vector into the normalized InfoNCE space."""
        return F.normalize(self.contrast.s2_proj(s2_gap), dim=-1)

    def project_ssl_shared_s1(self, s1_gap: torch.Tensor) -> torch.Tensor:
        """Project an S1 GAP vector into the normalized InfoNCE space."""
        return F.normalize(self.contrast.s1_proj(s1_gap), dim=-1)

    def encode_ssl_shared(
        self,
        s2: torch.Tensor,
        s1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the two modality embeddings learned directly by SSL InfoNCE.

        The projection heads are deliberately exposed here instead of reusing the
        supervised MSRL model common space.  This keeps downstream
        ``SSLShared`` feature sets tied to ``contrast.s1_proj/s2_proj``.
        """
        s2_gap = self.encode_s2(s2).s2_gap
        s1_gap = self.encode_s1(s1).s1_gap
        return self.project_ssl_shared_s2(s2_gap), self.project_ssl_shared_s1(s1_gap)

    def pretrain_forward(
        self,
        s2: torch.Tensor,
        s1: torch.Tensor,
        s2_mask_ratio: float,
        s1_mask_ratio: float,
        has_s2: torch.Tensor | None = None,
        has_s1: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        self-supervision for paired or partially missing modality batches.

        A modality reconstructs itself when its counterpart is absent. Paired items use
        cross-modal fusion and additionally contribute contrastive/alignment losses.
        """
        b = s2.shape[0]
        has_s2 = torch.ones(b, dtype=torch.bool, device=s2.device) if has_s2 is None else has_s2.bool()
        has_s1 = torch.ones(b, dtype=torch.bool, device=s1.device) if has_s1 is None else has_s1.bool()
        paired = has_s2 & has_s1
        if not (has_s2 | has_s1).any():
            raise ValueError("Every pretraining item must contain S1, S2, or both.")

        s2_mask = random_mask(b, self.num_patches, s2_mask_ratio, s2.device)
        s1_mask = random_mask(b, self.num_patches, s1_mask_ratio, s1.device)
        bias = self.alibi.to(device=s2.device, dtype=s2.dtype)
        s2_tokens, s2_ids = self.s2_encoder(s2, bias, s2_mask)
        s1_tokens, s1_ids = self.s1_encoder(s1, bias, s1_mask)
        s2_gap = self.s2_gap_head(s2_tokens.mean(dim=1))
        s1_gap = self.s1_gap_head(s1_tokens.mean(dim=1))

        s2_joint, s1_joint = self._cross_fuse(
            s2_tokens, s1_tokens, s2_ids, s1_ids
        )
        pred_s2_self = self.s2_decoder(s2_tokens, s2_mask)
        pred_s1_self = self.s1_decoder(s1_tokens, s1_mask)
        pred_s2_cross = self.s2_decoder(s2_joint, s2_mask)
        pred_s1_cross = self.s1_decoder(s1_joint, s1_mask)
        paired4 = paired[:, None, None]
        pred_s2 = torch.where(paired4, pred_s2_cross, pred_s2_self)
        pred_s1 = torch.where(paired4, pred_s1_cross, pred_s1_self)
        target_s2 = patchify(s2, self.patch_size)
        target_s1 = patchify(s1, self.patch_size)

        mae_s2_each = self._masked_normalized_mse_per_item(pred_s2, target_s2, s2_mask.mask)
        mae_s1_each = self._masked_normalized_mse_per_item(pred_s1, target_s1, s1_mask.mask)
        if paired.any():
            contrastive = self.contrast(s1_gap[paired], s2_gap[paired])
            alignment = 1.0 - F.cosine_similarity(
                F.normalize(s2_gap[paired], dim=-1),
                F.normalize(s1_gap[paired], dim=-1),
                dim=-1,
            ).mean()
        else:
            contrastive = s2_gap.sum() * 0.0
            alignment = s2_gap.sum() * 0.0
        return {
            "contrastive": contrastive,
            "mae_s2": self._availability_mean(mae_s2_each, has_s2),
            "mae_s1": self._availability_mean(mae_s1_each, has_s1),
            "alignment": alignment,
        }

    def forward(self, s2: torch.Tensor, s1: torch.Tensor) -> MsrlOutput:
        bias = self.alibi.to(device=s2.device, dtype=s2.dtype)
        s2_tokens, _ = self.s2_encoder(s2, bias, None)
        s1_tokens, _ = self.s1_encoder(s1, bias, None)
        s2_joint, s1_joint = self._cross_fuse(
            s2_tokens, s1_tokens, None, None
        )
        joint_tokens = self.joint_norm(0.5 * (s2_joint + s1_joint))
        return MsrlOutput(
            s2_tokens=s2_tokens,
            s1_tokens=s1_tokens,
            joint_tokens=joint_tokens,
            s2_gap=self.s2_gap_head(s2_tokens.mean(dim=1)),
            s1_gap=self.s1_gap_head(s1_tokens.mean(dim=1)),
            joint_gap=joint_tokens.mean(dim=1),
        )

    def encoder_state_dict(self) -> dict[str, Any]:
        excluded = ("s2_decoder.", "s1_decoder.", "contrast.")
        return {k: v for k, v in self.state_dict().items() if not k.startswith(excluded)}

    def load_encoder_state_dict(self, state: dict[str, torch.Tensor], strict: bool = True) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        real_missing = [k for k in missing if not k.startswith(("s2_decoder.", "s1_decoder.", "contrast."))]
        if strict and (real_missing or unexpected):
            raise RuntimeError(f"Encoder checkpoint mismatch. Missing={real_missing}, unexpected={unexpected}")


def build_msrl_from_config(cfg: dict[str, Any]) -> FlexibleMsrl:
    d = cfg["data"]
    m = cfg["msrl_backbone"]
    return FlexibleMsrl(
        image_size=int(d["image_size"]),
        patch_size=int(m["patch_size"]),
        s2_channels=len(d["s2_bands"]),
        s1_channels=len(d["s1_bands"]),
        encoder_dim=int(m["encoder_dim"]),
        optical_depth=int(m["optical_depth"]),
        radar_depth=int(m["radar_depth"]),
        joint_depth=int(m["joint_depth"]),
        heads=int(m["heads"]),
        ffn_ratio=float(m.get("ffn_ratio", 4.0)),
        dropout=float(m.get("dropout", 0.0)),
        decoder_dim=int(m["decoder_dim"]),
        decoder_depth=int(m["decoder_depth"]),
        decoder_heads=int(m["decoder_heads"]),
        projection_dim=int(m.get("projection_dim", 128)),
        contrastive_temperature=float(m.get("contrastive_temperature", 0.10)),
        contrastive_queue_size=int(m.get("contrastive_queue_size", 0)),
    )
