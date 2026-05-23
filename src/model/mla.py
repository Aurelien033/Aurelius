"""Multi-head Latent Attention (MLA): compress KV into low-rank latent for efficient KV cache (DeepSeek-V2)."""  # noqa: E501

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rms_norm import RMSNorm
from .rope import RotaryEmbedding, apply_rope


@dataclass
class MLAConfig:
    d_model: int = 512
    n_heads: int = 8
    head_dim: int = 64
    kv_lora_rank: int = 64  # latent dimension for compressed KV
    q_lora_rank: int = 0  # if > 0, also compress queries (0 = no Q compression)
    rope_dim: int = 32  # portion of head_dim that gets RoPE (decoupled RoPE)
    dropout: float = 0.0
    max_seq_len: int = 8192
    rope_theta: float = 10000.0


class DownProjectKV(nn.Module):
    """Compress KV: d_model -> kv_lora_rank."""

    def __init__(self, d_model: int, kv_lora_rank: int) -> None:
        super().__init__()
        self.down_proj = nn.Linear(d_model, kv_lora_rank, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, T, kv_lora_rank)."""
        return self.down_proj(x)


class UpProjectKV(nn.Module):
    """Expand latent back to K and V heads."""

    def __init__(self, kv_lora_rank: int, n_heads: int, head_dim: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.k_up = nn.Linear(kv_lora_rank, n_heads * head_dim, bias=False)
        self.v_up = nn.Linear(kv_lora_rank, n_heads * head_dim, bias=False)

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """c: (B, T, kv_lora_rank) -> (K, V) each (B, n_heads, T, head_dim)."""
        B, T, _ = c.shape
        K = self.k_up(c).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_up(c).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        return K, V


class MultiHeadLatentAttention(nn.Module):
    """Full MLA layer with compressed KV cache."""

    def __init__(self, config: MLAConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        # Q projection
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)

        # KV down-projection and up-projection
        self.kv_down = DownProjectKV(config.d_model, config.kv_lora_rank)
        self.kv_up = UpProjectKV(config.kv_lora_rank, config.n_heads, config.head_dim)

        # Output projection
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.scale = config.head_dim**-0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, d_model)
            mask: optional attention mask
            past_kv: optional past latent cache (B, T_past, kv_lora_rank)

        Returns:
            (output, current_latent_c):
              - output: (B, T, d_model)
              - current_latent_c: (B, T_total, kv_lora_rank) — cache the latent, not K/V
        """
        B, T, _ = x.shape

        # Project Q
        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # (B, n_heads, T, head_dim)

        # Down-project x -> latent c
        c_new = self.kv_down(x)  # (B, T, kv_lora_rank)

        # Concat with past latent if provided
        if past_kv is not None:
            c = torch.cat([past_kv, c_new], dim=1)
        else:
            c = c_new

        # Up-project c -> K, V
        K, V = self.kv_up(c)  # each (B, n_heads, T_total, head_dim)

        # Scaled dot-product attention
        is_causal = mask is None and past_kv is None

        out = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=mask,
            dropout_p=self.config.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # (B, n_heads, T, head_dim) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), c


class MLABlock(nn.Module):
    """Transformer block using MLA + SwiGLU FFN."""

    def __init__(self, config: MLAConfig, d_ff: int) -> None:
        super().__init__()
        # Pre-norm -> MLA
        self.norm1 = RMSNorm(config.d_model)
        self.attn = MultiHeadLatentAttention(config)

        # Pre-norm -> SwiGLU FFN
        self.norm2 = RMSNorm(config.d_model)
        self.gate_proj = nn.Linear(config.d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (output, latent_cache)
        """
        # Pre-norm -> MLA -> residual
        h, latent_cache = self.attn(self.norm1(x), mask=mask, past_kv=past_kv)
        x = x + h

        # Pre-norm -> SwiGLU FFN -> residual
        h = self.norm2(x)
        h = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        x = x + h

        return x, latent_cache


class MultiheadLatentAttention(nn.Module):
    """MLA with expanded K/V cache API for AMC incremental decode (T07)."""

    def __init__(self, config: MLAConfig) -> None:
        super().__init__()
        self.cfg = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        kv_lrank = config.kv_lora_rank or max(32, config.d_model // 8)
        q_lrank = config.q_lora_rank or kv_lrank

        self.q_down = nn.Linear(config.d_model, q_lrank, bias=False)
        self.q_up = nn.Linear(q_lrank, config.d_model, bias=False)
        self.kv_down = nn.Linear(config.d_model, kv_lrank, bias=False)
        self.k_up = nn.Linear(kv_lrank, config.d_model, bias=False)
        self.v_up = nn.Linear(kv_lrank, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, seq_len, dim = x.shape
        cfg = self.cfg

        latents = self.kv_down(x)
        key = self.k_up(latents)
        value = self.v_up(latents)

        if kv_cache is not None:
            key = torch.cat([kv_cache[0], key], dim=1)
            value = torch.cat([kv_cache[1], value], dim=1)

        query = self.q_up(self.q_down(x))
        total_len = key.shape[1]

        query = query.view(batch, seq_len, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        key_heads = key.view(batch, total_len, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        value = value.view(batch, total_len, cfg.n_heads, cfg.head_dim).transpose(1, 2)

        cos, sin = self.rope(total_len, x.device, x.dtype)
        position_offset = kv_cache[0].shape[1] if kv_cache is not None else 0
        query, key_attn = apply_rope(
            query,
            key_heads,
            cos,
            sin,
            position_offset=position_offset,
        )

        scale = 1.0 / math.sqrt(cfg.head_dim)
        attn = (query @ key_attn.transpose(-2, -1)) * scale

        if kv_cache is not None:
            past_len = kv_cache[0].shape[1]
            mask = torch.triu(
                torch.ones(seq_len, total_len, device=x.device, dtype=torch.bool),
                diagonal=past_len + 1,
            )
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        else:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = attn @ value
        out = out.transpose(1, 2).reshape(batch, seq_len, dim)
        out = self.o_proj(out)

        if return_cache:
            key_cache = key_heads.transpose(1, 2).reshape(batch, total_len, dim)
            value_cache = value.transpose(1, 2).reshape(batch, total_len, dim)
            return out, (key_cache, value_cache)
        return out

    @property
    def cache_size_per_token(self) -> int:
        return 2 * self.cfg.kv_lora_rank

    def cache_reduction_vs_mha(self) -> float:
        mha_size = 2 * self.cfg.d_model
        return self.cache_size_per_token / mha_size


def compute_kv_cache_savings(config: MLAConfig) -> dict[str, int | float]:
    """Calculate memory per token: standard vs MLA.

    Returns:
        {"standard_per_token": int, "mla_per_token": int, "compression_ratio": float}
    """
    standard_per_token = 2 * config.n_heads * config.head_dim
    mla_per_token = config.kv_lora_rank
    compression_ratio = standard_per_token / mla_per_token
    return {
        "standard_per_token": standard_per_token,
        "mla_per_token": mla_per_token,
        "compression_ratio": compression_ratio,
    }
