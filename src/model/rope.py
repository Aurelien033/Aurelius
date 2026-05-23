"""Rotary Position Embedding (RoPE) for attention Q/K only (not SSM state)."""

from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE frequency table."""

    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cache: torch.Tensor | None = None
        self._sin_cache: torch.Tensor | None = None
        self._cached_seq_len = 0

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if seq_len <= self._cached_seq_len and self._cos_cache is not None:
            return
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cache = emb.cos().to(dtype)
        self._sin_cache = emb.sin().to(dtype)
        self._cached_seq_len = seq_len

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_cache(seq_len, device, dtype)
        assert self._cos_cache is not None and self._sin_cache is not None
        return self._cos_cache[:seq_len], self._sin_cache[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    position_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors; values are unchanged by design."""
    q_len = q.shape[-2]
    k_len = k.shape[-2]
    q_cos = cos[position_offset : position_offset + q_len]
    q_sin = sin[position_offset : position_offset + q_len]
    k_cos = cos[:k_len]
    k_sin = sin[:k_len]

    def _rotate(tensor: torch.Tensor, cos_slice: torch.Tensor, sin_slice: torch.Tensor) -> torch.Tensor:
        cos_b = cos_slice.view(1, 1, -1, cos_slice.shape[-1])
        sin_b = sin_slice.view(1, 1, -1, sin_slice.shape[-1])
        return tensor * cos_b + rotate_half(tensor) * sin_b

    return _rotate(q, q_cos, q_sin), _rotate(k, k_cos, k_sin)


__all__ = ["RotaryEmbedding", "apply_rope", "rotate_half"]
