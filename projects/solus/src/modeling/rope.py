"""
Solus-7B — RoPE (Rotary Position Embeddings)

CONVENTION
----------
head_dim_full = hidden_size // num_attention_heads   (= 128 for Solus-7B)
pair_dim      = head_dim_full // 2                   (=  64 for Solus-7B)

self.dim  = pair_dim — this is what gets stored in the cos/sin buffers.
             cos and sin each have shape (max_seq_len, pair_dim).

The q/k tensors that reach forward() have last dim = head_dim_full (= 2*pair_dim).
forward() splits those into (..., head_dim_full // 2) pair halves, then passes
the corresponding ready_positions cos/sin slices to apply_rope.

PAIR-BY-PAIR ROTATION
---------------------
For each pair ndx of half the vector:
  q[..., ndx] = even
  q[..., pair_dim + ndx] = odd
  cos_i = cos[:, ndx]   sin_i = sin[:, ndx]   (identical rotation angle per pair)
  rot = q_even * cos_i - q_odd * sin_i
  rot = q_even * sin_i + q_odd * cos_i
"""
from __future__ import annotations
import math, torch
from torch import Tensor
import torch.nn as nn


def precompute_rope_freqs(dim: int, seq_len: int,
                          theta: float = 10000.0,
                          device=None) -> tuple[Tensor, Tensor]:
    """Precompute cos/sin table.

    Parameters
    ----------
    dim    : pair dimension = head_dim_full // 2
    seq_len: maximum sequence
    """
    if device is None:
        device = torch.device("cpu")
    freqs = 1.0 / (theta ** (torch.arange(0, dim, dtype=torch.float32, device=device) / dim))
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    angles = torch.outer(positions, freqs)         # (seq_len, dim)
    return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE rotation.

    x   : (..., seq_len, 2*pair_dim)  — q or k
    cos : (..., seq_len, pair_dim)
    sin : (..., seq_len, pair_dim)

    Returns (..., seq_len, 2*pair_dim)
    """
    pair_dim = cos.shape[-1]          # = head_dim_full // 2
    e = x[..., :pair_dim]             # even-indexed pair elements → (..., pair_dim)
    o = x[..., pair_dim:]             # odd-indexed  pair elements → (..., pair_dim)
    return torch.cat([e * cos - o * sin, e * sin + o * cos], dim=-1)


class RotaryEmbedding(nn.Module):
    """Buffered RoPE module — precomputes cos/sin for max_seq_len once."""

    def __init__(self, pair_dim: int, max_seq_len: int = 8192,
                 theta: float = 10000.0,
                 *, rope_scaling: dict | None = None,
                 device=None):
        super().__init__()
        self.pair_dim    = pair_dim
        self.max_seq_len = max_seq_len
        self.theta       = theta
        self.rope_scaling = rope_scaling

        if device is None:
            device = torch.device("cpu")

        if rope_scaling and rope_scaling.get("factor"):
            factor = rope_scaling["factor"]
            self.theta       = theta * ((max_seq_len * factor) / max_seq_len) ** (pair_dim / (pair_dim - 2))
            self.max_seq_len = int(max_seq_len * factor)

        cos, sin = precompute_rope_freqs(pair_dim, self.max_seq_len, self.theta, device=device)
        # cos : (max_seq_len, pair_dim)
        # sin : (max_seq_len, pair_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @property
    def head_dim_full(self) -> int:
        """Full q/k last dimension (= 2 × pair_dim)."""
        return 2 * self.pair_dim

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """Apply RoPE to q or k.

        Parameters
        ----------
        x      : (..., seq_len, head_dim_full)   where head_dim_full = 2 * pair_dim
        offset : absolute starting token position

        Returns
        -------
        (..., seq_len, head_dim_full)
        """
        seq     = x.shape[-2]
        end     = offset + seq
        # Grab only the relevant position slices (no caching history leakage)
        # cos_sinusolar: rows offset..offset+seq, columns 0..pair_dim-1
        positions_cos = self.cos[offset:end, :]
        positions_sin = self.sin[offset:end, :]
        cos = positions_cos.unsqueeze(0)   # (1, seq, pair_dim)
        sin = positions_sin.unsqueeze(0)
        return apply_rope(x, cos, sin)

    def __repr__(self) -> str:
        s = self.rope_scaling if self.rope_scaling else "none"
        return (f"RotaryEmbedding(pair_dim={self.pair_dim}, "
                f"head_dim_full={self.head_dim_full}, "
                f"max_seq_len={self.max_seq_len}, theta={self.theta:.4f}, "
                f"scaling={s})")
