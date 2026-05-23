"""RMSNorm — Root Mean Square Layer Normalization."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMS normalization with learned per-element scale."""

    def __init__(self, dim: int, *, eps: float = 1e-6, bias: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f32 = x.to(torch.float32)
        rms = x_f32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        out = x_f32 * rms
        out = out.to(dtype)
        bias = self.bias if self.bias is not None else 0
        return out * self.weight + bias

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}, bias={self.bias is not None}"


__all__ = ["RMSNorm"]
