"""
Solus-7B — RMSNorm (Llama-style pre-normalization)
Stable, simpler replacement for LayerNorm that does NOT subtract the mean.
Equivalently: LayerNorm ÷ √(input² history), proven SOTA for Transformer pre-LN.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization — Llama 2/3 implementation.

    Formula:  x · w / rms(x)     where rms(x) = sqrt(mean(x²) + ε)
    No mean subtraction → empirically better for pre-LN cascades.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (…, hidden_size) float tensor

        Returns
        -------
        (…, hidden_size) float tensor — RMS-normalized + scaled
        """
        variance = x.pow(2).mean(-1, keepdim=True)       # (…, 1)
        x_norm = x * torch.rsqrt(variance + self.eps)    # (…, H)
        return self.weight * x_norm.reshape_as(x)        # (…, H) with element-wise scale

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"
