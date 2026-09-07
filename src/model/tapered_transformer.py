"""Tapered MLP utilities for Aurelius transformer blocks.

Paper basis: Bayat et al. 2026, "Tapered Language Models".
The important correction is that tapering should be applied to the MLP
intermediate width d_ff, not the residual dimension d_model. Keeping d_model
constant preserves attention/residual compatibility while reallocating FFN
capacity toward early layers.
"""

from __future__ import annotations

import math
from dataclasses import replace

import torch.nn as nn

from .config import AureliusConfig
from .ffn import SwiGLUFFN


def round_to_multiple(value: float, multiple: int = 256) -> int:
    """Round width to a hardware-friendly multiple."""
    if multiple <= 0:
        return int(round(value))
    return max(multiple, int(round(value / multiple) * multiple))


def cosine_taper_dff(
    baseline_dff: int,
    layer_idx: int,
    n_layers: int,
    start_mult: float = 1.5,
    end_mult: float = 0.5,
    multiple_of: int = 256,
) -> int:
    """Cosine tapered FFN width for one layer.

    Implements d_ff(l) = d_end + (d_start - d_end) *
    (1 + cos(pi*l/(L-1))) / 2, rounded for tensor-core efficiency.
    The default 1.5 -> 0.5 schedule preserves the average d_ff budget.
    """
    if n_layers <= 1:
        return round_to_multiple(baseline_dff, multiple_of)
    d_start = start_mult * baseline_dff
    d_end = end_mult * baseline_dff
    phase = math.pi * layer_idx / (n_layers - 1)
    width = d_end + (d_start - d_end) * (1.0 + math.cos(phase)) / 2.0
    return round_to_multiple(width, multiple_of)


def tapered_dff_schedule(
    baseline_dff: int,
    n_layers: int,
    start_mult: float = 1.5,
    end_mult: float = 0.5,
    multiple_of: int = 256,
) -> list[int]:
    """Return per-layer tapered FFN widths."""
    return [
        cosine_taper_dff(baseline_dff, i, n_layers, start_mult, end_mult, multiple_of)
        for i in range(n_layers)
    ]


class TaperedSwiGLUFFN(nn.Module):
    """Drop-in SwiGLU FFN with layer-specific tapered d_ff.

    This does not change d_model, attention heads, RoPE, or residual shape.
    """

    def __init__(self, config: AureliusConfig, layer_idx: int, n_layers: int):
        super().__init__()
        tapered_dff = cosine_taper_dff(config.d_ff, layer_idx, n_layers)
        self.layer_idx = layer_idx
        self.baseline_dff = config.d_ff
        self.tapered_dff = tapered_dff
        self.ffn = SwiGLUFFN(replace(config, d_ff=tapered_dff))

    def forward(self, x):
        return self.ffn(x)
