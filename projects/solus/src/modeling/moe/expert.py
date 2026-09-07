"""
Solus-MoE -- Single Expert

Identical architecture to SolusMLP (3 linear layers: gate, up, down).

Notable notes
==============

Output method: nil (skip is working.)
"""
from __future__ import annotations
import torch
import torch.nn as nn


class SolusExpert(nn.Module):
    """Single SwiGLU expert: gate(x) * up(x) -> down(output)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act       = nn.SiLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
