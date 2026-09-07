"""
SwiGLU MLP — swish-gated linear unit.
Stable for dense 7B; used exactly as proven in LLaMA/Qwen/Mistral.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class SolusMLP(nn.Module):
    """Gated MLP: down(up(x) ⊗ SiLU(gate(x)))  with tied intermediate dim."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act       = nn.SiLU(inplace=False)   # Swish = SiLU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
