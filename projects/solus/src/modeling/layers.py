"""
Solus-7B — Decoder layer.
Pre-Norm (RMSNorm) → GQA Attention → residual → pre-Norm → SwiGLU MLP → residual.
`layers.py` is only imported by `model.py` — NOT the reverse — so there are no circular imports.
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from .config  import SolusConfig
from .attention import SolusAttention
from .norm import RMSNorm
from .mlp import SolusMLP


class SolusDecoderLayer(nn.Module):
    """
    Single pre-LN Transformer decoder block:
        x ──► RMSNorm ──► GQA Attention ──► +x ──► RMSNorm ──► SwiGLU MLP ──► +x
    """

    def __init__(self, cfg: SolusConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.cfg       = cfg

        self.input_layernorm          = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn                = SolusAttention(cfg, layer_idx=layer_idx)
        self.post_attn_layernorm      = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp                      = SolusMLP(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        # ── Attention residual ──────────────────────────────────────────────
        residual_a = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states)
        hidden_states = residual_a + hidden_states

        # ── MLP residual ───────────────────────────────────────────────────
        residual_m = hidden_states
        hidden_states = self.post_attn_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual_m + hidden_states

        return hidden_states
