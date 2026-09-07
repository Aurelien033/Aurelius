"""
Solus-MoE -- Decoder Layer with MoE

Hybrid:   Pre-RMSNorm -> GQA Attention -> residual ->
          Post-Attn RMSNorm -> SolusMoELayer (sparse) -> residual

Used only for layers in cfg.moe_layer_indices.
All other  layers use the standard dense SolusDecoderLayer.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..config import SolusConfig
from ..attention import SolusAttention
from ..norm import RMSNorm
from .moe_layer  import SolusMoELayer


class SolusSparseDecoderLayer(nn.Module):
    """
    Pre-norm, GQA attention first, then sparse MoE.

    Sequence:
        x -> RMSNorm(attn_norm) -> SolusAttention -> +x -> residual[attn]
        -> RMSNorm(moe_norm)   -> SolusMoELayer  -> +x -> residual[MoE]
    """

    def __init__(self, cfg: SolusConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx   = layer_idx
        self.cfg         = cfg
        self.input_layernorm     = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn           = SolusAttention(cfg, layer_idx=layer_idx)
        self.post_attn_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.moe_layer           = SolusMoELayer(cfg, layer_idx=layer_idx)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Optional[Tensor]]:
        # Attention residual
        residual_a = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states)
        hidden_states = residual_a + hidden_states

        # MoE residual -- capture auxiliary load-balance loss
        residual_m = hidden_states
        hidden_states = self.post_attn_layernorm(hidden_states)
        hidden_states, aux_loss = self.moe_layer(hidden_states)
        hidden_states = residual_m + hidden_states

        return hidden_states, aux_loss
