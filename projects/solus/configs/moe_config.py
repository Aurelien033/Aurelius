"""
configs/moe_config.py
=====================
Serialises the production Solus-MoE 8B hyper-parameter set as a
stand-alone Python config  (also convertible to JSON via cfg.to_pretrained()).

Use:
    from configs.moe_config import solus_moe_8b_cfg
    cfg = solus_moe_8b_cfg()          # SolusConfig instance

Architecture
------------
  Embeddings  V × H    :  525,336,576  (525.3 M)
  24 × Dense  24 ×:   4,756,537,344  (4756.5 M)
  8 × MoE     8  ×:   2,743,402,496  (2743.4 M)
  Final norm  H         :        4,096  (  0.0 M)
  ─────────────────────────────────────────────
  Total              :   8,025,280,512  (8025.3 M)

Component  formula
------------
  Dense per_layer = attn (QKVO proj)
                 + 3 × H × I_dense   (SwiGLU FFN: gate+up+down)
                 + 2 × H              (pre-attn + post-attn RMSNorm)
                 = 198,189,056

  MoE per_layer   = attn
                 + E × 3 × H × I_expert   (E experts × SwiGLU)
                 + H × E                  (router gate proj)
                 + 2 × H                  (2 × RMSNorm)
                 = 342,925,312

  E = moe_num_experts = 8
  I_dense  = intermediate_size         = 12,800
  I_expert = expert_intermediate_size  =  3,072
  MoE layers at indices: [0, 4, 8, 12, 16, 20, 24, 28]
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from modeling.config import SolusConfig


def solus_moe_8b_cfg() -> SolusConfig:
    """Return the exact Solus-MoE 8B configuration.

    Parameter count  : 8,025 M  (exact: 8,025,280,512)
    Hidden size      : 4,096
    Layers           : 32  (24 dense + 8 MoE)
    Attention heads  : 32 GQA  (key-value = 7 shared heads)
    MoE   experts    : 8  (top-k = 2 routing)
    MoE expert FFN   : inter = 3,072
    Vocab size       : 128,256
    Max seq length   : 8,192
    """
    return SolusConfig(
        name                          = "solus-moe-8b",
        architecture                  = "solus-moe-transformer",
        # ── Core dims ──────────────────────────────────────────────────────
        hidden_size                   = 4096,
        num_hidden_layers             = 32,
        num_attention_heads           = 32,
        num_key_value_heads           = 7,
        # ── Dense FFN ─────────────────────────────────────────────────────
        intermediate_size             = 12800,
        # ── MoE ───────────────────────────────────────────────────────────
        expert_intermediate_size      = 3072,
        moe_num_experts               = 8,
        moe_top_k                     = 2,
        moe_layer_indices             = [0, 4, 8, 12, 16, 20, 24, 28],
        # ── Position / normalization ───────────────────────────────────────
        max_position_embeddings       = 8192,
        rope_theta                    = 10000.0,
        rms_norm_eps                  = 1e-5,
        # ── Tokenizer ──────────────────────────────────────────────────────
        vocab_size                    = 128256,
        bos_token_id                  = 128000,
        eos_token_id                  = 128001,
        # ── Router ─────────────────────────────────────────────────────────
        moe_router_jitter_eps         = 0.01,
        moe_aux_loss_alpha            = 0.01,
    )


__all__ = ["solus_moe_8b_cfg"]
