"""Shared pytest fixtures for Solus-MoE unit tests."""
from __future__ import annotations
import sys, os

_PROJ = os.path.dirname(os.path.abspath(__file__))  # tests/unit/moe/
_SRC  = os.path.join(_PROJ, "..", "..", "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch
import pytest
from src.modeling.config import SolusConfig

# ── Micro config matching dense 1024/8/1 ──────────────────────────────────
@pytest.fixture(scope="session")
def cfg():
    return SolusConfig(
        hidden_size=1024,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=1,
        intermediate_size=1024 * 4,
        max_position_embeddings=2048,
        vocab_size=5000,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_dropout=0.0,
    )

@pytest.fixture(scope="session")
def moe_cfg():
    return SolusConfig(
        name="solus-moe-8b",
        architecture="solus-moe-transformer",
        hidden_size=1024,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=1,
        intermediate_size=1024 * 4,    # dense FFN (used in dense layers)
        expert_intermediate_size=256,  # 256 is MoE expert inter for test (tiny)
        moe_num_experts=4,
        moe_top_k=2,
        moe_layer_indices=[0],         # only layer 0 is MoE in this test config
        vocab_size=5000,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )
