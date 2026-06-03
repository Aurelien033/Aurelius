"""Shared pytest fixtures for Solus-7B unit tests."""
from __future__ import annotations
import sys, os

_PROJ = os.path.dirname(os.path.abspath(__file__))   # tests/
_SRC  = os.path.join(_PROJ, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import math
import torch
import pytest
from src.modeling.config import SolusConfig
from src.modeling.rope import RotaryEmbedding, apply_rope, precompute_rope_freqs

# ── Solus-7B canonical head dim ──────────────────────────────────────────
HEAD_DIM  = 128   # 4096 hidden / 32 heads
HIDDEN    = 1024  # 8 × 128 — small enough for fast testing

# ── Micro-config ──────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def cfg():
    return SolusConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=HIDDEN // HEAD_DIM,
        num_key_value_heads=1,
        head_dim=HEAD_DIM,
        intermediate_size=HIDDEN * 4,
        max_position_embeddings=2048,
        vocab_size=50512,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )
