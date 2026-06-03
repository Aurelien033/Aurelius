"""Tests for Ring 1 checkpoint loader (Tranche 5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.ring1_model_loader import (
    checkpoint_sha256_for_path,
    infer_amc_config_from_state_dict,
    load_ring1_amc_model,
    resolve_checkpoint_weights_path,
)

CHECKPOINT_ROOT = Path("checkpoints/aurelius-1.3b")


@pytest.mark.skipif(
    resolve_checkpoint_weights_path(CHECKPOINT_ROOT) is None,
    reason="local checkpoint not present",
)
def test_resolve_checkpoint_weights_path_finds_legacy_step_dir() -> None:
    weights = resolve_checkpoint_weights_path(CHECKPOINT_ROOT)
    assert weights is not None
    assert weights.name == "model.safetensors"
    assert "step-" in str(weights.parent)


@pytest.mark.skipif(
    resolve_checkpoint_weights_path(CHECKPOINT_ROOT) is None,
    reason="local checkpoint not present",
)
def test_checkpoint_sha256_is_not_placeholder() -> None:
    digest = checkpoint_sha256_for_path(CHECKPOINT_ROOT)
    assert digest != "dummy-checkpoint-not-loaded"
    assert len(digest) == 64


@pytest.mark.skipif(
    resolve_checkpoint_weights_path(CHECKPOINT_ROOT) is None,
    reason="local checkpoint not present",
)
def test_load_ring1_amc_model_smoke() -> None:
    model, weights_path, digest = load_ring1_amc_model(CHECKPOINT_ROOT)
    assert weights_path.exists()
    assert digest != "dummy-checkpoint-not-loaded"
    assert model.config.vocab_size > 0
    assert model.config.use_hlm_bank is True


def test_infer_amc_config_from_state_dict() -> None:
    import torch

    state = {
        "embed.weight": torch.zeros(256, 64),
        "layers.0.attn.q_proj.weight": torch.zeros(64, 64),
        "layers.0.ffn.gate_proj.weight": torch.zeros(128, 64),
        "layers.1.attn.q_proj.weight": torch.zeros(64, 64),
    }
    config = infer_amc_config_from_state_dict(state)
    assert config.vocab_size == 256
    assert config.d_model == 64
    assert config.n_layers == 2
    assert config.d_ff == 128
