"""Tests for AMCTransformer parameter counting and config validation."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.count_params import (
    count_by_category,
    count_parameters,
    validate_amc_config,
    validate_config,
)
from src.model.amc_transformer import AMCTransformer, AMCTransformerConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _small_config(**overrides: object) -> AMCTransformerConfig:
    base: dict[str, object] = dict(
        vocab_size=1000,
        d_model=128,
        n_layers=4,
        n_heads=4,
        ssm_d_state=32,
        ssm_headdim=32,
        ssm_expand=2,
    )
    base.update(overrides)
    return AMCTransformerConfig(**base)


def test_count_params_small_model() -> None:
    model = AMCTransformer(_small_config())
    counts = count_by_category(model)
    assert sum(counts.values()) > 0
    assert counts["ssm_layers"] > 0
    assert counts["mla_layers"] > 0


def test_count_params_total() -> None:
    model = AMCTransformer(_small_config())
    counts = count_parameters(model)
    assert counts["total"] == sum(count_by_category(model).values())


def test_count_params_by_module() -> None:
    model = AMCTransformer(_small_config())
    counts = count_by_category(model)
    assert counts["embed"] > 0
    assert counts["mla_layers"] > 0
    assert counts["ssm_layers"] > 0


def test_validate_config_catches_bad_combo() -> None:
    cfg = _small_config(d_model=127)
    errors = validate_config(cfg)
    assert errors
    assert any("divisible" in error for error in errors)


def test_invalid_config_detected() -> None:
    cfg = _small_config(n_layers=2, vocab_size=50, max_seq_len=32)
    errors = validate_amc_config(cfg)
    assert len(errors) >= 2


def test_1b_config_is_in_range() -> None:
    config_path = _REPO_ROOT / "configs" / "amc_forge_1b.yaml"
    with config_path.open() as handle:
        raw = yaml.safe_load(handle)
    cfg = AMCTransformerConfig(**raw["model"])
    errors = validate_config(cfg)
    assert not errors, f"config errors: {errors}"
    model = AMCTransformer(cfg)
    total = sum(param.numel() for param in model.parameters())
    assert 0.8e9 < total < 1.2e9, f"model size {total:,} not in [800M, 1.2B]"
