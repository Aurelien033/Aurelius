"""Tests for AMC Forge 1B full-config validation (T18)."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.validate_amc_forge_config import validate_amc_forge_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "configs" / "amc_forge_1b.yaml"


def _load_config() -> dict:
    with _CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_forge_config_passes_validation() -> None:
    errors = validate_amc_forge_config(_load_config(), repo_root=_REPO_ROOT)
    assert errors == [], f"unexpected errors: {errors}"


def test_forge_config_rejects_bad_lr() -> None:
    raw = _load_config()
    raw["training"]["learning_rate"] = 1e-3
    errors = validate_amc_forge_config(raw, repo_root=_REPO_ROOT)
    assert any("learning_rate" in error for error in errors)


def test_forge_config_rejects_loss_weight_drift() -> None:
    raw = _load_config()
    raw["training"]["loss_weights"]["sft"] = 0.90
    errors = validate_amc_forge_config(raw, repo_root=_REPO_ROOT)
    assert any("loss_weights" in error for error in errors)


def test_forge_config_rejects_data_seq_longer_than_model() -> None:
    raw = _load_config()
    raw["data"]["max_seq_len"] = raw["model"]["max_seq_len"] + 1
    errors = validate_amc_forge_config(raw, repo_root=_REPO_ROOT)
    assert any("data.max_seq_len" in error for error in errors)
