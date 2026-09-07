"""Tests for AMC training launcher (T20)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from src.training.launch_amc_training import (
    train_config_from_yaml,
    validate_data_dir,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _prepare_fixture(tmp_path: Path) -> Path:
    out_dir = tmp_path / "tokenized"
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "prepare_training_data.py"),
        "--output-dir",
        str(out_dir),
        "--train-tokens",
        "4096",
        "--max-seq-len",
        "32",
        "--n-docs",
        "20",
        "--train-split",
        "0.8",
    ]
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)  # noqa: S603 - cmd is built from trusted repo paths and literals
    return out_dir


def _tiny_config_yaml(tmp_path: Path) -> Path:
    cfg = {
        "model": {
            "name": "amc-tiny-test",
            "vocab_size": 128000,
            "d_model": 128,
            "n_layers": 4,
            "n_heads": 4,
            "kv_lrank": 32,
            "ssm_d_state": 32,
            "ssm_expand": 2,
            "ssm_headdim": 32,
            "d_conv": 4,
            "max_seq_len": 64,
            "tie_embeddings": True,
            "promotion_temperature": 0.5,
        },
        "training": {
            "batch_size": 2,
            "gradient_accumulation": 1,
            "learning_rate": 3e-4,
            "promotion_gate_lr": 1e-4,
            "surprise_head_lr": 1e-5,
            "weight_decay": 0.01,
            "warmup_steps": 1,
            "max_steps": 3,
            "eval_every": 2,
            "checkpoint_every": 10,
            "grad_clip": 1.0,
            "loss_weights": {
                "sft": 0.70,
                "surprise": 0.15,
                "consistency": 0.10,
                "promotion": 0.05,
            },
        },
        "evaluation": {"eval_batch_size": 2},
        "compute": {"precision": "fp32"},
    }
    path = tmp_path / "tiny_amc.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_validate_data_dir_requires_manifest(tmp_path: Path) -> None:
    errors = validate_data_dir(tmp_path / "missing")
    assert any("manifest" in err for err in errors)


def test_train_config_from_yaml_respects_overrides() -> None:
    raw = {
        "model": {"name": "x"},
        "training": {"batch_size": 8, "max_steps": 100, "loss_weights": {"sft": 0.7}},
    }
    cfg = train_config_from_yaml(raw, max_steps=5, batch_size=1)
    assert cfg.max_steps == 5
    assert cfg.batch_size == 1


def test_launch_amc_training_sanity_run(tmp_path: Path) -> None:
    data_dir = _prepare_fixture(tmp_path)
    assert not validate_data_dir(data_dir)
    config_path = _tiny_config_yaml(tmp_path)
    log_dir = tmp_path / "logs"
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "src" / "training" / "launch_amc_training.py"),
        "--config",
        str(config_path),
        "--data",
        str(data_dir),
        "--log_dir",
        str(log_dir),
        "--max-steps",
        "2",
        "--skip-config-validation",
        "--num-workers",
        "0",
    ]
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)  # noqa: S603 - cmd is built from trusted repo paths and literals
    metrics_path = log_dir / "training.jsonl"
    assert metrics_path.is_file()
    lines = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) >= 2
    assert "total_loss" in lines[0]
    assert (log_dir / "checkpoints" / "checkpoint-final.pt").is_file()
