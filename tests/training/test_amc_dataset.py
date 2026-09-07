"""Tests for AMC memmap dataset loader (T19)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from src.training.amc_data import AMCDataCollator
from src.training.amc_dataset import AMCDataset

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _prepare_fixture(tmp_path: Path) -> Path:
    out_dir = tmp_path / "tokenized"
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "prepare_training_data.py"),
        "--output-dir",
        str(out_dir),
        "--train-tokens",
        "8192",
        "--max-seq-len",
        "64",
        "--n-docs",
        "40",
        "--train-split",
        "0.8",
    ]
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)  # noqa: S603 - cmd is built from trusted repo paths and literals
    return out_dir


def test_prepare_training_data_writes_manifest(tmp_path: Path) -> None:
    out_dir = _prepare_fixture(tmp_path)
    manifest_path = out_dir / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["train"]["n_sequences"] > 0
    assert manifest["eval"]["n_sequences"] > 0
    assert Path(manifest["train"]["input_ids_path"]).is_file()


def test_amc_dataset_loads_shapes(tmp_path: Path) -> None:
    out_dir = _prepare_fixture(tmp_path)
    dataset = AMCDataset(out_dir, split="train")
    item = dataset[0]
    assert item.input_ids.shape[1] == dataset.max_seq_len
    assert item.importance_labels.shape == item.input_ids.shape
    assert torch.equal(item.target_ids[:, :-1], item.input_ids[:, 1:])


def test_amc_dataset_collator_batches(tmp_path: Path) -> None:
    out_dir = _prepare_fixture(tmp_path)
    dataset = AMCDataset(out_dir, split="train")
    batch = AMCDataCollator()([dataset[0], dataset[1]])
    assert batch.input_ids.shape[0] == 2
