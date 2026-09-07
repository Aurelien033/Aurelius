"""Tests for Ring 1 reproducibility pack builder (Tranche 3)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import yaml

from src.eval.ring1_repro_pack import build_manifest, validate_pack, write_repro_pack


def test_build_manifest_contains_hashes() -> None:
    config = {
        "tranche": 3,
        "seeds": {"base_seed": 1},
        "model": {"checkpoint_path": "checkpoints/aurelius-1.3b"},
    }
    manifest = build_manifest(
        config=config,
        artifact_paths={"traces": "data/traces.jsonl"},
        repo_root=Path("."),
    )
    assert manifest.config_hash
    assert manifest.git_sha
    assert manifest.pack_id.startswith("ring1-repro-")


def test_write_and_validate_repro_pack() -> None:
    config = yaml.safe_load(Path("configs/ring1_tranche3.yaml").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        traces = tmp / "traces.jsonl"
        traces.write_text('{"trace_id":"x","steps":[]}\n', encoding="utf-8")
        dreambank = tmp / "dreambank_run"
        dreambank.mkdir()
        (dreambank / "pre_cycle_model_hash.txt").write_text("abc\n", encoding="utf-8")
        (dreambank / "post_cycle_model_hash.txt").write_text("abc\n", encoding="utf-8")
        metrics = tmp / "metrics"
        metrics.mkdir()
        (metrics / "aggregated_metrics.json").write_text("{}", encoding="utf-8")

        tarball, pack_dir = write_repro_pack(
            config=config,
            output_dir=tmp / "out",
            traces_path=traces,
            dreambank_dir=dreambank,
            metrics_dir=metrics,
            repo_root=Path("."),
        )
        assert tarball.exists()
        assert (pack_dir / "manifest.json").exists()
        errors = validate_pack(pack_dir)
        assert errors == []
        manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["config_hash"]
