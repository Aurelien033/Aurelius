"""Tests for scripts/evaluate_forge.py (T22)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_evaluate_forge_smoke_writes_summary(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-final.pt"
    checkpoint.write_bytes(b"placeholder")
    out_dir = tmp_path / "eval"
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "evaluate_forge.py"),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(out_dir),
        "--profile",
        "smoke",
        "--mode",
        "oracle",
    ]
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)  # noqa: S603 - cmd is built from trusted repo paths and literals
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "amc_memory" in summary["results"]
    assert "gsm8k" in summary["results"]
    assert "mmlu" in summary["results"]
    assert summary["results"]["gsm8k"]["accuracy"] == 1.0
