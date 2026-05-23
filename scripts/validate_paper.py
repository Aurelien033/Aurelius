#!/usr/bin/env python3
"""Validate paper/main.tex skeleton (T32)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
MAIN = _REPO / "paper" / "main.tex"


def main() -> int:
    if not MAIN.is_file():
        print(f"MISSING {MAIN}", file=sys.stderr)
        return 1

    text = MAIN.read_text(encoding="utf-8")
    required = [
        r"\begin{abstract}",
        r"\section{Introduction}",
        r"\section{Experiments}",
        r"tab:ablation",
        r"tab:safety",
    ]
    missing = [token for token in required if token not in text]
    if missing:
        print("Missing LaTeX fragments:", ", ".join(missing), file=sys.stderr)
        return 1

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/paper/test_paper_skeleton.py", "-q"],
        cwd=_REPO,
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
