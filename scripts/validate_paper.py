#!/usr/bin/env python3
"""Validate paper skeleton (T32–T34)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
PAPER = _REPO / "paper"


def _collect_tex() -> str:
    parts = [(PAPER / "main.tex").read_text(encoding="utf-8")]
    for path in sorted(PAPER.rglob("*.tex")):
        if path.name == "main.tex":
            continue
        parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def main() -> int:
    if not (PAPER / "main.tex").is_file():
        print(f"MISSING {PAPER / 'main.tex'}", file=sys.stderr)
        return 1

    text = _collect_tex()
    required = [
        r"\begin{abstract}",
        r"\section{Introduction}",
        r"\section{Experiments}",
        "tab:ablation",
        "tab:safety",
        "TODO",
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
