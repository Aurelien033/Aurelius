#!/usr/bin/env python3
"""Write release/dist/README_HF.md from reproducibility model card (no upload)."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
SRC = _REPO / "docs/reproducibility/results/model_card.md"
OUT = _REPO / "release/dist/README_HF.md"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    body = (
        SRC.read_text(encoding="utf-8")
        if SRC.is_file()
        else "# AMC Model Card\n\nTODO: model_card.md missing\n"
    )
    header = (
        "# Hugging Face README (generated)\n\n"
        "> TODO: set `weights_path` when checkpoints are available.\n\n"
    )
    OUT.write_text(header + body, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
