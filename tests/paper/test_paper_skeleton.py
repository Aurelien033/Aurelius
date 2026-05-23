"""Validate AMC paper LaTeX skeleton (T32)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER = REPO_ROOT / "paper"
MAIN_TEX = PAPER / "main.tex"

REQUIRED_SECTIONS = (
    r"\section{Introduction}",
    r"\section{Related Work}",
    r"\section{Architecture}",
    r"\section{Training}",
    r"\section{Experiments}",
    r"\section{Analysis}",
    r"\section{Conclusion}",
)

REQUIRED_LABELS = (
    "sec:intro",
    "sec:architecture",
    "sec:experiments",
    "tab:ablation",
    "tab:safety",
)


def test_main_tex_exists() -> None:
    assert MAIN_TEX.is_file()


def test_title_and_abstract() -> None:
    text = MAIN_TEX.read_text(encoding="utf-8")
    assert "Aurelian Memory Core" in text
    assert r"\begin{abstract}" in text
    assert "trust-aware memory contract" in text
    assert "constitutional memory alignment" in text


@pytest.mark.parametrize("section", REQUIRED_SECTIONS)
def test_required_sections_present(section: str) -> None:
    text = MAIN_TEX.read_text(encoding="utf-8")
    assert section in text


@pytest.mark.parametrize("label", REQUIRED_LABELS)
def test_required_labels_present(label: str) -> None:
    text = MAIN_TEX.read_text(encoding="utf-8")
    assert label in text


def test_makefile_exists() -> None:
    assert (PAPER / "Makefile").is_file()


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_latex_compiles() -> None:
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        cwd=PAPER,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout[-800:] + proc.stderr[-800:]
    assert (PAPER / "main.pdf").is_file()
