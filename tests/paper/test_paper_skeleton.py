"""Validate AMC paper LaTeX skeleton (T32–T34)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER = REPO_ROOT / "paper"
MAIN_TEX = PAPER / "main.tex"
CLAIMS = PAPER / "CLAIMS_LEDGER.md"

REQUIRED_INPUTS = (
    "sections/01_introduction.tex",
    "sections/02_related_work.tex",
    "sections/03_method.tex",
    "sections/05_experiments.tex",
    "sections/07_conclusion.tex",
)

REQUIRED_LABELS = (
    "sec:intro",
    "sec:method",
    "sec:experiments",
    "tab:ablation",
    "tab:safety",
)


def _paper_text() -> str:
    parts = [MAIN_TEX.read_text(encoding="utf-8")]
    for rel in REQUIRED_INPUTS:
        parts.append((PAPER / rel).read_text(encoding="utf-8"))
    return "\n".join(parts)


def test_main_tex_exists() -> None:
    assert MAIN_TEX.is_file()
    assert CLAIMS.is_file()


def test_title_and_abstract_todos() -> None:
    text = MAIN_TEX.read_text(encoding="utf-8")
    assert "Aurelian Memory Core" in text
    assert r"\begin{abstract}" in text
    assert "TODO" in text


def test_claims_ledger_covers_c1_c10() -> None:
    body = CLAIMS.read_text(encoding="utf-8")
    for cid in [f"C{i}" for i in range(1, 11)]:
        assert cid in body


@pytest.mark.parametrize("rel", REQUIRED_INPUTS)
def test_section_files_exist(rel: str) -> None:
    assert (PAPER / rel).is_file()


@pytest.mark.parametrize("label", REQUIRED_LABELS)
def test_required_labels_present(label: str) -> None:
    assert label in _paper_text()


def test_method_has_correct_st_not_wrong_formula() -> None:
    method = (PAPER / "sections/03_method.tex").read_text(encoding="utf-8")
    assert "hard.detach" not in method
    assert "operatorname{sg}" in method


def test_experiments_no_fake_xx() -> None:
    exp = (PAPER / "sections/05_experiments.tex").read_text(encoding="utf-8")
    assert "X.XX" not in exp


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
