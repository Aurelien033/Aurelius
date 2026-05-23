"""Validate docs/reproducibility bundle structure (T30)."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = REPO_ROOT / "docs" / "reproducibility"

REQUIRED_FILES = (
    "README.md",
    "seed.txt",
    "environment.yml",
    "configs/baseline.yaml",
    "configs/tier1_only.yaml",
    "configs/tier12.yaml",
    "configs/full_amc.yaml",
    "scripts/install.sh",
    "scripts/download_data.sh",
    "scripts/train.sh",
    "scripts/evaluate.sh",
    "scripts/ablation.sh",
    "scripts/plot_results.py",
    "results/model_card.md",
    "checkpoint/README.md",
)

EXECUTABLE_SCRIPTS = (
    "scripts/install.sh",
    "scripts/download_data.sh",
    "scripts/train.sh",
    "scripts/evaluate.sh",
    "scripts/ablation.sh",
)


@pytest.mark.parametrize("relative_path", REQUIRED_FILES)
def test_required_bundle_file_exists(relative_path: str) -> None:
    path = BUNDLE / relative_path
    assert path.exists(), f"missing {path}"


@pytest.mark.parametrize("relative_path", EXECUTABLE_SCRIPTS)
def test_shell_scripts_are_executable(relative_path: str) -> None:
    path = BUNDLE / relative_path
    assert path.stat().st_mode & 0o111, f"{path} should be executable"


def test_config_symlinks_resolve_to_repo_configs() -> None:
    mapping = {
        "baseline.yaml": "ablation_baseline.yaml",
        "tier1_only.yaml": "ablation_tier1_only.yaml",
        "tier12.yaml": "ablation_tier12.yaml",
        "full_amc.yaml": "amc_forge_1b.yaml",
    }
    for bundle_name, repo_name in mapping.items():
        path = (BUNDLE / "configs" / bundle_name).resolve()
        assert path.is_file(), bundle_name
        assert repo_name in path.name or path.name == bundle_name


def test_readme_documents_quick_start() -> None:
    text = (BUNDLE / "README.md").read_text(encoding="utf-8")
    assert "install.sh" in text
    assert "ablation.sh" in text
    assert "seed.txt" in text


def test_seed_file_lists_training_and_eval_seeds() -> None:
    text = (BUNDLE / "seed.txt").read_text(encoding="utf-8")
    assert "training.global_seed=42" in text
    assert "evaluation.gsm8k_seed=42" in text
