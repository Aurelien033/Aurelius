"""Ring 1 reproducibility pack builder — manifest + tarball layout."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.eval.ring1_trace_logger import compute_checkpoint_sha256, compute_config_hash, get_git_sha


@dataclass
class ReproManifest:
    pack_id: str
    created_at: str
    git_sha: str
    config_hash: str
    checkpoint_sha256: str
    seeds: dict[str, Any]
    artifact_paths: dict[str, str] = field(default_factory=dict)
    code_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_critical_modules(root: Path) -> dict[str, str]:
    modules = [
        "src/eval/ring1_trace_logger.py",
        "src/eval/ring1_agent.py",
        "src/eval/ring1_amc_glue.py",
        "src/eval/ring1_dreambank_runner.py",
        "src/eval/ring1_eval_harness.py",
        "src/eval/ring1_repro_pack.py",
        "scripts/ring1_trace_collector.py",
    ]
    hashes: dict[str, str] = {}
    for relative in modules:
        path = root / relative
        if path.exists():
            hashes[relative] = sha256_file(path)
    return hashes


def build_manifest(
    *,
    config: dict[str, Any],
    artifact_paths: dict[str, str],
    repo_root: Path,
) -> ReproManifest:
    now = datetime.now(UTC)
    git_sha = get_git_sha()
    short_sha = git_sha[:8] if git_sha != "unknown" else "unknown"
    pack_id = f"ring1-repro-{now.strftime('%Y%m%d')}-{short_sha}"
    checkpoint = config.get("model", {}).get("checkpoint_path")
    return ReproManifest(
        pack_id=pack_id,
        created_at=now.isoformat(),
        git_sha=git_sha,
        config_hash=compute_config_hash(config),
        checkpoint_sha256=compute_checkpoint_sha256(checkpoint),
        seeds=config.get("seeds", {}),
        artifact_paths=artifact_paths,
        code_hashes=hash_critical_modules(repo_root),
    )


def write_repro_pack(
    *,
    config: dict[str, Any],
    output_dir: Path,
    traces_path: Path,
    dreambank_dir: Path | None,
    metrics_dir: Path | None,
    repo_root: Path,
) -> Path:
    """Assemble reproducibility pack directory per Ring 1 Repro Spec v0.1 (smoke subset)."""
    manifest = build_manifest(
        config=config,
        artifact_paths={
            "traces": str(traces_path),
            "dreambank": str(dreambank_dir) if dreambank_dir else "",
            "metrics": str(metrics_dir) if metrics_dir else "",
        },
        repo_root=repo_root,
    )
    pack_root = output_dir / manifest.pack_id
    if pack_root.exists():
        shutil.rmtree(pack_root)

    config_dir = pack_root / "config"
    seeds_dir = pack_root / "seeds"
    traces_dir = pack_root / "traces" / "test_split"
    metrics_out = pack_root / "metrics"
    dreambank_out = pack_root / "dreambank_run"
    verification_dir = pack_root / "verification"
    for directory in (
        config_dir,
        seeds_dir,
        traces_dir,
        metrics_out,
        dreambank_out,
        verification_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    (config_dir / "experiment_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    if Path("configs/ring1_tranche2.yaml").exists():
        shutil.copy("configs/ring1_tranche2.yaml", config_dir / "inference_config.yaml")
    if Path("configs/ring1_tranche3.yaml").exists():
        shutil.copy("configs/ring1_tranche3.yaml", config_dir / "dreambank_config.yaml")

    (seeds_dir / "master_seeds.json").write_text(
        json.dumps(config.get("seeds", {}), indent=2),
        encoding="utf-8",
    )

    if traces_path.exists():
        shutil.copy(traces_path, traces_dir / "traces.jsonl")
        sidecars = traces_path.parent / "sidecars"
        if sidecars.exists():
            shutil.copytree(sidecars, traces_dir / "sidecars", dirs_exist_ok=True)

    if metrics_dir and metrics_dir.exists():
        for artifact in metrics_dir.glob("*.json"):
            shutil.copy(artifact, metrics_out / artifact.name)

    if dreambank_dir and dreambank_dir.exists():
        for artifact in dreambank_dir.iterdir():
            if artifact.is_file():
                shutil.copy(artifact, dreambank_out / artifact.name)

    (pack_root / "code_hashes.txt").write_text(
        "\n".join(f"{path} {digest}" for path, digest in manifest.code_hashes.items()) + "\n",
        encoding="utf-8",
    )
    (pack_root / "README.md").write_text(
        _readme_text(manifest.pack_id),
        encoding="utf-8",
    )
    (pack_root / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    verify_src = repo_root / "verification" / "verify_ring1_tranche3.sh"
    if verify_src.exists():
        shutil.copy(verify_src, verification_dir / "verify.sh")
        (verification_dir / "verify.sh").chmod(0o755)

    tarball = output_dir / f"{manifest.pack_id}.tar.gz"
    with tarfile.open(tarball, "w:gz") as archive:
        archive.add(pack_root, arcname=pack_root.name)
    return tarball, pack_root


def validate_pack(pack_dir: Path) -> list[str]:
    """Validate minimum repro pack contents."""
    errors: list[str] = []
    required = [
        "README.md",
        "manifest.json",
        "config/experiment_config.yaml",
        "seeds/master_seeds.json",
        "verification/verify.sh",
    ]
    for relative in required:
        if not (pack_dir / relative).exists():
            errors.append(f"missing {relative}")
    dreambank = pack_dir / "dreambank_run"
    if not (dreambank / "pre_cycle_model_hash.txt").exists():
        errors.append("missing dreambank_run/pre_cycle_model_hash.txt")
    if not (dreambank / "post_cycle_model_hash.txt").exists():
        errors.append("missing dreambank_run/post_cycle_model_hash.txt")
    metrics = pack_dir / "metrics"
    if not metrics.exists() or not any(metrics.glob("*.json")):
        errors.append("missing metrics/*.json")
    return errors


def _readme_text(pack_id: str) -> str:
    return f"""# Ring 1 Reproducibility Pack — {pack_id}

## One-command replay

```bash
./verification/verify.sh
```

## Contents

- `config/` — experiment, inference, and DreamBank configs
- `traces/test_split/` — held-out trace JSONL + memory sidecars
- `dreambank_run/` — sleep cycle logs + pre/post param hashes
- `metrics/` — aggregated metrics and bootstrap CI
- `manifest.json` — config hash, git SHA, artifact paths

Regenerate traces if missing:

```bash
python scripts/ring1_trace_collector.py --config configs/ring1_tranche2.yaml --num_traces 100
python scripts/ring1_tranche3_workflow.py --config configs/ring1_tranche3.yaml
```
"""
