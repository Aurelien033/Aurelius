"""Tests for P0.5 VEL result registry (OC-10)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.vel_result_registry import (
    DEFAULT_REGISTRY_PATH,
    REDACTED,
    VELClaimType,
    VELDecision,
    VELMetric,
    VELRegistry,
    VELRegistryError,
    VELResultRecord,
    VELRunStatus,
    VELSource,
    VELSourceType,
    build_vel_record,
    collect_vel_environment,
    compute_record_hash,
    sanitize_vel_command,
    sanitize_vel_value,
    sha256_file,
    stable_json_dumps,
)


def _env():
    return collect_vel_environment(git=False)


def _record(
    *,
    record_id: str = "rec-1",
    run_status: VELRunStatus = VELRunStatus.PASS,
    decision: VELDecision = VELDecision.ACCEPT,
    command: tuple[str, ...] = ("/venv/bin/python", "-m", "pytest", "tests/eval/"),
    pivot_reason: str | None = None,
    raw_output_path: str | None = "/tmp/out.json",
    metadata: dict | None = None,
) -> VELResultRecord:
    return VELResultRecord(
        record_id=record_id,
        created_at="2026-05-22T12:00:00+00:00",
        claim_type=VELClaimType.BENCHMARK,
        run_status=run_status,
        decision=decision,
        command=command,
        config_path=None,
        config_sha256=None,
        seed=42,
        checkpoint_sha=None,
        raw_output_path=raw_output_path,
        metrics=(VELMetric(name="overall_score", value=0.9),),
        sources=(),
        environment=_env(),
        pivot_or_refine_reason=pivot_reason,
        notes=None,
        metadata=metadata or {},
    )


# ── Test 1: VELMetric rejects NaN/inf ────────────────────────────────────────


def test_vel_metric_rejects_nan_inf() -> None:
    VELMetric(name="ok", value=1.0)
    with pytest.raises(VELRegistryError, match="finite"):
        VELMetric(name="bad", value=float("nan"))
    with pytest.raises(VELRegistryError, match="finite"):
        VELMetric(name="bad", value=float("inf"))


# ── Test 2: VELSource validation ─────────────────────────────────────────────


def test_vel_source_requires_ids() -> None:
    VELSource(
        source_id="src-1",
        source_type=VELSourceType.CONFIG,
        path_or_url="configs/foo.yaml",
    )
    with pytest.raises(VELRegistryError, match="source_id"):
        VELSource(source_id="", source_type=VELSourceType.CONFIG, path_or_url="x")
    with pytest.raises(VELRegistryError, match="path_or_url"):
        VELSource(source_id="s", source_type=VELSourceType.CONFIG, path_or_url="")


# ── Test 3: benchmark claims require command ─────────────────────────────────


def test_benchmark_claim_requires_command() -> None:
    with pytest.raises(VELRegistryError, match="command"):
        VELResultRecord(
            record_id="r",
            created_at="2026-05-22T12:00:00+00:00",
            claim_type=VELClaimType.BENCHMARK,
            run_status=VELRunStatus.PASS,
            decision=VELDecision.ACCEPT,
            command=(),
            config_path=None,
            config_sha256=None,
            seed=None,
            checkpoint_sha=None,
            raw_output_path="/tmp/out.json",
            metrics=(),
            sources=(),
            environment=_env(),
        )


# ── Test 4: FAIL/ERROR cannot be ACCEPT without override ───────────────────


def test_fail_cannot_be_accept_without_override() -> None:
    with pytest.raises(VELRegistryError, match="ACCEPT"):
        _record(run_status=VELRunStatus.FAIL, decision=VELDecision.ACCEPT)
    _record(
        run_status=VELRunStatus.FAIL,
        decision=VELDecision.ACCEPT,
        metadata={"accept_override_reason": "manual review"},
    )


# ── Test 5: REFINE/PIVOT/REJECT require reason ───────────────────────────────


def test_pivot_refine_requires_reason() -> None:
    with pytest.raises(VELRegistryError, match="pivot_or_refine_reason"):
        _record(decision=VELDecision.PIVOT, pivot_reason=None)
    _record(decision=VELDecision.REFINE, pivot_reason="retry with seed 7")


# ── Test 6–7: sanitization ───────────────────────────────────────────────────


def test_secret_command_args_redacted() -> None:
    cmd = sanitize_vel_command(
        (
            "run.py",
            "--api-key=sk-live",
            "--token",
            "abc",
            "Authorization: Bearer xyz",
            "SAFE=ok",
        )
    )
    blob = " ".join(cmd)
    assert REDACTED in blob
    assert "sk-live" not in blob
    assert "Bearer xyz" not in blob
    assert "SAFE=ok" in blob


def test_secret_metadata_redacted_recursively() -> None:
    data = sanitize_vel_value({"api_key": "x", "nested": {"password": "y", "note": "z"}})
    assert data["api_key"] == REDACTED
    assert data["nested"]["password"] == REDACTED
    assert data["nested"]["note"] == "z"


# ── Test 8: sha256_file ──────────────────────────────────────────────────────


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("seed: 42\n", encoding="utf-8")
    digest = sha256_file(path)
    assert len(digest) == 64
    assert digest == sha256_file(path)


# ── Tests 9–12: registry JSONL ───────────────────────────────────────────────


def test_registry_append_and_read(tmp_path: Path) -> None:
    registry_path = tmp_path / "vel.jsonl"
    registry = VELRegistry(registry_path)
    record = _record(record_id="append-1")
    registry.append(record)
    lines = registry_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    loaded = registry.read_all()
    assert len(loaded) == 1
    assert loaded[0].record_id == "append-1"


def test_registry_ignores_blank_lines(tmp_path: Path) -> None:
    registry_path = tmp_path / "vel.jsonl"
    registry_path.write_text("\n\n", encoding="utf-8")
    registry = VELRegistry(registry_path)
    registry.append(_record(record_id="only"))
    loaded = registry.read_all()
    assert len(loaded) == 1


def test_registry_raises_on_malformed_json(tmp_path: Path) -> None:
    registry_path = tmp_path / "vel.jsonl"
    registry_path.write_text("{not json\n", encoding="utf-8")
    registry = VELRegistry(registry_path)
    with pytest.raises(VELRegistryError, match="malformed"):
        registry.read_all()


# ── Test 13–14: stable serialization / record hash ───────────────────────────


def test_stable_json_dumps_deterministic() -> None:
    payload = {"b": 1, "a": 2, "status": "pass"}
    assert stable_json_dumps(payload) == stable_json_dumps({"a": 2, "b": 1, "status": "pass"})


def test_record_hash_changes_with_metric() -> None:
    r1 = _record()
    r2 = _record()
    r2_dict = r2.to_dict()
    r2_dict["metrics"] = [{"name": "overall_score", "value": 0.1}]
    r2b = VELResultRecord.from_dict(r2_dict)
    assert compute_record_hash(r1) != compute_record_hash(r2b)


# ── Test 15: environment safe ──────────────────────────────────────────────────


def test_environment_collection_excludes_env_vars() -> None:
    env = collect_vel_environment(git=False)
    blob = json.dumps(env.to_dict())
    assert "environ" not in blob.lower()
    assert "API_KEY" not in blob
    assert env.python_version


# ── Test 16–17: no write on import ───────────────────────────────────────────


def test_import_does_not_create_default_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib
    import sys

    target = tmp_path / "docs" / "reports" / "vel_results.jsonl"
    monkeypatch.chdir(tmp_path)
    name = "src.eval.vel_result_registry"
    sys.modules.pop(name, None)
    importlib.import_module(name)
    assert not target.exists()


def test_append_creates_parent_directory(tmp_path: Path) -> None:
    registry_path = tmp_path / "nested" / "reports" / "vel.jsonl"
    VELRegistry(registry_path).append(_record(record_id="mkdir"))
    assert registry_path.exists()


# ── Test 18: config hash when path exists ────────────────────────────────────


def test_build_vel_record_computes_config_hash(tmp_path: Path) -> None:
    cfg = tmp_path / "bench.yaml"
    cfg.write_text("suite: amc\n", encoding="utf-8")
    record = build_vel_record(
        claim_type=VELClaimType.BENCHMARK,
        run_status=VELRunStatus.PASS,
        decision=VELDecision.ACCEPT,
        command=("python", "-m", "pytest"),
        config_path=str(cfg),
        raw_output_path="/tmp/out.json",
        metrics=[{"name": "score", "value": 1.0}],
    )
    assert record.config_sha256 == sha256_file(cfg)


def test_benchmark_requires_raw_output_or_reason() -> None:
    with pytest.raises(VELRegistryError, match="raw_output"):
        _record(raw_output_path=None)
    _record(
        raw_output_path=None,
        metadata={"raw_output_missing_reason": "dry-run"},
    )


def test_default_registry_path_constant() -> None:
    assert DEFAULT_REGISTRY_PATH.as_posix().endswith("docs/reports/vel_results.jsonl")
