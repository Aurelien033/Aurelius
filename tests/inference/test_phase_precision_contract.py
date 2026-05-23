"""Tests for P0.7 PPDQ phase-aware precision contract (OC-13)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.inference.phase_precision_contract import (
    ALL_SERVING_PHASES,
    REDACTED,
    FallbackPolicy,
    PhasePrecisionRequest,
    PhaseRiskTier,
    PrecisionDecision,
    PrecisionMode,
    PrecisionSupport,
    ServingPhase,
    build_phase_precision_plan,
    build_phase_telemetry_record,
    default_phase_precision_requests,
    detect_precision_capability,
    plan_hash,
    resolve_phase_precision,
    sanitize_precision_payload,
    validate_phase_precision_plan,
)


def test_module_imports_on_cpu() -> None:
    mod = importlib.import_module("src.inference.phase_precision_contract")
    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import numpy" not in source
    assert mod.SCHEMA_VERSION == "ppdq.v1"


def test_default_requests_contain_no_nvfp4() -> None:
    for req in default_phase_precision_requests():
        assert req.requested_precision != PrecisionMode.NVFP4


def test_default_plan_covers_all_phases() -> None:
    plan = build_phase_precision_plan()
    validate_phase_precision_plan(plan)
    phases = {d.phase for d in plan.decisions}
    assert phases == set(ALL_SERVING_PHASES)


def test_default_decode_is_bf16_not_nvfp4() -> None:
    plan = build_phase_precision_plan()
    decode = next(d for d in plan.decisions if d.phase == ServingPhase.DECODE)
    assert decode.effective_precision == PrecisionMode.BF16
    assert decode.effective_precision != PrecisionMode.NVFP4


def test_safety_verifier_never_nvfp4() -> None:
    req = PhasePrecisionRequest(
        phase=ServingPhase.SAFETY_VERIFIER,
        requested_precision=PrecisionMode.NVFP4,
        fallback_policy=FallbackPolicy.BF16,
        risk_tier=PhaseRiskTier.CRITICAL,
    )
    decision = resolve_phase_precision(req)
    assert decision.effective_precision != PrecisionMode.NVFP4


def test_commit_audit_never_nvfp4() -> None:
    req = PhasePrecisionRequest(
        phase=ServingPhase.COMMIT_AUDIT,
        requested_precision=PrecisionMode.NVFP4,
        fallback_policy=FallbackPolicy.FP32,
        risk_tier=PhaseRiskTier.CRITICAL,
    )
    decision = resolve_phase_precision(req)
    assert decision.effective_precision != PrecisionMode.NVFP4


def test_unsupported_nvfp4_prefill_falls_back_bf16() -> None:
    req = PhasePrecisionRequest(
        phase=ServingPhase.PROMPT_PREFILL,
        requested_precision=PrecisionMode.NVFP4,
        fallback_policy=FallbackPolicy.BF16,
        risk_tier=PhaseRiskTier.LOW,
    )
    decision = resolve_phase_precision(req, runner=lambda _cmd: None)
    assert decision.decision == PrecisionDecision.FALLBACK
    assert decision.effective_precision == PrecisionMode.BF16


def test_unsupported_nvfp4_error_policy_rejects() -> None:
    req = PhasePrecisionRequest(
        phase=ServingPhase.PROMPT_PREFILL,
        requested_precision=PrecisionMode.NVFP4,
        fallback_policy=FallbackPolicy.ERROR,
        risk_tier=PhaseRiskTier.LOW,
    )
    decision = resolve_phase_precision(req, runner=lambda _cmd: None)
    assert decision.decision == PrecisionDecision.REJECT
    assert decision.effective_precision == PrecisionMode.BF16


def test_effective_precision_never_unknown() -> None:
    plan = build_phase_precision_plan()
    for decision in plan.decisions:
        assert decision.effective_precision != PrecisionMode.UNKNOWN


def test_nvfp4_capability_does_not_crash_without_gpu() -> None:
    cap = detect_precision_capability(PrecisionMode.NVFP4, runner=lambda _cmd: None)
    assert cap.support in (PrecisionSupport.UNSUPPORTED, PrecisionSupport.UNKNOWN)


def test_nvidia_smi_probe_uses_shell_false(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []
    kwargs_seen: list[dict[str, object]] = []

    def runner(cmd: list[str]) -> SimpleNamespace:
        captured.append(cmd)
        return SimpleNamespace(returncode=1, stdout="")

    detect_precision_capability(PrecisionMode.NVFP4, runner=runner)
    assert captured[0] == ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]

    def fake_run(cmd: list[str], **kwargs: object) -> None:
        kwargs_seen.append(kwargs)
        raise FileNotFoundError

    monkeypatch.setattr(
        "src.inference.phase_precision_contract.subprocess.run",
        fake_run,
    )
    detect_precision_capability(PrecisionMode.NVFP4)
    assert kwargs_seen[0]["shell"] is False


def test_plan_validation_fails_on_missing_phase() -> None:
    partial = build_phase_precision_plan(
        requests=default_phase_precision_requests()[:2],
    )
    with pytest.raises(ValueError, match="missing phase"):
        validate_phase_precision_plan(partial)


def test_plan_hash_deterministic_for_equivalent_plans() -> None:
    requests = default_phase_precision_requests()
    plan_a = build_phase_precision_plan(requests=requests)
    plan_b = build_phase_precision_plan(requests=requests)
    assert plan_hash(plan_a) == plan_hash(plan_b)


def test_plan_hash_changes_when_effective_precision_changes() -> None:
    base = build_phase_precision_plan()
    alt_requests = list(default_phase_precision_requests())
    alt_requests[2] = PhasePrecisionRequest(
        phase=ServingPhase.DECODE,
        requested_precision=PrecisionMode.FP32,
        fallback_policy=FallbackPolicy.BF16,
        risk_tier=PhaseRiskTier.HIGH,
    )
    alt = build_phase_precision_plan(requests=alt_requests)
    assert plan_hash(alt) != plan_hash(base)


def test_telemetry_rejects_negative_latency() -> None:
    with pytest.raises(ValueError, match="latency_ms"):
        build_phase_telemetry_record(
            phase=ServingPhase.DECODE,
            effective_precision=PrecisionMode.BF16,
            latency_ms=-1.0,
        )


def test_telemetry_rejects_negative_tokens() -> None:
    with pytest.raises(ValueError, match="tokens"):
        build_phase_telemetry_record(
            phase=ServingPhase.DECODE,
            effective_precision=PrecisionMode.BF16,
            tokens=-1,
        )


def test_telemetry_allows_quality_delta_none() -> None:
    record = build_phase_telemetry_record(
        phase=ServingPhase.DECODE,
        effective_precision=PrecisionMode.BF16,
        quality_delta=None,
    )
    assert record.quality_delta is None


def test_metadata_secret_keys_redacted() -> None:
    out = sanitize_precision_payload({"api_key": "secret", "note": "ok"})
    assert out["api_key"] == REDACTED
    assert out["note"] == "ok"


def test_capability_metadata_has_no_environment_dump() -> None:
    cap = detect_precision_capability(PrecisionMode.BF16, runner=lambda _cmd: None)
    blob = json.dumps(cap.metadata)
    assert "environ" not in blob.lower()
    assert "PATH=" not in blob


def test_commit_audit_int8_fallback() -> None:
    req = PhasePrecisionRequest(
        phase=ServingPhase.COMMIT_AUDIT,
        requested_precision=PrecisionMode.INT8,
        fallback_policy=FallbackPolicy.FP32,
        risk_tier=PhaseRiskTier.CRITICAL,
    )
    decision = resolve_phase_precision(req)
    assert decision.effective_precision not in (PrecisionMode.INT8, PrecisionMode.NVFP4)
