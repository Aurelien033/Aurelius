"""Tests for P0.3 EWM Erase/Write Update Contract (OC-7)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from src.memory.amc_update_contract import (  # noqa: E402
    AMCCollisionPolicy,
    AMCContradictionPolicy,
    AMCEraseWriteUpdate,
    AMCSafetyTier,
    AMCUpdateReason,
    EWMUpdateContractError,
    admit_ewm_update,
    build_ewm_update,
    clamp_gate,
    summarize_gate,
    validate_gate,
)
from src.memory.sdb_runtime import REDACTED  # noqa: E402


def _now() -> datetime:
    return datetime(2026, 5, 22, 14, 0, 0, tzinfo=UTC)


def _update(
    *,
    decay_gate: float | torch.Tensor = 0.1,
    erase_gate: float | torch.Tensor = 0.2,
    write_gate: float | torch.Tensor = 0.3,
    trust_score: float = 0.8,
    provenance: dict | None = None,
    safety_tier: AMCSafetyTier = AMCSafetyTier.INTERNAL,
    update_reason: AMCUpdateReason = AMCUpdateReason.NEW,
    collision_policy: AMCCollisionPolicy = AMCCollisionPolicy.REJECT,
    contradiction_policy: AMCContradictionPolicy = AMCContradictionPolicy.QUARANTINE,
) -> AMCEraseWriteUpdate:
    return build_ewm_update(
        update_id="upd-1",
        memory_key=torch.tensor([1.0, 2.0]),
        memory_value=torch.tensor([3.0, 4.0]),
        decay_gate=decay_gate,
        erase_gate=erase_gate,
        write_gate=write_gate,
        trust_score=trust_score,
        provenance=provenance or {"source": "test"},
        safety_tier=safety_tier,
        update_reason=update_reason,
        collision_policy=collision_policy,
        contradiction_policy=contradiction_policy,
        created_at=_now(),
    )


# ── Test 1: independent gates ────────────────────────────────────────────────


def test_update_requires_independent_gates() -> None:
    update = _update(decay_gate=0.1, erase_gate=0.5, write_gate=0.9)
    assert update.decay_gate != update.erase_gate or update.decay_gate == 0.1
    assert update.erase_gate == 0.5
    assert update.write_gate == 0.9
    telemetry = admit_ewm_update(update)
    assert telemetry.decay_mean == pytest.approx(0.1)
    assert telemetry.erase_mean == pytest.approx(0.5)
    assert telemetry.write_mean == pytest.approx(0.9)
    assert hasattr(update, "decay_gate")
    assert hasattr(update, "erase_gate")
    assert hasattr(update, "write_gate")
    combined = getattr(update, "combined_gate", None)
    assert combined is None


# ── Test 2: gate validation ──────────────────────────────────────────────────


def test_gate_validation_rejects_invalid_values() -> None:
    with pytest.raises(EWMUpdateContractError, match="decay_gate"):
        validate_gate(float("nan"), name="decay_gate")
    with pytest.raises(EWMUpdateContractError, match="erase_gate"):
        validate_gate(float("inf"), name="erase_gate")
    with pytest.raises(EWMUpdateContractError, match="write_gate"):
        validate_gate(1.2, name="write_gate")
    with pytest.raises(EWMUpdateContractError, match="write_gate"):
        validate_gate(-0.1, name="write_gate")
    assert validate_gate(0.5, name="write_gate") == 0.5
    assert validate_gate(0.0, name="decay_gate") == 0.0
    assert validate_gate(1.0, name="erase_gate") == 1.0


def test_clamp_gate_only_when_explicit() -> None:
    assert clamp_gate(1.5) == 1.0
    assert clamp_gate(-0.2) == 0.0


# ── Test 3: tensor gate summaries JSON-compatible ────────────────────────────


def test_tensor_gate_summaries_json_compatible() -> None:
    gate = torch.tensor([0.1, 0.5, 0.9])
    summary = summarize_gate(gate)
    json.dumps(summary)
    assert summary["kind"] == "tensor"
    assert summary["shape"] == [3]
    assert "mean" in summary
    assert "min" in summary
    assert "max" in summary


# ── Test 4: erase/write correlation telemetry ────────────────────────────────


def test_erase_write_correlation_same_shape() -> None:
    erase = torch.tensor([0.0, 0.5, 1.0])
    write = torch.tensor([1.0, 0.5, 0.0])
    update = _update(erase_gate=erase, write_gate=write)
    telemetry = admit_ewm_update(update)
    assert telemetry.erase_write_correlation is not None
    assert -1.0 <= telemetry.erase_write_correlation <= 1.0


def test_erase_write_correlation_different_shape_returns_none() -> None:
    update = _update(
        erase_gate=torch.tensor([0.1, 0.2]),
        write_gate=torch.tensor([0.3]),
    )
    telemetry = admit_ewm_update(update)
    assert telemetry.erase_write_correlation is None


def test_constant_tensor_correlation_is_none() -> None:
    update = _update(
        erase_gate=torch.tensor([0.5, 0.5]),
        write_gate=torch.tensor([0.5, 0.5]),
    )
    telemetry = admit_ewm_update(update)
    assert telemetry.erase_write_correlation is None


# ── Test 5: trust_score bounds ───────────────────────────────────────────────


def test_trust_score_bounds() -> None:
    with pytest.raises(EWMUpdateContractError, match="trust_score"):
        _update(trust_score=-0.01)
    with pytest.raises(EWMUpdateContractError, match="trust_score"):
        _update(trust_score=1.01)
    _update(trust_score=0.0)
    _update(trust_score=1.0)


# ── Test 6: provenance required and redacted ─────────────────────────────────


def test_provenance_required_for_durable_reasons() -> None:
    with pytest.raises(EWMUpdateContractError, match="provenance"):
        build_ewm_update(
            update_id="u",
            memory_key=torch.zeros(2),
            memory_value=torch.zeros(2),
            decay_gate=0.1,
            erase_gate=0.1,
            write_gate=0.1,
            trust_score=0.5,
            provenance={},
            safety_tier=AMCSafetyTier.INTERNAL,
            update_reason=AMCUpdateReason.NEW,
            created_at=_now(),
        )
    with pytest.raises(EWMUpdateContractError, match="provenance"):
        build_ewm_update(
            update_id="u",
            memory_key=torch.zeros(2),
            memory_value=torch.zeros(2),
            decay_gate=0.1,
            erase_gate=0.1,
            write_gate=0.1,
            trust_score=0.5,
            provenance={},
            safety_tier=AMCSafetyTier.INTERNAL,
            update_reason=AMCUpdateReason.TOOL_RESULT,
            created_at=_now(),
        )
    update = build_ewm_update(
        update_id="u",
        memory_key=torch.zeros(2),
        memory_value=torch.zeros(2),
        decay_gate=0.1,
        erase_gate=0.1,
        write_gate=0.1,
        trust_score=0.5,
        provenance={"api_key": "secret", "ref": "ok"},
        safety_tier=AMCSafetyTier.INTERNAL,
        update_reason=AMCUpdateReason.CORRECTION,
        created_at=_now(),
    )
    audit = update.to_audit_dict()
    assert REDACTED in json.dumps(audit)
    assert "secret" not in json.dumps(audit["provenance"])


# ── Test 7: unsafe safety tier rejected ──────────────────────────────────────


def test_unsafe_safety_tier_rejected() -> None:
    update = _update(safety_tier=AMCSafetyTier.UNSAFE)
    telemetry = admit_ewm_update(update)
    assert telemetry.admitted is False
    assert telemetry.reject_code == "unsafe_memory"


# ── Test 8: collision policy explicit ────────────────────────────────────────


def test_collision_policy_explicit() -> None:
    update = build_ewm_update(
        update_id="u",
        memory_key=torch.zeros(2),
        memory_value=torch.zeros(2),
        decay_gate=0.1,
        erase_gate=0.1,
        write_gate=0.1,
        trust_score=0.5,
        provenance={"source": "x"},
        safety_tier=AMCSafetyTier.INTERNAL,
        update_reason=AMCUpdateReason.MANUAL,
        created_at=_now(),
    )
    assert update.collision_policy == AMCCollisionPolicy.REJECT
    overwrite = _update(collision_policy=AMCCollisionPolicy.OVERWRITE)
    assert overwrite.collision_policy == AMCCollisionPolicy.OVERWRITE
    with pytest.raises(EWMUpdateContractError):
        build_ewm_update(
            update_id="u",
            memory_key=torch.zeros(2),
            memory_value=torch.zeros(2),
            decay_gate=0.1,
            erase_gate=0.1,
            write_gate=0.1,
            trust_score=0.5,
            provenance={"source": "x"},
            safety_tier=AMCSafetyTier.INTERNAL,
            update_reason=AMCUpdateReason.MANUAL,
            collision_policy="not-a-policy",  # type: ignore[arg-type]
            created_at=_now(),
        )


# ── Test 9: contradiction policy explicit ────────────────────────────────────


def test_contradiction_policy_explicit() -> None:
    update = _update()
    assert update.contradiction_policy == AMCContradictionPolicy.QUARANTINE
    conflict = _update(
        update_reason=AMCUpdateReason.CONFLICT,
        contradiction_policy=AMCContradictionPolicy.REJECT,
    )
    assert conflict.contradiction_policy == AMCContradictionPolicy.REJECT
    with pytest.raises(EWMUpdateContractError):
        build_ewm_update(
            update_id="u",
            memory_key=torch.zeros(2),
            memory_value=torch.zeros(2),
            decay_gate=0.1,
            erase_gate=0.1,
            write_gate=0.1,
            trust_score=0.5,
            provenance={"source": "x"},
            safety_tier=AMCSafetyTier.INTERNAL,
            update_reason=AMCUpdateReason.CONFLICT,
            contradiction_policy=None,  # type: ignore[arg-type]
            created_at=_now(),
        )


# ── Test 10: no memory write side effects ────────────────────────────────────


def test_no_memory_write_side_effects() -> None:
    import inspect

    import src.memory.amc_update_contract as contract

    update = _update()
    admit_ewm_update(update)
    assert admit_ewm_update.__name__ == "admit_ewm_update"
    source = inspect.getsource(admit_ewm_update)
    assert "tier2" not in source.lower()
    assert "tier3" not in source.lower()
    assert "write(" not in source
    public_writes = [
        name
        for name in dir(contract)
        if "write" in name.lower()
        and callable(getattr(contract, name))
        and name not in {"AMCEraseWriteUpdate", "AMCEraseWriteTelemetry", "build_ewm_update"}
    ]
    assert public_writes == []


# ── Test 11: audit output excludes raw tensors ───────────────────────────────


def test_audit_output_excludes_raw_tensors() -> None:
    update = _update()
    audit = update.to_audit_dict()
    telemetry = admit_ewm_update(update).to_dict()
    blob = json.dumps({"audit": audit, "telemetry": telemetry})
    assert "[1.0, 2.0]" not in blob
    assert '"values"' not in blob
    assert audit["memory_key"]["kind"] in {"tensor", "scalar", "opaque"}
    assert "content_hash" in audit["memory_key"]


# ── Test 12: import is cheap ─────────────────────────────────────────────────


def test_import_is_cheap() -> None:
    import importlib
    import sys

    name = "src.memory.amc_update_contract"
    sys.modules.pop(name, None)
    mod = importlib.import_module(name)
    assert mod.SCHEMA_VERSION == "ewm/v1"
    assert "admit_ewm_update" in dir(mod)
