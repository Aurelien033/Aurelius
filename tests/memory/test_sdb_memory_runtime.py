"""Tests for SDB-Memory Runtime Contract (P0.1 / OC-9)."""

from __future__ import annotations

import pytest

from src.memory.sdb_runtime import (
    MemoryOperation,
    MemorySourceType,
    MemoryTargetTier,
    ProposalRejectedError,
    SDBMemoryRuntime,
    VerificationDecision,
    VerificationMismatchError,
    VerificationRequiredError,
    VerificationResult,
    sanitize_memory_payload,
    stable_hash,
    stable_json_dumps,
)


def _runtime() -> SDBMemoryRuntime:
    return SDBMemoryRuntime()


def _propose(
    runtime: SDBMemoryRuntime,
    *,
    payload: dict | None = None,
    evidence: list | None = None,
    proposal_id: str = "prop-1",
) -> object:
    return runtime.propose(
        proposal_id=proposal_id,
        session_id="sess-1",
        step=1,
        proposer="test",
        source_type=MemorySourceType.TOOL,
        target_tier=MemoryTargetTier.TIER2,
        operation=MemoryOperation.STORE,
        payload=payload or {"note": "hello"},
        evidence=evidence or [],
    )


# ── Test 1: proposal redacts secrets ─────────────────────────────────────────


def test_proposal_redacts_secrets() -> None:
    raw = {
        "api_key": "sk-secret",
        "token": "tok-secret",
        "nested": {"password": "pw-secret", "label": "visible"},
        "note": "keep-me",
    }
    sanitized = sanitize_memory_payload(raw)
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["token"] == "[REDACTED]"
    assert sanitized["nested"]["password"] == "[REDACTED]"
    assert sanitized["nested"]["label"] == "visible"
    assert sanitized["note"] == "keep-me"

    runtime = _runtime()
    proposal = runtime.propose(
        proposal_id="p-redact",
        session_id="s1",
        step=0,
        proposer="agent",
        source_type=MemorySourceType.TOOL,
        target_tier=MemoryTargetTier.TIER2,
        operation=MemoryOperation.STORE,
        payload=raw,
        evidence=[{"authorization": "Bearer xyz", "ref": "ok"}],
    )
    assert proposal.payload["api_key"] == "[REDACTED]"
    assert proposal.evidence[0]["authorization"] == "[REDACTED]"
    assert proposal.evidence[0]["ref"] == "ok"


# ── Test 2: commit requires accepted verification ────────────────────────────


def test_commit_requires_verification() -> None:
    runtime = _runtime()
    proposal = _propose(runtime)
    with pytest.raises(VerificationRequiredError):
        runtime.commit(proposal, verification_result=None)
    assert runtime.replay_events()[-1].event_type == "proposed"


def test_commit_rejects_forged_verification_without_runtime_verify() -> None:
    runtime = _runtime()
    proposal = _propose(runtime)
    forged = VerificationResult(
        proposal_id=proposal.proposal_id,
        verifier="attacker",
        decision=VerificationDecision.ACCEPT,
        reason="forged",
        deterministic=True,
        checks={},
        created_at=proposal.created_at,
    )
    with pytest.raises(VerificationRequiredError):
        runtime.commit(proposal, verification_result=forged)


def test_commit_rejects_payload_mutation_after_verify() -> None:
    runtime = _runtime()
    proposal = _propose(runtime, payload={"note": "original"})
    verification = runtime.verify(
        proposal,
        verifier="policy",
        decision=VerificationDecision.ACCEPT,
        reason="ok",
    )
    proposal.payload["note"] = "mutated"
    with pytest.raises(VerificationMismatchError):
        runtime.commit(proposal, verification_result=verification)


# ── Test 3: rejected proposal cannot commit ──────────────────────────────────


def test_rejected_proposal_cannot_commit() -> None:
    runtime = _runtime()
    proposal = _propose(runtime)
    verification = runtime.verify(
        proposal,
        verifier="policy",
        decision=VerificationDecision.REJECT,
        reason="unsafe",
        checks={"policy_ok": False},
    )
    with pytest.raises(ProposalRejectedError):
        runtime.commit(proposal, verification_result=verification)
    assert not any(e.event_type == "committed" for e in runtime.replay_events())


# ── Test 4: quarantined proposal cannot commit ───────────────────────────────


def test_quarantined_proposal_cannot_commit() -> None:
    runtime = _runtime()
    proposal = _propose(runtime)
    verification = runtime.verify(
        proposal,
        verifier="policy",
        decision=VerificationDecision.QUARANTINE,
        reason="needs review",
        checks={"quarantine": True},
    )
    with pytest.raises(ProposalRejectedError):
        runtime.commit(proposal, verification_result=verification)
    assert not any(e.event_type == "committed" for e in runtime.replay_events())


# ── Test 5: accepted proposal can commit ─────────────────────────────────────


def test_accepted_proposal_can_commit() -> None:
    runtime = _runtime()
    proposal = _propose(runtime, proposal_id="p-accept")
    verification = runtime.verify(
        proposal,
        verifier="admission",
        decision=VerificationDecision.ACCEPT,
        reason="ok",
        checks={"schema_ok": True},
        deterministic=True,
    )
    commit = runtime.commit(proposal, verification_result=verification)
    assert commit.proposal_id == "p-accept"
    assert commit.replay_hash
    types = [e.event_type for e in runtime.replay_events()]
    assert types.index("proposed") < types.index("verified")
    assert types.index("verified") < types.index("committed")


# ── Test 6: verification must match proposal ───────────────────────────────


def test_verification_must_match_proposal() -> None:
    runtime = _runtime()
    proposal_a = _propose(runtime, proposal_id="a")
    proposal_b = _propose(runtime, proposal_id="b")
    verification_a = runtime.verify(
        proposal_a,
        verifier="v",
        decision=VerificationDecision.ACCEPT,
        reason="ok",
        checks={},
        deterministic=True,
    )
    with pytest.raises(VerificationMismatchError):
        runtime.commit(proposal_b, verification_result=verification_a)


def test_wrong_proposal_verification_is_mismatch_even_when_target_unverified() -> None:
    """Mismatch on proposal_id is raised before 'must verify through runtime'."""
    runtime = _runtime()
    proposal_a = _propose(runtime, proposal_id="verified-only")
    proposal_b = _propose(runtime, proposal_id="never-verified")
    verification_a = runtime.verify(
        proposal_a,
        verifier="v",
        decision=VerificationDecision.ACCEPT,
        reason="ok",
        checks={},
        deterministic=True,
    )
    assert "never-verified" not in runtime._verified_proposal_ids  # noqa: SLF001
    with pytest.raises(VerificationMismatchError) as exc_info:
        runtime.commit(proposal_b, verification_result=verification_a)
    assert "never-verified" in str(exc_info.value)
    assert "verified-only" in str(exc_info.value)


# ── Test 7: replay hashes are stable ─────────────────────────────────────────


def test_replay_hashes_are_stable() -> None:
    canonical = {
        "proposal_id": "p-stable",
        "operation": "store",
        "target_tier": "tier2",
        "source_type": "tool",
    }
    h1 = stable_hash(canonical)
    h2 = stable_hash(dict(canonical))
    assert h1 == h2
    assert stable_json_dumps({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_equivalent_flows_produce_stable_commit_replay_hash() -> None:
    def _flow() -> str:
        runtime = SDBMemoryRuntime()
        proposal = runtime.propose(
            proposal_id="p-eq",
            session_id="s",
            step=1,
            proposer="p",
            source_type=MemorySourceType.SYSTEM,
            target_tier=MemoryTargetTier.TIER1,
            operation=MemoryOperation.STORE,
            payload={"k": "v"},
            evidence=[],
        )
        verification = runtime.verify(
            proposal,
            verifier="v",
            decision=VerificationDecision.ACCEPT,
            reason="ok",
            checks={"ok": True},
            deterministic=True,
        )
        commit = runtime.commit(proposal, verification_result=verification)
        return commit.replay_hash

    assert _flow() == _flow()


# ── Test 8: event ordering ───────────────────────────────────────────────────


def test_event_ordering_propose_verify_commit() -> None:
    runtime = _runtime()
    proposal = _propose(runtime, proposal_id="p-order")
    verification = runtime.verify(
        proposal,
        verifier="v",
        decision=VerificationDecision.ACCEPT,
        reason="ok",
        checks={},
        deterministic=True,
    )
    runtime.commit(proposal, verification_result=verification)
    events = runtime.replay_events()
    assert [e.event_type for e in events] == ["proposed", "verified", "committed"]


def test_event_ordering_propose_verify_reject() -> None:
    runtime = _runtime()
    proposal = _propose(runtime, proposal_id="p-rej")
    verification = runtime.verify(
        proposal,
        verifier="v",
        decision=VerificationDecision.REJECT,
        reason="no",
        checks={},
    )
    runtime.reject(proposal, verification_result=verification)
    events = runtime.replay_events()
    assert [e.event_type for e in events] == ["proposed", "verified", "rejected"]


# ── Optional: deterministic_required rejects non-deterministic verification ──


def test_deterministic_required_rejects_non_deterministic_verification() -> None:
    runtime = _runtime()
    proposal = _propose(runtime)
    verification = runtime.verify(
        proposal,
        verifier="v",
        decision=VerificationDecision.ACCEPT,
        reason="ok",
        checks={},
        deterministic=False,
    )
    with pytest.raises(ProposalRejectedError):
        runtime.commit(
            proposal,
            verification_result=verification,
            deterministic_required=True,
        )
