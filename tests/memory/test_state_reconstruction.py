"""Tests for SDB event-log state reconstruction (T12)."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

from plugins.memory.episodic_memory import EpisodicMemory
from src.memory.amc_tier3 import AMCTier3Config, AMCTier3Hook, TrustLevel
from src.memory.sdb_persistent_log import SDBPersistentLog
from src.memory.sdb_runtime import (
    MemoryOperation,
    MemorySourceType,
    MemoryTargetTier,
    ReplayEvent,
    SDBMemoryRuntime,
    VerificationDecision,
)
from src.memory.state_reconstruction import StateReconstructor


def _tmp_log() -> tuple[tempfile.TemporaryDirectory[str], SDBPersistentLog]:
    tmp = tempfile.TemporaryDirectory()
    log = SDBPersistentLog(Path(tmp.name) / "events.db")
    return tmp, log


def _make_event(
    runtime: SDBMemoryRuntime,
    *,
    tier: MemoryTargetTier,
    op: MemoryOperation,
    payload: dict | None = None,
    step: int = 1,
    proposal_id: str | None = None,
) -> object:
    proposal = runtime.propose(
        proposal_id=proposal_id,
        session_id="s1",
        step=step,
        proposer="test",
        source_type=MemorySourceType.USER,
        target_tier=tier,
        operation=op,
        payload=payload or {"note": f"test-{step}"},
    )
    verification = runtime.verify(
        proposal,
        verifier="test",
        decision=VerificationDecision.ACCEPT,
        reason="ok",
    )
    runtime.commit(proposal, verification_result=verification)
    return proposal


def test_reconstruct_empty() -> None:
    tmp, log = _tmp_log()
    with tmp:
        recon = StateReconstructor(log)
        state = recon.reconstruct_at()
        assert state.tier2_count == 0
        assert state.tier3_store_count == 0
        log.close()


def test_reconstruct_after_propose_verify_commit() -> None:
    tmp, log = _tmp_log()
    with tmp:
        runtime = SDBMemoryRuntime(persistent_log=log)
        _make_event(
            runtime,
            tier=MemoryTargetTier.TIER2,
            op=MemoryOperation.STORE,
            payload={"content": "hello", "importance": 0.9, "role": "user"},
        )
        state = StateReconstructor(log).reconstruct_at()
        assert state.tier2_count == 1
        assert state.tier2.retrieve_recent(1)[0].content == "hello"
        log.close()


def test_reconstruct_after_reject() -> None:
    tmp, log = _tmp_log()
    with tmp:
        runtime = SDBMemoryRuntime(persistent_log=log)
        proposal = runtime.propose(
            session_id="s1",
            step=1,
            proposer="test",
            source_type=MemorySourceType.USER,
            target_tier=MemoryTargetTier.TIER2,
            operation=MemoryOperation.STORE,
            payload={"content": "blocked", "role": "user"},
        )
        verification = runtime.verify(
            proposal,
            verifier="test",
            decision=VerificationDecision.REJECT,
            reason="no",
        )
        runtime.reject(proposal, verification_result=verification)
        state = StateReconstructor(log).reconstruct_at()
        assert state.tier2_count == 0
        log.close()


def test_reconstruct_after_quarantine() -> None:
    tmp, log = _tmp_log()
    with tmp:
        runtime = SDBMemoryRuntime(persistent_log=log)
        _make_event(
            runtime,
            tier=MemoryTargetTier.TIER3,
            op=MemoryOperation.QUARANTINE,
            payload={"key": "ext.fact", "value": "dubious", "confidence": 0.1},
        )
        state = StateReconstructor(log).reconstruct_at()
        assert state.tier3_quarantine_count == 1
        assert "ext.fact" in state.tier3._quarantine
        log.close()


def test_reconstruct_after_promotion() -> None:
    tmp, log = _tmp_log()
    with tmp:
        runtime = SDBMemoryRuntime(persistent_log=log)
        _make_event(
            runtime,
            tier=MemoryTargetTier.TIER3,
            op=MemoryOperation.PROMOTE,
            payload={
                "key": "pref.theme",
                "value": "dark",
                "confidence": 0.95,
                "trust_level": str(TrustLevel.TRUSTED),
            },
        )
        state = StateReconstructor(log).reconstruct_at()
        assert state.tier3_store_count == 1
        assert state.tier3._store["pref.theme"].trust_level is TrustLevel.TRUSTED
        log.close()


def test_reconstruct_after_revoke() -> None:
    tmp, log = _tmp_log()
    with tmp:
        runtime = SDBMemoryRuntime(persistent_log=log)
        _make_event(
            runtime,
            tier=MemoryTargetTier.TIER3,
            op=MemoryOperation.PROMOTE,
            payload={"key": "k1", "value": "v", "confidence": 0.9},
        )
        log.append(
            ReplayEvent(
                event_id="revoke-evt",
                event_type="committed",
                proposal_id="revoke-prop",
                timestamp=datetime.now(UTC),
                metadata={
                    "target_tier": "tier3",
                    "operation": "revoke",
                    "payload": {"key": "k1"},
                },
                replay_hash="revoke-hash",
            )
        )
        state = StateReconstructor(log).reconstruct_at()
        assert state.tier3._store["k1"].trust_level is TrustLevel.REVOKED
        log.close()


def test_reconstruct_deterministic() -> None:
    tmp, log = _tmp_log()
    with tmp:
        runtime = SDBMemoryRuntime(persistent_log=log)
        for step in range(3):
            _make_event(
                runtime,
                tier=MemoryTargetTier.TIER2,
                op=MemoryOperation.STORE,
                step=step,
                payload={"content": f"m{step}", "importance": 0.9, "role": "user"},
            )
        recon = StateReconstructor(log)
        first = recon.reconstruct_at()
        second = recon.reconstruct_at()
        assert first.tier2_count == second.tier2_count
        assert first.tier3_store_count == second.tier3_store_count
        log.close()


def test_reconstruct_at_subset_sequence() -> None:
    tmp, log = _tmp_log()
    with tmp:
        runtime = SDBMemoryRuntime(persistent_log=log)
        for step in range(5):
            _make_event(
                runtime,
                tier=MemoryTargetTier.TIER2,
                op=MemoryOperation.STORE,
                step=step,
                payload={"content": f"m{step}", "importance": 0.9, "role": "user"},
            )
        recon = StateReconstructor(log)
        early = recon.reconstruct_at(target_seq=3)
        full = recon.reconstruct_at(target_seq=None)
        assert early.events_replayed < full.events_replayed
        assert early.tier2_count <= full.tier2_count
        log.close()


def test_diff_shows_added_removed() -> None:
    tmp, log = _tmp_log()
    with tmp:
        runtime = SDBMemoryRuntime(persistent_log=log)
        _make_event(
            runtime,
            tier=MemoryTargetTier.TIER2,
            op=MemoryOperation.STORE,
            step=1,
            payload={"content": "a", "importance": 0.9, "role": "user"},
        )
        _make_event(
            runtime,
            tier=MemoryTargetTier.TIER2,
            op=MemoryOperation.STORE,
            step=2,
            payload={"content": "b", "importance": 0.9, "role": "user"},
        )
        diff = StateReconstructor(log).diff(seq_a=3, seq_b=None)
        assert diff.events_between >= 1
        log.close()


def test_verify_against_live_matches() -> None:
    tmp, log = _tmp_log()
    with tmp:
        runtime = SDBMemoryRuntime(persistent_log=log)
        for step in range(3):
            _make_event(
                runtime,
                tier=MemoryTargetTier.TIER2,
                op=MemoryOperation.STORE,
                step=step,
                payload={"content": f"m{step}", "importance": 0.9, "role": "user"},
            )
        _make_event(
            runtime,
            tier=MemoryTargetTier.TIER3,
            op=MemoryOperation.PROMOTE,
            payload={"key": "k", "value": "v", "confidence": 0.9},
            step=10,
        )

        live_t2 = EpisodicMemory()
        for step in range(3):
            live_t2.store(role="user", content=f"m{step}", importance=0.9)
        live_t3 = AMCTier3Hook(AMCTier3Config())
        live_t3.promote(key="k", value="v", confidence=0.9, trust_level=TrustLevel.TRUSTED)

        result = StateReconstructor(log).verify_against_live(live_t2, live_t3)
        assert result.matches is True
        log.close()
