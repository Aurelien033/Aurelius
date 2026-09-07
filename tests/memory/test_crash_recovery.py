"""End-to-end crash recovery tests for SDB WAL + checkpoint replay (T14)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.memory.amc_checkpoint import save_amc_checkpoint
from src.memory.sdb_persistent_log import SDBPersistentLog
from src.memory.sdb_runtime import (
    MemoryOperation,
    MemorySourceType,
    MemoryTargetTier,
    SDBMemoryRuntime,
    VerificationDecision,
)
from src.memory.state_reconstruction import StateReconstructor


def _commit_cycle(
    runtime: SDBMemoryRuntime,
    *,
    step: int,
    content: str,
) -> None:
    proposal = runtime.propose(
        session_id="s1",
        step=step,
        proposer="test",
        source_type=MemorySourceType.USER,
        target_tier=MemoryTargetTier.TIER2,
        operation=MemoryOperation.STORE,
        payload={"content": content, "importance": 0.9, "role": "user"},
    )
    verification = runtime.verify(
        proposal,
        verifier="checker",
        decision=VerificationDecision.ACCEPT,
        reason="ok",
    )
    runtime.commit(proposal, verification_result=verification)


def test_crash_recovery_checkpoint_plus_wal_replay() -> None:
    """Checkpoint + WAL tail replay matches full pre-crash reconstruction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "events.db"
        ckpt_path = Path(tmpdir) / "tier.ckpt"

        log = SDBPersistentLog(db_path)
        runtime = SDBMemoryRuntime(persistent_log=log)
        recon = StateReconstructor(log)

        for step in range(100):
            _commit_cycle(runtime, step=step, content=f"mem-{step}")

        snapshot = recon.reconstruct_at()
        save_amc_checkpoint(snapshot.tier2, snapshot.tier3, log, path=ckpt_path)
        wal_seq = log.save_checkpoint({"amc_checkpoint_path": str(ckpt_path)})

        for step in range(100, 150):
            _commit_cycle(runtime, step=step, content=f"mem-{step}")

        pre_crash = recon.reconstruct_at()
        assert pre_crash.tier2_count == 150
        assert wal_seq > 0
        log.close()

        log2 = SDBPersistentLog(db_path)
        assert log2.verify_chain_integrity() is True
        runtime2 = SDBMemoryRuntime(persistent_log=log2)
        recovered = runtime2.recover_from_crash(checkpoint_path=ckpt_path)

        match = StateReconstructor(log2).verify_against_live(
            pre_crash.tier2,
            pre_crash.tier3,
        )
        assert match.matches is True
        assert recovered.tier2_count == pre_crash.tier2_count
        assert recovered.events_replayed >= 150  # tail events after checkpoint
        log2.close()


def test_crash_recovery_from_wal() -> None:
    """Simulate crash recovery: close + reopen log, verify state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        log = SDBPersistentLog(db_path)
        runtime = SDBMemoryRuntime(persistent_log=log)
        for step in range(20):
            _commit_cycle(runtime, step=step, content=f"mem-{step}")
        assert log.event_count() >= 60
        log.close()

        log2 = SDBPersistentLog(db_path)
        assert log2.event_count() >= 60
        assert log2.verify_chain_integrity() is True

        runtime2 = SDBMemoryRuntime(persistent_log=log2)
        for step in range(20, 30):
            _commit_cycle(runtime2, step=step, content=f"mem-{step}")

        state = StateReconstructor(log2).reconstruct_at()
        assert state.events_replayed >= 90
        assert state.tier2_count == 30
        log2.close()


def test_wal_mode_survives_abrupt_close() -> None:
    """WAL mode survives even if the process does not close cleanly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        log = SDBPersistentLog(db_path)
        log._conn.execute("PRAGMA wal_checkpoint(FULL)")
        log.close()

        log2 = SDBPersistentLog(db_path)
        assert log2.verify_chain_integrity() is True
        log2.close()


def test_chain_integrity_after_partial_write() -> None:
    """Chain integrity holds after a normal close following many commits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        log = SDBPersistentLog(db_path)
        runtime = SDBMemoryRuntime(persistent_log=log)
        for step in range(10):
            _commit_cycle(runtime, step=step, content=f"m{step}")
        log.close()

        log2 = SDBPersistentLog(db_path)
        assert log2.verify_chain_integrity() is True
        log2.close()
