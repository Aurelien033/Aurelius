"""Tests for SQLite-backed SDB persistent event log (T03)."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from src.memory.sdb_persistent_log import SDBPersistentLog
from src.memory.sdb_runtime import (
    MemoryOperation,
    MemorySourceType,
    MemoryTargetTier,
    ReplayEvent,
    SDBMemoryRuntime,
    VerificationDecision,
)


def _event(
    *,
    event_id: str = "evt-1",
    proposal_id: str = "prop-1",
    event_type: str = "proposed",
) -> ReplayEvent:
    return ReplayEvent(
        event_id=event_id,
        event_type=event_type,
        proposal_id=proposal_id,
        timestamp=datetime.now(UTC),
        metadata={"note": "hello"},
        replay_hash=f"hash-{event_id}",
    )


def _tmp_db() -> tuple[tempfile.TemporaryDirectory[str], Path]:
    tmp = tempfile.TemporaryDirectory()
    return tmp, Path(tmp.name) / "sdb_events.db"


def test_init_creates_db_file() -> None:
    tmp, path = _tmp_db()
    with tmp:
        log = SDBPersistentLog(path)
        assert path.exists()
        log.close()


def test_append_returns_increasing_seq() -> None:
    tmp, path = _tmp_db()
    with tmp:
        log = SDBPersistentLog(path)
        seq1 = log.append(_event(event_id="a"))
        seq2 = log.append(_event(event_id="b"))
        assert seq2 > seq1
        log.close()


def test_append_stores_event_fields() -> None:
    tmp, path = _tmp_db()
    with tmp:
        log = SDBPersistentLog(path)
        event = _event(event_id="evt-store", proposal_id="p-99")
        log.append(event)
        row = log._conn.execute(
            "SELECT event_id, proposal_id, event_type FROM amc_events"
        ).fetchone()
        assert row == (event.event_id, event.proposal_id, event.event_type)
        log.close()


def test_replay_from_returns_events_in_order() -> None:
    tmp, path = _tmp_db()
    with tmp:
        log = SDBPersistentLog(path)
        log.append(_event(event_id="e1", proposal_id="p1"))
        log.append(_event(event_id="e2", proposal_id="p2"))
        events = log.replay_from(0)
        assert [e.event_id for e in events] == ["e1", "e2"]
        log.close()


def test_replay_from_with_limit() -> None:
    tmp, path = _tmp_db()
    with tmp:
        log = SDBPersistentLog(path)
        for idx in range(5):
            log.append(_event(event_id=f"e{idx}"))
        events = log.replay_from(0, limit=2)
        assert len(events) == 2
        log.close()


def test_chain_integrity_passes_when_untouched() -> None:
    tmp, path = _tmp_db()
    with tmp:
        log = SDBPersistentLog(path)
        for idx in range(3):
            log.append(_event(event_id=f"ok-{idx}"))
        assert log.verify_chain_integrity() is True
        log.close()


def test_chain_integrity_fails_on_tamper() -> None:
    tmp, path = _tmp_db()
    with tmp:
        log = SDBPersistentLog(path)
        log.append(_event(event_id="tamper-me"))
        log._conn.execute(
            "UPDATE amc_events SET replay_hash = ? WHERE event_id = ?",
            ("deadbeef", "tamper-me"),
        )
        assert log.verify_chain_integrity() is False
        log.close()


def test_save_and_load_checkpoint() -> None:
    tmp, path = _tmp_db()
    with tmp:
        log = SDBPersistentLog(path)
        log.append(_event())
        seq = log.save_checkpoint({"commits": 1, "events": 1})
        loaded = log.load_latest_checkpoint()
        assert loaded is not None
        assert loaded[0] == seq
        assert loaded[1]["commits"] == 1
        log.close()


def test_crash_recovery() -> None:
    tmp, path = _tmp_db()
    with tmp:
        log = SDBPersistentLog(path)
        log.append(_event(event_id="persist-1"))
        log.append(_event(event_id="persist-2"))
        log.close()

        log2 = SDBPersistentLog(path)
        assert log2.event_count() == 2
        assert log2.verify_chain_integrity() is True
        log2.close()


def test_wal_mode_survives_interrupt() -> None:
    tmp, path = _tmp_db()
    with tmp:
        log = SDBPersistentLog(path)
        for idx in range(10):
            log.append(_event(event_id=f"wal-{idx}"))
        log.close()

        conn = sqlite3.connect(str(path))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert str(mode).lower() == "wal"

        log2 = SDBPersistentLog(path)
        assert log2.event_count() == 10
        log2.close()


def test_massive_append_performance() -> None:
    # Threshold is deliberately generous: CI runners (shared, virtualized)
    # have shown 5.0-6.0s on this loop while dev machines sit well under.
    # Regression signal is order-of-magnitude, not a tight bound.
    max_seconds = float(os.environ.get("SDB_APPEND_MAX_SECONDS", "10.0"))
    tmp, path = _tmp_db()
    with tmp:
        log = SDBPersistentLog(path)
        start = time.perf_counter()
        for idx in range(10_000):
            log.append(_event(event_id=f"bulk-{idx}"))
        elapsed = time.perf_counter() - start
        assert log.event_count() == 10_000
        assert elapsed < max_seconds
        log.close()


def test_metadata_json_is_sanitized() -> None:
    tmp, path = _tmp_db()
    with tmp:
        log = SDBPersistentLog(path)
        event = ReplayEvent(
            event_id="secret-evt",
            event_type="proposed",
            proposal_id="p-secret",
            timestamp=datetime.now(UTC),
            metadata={"api_key": "sk-live", "note": "visible"},
            replay_hash="hash-secret",
        )
        log.append(event)
        row = log._conn.execute(
            "SELECT metadata_json FROM amc_events WHERE event_id = ?",
            ("secret-evt",),
        ).fetchone()
        stored = json.loads(row[0])
        assert stored["api_key"] == "[REDACTED]"
        assert stored["note"] == "visible"
        assert "sk-live" not in row[0]
        log.close()


def test_runtime_integration_with_persistent_log() -> None:
    tmp, path = _tmp_db()
    with tmp:
        log = SDBPersistentLog(path)
        runtime = SDBMemoryRuntime(persistent_log=log)
        proposal = runtime.propose(
            session_id="s1",
            step=1,
            proposer="test",
            source_type=MemorySourceType.USER,
            target_tier=MemoryTargetTier.TIER2,
            operation=MemoryOperation.STORE,
            payload={"note": "test"},
        )
        verification = runtime.verify(
            proposal,
            verifier="checker",
            decision=VerificationDecision.ACCEPT,
            reason="ok",
        )
        runtime.commit(proposal, verification_result=verification)

        events = log.replay_from(0)
        assert len(events) == 3
        assert log.verify_chain_integrity() is True
        assert log.event_count() == len(runtime.replay_events())
        log.close()
