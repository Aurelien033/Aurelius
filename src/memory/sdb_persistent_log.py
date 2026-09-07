"""Persistent SDB event log — SQLite-backed append-only store.

Tables:
- amc_events: append-only event log with chain hash integrity
- amc_checkpoints: periodic state snapshots for fast recovery

Events form a hash chain: each stored hash chains to the previous event.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.memory.sdb_runtime import (
    ReplayEvent,
    sanitize_memory_payload,
    stable_hash,
    stable_json_dumps,
)

GENESIS_HASH = "genesis"
_EVENT_REPLAY_HASH_KEY = "__event_replay_hash"

EVENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS amc_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    event_type TEXT NOT NULL,
    proposal_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    replay_hash TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_wall_time REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_proposal_id ON amc_events(proposal_id);
CREATE INDEX IF NOT EXISTS idx_events_event_type ON amc_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_seq ON amc_events(seq);
"""

CHECKPOINT_SCHEMA = """
CREATE TABLE IF NOT EXISTS amc_checkpoints (
    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    seq INTEGER NOT NULL,
    checkpoint_blob TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (seq) REFERENCES amc_events(seq)
);
"""


def _chain_hash(prev_hash: str, event: ReplayEvent) -> str:
    chain_input = {
        "prev_hash": prev_hash,
        "event_id": event.event_id,
        "event_type": event.event_type,
        "proposal_id": event.proposal_id,
        "timestamp": event.timestamp.isoformat(),
        "replay_hash": event.replay_hash,
    }
    return stable_hash(chain_input)


@dataclass(frozen=True)
class StoredEventRow:
    seq: int
    event_id: str
    event_type: str
    proposal_id: str
    timestamp: str
    chain_hash: str
    prev_hash: str
    metadata_json: str


class SDBPersistentLog:
    """SQLite-backed append-only event log for SDB Memory Runtime."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(EVENT_SCHEMA)
        self._conn.executescript(CHECKPOINT_SCHEMA)

    def _last_hash(self) -> str:
        row = self._conn.execute(
            "SELECT replay_hash FROM amc_events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else GENESIS_HASH

    def _fsync_batch(self) -> None:
        self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")

    def append(self, event: ReplayEvent) -> int:
        """Append one event and return its sequence number."""
        prev_hash = self._last_hash()
        chain_hash = _chain_hash(prev_hash, event)

        stored_meta = sanitize_memory_payload(dict(event.metadata))
        stored_meta[_EVENT_REPLAY_HASH_KEY] = event.replay_hash
        meta_json = stable_json_dumps(stored_meta)

        cursor = self._conn.execute(
            """INSERT INTO amc_events
               (event_id, event_type, proposal_id, timestamp,
                replay_hash, prev_hash, metadata_json, created_wall_time)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.event_id,
                event.event_type,
                event.proposal_id,
                event.timestamp.isoformat(),
                chain_hash,
                prev_hash,
                meta_json,
                time.time(),
            ),
        )
        self._fsync_batch()
        row_id = cursor.lastrowid
        if row_id is None:
            raise RuntimeError("append failed to return sequence number")
        return int(row_id)

    def _row_to_event(self, row: tuple[Any, ...]) -> ReplayEvent:
        metadata = json.loads(row[5])
        event_replay_hash = metadata.pop(_EVENT_REPLAY_HASH_KEY, row[4])
        return ReplayEvent(
            event_id=row[0],
            event_type=row[1],
            proposal_id=row[2],
            timestamp=datetime.fromisoformat(row[3]),
            metadata=metadata,
            replay_hash=str(event_replay_hash),
        )

    def replay_from(self, from_seq: int = 0, limit: int | None = None) -> list[ReplayEvent]:
        """Replay events with sequence strictly greater than ``from_seq``."""
        query = (
            "SELECT event_id, event_type, proposal_id, timestamp, replay_hash, metadata_json "
            "FROM amc_events WHERE seq > ? ORDER BY seq ASC"
        )
        params: list[Any] = [from_seq]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_event(row) for row in rows]

    def iter_events(self) -> Iterator[StoredEventRow]:
        rows = self._conn.execute(
            """SELECT seq, event_id, event_type, proposal_id, timestamp,
                      replay_hash, prev_hash, metadata_json
               FROM amc_events ORDER BY seq ASC"""
        )
        for row in rows:
            yield StoredEventRow(
                seq=int(row[0]),
                event_id=row[1],
                event_type=row[2],
                proposal_id=row[3],
                timestamp=row[4],
                chain_hash=row[5],
                prev_hash=row[6],
                metadata_json=row[7],
            )

    def verify_chain_integrity(self) -> bool:
        """Return False if any event was tampered with."""
        prev_hash = GENESIS_HASH
        for stored in self.iter_events():
            metadata = json.loads(stored.metadata_json)
            event_replay_hash = metadata.get(_EVENT_REPLAY_HASH_KEY, stored.chain_hash)
            expected = stable_hash(
                {
                    "prev_hash": prev_hash,
                    "event_id": stored.event_id,
                    "event_type": stored.event_type,
                    "proposal_id": stored.proposal_id,
                    "timestamp": stored.timestamp,
                    "replay_hash": event_replay_hash,
                }
            )
            if stored.chain_hash != expected:
                return False
            if stored.prev_hash != prev_hash:
                return False
            prev_hash = stored.chain_hash
        return True

    def event_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM amc_events").fetchone()
        return int(row[0]) if row else 0

    def save_checkpoint(self, state_blob: dict[str, Any]) -> int:
        """Save a state snapshot at the current sequence point."""
        current_seq = self._conn.execute("SELECT MAX(seq) FROM amc_events").fetchone()[0]
        current_seq = int(current_seq or 0)
        blob = stable_json_dumps(sanitize_memory_payload(state_blob))
        self._conn.execute(
            "INSERT INTO amc_checkpoints (seq, checkpoint_blob, created_at) VALUES (?, ?, ?)",
            (current_seq, blob, datetime.now(UTC).isoformat()),
        )
        self._fsync_batch()
        return current_seq

    def load_latest_checkpoint(self) -> tuple[int, dict[str, Any]] | None:
        """Load the most recent checkpoint."""
        row = self._conn.execute(
            "SELECT seq, checkpoint_blob FROM amc_checkpoints "
            "ORDER BY checkpoint_id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return int(row[0]), json.loads(row[1])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SDBPersistentLog:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ["SDBPersistentLog", "GENESIS_HASH"]
