"""Tests for AMC Tier-2/Tier-3 msgpack checkpoint save/load."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from plugins.memory.episodic_memory import EpisodicMemory
from src.memory.amc_checkpoint import (
    SCHEMA_VERSION,
    load_amc_checkpoint,
    save_amc_checkpoint,
)
from src.memory.amc_tier3 import AMCTier3Config, AMCTier3Hook, TrustLevel
from src.memory.sdb_persistent_log import SDBPersistentLog
from src.memory.sdb_runtime import ReplayEvent


def _populated_tier2() -> EpisodicMemory:
    memory = EpisodicMemory(max_entries=100)
    memory.store(role="user", content="prefer dark mode", importance=0.9)
    memory.store(role="assistant", content="noted preference", importance=0.7)
    return memory


def _populated_tier3() -> AMCTier3Hook:
    hook = AMCTier3Hook(AMCTier3Config(min_confidence=0.5, quarantine_threshold=0.2))
    hook.promote(key="pref.theme", value="dark", confidence=0.95)
    hook.quarantine(key="ext.fact", value="unverified claim", confidence=0.1)
    return hook


def test_save_creates_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ckpt.msgpack"
        header = save_amc_checkpoint(_populated_tier2(), _populated_tier3(), path=path)
        assert path.is_file()
        assert header.schema_version == SCHEMA_VERSION


def test_roundtrip_preserves_tier2_entries() -> None:
    tier2 = _populated_tier2()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ckpt.msgpack"
        save_amc_checkpoint(tier2, _populated_tier3(), path=path)
        restored, _, _ = load_amc_checkpoint(path)
    assert len(restored) == len(tier2)
    assert restored.retrieve_recent(1)[0].content == tier2.retrieve_recent(1)[0].content


def test_roundtrip_preserves_tier3_trust_levels() -> None:
    tier3 = _populated_tier3()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ckpt.msgpack"
        save_amc_checkpoint(_populated_tier2(), tier3, path=path)
        _, restored, _ = load_amc_checkpoint(path)
    assert restored._store["pref.theme"].trust_level is TrustLevel.TRUSTED


def test_roundtrip_preserves_tier3_quarantine() -> None:
    tier3 = _populated_tier3()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ckpt.msgpack"
        save_amc_checkpoint(_populated_tier2(), tier3, path=path)
        _, restored, _ = load_amc_checkpoint(path)
    assert "ext.fact" in restored._quarantine
    assert restored._quarantine["ext.fact"].trust_level is TrustLevel.QUARANTINED


def test_header_counts_match() -> None:
    tier2 = _populated_tier2()
    tier3 = _populated_tier3()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ckpt.msgpack"
        header = save_amc_checkpoint(tier2, tier3, path=path)
        _, restored_t3, loaded_header = load_amc_checkpoint(path)
    assert header.tier2_entry_count == len(tier2)
    assert header.tier3_store_count == len(tier3._store)
    assert header.tier3_quarantine_count == len(tier3._quarantine)
    assert loaded_header.tier2_entry_count == len(tier2)
    assert len(restored_t3._store) == header.tier3_store_count


def test_schema_version_check() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ckpt.msgpack"
        save_amc_checkpoint(_populated_tier2(), _populated_tier3(), path=path)
        with pytest.raises(ValueError, match="!= expected"):
            load_amc_checkpoint(path, expected_version="amc_checkpoint/v0")


def test_backward_compatible_schema() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ckpt.msgpack"
        save_amc_checkpoint(_populated_tier2(), _populated_tier3(), path=path)
        tier2, tier3, header = load_amc_checkpoint(path)
    assert header.schema_version == SCHEMA_VERSION
    assert len(tier2) > 0
    assert len(tier3._store) > 0


def test_save_with_persistent_log_records_event_count() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "events.db"
        ckpt_path = Path(tmpdir) / "ckpt.msgpack"
        with SDBPersistentLog(db_path) as log:
            log.append(
                ReplayEvent(
                    event_id="e1",
                    event_type="propose",
                    proposal_id="p1",
                    timestamp=datetime.now(UTC),
                    metadata={},
                    replay_hash="hash1",
                )
            )
            header = save_amc_checkpoint(
                _populated_tier2(),
                _populated_tier3(),
                log,
                path=ckpt_path,
            )
        assert header.event_log_count == 1


def test_load_nonexistent_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_amc_checkpoint("/tmp/does-not-exist-amc-checkpoint.msgpack")


def test_empty_checkpoint_roundtrip() -> None:
    tier2 = EpisodicMemory()
    tier3 = AMCTier3Hook()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ckpt.msgpack"
        save_amc_checkpoint(tier2, tier3, path=path)
        restored_t2, restored_t3, header = load_amc_checkpoint(path)
    assert len(restored_t2) == 0
    assert len(restored_t3._store) == 0
    assert len(restored_t3._quarantine) == 0
    assert header.tier2_entry_count == 0
