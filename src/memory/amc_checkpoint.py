"""AMC memory hierarchy checkpoint — save/load Tier-2 + Tier-3 state (msgpack)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import msgpack

from plugins.memory.episodic_memory import EpisodicMemory, MemoryEntry
from src.memory.amc_tier3 import (
    AMCTier3Config,
    AMCTier3Hook,
    DecayPolicy,
    Tier3Entry,
    TrustLevel,
)
from src.memory.sdb_persistent_log import SDBPersistentLog

SCHEMA_VERSION = "amc_checkpoint/v1"
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})


@dataclass(frozen=True)
class AMCCheckpointHeader:
    schema_version: str
    created_at: str
    amc_version: str
    tier2_entry_count: int
    tier3_store_count: int
    tier3_quarantine_count: int
    event_log_count: int


def _header_to_dict(header: AMCCheckpointHeader) -> dict[str, Any]:
    return asdict(header)


def _header_from_dict(data: dict[str, Any]) -> AMCCheckpointHeader:
    return AMCCheckpointHeader(
        schema_version=str(data["schema_version"]),
        created_at=str(data["created_at"]),
        amc_version=str(data["amc_version"]),
        tier2_entry_count=int(data["tier2_entry_count"]),
        tier3_store_count=int(data["tier3_store_count"]),
        tier3_quarantine_count=int(data["tier3_quarantine_count"]),
        event_log_count=int(data["event_log_count"]),
    )


def _serialize_tier2(tier2: EpisodicMemory) -> dict[str, Any]:
    return {
        "max_entries": tier2._max_entries,
        "entries": [
            {
                "role": entry.role,
                "content": entry.content,
                "importance": entry.importance,
                "id": entry.id,
                "timestamp": entry.timestamp,
                "session_id": entry.session_id,
                "step": entry.step,
            }
            for entry in tier2._entries
        ],
    }


def _deserialize_tier2(data: dict[str, Any]) -> EpisodicMemory:
    memory = EpisodicMemory(max_entries=int(data.get("max_entries", 1000)))
    for row in data.get("entries", []):
        memory._entries.append(
            MemoryEntry(
                role=str(row["role"]),
                content=str(row["content"]),
                importance=float(row.get("importance", 1.0)),
                id=str(row.get("id", "")),
                timestamp=str(row.get("timestamp", datetime.now(UTC).isoformat())),
                session_id=row.get("session_id"),
                step=row.get("step"),
            )
        )
    return memory


def _tier3_entry_to_dict(entry: Tier3Entry) -> dict[str, Any]:
    return {
        "key": entry.key,
        "value": entry.value,
        "source_tier2_id": entry.source_tier2_id,
        "trust_level": str(entry.trust_level),
        "confidence": entry.confidence,
        "last_verified_at": entry.last_verified_at,
        "created_at": entry.created_at,
        "tags": sorted(entry.tags),
        "decay_policy": {
            "half_life_seconds": entry.decay_policy.half_life_seconds,
            "max_age_seconds": entry.decay_policy.max_age_seconds,
        },
    }


def _tier3_entry_from_dict(data: dict[str, Any]) -> Tier3Entry:
    decay_raw = data.get("decay_policy") or {}
    return Tier3Entry(
        key=str(data["key"]),
        value=data["value"],
        source_tier2_id=data.get("source_tier2_id"),
        trust_level=TrustLevel(data.get("trust_level", TrustLevel.UNVERIFIED)),
        confidence=float(data.get("confidence", 1.0)),
        last_verified_at=data.get("last_verified_at"),
        created_at=float(data.get("created_at", 0.0)),
        tags=frozenset(data.get("tags", [])),
        decay_policy=DecayPolicy(
            half_life_seconds=float(decay_raw.get("half_life_seconds", 86400.0)),
            max_age_seconds=float(decay_raw.get("max_age_seconds", 2592000.0)),
        ),
    )


def _serialize_tier3(tier3: AMCTier3Hook) -> dict[str, Any]:
    return {
        "config": asdict(tier3.config),
        "store": {key: _tier3_entry_to_dict(entry) for key, entry in tier3._store.items()},
        "quarantine": {
            key: _tier3_entry_to_dict(entry) for key, entry in tier3._quarantine.items()
        },
        "promoted_count": tier3._promoted_count,
        "quarantined_count": tier3._quarantined_count,
    }


def _deserialize_tier3(data: dict[str, Any]) -> AMCTier3Hook:
    config_raw = data.get("config") or {}
    hook = AMCTier3Hook(AMCTier3Config(**config_raw))
    hook._store = {
        str(key): _tier3_entry_from_dict(entry) for key, entry in (data.get("store") or {}).items()
    }
    hook._quarantine = {
        str(key): _tier3_entry_from_dict(entry)
        for key, entry in (data.get("quarantine") or {}).items()
    }
    hook._promoted_count = int(data.get("promoted_count", len(hook._store)))
    hook._quarantined_count = int(data.get("quarantined_count", len(hook._quarantine)))
    return hook


def save_amc_checkpoint(
    tier2: EpisodicMemory,
    tier3: AMCTier3Hook,
    persistent_log: SDBPersistentLog | None = None,
    *,
    path: str | Path,
    amc_version: str = "0.1.0",
) -> AMCCheckpointHeader:
    """Save Tier-2 episodic memory and Tier-3 LTS hook to a msgpack checkpoint file."""
    event_count = persistent_log.event_count() if persistent_log is not None else 0

    header = AMCCheckpointHeader(
        schema_version=SCHEMA_VERSION,
        created_at=datetime.now(UTC).isoformat(),
        amc_version=amc_version,
        tier2_entry_count=len(tier2),
        tier3_store_count=len(tier3._store),
        tier3_quarantine_count=len(tier3._quarantine),
        event_log_count=event_count,
    )

    payload = {
        "header": _header_to_dict(header),
        "tier2": _serialize_tier2(tier2),
        "tier3": _serialize_tier3(tier3),
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(msgpack.packb(payload, use_bin_type=True))
    return header


def load_amc_checkpoint(
    path: str | Path,
    *,
    expected_version: str | None = None,
) -> tuple[EpisodicMemory, AMCTier3Hook, AMCCheckpointHeader]:
    """Load Tier-2 and Tier-3 state from a checkpoint file."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint not found: {source}")

    payload = msgpack.unpackb(source.read_bytes(), raw=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")

    header = _header_from_dict(payload["header"])
    if expected_version is not None and header.schema_version != expected_version:
        raise ValueError(
            f"checkpoint schema {header.schema_version} != expected {expected_version}"
        )
    if header.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported checkpoint schema: {header.schema_version}")

    tier2 = _deserialize_tier2(payload["tier2"])
    tier3 = _deserialize_tier3(payload["tier3"])
    return tier2, tier3, header


__all__ = [
    "AMCCheckpointHeader",
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "load_amc_checkpoint",
    "save_amc_checkpoint",
]
