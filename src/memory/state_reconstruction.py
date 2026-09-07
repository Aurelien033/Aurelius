"""Replay an SDB event log to reconstruct Tier-2 + Tier-3 state.

Given the ordered event log, reach the exact memory state that existed at any
point in history. Only ``committed`` events mutate state; ``rejected`` events
are ignored. Reconstruction uses event metadata only (no wall-clock sampling).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from plugins.memory.episodic_memory import EpisodicMemory
from src.memory.amc_tier3 import AMCTier3Config, AMCTier3Hook, TrustLevel
from src.memory.sdb_persistent_log import SDBPersistentLog
from src.memory.sdb_runtime import ReplayEvent


@dataclass
class ReconstructedState:
    """Snapshot of reconstructed memory state at a sequence point."""

    target_seq: int
    tier2: EpisodicMemory
    tier3: AMCTier3Hook
    events_replayed: int
    tier2_count: int
    tier3_store_count: int
    tier3_quarantine_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateDiff:
    """Difference between two reconstructed states."""

    tier2_added: list[str] = field(default_factory=list)
    tier2_removed: list[str] = field(default_factory=list)
    tier3_promoted: list[str] = field(default_factory=list)
    tier3_quarantined: list[str] = field(default_factory=list)
    tier3_revoked: list[str] = field(default_factory=list)
    events_between: int = 0


@dataclass(frozen=True)
class StateMatchResult:
    """Outcome of comparing reconstructed state to a live reference."""

    matches: bool
    tier2_mismatch: tuple[str, ...] = ()
    tier3_mismatch: tuple[str, ...] = ()


def _event_payload(meta: dict[str, Any]) -> dict[str, Any]:
    raw = meta.get("payload")
    if isinstance(raw, dict):
        return raw
    return {}


class StateReconstructor:
    """Replay SDB event log and rebuild Tier-2/3 state."""

    def __init__(self, persistent_log: SDBPersistentLog) -> None:
        self.log = persistent_log

    def reconstruct_at(self, target_seq: int | None = None) -> ReconstructedState:
        """Replay events up to ``target_seq`` and rebuild state.

        ``target_seq`` is the number of events to apply (first N in log order).
        ``None`` replays all events.
        """
        return self.reconstruct_with_base(
            EpisodicMemory(),
            AMCTier3Hook(AMCTier3Config()),
            from_seq=0,
            target_seq=target_seq,
        )

    def reconstruct_with_base(
        self,
        tier2: EpisodicMemory,
        tier3: AMCTier3Hook,
        *,
        from_seq: int = 0,
        target_seq: int | None = None,
    ) -> ReconstructedState:
        """Apply committed events after ``from_seq`` onto existing Tier-2/Tier-3 state."""
        events = self.log.replay_from(from_seq)
        if target_seq is not None:
            events = events[:target_seq]

        committed_tier2_payloads: dict[str, dict[str, Any]] = {}
        committed_tier3_payloads: dict[str, dict[str, Any]] = {}

        for event in events:
            if event.event_type == "committed":
                self._apply_committed(
                    event,
                    tier2,
                    tier3,
                    committed_tier2_payloads,
                    committed_tier3_payloads,
                )

        return ReconstructedState(
            target_seq=len(events) if target_seq is None else target_seq,
            tier2=tier2,
            tier3=tier3,
            events_replayed=len(events),
            tier2_count=len(tier2),
            tier3_store_count=len(tier3._store),
            tier3_quarantine_count=len(tier3._quarantine),
            metadata={"from_seq": from_seq},
        )

    def _apply_committed(
        self,
        event: ReplayEvent,
        tier2: EpisodicMemory,
        tier3: AMCTier3Hook,
        t2_payloads: dict[str, dict[str, Any]],
        t3_payloads: dict[str, dict[str, Any]],
    ) -> None:
        meta = event.metadata
        payload = _event_payload(meta)
        tier = str(meta.get("target_tier", ""))
        op = str(meta.get("operation", ""))
        pid = event.proposal_id

        if tier == "tier2" and op == "store":
            entry = tier2.store(
                role=str(payload.get("role", meta.get("role", "system"))),
                content=str(payload.get("content", meta.get("content", ""))),
                importance=float(payload.get("importance", meta.get("importance", 1.0))),
            )
            memory_entry_id = meta.get("memory_entry_id")
            if memory_entry_id:
                entry.id = str(memory_entry_id)
            t2_payloads[pid] = {"entry_id": entry.id, "content": entry.content}

        elif tier == "tier3" and op == "promote":
            key = str(payload.get("key", meta.get("key", pid)))
            value = payload.get("value", meta.get("value", ""))
            confidence = float(payload.get("confidence", meta.get("confidence", 0.5)))
            trust_raw = payload.get("trust_level", meta.get("trust_level"))
            trust = TrustLevel(trust_raw) if trust_raw else None
            tier3.promote(
                key=key,
                value=value,
                confidence=confidence,
                trust_level=trust,
            )
            t3_payloads[pid] = {"key": key}

        elif tier == "tier3" and op == "quarantine":
            key = str(payload.get("key", meta.get("key", pid)))
            value = payload.get("value", meta.get("value", ""))
            confidence = float(payload.get("confidence", meta.get("confidence", 0.0)))
            tier3.quarantine(key=key, value=value, confidence=confidence)
            t3_payloads[pid] = {"key": key, "quarantined": True}

        elif op == "revoke":
            key = payload.get("key", meta.get("key"))
            if key is not None and str(key) in tier3._store:
                tier3._store[str(key)].revoke()

    def diff(self, seq_a: int, seq_b: int | None = None) -> StateDiff:
        """Compute state differences between two sequence points."""
        state_a = self.reconstruct_at(seq_a)
        state_b = self.reconstruct_at(seq_b)

        ids_a = {entry.id for entry in state_a.tier2._entries}
        ids_b = {entry.id for entry in state_b.tier2._entries}

        keys_a_store = set(state_a.tier3._store.keys())
        keys_b_store = set(state_b.tier3._store.keys())
        keys_a_q = set(state_a.tier3._quarantine.keys())
        keys_b_q = set(state_b.tier3._quarantine.keys())

        revoked = [
            key
            for key in keys_a_store & keys_b_store
            if state_a.tier3._store[key].trust_level != state_b.tier3._store[key].trust_level
            and state_b.tier3._store[key].trust_level is TrustLevel.REVOKED
        ]

        return StateDiff(
            tier2_added=sorted(ids_b - ids_a),
            tier2_removed=sorted(ids_a - ids_b),
            tier3_promoted=sorted(keys_b_store - keys_a_store),
            tier3_quarantined=sorted(keys_b_q - keys_a_q),
            tier3_revoked=sorted(revoked),
            events_between=state_b.events_replayed - state_a.events_replayed,
        )

    def verify_against_live(
        self,
        live_tier2: EpisodicMemory,
        live_tier3: AMCTier3Hook,
    ) -> StateMatchResult:
        """Verify full replay matches the supplied live Tier-2/Tier-3 state."""
        reconstructed = self.reconstruct_at(None)

        def _tier2_signature(memory: EpisodicMemory) -> list[tuple[str, str, float]]:
            return sorted(
                (entry.role, entry.content, round(entry.importance, 6))
                for entry in memory._entries
            )

        tier2_mismatch: list[str] = []
        if _tier2_signature(live_tier2) != _tier2_signature(reconstructed.tier2):
            tier2_mismatch.append("tier2_entries")

        def _tier3_store_signature(hook: AMCTier3Hook) -> dict[str, tuple[Any, TrustLevel]]:
            return {
                key: (entry.value, entry.trust_level)
                for key, entry in hook._store.items()
            }

        tier3_mismatch: list[str] = []
        if _tier3_store_signature(live_tier3) != _tier3_store_signature(reconstructed.tier3):
            tier3_mismatch.append("tier3_store")

        live_q = {
            key: (entry.value, entry.trust_level)
            for key, entry in live_tier3._quarantine.items()
        }
        recon_q = {
            key: (entry.value, entry.trust_level)
            for key, entry in reconstructed.tier3._quarantine.items()
        }
        if live_q != recon_q:
            tier3_mismatch.append("tier3_quarantine")

        matches = not tier2_mismatch and not tier3_mismatch
        return StateMatchResult(
            matches=matches,
            tier2_mismatch=tuple(tier2_mismatch),
            tier3_mismatch=tuple(tier3_mismatch),
        )


__all__ = [
    "ReconstructedState",
    "StateDiff",
    "StateMatchResult",
    "StateReconstructor",
]
