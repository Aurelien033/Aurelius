"""AMC Tier 3 — Long-term store with quarantine, trust scoring, and decay policy.

Tier 3 is the durable layer of the Aurelian Memory Core.  Unlike Tier 1 (working)
and Tier 2 (episodic), Tier 3 entries are:
- Synthesised from Tier 2 by the :class:`AMCTier3Hook`; they are never written
  directly by arbitrary generation.
- Tagged with provenance, confidence, ``last_verified_at``, and ``trust_level``.
- Subject to a configurable decay policy so stale high-importance entries do not
  dominate retrieval.
- Quarantined when sources are flagged as untrusted (external, adversarial, or
  low-confidence). Quarantined entries are not promoted to the active store until
  a ``consolidate()`` pass verifies them.

Design reference: :doc:`docs/AMC_FIRST_ARCHITECTURE.md`
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from plugins.memory.long_term_memory import (
    LongTermMemory,
)
from src._compat import StrEnum

# ── Trust level ──────────────────────────────────────────────────────────────


class TrustLevel(StrEnum):
    """Confidence tier for a Tier 3 memory entry."""

    TRUSTED = "trusted"  # multi-session verified
    UNVERIFIED = "unverified"  # single source, pending verification
    QUARANTINED = "quarantined"  # flagged as untrusted; excluded from retrieval
    REVOKED = "revoked"  # superseded or retracted


# ── Decay policy ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DecayPolicy:
    """Rules governing how importance decays over time.

    After ``half_life_seconds`` have elapsed, importance is halved.  After
    ``max_age_seconds`` the entry is pruned on the next consolidation pass.
    """

    half_life_seconds: float = 86400.0  # 1 day
    max_age_seconds: float = 2592000.0  # 30 days


# ── Tier-3 entry ─────────────────────────────────────────────────────────────


@dataclass
class Tier3Entry:
    """A single durable memory in the Tier-3 store.

    ``source_tier2_id`` links back to the originating Tier-2 entry so the
    consolidation cycle can re-score the entry if new evidence arrives.
    """

    key: str
    value: Any
    source_tier2_id: str | None = None
    trust_level: TrustLevel = TrustLevel.UNVERIFIED
    confidence: float = 1.0  # 0.0 — 1.0 at time of promotion
    last_verified_at: float | None = None
    created_at: float = field(default_factory=time.monotonic)
    tags: frozenset[str] = field(default_factory=frozenset)
    decay_policy: DecayPolicy = field(default_factory=DecayPolicy)

    # ── helpers ──────────────────────────────────────────────────────────────

    def decayed_importance(self, now: float | None = None) -> float:
        """Return importance adjusted for elapsed time."""
        now = now or time.monotonic()
        if self.trust_level is TrustLevel.QUARANTINED:
            return 0.0
        elapsed = now - self.created_at
        if elapsed <= 0:
            return float(self.confidence)
        half_lives = elapsed / self.decay_policy.half_life_seconds
        decay_factor = 0.5**half_lives
        return round(self.confidence * decay_factor, 6)

    def is_expired(self, now: float | None = None) -> bool:
        now = now or time.monotonic()
        return (now - self.created_at) > self.decay_policy.max_age_seconds

    def verify(self, confidence: float = 1.0) -> None:
        """Promote to TRUSTED after external verification."""
        self.last_verified_at = time.monotonic()
        self.confidence = max(0.0, min(1.0, confidence))
        if self.confidence >= 0.7:
            self.trust_level = TrustLevel.TRUSTED
        else:
            self.trust_level = TrustLevel.UNVERIFIED

    def revoke(self) -> None:
        self.trust_level = TrustLevel.REVOKED


# ── Tier-3 hook ──────────────────────────────────────────────────────────────


@dataclass
class AMCTier3Config:
    """Configuration for the Tier-3 LTS hook."""

    min_confidence: float = 0.6  # minimum confidence to promote (avoid QUARANTINED)
    quarantine_threshold: float = 0.3  # below this → flagged as untrusted source
    max_entries: int = 10000
    consolidation_batch: int = 128
    trust_promotion_threshold: float = 0.7

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if not 0.0 <= self.quarantine_threshold <= 1.0:
            raise ValueError("quarantine_threshold must be in [0, 1]")
        if self.max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        # quarantine_threshold must be strictly below min_confidence so the two
        # gates form a non-empty gap:
        # [0, qt) → quarantine, [qt, mc) → unverified, [mc, 1] → trusted.
        if self.quarantine_threshold >= self.min_confidence:
            raise ValueError(
                f"quarantine_threshold ({self.quarantine_threshold}) must be "
                f"less than min_confidence ({self.min_confidence})"
            )


@dataclass
class Tier3ConsolidationResult:
    """Result of a Tier-3 consolidation pass."""

    promoted: int = 0
    quarantined: int = 0
    revoked: int = 0
    expired_pruned: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "promoted": self.promoted,
            "quarantined": self.quarantined,
            "revoked": self.revoked,
            "expired_pruned": self.expired_pruned,
            "errors": self.errors,
        }


@dataclass
class Tier3Stats:
    """Snapshot of Tier-3 store health."""

    total_entries: int = 0
    trusted: int = 0
    unverified: int = 0
    quarantined: int = 0
    revoked: int = 0
    avg_confidence: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "total_entries": self.total_entries,
            "trusted": self.trusted,
            "unverified": self.unverified,
            "quarantined": self.quarantined,
            "revoked": self.revoked,
            "avg_confidence": self.avg_confidence,
        }


class AMCTier3Hook:
    """AMC Tier-3 long-term store hook.

    Responsibilities:
    - **promote**: move a score-confirmed observation from Tier 2 to durable LTS.
    - **quarantine**: hold a low-confidence / externally-sourced observation in
      isolation until it passes a manual or automatic verification gate.
    - **consolidate**: sweep quarantined entries → promote if confidence now
      sufficient, otherwise expire.
    - **prioritize**: return active (trusted + unverified) entries sorted by
      decayed importance; quarantined entries are always excluded.
    """

    def __init__(self, config: AMCTier3Config | None = None) -> None:
        self.config = config or AMCTier3Config()
        self._store: dict[str, Tier3Entry] = {}
        self._quarantine: dict[str, Tier3Entry] = {}
        self._ltm = LongTermMemory()
        self._promoted_count: int = 0
        self._quarantined_count: int = 0

    # ── Promote / Quarantine ─────────────────────────────────────────────────

    def promote(
        self,
        *,
        key: str,
        value: Any,
        source_tier2_id: str | None = None,
        confidence: float = 1.0,
        trust_level: TrustLevel | None = None,
        tags: frozenset[str] | None = None,
    ) -> Tier3Entry | None:
        """Promote an observation to Tier 3, subject to the confidence gate.

        * If ``confidence < quarantine_threshold`` → :meth:`quarantine` is called and
          its result returned (trust_level forced to :class:`TrustLevel.QUARANTINED`).
        * Otherwise a tier-3 entry is created; ``trust_level`` defaults to TRUSTED
          when ``confidence >= trust_promotion_threshold`` to reduce boilerplate at
          call sites.
        """
        confidence = max(0.0, min(1.0, confidence))

        if confidence < self.config.quarantine_threshold:
            return self.quarantine(
                key=key,
                value=value,
                source_tier2_id=source_tier2_id,
                confidence=confidence,
                tags=tags,
            )

        if trust_level is None:
            trust_level = (
                TrustLevel.TRUSTED
                if confidence >= self.config.trust_promotion_threshold
                else TrustLevel.UNVERIFIED
            )

        entry = Tier3Entry(
            key=key,
            value=value,
            source_tier2_id=source_tier2_id,
            trust_level=trust_level,
            confidence=confidence,
            tags=tags or frozenset(),
        )
        self._prune_if_full()
        self._store[key] = entry
        self._promoted_count += 1
        if trust_level is TrustLevel.TRUSTED:
            self._ltm.store(key, value, importance=confidence)
        return entry

    def quarantine(
        self,
        *,
        key: str,
        value: Any,
        source_tier2_id: str | None = None,
        confidence: float = 0.0,
        tags: frozenset[str] | None = None,
    ) -> Tier3Entry:
        """Place a low-confidence entry in quarantine.  Not visible to :meth:`prioritize`."""
        entry = Tier3Entry(
            key=key,
            value=value,
            source_tier2_id=source_tier2_id,
            trust_level=TrustLevel.QUARANTINED,
            confidence=confidence,
            tags=tags or frozenset(),
        )
        self._quarantine[key] = entry
        self._quarantined_count += 1
        return entry

    # ── Consolidation ────────────────────────────────────────────────────────

    def consolidate(self) -> Tier3ConsolidationResult:
        """Sweep quarantined entries and prune expired active entries."""
        result = Tier3ConsolidationResult()
        now = time.monotonic()

        # Quarantine sweep
        to_promote: dict[str, Tier3Entry] = {}
        to_keep_quarantined: dict[str, Tier3Entry] = {}
        for key, entry in self._quarantine.items():
            if entry.is_expired(now):
                result.expired_pruned += 1
                continue
            if entry.confidence >= self.config.min_confidence:
                entry.trust_level = TrustLevel.UNVERIFIED
                to_promote[key] = entry
                result.promoted += 1
            else:
                to_keep_quarantined[key] = entry
        self._quarantine = to_keep_quarantined
        for key, entry in to_promote.items():
            self._store[key] = entry
            self._ltm.store(key, entry.value, importance=entry.confidence)

        # Active-store expiry sweep
        expired_keys = [k for k, v in self._store.items() if v.is_expired(now)]
        for k in expired_keys:
            self._store.pop(k, None)
            result.expired_pruned += 1

        return result

    # ── Prioritization / retrieval ────────────────────────────────────────────

    def prioritize(self, limit: int = 5) -> list[Tier3Entry]:
        """Return active entries sorted by decayed importance (highest first).

        Quarantined and revoked entries are always excluded.
        """
        scored = [
            (e.decayed_importance(), e)
            for e in self._store.values()
            if e.trust_level not in (TrustLevel.QUARANTINED, TrustLevel.REVOKED)
        ]
        scored.sort(key=lambda p: p[0], reverse=True)
        return [e for _, e in scored[:limit]]

    def verify_and_promote(self, key: str, confidence: float = 1.0) -> Tier3Entry | None:
        """Manually verify a quarantined entry and promote it to active."""
        if key in self._quarantine:
            entry = self._quarantine.pop(key)
            entry.verify(confidence)
            self._store[key] = entry
            self._ltm.store(key, entry.value, importance=entry.confidence)
            return entry
        return None

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> Tier3Stats:
        trusted = sum(1 for e in self._store.values() if e.trust_level is TrustLevel.TRUSTED)
        unverified = sum(1 for e in self._store.values() if e.trust_level is TrustLevel.UNVERIFIED)
        active = trusted + unverified
        avg_conf = sum(e.confidence for e in self._store.values()) / active if active else 0.0
        return Tier3Stats(
            total_entries=len(self._store) + len(self._quarantine),
            trusted=trusted,
            unverified=unverified,
            quarantined=len(self._quarantine),
            revoked=sum(1 for e in self._store.values() if e.trust_level is TrustLevel.REVOKED),
            avg_confidence=round(avg_conf, 4),
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _prune_if_full(self) -> None:
        if len(self._store) >= self.config.max_entries:
            # Drop the entry with the lowest decayed importance
            if self._store:
                worst_key = min(
                    self._store,
                    key=lambda k: self._store[k].decayed_importance(),
                )
                del self._store[worst_key]
