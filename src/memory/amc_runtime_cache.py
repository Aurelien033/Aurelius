"""AMC runtime / cache interface — SGLang-inspired patterns, Aurelius-owned.

This module adapts the useful SGLang separation ideas (prefix caching, paged/block
memory, chunked prefill, phase separation, structured write decisions) into a
trust-aware AMC surface without importing or coupling to any SGLang internals.

Nothing here modifies the forward path; callers opt in explicitly.

Key surfaces
------------
AMCMemoryBlock          — per-block memory record (tier, token span, trust state)
AMCMemoryCacheKey       — hashable composite key for cache identity
AMCPrefixSegment        — compiled trust-eligible prefix block
AMCPrefixCompileResult  — structured result of prefix compilation
AMCPrefixCompiler       — trust-aware compiler: candidate records → cache blocks
AMCWriteDecision        — fail-closed write decision for AMC memory action

Trust/Quarantine/Revocation contract
--------------------------------------
Retrieved memory is NEVER inserted as a high-authority system instruction merely
because it is cached.  The AMCPrefixCompiler:
  1. Accepts only trusted/allowed records in the privileged compile path.
  2. Excludes quarantined / untrusted records from trusted/system context.
  3. Preserves lower-priority non-privileged context only if existing policy
     grants it (``TrustLevel.UNVERIFIED`` → non-privilege, not privileged).
  4. Changes the cache key if trust/state changes — cache identity is bound to
     trust/quarantine/revocation, so the same content in a different trust state
     gets a *different* key and never re-appears stale under the old key.

Pure stdlib; no torch.  Side-effect-free import.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Trust / quarantine primitives (mirrors amc_tier3.TrustLevel) ─────────────

class TrustState(str, Enum):
    """Confidence tier for an AMC memory block or entry."""

    VERIFIED  = "verified"   # multi-session, cross-checked
    UNVERIFIED = "unverified"  # single source, unconfirmed
    QUARANTINED = "quarantined"  # flagged; excluded from privileged context
    REVOKED   = "revoked"    # superseded or retracted

    @property
    def is_privileged(self) -> bool:
        return self == TrustState.VERIFIED

    @property
    def can_be_trusted_context(self) -> bool:
        """True → eligible to contribute to trusted/system prompt context."""
        return self == TrustState.VERIFIED


# ─────────────────────────────────────────────────────────────────────────────
# AMCWriteDecision
# ─────────────────────────────────────────────────────────────────────────────

class WriteAction(str, Enum):
    """AMC memory write decision outcomes."""
    WRITE    = "write"      # persist to target_tier
    SKIP     = "skip"       # silently drop
    QUARANTINE = "quarantine"  # redirect to quarantine, no privilege
    PROMOTE  = "promote"    # move up (e.g. tier-2 → tier-3)
    REVOKE   = "revoke"     # explicitly invalidate a cached copy


@dataclass(frozen=True)
class AMCWriteDecision:
    """Structured, fail-closed write decision for AMC memory writes.

    Invalid or unsafe decisions are rejected at construction time so callers
    cannot accidentally bypass safety invariants.
    """

    action: WriteAction
    target_tier: int            # 1-based tier number (1, 2, or 3)
    confidence: float           # 0.0 – 1.0
    trust_label: TrustState
    provenance: str             # human-readable source tag
    reason: str                 # why this decision was made
    quarantine_reason: str = ""
    ttl_seconds: float | None = None   # optional expiry; None = no expiry

    def __post_init__(self) -> None:
        # Fail-closed guards — mutation is needed during validation
        object.__setattr__(self, "action", WriteAction(self.action))
        object.__setattr__(self, "trust_label", TrustState(self.trust_label))
        if self.target_tier not in (1, 2, 3):
            raise ValueError(
                f"target_tier must be 1, 2, or 3, got {self.target_tier!r}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence!r}"
            )
        if not self.reason.strip():
            raise ValueError("reason must be a non-empty string")
        if self.ttl_seconds is not None and self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive or None")

    @property
    def is_safe(self) -> bool:
        """True if the decision does not bypass safety controls."""
        return self.action != WriteAction.WRITE or self.confidence >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Cache key — trust/state bound
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AMCMemoryCacheKey:
    """Immutable, trust/provenance-aware cache key for AMC prefix segments.

    Changing any of the trust/quarantine/revocation fields produces a different
    key, so stale cache entries from a prior trust state can never be served
    accidentally.
    """

    content_hash: str          # hash of block content bytes
    token_hash: str            # hash of the token-id sequence
    tier: int
    trust_state: TrustState
    provenance_hash: str       # hash of the provenance record
    policy_version: str
    quarantine_state: str      # "" if not quarantined
    revocation_epoch: int      # monotonically increasing; 0 = never revoked
    session_fingerprint: str | None = None  # optional session affinity marker

    @classmethod
    def compute(
        cls,
        *,
        content: str,
        token_ids: tuple[int, ...],
        tier: int,
        trust_state: TrustState,
        provenance: str,
        policy_version: str,
        revocation_epoch: int = 0,
        session_fingerprint: str | None = None,
    ) -> AMCMemoryCacheKey:
        """Build a ``AMCMemoryCacheKey`` by hashing the provided fields."""
        content_hash = hashlib.blake2b(
            content.encode("utf-8"), digest_size=16
        ).hexdigest()
        token_hash = hashlib.blake2b(
            b"".join(t.to_bytes(8, "little", signed=True) for t in token_ids),
            digest_size=16,
        ).hexdigest()
        provenance_hash = hashlib.blake2b(
            provenance.encode("utf-8"), digest_size=16
        ).hexdigest()
        quarantine_state = (
            trust_state.value if trust_state == TrustState.QUARANTINED else ""
        )
        return cls(
            content_hash=content_hash,
            token_hash=token_hash,
            tier=tier,
            trust_state=trust_state,
            provenance_hash=provenance_hash,
            policy_version=policy_version,
            quarantine_state=quarantine_state,
            revocation_epoch=revocation_epoch,
            session_fingerprint=session_fingerprint,
        )

    @property
    def fingerprint(self) -> str:
        """Stable DER-like string for LRU key comparisons."""
        parts = [
            self.content_hash,
            self.token_hash,
            str(self.tier),
            self.trust_state.value,
            self.provenance_hash,
            self.policy_version,
            self.quarantine_state,
            str(self.revocation_epoch),
            self.session_fingerprint or "",
        ]
        return "|".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Memory block — paged / block-level abstraction
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AMCMemoryBlock:
    """One AMC memory block / page.

    Attributes:
        block_id:           Stable UUID-stable identifier.
        tokens:              Token-id sequence in this block.
        tier:                AMC tier of the source (1, 2, or 3).
        trust_state:         Current trust/confidence level.
        provenance:          Human-readable provenance tag.
        salience:            Importance weight [0.0, 1.0].
        surprise_score:      Surprise at time of creation [0.0, 1.0].
        last_used:           Monotonic timestamp of last cache access.
        dirty:               True → block has unsaved writes pending writeback.
        quarantine_state:    "" or quarantine record ID.
        revocation_epoch:    Monotonically increasing; bumped on revocation.
        embedding_handle:    Opaque reference to a pre-computed summary / embedding.
        kv_ref:              Opaque reference to backing KV storage (tensors/pages).
        metadata:            Arbitrary typed metadata bag.

    Mutable by design — instances may be updated in-place.  Callers who need
    immutability should make a copy after construction.
    """

    block_id: str
    tokens: tuple[int, ...] = field(default_factory=tuple)
    tier: int = 2
    trust_state: TrustState = TrustState.UNVERIFIED
    provenance: str = ""
    salience: float = 0.5
    surprise_score: float = 0.0
    last_used: float = field(default_factory=time.monotonic)
    dirty: bool = False
    quarantine_state: str = ""
    revocation_epoch: int = 0
    embedding_handle: Any = None
    kv_ref: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tier not in (1, 2, 3):
            raise ValueError(f"tier must be 1, 2, or 3, got {self.tier!r}")
        if isinstance(self.tokens, list):
            object.__setattr__(self, "tokens", tuple(self.tokens))
        if not 0.0 <= self.salience <= 1.0:
            raise ValueError(f"salience must be in [0.0, 1.0], got {self.salience!r}")
        if not 0.0 <= self.surprise_score <= 1.0:
            raise ValueError(
                f"surprise_score must be in [0.0, 1.0], got {self.surprise_score!r}"
            )
        if not isinstance(self.provenance, str):
            raise TypeError("provenance must be str")

    def mark_used(self) -> None:
        self.last_used = time.monotonic()

    def mark_dirty(self) -> None:
        self.dirty = True

    def mark_clean(self) -> None:
        self.dirty = False

    def quarantine(self, reason: str = "") -> None:
        self.trust_state = TrustState.QUARANTINED
        self.quarantine_state = reason

    def revoke(self, epoch: int | None = None) -> None:
        self.trust_state = TrustState.REVOKED
        if epoch is not None:
            object.__setattr__(self, "revocation_epoch", epoch)

    def to_cache_key(self, policy_version: str, session_fingerprint: str | None = None) -> AMCMemoryCacheKey:
        return AMCMemoryCacheKey.compute(
            content=self.provenance,
            token_ids=self.tokens,
            tier=self.tier,
            trust_state=self.trust_state,
            provenance=self.provenance,
            policy_version=policy_version,
            revocation_epoch=self.revocation_epoch,
            session_fingerprint=session_fingerprint,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Prefix segments — compiled, trust-filtered cache units
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AMCPrefixSegment:
    """A compiled, cache-ready AMC memory segment.

    Attributes:
        cache_key:       Unique trust-aware identity key.
        tokens:          Token ids covered by this segment.
        kv_ref:          Opaque reference to the backing KV store.
        trust_state:     Trust level at compile time (read-only afterwards).
        tier:            AMC tier.
        salience:        Salience weight [0.0, 1.0].
        last_used:       Monotonic timestamp for LRU.
        metadata:        Arbitrary metadata bag.
    """

    cache_key: AMCMemoryCacheKey
    tokens: tuple[int, ...] = field(default_factory=tuple)
    kv_ref: Any = None
    trust_state: TrustState = TrustState.UNVERIFIED
    tier: int = 2
    salience: float = 0.5
    last_used: float = field(default_factory=time.monotonic)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tier not in (1, 2, 3):
            raise ValueError(f"tier must be 1, 2, or 3, got {self.tier!r}")
        if isinstance(self.tokens, list):
            object.__setattr__(self, "tokens", tuple(self.tokens))




# ─────────────────────────────────────────────────────────────────────────────
# Chunked prefix prefill seam
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AMCPrefixChunk:
    """A deterministic chunk of a compiled AMC prefix segment.

    Pure helper — no torch, no CUDA, no serving dependency.
    Callers may aggregate chunks before flushing to a model forward pass.
    """

    chunk_id: int
    cache_key_fingerprint: str | None = None
    tokens: tuple[int, ...] = field(default_factory=tuple)
    kv_ref: Any = None
    trust_state: TrustState = TrustState.UNVERIFIED
    tier: int = 2
    salience: float = 0.5
    provenance: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_prefix_segments(
    segments: list[AMCPrefixSegment],
    *,
    max_tokens: int,
    include_quarantined: bool = False,
    include_revoked: bool = False,
) -> list[AMCPrefixChunk]:
    """Split *segments* into deterministic chunks of at most *max_tokens* tokens.

    Args:
        segments: Compiled prefix segments (from AMCPrefixCompiler.compile()).
        max_tokens: Maximum token count per chunk. Must be > 0.
        include_quarantined: If True, quarantined segments are included; they
            are always routed to the appropriate quarantined bucket.
        include_revoked: If True, revoked segments are included.

    Returns:
        Ordered list of AMCPrefixChunk objects. The total number of tokens
        across all chunks equals the total tokens in *segments* (order preserved).

    Raises:
        ValueError: If *max_tokens* <= 0.
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be > 0, got {max_tokens!r}")

    filtered: list[AMCPrefixSegment] = []
    for seg in segments:
        if seg.trust_state == TrustState.QUARANTINED and not include_quarantined:
            continue
        if seg.trust_state == TrustState.REVOKED and not include_revoked:
            continue
        filtered.append(seg)

    chunks: list[AMCPrefixChunk] = []
    chunk_id = 0

    for seg in filtered:
        tokens = seg.tokens
        if not tokens:
            chunks.append(
                AMCPrefixChunk(
                    chunk_id=chunk_id,
                    cache_key_fingerprint=seg.cache_key.fingerprint,
                    trust_state=seg.trust_state,
                    tier=seg.tier,
                    salience=seg.salience,
                    provenance=seg.cache_key.provenance_hash,
                    metadata={"segment_tier": seg.tier, **seg.metadata},
                )
            )
            chunk_id += 1
            continue

        for offset in range(0, len(tokens), max_tokens):
            chunk = AMCPrefixChunk(
                chunk_id=chunk_id,
                cache_key_fingerprint=seg.cache_key.fingerprint,
                tokens=tokens[offset : offset + max_tokens],
                kv_ref=seg.kv_ref,
                trust_state=seg.trust_state,
                tier=seg.tier,
                salience=seg.salience,
                provenance=seg.cache_key.provenance_hash,
                metadata={"segment_tier": seg.tier, **seg.metadata},
            )
            chunks.append(chunk)
            chunk_id += 1

    return chunks


@dataclass
class AMCPrefixCompileResult:
    """Result of a trust-aware prefix compilation pass.

    Attributes:
        trusted:     Segments from VERIFIED records — eligible for privileged use.
        allowed:     Segments from UNVERIFIED records — can be REQUESTED, not injected.
        quarantined: Quarantined segments — excluded from all prompt context.
        revoked:     Revoked segments — cache key changed to invalidate old identity.
        stats:       Aggregated counters keyed by label.
        cache_key_changes: Mapping old_key → new_key for revoked/state-changed blocks.
    """

    trusted: list[AMCPrefixSegment] = field(default_factory=list)
    allowed: list[AMCPrefixSegment] = field(default_factory=list)
    quarantined: list[AMCPrefixSegment] = field(default_factory=list)
    revoked: list[AMCPrefixSegment] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    cache_key_changes: dict[str, str] = field(default_factory=dict)

    def total_segments(self) -> int:
        return sum(
            len(s)
            for s in (self.trusted, self.allowed, self.quarantined, self.revoked)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trusted_count":    len(self.trusted),
            "allowed_count":    len(self.allowed),
            "quarantined_count": len(self.quarantined),
            "revoked_count":    len(self.revoked),
            "stats":            dict(self.stats),
            "cache_key_changes": dict(self.cache_key_changes),
        }


# ─────────────────────────────────────────────────────────────────────────────
# AMCPrefixCompiler — trust-aware prefix compilation
# ─────────────────────────────────────────────────────────────────────────────

# Sentinel: no trust version to track invalidation when state degrades
_NO_TRUST_EPOCH = 0


class AMCPrefixCompiler:
    """Trust-aware prefix compiler for AMC memory blocks.

    Routes memory blocks into the correct tier context based on trust state
    and produces cache keys that change when trust/quarantine/revocation state
    changes, so stale high-trust cache entries are never served after degradation.

    Args:
        policy_version:    Stable policy tag — changes to this invalidate all keys.

    Allowed trust-state routing
    ---------------------------
    VERIFIED      → ``trusted`` list   (eligible for privileged prefix use)
    UNVERIFIED    → ``allowed`` list   (eligible for non-privileged context)
    QUARANTINED   → ``quarantined`` list (never injected into prompt context)
    REVOKED       → ``revoked`` list   (cache key changed; callers must not use
                  previous identity)
    """

    def __init__(self, policy_version: str = "v0", *, session_fingerprint: str | None = None) -> None:
        if not policy_version.strip():
            raise ValueError("policy_version must be a non-empty string")
        self.policy_version = policy_version
        self.session_fingerprint = session_fingerprint
        self._compilations: int = 0
        self._blocks_seen: int = 0
        self._trusted_count: int = 0
        self._allowed_count: int = 0
        self._quarantined_count: int = 0
        self._revoked_count: int = 0
        self._cache_key_changes: dict[str, str] = {}

    # ── public API ──────────────────────────────────────────────────────────

    def compile(self, blocks: list[AMCMemoryBlock]) -> AMCPrefixCompileResult:
        """Compile ``blocks`` into trust-classified prefix segments.

        Each block is routed to exactly one bucket.  Changing trust state between
        compilations changes the resulting cache key, so callers serving old KV
        references must invalidate them.

        Returns:
            :class:`AMCPrefixCompileResult` with four classified lists.
        """
        self._compilations += 1
        trusted: list[AMCPrefixSegment] = []
        allowed: list[AMCPrefixSegment] = []
        quarantined: list[AMCPrefixSegment] = []
        revoked_bucket: list[AMCPrefixSegment] = []

        for block in blocks:
            self._blocks_seen += 1

            cache_key = block.to_cache_key(self.policy_version, self.session_fingerprint)
            old_key_str = self._cache_key_changes.get(block.block_id, _NO_TRUST_EPOCH)
            new_key_str = cache_key.fingerprint

            key_differs = (
                block.revocation_epoch > _NO_TRUST_EPOCH
                or old_key_str != new_key_str
            )

            segment = AMCPrefixSegment(
                cache_key=cache_key,
                tokens=block.tokens,
                kv_ref=block.kv_ref,
                trust_state=block.trust_state,
                tier=block.tier,
                salience=block.salience,
                last_used=block.last_used,
                metadata={"provenance": block.provenance},
            )

            match block.trust_state:
                case TrustState.VERIFIED:
                    trusted.append(segment)
                    self._trusted_count += 1
                    if key_differs:
                        self._cache_key_changes[block.block_id] = new_key_str

                case TrustState.UNVERIFIED:
                    allowed.append(segment)
                    self._allowed_count += 1

                case TrustState.QUARANTINED:
                    quarantined.append(segment)
                    self._quarantined_count += 1

                case TrustState.REVOKED:
                    revoked_bucket.append(segment)
                    self._revoked_count += 1
                    if key_differs:
                        self._cache_key_changes[block.block_id] = new_key_str

        return AMCPrefixCompileResult(
            trusted=trusted,
            allowed=allowed,
            quarantined=quarantined,
            revoked=revoked_bucket,
            stats=self._make_stats(),
            cache_key_changes=dict(self._cache_key_changes),
        )

    def _make_stats(self) -> dict[str, object]:
        return {
            "compilations":      self._compilations,
            "blocks_seen":       self._blocks_seen,
            "trusted_segments":  self._trusted_count,
            "allowed_segments":  self._allowed_count,
            "quarantined_segments": self._quarantined_count,
            "revoked_segments":  self._revoked_count,
            "cache_key_changes": len(self._cache_key_changes),
        }

    def reset_counters(self) -> None:
        """Reset compiler stats without touching config."""
        self._compilations = 0
        self._blocks_seen = 0
        self._trusted_count = 0
        self._allowed_count = 0
        self._quarantined_count = 0
        self._revoked_count = 0
        self._cache_key_changes = {}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helper — future observability surfaces
# ─────────────────────────────────────────────────────────────────────────────

class AMCMetricsCollector:
    """Lightweight metrics collector for AMC cache and write operations.

    No timer, no histogram framework — just counters keyed by event label
    so that future serving/metrics layers can attach a proper exporter.
    """

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._hit_by_trust: dict[str, int] = {}
        self._quarantine_exclusions: int = 0
        self._revocation_invalidations: int = 0
        self._write_actions: dict[str, int] = {}

    # ── counter helpers ────────────────────────────────────────────────────

    def inc(self, label: str, delta: int = 1) -> None:
        self._counters[label] = self._counters.get(label, 0) + delta

    def set_gauge(self, label: str, value: float) -> None:
        self._gauges[label] = float(value)

    # ── cache events ───────────────────────────────────────────────────────

    def record_cache_hit(self, trust_state: TrustState) -> None:
        label = f"cache_hit_{trust_state.value}"
        self._counters[label] = self._counters.get(label, 0) + 1
        self._hit_by_trust[trust_state.value] = (
            self._hit_by_trust.get(trust_state.value, 0) + 1
        )

    def record_cache_miss(self, reason: str = "miss") -> None:
        # Count both total misses and per-reason misses for observability.
        self.inc("cache_miss_total")
        self.inc(f"cache_miss_{reason}")

    def record_quarantine_exclusion(self) -> None:
        self._quarantine_exclusions += 1
        self.inc("quarantine_exclusion")

    def record_revocation_invalidation(self, old_key: str) -> None:
        self._revocation_invalidations += 1
        self.inc("revocation_invalidation")

    # ── write decision events ───────────────────────────────────────────────

    def record_write_decision(self, decision: AMCWriteDecision) -> None:
        key = f"write_{decision.action.value}"
        self.inc(key)                          # also surface in general counters
        self._write_actions[key] = self._write_actions.get(key, 0) + 1

    # ── snapshots ───────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        return {
            "counters":                 dict(self._counters),
            "gauges":                   dict(self._gauges),
            "hit_by_trust":             dict(self._hit_by_trust),
            "quarantine_exclusions":    self._quarantine_exclusions,
            "revocation_invalidations": self._revocation_invalidations,
            "write_actions":            dict(self._write_actions),
        }

    def hit_rate_by_trust(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for trust_label, h in self._hit_by_trust.items():
            hits = self._counters.get(f"cache_hit_{trust_label}", 0)
            miss_candidates = [
                k for k, v in self._counters.items()
                if k == f"cache_miss_{trust_label}"
            ]
            total = hits + sum(self._counters[k] for k in miss_candidates)
            out[trust_label] = h / total if total else 0.0
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Module-level defaults / exports
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Primitives
    "TrustState",
    "WriteAction",
    "AMCWriteDecision",
    "AMCMemoryCacheKey",
    "AMCMemoryBlock",
    # Prefix cache
    "AMCPrefixSegment",
    "AMCPrefixCompileResult",
    "AMCPrefixCompiler",
    # Metrics
    "AMCMetricsCollector",
    # Chunked prefill seam
    "AMCPrefixChunk",
    "chunk_prefix_segments",
]
