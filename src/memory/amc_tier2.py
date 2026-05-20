"""Minimal Tier-2 episodic hook for AMC-first experiments.

This is intentionally small. It is not the full Aurelian Memory Core; it is the
first stable seam that lets runtimes decide whether a turn is surprising enough
to write into episodic memory and retrieve a compact prompt context later.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b

from .amc_runtime_cache import AMCMemoryBlock, TrustState
from .episodic_memory import EpisodicMemory, MemoryEntry


@dataclass(frozen=True)
class AMCTier2Config:
    """Configuration for the Tier-2 episodic memory hook."""

    surprise_threshold: float = 0.5
    max_retrieved: int = 5
    default_importance: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.surprise_threshold <= 1.0:
            raise ValueError("surprise_threshold must be between 0.0 and 1.0")
        if self.max_retrieved < 1:
            raise ValueError("max_retrieved must be >= 1")
        if self.default_importance < 0.0:
            raise ValueError("default_importance must be >= 0.0")


@dataclass(frozen=True)
class AMCTier2AblationResult:
    """No-memory vs Tier-2 prompt-context comparison for benchmark logs."""

    query: str
    no_memory_context: str
    tier2_context: str
    retrieved_entries: int
    context_delta_chars: int
    expected_terms: tuple[str, ...] = ()
    expected_terms_found: tuple[str, ...] = ()
    expected_term_recall: float = 0.0
    memory_available: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable evidence payload."""
        return {
            "query": self.query,
            "no_memory_context": self.no_memory_context,
            "tier2_context": self.tier2_context,
            "retrieved_entries": self.retrieved_entries,
            "context_delta_chars": self.context_delta_chars,
            "expected_terms": list(self.expected_terms),
            "expected_terms_found": list(self.expected_terms_found),
            "expected_term_recall": self.expected_term_recall,
            "memory_available": self.memory_available,
        }


class AMCTier2Hook:
    """Small AMC Tier-2 adapter around the existing episodic memory store."""

    def __init__(
        self,
        config: AMCTier2Config | None = None,
        episodic_memory: EpisodicMemory | None = None,
    ) -> None:
        self.config = config or AMCTier2Config()
        self.episodic = episodic_memory or EpisodicMemory()
        self._observed_events = 0
        self._stored_events = 0
        self._skipped_events = 0
        self._retrieval_calls = 0
        self._retrieval_query_hits = 0
        self._retrieval_fallbacks = 0
        self._retrieved_entries = 0

    def observe(
        self,
        role: str,
        content: str,
        *,
        surprise: float,
        importance: float | None = None,
    ) -> MemoryEntry | None:
        """Store one event if it crosses the Tier-2 surprise gate."""
        if not content.strip():
            return None
        if not 0.0 <= surprise <= 1.0:
            raise ValueError("surprise must be between 0.0 and 1.0")
        self._observed_events += 1
        if surprise < self.config.surprise_threshold:
            self._skipped_events += 1
            return None
        entry = self.episodic.store(
            role=role,
            content=content,
            importance=self.config.default_importance if importance is None else importance,
        )
        self._stored_events += 1
        return entry

    def retrieve(self, query: str, *, limit: int | None = None) -> list[MemoryEntry]:
        """Retrieve matching Tier-2 memories, falling back to recent entries."""
        n = limit or self.config.max_retrieved
        if n < 1:
            raise ValueError("limit must be >= 1")
        self._retrieval_calls += 1
        matches = self.episodic.search(query.strip()) if query.strip() else []
        if matches:
            entries = matches[:n]
            self._retrieval_query_hits += 1
            self._retrieved_entries += len(entries)
            return entries
        entries = self.episodic.retrieve_recent(n)
        if entries:
            self._retrieval_fallbacks += 1
        self._retrieved_entries += len(entries)
        return entries

    @staticmethod
    def _format_context(entries: list[MemoryEntry]) -> str:
        if not entries:
            return ""
        lines = ["Tier-2 episodic memory:"]
        lines.extend(f"- {entry.role}: {entry.content}" for entry in entries)
        return "\n".join(lines)

    def build_context(self, query: str, *, limit: int | None = None) -> str:
        """Build a compact prompt-context block from retrieved Tier-2 memories."""
        return self._format_context(self.retrieve(query, limit=limit))

    def run_no_memory_ablation(
        self,
        query: str,
        *,
        expected_terms: tuple[str, ...] = (),
        limit: int | None = None,
    ) -> AMCTier2AblationResult:
        """Compare empty context with Tier-2 context for one query.

        This does not claim model-quality improvement. It records the evidence a
        benchmark can inspect before deciding whether Tier-2 retrieval is worth
        wiring into a generation path.
        """
        no_memory_context = ""
        entries = self.retrieve(query, limit=limit)
        tier2_context = self._format_context(entries)
        context_lower = tier2_context.lower()
        found = tuple(term for term in expected_terms if term.lower() in context_lower)
        recall = len(found) / len(expected_terms) if expected_terms else 0.0
        return AMCTier2AblationResult(
            query=query,
            no_memory_context=no_memory_context,
            tier2_context=tier2_context,
            retrieved_entries=len(entries),
            context_delta_chars=len(tier2_context) - len(no_memory_context),
            expected_terms=expected_terms,
            expected_terms_found=found,
            expected_term_recall=recall,
            memory_available=bool(tier2_context),
        )

    def stats(self) -> dict[str, int | float]:
        """Return small operational stats for health checks and benchmark logs."""
        write_rate = self._stored_events / self._observed_events if self._observed_events else 0.0
        retrieval_hit_rate = (
            self._retrieval_query_hits / self._retrieval_calls if self._retrieval_calls else 0.0
        )
        average_retrieved = (
            self._retrieved_entries / self._retrieval_calls if self._retrieval_calls else 0.0
        )
        return {
            "episodic_entries": len(self.episodic),
            "surprise_threshold": self.config.surprise_threshold,
            "max_retrieved": self.config.max_retrieved,
            "observed_events": self._observed_events,
            "stored_events": self._stored_events,
            "skipped_events": self._skipped_events,
            "write_rate": write_rate,
            "retrieval_calls": self._retrieval_calls,
            "retrieval_query_hits": self._retrieval_query_hits,
            "retrieval_fallbacks": self._retrieval_fallbacks,
            "retrieved_entries": self._retrieved_entries,
            "average_retrieved_per_call": average_retrieved,
            "retrieval_hit_rate": retrieval_hit_rate,
        }


    def build_runtime_blocks(
        self,
        entries: list[MemoryEntry],
        *,
        policy_version: str = "v0",
        trust_override: TrustState | None = None,
        tier: int = 2,
    ) -> list[AMCMemoryBlock]:
        """Convert *entries* to AMCMemoryBlock objects usable by AMCPrefixCompiler.

        Deliberately conservative defaults — MemoryEntry has no explicit trust
        annotation so blocks default to ``TrustState.UNVERIFIED`` unless the
        caller passes *trust_override*.  This means nothing is silently promoted
        to a privileged cache segment.

        Provenance is derived from ``role`` and ``session_id`` when present.

        Args:
            entries: MemoryEntry list from ``retrieve()`` or ``episodic`` .
            policy_version: Passed through to the cache key.
            trust_override: If provided, overrides the default UNVERIFIED trust
                for every block.  Callers that hold explicit admission/trust
                attestations may set this to TrustState.VERIFIED.
            tier: AMC tier to stamp on every block (default 2).

        Returns:
            list[AMCMemoryBlock] in the same order as *entries*.
        """
        blocks: list[AMCMemoryBlock] = []
        default_trust = trust_override if trust_override is not None else TrustState.UNVERIFIED
        for entry in entries:
            provenance_parts: list[str] = []
            if entry.role:
                provenance_parts.append(entry.role)
            if entry.session_id:
                provenance_parts.append(f"session:{entry.session_id[:8]}")
            provenance = "|".join(provenance_parts) if provenance_parts else "tier2:unknown"

            # Content hash — deterministic content identity over blake2b-16.
            content_hash = blake2b(entry.content.encode("utf-8"), digest_size=16).hexdigest()  # 32-char hex

            # Derive a pseudo token-id sequence from the content hash
            # so every unique content maps to a unique-but-deterministic token tuple.
            token_hash = blake2b(entry.content.encode("utf-8"), digest_size=8).digest()  # 8 bytes
            token_ids = tuple(token_hash)  # 8 ints in [0,255]

            salience = max(0.0, min(1.0, entry.importance))
            provenance_hash = blake2b(provenance.encode("utf-8"), digest_size=16).hexdigest()

            block = AMCMemoryBlock(
                block_id=entry.id,         # stable identifier from MemoryEntry
                tokens=token_ids,
                tier=tier,
                trust_state=default_trust,
                provenance=provenance,
                salience=salience,
                surprise_score=salience,   # reuse importance as initial surprise
                quarantine_state="",
                revocation_epoch=0,
                metadata={"episodic_role": entry.role, "content_hash": content_hash[:12]},
            )
            blocks.append(block)
        return blocks


__all__ = ["AMCTier2AblationResult", "AMCTier2Config", "AMCTier2Hook"]
