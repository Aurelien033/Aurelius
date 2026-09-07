"""Skill crystallization — compress frequently retrieved Tier-3 facts into skills."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.memory.amc_tier3 import AMCTier3Hook, Tier3Entry, TrustLevel

GenerateFn = Callable[[str], str]


@dataclass(frozen=True)
class CrystallizationProposal:
    source_key: str
    source_value: Any
    retrieval_count: int
    history: tuple[dict[str, Any], ...] = field(default_factory=tuple)


class SkillCrystallizer:
    """Promote frequently retrieved Tier-3 facts into compressed skill entries."""

    def __init__(
        self,
        tier3_hook: AMCTier3Hook,
        generate_fn: GenerateFn,
        *,
        retrieval_threshold: int = 5,
    ) -> None:
        self.tier3 = tier3_hook
        self.generate = generate_fn
        self.threshold = retrieval_threshold
        self._retrieval_counts: dict[str, int] = {}
        self._retrieval_history: dict[str, list[dict[str, Any]]] = {}

    def record_retrieval(self, entry: Tier3Entry, query_context: str) -> None:
        key = entry.key
        if key.startswith("constitutional:") or key.startswith("crystal:"):
            return
        self._retrieval_counts[key] = self._retrieval_counts.get(key, 0) + 1
        self._retrieval_history.setdefault(key, []).append(
            {
                "query": query_context,
                "timestamp": time.time(),
            }
        )

    def scan_for_crystallization(self) -> list[CrystallizationProposal]:
        proposals: list[CrystallizationProposal] = []
        for key, count in self._retrieval_counts.items():
            if count < self.threshold:
                continue
            entry = self.tier3._store.get(key)
            if entry is None or entry.trust_level != TrustLevel.TRUSTED:
                continue
            proposals.append(
                CrystallizationProposal(
                    source_key=key,
                    source_value=entry.value,
                    retrieval_count=count,
                    history=tuple(self._retrieval_history.get(key, [])),
                )
            )
        return proposals

    def crystallize(self, proposal: CrystallizationProposal) -> Tier3Entry | None:
        source = self.tier3._store.get(proposal.source_key)
        if source is None or source.trust_level != TrustLevel.TRUSTED:
            return None

        history_summary = "\n".join(f"- {item['query']}" for item in proposal.history[-10:])
        prompt = f"""
Compress this frequently-retrieved fact into a more abstract, general principle.

Original fact: {proposal.source_value}
Retrieved {proposal.retrieval_count} times in contexts:
{history_summary}

Output ONLY the compressed principle (one sentence):
""".strip()
        compressed = self.generate(prompt).strip()
        if not compressed:
            return None

        new_key = f"crystal:{proposal.source_key}"
        elevated_confidence = min(1.0, float(source.confidence) + 0.05)
        entry = self.tier3.promote(
            key=new_key,
            value=compressed,
            confidence=elevated_confidence,
            source_tier2_id=source.source_tier2_id,
            tags=frozenset({"crystallized", "skill"}),
        )
        if entry is None:
            return None

        source.tags = frozenset(set(source.tags) | {"stale"})
        self._retrieval_counts[proposal.source_key] = 0
        self._retrieval_history[proposal.source_key] = []
        return entry

    def run_cycle(self) -> list[Tier3Entry]:
        results: list[Tier3Entry] = []
        for proposal in self.scan_for_crystallization():
            entry = self.crystallize(proposal)
            if entry is not None:
                results.append(entry)
        return results


__all__ = ["CrystallizationProposal", "SkillCrystallizer"]
