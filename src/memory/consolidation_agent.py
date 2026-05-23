"""Post-session memory consolidation: reflect on a transcript and update Tier-3."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import blake2b
from typing import Any

from src.memory.amc_tier2 import AMCTier2Hook
from src.memory.amc_tier3 import AMCTier3Hook, Tier3Entry, TrustLevel

GenerateFn = Callable[[str], str]

_GREETING_MARKERS = frozenset({"hi", "hello", "hey", "thanks", "thank", "ok", "okay"})
_TRANSIENT_NUMBER_RE = re.compile(r"^[\d\s\.,$%-]+$")

_PROPOSAL_BLOCK_RE = re.compile(
    r"FACT:\s*(?P<fact>.+?)\s*"
    r"CONFIDENCE:\s*(?P<confidence>[0-9.]+)\s*"
    r"TAGS:\s*(?P<tags>[^\n]+?)\s*"
    r"CONTRADICTS:\s*(?P<contradicts>[^\n]+)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ConsolidationProposal:
    fact: str
    confidence: float
    tags: tuple[str, ...]
    contradicts: str | None
    key: str = ""

    def __post_init__(self) -> None:
        if not self.key:
            digest = blake2b(self.fact.encode("utf-8"), digest_size=8).hexdigest()
            object.__setattr__(self, "key", f"fact:{digest}")


@dataclass
class ConsolidationReport:
    promoted: list[Tier3Entry] = field(default_factory=list)
    quarantined: list[Tier3Entry] = field(default_factory=list)
    rejected: list[ConsolidationProposal] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "promoted_count": len(self.promoted),
            "quarantined_count": len(self.quarantined),
            "rejected_count": len(self.rejected),
            "promoted_keys": [entry.key for entry in self.promoted],
            "quarantined_keys": [entry.key for entry in self.quarantined],
            "rejected_facts": [proposal.fact for proposal in self.rejected],
        }


class ConsolidationAgent:
    """Post-session agent that reviews transcripts and consolidates Tier-3 memory."""

    PROMPT_TEMPLATE = """
You are the memory consolidation agent for the Aurelius AMC system.
Review this session and identify facts worth persisting long-term.

Session transcript:
{transcript}

Existing long-term memories (Tier-3):
{existing_memories}

For each fact worth remembering, output in this format:
FACT: <the fact>
CONFIDENCE: <0.0-1.0>
TAGS: <comma-separated tags>
CONTRADICTS: <existing memory ID or "none">

Only promote:
- User preferences and corrections
- Architecture decisions
- Durable facts (not temporary numbers, greetings, or transient tool output)
""".strip()

    def __init__(
        self,
        tier2_hook: AMCTier2Hook,
        tier3_hook: AMCTier3Hook,
        generate_fn: GenerateFn | None = None,
        *,
        min_confidence: float = 0.6,
    ) -> None:
        self.tier2 = tier2_hook
        self.tier3 = tier3_hook
        self.generate = generate_fn
        self.min_confidence = min_confidence

    def _format_transcript(self, session_messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for message in session_messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else "(empty session)"

    def _format_existing(self, entries: list[Tier3Entry]) -> str:
        if not entries:
            return "(none)"
        return "\n".join(f"- {entry.key}: {entry.value}" for entry in entries)

    @staticmethod
    def parse_proposals(output: str) -> list[ConsolidationProposal]:
        proposals: list[ConsolidationProposal] = []
        for match in _PROPOSAL_BLOCK_RE.finditer(output):
            fact = match.group("fact").strip()
            if not fact:
                continue
            try:
                confidence = float(match.group("confidence"))
            except ValueError:
                continue
            tags_raw = match.group("tags").strip()
            tags = tuple(tag.strip() for tag in tags_raw.split(",") if tag.strip())
            contradicts_raw = match.group("contradicts").strip()
            contradicts = None if contradicts_raw.lower() in {"none", ""} else contradicts_raw
            proposals.append(
                ConsolidationProposal(
                    fact=fact,
                    confidence=max(0.0, min(1.0, confidence)),
                    tags=tags,
                    contradicts=contradicts,
                )
            )
        return proposals

    @staticmethod
    def should_skip_proposal(proposal: ConsolidationProposal) -> bool:
        fact_lower = proposal.fact.lower().strip()
        if not fact_lower:
            return True
        tokens = fact_lower.split()
        if len(tokens) <= 4 and any(token in _GREETING_MARKERS for token in tokens):
            return True
        if _TRANSIENT_NUMBER_RE.fullmatch(fact_lower):
            return True
        if fact_lower.isdigit():
            return True
        return False

    def _build_prompt(self, session_messages: list[dict[str, Any]]) -> str:
        existing = self.tier3.prioritize(limit=20)
        return self.PROMPT_TEMPLATE.format(
            transcript=self._format_transcript(session_messages),
            existing_memories=self._format_existing(existing),
        )

    def reflect(self, session_messages: list[dict[str, Any]]) -> ConsolidationReport:
        if not session_messages:
            return ConsolidationReport()

        if self.generate is None:
            output = ""
        else:
            output = self.generate(self._build_prompt(session_messages))

        proposals = self.parse_proposals(output)
        return self.apply_proposals(proposals)

    def reflect_from_output(self, output: str) -> ConsolidationReport:
        """Apply consolidation from pre-generated model text (testing seam)."""
        return self.apply_proposals(self.parse_proposals(output))

    def apply_proposals(self, proposals: list[ConsolidationProposal]) -> ConsolidationReport:
        report = ConsolidationReport()
        for proposal in proposals:
            if self.should_skip_proposal(proposal):
                report.rejected.append(proposal)
                continue
            if proposal.confidence < self.min_confidence:
                report.rejected.append(proposal)
                continue

            if proposal.contradicts and not proposal.contradicts.startswith("constitutional:"):
                existing = self.tier3._store.get(proposal.contradicts)
                if existing is not None and existing.trust_level != TrustLevel.QUARANTINED:
                    existing.trust_level = TrustLevel.QUARANTINED
                    report.quarantined.append(existing)

            entry = self.tier3.promote(
                key=proposal.key,
                value=proposal.fact,
                confidence=proposal.confidence,
                tags=frozenset(proposal.tags),
            )
            if entry is None:
                report.rejected.append(proposal)
                continue
            if proposal.confidence >= 0.85:
                entry.verify(proposal.confidence)
            report.promoted.append(entry)
        return report


def reflect_and_consolidate(
    session_messages: list[dict[str, Any]],
    *,
    tier2_hook: AMCTier2Hook,
    tier3_hook: AMCTier3Hook,
    generate_fn: GenerateFn | None = None,
) -> ConsolidationReport:
    """Run post-session consolidation (ReAct loop integration entry point)."""
    _ = tier2_hook
    agent = ConsolidationAgent(tier2_hook, tier3_hook, generate_fn=generate_fn)
    return agent.reflect(session_messages)


__all__ = [
    "ConsolidationAgent",
    "ConsolidationProposal",
    "ConsolidationReport",
    "reflect_and_consolidate",
]
