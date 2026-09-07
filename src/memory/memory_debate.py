"""Memory Debate — Adjudicated Tier-3 promotion protocol.

Structured 3-agent debate (proposer → skeptic → judge) for deciding
whether an AMCMemoryBlock earns Tier-3 admission. Each agent receives
progressively more context — skeptic sees proposer's argument, judge
sees both. The final verdict (with full argument traces) is returned
as an immutable DebateVerdict.

This module orchestrates the protocol; it does not implement the agents.
Proposer/skeptic/judge callables are injected, so any LLM or rule-based
backend can slot in.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum

from src.memory.amc_runtime_cache import AMCMemoryBlock


class DebateDecision(StrEnum):
    ADMIT = "admit"
    QUARANTINE = "quarantine"
    REJECT = "reject"


@dataclass(frozen=True)
class DebateVerdict:
    decision: DebateDecision
    reason: str
    proposer_argument: str
    skeptic_argument: str
    judge_confidence: float
    block_id: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.judge_confidence <= 1.0):
            raise ValueError(f"judge_confidence must be in [0, 1], got {self.judge_confidence}")
        if not isinstance(self.decision, DebateDecision):
            raise TypeError(f"decision must be DebateDecision, got {type(self.decision).__name__}")


# Callable signatures for injectable debate agents
ProposerFn = Callable[[AMCMemoryBlock], str]
SkepticFn = Callable[[AMCMemoryBlock, str], str]
JudgeFn = Callable[[AMCMemoryBlock, str, str], DebateVerdict]


class MemoryDebateController:
    """Pure orchestration of the 3-step debate protocol.

    Does not mutate the input block; returns a verdict the caller applies.
    """

    def run_debate(
        self,
        block: AMCMemoryBlock,
        *,
        propose_fn: ProposerFn,
        skeptic_fn: SkepticFn,
        judge_fn: JudgeFn,
    ) -> DebateVerdict:
        # Step 1: proposer argues alone
        proposer_arg = propose_fn(block)

        # Step 2: skeptic sees proposer's argument
        skeptic_arg = skeptic_fn(block, proposer_arg)

        # Step 3: judge sees both arguments
        verdict = judge_fn(block, proposer_arg, skeptic_arg)

        # Enforce block_id consistency — verdict's block_id must match the
        # block being debated, regardless of what judge_fn returned.
        return DebateVerdict(
            decision=verdict.decision,
            reason=verdict.reason,
            proposer_argument=proposer_arg,
            skeptic_argument=skeptic_arg,
            judge_confidence=verdict.judge_confidence,
            block_id=block.block_id,
        )

    def batch_debate(
        self,
        blocks: Iterable[AMCMemoryBlock],
        *,
        propose_fn: ProposerFn,
        skeptic_fn: SkepticFn,
        judge_fn: JudgeFn,
    ) -> list[DebateVerdict]:
        """Run the debate protocol on every block, return ordered verdict list."""
        return [
            self.run_debate(b, propose_fn=propose_fn, skeptic_fn=skeptic_fn, judge_fn=judge_fn)
            for b in blocks
        ]
