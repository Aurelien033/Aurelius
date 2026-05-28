# Memory Debate — Adjudicated Tier-3 Promotion

> **For Hermes:** Implement inline, strict TDD. Do not push.

**Goal:** Add a verifier-heavy promotion protocol for Tier-3 memory: each
candidate block passes through a structured 3-agent debate (proposer,
skeptic, judge) before being admitted. The final verdict — with full
argument provenance — is attached to the block so Tier-3 becomes
auditable.

**Architecture:** Pure-Python orchestrator with injectable callables for
propose / skeptic / judge. Keeps LLM concerns orthogonal: this module
is the debate *protocol*, not the debate *voices*. Downstream, any
LLM-or-rule-based callable can be slotted in.

**Depends on:** Everything up to commit `5a372fd1` (TrustRAG).

---

## Truth surface

HEAD: `5a372fd1 memory(trustrag): quarantine-aware, trust-bound retrieval controller`
Combined suite: 121 tests green.

## Claim / not-claimed

**Claim:**
- A structured debate protocol over `AMCMemoryBlock` produces
  deterministic, auditable verdicts (`admit`/`quarantine`/`reject`).
- Verdict traces are immutable dataclasses carrying proposer argument,
  skeptic argument, judge decision, judge reason, and judge confidence.
- Injected callables receive the block plus prior arguments, so each
  agent sees progressively more context (skeptic sees proposer argument,
  judge sees both).

**Not claimed:**
- We don't implement LLM-backed proposer / skeptic / judge — those are
  injectable callables.
- We don't actually mutate the block's trust state; the controller returns
  a verdict and the caller is responsible for applying it (this keeps
  the controller pure and testable).

## File map

Create:
- `src/memory/memory_debate.py` — `MemoryDebateController`, `DebateVerdict`, `DebateDecision` enum.
- `tests/memory/test_memory_debate.py` — tests.

Modify:
- `docs/research-brief.md` — append section 10.

## API contract

```python
from collections.abc import Callable
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
    block_id: str           # trace: which block was adjudicated

    def __post_init__(self) -> None:
        if not 0.0 <= self.judge_confidence <= 1.0:
            raise ValueError(
                f"judge_confidence must be in [0, 1], got {self.judge_confidence}"
            )
        if not isinstance(self.decision, DebateDecision):
            raise TypeError(f"decision must be DebateDecision, got {type(self.decision)}")


# Injectable signatures
ProposerFn = Callable[[AMCMemoryBlock], str]
SkepticFn = Callable[[AMCMemoryBlock, str], str]
JudgeFn = Callable[[AMCMemoryBlock, str, str], DebateVerdict]


class MemoryDebateController:
    def run_debate(
        self,
        block: AMCMemoryBlock,
        *,
        propose_fn: ProposerFn,
        skeptic_fn: SkepticFn,
        judge_fn: JudgeFn,
    ) -> DebateVerdict:
        proposer_arg = propose_fn(block)
        skeptic_arg = skeptic_fn(block, proposer_arg)
        verdict = judge_fn(block, proposer_arg, skeptic_arg)
        # Enforce block_id consistency regardless of what judge_fn returns
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
        return [
            self.run_debate(
                b, propose_fn=propose_fn, skeptic_fn=skeptic_fn, judge_fn=judge_fn
            )
            for b in blocks
        ]
```

## Tranches

### MD-00: Pre-flight — verify 121 tests green.

### MD-01: Core debate + TDD

**Tests:**
- `test_debate_verdict_frozen`
- `test_debate_verdict_rejects_bad_confidence`
- `test_debate_verdict_rejects_bad_decision_type`
- `test_run_debate_calls_proposer_first`
- `test_run_debate_passes_proposer_arg_to_skeptic`
- `test_run_debate_passes_both_args_to_judge`
- `test_run_debate_returns_judge_verdict_with_block_id_override`
- `test_run_debate_handles_admit_quarantine_reject_each`
- `test_batch_debate_returns_one_verdict_per_block`
- `test_batch_debate_empty_store_returns_empty`
- `test_debate_is_pure_does_not_mutate_block`

### MD-02: Append section 10 to research-brief.md.

## Done criteria

- 11 new tests green.
- Combined suite: 121 + 11 = 132 tests green.
- No mutations to any prior-tranche files.
