"""Tests for the Memory Debate controller — adjudicated Tier-3 promotion."""

from __future__ import annotations

import dataclasses

import pytest

from src.memory.amc_runtime_cache import AMCMemoryBlock, TrustState
from src.memory.memory_debate import (
    DebateDecision,
    DebateVerdict,
    MemoryDebateController,
)


def _block(block_id: str = "blk-1", **kw) -> AMCMemoryBlock:
    base = dict(
        block_id=block_id,
        tokens=(1, 2, 3),
        trust_state=TrustState.UNVERIFIED,
        provenance="test",
        quarantine_state="",
        revocation_epoch=0,
    )
    base.update(kw)
    return AMCMemoryBlock(**base)


def _make_verdict(decision: DebateDecision, *, block_id: str = "blk-1",
                   reason: str = "r", prop: str = "p", sk: str = "s",
                   conf: float = 0.5) -> DebateVerdict:
    return DebateVerdict(
        decision=decision,
        reason=reason,
        proposer_argument=prop,
        skeptic_argument=sk,
        judge_confidence=conf,
        block_id=block_id,
    )


# ── DebateVerdict contract ─────────────────────────────────────────────────


def test_debate_verdict_frozen() -> None:
    v = _make_verdict(DebateDecision.ADMIT)
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        v.reason = "tampered"  # type: ignore[misc]


def test_debate_verdict_rejects_bad_confidence() -> None:
    for bad in [-0.1, 1.5, 2.0]:
        with pytest.raises(ValueError, match="judge_confidence"):
            DebateVerdict(
                decision=DebateDecision.ADMIT,
                reason="r",
                proposer_argument="p",
                skeptic_argument="s",
                judge_confidence=bad,
                block_id="x",
            )


def test_debate_verdict_rejects_bad_decision_type() -> None:
    with pytest.raises(TypeError, match="DebateDecision"):
        DebateVerdict(
            decision="admit",  # type: ignore[arg-type]
            reason="r",
            proposer_argument="p",
            skeptic_argument="s",
            judge_confidence=0.5,
            block_id="x",
        )


# ── run_debate orchestration ───────────────────────────────────────────────


def test_run_debate_calls_proposer_first() -> None:
    calls: list[str] = []

    def propose(b):
        calls.append("propose")
        return "p-arg"

    def skeptic(b, p_arg):
        calls.append("skeptic")
        return "s-arg"

    def judge(b, p_arg, s_arg):
        calls.append("judge")
        return _make_verdict(DebateDecision.ADMIT, block_id=b.block_id)

    v = MemoryDebateController().run_debate(
        _block(), propose_fn=propose, skeptic_fn=skeptic, judge_fn=judge,
    )
    assert calls == ["propose", "skeptic", "judge"]
    assert v.decision == DebateDecision.ADMIT


def test_run_debate_passes_proposer_arg_to_skeptic() -> None:
    received: list[str] = []

    def propose(b): return "hello proposer"
    def skeptic(b, p):
        received.append(p)
        return "skeptic-replies"
    def judge(b, p, s):
        return _make_verdict(DebateDecision.ADMIT, block_id=b.block_id)

    MemoryDebateController().run_debate(
        _block(), propose_fn=propose, skeptic_fn=skeptic, judge_fn=judge,
    )
    assert received == ["hello proposer"]


def test_run_debate_passes_both_args_to_judge() -> None:
    received: list[tuple[str, str]] = []

    def propose(b): return "P"
    def skeptic(b, p): return "S"
    def judge(b, p, s):
        received.append((p, s))
        return _make_verdict(DebateDecision.ADMIT, block_id=b.block_id)

    MemoryDebateController().run_debate(
        _block(), propose_fn=propose, skeptic_fn=skeptic, judge_fn=judge,
    )
    assert received == [("P", "S")]


def test_run_debate_returns_judge_verdict_with_block_id_override() -> None:
    """Judge may (accidentally) tag verdict with wrong block_id — controller overrides."""
    def prop(b): return ""
    def sk(b, p): return ""
    def judge(b, p, s):
        # Judge returns verdict tagged with a DIFFERENT block_id
        return _make_verdict(DebateDecision.ADMIT, block_id="WRONG-ID")

    real_block = _block(block_id="actual-id")
    v = MemoryDebateController().run_debate(
        real_block, propose_fn=prop, skeptic_fn=sk, judge_fn=judge,
    )
    assert v.block_id == "actual-id"


def test_run_debate_handles_admit_quarantine_reject_each() -> None:
    """All three decision classes propagate cleanly through the protocol."""
    for decision in (DebateDecision.ADMIT, DebateDecision.QUARANTINE, DebateDecision.REJECT):
        def prop(b): return "p"
        def sk(b, p): return "s"
        def judge(b, p, s, d=decision):
            return _make_verdict(d, block_id=b.block_id)

        v = MemoryDebateController().run_debate(
            _block(), propose_fn=prop, skeptic_fn=sk, judge_fn=judge,
        )
        assert v.decision == decision


def test_batch_debate_returns_one_verdict_per_block() -> None:
    def prop(b): return "p"
    def sk(b, p): return "s"
    def judge(b, p, s): return _make_verdict(DebateDecision.ADMIT, block_id=b.block_id)

    blocks = [_block(f"id-{i}") for i in range(5)]
    vs = MemoryDebateController().batch_debate(
        blocks, propose_fn=prop, skeptic_fn=sk, judge_fn=judge,
    )
    assert len(vs) == 5
    assert [v.block_id for v in vs] == [f"id-{i}" for i in range(5)]


def test_batch_debate_empty_store_returns_empty() -> None:
    def prop(b): return ""
    def sk(b, p): return ""
    def judge(b, p, s): return _make_verdict(DebateDecision.REJECT, block_id=b.block_id)

    vs = MemoryDebateController().batch_debate(
        [], propose_fn=prop, skeptic_fn=sk, judge_fn=judge,
    )
    assert vs == []


def test_debate_is_pure_does_not_mutate_block() -> None:
    """The debate protocol must not mutate the block under adjudication."""
    block = _block(block_id="immutable", trust_state=TrustState.UNVERIFIED)
    snapshot = block.tokens, block.trust_state, block.provenance, block.quarantine_state

    def prop(b): return "p"
    def sk(b, p): return "s"
    def judge(b, p, s): return _make_verdict(DebateDecision.ADMIT, block_id=b.block_id)

    MemoryDebateController().run_debate(
        block, propose_fn=prop, skeptic_fn=sk, judge_fn=judge,
    )
    # Block fields untouched
    assert (block.tokens, block.trust_state, block.provenance, block.quarantine_state) == snapshot
