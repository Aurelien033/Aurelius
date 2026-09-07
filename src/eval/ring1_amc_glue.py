"""Ring 1 AMC glue — read-only adapters from live memory surfaces to trace events.

Tranche 2: wires AMCTier2Hook + SDBMemoryRuntime + stub MCTS without modifying
core AMC model or alignment code.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any

from src.eval.ring1_trace_logger import MCTSStats, MemoryReadEvent, MemoryWriteEvent
from src.memory.amc_tier2 import AMCTier2Config, AMCTier2Hook
from src.memory.sdb_runtime import (
    MemoryOperation,
    MemorySourceType,
    MemoryTargetTier,
    SDBMemoryRuntime,
    VerificationDecision,
)
from src.reasoning.mcts_reasoner import MCTSNode, MCTSReasoner

CONSTITUTIONAL_PRINCIPLES = (
    "Prefer verified sources over speculative claims.",
    "Backtrack when tool output contradicts stored episodic memory.",
)


@dataclass
class Ring1MemorySession:
    """Per-trace AMC memory session (Tier-2 + SDB audit + working-memory stub)."""

    session_id: str
    tier2: AMCTier2Hook
    sdb: SDBMemoryRuntime
    layer_range: list[int] = field(default_factory=lambda: [8, 12, 16, 20, 24])
    working_memory: dict[int, str] = field(default_factory=dict)
    _stored_entry_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: dict[str, Any], session_id: str) -> Ring1MemorySession:
        threshold = config.get("surprise_threshold", 0.65)
        tier2 = AMCTier2Hook(AMCTier2Config(surprise_threshold=threshold))
        return cls(
            session_id=session_id,
            tier2=tier2,
            sdb=SDBMemoryRuntime(),
            layer_range=list(config.get("layer_range", [8, 12, 16, 20, 24])),
        )

    def reset(self) -> None:
        self.working_memory.clear()
        self._stored_entry_ids.clear()


def read_memory_for_think(
    session: Ring1MemorySession,
    *,
    observation: str,
    step_id: int,
    rng: random.Random,
) -> list[MemoryReadEvent]:
    """Consult working, episodic, and constitutional memory during Think."""
    reads: list[MemoryReadEvent] = []

    if step_id == 1:
        principle = rng.choice(CONSTITUTIONAL_PRINCIPLES)
        reads.append(
            MemoryReadEvent(
                source="constitutional",
                content=principle,
                influenced_action=False,
                layer=None,
                step_id=step_id,
            )
        )
        return reads

    for layer, content in session.working_memory.items():
        reads.append(
            MemoryReadEvent(
                source="working",
                content=content,
                influenced_action=False,
                layer=layer,
                step_id=step_id,
            )
        )

    entries = session.tier2.retrieve(observation)
    for entry in entries:
        influenced = any(token in observation.lower() for token in _content_tokens(entry.content))
        reads.append(
            MemoryReadEvent(
                source="episodic",
                content=entry.content,
                influenced_action=influenced,
                layer=rng.choice(session.layer_range),
                step_id=step_id,
            )
        )

    if step_id > 2 and not any(read.influenced_action for read in reads) and entries:
        reads[-1] = MemoryReadEvent(
            source="episodic",
            content=entries[-1].content,
            influenced_action=True,
            layer=rng.choice(session.layer_range),
            step_id=step_id,
        )

    return reads


def write_memory_after_step(
    session: Ring1MemorySession,
    *,
    step_id: int,
    observation: str,
    reflection_summary: str,
    action_failed: bool,
    surprise_override: float | None,
    rng: random.Random,
) -> list[MemoryWriteEvent]:
    """Apply Integration Contract write points: failed step, insight, or reflect."""
    layer = rng.choice(session.layer_range)
    if surprise_override is not None:
        surprise_score = surprise_override
    elif action_failed:
        surprise_score = round(rng.uniform(0.72, 0.98), 3)
    elif "insight" in reflection_summary.lower() or step_id == 1:
        surprise_score = round(rng.uniform(0.68, 0.95), 3)
    else:
        surprise_score = round(rng.uniform(0.25, 0.92), 3)

    proposed_content = f"step_{step_id}: {observation[:160]}"
    if reflection_summary:
        proposed_content = f"{proposed_content} | reflect: {reflection_summary[:80]}"

    proposal = session.sdb.propose(
        session_id=session.session_id,
        step=step_id,
        proposer="ring1_agent",
        source_type=MemorySourceType.ASSISTANT,
        target_tier=MemoryTargetTier.TIER2,
        operation=MemoryOperation.STORE,
        payload={"content": proposed_content, "surprise": surprise_score, "layer": layer},
    )

    entry = session.tier2.observe(
        role="assistant",
        content=proposed_content,
        surprise=surprise_score,
        importance=min(1.0, surprise_score),
    )

    if entry is not None:
        verification = session.sdb.verify(
            proposal,
            verifier="ring1_surprise_gate",
            decision=VerificationDecision.ACCEPT,
            reason="surprise threshold crossed",
            checks={"surprise_score": surprise_score, "layer": layer},
        )
        session.sdb.commit(
            proposal,
            verification_result=verification,
            memory_entry_id=entry.id,
        )
        session._stored_entry_ids.append(entry.id)
        session.working_memory[layer] = proposed_content[:120]
        return [
            MemoryWriteEvent(
                layer=layer,
                surprise_score=surprise_score,
                decision="promote",
                proposed_content=proposed_content,
                verification_passed=True,
                target_tier="episodic",
                step_id=step_id,
            )
        ]

    if surprise_score >= session.tier2.config.surprise_threshold:
        decision = "quarantine"
        target_tier = "working"
    elif surprise_score >= 0.4:
        decision = "quarantine"
        target_tier = "working"
    else:
        decision = "reject"
        target_tier = "working"

    verification = session.sdb.verify(
        proposal,
        verifier="ring1_surprise_gate",
        decision=VerificationDecision.REJECT if decision == "reject" else VerificationDecision.QUARANTINE,
        reason="below surprise threshold or verification failed",
        checks={"surprise_score": surprise_score},
    )
    session.sdb.reject(proposal, verification_result=verification)

    return [
        MemoryWriteEvent(
            layer=layer,
            surprise_score=surprise_score,
            decision=decision,
            proposed_content=proposed_content,
            verification_passed=False,
            target_tier=target_tier,
            step_id=step_id,
        )
    ]


def run_mcts_think(
    *,
    observation: str,
    candidate_actions: list[str],
    memory_reads: list[MemoryReadEvent],
    config: dict[str, Any],
    rng: random.Random,
) -> MCTSStats:
    """Run stub MCTS where memory-augmented candidates receive higher priors."""
    mcts_cfg = config.get("mcts", {})
    reasoner = MCTSReasoner(
        c_puct=mcts_cfg.get("c_puct", 1.414),
        max_depth=mcts_cfg.get("max_depth", 6),
        max_simulations=mcts_cfg.get("max_simulations", 16),
    )
    budget = mcts_cfg.get("simulation_budget", 12)

    memory_terms = _memory_terms(memory_reads)
    priors = [_action_prior(action, memory_terms, rng) for action in candidate_actions]
    root = reasoner.create_root(observation)
    children = reasoner.expand(root, candidate_actions, priors=priors)

    nodes_using_memory = 0
    value_estimates: list[float] = []
    for child in children:
        value = _evaluate_action(child.state, memory_terms, rng)
        value_estimates.append(value)
        path = [root, child]
        reasoner.backup_path(path, value)
        if _action_uses_memory(child.state, memory_terms):
            nodes_using_memory += 1

    rollout_path = reasoner.rollout_path(root, budget=min(budget, len(children) or 1))
    if len(rollout_path) > 1:
        selected = reasoner.best_child(root)
    elif children:
        selected = max(children, key=lambda node: node.visits)
    else:
        selected = MCTSNode(state=candidate_actions[0], parent_id=root.id)

    memory_consulted = bool(memory_reads) or bool(memory_terms)
    if memory_consulted and nodes_using_memory == 0 and memory_terms:
        nodes_using_memory = 1

    return MCTSStats(
        num_expansions=len(children),
        selected_action=selected.state,
        memory_consulted=memory_consulted,
        nodes_using_memory=nodes_using_memory,
        value_estimates=value_estimates[:5],
    )


def memory_delta_summary(session: Ring1MemorySession, domain: str) -> str:
    stats = session.tier2.stats()
    return (
        f"Active memory for {domain}: {stats['episodic_entries']} episodic entries, "
        f"{len(session.working_memory)} working layers, "
        f"{session.sdb.replay_events().__len__()} SDB audit events."
    )


def _content_tokens(content: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", content.lower()) if len(token) > 3}


def _memory_terms(reads: list[MemoryReadEvent]) -> set[str]:
    terms: set[str] = set()
    for read in reads:
        terms.update(_content_tokens(read.content))
    return terms


def _action_prior(action: str, memory_terms: set[str], rng: random.Random) -> float:
    action_terms = _content_tokens(action)
    overlap = len(action_terms & memory_terms)
    base = 0.25 + 0.15 * overlap
    if overlap > 0:
        base += 0.35
    return min(0.95, base + rng.uniform(0.01, 0.08))


def _evaluate_action(action: str, memory_terms: set[str], rng: random.Random) -> float:
    overlap = len(_content_tokens(action) & memory_terms)
    raw = 0.35 + 0.2 * overlap + rng.uniform(-0.05, 0.15)
    return max(-1.0, min(1.0, math.tanh(raw)))


def _action_uses_memory(action: str, memory_terms: set[str]) -> bool:
    if not memory_terms:
        return False
    return bool(_content_tokens(action) & memory_terms)
