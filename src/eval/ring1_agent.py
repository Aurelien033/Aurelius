"""Ring 1 integrated agent — Observe → Think(MCTS) → Act → Reflect with live AMC memory.

Tranche 2 replaces the dummy RNG memory path with AMCTier2Hook + SDB audit +
memory-influenced MCTS action selection.
"""

from __future__ import annotations

import random
import uuid
from datetime import UTC, datetime
from typing import Any

from src.eval.ring1_amc_glue import (
    Ring1MemorySession,
    memory_delta_summary,
    read_memory_for_think,
    run_mcts_think,
    write_memory_after_step,
)
from src.eval.ring1_dummy_agent import (
    ACTION_TEMPLATES,
    DOMAINS,
    OBSERVATION_TEMPLATES,
    Ring1DummyAgent,
)
from src.eval.ring1_trace_logger import Ring1Trace, Ring1TraceLogger, TraceStep

MULTIHOP_FACTS = {
    "entity_3": "Marie Curie",
    "entity_10": "Albert Einstein",
    "entity_7": "Rosalind Franklin",
    "entity_15": "Isaac Newton",
}


class Ring1Agent:
    """Memory-integrated Ring 1 agent using live Tier-2 + SDB surfaces."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._length_sampler = Ring1DummyAgent(config)

    def sample_trace_length(self, rng: random.Random, max_steps: int = 12) -> int:
        return self._length_sampler.sample_trace_length(rng, max_steps=max_steps)

    def generate_trace(
        self,
        *,
        seed: int,
        logger: Ring1TraceLogger,
        max_steps: int = 12,
        domain: str | None = None,
        tags: list[str] | None = None,
    ) -> Ring1Trace:
        rng = random.Random(seed)
        chosen_domain = domain or rng.choice(DOMAINS)
        num_steps = self.sample_trace_length(rng, max_steps=max_steps)
        session = Ring1MemorySession.from_config(self.config, session_id=str(uuid.uuid4()))
        steps: list[TraceStep] = []
        trace_tags = list(tags or ["integration:amc_tier2"])
        prior_observations: list[str] = []

        for step_id in range(1, num_steps + 1):
            step = self._run_step(
                step_id=step_id,
                domain=chosen_domain,
                rng=rng,
                session=session,
                prior_observations=prior_observations,
                is_final=step_id == num_steps,
            )
            prior_observations.append(step.observation)
            steps.append(step)

        if not any(tag.startswith("failure_mode:") for tag in trace_tags):
            if any(
                read.influenced_action
                for step in steps
                for read in step.memory_reads
                if read.source == "episodic"
            ):
                trace_tags.append("failure_mode:successful_retrieval")
            else:
                trace_tags.append("failure_mode:high_surprise_dead_end")

        cross_step = _count_cross_step_memory_use(steps)
        trace_tags.append(f"cross_step_reads:{cross_step}")

        final_outcome = "success" if cross_step >= 1 and rng.random() > 0.12 else "partial"
        return logger.build_trace(
            seed=seed,
            domain=chosen_domain,
            steps=steps,
            final_outcome=final_outcome,
            tags=trace_tags,
        )

    def _run_step(
        self,
        *,
        step_id: int,
        domain: str,
        rng: random.Random,
        session: Ring1MemorySession,
        prior_observations: list[str],
        is_final: bool,
    ) -> TraceStep:
        observation = self._make_observation(domain, step_id, rng, prior_observations)

        memory_reads = read_memory_for_think(
            session,
            observation=observation,
            step_id=step_id,
            rng=rng,
        )

        candidate_actions = self._candidate_actions(domain, step_id, observation, memory_reads, rng)
        mcts = run_mcts_think(
            observation=observation,
            candidate_actions=candidate_actions,
            memory_reads=memory_reads,
            config=self.config,
            rng=rng,
        )

        action_failed = rng.random() < 0.12
        action = {
            "tool": domain,
            "parameters": {
                "step_id": step_id,
                "selected_action": mcts.selected_action,
                "memory_informed": any(read.influenced_action for read in memory_reads),
            },
            "response": {
                "status": "error" if action_failed else "ok",
                "payload": self._action_payload(domain, mcts.selected_action, memory_reads),
            },
        }

        reflection = self._reflect(
            step_id=step_id,
            observation=observation,
            action=action,
            memory_reads=memory_reads,
            is_final=is_final,
            rng=rng,
        )

        memory_writes = write_memory_after_step(
            session,
            step_id=step_id,
            observation=observation,
            reflection_summary=reflection["summary"],
            action_failed=action_failed,
            surprise_override=0.88 if step_id == 1 else None,
            rng=rng,
        )

        return TraceStep(
            step_id=step_id,
            observation=observation,
            mcts=mcts,
            memory_reads=memory_reads,
            memory_writes=memory_writes,
            action=action,
            reflection=reflection,
            memory_delta_summary=memory_delta_summary(session, domain),
            timestamp=datetime.now(UTC).isoformat(),
        )

    def _make_observation(
        self,
        domain: str,
        step_id: int,
        rng: random.Random,
        prior_observations: list[str],
    ) -> str:
        templates = OBSERVATION_TEMPLATES[domain]
        template = templates[(step_id - 1) % len(templates)]
        entity_a = f"entity_{rng.randint(1, 9)}"
        entity_b = f"entity_{rng.randint(10, 19)}"
        observation = template.format(
            entity_a=entity_a,
            entity_b=entity_b,
            source_id=rng.randint(100, 999),
            doc_id=f"doc_{step_id}",
            year=2000 + rng.randint(0, 25),
            query=f"query_{step_id}",
            refined_query=f"refined_{entity_a}",
            subtask=f"subtask_{step_id}",
            budget=rng.randint(1, 5),
            resource=f"resource_{rng.randint(1, 3)}",
            step_hint=step_id + 1,
            metric="completion_rate",
            threshold=round(rng.uniform(0.5, 0.9), 2),
            entity=entity_a,
            action=f"action_{step_id}",
        )
        if domain == "multi_hop_qa" and step_id > 1 and prior_observations:
            fact_key = entity_a if entity_a in MULTIHOP_FACTS else entity_b
            if fact_key in MULTIHOP_FACTS:
                observation = (
                    f"{observation} Prior step stored: {fact_key}={MULTIHOP_FACTS[fact_key]}."
                )
        return observation

    def _candidate_actions(
        self,
        domain: str,
        step_id: int,
        observation: str,
        memory_reads: list[Any],
        rng: random.Random,
    ) -> list[str]:
        template = ACTION_TEMPLATES[domain]
        base = [
            template.format(entity=f"entity_{step_id}", query=f"query_{step_id}", action=f"action_{step_id}"),
            template.format(
                entity=f"entity_{step_id + 1}",
                query=f"query_{step_id}_alt",
                action=f"action_{step_id}_alt",
            ),
            f"clarify({observation[:40]})",
        ]
        for read in memory_reads:
            if read.influenced_action and read.source == "episodic":
                for token in read.content.split():
                    if token.startswith("entity_"):
                        base.append(template.format(entity=token, query=token, action=token))
        if domain == "multi_hop_qa":
            for key, name in MULTIHOP_FACTS.items():
                if key in observation:
                    base.append(f"lookup({key}) -> {name}")
        rng.shuffle(base)
        return base[:5]

    def _action_payload(
        self,
        domain: str,
        selected_action: str,
        memory_reads: list[Any],
    ) -> str:
        influenced = [read.content[:60] for read in memory_reads if read.influenced_action]
        if influenced:
            return f"Executed {selected_action} using memory: {influenced[0]}"
        return f"Executed {selected_action} for {domain}"

    def _reflect(
        self,
        *,
        step_id: int,
        observation: str,
        action: dict[str, Any],
        memory_reads: list[Any],
        is_final: bool,
        rng: random.Random,
    ) -> dict[str, Any]:
        used_memory = any(read.influenced_action for read in memory_reads)
        failed = action["response"]["status"] == "error"
        if failed:
            decision = "backtrack"
            summary = f"Step {step_id} failed; insight: store contradiction from {observation[:50]}"
        elif used_memory:
            decision = "continue"
            summary = f"Step {step_id} succeeded using episodic recall; reusable insight captured."
        elif is_final:
            decision = "continue"
            summary = f"Final step {step_id} completed."
        else:
            decision = "continue" if rng.random() > 0.15 else "request_clarification"
            summary = f"Step {step_id} observation processed."
        return {
            "summary": summary,
            "decision": decision,
            "confidence": round(rng.uniform(0.55, 0.95), 3),
            "memory_used": used_memory,
        }


def _count_cross_step_memory_use(steps: list[TraceStep]) -> int:
    count = 0
    for step in steps:
        if step.step_id <= 1:
            continue
        for read in step.memory_reads:
            if read.source == "episodic" and read.influenced_action:
                count += 1
    return count
