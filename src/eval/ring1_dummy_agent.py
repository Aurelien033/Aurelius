"""Ring 1 dummy agent — minimal Observe → Think → Act → Reflect loop.

Generates valid 4–12 step traces with dummy MCTS and memory events.
No real tool calling, MCTS, or AMC integration in Tranche 1.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from typing import Any

from src.eval.ring1_trace_logger import (
    MCTSStats,
    MemoryReadEvent,
    MemoryWriteEvent,
    Ring1Trace,
    Ring1TraceLogger,
    TraceStep,
)

DOMAINS = ["multi_hop_qa", "tool_use", "closed_world_planning"]

OBSERVATION_TEMPLATES = {
    "multi_hop_qa": [
        "Question: Which scientist discovered {entity_a} before {entity_b}?",
        "Follow-up: The prior answer conflicts with source {source_id}.",
        "New evidence: Document {doc_id} mentions {entity_a} in {year}.",
    ],
    "tool_use": [
        "Task: Search for {query} and summarize findings.",
        "Tool returned partial results for {query}; status=timeout.",
        "Retry request: refine query to {refined_query}.",
    ],
    "closed_world_planning": [
        "Goal: Complete subtask {subtask} within budget {budget}.",
        "Constraint update: {resource} is unavailable until step {step_hint}.",
        "Progress check: {metric} is below threshold {threshold}.",
    ],
}

ACTION_TEMPLATES = {
    "multi_hop_qa": "lookup({entity})",
    "tool_use": "web_search(query={query})",
    "closed_world_planning": "plan_step(action={action})",
}


class Ring1DummyAgent:
    """Stub agent that produces spec-compliant traces for infrastructure testing."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.length_distribution: dict[str, float] = config.get(
            "trace_length_distribution",
            {"4": 0.15, "6-8": 0.55, "10-12": 0.30},
        )
        self.layer_range: list[int] = config.get("layer_range", [8, 12, 16, 20, 24])
        self.surprise_threshold: float = config.get("surprise_threshold", 0.65)

    def sample_trace_length(self, rng: random.Random, max_steps: int = 12) -> int:
        bucket = rng.choices(
            population=list(self.length_distribution.keys()),
            weights=list(self.length_distribution.values()),
            k=1,
        )[0]
        if bucket == "4":
            length = 4
        elif bucket == "6-8":
            length = rng.randint(6, 8)
        else:
            length = rng.randint(10, 12)
        return min(max(length, 4), max_steps)

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
        memory_store: list[dict[str, Any]] = []
        steps: list[TraceStep] = []
        trace_tags = list(tags or [])
        saw_influenced_read = False

        for step_id in range(1, num_steps + 1):
            force_influenced_read = (
                not saw_influenced_read and step_id == num_steps and len(memory_store) > 0
            )
            step = self._run_step(
                step_id=step_id,
                domain=chosen_domain,
                rng=rng,
                memory_store=memory_store,
                is_final=step_id == num_steps,
                force_influenced_read=force_influenced_read,
            )
            if any(read.influenced_action for read in step.memory_reads):
                saw_influenced_read = True
            steps.append(step)

        if not any(tag.startswith("failure_mode:") for tag in trace_tags):
            if memory_store and rng.random() < 0.4:
                trace_tags.append("failure_mode:high_value_unused_promotion")
            elif rng.random() < 0.3:
                trace_tags.append("failure_mode:successful_retrieval")

        final_outcome = "success" if rng.random() > 0.15 else "partial"
        trace = logger.build_trace(
            seed=seed,
            domain=chosen_domain,
            steps=steps,
            final_outcome=final_outcome,
            tags=trace_tags,
        )
        return trace

    def _run_step(
        self,
        *,
        step_id: int,
        domain: str,
        rng: random.Random,
        memory_store: list[dict[str, Any]],
        is_final: bool,
        force_influenced_read: bool = False,
    ) -> TraceStep:
        observation = self._make_observation(domain, step_id, rng)
        memory_reads = self._make_memory_reads(
            step_id,
            memory_store,
            rng,
            force_influenced_read=force_influenced_read,
        )
        memory_consulted = bool(memory_reads) or rng.random() < 0.35
        mcts = MCTSStats(
            num_expansions=rng.randint(4, 24),
            selected_action=self._make_action(domain, step_id, rng),
            memory_consulted=memory_consulted,
            nodes_using_memory=rng.randint(0, 8) if memory_consulted else 0,
            value_estimates=[round(rng.uniform(0.2, 0.95), 3) for _ in range(3)],
        )
        memory_writes = self._make_memory_writes(step_id, observation, rng, memory_store)
        action = {
            "tool": domain,
            "parameters": {"step_id": step_id, "selected_action": mcts.selected_action},
            "response": {
                "status": "ok" if rng.random() > 0.1 else "error",
                "payload": f"dummy-response-for-step-{step_id}",
            },
        }
        reflection = {
            "summary": f"Reflected on step {step_id} outcome.",
            "decision": "continue" if not is_final or rng.random() > 0.2 else "backtrack",
            "confidence": round(rng.uniform(0.4, 0.95), 3),
        }
        memory_delta_summary = (
            f"Active memory contains {len(memory_store)} entries relevant to {domain}."
        )
        return TraceStep(
            step_id=step_id,
            observation=observation,
            mcts=mcts,
            memory_reads=memory_reads,
            memory_writes=memory_writes,
            action=action,
            reflection=reflection,
            memory_delta_summary=memory_delta_summary,
            timestamp=datetime.now(UTC).isoformat(),
        )

    def _make_observation(self, domain: str, step_id: int, rng: random.Random) -> str:
        templates = OBSERVATION_TEMPLATES[domain]
        template = templates[(step_id - 1) % len(templates)]
        return template.format(
            entity_a=f"entity_{rng.randint(1, 9)}",
            entity_b=f"entity_{rng.randint(10, 19)}",
            source_id=rng.randint(100, 999),
            doc_id=f"doc_{step_id}",
            year=2000 + rng.randint(0, 25),
            query=f"query_{step_id}",
            refined_query=f"refined_query_{step_id}",
            subtask=f"subtask_{step_id}",
            budget=rng.randint(1, 5),
            resource=f"resource_{rng.randint(1, 3)}",
            step_hint=step_id + 1,
            metric="completion_rate",
            threshold=round(rng.uniform(0.5, 0.9), 2),
            entity=f"entity_{step_id}",
            action=f"action_{step_id}",
        )

    def _make_action(self, domain: str, step_id: int, rng: random.Random) -> str:
        template = ACTION_TEMPLATES[domain]
        if domain == "multi_hop_qa":
            return template.format(entity=f"entity_{step_id}")
        if domain == "tool_use":
            return template.format(query=f"query_{step_id}")
        return template.format(action=f"action_{step_id}")

    def _make_memory_reads(
        self,
        step_id: int,
        memory_store: list[dict[str, Any]],
        rng: random.Random,
        force_influenced_read: bool = False,
    ) -> list[MemoryReadEvent]:
        if step_id == 1 or not memory_store:
            if step_id > 1 and rng.random() < 0.25:
                return [
                    MemoryReadEvent(
                        source="working",
                        content="No prior episodic memory; using working context only.",
                        influenced_action=False,
                        layer=rng.choice(self.layer_range),
                        step_id=step_id,
                    )
                ]
            return []

        candidates = memory_store[-min(3, len(memory_store)) :]
        chosen = rng.choice(candidates)
        influenced = force_influenced_read or (step_id > 2 and rng.random() > 0.2)
        return [
            MemoryReadEvent(
                source=chosen["tier"],
                content=chosen["content"],
                influenced_action=influenced,
                layer=chosen["layer"],
                step_id=step_id,
            )
        ]

    def _make_memory_writes(
        self,
        step_id: int,
        observation: str,
        rng: random.Random,
        memory_store: list[dict[str, Any]],
    ) -> list[MemoryWriteEvent]:
        if step_id == 1:
            surprise_score = round(rng.uniform(0.75, 0.98), 3)
            decision = "promote"
            target_tier = "episodic"
            verification_passed = True
        else:
            surprise_score = round(rng.uniform(0.2, 0.98), 3)
            if surprise_score >= self.surprise_threshold:
                decision = "promote"
                target_tier = "episodic"
                verification_passed = rng.random() > 0.05
            elif surprise_score >= 0.4:
                decision = "quarantine"
                target_tier = "working"
                verification_passed = False
            else:
                decision = "reject"
                target_tier = "working"
                verification_passed = False

        layer = rng.choice(self.layer_range)
        proposed_content = f"step_{step_id}: {observation[:120]}"
        event = MemoryWriteEvent(
            layer=layer,
            surprise_score=surprise_score,
            decision=decision,
            proposed_content=proposed_content,
            verification_passed=verification_passed,
            target_tier=target_tier,
            step_id=step_id,
        )

        if decision == "promote" and verification_passed:
            memory_store.append(
                {
                    "tier": target_tier,
                    "content": proposed_content,
                    "layer": layer,
                    "step_id": step_id,
                }
            )
        return [event]
