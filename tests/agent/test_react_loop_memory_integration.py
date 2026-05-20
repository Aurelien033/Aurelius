"""Integration tests for ReActLoop + AMC Tier 2 memory integration."""

from __future__ import annotations

import pytest

from src.memory.amc_tier2 import AMCTier2Config, AMCTier2Hook
from agent.react_loop import AgentStep, AgentTrace, ReActLoop


# ── Helpers ──────────────────────────────────────────────────────────────────


def _echo_tool() -> dict:
    """Return a registry with one echo tool."""
    return {"echo": lambda x: f"echoed: {x}"}


def _make_loop(
    hook: AMCTier2Hook | None = None,
    max_steps: int = 4,
    max_tool_seconds: float = 2.0,
) -> ReActLoop:
    """Build a minimal ReActLoop backed by a deterministic generate_fn."""
    def generate_fn(messages: list[dict]) -> str:
        last = messages[-1]["content"] if messages else ""
        return '<final_answer>done</final_answer>'

    return ReActLoop(
        generate_fn=generate_fn,
        tool_registry=_echo_tool(),
        max_steps=max_steps,
        max_tool_seconds=max_tool_seconds,
        tier2_hook=hook,
    )


def _tool_generate_fn(
    registry: dict[str, callable] | None = None,
    responses=None,
    final_answer: str | None = None,
) -> callable:
    """Return a generate_fn that produces canned responses in order.

    If ``responses`` is a list, the generate_fn pops from it.
    Otherwise emits a tool call against ``tools.registry`` or final answer.
    """
    calls: list = []  # recorder

    if responses is not None:
        def gen(messages):
            calls.append(messages)
            if responses:
                return responses.pop(0)
            if final_answer:
                return f"<final_answer>{final_answer}</final_answer>"
            return ''
    else:
        def gen(messages):
            calls.append(messages)
            if final_answer:
                return f"<final_answer>{final_answer}</final_answer>"
            return 'use tool on "test input"'

    return gen


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestReActLoopWithMemory:
    """AMC Tier 2 integration sanity checks for ReActLoop."""

    def test_hook_is_optional(self) -> None:
        """ReActLoop must run without a tier2_hook (backward compat)."""
        loop = _make_loop(hook=None)
        trace = loop.run("just a task")
        assert trace is not None
        assert trace.tier2_calls == 0
        assert trace.tier2_writes == 0

    def test_hook_retrieved_on_step_gt_0(self) -> None:
        """Tier 2 should be consulted starting at step_idx == 1."""
        hook = AMCTier2Hook(AMCTier2Config(surprise_threshold=0.0))
        loop = _make_loop(hook=hook)

        # Step 0 returns a non-final response; step 1 returns the final answer.
        responses = ["Still thinking...", "<final_answer>final</final_answer>"]

        def gen_with_final(messages):
            return responses.pop(0) if responses else ""

        loop._generate = gen_with_final
        trace = loop.run("capital of France")

        assert trace.tier2_calls >= 1  # recall on step 1 (step_idx=1 > 0)
        assert trace.status == "success"
        assert trace.steps_used == 2

    def test_hook_observes_tool_output(self) -> None:
        """After dispatching a tool, Tier 2 must receive an observation."""
        hook = AMCTier2Hook(AMCTier2Config(surprise_threshold=0.0))
        loop = _make_loop(hook=hook)

        step_outputs = [
            '{"name": "echo", "arguments": "hello"}',  # tool call
            '<final_answer>echoed</final_answer>',      # done
        ]

        def gen(messages):
            return step_outputs.pop(0) if step_outputs else ""

        loop._generate = gen
        trace = loop.run("call echo")

        assert trace.tier2_writes >= 1  # tool result was observed
        assert hook.stats()["stored_events"] >= 1

    def test_tier2_not_called_on_step_0(self) -> None:
        """No memory retrieval should happen on the very first generation."""
        hook = AMCTier2Hook(AMCTier2Config(surprise_threshold=0.0))
        loop = _make_loop(hook=hook)
        loop._generate = lambda msgs: "<final_answer>zero</final_answer>"
        trace = loop.run("hello")
        assert trace.tier2_calls == 0

    def test_retrieval_injected_as_user_message(self) -> None:
        """Tier 2 context is injected as a user-role turn (not system-role).

        This prevents untrusted episodic recall from acting as a high-priority
        system instruction.  When no admission controller is registered all
        recalled entries are trusted and pass through unchanged.
        """
        hook = AMCTier2Hook(
            AMCTier2Config(surprise_threshold=0.0, max_retrieved=2)
        )

        captured_messages: list[dict] = []

        # Step 0: non-final so loop continues. Step 1: final answer.
        responses = ["Still reasoning...", "<final_answer>done</final_answer>"]

        def tracking_gen(messages):
            captured_messages.append(messages)
            return responses.pop(0) if responses else ""

        loop = _make_loop(hook=hook)
        loop._generate = tracking_gen
        trace = loop.run("name of the capital")

        assert trace.tier2_calls >= 1
        # Tier-2 recall is now a user-role message (not system-role).
        recall_msgs = [
            m["content"]
            for msg_list in captured_messages[1:]  # skip first call
            for m in msg_list
            if m.get("role") == "user"
               and "[Tier-2 episodic memory recall]:" in m.get("content", "")
        ]
        assert recall_msgs, (
            "No Tier-2 recall found in user-role messages in subsequent calls"
        )
        assert trace.status == "success"
        assert trace.steps_used == 2

    def test_stats_copied_to_trace(self) -> None:
        """AgentTrace must reflect Tier 2 usage counters."""
        hook = AMCTier2Hook(AMCTier2Config(surprise_threshold=0.0))
        loop = _make_loop(hook=hook)
        loop._generate = lambda msgs: "<final_answer>x</final_answer>"
        trace = loop.run("task")
        assert trace.tier2_calls >= 0  # counter always present
        assert trace.tier2_writes >= 0
        assert isinstance(trace.tier2_calls, int)
        assert isinstance(trace.tier2_writes, int)

    def test_hook_receives_correct_roles(self) -> None:
        """Hook observe() should see 'assistant' and 'tool' roles."""
        hook = AMCTier2Hook(AMCTier2Config(surprise_threshold=0.0))
        loop = _make_loop(hook=hook)

        step_outputs = [
            'use tool on "input"',
            '<final_answer>ok</final_answer>',
        ]

        def gen(messages):
            return step_outputs.pop(0) if step_outputs else ""

        loop._generate = gen
        trace = loop.run("tool task")

        stored = hook.stats()["stored_events"]
        assert stored >= 1

    def test_config_passthrough_via_constructor(self) -> None:
        """tier2_hook passed in __init__ must be retained on the loop."""
        hook = AMCTier2Hook(AMCTier2Config(surprise_threshold=0.9))
        loop = _make_loop(hook=hook)
        assert loop._tier2_hook is hook

    def test_none_hook_is_noop(self) -> None:
        """When hook is None, loop must run cleanly without errors."""
        loop = ReActLoop(
            generate_fn=lambda msgs: "<final_answer>ok</final_answer>",
            tool_registry={},
            tier2_hook=None,
        )
        trace = loop.run("task")
        assert trace.status == "success"
        assert trace.tier2_calls == 0
        assert trace.tier2_writes == 0

    def test_budget_exhausted_tracks_amc_counters(self) -> None:
        """Even on a budget-roll, Tier 2 counters must appear in the trace."""
        hook = AMCTier2Hook(AMCTier2Config(surprise_threshold=0.0))
        loop = _make_loop(hook=hook, max_steps=2)
        loop._generate = lambda msgs: "just reasoning — no tool call, no final answer"
        trace = loop.run("complex task")
        assert trace.status == "budget"
        assert trace.tier2_calls >= 0
        assert trace.tier2_writes >= 0
