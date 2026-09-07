"""Tests for SLR end-to-end agent integration (T26)."""

from __future__ import annotations

import pytest

from src.agent.react_loop import ReActLoop
from src.agent.slr_integration import prepare_slr_recall_context, resolve_recall_context
from src.memory.amc_tier2 import AMCTier2Hook
from src.memory.amc_tier3 import AMCTier3Hook
from src.memory.sdb_runtime import SDBMemoryRuntime
from src.reasoning.stochastic_latent_recall import (
    SLRDisabledError,
    SLRQuery,
    default_slr_config,
    generate_slr_candidates,
)


def _query(query_id: str = "q-1") -> SLRQuery:
    return SLRQuery(
        query_id=query_id,
        query_text="What architecture does the user prefer?",
        memory_scope=("tier2", "tier3"),
        baseline_candidate_ids=(),
    )


def test_slr_disabled_by_default_raises() -> None:
    cfg = default_slr_config()
    assert cfg.enabled is False
    with pytest.raises(SLRDisabledError):
        generate_slr_candidates(config=cfg, query=_query())


def test_slr_creates_k_candidates() -> None:
    cfg = default_slr_config(enabled=True, k=4, replay_seed=7, noise_sigma=0.1)
    candidates = generate_slr_candidates(config=cfg, query=_query())
    assert len(candidates) == 4


def test_slr_deterministic_same_seed() -> None:
    cfg = default_slr_config(enabled=True, k=3, replay_seed=99, noise_sigma=0.2)
    first = generate_slr_candidates(config=cfg, query=_query("same"))
    second = generate_slr_candidates(config=cfg, query=_query("same"))
    assert [candidate.perturbation for candidate in first] == [
        candidate.perturbation for candidate in second
    ]


def test_slr_different_seed_different_candidates() -> None:
    q = _query("diff")
    cfg_a = default_slr_config(enabled=True, k=3, replay_seed=1, noise_sigma=0.2)
    cfg_b = default_slr_config(enabled=True, k=3, replay_seed=2, noise_sigma=0.2)
    a = generate_slr_candidates(config=cfg_a, query=q)
    b = generate_slr_candidates(config=cfg_b, query=q)
    assert [c.perturbation for c in a] != [c.perturbation for c in b]


def test_slr_records_to_sdb_log() -> None:
    runtime = SDBMemoryRuntime()
    cfg = default_slr_config(enabled=True, k=2, replay_seed=11, noise_sigma=0.0)
    context = prepare_slr_recall_context(
        config=cfg,
        query_text="Prefer MLA on even layers.",
        query_id="sess:step:1",
        sdb_runtime=runtime,
        session_id="sess",
        step_idx=1,
    )
    assert context is not None
    events = runtime.replay_events()
    assert len(events) == 1
    assert events[0].event_type == "slr_selection"
    assert events[0].metadata["session_id"] == "sess"


def test_slr_retrieval_uses_recall_keys() -> None:
    tier3 = AMCTier3Hook()
    tier2 = AMCTier2Hook()
    tier2.observe("user", "User prefers concise answers.", surprise=0.9)
    cfg = default_slr_config(enabled=True, k=1, replay_seed=5, noise_sigma=0.0)
    trial_query = _query("retrieve")
    candidates = generate_slr_candidates(config=cfg, query=trial_query)
    recall_key = candidates[0].recall_keys[0]
    tier3.promote(key=recall_key, value="Promoted via SLR recall key.", confidence=0.95)
    lines, entry_ids = resolve_recall_context(
        recall_keys=candidates[0].recall_keys,
        query_text=trial_query.query_text,
        tier2_hook=tier2,
        tier3_hook=tier3,
    )
    assert recall_key in entry_ids
    assert any("Promoted via SLR" in line for line in lines)
    assert any("concise" in line for line in lines)


def test_react_loop_injects_slr_context_when_enabled() -> None:
    messages_seen: list[list[dict]] = []

    def _generate(messages: list[dict]) -> str:
        messages_seen.append(messages)
        if len(messages_seen) == 1:
            return "Thinking..."
        return "<final_answer>done</final_answer>"

    loop = ReActLoop(
        generate_fn=_generate,
        tool_registry={},
        max_steps=2,
        slr_config=default_slr_config(enabled=True, k=2, replay_seed=3, noise_sigma=0.1),
        session_id="test-session",
    )
    trace = loop.run("Explain AMC memory tiers.", system_prompt="You are helpful.")
    assert trace.status == "success"
    assert len(messages_seen) >= 2
    second_turn = messages_seen[1]
    assert any("SLR stochastic recall" in msg.get("content", "") for msg in second_turn)
