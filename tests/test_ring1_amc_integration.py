"""Tranche 2 integration tests — live AMC memory in Ring 1 traces."""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from src.eval.ring1_agent import Ring1Agent
from src.eval.ring1_amc_glue import (
    Ring1MemorySession,
    read_memory_for_think,
    write_memory_after_step,
)
from src.eval.ring1_trace_logger import Ring1TraceLogger, load_traces, validate_trace


def _tranche2_config() -> dict:
    return {
        "tranche": 2,
        "agent": {"type": "integrated"},
        "seeds": {"base_seed": 1},
        "model": {"checkpoint_path": "checkpoints/aurelius-1.3b"},
        "domains": ["multi_hop_qa"],
        "trace_length_distribution": {"4": 1.0},
        "layer_range": [8, 16, 24],
        "surprise_threshold": 0.5,
        "mcts": {"simulation_budget": 8, "max_simulations": 8},
    }


def test_tier2_write_and_read_cross_step() -> None:
    import random

    rng = random.Random(7)
    session = Ring1MemorySession.from_config(_tranche2_config(), session_id="test-session")

    writes = write_memory_after_step(
        session,
        step_id=1,
        observation="Question about entity_3",
        reflection_summary="insight captured",
        action_failed=False,
        surprise_override=0.9,
        rng=rng,
    )
    assert writes[0].decision == "promote"

    reads = read_memory_for_think(session, observation="Follow-up about entity_3", step_id=3, rng=rng)
    assert any(read.source == "episodic" for read in reads)
    assert any(read.influenced_action for read in reads)


def test_integrated_agent_trace_validates() -> None:
    config = _tranche2_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Ring1TraceLogger(Path(tmpdir), config)
        trace = Ring1Agent(config).generate_trace(seed=42, logger=logger, max_steps=12)
        validate_trace(trace)
        logger.write_trace(trace)

        payload = load_traces(Path(tmpdir) / "traces.jsonl")[0]
        assert payload["metadata"]["tags"]
        assert any(
            read.get("influenced_action")
            for step in payload["steps"]
            for read in step["memory_reads"]
        )


def test_integrated_traces_show_mcts_memory_consultation() -> None:
    config = _tranche2_config()
    logger = Ring1TraceLogger(Path("/tmp/unused"), config)
    trace = Ring1Agent(config).generate_trace(seed=99, logger=logger)
    consulted = [step for step in trace.steps if step.mcts.memory_consulted]
    assert consulted
    assert any(step.mcts.num_expansions >= 1 for step in trace.steps)


def test_sdb_audit_events_recorded() -> None:
    import random

    rng = random.Random(1)
    session = Ring1MemorySession.from_config(_tranche2_config(), session_id="audit-session")
    write_memory_after_step(
        session,
        step_id=1,
        observation="store this",
        reflection_summary="insight",
        action_failed=False,
        surprise_override=0.91,
        rng=rng,
    )
    events = session.sdb.replay_events()
    event_types = {event.event_type for event in events}
    assert "proposed" in event_types
    assert "verified" in event_types
    assert "committed" in event_types


def test_tranche2_config_exists() -> None:
    path = Path("configs/ring1_tranche2.yaml")
    assert path.exists()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["tranche"] == 2
    assert config["agent"]["type"] == "integrated"
    assert config["collection"]["default_num_traces"] >= 100


def test_batch_integrated_traces_have_cross_step_memory() -> None:
    config = _tranche2_config()
    logger = Ring1TraceLogger(Path("/tmp/unused"), config)
    agent = Ring1Agent(config)
    cross_step_total = 0
    for seed in range(20):
        trace = agent.generate_trace(seed=1000 + seed, logger=logger)
        validate_trace(trace)
        for step in trace.steps:
            if step.step_id > 1:
                cross_step_total += sum(
                    1 for read in step.memory_reads if read.source == "episodic" and read.influenced_action
                )
    assert cross_step_total >= 15
