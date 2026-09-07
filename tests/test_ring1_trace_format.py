"""Tests for Ring 1 trace format compliance (Trace Specification v0.1)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from src.eval.ring1_dummy_agent import Ring1DummyAgent
from src.eval.ring1_trace_logger import (
    MCTSStats,
    MemoryReadEvent,
    MemoryWriteEvent,
    Ring1Trace,
    Ring1TraceLogger,
    TraceStep,
    load_traces,
    validate_trace,
)


def _minimal_config() -> dict:
    return {
        "tranche": 1,
        "seeds": {"base_seed": 1},
        "model": {"checkpoint_path": "checkpoints/aurelius-1.3b"},
        "domains": ["multi_hop_qa"],
        "trace_length_distribution": {"4": 1.0},
        "layer_range": [8, 16],
        "surprise_threshold": 0.5,
    }


def _sample_step(step_id: int = 1, influenced_read: bool = True) -> TraceStep:
    return TraceStep(
        step_id=step_id,
        observation=f"observation-{step_id}",
        mcts=MCTSStats(
            num_expansions=8,
            selected_action="lookup(entity_a)",
            memory_consulted=True,
            nodes_using_memory=3,
            value_estimates=[0.5, 0.7],
        ),
        memory_reads=[
            MemoryReadEvent(
                source="episodic",
                content="prior fact",
                influenced_action=influenced_read,
                layer=8,
            )
        ],
        memory_writes=[
            MemoryWriteEvent(
                layer=8,
                surprise_score=0.9,
                decision="promote",
                proposed_content="stored fact",
                verification_passed=True,
                target_tier="episodic",
            )
        ],
        action={"tool": "lookup", "parameters": {}, "response": {"status": "ok"}},
        reflection={"summary": "ok", "decision": "continue", "confidence": 0.8},
        memory_delta_summary="1 episodic entry",
        timestamp="2026-05-31T00:00:00+00:00",
    )


def test_validate_trace_requires_four_to_twelve_steps() -> None:
    config = _minimal_config()
    logger = Ring1TraceLogger(Path("/tmp/unused"), config)
    agent = Ring1DummyAgent(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        logger = Ring1TraceLogger(output_dir, config)
        trace = agent.generate_trace(seed=99, logger=logger, max_steps=12)
        validate_trace(trace)

        bad_trace = Ring1Trace(
            trace_id=trace.trace_id,
            model_checkpoint_sha256=trace.model_checkpoint_sha256,
            config_hash=trace.config_hash,
            seed=trace.seed,
            total_steps=3,
            domain=trace.domain,
            steps=trace.steps[:3],
            final_outcome=trace.final_outcome,
            metadata=trace.metadata,
        )
        with pytest.raises(ValueError, match="total_steps must be 4-12"):
            validate_trace(bad_trace)


def test_trace_logger_writes_jsonl_and_sidecar() -> None:
    config = _minimal_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        logger = Ring1TraceLogger(output_dir, config)
        agent = Ring1DummyAgent(config)
        trace = agent.generate_trace(seed=7, logger=logger)
        logger.write_trace(trace)

        jsonl_path = output_dir / "traces.jsonl"
        assert jsonl_path.exists()
        traces = load_traces(jsonl_path)
        assert len(traces) == 1
        assert traces[0]["trace_id"] == trace.trace_id
        assert traces[0]["config_hash"] == logger.config_hash
        assert "git_sha" in traces[0]["metadata"]

        sidecar = output_dir / "sidecars" / f"{trace.trace_id}_memory_events.ndjson"
        assert sidecar.exists()
        lines = [line for line in sidecar.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert lines
        first_event = json.loads(lines[0])
        assert first_event["event_type"] in {"memory_read", "memory_write"}
        assert first_event["trace_id"] == trace.trace_id


def test_generated_trace_has_required_step_fields() -> None:
    config = _minimal_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Ring1TraceLogger(Path(tmpdir), config)
        trace = Ring1DummyAgent(config).generate_trace(seed=11, logger=logger)
        payload = trace.to_dict()
        for step in payload["steps"]:
            assert "observation" in step
            assert "mcts" in step
            assert "memory_reads" in step
            assert "memory_writes" in step
            assert "action" in step
            assert "reflection" in step
            assert step["reflection"]["decision"] in {
                "continue",
                "backtrack",
                "request_clarification",
            }


def test_reproducibility_same_seed_same_length() -> None:
    config = _minimal_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Ring1TraceLogger(Path(tmpdir), config)
        agent = Ring1DummyAgent(config)
        trace_a = agent.generate_trace(seed=123, logger=logger)
        trace_b = agent.generate_trace(seed=123, logger=logger)
        assert trace_a.total_steps == trace_b.total_steps
        assert trace_a.domain == trace_b.domain


def test_config_file_loads() -> None:
    config_path = Path("configs/ring1_tranche1.yaml")
    assert config_path.exists()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert config["tranche"] == 1
    assert config["collection"]["default_num_traces"] >= 50


def test_manual_trace_validation_passes() -> None:
    config = _minimal_config()
    logger = Ring1TraceLogger(Path("/tmp/unused"), config)
    steps = [_sample_step(step_id=i, influenced_read=(i >= 2)) for i in range(1, 5)]
    trace = logger.build_trace(
        seed=1,
        domain="multi_hop_qa",
        steps=steps,
        final_outcome="success",
        tags=["failure_mode:successful_retrieval"],
    )
    validate_trace(trace)
