"""Ring 1 trace logger — canonical JSONL + memory_events.ndjson sidecar writer.

Implements the Trace Specification v0.1 format for Ring 1 Tranche 1.
Sidecar logging only; does not modify AMC or DreamBank source code.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VALID_DOMAINS = frozenset(
    {
        "multi_hop_qa",
        "tool_use",
        "closed_world_planning",
    }
)

VALID_MEMORY_SOURCES = frozenset({"working", "episodic", "constitutional"})
VALID_WRITE_DECISIONS = frozenset({"promote", "reject", "quarantine"})
VALID_TARGET_TIERS = frozenset({"working", "episodic", "constitutional"})
VALID_REFLECTION_DECISIONS = frozenset({"continue", "backtrack", "request_clarification"})
VALID_OUTCOMES = frozenset({"success", "failure", "partial"})


def compute_config_hash(config: dict[str, Any]) -> str:
    """Deterministic SHA-256 hash of a config dict."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_git_sha() -> str:
    """Return current git commit SHA, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 — git is a PATH-resolved system dependency
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def compute_checkpoint_sha256(checkpoint_path: str | None) -> str:
    """Hash checkpoint weights if present; otherwise return placeholder."""
    from src.eval.ring1_model_loader import checkpoint_sha256_for_path

    return checkpoint_sha256_for_path(checkpoint_path)


@dataclass
class MemoryReadEvent:
    source: str
    content: str
    influenced_action: bool
    layer: int | None = None
    step_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": self.source,
            "content": self.content,
            "influenced_action": self.influenced_action,
        }
        if self.layer is not None:
            payload["layer"] = self.layer
        return payload

    def to_sidecar(self, trace_id: str, timestamp: str) -> dict[str, Any]:
        return {
            "event_type": "memory_read",
            "trace_id": trace_id,
            "step_id": self.step_id,
            "timestamp": timestamp,
            **self.to_dict(),
        }


@dataclass
class MemoryWriteEvent:
    layer: int
    surprise_score: float
    decision: str
    proposed_content: str
    verification_passed: bool
    target_tier: str
    step_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "surprise_score": self.surprise_score,
            "decision": self.decision,
            "proposed_content": self.proposed_content,
            "verification_passed": self.verification_passed,
            "target_tier": self.target_tier,
        }

    def to_sidecar(self, trace_id: str, timestamp: str) -> dict[str, Any]:
        return {
            "event_type": "memory_write",
            "trace_id": trace_id,
            "step_id": self.step_id,
            "timestamp": timestamp,
            **self.to_dict(),
        }


@dataclass
class MCTSStats:
    num_expansions: int
    selected_action: str
    memory_consulted: bool
    nodes_using_memory: int
    value_estimates: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_expansions": self.num_expansions,
            "selected_action": self.selected_action,
            "memory_consulted": self.memory_consulted,
            "nodes_using_memory": self.nodes_using_memory,
            "value_estimates": self.value_estimates,
        }


@dataclass
class TraceStep:
    step_id: int
    observation: str
    mcts: MCTSStats
    memory_reads: list[MemoryReadEvent]
    memory_writes: list[MemoryWriteEvent]
    action: dict[str, Any]
    reflection: dict[str, Any]
    memory_delta_summary: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "observation": self.observation,
            "timestamp": self.timestamp,
            "mcts": self.mcts.to_dict(),
            "memory_reads": [event.to_dict() for event in self.memory_reads],
            "memory_writes": [event.to_dict() for event in self.memory_writes],
            "action": self.action,
            "reflection": self.reflection,
            "memory_delta_summary": self.memory_delta_summary,
        }


@dataclass
class Ring1Trace:
    trace_id: str
    model_checkpoint_sha256: str
    config_hash: str
    seed: int
    total_steps: int
    domain: str
    steps: list[TraceStep]
    final_outcome: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "model_checkpoint_sha256": self.model_checkpoint_sha256,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "total_steps": self.total_steps,
            "domain": self.domain,
            "steps": [step.to_dict() for step in self.steps],
            "final_outcome": self.final_outcome,
            "metadata": self.metadata,
        }


class Ring1TraceLogger:
    """Writes Ring 1 traces to JSONL plus per-trace memory event sidecars."""

    def __init__(
        self,
        output_dir: Path,
        config: dict[str, Any],
        checkpoint_path: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.config_hash = compute_config_hash(config)
        self.checkpoint_sha256 = compute_checkpoint_sha256(checkpoint_path)
        self.git_sha = get_git_sha()
        self.collection_timestamp = datetime.now(UTC).isoformat()
        self._jsonl_path = self.output_dir / "traces.jsonl"
        self._sidecar_dir = self.output_dir / "sidecars"
        self._sidecar_dir.mkdir(parents=True, exist_ok=True)

    def build_trace(
        self,
        *,
        seed: int,
        domain: str,
        steps: list[TraceStep],
        final_outcome: str,
        tags: list[str] | None = None,
        trace_id: str | None = None,
    ) -> Ring1Trace:
        metadata = {
            "git_sha": self.git_sha,
            "collection_timestamp": self.collection_timestamp,
            "tags": tags or [],
        }
        return Ring1Trace(
            trace_id=trace_id or str(uuid.uuid4()),
            model_checkpoint_sha256=self.checkpoint_sha256,
            config_hash=self.config_hash,
            seed=seed,
            total_steps=len(steps),
            domain=domain,
            steps=steps,
            final_outcome=final_outcome,
            metadata=metadata,
        )

    def write_trace(self, trace: Ring1Trace) -> Path:
        """Append one trace to JSONL and write its memory sidecar."""
        validate_trace(trace)
        with self._jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=True) + "\n")
        self._write_sidecar(trace)
        return self._jsonl_path

    def _write_sidecar(self, trace: Ring1Trace) -> Path:
        sidecar_path = self._sidecar_dir / f"{trace.trace_id}_memory_events.ndjson"
        with sidecar_path.open("w", encoding="utf-8") as handle:
            for step in trace.steps:
                timestamp = step.timestamp or self.collection_timestamp
                for read_event in step.memory_reads:
                    read_event.step_id = step.step_id
                    handle.write(
                        json.dumps(
                            read_event.to_sidecar(trace.trace_id, timestamp), ensure_ascii=True
                        )
                        + "\n"
                    )
                for write_event in step.memory_writes:
                    write_event.step_id = step.step_id
                    handle.write(
                        json.dumps(
                            write_event.to_sidecar(trace.trace_id, timestamp), ensure_ascii=True
                        )
                        + "\n"
                    )
        return sidecar_path


def validate_trace(trace: Ring1Trace) -> None:
    """Raise ValueError if trace violates Trace Specification v0.1."""
    if trace.domain not in VALID_DOMAINS:
        raise ValueError(f"Invalid domain: {trace.domain}")
    if trace.final_outcome not in VALID_OUTCOMES:
        raise ValueError(f"Invalid final_outcome: {trace.final_outcome}")
    if not (4 <= trace.total_steps <= 12):
        raise ValueError(f"total_steps must be 4-12, got {trace.total_steps}")
    if trace.total_steps != len(trace.steps):
        raise ValueError("total_steps does not match number of steps")
    if not trace.trace_id:
        raise ValueError("trace_id is required")
    if not trace.config_hash:
        raise ValueError("config_hash is required")
    if not trace.model_checkpoint_sha256:
        raise ValueError("model_checkpoint_sha256 is required")
    if "git_sha" not in trace.metadata:
        raise ValueError("metadata.git_sha is required for reproducibility")
    if "collection_timestamp" not in trace.metadata:
        raise ValueError("metadata.collection_timestamp is required")

    saw_write = False
    saw_used_read = False

    for step in trace.steps:
        _validate_step(step)
        if step.memory_writes:
            saw_write = True
        for read_event in step.memory_reads:
            if read_event.influenced_action:
                saw_used_read = True

    if not saw_write:
        raise ValueError("Trace must contain at least one memory write event")
    if not saw_used_read:
        raise ValueError("Trace must contain at least one memory read that influenced action")


def _validate_step(step: TraceStep) -> None:
    if step.step_id < 1:
        raise ValueError(f"Invalid step_id: {step.step_id}")
    if not step.observation:
        raise ValueError(f"Step {step.step_id} missing observation")
    if not step.mcts.selected_action:
        raise ValueError(f"Step {step.step_id} missing mcts.selected_action")
    if step.mcts.num_expansions < 1:
        raise ValueError(f"Step {step.step_id} must have num_expansions >= 1")
    if not step.action:
        raise ValueError(f"Step {step.step_id} missing action")
    if not step.reflection:
        raise ValueError(f"Step {step.step_id} missing reflection")

    decision = step.reflection.get("decision")
    if decision not in VALID_REFLECTION_DECISIONS:
        raise ValueError(f"Step {step.step_id} has invalid reflection decision: {decision}")

    for read_event in step.memory_reads:
        if read_event.source not in VALID_MEMORY_SOURCES:
            raise ValueError(f"Invalid memory read source: {read_event.source}")

    for write_event in step.memory_writes:
        if write_event.decision not in VALID_WRITE_DECISIONS:
            raise ValueError(f"Invalid memory write decision: {write_event.decision}")
        if write_event.target_tier not in VALID_TARGET_TIERS:
            raise ValueError(f"Invalid target_tier: {write_event.target_tier}")
        if not (0.0 <= write_event.surprise_score <= 1.0):
            raise ValueError("surprise_score must be between 0 and 1")


def load_traces(jsonl_path: Path) -> list[dict[str, Any]]:
    """Load traces from a JSONL file."""
    traces: list[dict[str, Any]] = []
    with Path(jsonl_path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces
