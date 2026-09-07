"""Persistent progress store for the Aurelius progress dashboard.

This module is intentionally dependency-light: it stores roadmap, events, and
training/evaluation metrics as JSON/JSONL files under ``data/aurelius_progress``.
The dashboard reads this state without requiring Prometheus, WandB, or any other
external service. Production deployments can later add collectors that write the
same JSONL schema.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "aurelius_progress"
DEFAULT_STATE_PATH = DEFAULT_DATA_DIR / "state.json"
DEFAULT_EVENTS_PATH = DEFAULT_DATA_DIR / "events.jsonl"
DEFAULT_METRICS_PATH = DEFAULT_DATA_DIR / "metrics.jsonl"

STATUS_WEIGHTS: dict[str, float] = {
    "done": 1.0,
    "review": 0.9,
    "in_progress": 0.6,
    "planned": 0.2,
    "blocked": 0.05,
    "not_started": 0.0,
}

STATUS_ORDER = {
    "blocked": 0,
    "not_started": 1,
    "planned": 2,
    "in_progress": 3,
    "review": 4,
    "done": 5,
}


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp suitable for JSON state."""

    return datetime.now(UTC).isoformat()


def _status_weight(status: str | None) -> float:
    return STATUS_WEIGHTS.get((status or "not_started").lower(), 0.0)


def _normalize_status(status: str | None) -> str:
    normalized = (status or "not_started").lower().replace(" ", "_")
    return normalized if normalized in STATUS_WEIGHTS else "not_started"


def _default_milestone(
    milestone_id: str,
    title: str,
    status: str,
    weight: float,
    next_action: str,
    evidence: str | None = None,
    owner: str = "Aurelius",
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": milestone_id,
        "title": title,
        "status": status,
        "weight": weight,
        "owner": owner,
        "evidence": evidence or "",
        "notes": "",
        "next_action": next_action,
        "blockers": blockers or [],
        "updated_at": utc_now_iso(),
    }


def _default_roadmap() -> dict[str, Any]:
    """Seed a detailed research/product roadmap for Aurelius.

    The roadmap is deliberately evidence-oriented: every milestone has an
    ``evidence`` field and a ``next_action``. This makes the dashboard useful as
    a management surface rather than a static checklist.
    """

    roadmap = [
        {
            "id": "phase-0",
            "phase": "Phase 0",
            "title": "Progress telemetry foundation",
            "timeframe": "Immediate",
            "status": "in_progress",
            "weight": 1.0,
            "objective": (
                "Make Aurelius progress observable before scale training begins. "
                "Capture roadmap status, training metrics, eval gates, data quality, "
                "interpretability experiments, safety results, and compute spend in "
                "one local dashboard."
            ),
            "milestones": [
                _default_milestone(
                    "phase0-dashboard-store",
                    "Persistent progress store",
                    "done",
                    1.0,
                    "Use the progress store as the canonical local state file.",
                    evidence=(
                        "src/monitoring/progress_store.py writes state, events, and metrics JSONL."
                    ),
                ),
                _default_milestone(
                    "phase0-live-dashboard",
                    "Live local dashboard",
                    "done",
                    1.0,
                    "Open the dashboard and verify /api/snapshot returns JSON.",
                    evidence="gateway/aurelius_progress_dashboard.py serves the dashboard.",
                ),
                _default_milestone(
                    "phase0-cli",
                    "CLI update helpers",
                    "done",
                    0.8,
                    "Add commands for event, metric, milestone, and snapshot updates.",
                    evidence="scripts/aurelius_progress.py provides local update primitives.",
                ),
                _default_milestone(
                    "phase0-training-hook",
                    "Training loop hook",
                    "not_started",
                    1.2,
                    "Emit one metric event per training step or logging interval.",
                    blockers=["No active training run yet"],
                ),
                _default_milestone(
                    "phase0-eval-hook",
                    "Evaluation gate hook",
                    "not_started",
                    1.0,
                    "Write eval results as dashboard events and metrics.",
                    blockers=["Need canonical eval harness output format"],
                ),
            ],
        },
        {
            "id": "phase-1",
            "phase": "Phase 1",
            "title": "AMC-first baseline and reproducibility",
            "timeframe": "0-3 months",
            "status": "in_progress",
            "weight": 1.4,
            "objective": (
                "Establish a reproducible Aurelian Memory Core baseline before "
                "adding larger-scale architectural complexity."
            ),
            "milestones": [
                _default_milestone(
                    "phase1-dataset-v1",
                    "Reasoning-centered SFT/pretrain dataset v1",
                    "done",
                    1.0,
                    "Regenerate and validate the Aurelius reasoning dataset.",
                    evidence="data/aurelius_reasoning_sft_v1 generated with 238 SFT rows.",
                ),
                _default_milestone(
                    "phase1-baseline-1-4b",
                    "1.4B dense baseline training recipe",
                    "in_progress",
                    1.6,
                    "Run a small local smoke train and record loss/throughput.",
                    blockers=["GPU availability and final optimizer config"],
                ),
                _default_milestone(
                    "phase1-checkpointing",
                    "Checkpoint/resume correctness",
                    "not_started",
                    1.0,
                    "Resume from checkpoint and verify optimizer state parity.",
                ),
                _default_milestone(
                    "phase1-long-context-smoke",
                    "Long-context smoke benchmark",
                    "not_started",
                    0.8,
                    "Run RULER-style minimal probes before full long-context training.",
                ),
            ],
        },
        {
            "id": "phase-2",
            "phase": "Phase 2",
            "title": "Reasoning circuits and mechanistic interpretability",
            "timeframe": "3-9 months",
            "status": "planned",
            "weight": 1.8,
            "objective": (
                "Measure how Aurelius reasons internally: circuits, memory gates, "
                "latent planning, metacognition, and verification behavior."
            ),
            "milestones": [
                _default_milestone(
                    "phase2-sae-v0",
                    "Sparse autoencoder feature atlas v0",
                    "planned",
                    1.2,
                    "Train SAE on selected residual-stream slices.",
                ),
                _default_milestone(
                    "phase2-probes",
                    "Layer probes for reasoning state",
                    "not_started",
                    1.0,
                    "Train probes for uncertainty, plan depth, and verification state.",
                ),
                _default_milestone(
                    "phase2-causal-tracing",
                    "Causal tracing for answer-critical tokens",
                    "not_started",
                    1.0,
                    "Run intervention traces on math/logic tasks.",
                ),
                _default_milestone(
                    "phase2-metacog-evals",
                    "Metacognition eval suite",
                    "not_started",
                    1.0,
                    "Create tasks where self-checking changes the final answer.",
                ),
            ],
        },
        {
            "id": "phase-3",
            "phase": "Phase 3",
            "title": "Scale path to 10B under budget",
            "timeframe": "6-15 months",
            "status": "planned",
            "weight": 1.5,
            "objective": (
                "Prove that a 10B Aurelius model is feasible under the $42-80K "
                "text-only target using stable architecture and data quality."
            ),
            "milestones": [
                _default_milestone(
                    "phase3-data-quality-gate",
                    "Data quality gate before scale",
                    "not_started",
                    1.4,
                    "Measure dedup rate, contamination risk, toxicity, and domain balance.",
                ),
                _default_milestone(
                    "phase3-ddp-fsdp",
                    "Distributed training harness",
                    "not_started",
                    1.4,
                    "Validate DDP/FSDP on rented GPUs with checkpoint recovery.",
                ),
                _default_milestone(
                    "phase3-architecture-ablation",
                    "Architecture ablation vs larger dense Transformer",
                    "not_started",
                    1.2,
                    "Kill mechanisms that do not beat +20% dense Transformer baseline.",
                ),
                _default_milestone(
                    "phase3-cost-model",
                    "Cloud cost model and rental plan",
                    "in_progress",
                    1.0,
                    "Track H100/B200 rental quotes and token budget.",
                ),
            ],
        },
        {
            "id": "phase-4",
            "phase": "Phase 4",
            "title": "Alignment, safety, and reward modeling",
            "timeframe": "9-18 months",
            "status": "planned",
            "weight": 1.3,
            "objective": (
                "Build safety as measurement: refusal behavior, red-team coverage, "
                "reward models, and post-training stability."
            ),
            "milestones": [
                _default_milestone(
                    "phase4-safety-evals",
                    "Safety eval harness",
                    "not_started",
                    1.0,
                    "Run refusal, helpfulness, and safe-redirection evals.",
                ),
                _default_milestone(
                    "phase4-reward-model",
                    "Reward model v0",
                    "not_started",
                    1.0,
                    "Train preference model from curated chosen/rejected data.",
                ),
                _default_milestone(
                    "phase4-grpo-stability",
                    "GRPO/RL stability guardrails",
                    "not_started",
                    1.2,
                    "Monitor reward collapse, KL drift, and refusal regression.",
                ),
            ],
        },
        {
            "id": "phase-5",
            "phase": "Phase 5",
            "title": "Agentic deployment and human feedback loop",
            "timeframe": "12-24 months",
            "status": "planned",
            "weight": 1.2,
            "objective": (
                "Turn Aurelius into a reliable research agent with memory, tools, "
                "planning, sandboxing, and human feedback capture."
            ),
            "milestones": [
                _default_milestone(
                    "phase5-agent-memory",
                    "Persistent agent memory",
                    "not_started",
                    1.0,
                    "Store and retrieve task context with provenance.",
                ),
                _default_milestone(
                    "phase5-tool-sandbox",
                    "Tool sandbox and audit trail",
                    "not_started",
                    1.0,
                    "Log tool calls, approvals, and safety outcomes.",
                ),
                _default_milestone(
                    "phase5-human-feedback",
                    "Human feedback capture",
                    "not_started",
                    0.8,
                    "Collect preference labels from dashboard-reviewed outputs.",
                ),
            ],
        },
        {
            "id": "phase-6",
            "phase": "Phase 6",
            "title": "Papers, reproducibility, and public artifact release",
            "timeframe": "18-30 months",
            "status": "planned",
            "weight": 1.0,
            "objective": (
                "Prepare ICLR/NeurIPS submission artifacts with ablations, "
                "reproducible configs, and mechanistic evidence."
            ),
            "milestones": [
                _default_milestone(
                    "phase6-paper-outline",
                    "Paper outline and contribution claims",
                    "in_progress",
                    0.8,
                    "Separate proven claims from roadmap/speculation.",
                ),
                _default_milestone(
                    "phase6-repro-pack",
                    "Reproducibility package",
                    "not_started",
                    1.0,
                    "Freeze configs, seeds, data manifests, and eval scripts.",
                ),
                _default_milestone(
                    "phase6-ablation-table",
                    "Ablation and cost table",
                    "not_started",
                    1.0,
                    "Show why each mechanism survives the +20% dense baseline gate.",
                ),
            ],
        },
    ]

    tracks = [
        {
            "id": "data",
            "name": "Data pipeline and curriculum",
            "weight": 1.4,
            "owner": "Aurelius",
            "objective": "High-quality, reasoning-centered data with provenance.",
            "milestone_ids": [
                "phase1-dataset-v1",
                "phase3-data-quality-gate",
            ],
        },
        {
            "id": "training",
            "name": "Training and scale engineering",
            "weight": 1.5,
            "owner": "Aurelius",
            "objective": "Stable pretraining, checkpointing, and distributed scale path.",
            "milestone_ids": [
                "phase1-baseline-1-4b",
                "phase1-checkpointing",
                "phase3-ddp-fsdp",
                "phase3-cost-model",
            ],
        },
        {
            "id": "reasoning",
            "name": "Reasoning and mechanistic interpretability",
            "weight": 1.8,
            "owner": "Aurelius",
            "objective": "Understand internal reasoning circuits, not just benchmark scores.",
            "milestone_ids": [
                "phase2-sae-v0",
                "phase2-probes",
                "phase2-causal-tracing",
                "phase2-metacog-evals",
            ],
        },
        {
            "id": "safety",
            "name": "Safety, alignment, and evaluation",
            "weight": 1.3,
            "owner": "Aurelius",
            "objective": "Measure safety and alignment as first-class research signals.",
            "milestone_ids": [
                "phase4-safety-evals",
                "phase4-reward-model",
                "phase4-grpo-stability",
            ],
        },
        {
            "id": "agentic",
            "name": "Agentic deployment",
            "weight": 1.1,
            "owner": "Aurelius",
            "objective": "Reliable agent loop, memory, tools, and human feedback.",
            "milestone_ids": [
                "phase5-agent-memory",
                "phase5-tool-sandbox",
                "phase5-human-feedback",
            ],
        },
        {
            "id": "publication",
            "name": "Paper and reproducibility",
            "weight": 1.0,
            "owner": "Aurelius",
            "objective": "Convert research progress into credible ICLR/NeurIPS artifacts.",
            "milestone_ids": [
                "phase6-paper-outline",
                "phase6-repro-pack",
                "phase6-ablation-table",
            ],
        },
    ]

    return {
        "version": "1.0.0",
        "project": {
            "name": "Aurelius",
            "mission": (
                "Build a research-grade model whose internal reasoning mechanisms "
                "are measured, interpretable, and reproducible."
            ),
            "target_venues": ["ICLR 2027", "NeurIPS 2027"],
            "practical_scale": "10B parameters with rented cloud GPUs",
            "budget_target": "$42-80K text-only target, hard ceiling below $500K",
            "repo": str(PROJECT_ROOT),
        },
        "tracks": tracks,
        "roadmap": roadmap,
        "current_focus": [
            "Keep the dashboard state live: every training/eval run should append metrics.",
            "Prove AMC-first baseline before adding architectural complexity.",
            "Treat data quality as the primary bottleneck.",
            "Kill mechanisms unless they beat a larger dense Transformer baseline.",
        ],
        "risk_register": [
            {
                "id": "risk-rep-collapse",
                "title": "Representation collapse during RL/GRPO",
                "severity": "high",
                "status": "watching",
                "mitigation": (
                    "Track KL drift, reward variance, feature entropy, and refusal regression."
                ),
            },
            {
                "id": "risk-data-contamination",
                "title": "Benchmark/data contamination",
                "severity": "high",
                "status": "watching",
                "mitigation": "Record dataset hashes, source manifests, and contamination probes.",
            },
            {
                "id": "risk-scope-creep",
                "title": "Architecture scope creep",
                "severity": "medium",
                "status": "watching",
                "mitigation": "Require +20% dense Transformer ablation gate for each mechanism.",
            },
        ],
        "metric_targets": {
            "train_loss": {"direction": "lower_is_better", "target": None},
            "val_loss": {"direction": "lower_is_better", "target": None},
            "tokens_per_sec": {"direction": "higher_is_better", "target": None},
            "gpu_util_pct": {"direction": "healthy_range", "target": [60.0, 95.0]},
            "reward_score": {"direction": "higher_is_better", "target": None},
            "refusal_accuracy": {"direction": "higher_is_better", "target": 0.95},
        },
        "updated_at": utc_now_iso(),
    }


def ensure_state(
    state_path: str | Path = DEFAULT_STATE_PATH,
    *,
    overwrite: bool = False,
) -> Path:
    """Create the default state file if it does not exist."""

    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not path.exists():
        save_state(_default_roadmap(), path)
    return path


def load_state(state_path: str | Path = DEFAULT_STATE_PATH) -> dict[str, Any]:
    """Load state, creating the default file if needed."""

    path = ensure_state(state_path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_state(state: dict[str, Any], state_path: str | Path = DEFAULT_STATE_PATH) -> None:
    """Atomically save state to JSON."""

    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = utc_now_iso()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Read a JSONL file from newest to oldest when limit is provided."""

    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit is not None:
        return rows[-limit:]
    return rows


def append_event(
    message: str,
    *,
    kind: str = "general",
    severity: str = "info",
    metadata: dict[str, Any] | None = None,
    events_path: str | Path = DEFAULT_EVENTS_PATH,
) -> dict[str, Any]:
    """Append a human-readable progress event."""

    event = {
        "id": uuid.uuid4().hex,
        "timestamp": time.time(),
        "timestamp_iso": utc_now_iso(),
        "kind": kind,
        "severity": severity,
        "message": message,
        "metadata": metadata or {},
    }
    _append_jsonl(events_path, event)
    return event


def append_metric(
    name: str,
    value: float,
    *,
    step: int | None = None,
    run_id: str = "default",
    unit: str | None = None,
    tags: dict[str, Any] | None = None,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
) -> dict[str, Any]:
    """Append a scalar metric sample."""

    metric = {
        "id": uuid.uuid4().hex,
        "timestamp": time.time(),
        "timestamp_iso": utc_now_iso(),
        "run_id": run_id,
        "name": name,
        "value": float(value),
        "step": step,
        "unit": unit,
        "tags": tags or {},
    }
    _append_jsonl(metrics_path, metric)
    return metric


def log_training_metric(
    run_id: str,
    step: int,
    metrics: dict[str, Any],
    *,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
) -> list[dict[str, Any]]:
    """Write a dictionary of training metrics as scalar samples."""

    records: list[dict[str, Any]] = []
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and value == value:
            records.append(
                append_metric(
                    str(name),
                    float(value),
                    step=step,
                    run_id=run_id,
                    metrics_path=metrics_path,
                )
            )
    return records


def update_milestone(
    milestone_id: str,
    *,
    status: str | None = None,
    evidence: str | None = None,
    notes: str | None = None,
    owner: str | None = None,
    blockers: list[str] | None = None,
    next_action: str | None = None,
    state_path: str | Path = DEFAULT_STATE_PATH,
) -> dict[str, Any]:
    """Update a roadmap milestone and return the updated milestone."""

    state = load_state(state_path)
    status = _normalize_status(status)
    found: dict[str, Any] | None = None

    for phase in state["roadmap"]:
        for milestone in phase["milestones"]:
            if milestone["id"] == milestone_id:
                found = milestone

    if found is None:
        raise KeyError(f"Unknown milestone id: {milestone_id}")

    if status:
        found["status"] = status
    if evidence is not None:
        found["evidence"] = evidence
    if notes is not None:
        found["notes"] = notes
    if owner is not None:
        found["owner"] = owner
    if blockers is not None:
        found["blockers"] = blockers
    if next_action is not None:
        found["next_action"] = next_action
    found["updated_at"] = utc_now_iso()

    save_state(state, state_path)
    return copy.deepcopy(found)


def _milestone_progress(milestones: list[dict[str, Any]]) -> float:
    if not milestones:
        return 0.0
    weighted = sum(
        float(m.get("weight", 1.0)) * _status_weight(m.get("status")) for m in milestones
    )
    total = sum(float(m.get("weight", 1.0)) for m in milestones)
    return 100.0 * weighted / total if total else 0.0


def _overall_progress(state: dict[str, Any]) -> float:
    phases = state.get("roadmap", [])
    weighted = sum(
        float(p.get("weight", 1.0)) * (_milestone_progress(p.get("milestones", [])) / 100.0)
        for p in phases
    )
    total = sum(float(p.get("weight", 1.0)) for p in phases)
    return 100.0 * weighted / total if total else 0.0


def _track_progress(state: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {
        milestone["id"]: milestone
        for phase in state.get("roadmap", [])
        for milestone in phase.get("milestones", [])
    }
    tracks: list[dict[str, Any]] = []
    for track in state.get("tracks", []):
        milestones = [by_id[mid] for mid in track.get("milestone_ids", []) if mid in by_id]
        tracks.append(
            {
                "id": track.get("id"),
                "name": track.get("name"),
                "owner": track.get("owner"),
                "objective": track.get("objective"),
                "weight": track.get("weight", 1.0),
                "progress": _milestone_progress(milestones),
                "milestones": milestones,
            }
        )
    return tracks


def _metric_series(
    metrics: list[dict[str, Any]], name: str, limit: int = 200
) -> list[dict[str, Any]]:
    rows = [m for m in metrics if m.get("name") == name]
    rows.sort(
        key=lambda row: (
            row.get("step") is None,
            row.get("step") or 0,
            row.get("timestamp", 0),
        )
    )
    return rows[-limit:]


def _aggregate_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    by_name: dict[str, list[float]] = {}
    latest: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        name = str(metric.get("name", "unknown"))
        try:
            value = float(metric["value"])
        except (KeyError, TypeError, ValueError):
            continue
        by_name.setdefault(name, []).append(value)
        latest[name] = metric

    aggregate: dict[str, Any] = {}
    for name, values in by_name.items():
        aggregate[name] = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": mean(values),
            "latest": values[-1],
            "latest_step": latest.get(name, {}).get("step"),
            "latest_run_id": latest.get(name, {}).get("run_id"),
        }
    return aggregate


def _risk_summary(state: dict[str, Any]) -> dict[str, Any]:
    risks = state.get("risk_register", [])
    return {
        "total": len(risks),
        "high": sum(1 for r in risks if r.get("severity") == "high"),
        "medium": sum(1 for r in risks if r.get("severity") == "medium"),
        "low": sum(1 for r in risks if r.get("severity") == "low"),
        "items": risks,
    }


def build_snapshot(
    *,
    state_path: str | Path = DEFAULT_STATE_PATH,
    events_path: str | Path = DEFAULT_EVENTS_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    metric_limit: int = 1000,
    event_limit: int = 200,
) -> dict[str, Any]:
    """Build the full JSON snapshot consumed by the dashboard."""

    state = load_state(state_path)
    metrics = read_jsonl(metrics_path, limit=metric_limit)
    events = read_jsonl(events_path, limit=event_limit)
    tracks = _track_progress(state)
    overall = _overall_progress(state)
    aggregate = _aggregate_metrics(metrics)
    latest_metric_names = sorted(aggregate.keys())

    return {
        "generated_at": utc_now_iso(),
        "project": state.get("project", {}),
        "overall": {
            "progress": overall,
            "status": "blocked" if overall < 1 else "in_progress",
            "roadmap_phases": len(state.get("roadmap", [])),
            "milestones": sum(
                len(phase.get("milestones", [])) for phase in state.get("roadmap", [])
            ),
        },
        "tracks": tracks,
        "roadmap": state.get("roadmap", []),
        "current_focus": state.get("current_focus", []),
        "risk_summary": _risk_summary(state),
        "latest_metrics": aggregate,
        "metric_names": latest_metric_names,
        "metric_series": {
            name: _metric_series(metrics, name, limit=200) for name in latest_metric_names
        },
        "recent_events": events,
        "state_hash": hashlib.sha256(json.dumps(state, sort_keys=True).encode("utf-8")).hexdigest(),
    }


def import_metrics_from_jsonl(
    source_path: str | Path,
    *,
    run_id: str | None = None,
    state_path: str | Path = DEFAULT_STATE_PATH,
    events_path: str | Path = DEFAULT_EVENTS_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
) -> dict[str, Any]:
    """Import metric rows from another JSONL source.

    Accepted row shapes:
    - ``{"name": "train_loss", "value": 2.1, "step": 100}``
    - ``{"step": 100, "train_loss": 2.1, "val_loss": 1.9}``
    """

    path = Path(source_path)
    imported = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "name" in row and "value" in row:
                append_metric(
                    str(row["name"]),
                    float(row["value"]),
                    step=row.get("step"),
                    run_id=run_id or str(row.get("run_id", "imported")),
                    unit=row.get("unit"),
                    tags=row.get("tags") or {},
                    metrics_path=metrics_path,
                )
                imported += 1
            else:
                step = row.get("step")
                for name, value in row.items():
                    if name in {"timestamp", "timestamp_iso", "run_id", "step", "tags", "unit"}:
                        continue
                    if isinstance(value, (int, float)) and value == value:
                        append_metric(
                            name,
                            float(value),
                            step=step,
                            run_id=run_id or str(row.get("run_id", "imported")),
                            unit=row.get("unit"),
                            tags=row.get("tags") or {},
                            metrics_path=metrics_path,
                        )
                        imported += 1

    append_event(
        f"Imported {imported} metric samples from {path}",
        kind="metrics",
        severity="info",
        metadata={"source_path": str(path), "run_id": run_id},
        events_path=events_path,
    )
    snapshot = build_snapshot(
        state_path=state_path,
        events_path=events_path,
        metrics_path=metrics_path,
    )
    return {"imported": imported, "snapshot": snapshot}
