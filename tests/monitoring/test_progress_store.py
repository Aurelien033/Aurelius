"""Tests for the progress store used by the Aurelius dashboard."""

from __future__ import annotations

import json

from src.monitoring.progress_store import (
    append_event,
    append_metric,
    build_snapshot,
    ensure_state,
    import_metrics_from_jsonl,
    update_milestone,
)


def test_progress_store_seed_metrics_and_snapshot(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    events_path = tmp_path / "events.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"

    ensure_state(state_path, overwrite=True)
    append_event("Started smoke training", kind="training", events_path=events_path)
    append_metric("train_loss", 2.5, step=1, run_id="smoke", metrics_path=metrics_path)
    append_metric("train_loss", 2.1, step=2, run_id="smoke", metrics_path=metrics_path)

    milestone = update_milestone(
        "phase1-baseline-1-4b",
        status="in_progress",
        evidence="local smoke test",
        state_path=state_path,
    )

    snapshot = build_snapshot(
        state_path=state_path,
        events_path=events_path,
        metrics_path=metrics_path,
    )

    assert milestone["status"] == "in_progress"
    assert snapshot["overall"]["milestones"] >= 20
    assert snapshot["latest_metrics"]["train_loss"]["latest"] == 2.1
    assert snapshot["recent_events"][0]["message"] == "Started smoke training"


def test_import_metrics_from_jsonl(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    events_path = tmp_path / "events.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    source_path = tmp_path / "source.jsonl"

    ensure_state(state_path, overwrite=True)
    rows = [
        {"step": 1, "train_loss": 3.0, "val_loss": 2.8},
        {"name": "tokens_per_sec", "value": 120.0, "step": 1},
    ]
    source_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    result = import_metrics_from_jsonl(
        source_path,
        run_id="imported",
        state_path=state_path,
        events_path=events_path,
        metrics_path=metrics_path,
    )

    assert result["imported"] == 3
    snapshot = build_snapshot(
        state_path=state_path,
        events_path=events_path,
        metrics_path=metrics_path,
    )
    assert snapshot["latest_metrics"]["tokens_per_sec"]["latest"] == 120.0
