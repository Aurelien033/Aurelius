"""CLI helpers for the Aurelius progress dashboard.

Examples:

    python scripts/aurelius_progress.py seed --overwrite
    python scripts/aurelius_progress.py snapshot
    python scripts/aurelius_progress.py event "Started 1.4B smoke training" --kind training --severity info
    python scripts/aurelius_progress.py metric train_loss 2.31 --step 100 --run-id smoke-1
    python scripts/aurelius_progress.py milestone phase1-baseline-1-4b --status in_progress
    python scripts/aurelius_progress.py import-metrics logs/metrics.jsonl --run-id smoke-1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.monitoring.progress_store import (
    append_event,
    append_metric,
    build_snapshot,
    ensure_state,
    import_metrics_from_jsonl,
    update_milestone,
)


def _json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("JSON value must be an object")
    return value


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aurelius progress dashboard CLI")
    parser.add_argument("--state", default=None, help="Override state.json path.")
    parser.add_argument("--events", default=None, help="Override events JSONL path.")
    parser.add_argument("--metrics", default=None, help="Override metrics JSONL path.")
    sub = parser.add_subparsers(dest="command", required=True)

    seed = sub.add_parser("seed", help="Create or overwrite the default roadmap state.")
    seed.add_argument("--overwrite", action="store_true")

    sub.add_parser("snapshot", help="Print the full dashboard snapshot JSON.")

    sub.add_parser("list-milestones", help="List roadmap milestones.")

    event = sub.add_parser("event", help="Append a dashboard event.")
    event.add_argument("message")
    event.add_argument("--kind", default="general")
    event.add_argument("--severity", default="info", choices=["info", "warning", "error", "critical"])
    event.add_argument("--metadata", default=None, help="JSON object metadata.")

    metric = sub.add_parser("metric", help="Append a scalar metric sample.")
    metric.add_argument("name")
    metric.add_argument("value", type=float)
    metric.add_argument("--step", type=int, default=None)
    metric.add_argument("--run-id", default="default")
    metric.add_argument("--unit", default=None)
    metric.add_argument("--tags", default=None, help="JSON object tags.")

    milestone = sub.add_parser("milestone", help="Update a roadmap milestone.")
    milestone.add_argument("milestone_id")
    milestone.add_argument("--status", default=None)
    milestone.add_argument("--evidence", default=None)
    milestone.add_argument("--notes", default=None)
    milestone.add_argument("--owner", default=None)
    milestone.add_argument("--blockers", default=None, help="JSON array of blockers.")
    milestone.add_argument("--next-action", default=None)

    imported = sub.add_parser("import-metrics", help="Import metric JSONL into the progress store.")
    imported.add_argument("path")
    imported.add_argument("--run-id", default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    state_path = Path(args.state) if args.state else None
    events_path = Path(args.events) if args.events else None
    metrics_path = Path(args.metrics) if args.metrics else None

    if args.command == "seed":
        path = ensure_state(state_path or "data/aurelius_progress/state.json", overwrite=args.overwrite)
        _print_json({"ok": True, "state": str(path)})
        return

    if args.command == "snapshot":
        _print_json(
            build_snapshot(
                state_path=state_path or "data/aurelius_progress/state.json",
                events_path=events_path or "data/aurelius_progress/events.jsonl",
                metrics_path=metrics_path or "data/aurelius_progress/metrics.jsonl",
            )
        )
        return

    if args.command == "list-milestones":
        snapshot = build_snapshot(
            state_path=state_path or "data/aurelius_progress/state.json",
            events_path=events_path or "data/aurelius_progress/events.jsonl",
            metrics_path=metrics_path or "data/aurelius_progress/metrics.jsonl",
        )
        rows = []
        for phase in snapshot["roadmap"]:
            for milestone in phase["milestones"]:
                rows.append(
                    {
                        "phase": phase["phase"],
                        "phase_title": phase["title"],
                        **milestone,
                    }
                )
        _print_json({"milestones": rows})
        return

    if args.command == "event":
        event_record = append_event(
            args.message,
            kind=args.kind,
            severity=args.severity,
            metadata=_json_object(args.metadata),
            events_path=events_path or "data/aurelius_progress/events.jsonl",
        )
        _print_json({"ok": True, "event": event_record})
        return

    if args.command == "metric":
        metric_record = append_metric(
            args.name,
            args.value,
            step=args.step,
            run_id=args.run_id,
            unit=args.unit,
            tags=_json_object(args.tags),
            metrics_path=metrics_path or "data/aurelius_progress/metrics.jsonl",
        )
        _print_json({"ok": True, "metric": metric_record})
        return

    if args.command == "milestone":
        milestone_record = update_milestone(
            args.milestone_id,
            status=args.status,
            evidence=args.evidence,
            notes=args.notes,
            owner=args.owner,
            blockers=json.loads(args.blockers) if args.blockers else None,
            next_action=args.next_action,
            state_path=state_path or "data/aurelius_progress/state.json",
        )
        _print_json({"ok": True, "milestone": milestone_record})
        return

    if args.command == "import-metrics":
        result = import_metrics_from_jsonl(
            args.path,
            run_id=args.run_id,
            state_path=state_path or "data/aurelius_progress/state.json",
            events_path=events_path or "data/aurelius_progress/events.jsonl",
            metrics_path=metrics_path or "data/aurelius_progress/metrics.jsonl",
        )
        _print_json({"ok": True, "imported": result["imported"]})
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
