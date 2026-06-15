#!/usr/bin/env python3
"""Repair/resume first-light-3651bb11 after receipt serialization failure.

This script is intentionally narrow:
- targets an existing run directory;
- reconstructs completed primary rows from checkpoint.jsonl;
- loads already-emitted primary trace/ledger artifacts;
- runs only the missing PILOT-labeled dense temp=0.7/top-p=0.95 rows;
- reruns frozen analysis and receipt assembly with JSON/YAML-safe serialization.

It does NOT rerun dense/random/entropy primary generations.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import first_light_runner_v3 as fl

RUN_ID = "first-light-3651bb11"
RUN_DIR = Path("/Users/christienantonio/Desktop/AI:ML Research/first_light_receipt") / RUN_ID


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def reconstruct_results_from_checkpoint(path: Path) -> list[fl.RunResult]:
    results: list[fl.RunResult] = []
    for row in load_jsonl(path):
        results.append(
            fl.RunResult(
                instance_id=row["instance_id"],
                policy=row["policy"],
                seed=int(row["seed"]),
                passed=bool(row["passed"]),
                error=row.get("error") or "",
                completion="",
                realized_tokens=int(row.get("realized_tokens") or 0),
                family=row.get("family") or "",
                skip_layers=[],
                entropy_data=None,
                run_label=row.get("label") or "primary",
            )
        )
    return results


def main() -> int:
    if not RUN_DIR.exists():
        print(f"BLOCKER: missing run directory {RUN_DIR}", file=sys.stderr)
        return 2

    runner = fl.FirstLightRunner(run_id=RUN_ID)
    runner.setup()

    # Load completed primary generations and existing evidence emitted before the
    # receipt failure. These are the completed dense/random/entropy rows.
    runner.all_results = reconstruct_results_from_checkpoint(runner.checkpoint_path)
    runner.trace_rows = load_jsonl(RUN_DIR / "traces_forced_entropy.jsonl")
    runner.ledger_rows = load_jsonl(RUN_DIR / "ledger_forced_entropy.jsonl")

    if not runner.trace_rows or not runner.ledger_rows:
        print("BLOCKER: missing existing forced-entropy trace/ledger rows", file=sys.stderr)
        return 3

    runner.amendment_log = [
        {
            "timestamp": fl.now_iso(),
            "kind": "repair_after_receipt_serialization_failure",
            "detail": "Original run completed primary dense/random/entropy generations and frozen analysis, then failed while writing receipt JSON because numpy scalar booleans were not JSON serializable. Patched to_jsonable + safe YAML dump; no primary generation values changed.",
            "affected_artifacts": [
                str(RUN_DIR / f"{RUN_ID}_receipt.json"),
                str(RUN_DIR / f"{RUN_ID}_receipt.yaml"),
            ],
        },
        {
            "timestamp": fl.now_iso(),
            "kind": "pilot_resume_key_repair",
            "detail": "Original runner checkpoint key omitted block label, so the PILOT dense temp=0.7 rows collided with greedy dense_uniform rows and were skipped. Patched checkpoint namespace and ran PILOT-labeled dense_uniform temp=0.7/top_p=0.95 rows without rerunning primary policies.",
            "affected_artifacts": [
                str(RUN_DIR / "checkpoint.jsonl"),
                str(RUN_DIR / "completions/pilot/dense_uniform"),
                str(RUN_DIR / "results_pilot.json"),
                str(RUN_DIR / "traces_pilot.jsonl"),
                str(RUN_DIR / "ledger_pilot.jsonl"),
            ],
        },
    ]

    # Fresh thermal warm for the resumed pilot block. Primary measured blocks had
    # already been warmed in the original process; this makes the repair pilot block
    # honest after process restart.
    runner.thermal_warm(duration_s=600)

    self_proofs = {
        "p1_routing": fl.self_proof_p1(runner.tok, runner.model),
        "p2_same_instances": fl.self_proof_p2(runner.pinned_ids["fl_subset"]),
        "p3_verifier_sanity": fl.self_proof_p3(runner.tok, runner.model),
        "p4_computed_verdicts": "VERIFIED_DURING_ANALYSIS",
    }

    runner.run_pilot_block()
    runner.run_analysis()
    runner.assemble_receipt(self_proofs)

    print("REPAIR COMPLETE")
    print(f"Receipt YAML: {RUN_DIR / f'{RUN_ID}_receipt.yaml'}")
    print(f"Receipt JSON: {RUN_DIR / f'{RUN_ID}_receipt.json'}")
    print(f"Traces: {RUN_DIR / 'traces.jsonl'}")
    print(f"Ledger: {RUN_DIR / 'ledger.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
