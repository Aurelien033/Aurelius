#!/usr/bin/env python3
"""Reassemble corrected First Light receipt from existing artifacts only.

No model load and no generation. Fixes v3 receipt analysis defects:
- H-FL-1 validates final trace rows instead of assuming smoke success.
- H-FL-3 uses entropy-random sign per prereg.
- H-FL-4 reads measured TPS from ledger and pinned byte predictions.
- Receipt references RSS sidecar and pilot repair artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import first_light_runner_v3 as fl

RUN_ID = "first-light-3651bb11"
RUN_DIR = Path("/Users/christienantonio/Desktop/AI:ML Research/first_light_receipt") / RUN_ID
ROOT = Path("/Users/christienantonio/Desktop/AI:ML Research")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def reconstruct_results(path: Path) -> list[fl.RunResult]:
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
    old_receipt = json.load(open(RUN_DIR / f"{RUN_ID}_receipt.json"))
    runner = fl.FirstLightRunner(run_id=RUN_ID)
    runner.schema = fl.load_json(fl.SCHEMA_PATH)
    runner.manifest = fl.load_yaml(fl.MANIFEST_PATH)
    runner.pinned_ids = fl.load_pinned_instance_ids()
    runner.all_results = reconstruct_results(RUN_DIR / "checkpoint.jsonl")
    runner.trace_rows = load_jsonl(RUN_DIR / "traces.jsonl")
    runner.ledger_rows = load_jsonl(RUN_DIR / "ledger.jsonl")
    runner.rss_log = old_receipt.get("rss_log", [])
    runner.thermal_log = old_receipt.get("thermal_log", [])
    runner.amendment_log = old_receipt.get("execution", {}).get("amendment_log", []) + [
        {
            "timestamp": fl.now_iso(),
            "kind": "analysis_receipt_correction",
            "detail": "Corrected v3 receipt analysis without rerunning generations: H-FL-1 now validates final trace rows; H-FL-3 sign fixed to entropy-random per prereg; H-FL-4 reads measured TPS from ledger_forced_entropy.jsonl and pinned first_light_byte_predictions.yaml instead of placeholder constants.",
            "affected_artifacts": [
                str(RUN_DIR / f"{RUN_ID}_receipt.json"),
                str(RUN_DIR / f"{RUN_ID}_receipt.yaml"),
                str(RUN_DIR / "ledger_forced_entropy.jsonl"),
                str(ROOT / "first_light_byte_predictions.yaml"),
            ],
        },
        {
            "timestamp": fl.now_iso(),
            "kind": "rss_sidecar_repair",
            "detail": "v3 runner recorded the empty-process RSS baseline after model load. Added rss_sidecar_measurement.json: fresh process, same pinned model revision, psutil RSS/USS + resource peak RSS + torch.mps allocator bytes. Treat sidecar as RSS gate evidence; primary receipt rss_log retained as provenance only.",
            "affected_artifacts": [str(RUN_DIR / "rss_sidecar_measurement.json")],
        },
    ]
    analysis = runner.run_analysis()
    self_proofs = old_receipt.get("self_proofs", {})
    self_proofs["p4_computed_verdicts"] = {
        "status": "VERIFIED_BY_REASSEMBLE_SCRIPT",
        "source": str(Path(__file__).resolve()),
        "hypotheses": analysis.get("hypotheses", {}),
    }
    runner.assemble_receipt(self_proofs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
