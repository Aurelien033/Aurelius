#!/usr/bin/env python3
"""Independent artifact-level validator for First Light v4.

Reads completed v4 artifacts; does not import or trust runner analysis code; does
not load the model; does not rerun generation. Produces JSON + Markdown reports
inside the run directory.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from jsonschema import Draft7Validator

RESEARCH = Path("/Users/christienantonio/Desktop/AI:ML Research")
RUN_DIR = RESEARCH / "first_light_receipt" / "first-light-v4-24515383"
RUN_ID = RUN_DIR.name
SCHEMA_PATH = RESEARCH / "directive_trace_schema.json"
MANIFEST_PATH = RESEARCH / "gym-v0.2" / "gym_manifest.yaml"
BYTE_PRED_PATH = RESEARCH / "first_light_byte_predictions.yaml"
RECEIPT_PATH = RUN_DIR / f"{RUN_ID}_receipt.yaml"
SEEDS = [1337, 2026, 7]
POLICIES = ["dense_uniform", "random_matched_mix", "heuristic_entropy"]
FAMILIES = ["F1_mbpp", "F2_json", "F3_type"]
B_BOOT = 10000


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    raise ValueError(f"{path}:{n}: invalid JSONL: {e}") from e
    return rows


def safe_rate(num: int, den: int) -> float:
    return float(num / den) if den else float("nan")


def bootstrap_rate(vals: list[bool], b: int = B_BOOT, seed: int = 20260614) -> tuple[float, list[float]]:
    arr = np.array([int(v) for v in vals], dtype=np.float64)
    rng = np.random.RandomState(seed)
    if len(arr) == 0:
        return float("nan"), [float("nan"), float("nan")]
    bs = [float(np.mean(arr[rng.randint(0, len(arr), len(arr))])) for _ in range(b)]
    return float(arr.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def bootstrap_delta(vals_a: list[bool], vals_b: list[bool], b: int = B_BOOT, seed: int = 20260614) -> tuple[float, list[float]]:
    a = np.array([int(v) for v in vals_a], dtype=np.float64)
    c = np.array([int(v) for v in vals_b], dtype=np.float64)
    d = a - c
    rng = np.random.RandomState(seed)
    bs = [float(np.mean(d[rng.randint(0, len(d), len(d))])) for _ in range(b)]
    return float(d.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def median(vals: list[float]) -> float:
    vals = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    return float(statistics.median(vals)) if vals else 0.0


def p95(vals: list[float]) -> float:
    vals = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    return float(np.percentile(vals, 95)) if vals else 0.0


def main() -> int:
    blockers: list[str] = []
    warnings: list[str] = []

    required = [
        "results_smoke.json", "traces_smoke.jsonl", "ledger_smoke.jsonl",
        "results_main.json", "traces_main.jsonl", "ledger_main.jsonl",
        "results_pilot.json", "traces_pilot.jsonl", "ledger_pilot.jsonl",
        f"{RUN_ID}_receipt.yaml",
    ]
    missing = [name for name in required if not (RUN_DIR / name).exists()]
    if missing:
        blockers.append(f"missing required artifacts: {missing}")

    schema = read_json(SCHEMA_PATH)
    validator = Draft7Validator(schema)
    manifest = yaml.safe_load(MANIFEST_PATH.read_text())
    receipt = yaml.safe_load(RECEIPT_PATH.read_text()) if RECEIPT_PATH.exists() else {}
    byte_pred = yaml.safe_load(BYTE_PRED_PATH.read_text())

    fl_ids = list(manifest["fl_subset_ids"])
    smoke_ids = list(manifest["smoke_set_ids"])
    fl_id_hash = sha16(json.dumps(sorted(fl_ids)))
    manifest_hash = manifest.get("manifest_hash")

    results = {tag: read_json(RUN_DIR / f"results_{tag}.json") for tag in ["smoke", "main", "pilot"] if (RUN_DIR / f"results_{tag}.json").exists()}
    traces = {tag: read_jsonl(RUN_DIR / f"traces_{tag}.jsonl") for tag in ["smoke", "main", "pilot"] if (RUN_DIR / f"traces_{tag}.jsonl").exists()}
    ledgers = {tag: read_jsonl(RUN_DIR / f"ledger_{tag}.jsonl") for tag in ["smoke", "main", "pilot"] if (RUN_DIR / f"ledger_{tag}.jsonl").exists()}

    # Completeness/pairing checks.
    expected_rows = {"smoke": len(smoke_ids) * len(POLICIES), "main": len(fl_ids) * len(POLICIES) * len(SEEDS), "pilot": len(fl_ids) * len(POLICIES) * len(SEEDS)}
    expected_traces = {tag: expected_rows[tag] * 14 for tag in expected_rows}
    for tag, exp in expected_rows.items():
        got = len(results.get(tag, []))
        if got != exp:
            blockers.append(f"{tag} results rows {got} != expected {exp}")
        lg = len(ledgers.get(tag, []))
        if lg != got:
            blockers.append(f"{tag} ledger rows {lg} != results rows {got}")
        tg = len(traces.get(tag, []))
        if tg != expected_traces[tag]:
            blockers.append(f"{tag} trace rows {tg} != expected {expected_traces[tag]}")

    # Trace schema checks.
    trace_summary: dict[str, Any] = {}
    for tag, rows in traces.items():
        schema_errors = 0
        reasons = collections.Counter()
        modes = collections.Counter()
        rungs = collections.Counter()
        decisions = collections.Counter()
        for row in rows:
            err = next(validator.iter_errors(row), None)
            if err is not None:
                schema_errors += 1
            reasons[row.get("reason_code", "MISSING")] += 1
            modes[row.get("execution_mode", "MISSING")] += 1
            rungs[row.get("estimator_rung", "MISSING")] += 1
            decisions[row.get("decision", "MISSING")] += 1
        if schema_errors:
            blockers.append(f"{tag} trace schema errors: {schema_errors}")
        if set(reasons) != {"FORCED_BASELINE"}:
            blockers.append(f"{tag} reason-code coverage not 100% FORCED_BASELINE: {dict(reasons)}")
        trace_summary[tag] = {"rows": len(rows), "schema_errors": schema_errors, "reason_codes": dict(reasons), "execution_modes": dict(modes), "estimator_rungs": dict(rungs), "decisions": dict(decisions)}

    # Main pairing by instance/policy/seed.
    main = results.get("main", [])
    keys = collections.Counter((r["instance_id"], r["policy"], int(r["seed"])) for r in main)
    dup = [k for k, v in keys.items() if v != 1]
    if dup:
        blockers.append(f"duplicate/missing uniqueness issue in main keys; examples={dup[:5]}")
    by_instance: dict[str, dict[str, list[bool]]] = {iid: {p: [] for p in POLICIES} for iid in fl_ids}
    family_by_id: dict[str, str] = {}
    for r in main:
        iid = r["instance_id"]
        if iid not in by_instance:
            blockers.append(f"main result has id outside manifest: {iid}")
            continue
        if int(r["seed"]) not in SEEDS:
            blockers.append(f"main result has unexpected seed: {r}")
        if r["policy"] not in POLICIES:
            blockers.append(f"main result has unexpected policy: {r}")
        by_instance[iid][r["policy"]].append(bool(r["passed"]))
        family_by_id[iid] = r["family"]
    paired_ids = [iid for iid in fl_ids if all(len(by_instance[iid][p]) == len(SEEDS) for p in POLICIES)]
    if len(paired_ids) != len(fl_ids):
        blockers.append(f"paired main instance count {len(paired_ids)} != manifest fl count {len(fl_ids)}")

    instance_pass = {p: [any(by_instance[iid][p]) for iid in paired_ids] for p in POLICIES}
    per_policy = {}
    for p in POLICIES:
        m, ci = bootstrap_rate(instance_pass[p])
        per_policy[p] = {"passes": int(sum(instance_pass[p])), "n": len(instance_pass[p]), "rate": m, "ci95": ci}
    dr, dr_ci = bootstrap_delta(instance_pass["dense_uniform"], instance_pass["random_matched_mix"])
    er, er_ci = bootstrap_delta(instance_pass["heuristic_entropy"], instance_pass["random_matched_mix"])
    de, de_ci = bootstrap_delta(instance_pass["dense_uniform"], instance_pass["heuristic_entropy"])

    per_family: dict[str, Any] = {}
    for fam in FAMILIES:
        ids = [iid for iid in paired_ids if family_by_id.get(iid) == fam]
        per_family[fam] = {}
        for p in POLICIES:
            vals = [any(by_instance[iid][p]) for iid in ids]
            per_family[fam][p] = {"passes": int(sum(vals)), "n": len(vals), "rate": safe_rate(int(sum(vals)), len(vals))}

    # McNemar dense vs random.
    dense_vals = instance_pass["dense_uniform"]
    random_vals = instance_pass["random_matched_mix"]
    a = sum(1 for x, y in zip(dense_vals, random_vals) if x and y)
    b = sum(1 for x, y in zip(dense_vals, random_vals) if x and not y)
    c = sum(1 for x, y in zip(dense_vals, random_vals) if not x and y)
    d = sum(1 for x, y in zip(dense_vals, random_vals) if not x and not y)

    # Ledger throughput.
    tps_by_tag_policy: dict[str, dict[str, dict[str, float]]] = {}
    for tag, rows in ledgers.items():
        tps_by_tag_policy[tag] = {}
        for p in POLICIES:
            vals = [float(r["cost"]["wall_clock_sustained_tps"]["p50"]) for r in rows if r.get("policy", {}).get("baseline_type") == p]
            tps_by_tag_policy[tag][p] = {"median": median(vals), "p95": p95(vals), "n": len(vals)}

    pred_dense = float(byte_pred["throughput_predictions"]["dense_predicted_tps_p50"])
    pred_skip = float(byte_pred["throughput_predictions"]["skip_predicted_tps_p50"])
    main_dense_tps = tps_by_tag_policy.get("main", {}).get("dense_uniform", {}).get("median", 0.0)
    main_random_tps = tps_by_tag_policy.get("main", {}).get("random_matched_mix", {}).get("median", 0.0)
    main_entropy_tps = tps_by_tag_policy.get("main", {}).get("heuristic_entropy", {}).get("median", 0.0)
    forced_vals = []
    for p in ["random_matched_mix", "heuristic_entropy"]:
        forced_vals.extend([float(r["cost"]["wall_clock_sustained_tps"]["p50"]) for r in ledgers.get("main", []) if r.get("policy", {}).get("baseline_type") == p])
    forced_median = median(forced_vals)
    dense_err = abs(main_dense_tps - pred_dense) / pred_dense if pred_dense else 1.0
    skip_err = abs(forced_median - pred_skip) / pred_skip if pred_skip else 1.0

    hfl = {
        "H-FL-1": {"pass": len(traces.get("main", [])) == expected_traces["main"] and trace_summary.get("main", {}).get("schema_errors") == 0, "detail": f"main traces={len(traces.get('main', []))}, schema_errors={trace_summary.get('main', {}).get('schema_errors')}"},
        "H-FL-2": {"pass": dr_ci[0] > 0, "delta": dr, "ci95": dr_ci, "detail": f"dense-random {dr*100:+.1f}pp CI[{dr_ci[0]*100:+.1f},{dr_ci[1]*100:+.1f}]"},
        "H-FL-3": {"entropy_wins": er_ci[0] > 0, "tie": abs(er_ci[0]) <= 0.08 and abs(er_ci[1]) <= 0.08, "delta": er, "ci95": er_ci, "detail": f"entropy-random {er*100:+.1f}pp CI[{er_ci[0]*100:+.1f},{er_ci[1]*100:+.1f}]"},
        "H-FL-4": {"pass": dense_err <= 0.20 and skip_err <= 0.20, "dense_error": dense_err, "skip_error": skip_err, "detail": f"dense {main_dense_tps:.2f} vs {pred_dense:.2f} ({dense_err*100:.1f}%); forced {forced_median:.2f} vs {pred_skip:.2f} ({skip_err*100:.1f}%)"},
    }

    # Receipt cross-checks.
    receipt_checks = {}
    if receipt:
        receipt_checks["run_id_matches"] = receipt.get("run_id") == RUN_ID
        receipt_checks["manifest_hash_matches"] = receipt.get("gym_manifest_hash") == manifest_hash
        receipt_checks["p2_hash_matches_manifest"] = receipt.get("self_proofs", {}).get("p2_same_instances", {}).get("id_hash") == fl_id_hash
        receipt_checks["p2_n_matches"] = receipt.get("self_proofs", {}).get("p2_same_instances", {}).get("n") == len(fl_ids)
        receipt_checks["per_policy_matches"] = {
            p: abs(float(receipt.get("analysis", {}).get("per_policy", {}).get(p, {}).get("rate", -1.0)) - per_policy[p]["rate"]) < 1e-12
            for p in POLICIES
        }
        for k, ok in receipt_checks.items():
            if isinstance(ok, bool) and not ok:
                blockers.append(f"receipt cross-check failed: {k}")
            elif isinstance(ok, dict) and not all(ok.values()):
                blockers.append(f"receipt cross-check failed: {k}={ok}")
        if (RUN_DIR / f"{RUN_ID}_receipt.json").exists() is False:
            warnings.append("receipt JSON missing; YAML exists and parses. Normalize to JSON if downstream tooling requires it.")

    # Ledger field caveats.
    zero_byte_rows = sum(1 for r in ledgers.get("main", []) if r.get("cost", {}).get("bytes_per_token") == 0.0)
    zero_flop_rows = sum(1 for r in ledgers.get("main", []) if r.get("cost", {}).get("flops_total") == 0.0)
    if zero_byte_rows:
        warnings.append(f"main ledger bytes_per_token is zero on {zero_byte_rows} rows; OK only if v4 contract treats TPS-only ledger as sufficient.")
    if zero_flop_rows:
        warnings.append(f"main ledger flops_total is zero on {zero_flop_rows} rows; OK only if v4 contract treats TPS-only ledger as sufficient.")
    # Pilot policy caveat.
    pilot_policies = collections.Counter(r.get("policy") for r in results.get("pilot", []))
    if set(pilot_policies) == set(POLICIES):
        warnings.append("pilot ran all three policies (2700 rows), not dense-only; acceptable if v4 audit/prereg intended all-policy sampling.")

    verdict = "BLOCKED_RERUN_RECOMMENDED" if blockers else "VALIDATED_NO_RERUN_NEEDED"
    report = {
        "run_id": RUN_ID,
        "verdict": verdict,
        "blockers": blockers,
        "warnings": warnings,
        "manifest": {"manifest_hash": manifest_hash, "fl_subset_hash_computed": fl_id_hash, "n_fl": len(fl_ids), "n_smoke": len(smoke_ids)},
        "artifact_counts": {tag: {"results": len(results.get(tag, [])), "traces": len(traces.get(tag, [])), "ledger": len(ledgers.get(tag, [])), "expected_results": expected_rows[tag], "expected_traces": expected_traces[tag]} for tag in expected_rows},
        "trace_summary": trace_summary,
        "per_policy": per_policy,
        "paired_deltas": {"dense_minus_random": {"mean": dr, "ci95": dr_ci}, "entropy_minus_random": {"mean": er, "ci95": er_ci}, "dense_minus_entropy": {"mean": de, "ci95": de_ci}},
        "mcnemar_dense_vs_random": {"a": a, "b": b, "c": c, "d": d},
        "per_family": per_family,
        "throughput": {"predicted_dense_p50": pred_dense, "predicted_skip_p50": pred_skip, "main_policy_tps": tps_by_tag_policy.get("main", {}), "forced_skip_combined_median": forced_median, "dense_abs_pct_error": dense_err, "skip_abs_pct_error": skip_err},
        "hypotheses": hfl,
        "receipt_checks": receipt_checks,
        "source_artifacts": {"receipt_yaml": str(RECEIPT_PATH), "run_dir": str(RUN_DIR), "schema": str(SCHEMA_PATH), "manifest": str(MANIFEST_PATH), "byte_predictions": str(BYTE_PRED_PATH)},
    }

    out_json = RUN_DIR / f"{RUN_ID}_INDEPENDENT_VALIDATION.json"
    out_md = RUN_DIR / f"{RUN_ID}_INDEPENDENT_VALIDATION.md"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=False))

    def pct(x: float) -> str:
        return f"{x*100:.1f}%"

    lines = [
        f"# Independent validation — {RUN_ID}",
        "",
        f"Verdict: **{verdict}**",
        "",
        "## Blockers",
    ]
    lines.extend([f"- {b}" for b in blockers] if blockers else ["- None"])
    lines.extend([
        "",
        "## Warnings",
    ])
    lines.extend([f"- {w}" for w in warnings] if warnings else ["- None"])
    lines.extend([
        "",
        "## H-FL verdicts",
        "",
        "| Hypothesis | Result | Detail |",
        "|---|---|---|",
        f"| H-FL-1 | {'PASS' if hfl['H-FL-1']['pass'] else 'FAIL'} | {hfl['H-FL-1']['detail']} |",
        f"| H-FL-2 | {'PASS' if hfl['H-FL-2']['pass'] else 'FAIL'} | {hfl['H-FL-2']['detail']} |",
        f"| H-FL-3 | {'ENTROPY_WINS' if hfl['H-FL-3']['entropy_wins'] else ('TIE' if hfl['H-FL-3']['tie'] else 'FAIL_ENTROPY_BEHIND')} | {hfl['H-FL-3']['detail']} |",
        f"| H-FL-4 | {'PASS' if hfl['H-FL-4']['pass'] else 'FAIL'} | {hfl['H-FL-4']['detail']} |",
        "",
        "## Per policy",
        "",
        "| Policy | Passes/n | Rate | 95% CI |",
        "|---|---:|---:|---:|",
    ])
    for p in POLICIES:
        pp = per_policy[p]
        lines.append(f"| {p} | {pp['passes']}/{pp['n']} | {pct(pp['rate'])} | [{pct(pp['ci95'][0])}, {pct(pp['ci95'][1])}] |")
    lines += [
        "",
        "## Per family",
        "",
        "| Family | dense | random | entropy |",
        "|---|---:|---:|---:|",
    ]
    for fam in FAMILIES:
        pf = per_family[fam]
        lines.append(f"| {fam} | {pf['dense_uniform']['passes']}/{pf['dense_uniform']['n']} ({pct(pf['dense_uniform']['rate'])}) | {pf['random_matched_mix']['passes']}/{pf['random_matched_mix']['n']} ({pct(pf['random_matched_mix']['rate'])}) | {pf['heuristic_entropy']['passes']}/{pf['heuristic_entropy']['n']} ({pct(pf['heuristic_entropy']['rate'])}) |")
    lines += [
        "",
        "## Throughput",
        "",
        f"- Dense median TPS: {main_dense_tps:.4f}; predicted {pred_dense:.4f}; error {dense_err*100:.1f}%",
        f"- Forced skip combined median TPS: {forced_median:.4f}; predicted {pred_skip:.4f}; error {skip_err*100:.1f}%",
        f"- Random median TPS: {main_random_tps:.4f}",
        f"- Entropy median TPS: {main_entropy_tps:.4f}",
        "",
        "## Artifacts",
        f"- JSON report: `{out_json}`",
        f"- Run dir: `{RUN_DIR}`",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"verdict": verdict, "blockers": blockers, "warnings": warnings, "report_json": str(out_json), "report_md": str(out_md), "hfl": hfl}, indent=2))
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
