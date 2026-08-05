#!/usr/bin/env python3
"""Post-run re-analysis for First Light — INDEPENDENT of the runner's own analysis stage.

Motivation (verification 2026-06-13): the v3 runner's analyze() fabricates H-FL-4 from hardcoded
placeholders (measured_tps_dense=15.0 #TODO) and computes H-FL-1 as a tautology. This script
reconstructs ALL FOUR hypothesis verdicts from the raw on-disk artifacts only — traces.jsonl,
checkpoint.jsonl, ledger.jsonl — so the receipt's numbers can be confirmed or overridden without
trusting (or re-running) the model. Pure stdlib + the trace schema. No torch, no MPS.

Usage: python postrun_reanalysis.py <run_dir>   (defaults to the smoke artifacts for dry-run)
"""
import json
import statistics
import sys
from pathlib import Path

RESEARCH = Path("/Users/christienantonio/Desktop/AI:ML Research")
SCHEMA = json.load(open(RESEARCH / "directive_trace_schema.json"))
POLICIES = ["dense_uniform", "random_matched_mix", "heuristic_entropy"]
MDE_BAND = 0.08  # pre-stated ~6-8pp First-Light resolution


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def validate_schema(rows):
    """H-FL-1 done honestly: validate EVERY trace row + check 100% reason-code coverage."""
    try:
        from jsonschema import Draft7Validator
        v = Draft7Validator(SCHEMA)
        errs = sum(1 for r in rows for _ in v.iter_errors(r))
    except ImportError:
        errs = -1  # validator unavailable; report rather than fake
    missing_rc = sum(1 for r in rows if not r.get("reason_code"))
    return {"n_rows": len(rows), "schema_errors": errs, "missing_reason_code": missing_rc,
            "coverage_100pct": (missing_rc == 0 and len(rows) > 0)}


def pass_by_instance(records):
    """instance_id -> policy -> True if any seed passed (paired, instance-level)."""
    out = {}
    for r in records:
        out.setdefault(r["instance_id"], {}).setdefault(r["policy"], []).append(bool(r["passed"]))
    return {iid: {pol: any(v) for pol, v in pols.items()} for iid, pols in out.items()}


def bootstrap_delta_ci(pairs, B=10000, seed=20260612):
    """Paired bootstrap CI of mean(a-b) over instances. pairs = list[(a_bool, b_bool)]."""
    import random
    rng = random.Random(seed)
    n = len(pairs)
    if n == 0:
        return 0.0, (0.0, 0.0)
    point = statistics.fmean(int(a) - int(b) for a, b in pairs)
    means = []
    for _ in range(B):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(int(a) - int(b) for a, b in s))
    means.sort()
    lo = means[int(0.025 * B)]
    hi = means[int(0.975 * B)]
    return point, (lo, hi)


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        RESEARCH / "first_light_receipt/first-light-3651bb11"
    # smoke dry-run uses the *_smoke files; full run uses the bare names
    suffix = "_smoke" if not (run_dir / "checkpoint.jsonl").exists() and \
        (run_dir / "results_smoke.json").exists() else ""
    print(f"Re-analysis of: {run_dir} (suffix='{suffix or 'full'}')")

    # ---- records: prefer checkpoint.jsonl (full run); fall back to results_smoke.json ----
    records = load_jsonl(run_dir / "checkpoint.jsonl")
    if not records and (run_dir / "results_smoke.json").exists():
        records = json.load(open(run_dir / "results_smoke.json"))
    traces = load_jsonl(run_dir / f"traces{suffix}.jsonl") or load_jsonl(run_dir / "traces.jsonl")
    ledger = load_jsonl(run_dir / f"ledger{suffix}.jsonl") or load_jsonl(run_dir / "ledger.jsonl")
    if not records:
        print("  no records yet — run still early"); return

    # Partition to the 300 pinned fl_subset ONLY — checkpoint also holds the 10 smoke instances.
    import yaml
    manifest = yaml.safe_load(open(RESEARCH / "gym-v0.1-FL/gym_manifest.yaml"))
    fl_only = set(manifest["fl_subset_ids"])
    records = [r for r in records if r["instance_id"] in fl_only]
    bi = pass_by_instance(records)
    paired = [iid for iid in bi if all(p in bi[iid] for p in POLICIES)]
    print(f"  records={len(records)}  instances={len(bi)}  fully-paired={len(paired)}")

    # ---- P2 independent: identical instance set per policy ----
    sets = {p: frozenset(iid for iid in bi if p in bi[iid]) for p in POLICIES}
    same_set = len(set(sets.values())) == 1
    print(f"  P2 same-instance-set across policies: {same_set} "
          f"(sizes {{ {', '.join(f'{p.split(chr(95))[0]}:{len(s)}' for p,s in sets.items())} }})")

    # ---- H-FL-1: honest schema + coverage over ALL traces ----
    h1 = validate_schema(traces)
    print(f"  H-FL-1 (recomputed): rows={h1['n_rows']} schema_errors={h1['schema_errors']} "
          f"missing_reason_code={h1['missing_reason_code']} -> "
          f"{'PASS' if h1['coverage_100pct'] and h1['schema_errors'] == 0 else 'CHECK'}")

    # ---- per-policy rates (instance-level, paired) ----
    for p in POLICIES:
        arr = [bi[iid][p] for iid in paired]
        rate = statistics.fmean(int(x) for x in arr) if arr else 0.0
        print(f"    {p:20s} {rate*100:5.1f}%  (n={len(arr)})")

    # ---- H-FL-2 / H-FL-3 from paired bootstrap (reproduce the runner) ----
    dr = bootstrap_delta_ci([(bi[i]["dense_uniform"], bi[i]["random_matched_mix"]) for i in paired])
    re_ = bootstrap_delta_ci([(bi[i]["random_matched_mix"], bi[i]["heuristic_entropy"]) for i in paired])
    print(f"  H-FL-2 Δ(dense-random)={dr[0]*100:+.1f}pp CI[{dr[1][0]*100:+.1f},{dr[1][1]*100:+.1f}] "
          f"-> {'PASS' if dr[1][0] > 0 else 'FAIL->harness-bug protocol'}")
    ent_minus_rand = (-re_[0], (-re_[1][1], -re_[1][0]))
    tie = abs(ent_minus_rand[1][0]) <= MDE_BAND and abs(ent_minus_rand[1][1]) <= MDE_BAND
    verdict3 = "ENTROPY WINS" if ent_minus_rand[1][0] > 0 else ("TIE" if tie else "ENTROPY BEHIND")
    print(f"  H-FL-3 Δ(entropy-random)={ent_minus_rand[0]*100:+.1f}pp "
          f"CI[{ent_minus_rand[1][0]*100:+.1f},{ent_minus_rand[1][1]*100:+.1f}] -> {verdict3}")

    # ---- H-FL-4 HONEST: from measured ledger tps, not the 15.0/18.0 placeholder ----
    if ledger:
        def tps_for(pred):
            vals = []
            for r in ledger:
                w = r.get("wall_clock_sustained_tps", {})
                t = w.get("p50") if isinstance(w, dict) else None
                if t and pred(r):
                    vals.append(t)
            return vals
        dense_tps = tps_for(lambda r: r.get("policy") == "dense_uniform")
        skip_tps = tps_for(lambda r: r.get("policy") in ("random_matched_mix", "heuristic_entropy"))
        def p(v, q):
            return statistics.quantiles(v, n=100)[q-1] if len(v) >= 2 else (v[0] if v else 0.0)
        if dense_tps and skip_tps:
            print(f"  H-FL-4 (recomputed from ledger): dense p50={p(dense_tps,50):.1f} "
                  f"p95={p(dense_tps,95):.1f} tok/s (n={len(dense_tps)}); "
                  f"skip p50={p(skip_tps,50):.1f} p95={p(skip_tps,95):.1f} (n={len(skip_tps)}). "
                  f"Compare vs pinned byte predictions; runner's 15.0/18.0 placeholder is INVALID.")
        else:
            print("  H-FL-4: ledger present but no tps values yet")
    else:
        print("  H-FL-4: no ledger yet")

    print("\n  NOTE: runner analyze() H-FL-4 uses hardcoded 15.0/18.0 (TODO) — OVERRIDE with the "
          "ledger-derived numbers above. H-FL-1 here is recomputed over all traces, not assumed.")


if __name__ == "__main__":
    main()
