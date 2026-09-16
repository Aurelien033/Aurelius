#!/usr/bin/env python3
"""flip_analysis.py — base-vs-model flip / regression table from two `eval_code_bench.py --dump` files.

The aggregate pass@1 can be identical (e.g. MBPP 125 == 125) while the two models actually solve a
DIFFERENT set of problems. This aligns the two dumps by task id and shows the 2x2 contingency so you
can tell "RLVR changed almost nothing" (high overlap, ~0 churn) from "RLVR reshuffled" (real churn,
possibly with regressions).

  # 1) produce two dumps (greedy-only is enough + fast; add --passk for the oracle table too)
  python docs/training/eval_code_bench.py --model Qwen/Qwen3-8B        --bench mbpp --n 200 --dump base_mbpp.json
  python docs/training/eval_code_bench.py --model /content/aurelius-rlvr --bench mbpp --n 200 --dump rlvr_mbpp.json
  # 2) compare (A=base, B=rlvr)
  python docs/training/flip_analysis.py base_mbpp.json rlvr_mbpp.json
"""
import json, sys


def load(path):
    d = json.loads(open(path).read())
    return d, {r["tid"]: r for r in d["results"]}


def table(am, bm, key):
    tids = [t for t in am if t in bm]
    if any(am[t][key] is None or bm[t][key] is None for t in tids):
        return None                                                  # key absent in one dump (no --passk)
    both = a_only = b_only = neither = 0
    a_only_ids, b_only_ids = [], []
    for t in tids:
        av, bv = am[t][key], bm[t][key]
        if   av and bv:      both += 1
        elif av and not bv:  a_only += 1; a_only_ids.append(t)        # B (rlvr) REGRESSED these
        elif not av and bv:  b_only += 1; b_only_ids.append(t)        # B (rlvr) GAINED these
        else:                neither += 1
    unmatched = len(set(am) ^ set(bm))
    return dict(n=len(tids), both=both, a_only=a_only, b_only=b_only, neither=neither,
                a_pass=both + a_only, b_pass=both + b_only, net=b_only - a_only,
                churn=a_only + b_only, a_only_ids=a_only_ids, b_only_ids=b_only_ids,
                unmatched=unmatched)


def show(t, key, an, bn):
    if t is None:
        print(f"\n[{key}] unavailable — one dump lacked --passk"); return
    print(f"\n=== {key.upper()} flip table   A={an}   B={bn} ===")
    print(f"  aligned problems     : {t['n']}" + (f"   (+{t['unmatched']} unmatched, ignored)" if t['unmatched'] else ""))
    print(f"  both pass            : {t['both']}")
    print(f"  A-only  (B regressed): {t['a_only']}   {t['a_only_ids'][:15]}")
    print(f"  B-only  (B gained)   : {t['b_only']}   {t['b_only_ids'][:15]}")
    print(f"  both fail            : {t['neither']}")
    print(f"  A pass {t['a_pass']}  |  B pass {t['b_pass']}  |  NET (B-A) {t['net']:+d}  |  churn {t['churn']}")
    if t['churn'] == 0:
        print("  READ: zero churn -> B solves the IDENTICAL set as A (the adapter changed nothing decisive).")
    else:
        print(f"  READ: churn {t['churn']} -> models differ per-problem even if the totals look close;"
              f" {t['a_only']} regressions vs {t['b_only']} gains.")


def main():
    if len(sys.argv) != 3:
        print("usage: flip_analysis.py A.json B.json   (A=baseline, B=candidate)"); sys.exit(1)
    da, am = load(sys.argv[1]); db, bm = load(sys.argv[2])
    an, bn = da["model"], db["model"]
    print(f"A = {an}   ({da['bench']}, n={da['n']}, think={da.get('think')})")
    print(f"B = {bn}   ({db['bench']}, n={db['n']}, think={db.get('think')})")
    if da["bench"] != db["bench"]:
        print("WARNING: different benchmarks — flip table is not meaningful")
    show(table(am, bm, "greedy"), "greedy", an, bn)
    show(table(am, bm, "oracle"), "oracle", an, bn)


if __name__ == "__main__":
    main()
