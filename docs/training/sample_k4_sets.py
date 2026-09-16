#!/usr/bin/env python3
"""Pin a FIXED sample of S distinct k-sets (default k=4, S=300 of the C(14,4)=1001), seeded.
The SAME S sets are used for every task in the k=4 routing experiment, so the matrix is a clean N x S
and global_best_set is comparable across tasks (the permanent-prune candidate).

Run:  python docs/training/sample_k4_sets.py            # writes eval_data/k4_sets_v0.1.json
"""
import argparse, json, hashlib, random, itertools
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROUTABLE = list(range(7, 21))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--s", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260617)
    ap.add_argument("--out", default=str(REPO / "eval_data" / "k4_sets_v0.1.json"))
    a = ap.parse_args()
    allsets = [list(c) for c in itertools.combinations(ROUTABLE, a.k)]    # 1001 for k=4
    if a.s > len(allsets):
        print(f"  WARN: s={a.s} > total {len(allsets)}; using all"); a.s = len(allsets)
    rng = random.Random(a.seed)
    pick = sorted(rng.sample(allsets, a.s))
    h = hashlib.sha256(json.dumps(pick).encode()).hexdigest()[:16]
    out = {"id": "k4_sets_v0.1", "k": a.k, "s": len(pick), "of_total": len(allsets),
           "coverage_pct": round(100 * len(pick) / len(allsets), 1), "seed": a.seed, "hash": h, "sets": pick}
    Path(a.out).write_text(json.dumps(out))
    print(f"wrote {a.out}: {len(pick)} k={a.k} sets of {len(allsets)} ({out['coverage_pct']}%), hash={h}")


if __name__ == "__main__":
    main()
