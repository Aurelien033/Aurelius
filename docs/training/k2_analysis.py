#!/usr/bin/env python3
"""Selector ablation on the exact k=2 pair matrix (CPU-only; reads rows.jsonl).

The decisive test post-v2: a near-dense skip set EXISTS at k=2 (true ceiling 96.7%), but can a CHEAP
selector FIND it on tasks it has never seen — or are good pairs only reachable by verifier search?
Uses TASK-LEVEL cross-validation (train on some tasks, evaluate on UNSEEN tasks) so we measure
generalization, not memorization of task difficulty.

Cheap selectors (no prompt features -> computable from the matrix alone):
  random           : mean pass over all 91 pairs  (the bar to beat)
  global_best_pair : rank pairs by TRAIN pass-rate, pick top-1, apply to TEST tasks  (structural top-1)
  best_of_top{m}   : from the top-m TRAIN pairs, a TEST task counts solved if ANY passes  (cheap multi-try)
  dense            : the full model (context)
  ceiling          : oracle, any of 91 passes (upper bound)

If global_best_pair >> random on held-out tasks -> a trivially cheap, zero-feature selector works:
selection is learnable and cheap. If global_best_pair ~= random but best_of_topM is high -> good pairs
are task-specific (need prompt features / a few tries). If nothing beats random -> good pairs exist but
aren't cheaply findable (the honest negative). A prompt-conditioned (activation-feature) selector needs
model forwards and is a separate follow-up.
"""
import json, sys, random
from collections import defaultdict
import numpy as np


def main(path):
    rows = [json.loads(l) for l in open(path)]
    tasks = sorted({r["iid"] for r in rows})
    pairs = sorted({tuple(r["pair"]) for r in rows if r["pair"] is not None})
    M = defaultdict(dict); dense = {}
    for r in rows:
        if r["dense"]: dense[r["iid"]] = r["passed"]
        else: M[r["iid"]][tuple(r["pair"])] = r["passed"]

    # ---- independent re-derivation of the headline (verification) ----
    ceiling = np.mean([1 if any(M[t].values()) else 0 for t in tasks])
    avg_random = np.mean([v for t in tasks for v in M[t].values()])
    dense_rate = np.mean([dense[t] for t in tasks]) if dense else float("nan")
    print(f"RE-DERIVED from matrix ({len(tasks)} tasks × {len(pairs)} pairs): "
          f"ceiling {ceiling*100:.1f}%  avg-random {avg_random*100:.1f}%  dense {dense_rate*100:.1f}%")
    by_fam = defaultdict(list)
    for r in rows:
        if not r["dense"]: by_fam[r["family"]].append(r["passed"])
    for f, v in by_fam.items():
        print(f"    {f}: avg-random {np.mean(v)*100:.1f}%")

    # ---- task-level K-fold CV of cheap selectors ----
    rng = random.Random(0); ts = tasks[:]; rng.shuffle(ts); K = 5
    folds = [ts[i::K] for i in range(K)]
    res = defaultdict(list); chosen = []
    for k in range(K):
        test = set(folds[k]); train = [t for t in tasks if t not in test]
        pr = {p: np.mean([M[t][p] for t in train]) for p in pairs}
        ranked = sorted(pairs, key=lambda p: -pr[p]); best = ranked[0]; chosen.append(best)
        res["random"].append(np.mean([v for t in test for v in M[t].values()]))
        res["dense"].append(np.mean([dense[t] for t in test]) if dense else float("nan"))
        res["global_best_pair"].append(np.mean([M[t][best] for t in test]))
        for m in [3, 5, 10]:
            topm = ranked[:m]
            res[f"best_of_top{m}"].append(np.mean([1 if any(M[t][p] for p in topm) else 0 for t in test]))
        res["ceiling"].append(np.mean([1 if any(M[t].values()) else 0 for t in test]))
    print("\n5-fold task-level CV (evaluated on HELD-OUT tasks):")
    for pol in ["random", "dense", "global_best_pair", "best_of_top3", "best_of_top5", "best_of_top10", "ceiling"]:
        v = np.array(res[pol]); print(f"  {pol:18s}: {v.mean()*100:5.1f}%  (±{v.std()*100:.1f})")
    print(f"  top-1 pair chosen per fold: {[f'{i}-{j}' for i,j in chosen]}  "
          f"(stable = selection signal is robust)")

    # ---- mechanistic structure (full data) ----
    pr_all = {p: np.mean([M[t][p] for t in tasks]) for p in pairs}
    top = sorted(pairs, key=lambda p: -pr_all[p])[:8]; bot = sorted(pairs, key=lambda p: pr_all[p])[:8]
    print("\ntop pairs (full-data pass rate):", [(f"{i}-{j}", round(pr_all[(i,j)]*100)) for i, j in top])
    print("worst pairs:", [(f"{i}-{j}", round(pr_all[(i,j)]*100)) for i, j in bot])
    incl = {}
    for L in range(7, 21):
        inc = [M[t][p] for t in tasks for p in pairs if L in p]
        exc = [M[t][p] for t in tasks for p in pairs if L not in p]
        incl[L] = np.mean(inc) - np.mean(exc)
    print("layer-inclusion effect (pass Δ when a layer is in the skipped pair):")
    for L in sorted(incl, key=lambda L: incl[L]):
        print(f"    L{L}: {incl[L]*100:+.1f}pp")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "/Users/christienantonio/Downloads/k2_matrix/k2_matrix/rows.jsonl")
