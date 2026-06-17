#!/usr/bin/env python3
"""Per-task selector ablation A/B/C with leave-tasks-out CV (numpy-only; reads rows.jsonl + features.npz).

The decisive test: does a PROMPT-CONDITIONED selector (C_full) beat the best GLOBAL pair (B) on HELD-OUT
tasks, and is the gain due to the PROMPT features (C_full > C_struct)?
  A random        : mean over 91 pairs                  (floor)
  B global_pair   : top-1 pair by TRAIN pass-rate        (the bound to beat)
  C_struct        : logistic on STRUCTURAL pair features (controls for "a learned model")
  C_full          : logistic on STRUCTURAL + PROMPT feat (the test condition)
  oracle          : per-task best of 91                  (ceiling)

Run:  python docs/training/selector_train.py rows.jsonl features.npz
"""
import sys, json, itertools
from collections import defaultdict
import numpy as np

ROUTABLE = list(range(7, 21))
PAIRS = list(itertools.combinations(ROUTABLE, 2))


def struct_feats(i, j):
    return [(i - 7) / 13, (j - 7) / 13, abs(i - j) / 13, 1.0 if abs(i - j) == 1 else 0.0,
            1.0 if 11 in (i, j) else 0.0, 1.0 if 14 in (i, j) else 0.0, 1.0 if 15 in (i, j) else 0.0,
            float((i <= 10) + (j <= 10)), float((i >= 17) + (j >= 17))]


def logistic_fit(X, y, l2=1.0, iters=400, lr=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Xs.shape[1]); n = len(y)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xs @ w))
        g = Xs.T @ (p - y) / n + l2 * np.r_[w[:-1], 0] / n
        w -= lr * g
    return mu, sd, w


def logistic_pred(model, X):
    mu, sd, w = model
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    return 1 / (1 + np.exp(-Xs @ w))


def main(matrix_path, feats_path):
    rows = [json.loads(l) for l in open(matrix_path)]
    F = np.load(feats_path, allow_pickle=True)
    task_ids = [str(t) for t in F["task_ids"]]
    pairs = [tuple(int(x) for x in p) for p in F["pairs"]]
    layer_feats, pair_align = F["layer_feats"], F["pair_align"]
    routable = [int(x) for x in F["routable"]]
    Lidx = {L: k for k, L in enumerate(routable)}; pidx = {p: k for k, p in enumerate(pairs)}
    tfi = {t: k for k, t in enumerate(task_ids)}

    M = defaultdict(dict); dense = {}; fam = {}
    for r in rows:
        fam[r["iid"]] = r["family"]
        if r["dense"]: dense[r["iid"]] = int(r["passed"])
        else: M[r["iid"]][tuple(r["pair"])] = int(r["passed"])
    tasks = [t for t in task_ids if t in M and len(M[t]) == len(PAIRS)]
    print(f"tasks with full matrix + features: {len(tasks)}  (feat {len(task_ids)}, matrix {len(M)})")
    SF = {p: np.array(struct_feats(*p)) for p in PAIRS}

    def build(task_list, use_prompt, rate):
        # rate[p] = global TRAIN pass-rate (so C_struct can recover the B baseline; C_full adds prompt on top)
        X, y, key = [], [], []
        for t in task_list:
            tf = tfi[t]
            for p in PAIRS:
                f = list(SF[p]) + [rate[p]]
                if use_prompt:
                    f += list(layer_feats[tf, Lidx[p[0]]]) + list(layer_feats[tf, Lidx[p[1]]]) + [pair_align[tf, pidx[p]]]
                X.append(f); y.append(M[t][p]); key.append((t, p))
        return np.array(X, float), np.array(y, float), key

    # early read: does the mechanistic pair_align feature correlate with pass at all?
    pa = np.array([pair_align[tfi[t], pidx[p]] for t in tasks for p in PAIRS])
    yy = np.array([M[t][p] for t in tasks for p in PAIRS])
    print(f"EARLY READ: corr(pair_align, pass) = {np.corrcoef(pa, yy)[0,1]:+.3f}  (|.|~0 => headline feature weak)")

    rng = np.random.RandomState(0); ts = tasks[:]; rng.shuffle(ts); K = 5
    folds = [ts[i::K] for i in range(K)]
    res = defaultdict(list)
    for k in range(K):
        test = folds[k]; train = [t for t in tasks if t not in set(test)]
        rate = {p: np.mean([M[t][p] for t in train]) for p in PAIRS}
        bestB = max(PAIRS, key=lambda p: rate[p])
        res["random"].append(np.mean([M[t][p] for t in test for p in PAIRS]))
        res["dense"].append(np.mean([dense[t] for t in test]))
        res["B_global_pair"].append(np.mean([M[t][bestB] for t in test]))
        res["oracle"].append(np.mean([1 if any(M[t].values()) else 0 for t in test]))
        for use_prompt, name in [(False, "C_struct"), (True, "C_full")]:
            Xtr, ytr, _ = build(train, use_prompt, rate); model = logistic_fit(Xtr, ytr)
            sel = []
            for t in test:
                Xt, _, keyt = build([t], use_prompt, rate); P = logistic_pred(model, Xt)
                sel.append(M[t][keyt[int(np.argmax(P))][1]])
            res[name].append(np.mean(sel))
    print("\n5-fold task-level CV (held-out tasks):")
    for name in ["random", "B_global_pair", "C_struct", "C_full", "dense", "oracle"]:
        v = np.array(res[name]); print(f"  {name:14s}: {v.mean()*100:5.1f}%  (±{v.std()*100:.1f})")
    cf, b, cs = (np.array(res[x]) for x in ["C_full", "B_global_pair", "C_struct"])
    print(f"\n  H-SEL-0  C_full − B        = {(cf.mean()-b.mean())*100:+.1f}pp  (prompt selector beats global pair?)")
    print(f"  H-SEL-1  C_full − C_struct = {(cf.mean()-cs.mean())*100:+.1f}pp  (gain from PROMPT features, not just a model?)")
    print("  VERDICT: H-SEL-0 positive (CI>0) => prompt-conditioned routing is REAL; ~0 => honest KILL.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
