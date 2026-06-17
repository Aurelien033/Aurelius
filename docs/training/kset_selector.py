#!/usr/bin/env python3
"""Set-generic selector ablation with leave-tasks-out CV (k>=3 generalization of selector_train).
Reads kset_matrix rows.jsonl + kset_feats features.npz. Numpy-only logistic.

policies (held-out 5-fold CV):
  random          : mean over the S sampled sets                          (floor)
  dense           : full model                                           (the bar)
  global_best_set : top-1 set by TRAIN pass-rate = the PERMANENT PRUNE    (H-K4-0: >= dense?)
  C_struct        : logistic on STRUCTURAL set features (+ global rate)
  C_full          : + per-task PROMPT features (agg layer feats + set logit-space interference) (H-K4-2)
  verified_top{3,5}: any-pass over the top-M train sets (self-verifying)
  sampled_ceiling : best of the S per task (verifier-assisted; LOWER bound on the true k-set oracle)

Run:  python docs/training/kset_selector.py rows.jsonl features.npz
"""
import sys, json
from collections import defaultdict
import numpy as np

POISON = {11, 14, 15}


def struct_set(s):
    s = sorted(s); runs = 1; longest = 1
    for a, b in zip(s, s[1:]):
        if b - a == 1: runs += 1; longest = max(longest, runs)
        else: runs = 1
    return [len(set(s) & POISON), sum(1 for a, b in zip(s, s[1:]) if b - a == 1), longest,
            (s[-1] - s[0]) / 13, sum(1 for x in s if x <= 10), sum(1 for x in s if x >= 17),
            (sum(s) / len(s) - 7) / 13]


def logistic_fit(X, y, l2=1.0, iters=400, lr=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))]); w = np.zeros(Xs.shape[1]); n = len(y)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xs @ w)); w -= lr * (Xs.T @ (p - y) / n + l2 * np.r_[w[:-1], 0] / n)
    return mu, sd, w


def logistic_pred(m, X):
    mu, sd, w = m
    return 1 / (1 + np.exp(-np.hstack([(X - mu) / sd, np.ones((len(X), 1))]) @ w))


def main(matrix_path, feats_path):
    rows = [json.loads(l) for l in open(matrix_path)]
    F = np.load(feats_path, allow_pickle=True)
    task_ids = [str(t) for t in F["task_ids"]]
    sets = [tuple(int(x) for x in s) for s in F["sets"]]
    layer_feats, set_interf = F["layer_feats"], F["set_interf"]
    routable = [int(x) for x in F["routable"]]
    Lpos = {L: k for k, L in enumerate(routable)}; sidx = {s: k for k, s in enumerate(sets)}
    tfi = {t: k for k, t in enumerate(task_ids)}
    M = defaultdict(dict); dense = {}
    for r in rows:
        if r["dense"]: dense[r["iid"]] = int(r["passed"])
        else: M[r["iid"]][tuple(r["set"])] = int(r["passed"])
    tasks = [t for t in task_ids if t in M and len(M[t]) == len(sets)]
    k = len(sets[0])
    print(f"k={k}: tasks with full matrix + features: {len(tasks)} (feat {len(task_ids)}, matrix {len(M)}); {len(sets)} sets")

    SS = {s: np.array(struct_set(s)) for s in sets}

    def agg_layer(t, s):
        return list(np.mean([layer_feats[tfi[t], Lpos[L]] for L in s], axis=0))

    def build(task_list, use_prompt, rate):
        X, y, key = [], [], []
        for t in task_list:
            for s in sets:
                f = list(SS[s]) + [rate[s]]
                if use_prompt: f += agg_layer(t, s) + [set_interf[tfi[t], sidx[s]]]
                X.append(f); y.append(M[t][s]); key.append((t, s))
        return np.array(X, float), np.array(y, float), key

    inter = np.array([set_interf[tfi[t], sidx[s]] for t in tasks for s in sets])
    yy = np.array([M[t][s] for t in tasks for s in sets])
    print(f"EARLY READ: corr(set_interference, pass) = {np.corrcoef(inter, yy)[0,1]:+.3f}  (|.|~0 => feature weak)")

    rng = np.random.RandomState(0); ts = tasks[:]; rng.shuffle(ts); K = 5
    folds = [ts[i::K] for i in range(K)]
    res = defaultdict(list)
    for kf in range(K):
        test = folds[kf]; train = [t for t in tasks if t not in set(test)]
        rate = {s: np.mean([M[t][s] for t in train]) for s in sets}
        ranked = sorted(sets, key=lambda s: -rate[s]); best = ranked[0]
        res["random"].append(np.mean([M[t][s] for t in test for s in sets]))
        res["dense"].append(np.mean([dense[t] for t in test]))
        res["global_best_set"].append(np.mean([M[t][best] for t in test]))
        res["sampled_ceiling"].append(np.mean([1 if any(M[t].values()) else 0 for t in test]))
        for Mtop in [3, 5]:
            top = ranked[:Mtop]
            res[f"verified_top{Mtop}"].append(np.mean([1 if any(M[t][s] for s in top) else 0 for t in test]))
        for up, name in [(False, "C_struct"), (True, "C_full")]:
            Xtr, ytr, _ = build(train, up, rate); mdl = logistic_fit(Xtr, ytr)
            sel = []
            for t in test:
                Xt, _, keyt = build([t], up, rate); P = logistic_pred(mdl, Xt)
                sel.append(M[t][keyt[int(np.argmax(P))][1]])
            res[name].append(np.mean(sel))
    print(f"\n5-fold task-level CV (held-out tasks), k={k}:")
    for n in ["random", "dense", "global_best_set", "C_struct", "C_full", "verified_top3", "verified_top5", "sampled_ceiling"]:
        v = np.array(res[n]); print(f"  {n:16s}: {v.mean()*100:5.1f}%  (±{v.std()*100:.1f})")
    g, d, cf, cs = (np.array(res[x]) for x in ["global_best_set", "dense", "C_full", "C_struct"])
    print(f"\n  H-K4-0  global_best_set − dense = {(g.mean()-d.mean())*100:+.1f}pp  (a PERMANENT {k}-layer prune >= full model? the prize)")
    print(f"  H-K4-2  C_full − C_struct       = {(cf.mean()-cs.mean())*100:+.1f}pp  (per-task PROMPT signal at k={k}?)")
    pr = {s: np.mean([M[t][s] for t in tasks]) for s in sets}
    top = sorted(sets, key=lambda s: -pr[s])[:6]
    print("\ntop sets (full-data rate):", [(",".join(map(str, s)), round(pr[s] * 100)) for s in top])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
