#!/usr/bin/env python3
"""E87 Phase B (CPU part) — train the cheap g_gain estimator on Phase-A oracle labels, compute the
rank statistic (H-E87-3), and emit each test instance's k=4 skip-set per policy for the chessboard.

g_gain target: per (instance, layer) the verified GAIN of NOT skipping = V(dense) - V(skip l).
Skip the k layers with the SMALLEST predicted gain (least harmful to skip). Features are cheap
(available from one dense forward pass): entropy H_l, |ΔH_l|, layer index.

The entropy baseline (First Light rung) skips the k smallest-|ΔH| layers — its implied gain-ranking
is monotone in |ΔH|. H-E87-3 compares Spearman ρ(g_gain, realized gain) vs ρ(|ΔH|, realized gain).

CPU-only; runs on partial labels as a dry-run and on full labels for real. The GPU chessboard
(e87_chessboard.py, written next) consumes the emitted skip-sets. No model load here.
"""
import json, sys, statistics
from pathlib import Path

OUT = Path("/Users/christienantonio/Desktop/AI:ML Research/e87_receipt")
ROUTABLE = list(range(7, 21))
K = 4


def load():
    rows = [json.loads(l) for l in open(OUT/"phase_a_labels.jsonl")]
    split = json.loads((OUT/"e87_split.json").read_text())
    return rows, set(split["train"]), set(split["test"])


def feats(pl, layer):
    """cheap features for (instance, layer)."""
    return [pl["H"] or 0.0, pl["dH"] or 0.0, float(layer), (pl["H"] or 0.0) * (pl["dH"] or 0.0)]


def fit_linear(X, y, l2=1.0):
    """tiny ridge regression via normal equations (numpy); the 'cheap estimator' the prereg wants."""
    import numpy as np
    X = np.array(X, float); y = np.array(y, float)
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = Xb.T @ Xb + l2 * np.eye(Xb.shape[1])
    w = np.linalg.solve(A, Xb.T @ y)
    return w


def predict(w, X):
    import numpy as np
    X = np.array(X, float)
    return (np.hstack([X, np.ones((len(X), 1))]) @ w).tolist()


def spearman(a, b):
    import numpy as np
    def rank(v):
        order = np.argsort(v); r = np.empty(len(v)); r[order] = np.arange(len(v)); return r
    ra, rb = rank(np.array(a, float)), rank(np.array(b, float))
    if np.std(ra) == 0 or np.std(rb) == 0: return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    rows, train, test = load()
    print(f"labels: {len(rows)} instances ({sum(r['instance_id'] in train for r in rows)} train, "
          f"{sum(r['instance_id'] in test for r in rows)} test); "
          f"dry-run={'YES (partial)' if len(rows) < 200 else 'FULL'}")

    # build training set from TRAIN instances only (no leakage)
    Xtr, ytr = [], []
    for r in rows:
        if r["instance_id"] not in train: continue
        for l in ROUTABLE:
            pl = r["per_layer"][str(l)]
            Xtr.append(feats(pl, l)); ytr.append(pl["gain"])
    if len(Xtr) < 20:
        print("  too few train labels yet for a meaningful fit — dry-run pipeline check only")
    w = fit_linear(Xtr, ytr) if Xtr else None
    wr = [round(float(x), 3) for x in w] if w is not None else []
    print(f"  trained g_gain on {len(Xtr)} (instance,layer) train pairs; weights={wr}")

    # rank statistic on TEST instances (H-E87-3): g_gain vs |dH| as gain-rankers
    g_gain_pred, dH_neg, realized = [], [], []
    test_rows = [r for r in rows if r["instance_id"] in test]
    for r in test_rows:
        for l in ROUTABLE:
            pl = r["per_layer"][str(l)]
            realized.append(pl["gain"])
            dH_neg.append(pl["dH"] or 0.0)               # entropy rung's implied gain proxy (monotone in |dH|)
            g_gain_pred.append(predict(w, [feats(pl, l)])[0] if w is not None else 0.0)
    if realized and any(realized):
        rho_gain = spearman(g_gain_pred, realized)
        rho_ent = spearman(dH_neg, realized)
        print(f"  H-E87-3 (rank stat, n={len(realized)} test layer-pairs):")
        print(f"    ρ(g_gain, realized gain)  = {rho_gain:+.3f}")
        print(f"    ρ(|ΔH|,  realized gain)   = {rho_ent:+.3f}   [entropy rung]")
        print(f"    Δρ (g_gain − entropy)     = {rho_gain - rho_ent:+.3f}   ({'g_gain ranks better' if rho_gain>rho_ent else 'entropy >= g_gain'})")

    # emit per-test-instance skip-sets for the GPU chessboard
    skipsets = {}
    for r in test_rows:
        pls = r["per_layer"]
        gain_pred = {l: (predict(w, [feats(pls[str(l)], l)])[0] if w is not None else 0.0) for l in ROUTABLE}
        true_gain = {l: pls[str(l)]["gain"] for l in ROUTABLE}
        dH = {l: (pls[str(l)]["dH"] or 0.0) for l in ROUTABLE}
        skipsets[r["instance_id"]] = {
            "learned_gain": sorted(sorted(ROUTABLE, key=lambda l: gain_pred[l])[:K]),
            "entropy": sorted(sorted(ROUTABLE, key=lambda l: dH[l])[:K]),       # smallest |ΔH|
            "oracle": sorted(sorted(ROUTABLE, key=lambda l: true_gain[l])[:K]), # smallest TRUE gain (ceiling)
            # random + dense are computed at run time
        }
    (OUT/"e87_skipsets.json").write_text(json.dumps({"k": K, "split_hash": json.loads((OUT/'e87_split.json').read_text())['split_hash'], "skipsets": skipsets}, indent=2))
    print(f"  wrote skip-sets for {len(skipsets)} test instances -> e87_skipsets.json")
    if w is not None:
        (OUT/"g_gain_model.json").write_text(json.dumps({"weights": list(w), "features": ["H","dH","layer","H*dH","bias"]}, indent=2))


if __name__ == "__main__":
    main()
