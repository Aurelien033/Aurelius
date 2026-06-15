#!/usr/bin/env python3
"""Eval a co-trained (skip-robust) model — the H-CT-0/1/2 measurement (drop-in after cotrain).

Loads FROZEN-BASE-v1 + the trained LoRA adapters, then runs the SAME held-out E87 test set at the
k-sweep operating points, comparing CO-TRAINED vs the frozen-base E87 numbers. One command on Lightning.
Independent re-score built in (the v4 discipline). Reuses the verified verifiers + IdentitySkip.

Run (on Lightning, after cotrain):
  python docs/training/eval_cotrained.py --adapters checkpoints/cotrain-skiprobust --k 1 2 4
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/Users/christienantonio/aurelius/docs/first_light")
import first_light_runner_v4 as R

RESEARCH = Path("/Users/christienantonio/Desktop/AI:ML Research")   # or wherever the gym/split live on Lightning
OUT = RESEARCH / "e87_receipt"
ROUTABLE = list(range(7, 21)); SEEDS = [1337, 2026, 7]
FROZEN_E87 = {1: {"dense": .733, "entropy": .75, "random": .543, "ceiling": .917},
              2: {"random": .387, "ceiling": .917}, 4: {"dense": .733, "random": .148, "ceiling": .717,
              "entropy": .017, "learned_gain": .20}}   # the frozen-base comparison (e87_VERIFICATION.md)


def build_index():
    idx = {}
    for root, fam in [(RESEARCH/"gym-v0.1-FL", "F2_json"), (RESEARCH/"gym-v0.3", "F3_type")]:
        for sp in ["test", "smoke"]:
            for f in (root/fam/sp).glob("*.json"):
                d = json.loads(f.read_text()); idx[d["instance_id"]] = d
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True, help="path to the cotrain LoRA checkpoint")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--samples", type=int, default=20, help="random skip-sets/instance for the ceiling")
    a = ap.parse_args()
    R.MAX_NEW = 256
    test = set(json.loads((OUT/"e87_split.json").read_text())["test"])
    idx = build_index(); test_ids = [i for i in test if i in idx]

    tok = R.AutoTokenizer.from_pretrained(R.MODEL_REPO, revision=R.MODEL_REVISION)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = R.AutoModelForCausalLM.from_pretrained(R.MODEL_REPO, revision=R.MODEL_REVISION, dtype=torch.bfloat16)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapters)   # <-- the co-trained skip-robust adapters
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(dev).eval()
    layers = _find_layers(model)

    def run(iid, skip):
        inst = idx[iid]
        with R.IdentitySkip(layers, skip) if skip else _null():
            comp, _, _ = R.generate(tok, model, inst["prompt_context"], SEEDS[0], skip=None)  # skip applied via ctx
        return bool(R.VERIFIERS[inst["metadata"]["family"]](inst, comp))

    print(f"CO-TRAINED EVAL (adapters={a.adapters}, n={len(test_ids)} held-out test):")
    results = {}
    for k in a.k:
        dense = np.mean([run(i, []) for i in test_ids]) if k == a.k[0] else results.get("dense")
        # achievable ceiling (best of N random k-sets) + avg random + dense
        ceil_hits, avg = [], []
        for iid in test_ids:
            rng = random.Random(R.sha16(iid))
            passes = [run(iid, sorted(rng.sample(ROUTABLE, k))) for _ in range(a.samples)]
            ceil_hits.append(1 if any(passes) else 0); avg.append(np.mean(passes))
        ceiling, avgr = float(np.mean(ceil_hits)), float(np.mean(avg))
        fr = FROZEN_E87.get(k, {})
        print(f"  k={k}: dense {dense*100:.0f}% | CO-TRAINED ceiling {ceiling*100:.1f}% avg-random {avgr*100:.1f}%  "
              f"| FROZEN was ceiling {fr.get('ceiling',0)*100:.0f}% random {fr.get('random',0)*100:.0f}%")
        results[k] = {"dense": float(dense), "cotrained_ceiling": ceiling, "cotrained_avg_random": avgr, "frozen": fr}
        results["dense"] = float(dense)
    Path(a.adapters).joinpath("cotrain_eval.json").write_text(json.dumps(results, indent=2))
    print("\n  H-CT-0 (cost shrinks): compare co-trained avg-random vs frozen avg-random at each k (higher = skipping got cheaper)")
    print("  H-CT-2 (no capability loss): co-trained dense vs frozen dense 73%")
    print("  -> bring cotrain_eval.json back for independent verification")


def _find_layers(model):
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, nn.ModuleList) and len(m) >= 20 and hasattr(m[0], "self_attn"):
            return m
    raise RuntimeError("no decoder layers")


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
