#!/usr/bin/env python3
"""Exact k=2 pair matrix — the TRUE exhaustive k=2 achievable ceiling + the SELECTION dataset.

Replaces the best-of-20 SAMPLED k=2 "ceiling" (caught as overstated 2026-06-15: claims.yaml
k_sweep_correction_2026_06_15) with the FULL enumeration of all C(14,2)=91 routable-layer pairs × the 60
held-out tasks. Produces the 60×91 pass matrix — the first real dataset for the selection problem.

Why this experiment (post-v2): co-training to make ALL skips cheap produced a uniformly-mediocre model
(v2 KILLED, all H-CT failed). That makes SELECTION the live question: can a CHEAP selector recover the
k=2 ceiling, or are good sets only found by (expensive) verifier-assisted search? This matrix is the
microscope — k=2 has only 91 sets, and non-additivity is exactly the pairwise interaction term.

Outputs (in <RESEARCH>/e87_receipt/k2_matrix_<tag>/):
  rows.jsonl        — one row per (task, pair, seed): {iid, family, pair, seed, passed, dense}
  summary.json      — TRUE k=2 ceiling/avg-random/dense (per seed), per-pair rates, per-task pass counts,
                      layer-inclusion effects, adjacency/span effects, cross-seed stability
  completions/      — every generation, re-scorable (the v4 discipline)

Run (Lightning GPU, frozen base):   python docs/training/k2_pair_matrix.py --seeds 1337
  add stability seed:               python docs/training/k2_pair_matrix.py --seeds 1337 2026
  map a co-trained model too:       python docs/training/k2_pair_matrix.py --adapters checkpoints/cotrain-v2
"""
import argparse, json, itertools, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "first_light"))
import first_light_runner_v4 as R

OUT = R.RESEARCH / "e87_receipt"
ROUTABLE = list(range(7, 21))
PAIRS = list(itertools.combinations(ROUTABLE, 2))   # C(14,2) = 91


def find_layers(model):
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, nn.ModuleList) and len(m) >= 20 and hasattr(m[0], "self_attn"):
            return m
    raise RuntimeError("no decoder layers")


class _null:
    def __enter__(self): return None
    def __exit__(self, *x): return False


def summarize(rows, seeds):
    """TRUE exhaustive k=2 aggregates from the full matrix, plus interaction diagnostics."""
    tasks = sorted({r["iid"] for r in rows})
    out = {"n_tasks": len(tasks), "n_pairs": len(PAIRS), "seeds": seeds, "per_seed": {}}
    for seed in seeds:
        sr = [r for r in rows if r["seed"] == seed]
        skip = [r for r in sr if not r["dense"]]
        dense = [r for r in sr if r["dense"]]
        # per-task: did ANY pair pass (true ceiling), mean over pairs (true avg-random)
        by_task = defaultdict(list)
        for r in skip: by_task[r["iid"]].append(r["passed"])
        ceiling = float(np.mean([1 if any(v) else 0 for v in by_task.values()]))
        avg_random = float(np.mean([r["passed"] for r in skip]))
        dense_rate = float(np.mean([r["passed"] for r in dense])) if dense else None
        # per-pair pass rate across tasks
        by_pair = defaultdict(list)
        for r in skip: by_pair[tuple(r["pair"])].append(r["passed"])
        pair_rate = {f"{i}-{j}": float(np.mean(v)) for (i, j), v in by_pair.items()}
        # layer inclusion effect: mean(pass | layer in pair) - mean(pass | layer not in pair)
        incl = {}
        for L in ROUTABLE:
            inc = [r["passed"] for r in skip if L in r["pair"]]
            exc = [r["passed"] for r in skip if L not in r["pair"]]
            incl[L] = float(np.mean(inc) - np.mean(exc)) if inc and exc else None
        # adjacency + span
        adj = [r["passed"] for r in skip if abs(r["pair"][0]-r["pair"][1]) == 1]
        non = [r["passed"] for r in skip if abs(r["pair"][0]-r["pair"][1]) != 1]
        out["per_seed"][seed] = {
            "true_ceiling": ceiling, "true_avg_random": avg_random, "dense": dense_rate,
            "tasks_with_a_passing_pair": int(sum(1 for v in by_task.values() if any(v))),
            "per_task_passcount": {t: int(sum(by_task[t])) for t in tasks},
            "pair_rate_sorted": dict(sorted(pair_rate.items(), key=lambda kv: -kv[1])),
            "layer_inclusion_effect": incl,
            "adjacent_pass": float(np.mean(adj)) if adj else None,
            "nonadjacent_pass": float(np.mean(non)) if non else None,
        }
    # cross-seed stability: of (task,pair) passing on seeds[0], fraction passing on seeds[1]
    if len(seeds) >= 2:
        s0 = {(r["iid"], tuple(r["pair"])): r["passed"] for r in rows if r["seed"] == seeds[0] and not r["dense"]}
        s1 = {(r["iid"], tuple(r["pair"])): r["passed"] for r in rows if r["seed"] == seeds[1] and not r["dense"]}
        both = [k for k in s0 if k in s1]
        pass0 = [k for k in both if s0[k]]
        out["cross_seed_stability"] = {
            "pass_on_seed0": len(pass0),
            "of_those_pass_on_seed1": float(np.mean([s1[k] for k in pass0])) if pass0 else None,
            "raw_agreement": float(np.mean([s0[k] == s1[k] for k in both])) if both else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default=None, help="optional LoRA adapters; default = FROZEN base")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1337])
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--smoke", action="store_true", help="2 tasks × 3 pairs — validate the pipeline before the full run")
    a = ap.parse_args()
    R.MAX_NEW = a.max_new
    pairs = PAIRS[:3] if a.smoke else PAIRS
    tag = ("smoke_" if a.smoke else "") + ("frozen" if not a.adapters else Path(a.adapters).name.replace("/", "_"))
    rd = OUT / f"k2_matrix_{tag}"; (rd/"completions").mkdir(parents=True, exist_ok=True)
    DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    test = sorted(set(json.loads((OUT/"e87_split.json").read_text())["test"]))
    idx = {}
    for root, fam in [(R.RESEARCH/"gym-v0.1-FL", "F2_json"), (R.RESEARCH/"gym-v0.3", "F3_type")]:
        for sp in ["test", "smoke"]:
            for f in (root/fam/sp).glob("*.json"):
                d = json.loads(f.read_text()); idx[d["instance_id"]] = d
    test = [i for i in test if i in idx]
    if a.smoke: test = test[:2]

    tok = R.AutoTokenizer.from_pretrained(R.MODEL_REPO, revision=R.MODEL_REVISION)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = R.AutoModelForCausalLM.from_pretrained(R.MODEL_REPO, revision=R.MODEL_REVISION, dtype=torch.bfloat16)
    if a.adapters:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, a.adapters).to(DEV).eval()
    else:
        model = base.to(DEV).eval()
    layers = find_layers(model)

    def gen(prompt, skip, seed):
        with R.IdentitySkip(layers, skip) if skip else _null():
            c, _, _ = R.generate(tok, model, prompt, seed); return c

    total = len(test) * (len(pairs) + 1) * len(a.seeds)
    print(f"k=2 EXACT pair matrix on {tag} ({DEV}): {len(test)} tasks × {len(pairs)} pairs (+dense) × "
          f"{len(a.seeds)} seed(s) = {total} generations", flush=True)
    rows = []
    for ti, iid in enumerate(test):
        inst = idx[iid]; vf = R.VERIFIERS[inst["metadata"]["family"]]; prompt = inst["prompt_context"]
        for seed in a.seeds:
            dc = gen(prompt, [], seed)
            rows.append({"iid": iid, "family": inst["metadata"]["family"], "pair": None, "seed": seed,
                         "passed": bool(vf(inst, dc)), "dense": True})
            for (i, j) in pairs:
                comp = gen(prompt, [i, j], seed)
                rows.append({"iid": iid, "family": inst["metadata"]["family"], "pair": [i, j], "seed": seed,
                             "passed": bool(vf(inst, comp)), "dense": False})
                (rd/"completions"/f"{iid.replace('/','_')}_{i}_{j}_s{seed}.txt").write_text(comp)
        print(f"  [{ti+1}/{len(test)}] {iid}", flush=True)
        with open(rd/"rows.jsonl", "w") as f:        # crash-safe incremental save
            for r in rows: f.write(json.dumps(r)+"\n")

    summary = summarize(rows, a.seeds)
    (rd/"summary.json").write_text(json.dumps(summary, indent=2))
    s0 = summary["per_seed"][a.seeds[0]]
    print(f"\n  TRUE k=2 (seed {a.seeds[0]}): ceiling {s0['true_ceiling']*100:.1f}%  "
          f"avg-random {s0['true_avg_random']*100:.1f}%  dense {(s0['dense'] or 0)*100:.1f}%")
    print(f"  (sampled best-of-20 was: ceiling 91.7%, avg-random 38.7% — now retired by this exhaustive run)")
    print(f"  matrix + completions -> {rd}  (re-scorable). Bring back rows.jsonl + summary.json.")


if __name__ == "__main__":
    main()
