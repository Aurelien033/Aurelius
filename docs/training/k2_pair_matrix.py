#!/usr/bin/env python3
"""Exact k=2 pair matrix (BATCHED, greedy) — the TRUE exhaustive k=2 ceiling + the SELECTION dataset.

Replaces the best-of-20 SAMPLED k=2 "ceiling" (caught as overstated 2026-06-15: claims.yaml
k_sweep_correction_2026_06_15) with the FULL enumeration of all C(14,2)=91 routable-layer pairs × the 60
held-out tasks. Produces the 60×91 pass matrix — the first real dataset for the selection problem.

Why batched: generation here is GREEDY (deterministic), so for a FIXED skip-set we can generate all 60
task completions in ONE batched forward. Qwen2.5-1.5B is GQA (2 KV heads) so the KV cache is tiny and the
whole 60-task batch fits a 16GB T4. Cost = 92 batched generations (91 pairs + 1 dense) instead of 5,460
sequential ones — minutes on a T4 instead of hours. (Greedy ⇒ seed is irrelevant; the across-seed
"stability" kill-shot needs temperature>0 and is a separate run.)

Outputs (in <RESEARCH>/e87_receipt/k2_matrix_<tag>/):
  rows.jsonl   — one row per (task, pair): {iid, family, pair, passed, dense}
  summary.json — TRUE k=2 ceiling/avg-random/dense, per-pair rates, layer-inclusion effects, adjacency/span
  completions/ — every generation, re-scorable (the v4 discipline)

Run (Colab T4, frozen base):     python docs/training/k2_pair_matrix.py
  map a co-trained model too:     python docs/training/k2_pair_matrix.py --adapters checkpoints/cotrain-v2
  smaller batch if OOM:           python docs/training/k2_pair_matrix.py --batch_size 20
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


def batched_greedy(tok, model, prompts, skip, layers, dev, batch_size, max_new):
    """Greedy-generate completions for many prompts under a FIXED skip-set, in chunks.
    Left-padding so the new tokens start at the same index for every row in a chunk."""
    outs = []
    for s in range(0, len(prompts), batch_size):
        chunk = prompts[s:s + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to(dev)
        with (R.IdentitySkip(layers, skip) if skip else _null()), torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        L = enc["input_ids"].shape[1]
        for row in gen[:, L:]:
            outs.append(tok.decode(row, skip_special_tokens=True))
    return outs


def summarize(rows):
    """TRUE exhaustive k=2 aggregates from the full matrix, plus interaction diagnostics."""
    tasks = sorted({r["iid"] for r in rows})
    skip = [r for r in rows if not r["dense"]]
    dense = [r for r in rows if r["dense"]]
    by_task = defaultdict(list)
    for r in skip: by_task[r["iid"]].append(r["passed"])
    ceiling = float(np.mean([1 if any(v) else 0 for v in by_task.values()]))
    avg_random = float(np.mean([r["passed"] for r in skip]))
    dense_rate = float(np.mean([r["passed"] for r in dense])) if dense else None
    by_pair = defaultdict(list)
    for r in skip: by_pair[tuple(r["pair"])].append(r["passed"])
    pair_rate = {f"{i}-{j}": float(np.mean(v)) for (i, j), v in by_pair.items()}
    incl = {}
    for Lr in ROUTABLE:
        inc = [r["passed"] for r in skip if Lr in r["pair"]]
        exc = [r["passed"] for r in skip if Lr not in r["pair"]]
        incl[Lr] = float(np.mean(inc) - np.mean(exc)) if inc and exc else None
    adj = [r["passed"] for r in skip if abs(r["pair"][0] - r["pair"][1]) == 1]
    non = [r["passed"] for r in skip if abs(r["pair"][0] - r["pair"][1]) != 1]
    return {
        "n_tasks": len(tasks), "n_pairs": len(PAIRS), "decoding": "greedy",
        "true_ceiling": ceiling, "true_avg_random": avg_random, "dense": dense_rate,
        "tasks_with_a_passing_pair": int(sum(1 for v in by_task.values() if any(v))),
        "sampled_was": {"ceiling": 0.917, "avg_random": 0.387, "note": "best-of-20 sampled — now retired"},
        "per_task_passcount": {t: int(sum(by_task[t])) for t in tasks},
        "pair_rate_sorted": dict(sorted(pair_rate.items(), key=lambda kv: -kv[1])),
        "layer_inclusion_effect": incl,
        "adjacent_pass": float(np.mean(adj)) if adj else None,
        "nonadjacent_pass": float(np.mean(non)) if non else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default=None, help="optional LoRA adapters; default = FROZEN base")
    ap.add_argument("--batch_size", type=int, default=30)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--out_dir", default=None, help="write outputs here (e.g. a Google Drive path) so they survive a VM recycle; resume reads from here too")
    ap.add_argument("--smoke", action="store_true", help="2 tasks × 3 pairs — validate the pipeline first")
    a = ap.parse_args()
    R.MAX_NEW = a.max_new
    pairs = PAIRS[:3] if a.smoke else PAIRS
    tag = ("smoke_" if a.smoke else "") + ("frozen" if not a.adapters else Path(a.adapters).name.replace("/", "_"))
    rd = Path(a.out_dir) if a.out_dir else (OUT / f"k2_matrix_{tag}")
    (rd / "completions").mkdir(parents=True, exist_ok=True)
    DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    test = sorted(set(json.loads((OUT / "e87_split.json").read_text())["test"]))
    idx = {}
    for root, fam in [(R.RESEARCH / "gym-v0.1-FL", "F2_json"), (R.RESEARCH / "gym-v0.3", "F3_type")]:
        for sp in ["test", "smoke"]:
            for f in (root / fam / sp).glob("*.json"):
                d = json.loads(f.read_text()); idx[d["instance_id"]] = d
    test = [i for i in test if i in idx]
    if a.smoke: test = test[:2]

    tok = R.AutoTokenizer.from_pretrained(R.MODEL_REPO, revision=R.MODEL_REVISION)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # decoder-only batched generation needs left padding
    base = R.AutoModelForCausalLM.from_pretrained(R.MODEL_REPO, revision=R.MODEL_REVISION, dtype=torch.bfloat16)
    if a.adapters:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, a.adapters).to(DEV).eval()
    else:
        model = base.to(DEV).eval()
    layers = find_layers(model)

    prompts = [idx[iid]["prompt_context"] for iid in test]
    vfs = [R.VERIFIERS[idx[iid]["metadata"]["family"]] for iid in test]
    fams = [idx[iid]["metadata"]["family"] for iid in test]

    # RESUME: rows.jsonl is written only after a pair fully completes (all 60 tasks), so it's never
    # partially-corrupt. Reload it and skip dense + any pairs already done (survives disconnects).
    rows, done_pairs, dense_done = [], set(), False
    if (rd / "rows.jsonl").exists():
        rows = [json.loads(l) for l in open(rd / "rows.jsonl")]
        done_pairs = {tuple(r["pair"]) for r in rows if r["pair"] is not None}
        dense_done = any(r["dense"] for r in rows)
    todo = [p for p in pairs if tuple(p) not in done_pairs]
    print(f"k=2 EXACT pair matrix on {tag} ({DEV}, bf16, batch {a.batch_size}): "
          f"{len(test)} tasks × {len(pairs)} pairs (+dense), GREEDY", flush=True)
    if done_pairs or dense_done:
        print(f"  RESUME: {len(done_pairs)}/{len(pairs)} pairs + dense({dense_done}) already done; "
              f"{len(todo)} pairs to go", flush=True)

    if not dense_done:
        dcomps = batched_greedy(tok, model, prompts, [], layers, DEV, a.batch_size, a.max_new)
        for ti, iid in enumerate(test):
            rows.append({"iid": iid, "family": fams[ti], "pair": None, "passed": bool(vfs[ti](idx[iid], dcomps[ti])), "dense": True})
        with open(rd / "rows.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
    print(f"  dense done: {np.mean([r['passed'] for r in rows if r['dense']])*100:.1f}%", flush=True)

    for pi, (i, j) in enumerate(todo):
        comps = batched_greedy(tok, model, prompts, [i, j], layers, DEV, a.batch_size, a.max_new)
        for ti, iid in enumerate(test):
            passed = bool(vfs[ti](idx[iid], comps[ti]))
            rows.append({"iid": iid, "family": fams[ti], "pair": [i, j], "passed": passed, "dense": False})
            (rd / "completions" / f"{iid.replace('/','_')}_{i}_{j}.txt").write_text(comps[ti])
        done_n = len(pairs) - len(todo) + (pi + 1)
        if done_n % 10 == 0 or pi == len(todo) - 1:
            print(f"  [{done_n}/{len(pairs)} pairs] last ({i},{j})", flush=True)
        with open(rd / "rows.jsonl", "w") as f:      # crash-safe incremental save
            for r in rows: f.write(json.dumps(r) + "\n")

    summary = summarize(rows)
    (rd / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  TRUE k=2: ceiling {summary['true_ceiling']*100:.1f}%  "
          f"avg-random {summary['true_avg_random']*100:.1f}%  dense {(summary['dense'] or 0)*100:.1f}%")
    print(f"  (sampled best-of-20 was ceiling 91.7% / avg-random 38.7% — now retired by this exhaustive run)")
    print(f"  matrix + completions -> {rd}  (re-scorable). Bring back rows.jsonl + summary.json.")


if __name__ == "__main__":
    main()
