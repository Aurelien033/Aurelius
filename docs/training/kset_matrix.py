#!/usr/bin/env python3
"""Set-generic skip-set matrix (the k>=3 generalization of k2_pair_matrix). Evaluates a FIXED list of
skip-sets (from --sets, e.g. k4_sets_v0.1.json) x a task split, greedy + batched + resumable.

For each task and each skip-set: generate (identity-skip those layers) and verify -> pass/fail. Output is
a clean N x S pass matrix = the supervision for the k=4 routing/selector ablation (kset_selector.py).
Greedy => batch all tasks per set in one forward (Qwen GQA => tiny KV cache => fits a T4).

Outputs (in --out_dir):
  rows.jsonl   — one row per (task, set): {iid, family, set, passed, dense}
  summary.json — TRUE-on-the-sample ceiling/avg-random/dense, per-set rates, layer-inclusion, poison-count
                 + adjacency effects (the mechanistic read at this k)

Run (Kaggle): python docs/training/kset_matrix.py --split eval_data/selector_split_v0.1.json \
                --sets eval_data/k4_sets_v0.1.json --out_dir /kaggle/working/k4 --dtype fp16 \
                --no_completions --batch_size 48
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "first_light"))
import first_light_runner_v4 as R

OUT = R.RESEARCH / "e87_receipt"
ROUTABLE = list(range(7, 21))
POISON = {11, 14, 15}   # from the k=2 result (layer-inclusion); used only for the summary read


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
    """Greedy-generate for many prompts under a FIXED skip-set, chunked, left-padded."""
    outs = []
    for s in range(0, len(prompts), batch_size):
        enc = tok(prompts[s:s + batch_size], return_tensors="pt", padding=True).to(dev)
        with (R.IdentitySkip(layers, skip) if skip else _null()), torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        L = enc["input_ids"].shape[1]
        outs.extend(tok.decode(r, skip_special_tokens=True) for r in gen[:, L:])
    return outs


def n_adjacent(s):
    s = sorted(s); return sum(1 for a, b in zip(s, s[1:]) if b - a == 1)


def summarize(rows, sets):
    tasks = sorted({r["iid"] for r in rows})
    skip = [r for r in rows if not r["dense"]]; dense = [r for r in rows if r["dense"]]
    by_task = defaultdict(list)
    for r in skip: by_task[r["iid"]].append(r["passed"])
    ceiling = float(np.mean([1 if any(v) else 0 for v in by_task.values()]))
    avg_random = float(np.mean([r["passed"] for r in skip]))
    dense_rate = float(np.mean([r["passed"] for r in dense])) if dense else None
    by_set = defaultdict(list)
    for r in skip: by_set[tuple(r["set"])].append(r["passed"])
    set_rate = {",".join(map(str, k)): float(np.mean(v)) for k, v in by_set.items()}
    incl = {}
    for Lr in ROUTABLE:
        inc = [r["passed"] for r in skip if Lr in r["set"]]
        exc = [r["passed"] for r in skip if Lr not in r["set"]]
        incl[Lr] = float(np.mean(inc) - np.mean(exc)) if inc and exc else None
    pois = defaultdict(list); adj = defaultdict(list)
    for r in skip:
        pois[len(set(r["set"]) & POISON)].append(r["passed"])
        adj[n_adjacent(r["set"])].append(r["passed"])
    return {
        "n_tasks": len(tasks), "n_sets": len(sets), "k": len(sets[0]) if sets else None, "decoding": "greedy",
        "ceiling_sampled": ceiling, "avg_random": avg_random, "dense": dense_rate,
        "note": "ceiling is best-of-sampled (a LOWER bound on the true k-set oracle)",
        "tasks_with_a_passing_set": int(sum(1 for v in by_task.values() if any(v))),
        "set_rate_sorted": dict(sorted(set_rate.items(), key=lambda kv: -kv[1])),
        "layer_inclusion_effect": incl,
        "pass_by_poison_count": {k: float(np.mean(v)) for k, v in sorted(pois.items())},
        "pass_by_n_adjacent": {k: float(np.mean(v)) for k, v in sorted(adj.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--sets", required=True, help="json with a 'sets' list of skip-sets")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--adapters", default=None)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="fp16")
    ap.add_argument("--no_completions", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    R.MAX_NEW = a.max_new
    sets = [list(s) for s in json.loads(Path(a.sets).read_text())["sets"]]
    if a.smoke: sets = sets[:3]
    rd = Path(a.out_dir); (rd / "completions").mkdir(parents=True, exist_ok=True)
    DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    DT = torch.float16 if a.dtype == "fp16" else torch.bfloat16

    test = sorted(set(json.loads(Path(a.split).read_text())["all"]))
    idx = {}
    for root, fam in [(R.RESEARCH / "gym-v0.1-FL", "F2_json"), (R.RESEARCH / "gym-v0.3", "F3_type")]:
        for sp in ["dev", "test", "smoke", "train"]:
            for f in (root / fam / sp).glob("*.json"):
                d = json.loads(f.read_text()); idx[d["instance_id"]] = d
    test = [i for i in test if i in idx]
    if a.smoke: test = test[:2]

    tok = R.AutoTokenizer.from_pretrained(R.MODEL_REPO, revision=R.MODEL_REVISION)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = R.AutoModelForCausalLM.from_pretrained(R.MODEL_REPO, revision=R.MODEL_REVISION, dtype=DT)
    if a.adapters:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, a.adapters).to(DEV).eval()
    else:
        model = base.to(DEV).eval()
    layers = find_layers(model)

    prompts = [idx[i]["prompt_context"] for i in test]
    vfs = [R.VERIFIERS[idx[i]["metadata"]["family"]] for i in test]
    fams = [idx[i]["metadata"]["family"] for i in test]

    # RESUME (rows.jsonl written only after a set fully completes => never partial)
    rows, done, dense_done = [], set(), False
    if (rd / "rows.jsonl").exists():
        rows = [json.loads(l) for l in open(rd / "rows.jsonl")]
        done = {tuple(r["set"]) for r in rows if r["set"] is not None}
        dense_done = any(r["dense"] for r in rows)
    todo = [s for s in sets if tuple(s) not in done]
    k = len(sets[0]) if sets else "?"
    print(f"k={k} matrix ({DEV}/{a.dtype}, batch {a.batch_size}): {len(test)} tasks × {len(sets)} sets (+dense)"
          f"{f' RESUME {len(done)}/{len(sets)} done' if done else ''}", flush=True)

    if not dense_done:
        dc = batched_greedy(tok, model, prompts, [], layers, DEV, a.batch_size, a.max_new)
        for ti, iid in enumerate(test):
            rows.append({"iid": iid, "family": fams[ti], "set": None, "passed": bool(vfs[ti](idx[iid], dc[ti])), "dense": True})
        with open(rd / "rows.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
    print(f"  dense done: {np.mean([r['passed'] for r in rows if r['dense']])*100:.1f}%", flush=True)

    for si, sk in enumerate(todo):
        comps = batched_greedy(tok, model, prompts, sk, layers, DEV, a.batch_size, a.max_new)
        for ti, iid in enumerate(test):
            passed = bool(vfs[ti](idx[iid], comps[ti]))
            rows.append({"iid": iid, "family": fams[ti], "set": sk, "passed": passed, "dense": False})
            if not a.no_completions:
                (rd / "completions" / f"{iid.replace('/','_')}_{'-'.join(map(str,sk))}.txt").write_text(comps[ti])
        n = len(sets) - len(todo) + (si + 1)
        if n % 20 == 0 or si == len(todo) - 1:
            print(f"  [{n}/{len(sets)} sets] last {sk}", flush=True)
        with open(rd / "rows.jsonl", "w") as f:      # crash-safe
            for r in rows: f.write(json.dumps(r) + "\n")

    summary = summarize(rows, sets)
    (rd / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  k={k}: ceiling(sampled) {summary['ceiling_sampled']*100:.1f}%  avg-random "
          f"{summary['avg_random']*100:.1f}%  dense {(summary['dense'] or 0)*100:.1f}%")
    print(f"  matrix -> {rd}. Next: kset_feats.py + kset_selector.py")


if __name__ == "__main__":
    main()
