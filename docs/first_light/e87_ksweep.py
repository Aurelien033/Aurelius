#!/usr/bin/env python3
"""E87 k-sweep — TRUE combinatorial achievable-ceiling at k=2 and k=4 (OBL-057).

The E87 chessboard's "oracle" was a greedy single-layer-marginal selector (invalid under layer
non-additivity). This runs a SAMPLED combinatorial oracle: per test instance, sample N random
k-subsets of the routable layers, run+verify each (greedy), and report:
  - achievable ceiling = fraction of instances where AT LEAST ONE of N skip-sets passes
  - average = mean single-skip-set pass rate (expected random)
Distinguishes 'selection is hard' (ceiling high, far above average) from 'multi-skip impossible'
(ceiling ≈ average ≈ low). k=1 is already exhaustive from Phase A (ceiling 91.7%); this adds k=2,4.
Reuses v4 IdentitySkip + verifiers. ~60 inst x {2 k-values} x N samples.
"""
import json, random, sys, time
from pathlib import Path
import torch
sys.path.insert(0, "/Users/christienantonio/aurelius/docs/first_light")
import first_light_runner_v4 as R

RESEARCH = Path("/Users/christienantonio/Desktop/AI:ML Research")
OUT = RESEARCH / "e87_receipt"
ROUTABLE = list(range(7, 21)); SEED = 1337; N = 20; KS = [2, 4]; MAX_NEW = 256


def build_index():
    idx = {}
    for root, fam in [(RESEARCH/"gym-v0.1-FL", "F2_json"), (RESEARCH/"gym-v0.3", "F3_type")]:
        for sp in ["test", "smoke"]:
            for f in (root/fam/sp).glob("*.json"):
                d = json.loads(f.read_text()); idx[d["instance_id"]] = d
    return idx


def main():
    R.MAX_NEW = MAX_NEW
    test = set(json.loads((OUT/"e87_split.json").read_text())["test"])
    idx = build_index()
    test_ids = [i for i in test if i in idx]
    print(f"E87 k-sweep: {len(test_ids)} test instances x k{KS} x N={N} samples")
    tok = R.AutoTokenizer.from_pretrained(R.MODEL_REPO, revision=R.MODEL_REVISION)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = R.AutoModelForCausalLM.from_pretrained(R.MODEL_REPO, revision=R.MODEL_REVISION,
        dtype=torch.bfloat16, low_cpu_mem_usage=True).to("mps").eval()

    ckpt = OUT / "ksweep_results.jsonl"
    done = set()
    if ckpt.exists():
        for l in open(ckpt):
            r = json.loads(l); done.add((r["instance_id"], r["k"], r["sample"]))

    t0 = time.time()
    with open(ckpt, "a") as out:
        for ii, iid in enumerate(test_ids):
            inst = idx[iid]; verify = R.VERIFIERS[inst["metadata"]["family"]]
            rng = random.Random(R.sha16(iid))
            for k in KS:
                for s in range(N):
                    if (iid, k, s) in done: continue
                    skip = sorted(rng.sample(ROUTABLE, k))
                    comp, _, _ = R.generate(tok, model, inst["prompt_context"], SEED, skip=skip)
                    out.write(json.dumps({"instance_id": iid, "family": inst["metadata"]["family"],
                        "k": k, "sample": s, "skip": skip, "passed": bool(verify(inst, comp))})+"\n")
                    out.flush()
            if (ii+1) % 10 == 0:
                print(f"  [{time.strftime('%H:%M:%S')}] {ii+1}/{len(test_ids)} instances ({(time.time()-t0)/(ii+1):.0f}s/inst)", flush=True)

    # analysis
    rows = [json.loads(l) for l in open(ckpt)]
    import numpy as np
    print(f"\nE87 ACHIEVABLE-CEILING vs k (n={len(test_ids)} test instances):")
    print(f"  k=0 dense: 73.3% | k=1 (exhaustive, Phase A): ceiling 91.7%, avg 54.3%")
    for k in KS:
        by_inst = {}
        for r in rows:
            if r["k"] == k: by_inst.setdefault(r["instance_id"], []).append(r["passed"])
        ceiling = np.mean([1 if any(v) else 0 for v in by_inst.values()])
        avg = np.mean([np.mean(v) for v in by_inst.values()])
        print(f"  k={k}: ACHIEVABLE CEILING {ceiling*100:.1f}%  |  avg random skip {avg*100:.1f}%  |  gap {(ceiling-avg)*100:+.1f}pp")
    res = {}
    for k in KS:
        by_inst = {}
        for r in rows:
            if r["k"] == k: by_inst.setdefault(r["instance_id"], []).append(r["passed"])
        res[f"k{k}"] = {"ceiling": float(np.mean([1 if any(v) else 0 for v in by_inst.values()])),
                        "avg": float(np.mean([np.mean(v) for v in by_inst.values()])), "N": N}
    res["k0_dense"] = 0.733; res["k1_ceiling"] = 0.917; res["k1_avg"] = 0.543
    (OUT/"e87_ksweep_analysis.json").write_text(json.dumps(res, indent=2))
    print("  wrote e87_ksweep_analysis.json")


if __name__ == "__main__":
    main()
