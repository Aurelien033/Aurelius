#!/usr/bin/env python3
"""E87 Phase B (GPU part) — the matched-k chessboard on held-out TEST instances. THE thesis test.

For each test instance x seed x policy {dense(k=0), random(k=4), entropy, learned_gain, oracle},
generate + verify. Skip-sets for entropy/learned_gain/oracle come from e87_skipsets.json (emitted by
e87_phase_b.py from FULL Phase-A labels). random is seeded at run time. Saves completions + verdicts;
an independent re-score (rescore parity) + paired bootstrap follow.

Run AFTER Phase A completes and e87_phase_b.py has produced e87_skipsets.json from full labels.
Reuses the v4 spike-verified IdentitySkip + verifiers. Checkpoint/resume.
"""
import json, random, sys, time
from pathlib import Path
import torch
sys.path.insert(0, "/Users/christienantonio/aurelius/docs/first_light")
import first_light_runner_v4 as R

RESEARCH = Path("/Users/christienantonio/Desktop/AI:ML Research")
OUT = RESEARCH / "e87_receipt"
ROUTABLE = list(range(7, 21)); K = 4; SEEDS = [1337, 2026, 7]; MAX_NEW = 256


def build_index():
    idx = {}
    for root, fam in [(RESEARCH/"gym-v0.1-FL", "F2_json"), (RESEARCH/"gym-v0.3", "F3_type")]:
        for sp in ["test", "smoke"]:
            for f in (root/fam/sp).glob("*.json"):
                d = json.loads(f.read_text()); idx[d["instance_id"]] = d
    return idx


def main():
    R.MAX_NEW = MAX_NEW
    skip = json.loads((OUT/"e87_skipsets.json").read_text())
    skipsets = skip["skipsets"]
    idx = build_index()
    test_ids = [i for i in skipsets if i in idx]
    print(f"E87 chessboard: {len(test_ids)} test instances x {len(SEEDS)} seeds x 5 policies "
          f"(split_hash {skip['split_hash']})")

    tok = R.AutoTokenizer.from_pretrained(R.MODEL_REPO, revision=R.MODEL_REVISION)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = R.AutoModelForCausalLM.from_pretrained(R.MODEL_REPO, revision=R.MODEL_REVISION,
        dtype=torch.bfloat16, low_cpu_mem_usage=True).to("mps").eval()

    ckpt = OUT / "chessboard_results.jsonl"
    done = set()
    if ckpt.exists():
        for l in open(ckpt):
            r = json.loads(l); done.add((r["instance_id"], r["policy"], r["seed"]))

    def policy_skip(iid, policy, seed):
        if policy == "dense": return None
        if policy == "random":
            return sorted(random.Random(R.sha16(iid+str(seed))).sample(ROUTABLE, K))
        return skipsets[iid][policy]   # entropy / learned_gain / oracle (deterministic per instance)

    t0 = time.time(); n = 0
    with open(ckpt, "a") as out:
        for iid in test_ids:
            inst = idx[iid]; fam = inst["metadata"]["family"]; verify = R.VERIFIERS[fam]
            for policy in ["dense", "random", "entropy", "learned_gain", "oracle"]:
                for seed in SEEDS:
                    if (iid, policy, seed) in done: continue
                    sk = policy_skip(iid, policy, seed)
                    comp, ntok, dt = R.generate(tok, model, inst["prompt_context"], seed, skip=sk)
                    passed = bool(verify(inst, comp))
                    cd = OUT/"chessboard_completions"/policy; cd.mkdir(parents=True, exist_ok=True)
                    (cd/f"{iid.replace('/','_')}_s{seed}.txt").write_text(comp)
                    out.write(json.dumps({"instance_id": iid, "family": fam, "policy": policy,
                        "seed": seed, "passed": passed, "skip": sk, "tps": ntok/dt if dt>0 else 0.0})+"\n")
                    out.flush(); n += 1
            if (test_ids.index(iid)+1) % 10 == 0:
                print(f"  [{time.strftime('%H:%M:%S')}] {test_ids.index(iid)+1}/{len(test_ids)} instances", flush=True)

    # ---- analysis ----
    rows = [json.loads(l) for l in open(ckpt)]
    bi = {}
    for r in rows: bi.setdefault(r["instance_id"], {}).setdefault(r["policy"], []).append(r["passed"])
    paired = [i for i in test_ids if all(p in bi[i] for p in ["dense","random","entropy","learned_gain","oracle"])]
    import numpy as np
    def rate(p): return float(np.mean([any(bi[i][p]) for i in paired]))
    def pboot(pa, pb, B=10000):
        rng = np.random.RandomState(20260614)
        d = np.array([int(any(bi[i][pa]))-int(any(bi[i][pb])) for i in paired])
        bs = [d[rng.randint(0,len(d),len(d))].mean() for _ in range(B)]
        return float(d.mean()), (float(np.percentile(bs,2.5)), float(np.percentile(bs,97.5)))
    print(f"\nE87 CHESSBOARD (n={len(paired)} held-out test instances, matched k={K}):")
    for p in ["dense","oracle","learned_gain","entropy","random"]:
        print(f"  {p:14s} {rate(p)*100:5.1f}%")
    og, ogc = pboot("oracle","random")
    lg, lgc = pboot("learned_gain","entropy")
    lr, lrc = pboot("learned_gain","random")
    print(f"\n  H-E87-0 (signal exists?) oracle-random = {og*100:+.1f}pp CI[{ogc[0]*100:+.1f},{ogc[1]*100:+.1f}] -> {'SIGNAL' if ogc[0]>0 else 'NO SIGNAL (thesis dead at this scale)'}")
    print(f"  H-E87-1 (THE THESIS)     learned-entropy = {lg*100:+.1f}pp CI[{lgc[0]*100:+.1f},{lgc[1]*100:+.1f}] -> {'g_gain BEATS entropy' if lgc[0]>0 else 'no win over entropy'}")
    print(f"  H-E87-2 (clears bar)     learned-random  = {lr*100:+.1f}pp CI[{lrc[0]*100:+.1f},{lrc[1]*100:+.1f}] -> {'beats random' if lrc[0]>0 else 'does NOT beat random'}")
    res = {"n": len(paired), "rates": {p: rate(p) for p in ["dense","oracle","learned_gain","entropy","random"]},
           "H-E87-0_oracle_vs_random": [og, list(ogc)], "H-E87-1_learned_vs_entropy": [lg, list(lgc)],
           "H-E87-2_learned_vs_random": [lr, list(lrc)]}
    (OUT/"e87_chessboard_analysis.json").write_text(json.dumps(res, indent=2))
    print(f"\n  wrote e87_chessboard_analysis.json")


if __name__ == "__main__":
    main()
