#!/usr/bin/env python3
"""LOCAL verification of a co-train — I run this on the M1 after the other model returns the adapters.

Design for integrity (every agent receipt this session had a defect): the other model produces ONLY the
trained LoRA adapters. THIS script does all the measurement, with the gold-standard harness + verifiers,
so their reported numbers are never trusted. It (1) self-proves the adapters are real + active,
(2) regenerates the held-out E87 test set on the CO-TRAINED model at k={1,2,4}, (3) compares to the
frozen-base E87 numbers to decide H-CT-0/1/2.

Run (local, after `pip install peft` and downloading the adapters):
  python docs/training/verify_cotrain.py --adapters /path/to/cotrain-skiprobust
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "first_light"))
import first_light_runner_v4 as R

OUT = R.RESEARCH / "e87_receipt"
ROUTABLE = list(range(7, 21)); SEEDS = [1337, 2026, 7]
# the FROZEN-base anchors (e87_VERIFICATION.md, all VERIFIED) — the comparison baseline:
FROZEN = {0: {"dense": .733},
          1: {"ceiling": .917, "avg_random": .543, "entropy": .75},
          2: {"ceiling": .917, "avg_random": .387},
          4: {"ceiling": .717, "avg_random": .148}}


def find_layers(model):
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, nn.ModuleList) and len(m) >= 20 and hasattr(m[0], "self_attn"):
            return m
    raise RuntimeError("no decoder layers")


def entropy_skip(tok, model, layers, prompt, k):
    inp = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    H = {}
    lmhead = model.get_output_embeddings()
    for li in ROUTABLE:
        logits = lmhead(out.hidden_states[li + 1][:, -1, :])
        p = torch.softmax(logits.float(), -1); H[li] = float(-(p * torch.log(p + 1e-10)).sum())
    dH = {ROUTABLE[i]: (abs(H[ROUTABLE[i]] - H[ROUTABLE[i-1]]) if i else 0.0) for i in range(len(ROUTABLE))}
    return sorted(dH, key=dH.get)[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 4])
    a = ap.parse_args()
    R.MAX_NEW = 256
    rd = OUT / f"cotrain_verify_{Path(a.adapters).name}"; (rd/"completions").mkdir(parents=True, exist_ok=True)

    test = sorted(set(json.loads((OUT/"e87_split.json").read_text())["test"]))
    idx = {}
    for root, fam in [(R.RESEARCH/"gym-v0.1-FL", "F2_json"), (R.RESEARCH/"gym-v0.3", "F3_type")]:
        for sp in ["test", "smoke"]:
            for f in (root/fam/sp).glob("*.json"):
                d = json.loads(f.read_text()); idx[d["instance_id"]] = d
    test = [i for i in test if i in idx]

    tok = R.AutoTokenizer.from_pretrained(R.MODEL_REPO, revision=R.MODEL_REVISION)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = R.AutoModelForCausalLM.from_pretrained(R.MODEL_REPO, revision=R.MODEL_REVISION, dtype=torch.bfloat16)
    from peft import PeftModel
    DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = PeftModel.from_pretrained(base, a.adapters).to(DEV).eval()
    layers = find_layers(model)

    # ---- SELF-PROOF: adapters real + active (re-derived locally, not trusted from their receipt) ----
    probe = tok("def add(a: int, b: int) -> int:", return_tensors="pt").to(DEV)
    with torch.no_grad():
        on = model(**probe).logits.float().cpu()
        with model.disable_adapter(): off = model(**probe).logits.float().cpu()
    diff = float((on - off).abs().max())
    print(f"SELF-PROOF: frozen-vs-cotrained logit diff = {diff:.2f}  -> {'adapters ACTIVE' if diff>0.5 else 'NO-OP — STOP'}")
    if diff <= 0.5:
        print("  adapters do not change the model; verification halted."); return

    def gen(prompt, skip):
        with R.IdentitySkip(layers, skip) if skip else _null():
            c, _, _ = R.generate(tok, model, prompt, SEEDS[0]); return c

    rows = []
    print(f"\nVERIFY co-train (adapters={a.adapters}, n={len(test)} held-out, my harness):")
    results = {"self_proof_logit_diff": diff}
    dense = np.mean([int(R.VERIFIERS[idx[i]['metadata']['family']](idx[i], gen(idx[i]['prompt_context'], []))) for i in test])
    results["cotrained_dense"] = float(dense)
    print(f"  k=0 dense: CO-TRAINED {dense*100:.1f}%  | FROZEN {FROZEN[0]['dense']*100:.0f}%  "
          f"-> H-CT-2 (no capability loss): {'PASS' if dense >= FROZEN[0]['dense']-0.05 else 'CHECK'}")
    for k in a.k:
        ceil, avg, ent = [], [], []
        for iid in test:
            inst = idx[iid]; vf = R.VERIFIERS[inst['metadata']['family']]; prompt = inst['prompt_context']
            rng = random.Random(R.sha16(iid))
            ps = []
            for s in range(a.samples):
                skip = sorted(rng.sample(ROUTABLE, k))
                comp = gen(prompt, skip); passed = bool(vf(inst, comp)); ps.append(passed)
                rows.append({"iid": iid, "k": k, "s": s, "skip": skip, "passed": passed})
                (rd/"completions"/f"{iid.replace('/','_')}_k{k}_s{s}.txt").write_text(comp)
            ceil.append(1 if any(ps) else 0); avg.append(np.mean(ps))
            # entropy controller (does routing beat random on the co-trained model? H-CT-1)
            esk = entropy_skip(tok, model, layers, prompt, k)
            ent.append(int(vf(inst, gen(prompt, esk))))
        c, av, e = float(np.mean(ceil)), float(np.mean(avg)), float(np.mean(ent))
        fr = FROZEN[k]
        results[f"k{k}"] = {"cotrained_ceiling": c, "cotrained_avg_random": av, "cotrained_entropy": e,
                            "frozen_avg_random": fr.get("avg_random"), "frozen_ceiling": fr.get("ceiling")}
        print(f"  k={k}: CO-TRAINED ceiling {c*100:.0f}% avg-random {av*100:.0f}% entropy {e*100:.0f}%  |  "
              f"FROZEN avg-random {fr['avg_random']*100:.0f}% ceiling {fr['ceiling']*100:.0f}%")
        print(f"       H-CT-0 (skipping cheaper): {'YES' if av > fr['avg_random']+0.05 else 'no'}  "
              f"| H-CT-1 (entropy>random on co-trained): {'YES' if e > av+0.05 else 'no'}")
    (rd/"verify_result.json").write_text(json.dumps(results, indent=2))
    with open(rd/"rows.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r)+"\n")
    print(f"\n  completions + records saved -> {rd}  (re-scorable)")
    print("  H-CT-0 cost-shrinks if co-trained avg-random > frozen at each k; H-CT-2 if dense preserved.")


class _null:
    def __enter__(self): return None
    def __exit__(self, *x): return False


if __name__ == "__main__":
    main()
