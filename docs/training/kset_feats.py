#!/usr/bin/env python3
"""Set-generic PROMPT-CONDITIONED features (the k>=3 generalization of feat_extract). ONE dense prefill per
task, no generation.

Per routable layer L (last token): contrib_norm, delta_norm, cos, entropy, logitlens_KL  (unchanged).
Per skip-SET S (the mechanistic non-additivity feature, generalized): with logitlens_delta_L = the layer's
image in logit space (lm_head(norm(h_{L+1})) - lm_head(norm(h_L))),
  set_interference(S) = || sum_{L in S} logitlens_delta_L ||^2 / sum_{L in S} || logitlens_delta_L ||^2
= 1 if the removed contributions are orthogonal; >1 if they CONSTRUCTIVELY interfere (predicted-bad set);
<1 if they cancel. This is exactly the cross-term that makes skip-set cost non-additive.

Run: python docs/training/kset_feats.py --split eval_data/selector_split_v0.1.json \
        --sets eval_data/k4_sets_v0.1.json --dtype fp16 --out /kaggle/working/k4/features.npz
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "first_light"))
import first_light_runner_v4 as R

ROUTABLE = list(range(7, 21))
FEATS = ["contrib_norm", "delta_norm", "cos", "entropy", "logitlens_KL"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="fp16")
    ap.add_argument("--out", default="features.npz")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    DT = torch.float16 if a.dtype == "fp16" else torch.bfloat16
    sets = [list(s) for s in json.loads(Path(a.sets).read_text())["sets"]]
    if a.smoke: sets = sets[:5]

    ids = sorted(set(json.loads(Path(a.split).read_text())["all"]))
    idx = {}
    for root, fam in [(R.RESEARCH / "gym-v0.1-FL", "F2_json"), (R.RESEARCH / "gym-v0.3", "F3_type")]:
        for sp in ["dev", "test", "smoke", "train"]:
            for f in (root / fam / sp).glob("*.json"):
                d = json.loads(f.read_text()); idx[d["instance_id"]] = d
    ids = [i for i in ids if i in idx]
    if a.smoke: ids = ids[:3]

    tok = R.AutoTokenizer.from_pretrained(R.MODEL_REPO, revision=R.MODEL_REVISION)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = R.AutoModelForCausalLM.from_pretrained(R.MODEL_REPO, revision=R.MODEL_REVISION, dtype=DT).to(DEV).eval()
    norm = model.model.norm; lm = model.get_output_embeddings()

    def lens(v): return lm(norm(v.unsqueeze(0)))[0].float()

    Lpos = {L: k for k, L in enumerate(ROUTABLE)}
    layer_feats = np.zeros((len(ids), len(ROUTABLE), len(FEATS)), dtype=np.float32)
    set_interf = np.zeros((len(ids), len(sets)), dtype=np.float32)
    fams = []
    print(f"kset_feats on {DEV}/{a.dtype}: {len(ids)} tasks × {len(sets)} sets", flush=True)
    for ti, iid in enumerate(ids):
        inst = idx[iid]; fams.append(inst["metadata"]["family"])
        enc = tok(inst["prompt_context"], return_tensors="pt").to(DEV)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hs = [h[0, -1, :] for h in out.hidden_states]
        pfin = torch.softmax(out.logits[0, -1, :].float(), -1)
        ldelta = {}
        for k, L in enumerate(ROUTABLE):
            hL, hL1 = hs[L], hs[L + 1]; rL = hL1 - hL
            llL1 = lens(hL1); pL1 = torch.softmax(llL1, -1)
            ent = float(-(pL1 * torch.log(pL1 + 1e-10)).sum())
            kl = float((pL1 * (torch.log(pL1 + 1e-10) - torch.log(pfin + 1e-10))).sum())
            layer_feats[ti, k] = [float(rL.norm() / (hL.norm() + 1e-6)), float(rL.norm()),
                                  float(torch.cosine_similarity(hL.float(), hL1.float(), dim=0)), ent, kl]
            ldelta[L] = llL1 - lens(hL)
        for si, sk in enumerate(sets):
            stacked = torch.stack([ldelta[L] for L in sk])          # (k, vocab)
            num = float((stacked.sum(0)).pow(2).sum())              # ||sum||^2
            den = float(stacked.pow(2).sum()) + 1e-6                # sum ||.||^2
            set_interf[ti, si] = num / den
        if (ti + 1) % 25 == 0 or ti == len(ids) - 1:
            print(f"  [{ti+1}/{len(ids)}]", flush=True)
    np.savez(a.out, task_ids=np.array(ids), families=np.array(fams), routable=np.array(ROUTABLE),
             sets=np.array(sets), feat_names=np.array(FEATS), layer_feats=layer_feats, set_interf=set_interf)
    print(f"saved {a.out}: layer_feats {layer_feats.shape}, set_interf {set_interf.shape}")


if __name__ == "__main__":
    main()
