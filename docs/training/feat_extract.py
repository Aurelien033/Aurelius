#!/usr/bin/env python3
"""PROMPT-CONDITIONED features for the per-task selector — ONE dense prefill per task, no generation.

Per routable layer L (h_L = residual into L, r_L = h_{L+1}-h_L = L's residual contribution, all at the
last token):
  contrib_norm = ||r_L||/||h_L||,  delta_norm = ||r_L||,  cos = cos(h_L,h_{L+1}),
  entropy = H(logit-lens readout of h_{L+1}),  logitlens_KL = KL(readout_L || final logits)
Per pair (i,j): pair_align = cos(logitlens_delta_i, logitlens_delta_j) where logitlens_delta_L =
lm_head(norm(h_{L+1})) - lm_head(norm(h_L)) is L's image in logit space. The cross term is the
MECHANISTIC non-additivity: bad pairs = two layers whose removed contributions point the same way in
task-relevant logit directions.

Run:  python docs/training/feat_extract.py --split eval_data/selector_split_v0.1.json --dtype fp16 --out features.npz
"""
import argparse, json, itertools, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "first_light"))
import first_light_runner_v4 as R

ROUTABLE = list(range(7, 21))
PAIRS = list(itertools.combinations(ROUTABLE, 2))   # 91
FEATS = ["contrib_norm", "delta_norm", "cos", "entropy", "logitlens_KL"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="fp16")
    ap.add_argument("--out", default="features.npz")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    DT = torch.float16 if a.dtype == "fp16" else torch.bfloat16

    ids = (sorted(set(json.loads(Path(a.split).read_text())["all"])) if a.split
           else sorted(set(json.loads((R.RESEARCH / "e87_receipt" / "e87_split.json").read_text())["test"])))
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

    def lens(v):  # logit lens of a hidden vector -> logits (float32, vocab)
        return lm(norm(v.unsqueeze(0)))[0].float()

    layer_feats = np.zeros((len(ids), len(ROUTABLE), len(FEATS)), dtype=np.float32)
    pair_align = np.zeros((len(ids), len(PAIRS)), dtype=np.float32)
    fams = []
    print(f"feat_extract on {DEV}/{a.dtype}: {len(ids)} tasks", flush=True)
    for ti, iid in enumerate(ids):
        inst = idx[iid]; fams.append(inst["metadata"]["family"])
        enc = tok(inst["prompt_context"], return_tensors="pt").to(DEV)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hs = [h[0, -1, :] for h in out.hidden_states]          # last token, 29 layers
        pfin = torch.softmax(out.logits[0, -1, :].float(), -1)
        ldelta = {}
        for k, L in enumerate(ROUTABLE):
            hL, hL1 = hs[L], hs[L + 1]; rL = hL1 - hL
            llL1 = lens(hL1)
            pL1 = torch.softmax(llL1, -1)
            ent = float(-(pL1 * torch.log(pL1 + 1e-10)).sum())
            kl = float((pL1 * (torch.log(pL1 + 1e-10) - torch.log(pfin + 1e-10))).sum())
            layer_feats[ti, k] = [float(rL.norm() / (hL.norm() + 1e-6)), float(rL.norm()),
                                  float(torch.cosine_similarity(hL.float(), hL1.float(), dim=0)), ent, kl]
            ldelta[L] = llL1 - lens(hL)
        for pi, (i, j) in enumerate(PAIRS):
            pair_align[ti, pi] = float(torch.cosine_similarity(ldelta[i], ldelta[j], dim=0))
        if (ti + 1) % 25 == 0 or ti == len(ids) - 1:
            print(f"  [{ti+1}/{len(ids)}]", flush=True)
    np.savez(a.out, task_ids=np.array(ids), families=np.array(fams), routable=np.array(ROUTABLE),
             pairs=np.array(PAIRS), feat_names=np.array(FEATS), layer_feats=layer_feats, pair_align=pair_align)
    print(f"saved {a.out}: layer_feats {layer_feats.shape}, pair_align {pair_align.shape}")


if __name__ == "__main__":
    main()
