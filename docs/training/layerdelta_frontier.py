#!/usr/bin/env python3
"""LayerDelta follow-ups (OBL-062 positive -> push it): the SUBSTITUTION FRONTIER + OUT-OF-FAMILY generalization.

Builds on layerdelta.py (cheap rank-r surrogate replacing a skip-set's layers, distilled to dense, MATCHED
dense at k=4 where identity-skip lost 12pp). Two questions, one script:

  A. FRONTIER  (--k_list 2,4,6,8,10,12): identity-skip pruning broke at ~2 layers; how far does SUBSTITUTION
     go? For each k, substitute the top-k SAFEST-to-skip layers (nested), train fresh surrogates, compare
     dense / identity-skip(S_k) / layerdelta(S_k). The curves show where substitution degrades.
  B. GENERALIZATION (--train_family F2_json --eval_family F3_type): distill surrogates on ONE family, eval on
     ANOTHER. If cross-family ~ same-family, the surrogates learned the layers' GENERAL function (a real model
     improvement), not task-specific tricks.

Run (Kaggle):
  FRONTIER:        python docs/training/layerdelta_frontier.py --k_list 2,4,6,8,10,12 \
                     --eval_split eval_data/selector_split_k4_v0.1.json --out_dir /kaggle/working/ldf --dtype bf16
  GENERALIZATION:  python docs/training/layerdelta_frontier.py --k_list 4 --train_family F2_json --eval_family F3_type \
                     --eval_split eval_data/selector_split_k4_v0.1.json --out_dir /kaggle/working/ldg --dtype bf16
"""
import argparse, json, sys, types, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "first_light"))
import first_light_runner_v4 as R

ROUTABLE = list(range(7, 21))
# layers ordered SAFEST-to-skip first (from the k=2 layer-inclusion effects); poison {11,14,15} last:
SAFE_ORDER = [18, 9, 16, 10, 8, 19, 13, 20, 12, 7, 17, 15, 11, 14]


def find_layers(model):
    for m in model.modules():
        if isinstance(m, nn.ModuleList) and len(m) >= 20 and hasattr(m[0], "self_attn"):
            return m
    raise RuntimeError("no decoder layers")


class Delta(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.V = nn.Linear(d, r, bias=False); self.U = nn.Linear(r, d, bias=True)
        nn.init.normal_(self.V.weight, std=1e-3); nn.init.zeros_(self.U.weight); nn.init.zeros_(self.U.bias)
    def forward(self, h): return self.U(self.V(h))


class _null:
    def __enter__(self): return None
    def __exit__(self, *x): return False


class Skip:
    def __init__(self, layers, sset, mode="zero", deltas=None):
        self.layers = layers; self.sset = sset; self.mode = mode; self.deltas = deltas; self.orig = {}
    def __enter__(self):
        for l in self.sset:
            self.orig[l] = self.layers[l].forward
            if self.mode == "zero":
                def fwd(self_layer, hidden_states, *a, **k): return hidden_states
            else:
                _d = self.deltas[l]
                def fwd(self_layer, hidden_states, *a, _d=_d, **k): return hidden_states + _d(hidden_states).to(hidden_states.dtype)
            self.layers[l].forward = types.MethodType(fwd, self.layers[l])
    def __exit__(self, *a):
        for l, o in self.orig.items(): self.layers[l].forward = o
        self.orig.clear()


def batched_gen(tok, model, prompts, ctx_factory, dev, bs, max_new):
    outs = []
    for s in range(0, len(prompts), bs):
        enc = tok(prompts[s:s+bs], return_tensors="pt", padding=True).to(dev)
        with ctx_factory(), torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        L = enc["input_ids"].shape[1]
        outs.extend(tok.decode(r, skip_special_tokens=True) for r in gen[:, L:])
    return outs


def train_surrogates(model, tok, layers, S, seqs, dev, dt, rank, steps, lr, batch, rng):
    d = model.config.hidden_size
    deltas = {l: Delta(d, rank).to(dev, dt) for l in S}
    for dl in deltas.values():
        for p in dl.parameters(): p.requires_grad_(True)
    opt = torch.optim.AdamW([p for dl in deltas.values() for p in dl.parameters()], lr=lr)
    for step in range(steps):
        bt = [seqs[rng.randrange(len(seqs))] for _ in range(batch)]
        maxlen = max(len(x[0]) for x in bt)
        ids = torch.full((len(bt), maxlen), tok.pad_token_id, dtype=torch.long)
        cmask = torch.zeros((len(bt), maxlen), dtype=torch.bool)
        for bi, (toks, cs) in enumerate(bt):
            ids[bi, :len(toks)] = torch.tensor(toks); cmask[bi, cs-1:len(toks)-1] = True
        ids = ids.to(dev); amask = (ids != tok.pad_token_id).long()
        cmask = cmask.to(dev)
        with torch.no_grad():
            t = model(input_ids=ids, attention_mask=amask).logits.float()
        with Skip(layers, S, "delta", deltas):
            s = model(input_ids=ids, attention_mask=amask).logits.float()
        tlp = F.log_softmax(t[cmask], -1)
        loss = (tlp.exp() * (tlp - F.log_softmax(s[cmask], -1))).sum(-1).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for dl in deltas.values() for p in dl.parameters()], 1.0)
        opt.step()
    return deltas, float(loss.item())


def passrate(tok, model, prompts, insts, vfs, ctx, dev, gb, mn):
    comps = batched_gen(tok, model, prompts, ctx, dev, gb, mn)
    return float(np.mean([bool(vfs[i](insts[i], comps[i])) for i in range(len(insts))]))


def load_idx():
    idx = {}
    for root, fam in [(R.RESEARCH/"gym-v0.1-FL", "F2_json"), (R.RESEARCH/"gym-v0.3", "F3_type")]:
        for sp in ["dev", "test", "smoke", "train"]:
            for f in (root/fam/sp).glob("*.json"):
                d = json.loads(f.read_text()); idx[d["instance_id"]] = d
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k_list", default="2,4,6,8,10,12")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--eval_split", required=True)
    ap.add_argument("--train_family", choices=["both", "F2_json", "F3_type"], default="both")
    ap.add_argument("--eval_family", choices=["both", "F2_json", "F3_type"], default="both")
    ap.add_argument("--n_train", type=int, default=120)
    ap.add_argument("--n_eval", type=int, default=0, help="cap eval set size (0=all) — for a tight GPU budget")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--gen_batch", type=int, default=24)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    ks = [int(x) for x in a.k_list.split(",")]
    R.MAX_NEW = a.max_new
    rd = Path(a.out_dir); rd.mkdir(parents=True, exist_ok=True)
    DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    DT = torch.float16 if a.dtype == "fp16" else torch.bfloat16
    rng = random.Random(1337)

    idx = load_idx()
    eval_ids = sorted(set(json.loads(Path(a.eval_split).read_text())["all"]))
    eval_ids = [i for i in eval_ids if i in idx and (a.eval_family == "both" or idx[i]["metadata"]["family"] == a.eval_family)]
    if a.n_eval: eval_ids = eval_ids[:a.n_eval]
    pool = [i for i in idx if i not in set(eval_ids) and (a.train_family == "both" or idx[i]["metadata"]["family"] == a.train_family)]
    rng.shuffle(pool); train_ids = pool[:a.n_train]
    if a.smoke: ks, eval_ids, train_ids, a.steps = [2, 4], eval_ids[:4], train_ids[:4], 6

    tok = R.AutoTokenizer.from_pretrained(R.MODEL_REPO, revision=R.MODEL_REVISION)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = R.AutoModelForCausalLM.from_pretrained(R.MODEL_REPO, revision=R.MODEL_REVISION, dtype=DT).to(DEV).eval()
    for p in model.parameters(): p.requires_grad_(False)
    layers = find_layers(model)
    print(f"LayerDelta-frontier {DEV}/{a.dtype}: k={ks} rank={a.rank} | train_fam={a.train_family}({len(train_ids)}) "
          f"eval_fam={a.eval_family}({len(eval_ids)}) | SAFE_ORDER nested", flush=True)

    # distillation data: dense completion per train task (once)
    tprompts = [idx[i]["prompt_context"] for i in train_ids]
    tcomps = batched_gen(tok, model, tprompts, lambda: _null(), DEV, a.gen_batch, a.max_new)
    seqs = []
    for pr, co in zip(tprompts, tcomps):
        pids = tok(pr)["input_ids"]; full = tok(pr + co)["input_ids"]
        if len(full) > len(pids) + 1: seqs.append((full, len(pids)))
    # dense + random baselines (k-independent), once
    eprompts = [idx[i]["prompt_context"] for i in eval_ids]; einst = [idx[i] for i in eval_ids]
    vfs = [R.VERIFIERS[i["metadata"]["family"]] for i in einst]
    dense = passrate(tok, model, eprompts, einst, vfs, lambda: _null(), DEV, a.gen_batch, a.max_new)
    print(f"  dense (k-independent): {dense*100:.1f}%   distill seqs {len(seqs)}", flush=True)

    rows = []
    for k in ks:
        S = sorted(SAFE_ORDER[:k])
        deltas, klf = train_surrogates(model, tok, layers, S, seqs, DEV, DT, a.rank, a.steps, a.lr, a.batch, rng)
        isk = passrate(tok, model, eprompts, einst, vfs, lambda: Skip(layers, S, "zero"), DEV, a.gen_batch, a.max_new)
        ld = passrate(tok, model, eprompts, einst, vfs, lambda: Skip(layers, S, "delta", deltas), DEV, a.gen_batch, a.max_new)
        row = {"k": k, "set": S, "dense": dense, "identity_skip": isk, "layerdelta": ld,
               "ld_minus_isk": round((ld-isk)*100, 1), "ld_minus_dense": round((ld-dense)*100, 1), "final_KL": round(klf, 4)}
        rows.append(row)
        print(f"  k={k:2d} {S}: identity_skip {isk*100:5.1f}%  layerdelta {ld*100:5.1f}%  "
              f"(LD-skip {row['ld_minus_isk']:+.1f}, LD-dense {row['ld_minus_dense']:+.1f})", flush=True)
        (rd/"frontier_result.json").write_text(json.dumps({"dense": dense, "train_family": a.train_family,
            "eval_family": a.eval_family, "rank": a.rank, "rows": rows}, indent=2))

    print(f"\n  FRONTIER (dense {dense*100:.1f}%):")
    for r in rows:
        print(f"    k={r['k']:2d}: identity-skip {r['identity_skip']*100:5.1f}%  ->  LayerDelta {r['layerdelta']*100:5.1f}%  "
              f"(recovers {r['ld_minus_isk']:+.1f}pp; vs dense {r['ld_minus_dense']:+.1f}pp)")
    print("  Read: how far in k does LayerDelta stay ~dense while identity-skip collapses?")
    print(f"  -> {rd}/frontier_result.json")


if __name__ == "__main__":
    main()
