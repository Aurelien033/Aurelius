#!/usr/bin/env python3
"""LayerDelta substitution (OBL-062 / layerdelta_preregistration.yaml) — the "soft prune".

Replace each layer l in a skip-set S with a CHEAP rank-r learned surrogate:  h_{l+1} = h_l + Delta_hat_l(h_l),
Delta_hat_l(h) = U_l(V_l h)  (starts at 0 = identity-skip). The surrogates are trained JOINTLY by distilling
the S-substituted model toward the DENSE model (logit-KL, teacher-forced on prompt + dense's own greedy
completion) on a TRAIN split disjoint from eval. Then eval on held-out tasks vs dense / identity-skip /
random. Tests H-LD-0 (beat identity-skip) and H-LD-1 (reach dense at less compute).

The k=4 prune loses ~12pp vs dense; LayerDelta asks whether a cheap surrogate RECOVERS that.

Run (Kaggle): python docs/training/layerdelta.py --set 8,15,16,19 --rank 32 \
   --eval_split eval_data/selector_split_k4_v0.1.json --out_dir /kaggle/working/ld --dtype bf16
"""
import argparse, json, sys, types, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "first_light"))
import first_light_runner_v4 as R

ROUTABLE = list(range(7, 21))


def find_layers(model):
    for m in model.modules():
        if isinstance(m, nn.ModuleList) and len(m) >= 20 and hasattr(m[0], "self_attn"):
            return m
    raise RuntimeError("no decoder layers")


class Delta(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.V = nn.Linear(d, r, bias=False)
        self.U = nn.Linear(r, d, bias=True)
        nn.init.normal_(self.V.weight, std=1e-3); nn.init.zeros_(self.U.weight); nn.init.zeros_(self.U.bias)  # Delta_hat(h)=0 at init = identity-skip
    def forward(self, h): return self.U(self.V(h))


class _null:
    def __enter__(self): return None
    def __exit__(self, *x): return False


class Skip:
    """ctx mgr: for l in sset, replace layer.forward. mode='zero' (identity-skip) or 'delta' (h+Delta(h))."""
    def __init__(self, layers, sset, mode="zero", deltas=None):
        self.layers = layers; self.sset = sset; self.mode = mode; self.deltas = deltas; self.orig = {}
    def __enter__(self):
        for l in self.sset:
            self.orig[l] = self.layers[l].forward
            if self.mode == "zero":
                def fwd(self_layer, hidden_states, *a, **k): return hidden_states
            else:
                d = self.deltas[l]
                def fwd(self_layer, hidden_states, *a, _d=d, **k): return hidden_states + _d(hidden_states).to(hidden_states.dtype)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="comma-separated skip-set, e.g. 8,15,16,19")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--eval_split", required=True)
    ap.add_argument("--n_train", type=int, default=150)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--gen_batch", type=int, default=24)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    S = [int(x) for x in a.set.split(",")]
    R.MAX_NEW = a.max_new
    rd = Path(a.out_dir); rd.mkdir(parents=True, exist_ok=True)
    DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    DT = torch.float16 if a.dtype == "fp16" else torch.bfloat16
    rng = random.Random(1337)

    # ---- data: eval = the split (held out from surrogate training); train = disjoint pool instances ----
    eval_ids = sorted(set(json.loads(Path(a.eval_split).read_text())["all"]))
    idx = {}
    for root, fam in [(R.RESEARCH/"gym-v0.1-FL", "F2_json"), (R.RESEARCH/"gym-v0.3", "F3_type")]:
        for sp in ["dev", "test", "smoke", "train"]:
            for f in (root/fam/sp).glob("*.json"):
                d = json.loads(f.read_text()); idx[d["instance_id"]] = d
    eval_ids = [i for i in eval_ids if i in idx]
    pool = [i for i in idx if i not in set(eval_ids)]; rng.shuffle(pool)
    train_ids = pool[:a.n_train]
    if a.smoke: eval_ids, train_ids, a.steps = eval_ids[:4], train_ids[:4], 6

    tok = R.AutoTokenizer.from_pretrained(R.MODEL_REPO, revision=R.MODEL_REVISION)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # batched generation needs left padding (training builds its own right-pad below)
    model = R.AutoModelForCausalLM.from_pretrained(R.MODEL_REPO, revision=R.MODEL_REVISION, dtype=DT).to(DEV).eval()
    for p in model.parameters(): p.requires_grad_(False)
    layers = find_layers(model)
    d = model.config.hidden_size
    deltas = {l: Delta(d, a.rank).to(DEV, DT) for l in S}
    for dl in deltas.values():
        for p in dl.parameters(): p.requires_grad_(True)
    print(f"LayerDelta on {DEV}/{a.dtype}: set={S} rank={a.rank} | train {len(train_ids)} eval {len(eval_ids)} | "
          f"surrogate params {sum(sum(p.numel() for p in dl.parameters()) for dl in deltas.values()):,}", flush=True)

    # ---- training data: dense greedy completion per train task (the distillation target) ----
    tprompts = [idx[i]["prompt_context"] for i in train_ids]
    tcomps = batched_gen(tok, model, tprompts, lambda: _null(), DEV, a.gen_batch, a.max_new)
    seqs = []
    for pr, co in zip(tprompts, tcomps):
        pids = tok(pr)["input_ids"]; full = tok(pr + co)["input_ids"]
        if len(full) > len(pids) + 1:
            seqs.append((full, len(pids)))   # (token ids, completion-start index)
    print(f"  distillation seqs: {len(seqs)} (prompt+dense-completion)", flush=True)

    # ---- train the surrogates: distill S-substituted -> dense on completion positions ----
    opt = torch.optim.AdamW([p for dl in deltas.values() for p in dl.parameters()], lr=a.lr)
    for step in range(a.steps):
        batch = [seqs[rng.randrange(len(seqs))] for _ in range(a.batch)]
        maxlen = max(len(x[0]) for x in batch)
        ids = torch.full((len(batch), maxlen), tok.pad_token_id, dtype=torch.long)
        cmask = torch.zeros((len(batch), maxlen), dtype=torch.bool)
        for bi, (toks, cs) in enumerate(batch):
            ids[bi, :len(toks)] = torch.tensor(toks)
            cmask[bi, cs-1:len(toks)-1] = True       # positions predicting completion tokens
        ids = ids.to(DEV); cmask = cmask.to(DEV)
        amask = (ids != tok.pad_token_id).long()
        with torch.no_grad():
            t_logits = model(input_ids=ids, attention_mask=amask).logits.float()          # dense teacher
        with Skip(layers, S, "delta", deltas):
            s_logits = model(input_ids=ids, attention_mask=amask).logits.float()          # student
        tlp = F.log_softmax(t_logits[cmask], -1)
        loss = (tlp.exp() * (tlp - F.log_softmax(s_logits[cmask], -1))).sum(-1).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for dl in deltas.values() for p in dl.parameters()], 1.0)
        opt.step()
        if step % 50 == 0 or step == a.steps - 1:
            print(f"  step {step:4d}  KL {loss.item():.4f}", flush=True)

    # ---- eval: 4 policies on the held-out split ----
    eprompts = [idx[i]["prompt_context"] for i in eval_ids]
    einst = [idx[i] for i in eval_ids]
    vfs = [R.VERIFIERS[i["metadata"]["family"]] for i in einst]
    rk = rng.sample(ROUTABLE, len(S))   # one fixed random skip-set of the same size
    policies = {"dense": lambda: _null(),
                "identity_skip": lambda: Skip(layers, S, "zero"),
                "random_skip": lambda: Skip(layers, rk, "zero"),
                "layerdelta": lambda: Skip(layers, S, "delta", deltas)}
    res = {}
    for name, ctx in policies.items():
        comps = batched_gen(tok, model, eprompts, ctx, DEV, a.gen_batch, a.max_new)
        passed = [bool(vfs[i](einst[i], comps[i])) for i in range(len(eval_ids))]
        res[name] = float(np.mean(passed))
        print(f"  {name:14s}: {res[name]*100:.1f}%", flush=True)

    out = {"set": S, "rank": a.rank, "n_train": len(seqs), "eval_n": len(eval_ids), "rates": res,
           "H_LD_0_layerdelta_minus_identityskip_pp": round((res["layerdelta"]-res["identity_skip"])*100, 1),
           "H_LD_1_layerdelta_minus_dense_pp": round((res["layerdelta"]-res["dense"])*100, 1)}
    (rd/"layerdelta_result.json").write_text(json.dumps(out, indent=2))
    print(f"\n  H-LD-0  layerdelta − identity_skip = {out['H_LD_0_layerdelta_minus_identityskip_pp']:+.1f}pp  (surrogate beats zeros?)")
    print(f"  H-LD-1  layerdelta − dense          = {out['H_LD_1_layerdelta_minus_dense_pp']:+.1f}pp  (soft prune >= full model?)")
    print(f"  -> {rd}/layerdelta_result.json")


if __name__ == "__main__":
    main()
