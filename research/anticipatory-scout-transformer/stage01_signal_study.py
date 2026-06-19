"""
Stage 0/1 — signal study on a FROZEN pretrained causal LM.
Spec: experiment-1-spec.md v2.

Tests the load-bearing premise WITHOUT training: does a cheap per-token disagreement
signal (MC-dropout JSD) predict where MORE COMPUTE (here: more transformer depth, via
early-exit) reduces per-token loss? With the controls that stop it from lying:
  - document-level cluster bootstrap (tokens are autocorrelated) -> CI lower bound is the gate
  - tail metric (NLL captured by top-x%-by-S vs oracle ceiling), not just global Spearman
  - iso-FLOP true policy simulation (signed realized benefit)
  - CALM single-pass-entropy baseline (free)
  - instrument-validity gate: dropout p=0 -> Plan-B, NOT a RED

NOTE: early-exit uses a raw LOGIT lens (final_norm + lm_head on hidden states). The spec
calls for a TUNED lens; logit-lens benefit is a rough proxy -> treat Variant-A results as
confirmatory. Variant B (cross-scale) needs byte-identical tokenizers; not implemented here.

Usage:
  python stage01_signal_study.py --model <hf-id> --data docs.txt --T 16 --tiers 0.5,0.75,1.0
Authored without execution — smoke-test with a tiny model and --max_docs 4 first.
"""
import argparse, math
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------- helpers -------------------------------
def find_final_norm(model):
    for path in ["model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm", "model.final_layernorm"]:
        obj = model
        try:
            for a in path.split("."): obj = getattr(obj, a)
            return obj
        except AttributeError:
            continue
    print("  [warn] final norm not found; using identity (logit-lens will be noisier)")
    return torch.nn.Identity()

def token_nll(logits, input_ids):                 # per-position NLL of the gold NEXT token
    lp = F.log_softmax(logits[:, :-1].float(), -1)
    tgt = input_ids[:, 1:]
    return -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # (B, L-1), indexed at PREDICTED token

def set_dropout(model, on):
    n = 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.train(on); n += (mod.p > 0)
    return n

# ------------------------------- core passes -------------------------------
@torch.no_grad()
def mc_dropout_jsd(model, ids, T):                # streaming JSD (never stacks T x V)
    set_dropout(model, True)
    mean_p, mean_H = None, None
    for _ in range(T):
        lp = F.log_softmax(model(ids).logits[:, :-1].float(), -1)
        p = lp.exp()
        mean_p = p / T if mean_p is None else mean_p + p / T
        h = -(p * lp).sum(-1)
        mean_H = h / T if mean_H is None else mean_H + h / T
    set_dropout(model, False)
    H_mean = -(mean_p * mean_p.clamp_min(1e-12).log()).sum(-1)
    return (H_mean - mean_H).clamp_min(0)         # (B, L-1)

@torch.no_grad()
def depth_benefit(model, fnorm, head, ids, tiers):
    out = model(ids, output_hidden_states=True)
    hs = out.hidden_states                         # tuple: [0]=emb, [i]=block-i out
    L = len(hs) - 1
    full_nll = token_nll(out.logits, ids)          # FULL = real logits (not lens) per spec C1
    tier_nll = []
    for frac in tiers:
        layer = max(1, min(L, round(frac * L)))
        lens_logits = head(fnorm(hs[layer]))
        tier_nll.append(token_nll(lens_logits, ids))
    entropy = -(F.softmax(out.logits[:, :-1].float(), -1)
                * F.log_softmax(out.logits[:, :-1].float(), -1)).sum(-1)  # CALM baseline, free
    return full_nll, tier_nll, entropy

# ------------------------------- stats -------------------------------
def spearman(a, b):
    ra = a.argsort().argsort().astype(float); rb = b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = math.sqrt((ra*ra).sum() * (rb*rb).sum()) + 1e-12
    return float((ra*rb).sum() / denom)

def doc_bootstrap_ci(S, B, doc_ids, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    docs = np.unique(doc_ids); rhos = []
    for _ in range(n_boot):
        pick = rng.choice(docs, size=len(docs), replace=True)
        idx = np.concatenate([np.where(doc_ids == d)[0] for d in pick])
        rhos.append(spearman(S[idx], B[idx]))
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return spearman(S, B), float(lo), float(hi)

def tail_capture(S, B, frac=0.2):                 # NLL benefit captured by top-frac by S vs oracle
    n = len(S); k = max(1, int(frac*n))
    by_s = B[np.argsort(-S)[:k]].sum()
    oracle = B[np.argsort(-B)[:k]].sum()
    return float(by_s / (oracle + 1e-12))

def iso_flop(S, B, frac=0.2):                      # true policy sim: allocate by S, credit signed B
    n = len(S); k = max(1, int(frac*n))
    targeted = B[np.argsort(-S)[:k]].sum()         # spend extra compute on top-frac by S
    uniform = B.sum() * frac                       # same budget spread evenly (expected)
    oracle = B[np.argsort(-B)[:k]].sum()           # ceiling: allocate by true B (spec §7.1 headroom)
    return float(targeted), float(uniform), float(oracle)

# ------------------------------- main -------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--data", required=True)
    p.add_argument("--ref_model", default="")     # optional independent difficulty label
    p.add_argument("--T", type=int, default=16); p.add_argument("--tiers", default="0.5,0.75,1.0")
    p.add_argument("--max_docs", type=int, default=200); p.add_argument("--maxlen", type=int, default=256)
    p.add_argument("--tail_frac", type=float, default=0.2); p.add_argument("--rho_gate", type=float, default=0.4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    tiers = [float(x) for x in args.tiers.split(",")]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()
    fnorm, head = find_final_norm(model), model.get_output_embeddings()

    # QC gate: the logit lens (head . final_norm . hidden[-1]) must reconstruct the real logits,
    # else early-exit benefit B is computed on the wrong scale and is meaningless. Refuse to score.
    with torch.no_grad():
        probe = tok("the quick brown fox jumps", return_tensors="pt").input_ids.to(args.device)
        o = model(probe, output_hidden_states=True)
        qc = torch.allclose(head(fnorm(o.hidden_states[-1])), o.logits, atol=1e-3)
    if not qc:
        raise SystemExit("[QC FAIL] head(final_norm(hidden[-1])) != logits -> wrong norm/head for this "
                         "architecture. Fix find_final_norm before scoring (early-exit B would be garbage).")
    print("[QC] logit-lens reconstruction matches full logits: OK")

    n_active = set_dropout(model, False)
    print(f"[instrument gate] dropout modules with p>0: {n_active}")
    if n_active == 0:
        print("  *** dropout p=0 -> MC-dropout is inert. This is an INSTRUMENT FAILURE, not a RED.")
        print("  *** Plan-B: inject activation noise, or use a seed/size committee. Continuing for plumbing test only.")

    docs = [d.strip() for d in open(args.data, encoding="utf-8").read().split("\n\n") if d.strip()][:args.max_docs]
    Sall, Ball, Doc, Ent = [], [], [], []
    marg_tier = 0                                  # marginal benefit at scout->next tier (index 0->1)
    for di, doc in enumerate(docs):
        ids = tok(doc, return_tensors="pt", truncation=True, max_length=args.maxlen).input_ids.to(args.device)
        if ids.shape[1] < 3: continue
        S = mc_dropout_jsd(model, ids, args.T)[0]
        full, tier_nll, ent = depth_benefit(model, fnorm, head, ids, tiers)
        # marginal benefit: NLL(tier 0) - NLL(tier 1) per spec (signed)
        B = (tier_nll[marg_tier] - tier_nll[marg_tier + 1])[0] if len(tier_nll) > 1 else (tier_nll[0] - full)[0]
        mask = torch.ones_like(S, dtype=torch.bool); mask[:1] = False   # drop BOS-adjacent
        Sall.append(S[mask].cpu().numpy()); Ball.append(B[mask].cpu().numpy())
        Ent.append(ent[0][mask].cpu().numpy()); Doc.append(np.full(mask.sum().item(), di))
        if di % 50 == 0: print(f"  doc {di}/{len(docs)}")

    S = np.concatenate(Sall); B = np.concatenate(Ball)
    E = np.concatenate(Ent); doc_ids = np.concatenate(Doc)
    print(f"\nscored tokens: {len(S)}  documents: {len(np.unique(doc_ids))}")

    rho, lo, hi = doc_bootstrap_ci(S, B, doc_ids)
    rho_e, lo_e, hi_e = doc_bootstrap_ci(E, B, doc_ids)       # CALM baseline
    cap = tail_capture(S, B, args.tail_frac)
    targ, unif, orac = iso_flop(S, B, args.tail_frac)
    print(f"\n[Spearman(surprise, benefit)]   rho={rho:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"[Spearman(entropy,  benefit)]   rho={rho_e:.3f}  95% CI [{lo_e:.3f}, {hi_e:.3f}]   (CALM baseline)")
    print(f"[tail capture top-{int(args.tail_frac*100)}%]  {cap:.2%} of oracle benefit")
    print(f"[iso-FLOP]  surprise-targeted={targ:.2f}  uniform={unif:.2f}  oracle={orac:.2f}  "
          f"(captured {targ/(orac+1e-9):.0%} of oracle)  -> {'WIN' if targ>unif else 'TIE/LOSS'}")
    green = (lo > args.rho_gate) and (targ > unif)
    strong = green and (lo > hi_e)                # JSD CI lower bound exceeds entropy CI UPPER bound
    print(f"\nVERDICT: {'STRONG-GREEN' if strong else 'GREEN' if green else 'YELLOW/RED'} "
          f"(gate: CI lower bound > {args.rho_gate} AND beats iso-FLOP uniform; "
          f"STRONG also needs JSD's CI lower bound to exceed the entropy CI upper bound)")
    print("Reminder: this is the NECESSARY proxy claim (depth), not the loop axis. Stage 2 is the build gate.")

if __name__ == "__main__":
    main()
