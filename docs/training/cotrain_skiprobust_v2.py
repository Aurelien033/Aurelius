#!/usr/bin/env python3
"""Skip-robust co-train v2 — fixes the v1 catastrophic forgetting (dense 73% -> 22%).

v1 failure root cause: both code datasets were gated, so the corpus collapsed to fineweb-edu (general
web text, NO code), and 2000 steps of strong LoRA (alpha/r=2) overwrote the model's code-repair
behavior. v2 fixes:
  1. CODE IN THE CORPUS  — MBPP (open) + optional open code, mixed ~60/40 with general text. (the #1 fix)
  2. GENTLER TRAINING    — lr 2e-4->5e-5, steps 2000->600, alpha 32->16 (adapter influence /2).
  3. LIVE DRIFT MONITOR  — every 100 steps, measure frozen-vs-cotrained logit drift on a code probe;
                           WARN + checkpoint when it crosses a threshold (v1 ended at 6.28 = way too far).
  4. offline corpus (v1's fix) + checkpoint every 200 steps.
Eval is identical: docs/training/verify_cotrain.py on the held-out E87 test set (the gold harness).

Run on Lightning:  python docs/training/cotrain_skiprobust_v2.py --steps 600 --out checkpoints/cotrain-v2
"""
import argparse, json, random, time, types
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, REV = "Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323"
ROUTABLE = list(range(7, 21))
SKIP_DIST = {0: 0.4, 1: 0.2, 2: 0.2, 4: 0.2}
DRIFT_WARN = 2.5   # logit-diff-from-base above this => over-drifting (v1 hit 6.28 and forgot everything)


def sample_skip(rng):
    ks, ws = zip(*SKIP_DIST.items()); k = rng.choices(ks, weights=ws, k=1)[0]
    return sorted(rng.sample(ROUTABLE, k)) if k else []


def find_decoder_layers(model):
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, nn.ModuleList) and len(m) >= 20 and hasattr(m[0], "self_attn"):
            return m
    raise RuntimeError("no decoder layers")


class IdentitySkip:
    def __init__(self, layers, idxs): self.layers = layers; self.idxs = idxs; self.orig = {}
    def __enter__(self):
        for i in self.idxs:
            self.orig[i] = self.layers[i].forward
            def ident(self_layer, hidden_states, *a, **k): return hidden_states
            self.layers[i].forward = types.MethodType(ident, self.layers[i])
    def __exit__(self, *a):
        for i, o in self.orig.items(): self.layers[i].forward = o
        self.orig.clear()


def build_corpus(tok, target_tokens, code_frac=0.6, cache="/tmp/cotrain_v2_corpus.npy"):
    """Download a fixed mix ONCE: CODE (MBPP, open) + GENERAL (fineweb-edu), then train offline.
    code_frac of the tokens are code — the key to NOT forgetting the code task."""
    if Path(cache).exists():
        arr = np.load(cache)
        if len(arr) >= target_tokens:
            print(f"  corpus cache hit: {len(arr):,} tokens", flush=True); return arr
    from datasets import load_dataset
    eot = tok.eos_token_id
    code, gen = [], []
    n_code = int(target_tokens * code_frac)
    # ---- CODE: MBPP (open) — text + reference solution + tests ----
    print("  loading code (MBPP)...", flush=True)
    for _ in range(50):   # cycle MBPP to reach the code budget (preservation, not memorization-of-new)
        try:
            ds = load_dataset("google-research-datasets/mbpp", "full", split="train")
        except Exception:
            ds = load_dataset("mbpp", "full", split="train")
        for ex in ds:
            blob = (ex.get("text", "") + "\n" + ex.get("code", "") + "\n" + "\n".join(ex.get("test_list", [])))
            code.extend(tok(blob)["input_ids"] + [eot])
        if len(code) >= n_code: break
    # optional extra open code (more diversity); non-fatal
    try:
        cp = load_dataset("codeparrot/github-code-clean", "Python-all", split="train", streaming=True, trust_remote_code=True)
        for ex in cp:
            code.extend(tok(ex.get("code", ""))["input_ids"] + [eot])
            if len(code) >= n_code: break
        print("  + codeparrot python", flush=True)
    except Exception as e:
        print(f"  (codeparrot skipped: {str(e)[:50]}) — MBPP only for code", flush=True)
    # ---- GENERAL: fineweb-edu ----
    print("  loading general (fineweb-edu)...", flush=True)
    for attempt in range(6):
        try:
            fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
            for ex in fw:
                t = ex.get("text") or ""
                if t: gen.extend(tok(t)["input_ids"] + [eot])
                if len(gen) >= target_tokens - n_code: break
            break
        except Exception as e:
            print(f"  fineweb retry {attempt+1}: {str(e)[:50]}", flush=True); time.sleep(5)
    arr = np.array((code[:n_code] + gen[:target_tokens - n_code]), dtype=np.uint16)
    np.save(cache, arr)
    print(f"  corpus ready: {len(arr):,} tokens ({len(code[:n_code]):,} code + {len(gen):,} general)", flush=True)
    return arr


def batches(arr, seq_len, batch, device, rng):
    n = batch * (seq_len + 1)
    while True:
        s = rng.randint(0, len(arr) - n - 1)
        t = torch.tensor(arr[s:s+n].astype("int64"), dtype=torch.long).view(batch, seq_len + 1).to(device)
        yield t[:, :-1], t[:, 1:]


def drift(model, tok, dev):
    probe = tok("def is_prime(n: int) -> bool:", return_tensors="pt").to(dev)
    with torch.no_grad():
        on = model(**probe).logits.float().cpu()
        with model.disable_adapter(): off = model(**probe).logits.float().cpu()
    return float((on - off).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)      # v1 was 2e-4
    ap.add_argument("--alpha", type=int, default=16)       # v1 was 32 (alpha/r 2 -> 1)
    ap.add_argument("--code_frac", type=float, default=0.6)
    ap.add_argument("--out", default="checkpoints/cotrain-v2")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke: a.steps = 30
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    rng = random.Random(1337); torch.manual_seed(1337)

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16).to(dev)
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=a.alpha, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=ROUTABLE, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    layers = find_decoder_layers(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)

    corpus = build_corpus(tok, max(a.steps * a.batch * (a.seq_len + 1) * 2, 300_000), code_frac=a.code_frac)
    data = batches(corpus, a.seq_len, a.batch, dev, random.Random(7))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.train(); losses = []; t0 = time.time()
    for step in range(a.steps):
        xb, yb = next(data); skip = sample_skip(rng)
        with IdentitySkip(layers, skip) if skip else _null():
            logits = model(input_ids=xb).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); opt.step()
        losses.append(loss.item())
        if step % 20 == 0 or step == a.steps - 1:
            print(f"  step {step:4d}  loss {loss.item():.3f}  k={len(skip)}  ({(time.time()-t0)/(step+1):.2f}s/step)", flush=True)
        if step > 0 and step % 100 == 0:
            model.eval(); d = drift(model, tok, dev); model.train()
            flag = "  ⚠ OVER-DRIFT (forgetting risk)" if d > DRIFT_WARN else ""
            print(f"  [drift @ step {step}: logit-diff-from-base = {d:.2f}{flag}]", flush=True)
            model.save_pretrained(a.out)

    model.eval(); d_final = drift(model, tok, dev)
    lora_norm = float(sum(p.detach().float().norm()**2 for n, p in model.named_parameters() if "lora" in n.lower() and p.requires_grad) ** 0.5)
    model.save_pretrained(a.out)
    init, final = float(np.mean(losses[:5])), float(np.mean(losses[-5:]))
    (Path(a.out)/"cotrain_result.json").write_text(json.dumps({
        "recipe": "v2", "model": {"repo": MODEL, "rev": REV}, "steps": a.steps, "lr": a.lr, "alpha": a.alpha,
        "code_frac": a.code_frac, "init_loss": init, "final_loss": final, "drop": init-final,
        "self_proof": {"final_drift_logit_diff": d_final, "lora_norm": lora_norm, "adapters_active": d_final > 0.5},
        "drift_note": f"final drift {d_final:.2f} (v1 ended at 6.28 = catastrophic; aim < {DRIFT_WARN})"}, indent=2))
    print(f"\n  init {init:.3f} -> final {final:.3f} | FINAL DRIFT {d_final:.2f} (v1=6.28; lower=less forgetting)")
    print(f"  adapters -> {a.out}/  — bring back the WHOLE folder for verify_cotrain.py")


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
