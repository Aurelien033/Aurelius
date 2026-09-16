#!/usr/bin/env python3
"""Skip-robust co-train — LoRA fine-tune FROZEN-BASE-v1 to be robust to layer-skipping (E87 continuation).

Tests whether co-training under the skip distribution shrinks the ~43pp k=4 skip cost E87 found on the
frozen base, and makes routing learnable. Trains ONLY LoRA adapters; per step samples a skip-set and
applies it (identity-skip on routable layers) during the forward, so the adapters learn to compensate
for skipped layers. Corpus streams from HF (no big download). Designed for a single 16-24GB GPU
(Lightning.AI free tier). Eval = re-run docs/first_light/e87_chessboard.py with the saved adapters.

Run (on Lightning, after `pip install peft datasets`):
  python docs/training/cotrain_skiprobust.py --steps 2000 --out checkpoints/cotrain-skiprobust
"""
import argparse, json, random, time, types
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, REV = "Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323"
ROUTABLE = list(range(7, 21))
SKIP_DIST = {0: 0.4, 1: 0.2, 2: 0.2, 4: 0.2}   # frozen at prereg (cotrain_skiprobust_preregistration.yaml)


def sample_skip(rng):
    ks, ws = zip(*SKIP_DIST.items())
    k = rng.choices(ks, weights=ws, k=1)[0]
    return sorted(rng.sample(ROUTABLE, k)) if k else []


def find_decoder_layers(model):
    """Robustly locate the decoder-layer ModuleList regardless of peft/HF wrapping."""
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, nn.ModuleList) and len(m) >= 20 and hasattr(m[0], "self_attn"):
            return m
    raise RuntimeError("could not locate decoder layers")


class IdentitySkip:
    """Spike-verified identity-skip; passes gradient through (identity), so LoRA can train under it."""
    def __init__(self, layers, idxs): self.layers = layers; self.idxs = idxs; self.orig = {}
    def __enter__(self):
        for i in self.idxs:
            self.orig[i] = self.layers[i].forward
            def ident(self_layer, hidden_states, *a, **k): return hidden_states
            self.layers[i].forward = types.MethodType(ident, self.layers[i])
    def __exit__(self, *a):
        for i, o in self.orig.items(): self.layers[i].forward = o
        self.orig.clear()


def load_fixed_corpus(tok, target_tokens, cache="/tmp/cotrain_corpus.npy"):
    """Download a fixed chunk of fineweb-edu ONCE (with retries) and tokenize to a local .npy, then
    train OFFLINE from it — no streaming during the loop, so a network hiccup can't crash training.
    Cached, so reruns are instant."""
    import numpy as np
    if Path(cache).exists():
        arr = np.load(cache)
        if len(arr) >= target_tokens:
            print(f"  corpus cache hit: {len(arr):,} tokens ({cache})", flush=True)
            return arr
    from datasets import load_dataset
    print(f"  downloading ~{target_tokens//1_000_000}M tokens once (then offline)...", flush=True)
    toks = []
    for attempt in range(6):
        try:
            ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
            for ex in ds:
                t = ex.get("text") or ""
                if t:
                    toks.extend(tok(t)["input_ids"] + [tok.eos_token_id])
                if len(toks) >= target_tokens:
                    break
            if len(toks) >= target_tokens:
                break
        except Exception as e:
            print(f"  download retry {attempt+1}/6 after: {str(e)[:60]}", flush=True); time.sleep(5)
    arr = np.array(toks[:target_tokens], dtype=np.uint16)
    np.save(cache, arr); print(f"  corpus ready: {len(arr):,} tokens -> {cache}", flush=True)
    return arr


def batches(arr, seq_len, batch, device, rng):
    """Yield random fixed-length windows from the in-memory corpus (offline, no network)."""
    n = batch * (seq_len + 1)
    while True:
        start = rng.randint(0, len(arr) - n - 1)
        chunk = arr[start:start + n].astype("int64")
        t = torch.tensor(chunk, dtype=torch.long).view(batch, seq_len + 1).to(device)
        yield t[:, :-1], t[:, 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="checkpoints/cotrain-skiprobust")
    ap.add_argument("--smoke", action="store_true", help="20-step smoke (verify loss drops) before the real run")
    a = ap.parse_args()
    if a.smoke: a.steps = 20
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    rng = random.Random(1337); torch.manual_seed(1337)

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16).to(dev)

    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                      layers_to_transform=ROUTABLE, task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    model.train()
    layers = find_decoder_layers(model)   # robust to peft wrapping
    print(f"  decoder layers located: {len(layers)}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)

    corpus = load_fixed_corpus(tok, target_tokens=max(a.steps * a.batch * (a.seq_len + 1) * 2, 200_000))
    data = batches(corpus, a.seq_len, a.batch, dev, random.Random(7))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    losses = []; t0 = time.time()
    for step in range(a.steps):
        xb, yb = next(data)
        skip = sample_skip(rng)
        with IdentitySkip(layers, skip) if skip else _null():
            out = model(input_ids=xb)
            logits = out.logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); opt.step()
        losses.append(loss.item())
        if step % 20 == 0 or step == a.steps - 1:
            import numpy as np
            print(f"  step {step:4d}  loss {loss.item():.3f}  k={len(skip)}  ({(time.time()-t0)/(step+1):.2f}s/step)", flush=True)
        if step > 0 and step % 500 == 0:
            model.save_pretrained(a.out); print(f"  [checkpoint saved @ step {step}]", flush=True)
    import numpy as np
    init, final = float(np.mean(losses[:5])), float(np.mean(losses[-5:]))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)

    # ---- SELF-PROOF (so the verifier can confirm the adapters are REAL + ACTIVE, not a no-op) ----
    model.eval()
    probe = tok("def add(a: int, b: int) -> int:", return_tensors="pt").to(dev)
    with torch.no_grad():
        lora_on = model(**probe).logits.float().cpu()
        with model.disable_adapter():          # peft: run the FROZEN base
            lora_off = model(**probe).logits.float().cpu()
    max_diff = float((lora_on - lora_off).abs().max())
    lora_norm = float(sum(p.detach().float().norm()**2 for n, p in model.named_parameters() if "lora" in n.lower() and p.requires_grad) ** 0.5)
    proof = {"frozen_vs_cotrained_max_logit_diff": max_diff, "lora_weight_norm": lora_norm,
             "adapters_active": max_diff > 0.5 and lora_norm > 0.0}
    (Path(a.out)/"cotrain_result.json").write_text(json.dumps({
        "model": {"repo": MODEL, "rev": REV}, "steps": a.steps, "seq_len": a.seq_len, "batch": a.batch,
        "init_loss": init, "final_loss": final, "drop": init-final, "skip_dist": SKIP_DIST, "smoke": a.smoke,
        "self_proof": proof,
        "verdict": "loss drops under skip distribution" if init-final > 0.3 else "CHECK: loss not dropping"}, indent=2))
    print(f"\n  init {init:.3f} -> final {final:.3f} (drop {init-final:.3f})")
    print(f"  SELF-PROOF: frozen-vs-cotrained logit diff {max_diff:.2f} (>0.5 => adapters active), lora_norm {lora_norm:.3f}")
    print(f"  adapters -> {a.out}/  (bring back the WHOLE folder + cotrain_result.json for verification)")


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
