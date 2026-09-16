#!/usr/bin/env python3
"""continued_pretrain.py — domain-adaptive continued pretraining on CLEAN docs (release Tier 1, optional).

Standard causal-LM training (no loss mask — predict every token) over packed domain documents, to inject
domain knowledge a base lacks BEFORE SFT. Use ONLY if the SFT eval shows a knowledge gap (style gaps are an
SFT/DPO problem, not a pretraining one) — per-dollar, more verified SFT data usually beats this. Data: the
v20 pretrain_docs.jsonl (23,448 clean deterministic-synthetic docs) and/or any {text} jsonl. LoRA by default
(domain adaptation); --full_ft for a real continued-pretrain on a big GPU.

Cost is LINEAR in tokens (~25M tok/A100-hr for a 3B model) — keep the budget modest (1-20B tokens), this is
NOT from-scratch pretraining (the open base already did the trillions).

Run (rented A100 / Colab Pro):
  python docs/training/continued_pretrain.py --base WeiboAI/VibeThinker-3B \
     --data data/aurelius_reasoning_sft_v20/pretrain_docs.jsonl --steps 2000 --out checkpoints/vibethinker-cpt
"""
import argparse, json, sys, random
from pathlib import Path
import numpy as np, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="WeiboAI/VibeThinker-3B")
    ap.add_argument("--data", required=True, help="comma-sep jsonl(s); uses the 'text' field (or 'response'/'prompt')")
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--full_ft", action="store_true", help="full continued-pretrain (big GPU); default = LoRA")
    ap.add_argument("--rank", type=int, default=32); ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--seq", type=int, default=1024); ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--out", default="checkpoints/cpt")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    DT = torch.float16 if a.dtype == "fp16" else torch.bfloat16
    rng = random.Random(1337); torch.manual_seed(1337)
    if a.smoke: a.steps, a.bs, a.seq = 8, 2, 64

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # ---- pack all docs into one token stream, then chunk into seq-len blocks (standard pretraining packing) ----
    stream = []
    eos = tok.eos_token_id
    for p in a.data.split(","):
        for l in open(p):
            d = json.loads(l)
            txt = d.get(a.text_key) or d.get("response") or d.get("prompt") or ""
            if txt.strip():
                stream.extend(tok(txt, add_special_tokens=False)["input_ids"] + [eos])
            if a.smoke and len(stream) > a.bs * (a.seq + 1) * 4: break
    arr = np.array(stream, dtype=np.int64)
    if len(arr) < a.bs * (a.seq + 1) + 1: print(f"  too few tokens ({len(arr)})"); return
    print(f"continued-pretrain {a.base} on {dev}/{a.dtype}: {len(arr)/1e6:.1f}M tokens, "
          f"{'FULL' if a.full_ft else 'LoRA r%d' % a.rank}, {a.steps} steps x bs{a.bs} x seq{a.seq}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=DT).to(dev)
    if not a.full_ft:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
    model.train()

    def batch():
        n = a.bs * (a.seq + 1)
        s = rng.randint(0, len(arr) - n - 1)
        t = torch.tensor(arr[s:s + n]).view(a.bs, a.seq + 1).to(dev)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95))
    for step in range(a.steps):
        ids, lab = batch()
        loss = model(input_ids=ids, labels=lab).loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); opt.step()
        if step % 50 == 0 or step == a.steps - 1:
            print(f"  step {step:5d}/{a.steps}  loss {loss.item():.3f}  ppl {torch.exp(loss).item():.1f}", flush=True)

    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print(f"\n  saved -> {a.out}/  ({'merge then ' if not a.full_ft else ''}SFT on top next). "
          "Sanity-check: ppl should drop; if it spikes, lower lr.", flush=True)


if __name__ == "__main__":
    main()
