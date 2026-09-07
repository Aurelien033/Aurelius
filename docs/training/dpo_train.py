#!/usr/bin/env python3
"""dpo_train.py — Direct Preference Optimization on CLEAN {prompt, chosen, rejected} pairs (release Stage 2).

Runs after sft_train.py. Sharpens the SFT model toward the *chosen* response and away from *rejected* using
DPO (Rafailov 2023). The reference policy is the same model with LoRA adapters DISABLED (no second copy in
memory) — so the SFT'd-then-DPO'd LoRA is trained against the frozen SFT base. Data: the v20 preferences.jsonl
(16,749 clean deterministic-synthetic pairs, no external-model outputs). Loss masked to response tokens.

Run (Colab Pro):
  python docs/training/dpo_train.py --base WeiboAI/VibeThinker-3B --adapter checkpoints/vibethinker-sft \
     --data data/aurelius_reasoning_sft_v20/preferences.jsonl --out checkpoints/vibethinker-dpo
"""
import argparse, json, sys, random
from pathlib import Path
import torch, torch.nn.functional as F
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sft_train import chat_encode   # reuse the exact chat-template + loss-mask encoder


def resp_logprob(model, ids, labels, dev):
    """Sum log-prob of the RESPONSE tokens (labels!=-100) under `model`. Causal shift: logits[t] predicts t+1."""
    out = model(input_ids=ids).logits[:, :-1, :].float()          # [1, T-1, V]
    tgt = labels[:, 1:]                                            # tokens at positions 1..T-1
    mask = (tgt != -100)
    lp = F.log_softmax(out, -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(-1)                                     # [1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="WeiboAI/VibeThinker-3B")
    ap.add_argument("--adapter", default=None, help="SFT LoRA dir to start from (recommended); else fresh LoRA on base")
    ap.add_argument("--data", required=True, help="jsonl with {prompt, chosen, rejected}")
    ap.add_argument("--system", default=None, help="optional system prompt (Aurelius persona)")
    ap.add_argument("--n", type=int, default=0, help="cap #pairs (0=all)")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--rank", type=int, default=32); ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-6); ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--out", default="checkpoints/dpo")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    DT = torch.float16 if a.dtype == "fp16" else torch.bfloat16
    rng = random.Random(1337); torch.manual_seed(1337)

    pairs = []
    for l in open(a.data):
        d = json.loads(l)
        if d["chosen"].strip() and d["rejected"].strip() and d["chosen"] != d["rejected"]:
            pairs.append((d["prompt"], d["chosen"], d["rejected"]))
    if a.n: pairs = pairs[:a.n]
    if a.smoke: pairs = pairs[:6]
    if not pairs: print("  NO usable pairs (chosen==rejected everywhere?)"); return

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=DT).to(dev)
    if a.adapter:                                                  # continue from the SFT adapter, made trainable
        model = PeftModel.from_pretrained(model, a.adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
    model.train()
    print(f"DPO {a.base}{' +'+a.adapter if a.adapter else ''} on {dev}/{a.dtype}: {len(pairs)} pairs, beta {a.beta}, lr {a.lr}", flush=True)

    def enc(prompt, resp):
        fid, lab = chat_encode(tok, prompt, resp, a.max_len, a.system)
        return torch.tensor([fid]).to(dev), torch.tensor([lab]).to(dev)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    steps = max(1, int(len(pairs) * a.epochs)); order = list(range(len(pairs))); pos = 0
    n_acc = 0
    for step in range(steps):
        if pos >= len(order): rng.shuffle(order); pos = 0
        p, ch, rj = pairs[order[pos]]; pos += 1
        ic, lc = enc(p, ch); ir, lr_ = enc(p, rj)
        pol_c, pol_r = resp_logprob(model, ic, lc, dev), resp_logprob(model, ir, lr_, dev)
        with torch.no_grad():
            with model.disable_adapter():                         # reference = SFT/base policy (adapters off)
                ref_c, ref_r = resp_logprob(model, ic, lc, dev), resp_logprob(model, ir, lr_, dev)
        logits = a.beta * ((pol_c - ref_c) - (pol_r - ref_r))     # DPO objective
        loss = -F.logsigmoid(logits).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); opt.step()
        n_acc += int((pol_c - pol_r).item() > 0)
        if step % 20 == 0 or step == steps - 1:
            print(f"  step {step:4d}/{steps}  loss {loss.item():.3f}  margin {(pol_c-pol_r).item():+.2f}  "
                  f"chosen>rejected {n_acc}/{step+1}", flush=True)

    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print(f"\n  DPO done. chosen>rejected on {n_acc}/{steps} ({n_acc/steps*100:.0f}%). saved -> {a.out}/", flush=True)
    print("  next: merge LoRA + eval gym pass-rate vs the SFT checkpoint to confirm DPO didn't regress quality.")


if __name__ == "__main__":
    main()
