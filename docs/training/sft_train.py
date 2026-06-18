#!/usr/bin/env python3
"""sft_train.py — gentle LoRA SFT of a CLEAN base on VERIFIED traces (release-track).

Instruction-tunes the base (default WeiboAI/VibeThinker-3B, MIT — clean, NOT Claude-distilled) on
{prompt -> verified response} traces (from make_traces.py and/or the existing verified-reasoning SFT). LoRA
+ low LR + few epochs ADD task/coding/verification behavior WITHOUT eroding the base's reasoning (avoid the
co-train's catastrophic forgetting). Loss is masked to RESPONSE tokens only. Evals the gym pass-rate
BEFORE and AFTER so you see whether the SFT actually helped. Colab Pro: bf16 (native-fast on A100/L4);
--full_ft for a full fine-tune on a big GPU, else LoRA fits 24GB.

Run (Colab Pro):
  python docs/training/sft_train.py --base WeiboAI/VibeThinker-3B --data data/traces_r1.jsonl \
     --epochs 2 --lr 1e-4 --out checkpoints/vibethinker-sft
"""
import argparse, json, sys, random
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "first_light"))
import first_light_runner_v4 as R


def chat_encode(tok, prompt, response, max_len, system=None):
    """Tokenize (system+)prompt+response with the chat template; labels = -100 on prompt, response tokens kept."""
    sys_msg = [{"role": "system", "content": system}] if system else []
    if getattr(tok, "chat_template", None):
        full = tok.apply_chat_template(sys_msg + [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}], tokenize=False)
        pre = tok.apply_chat_template(sys_msg + [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    else:
        pre = (system + "\n\n" if system else "") + prompt
        full = pre + response
    fid = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    plen = min(len(tok(pre, add_special_tokens=False)["input_ids"]), len(fid))
    labels = [-100] * plen + fid[plen:]
    return fid, labels


def gym_passrate(tok, model, idx, ids, dev, max_new=256):
    model.eval(); n = 0
    for s in range(0, len(ids), 8):
        chunk = ids[s:s+8]
        msgs = [(tok.apply_chat_template([{"role": "user", "content": idx[i]["prompt_context"]}], tokenize=False, add_generation_prompt=True)
                 if getattr(tok, "chat_template", None) else idx[i]["prompt_context"]) for i in chunk]
        enc = tok(msgs, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        L = enc["input_ids"].shape[1]
        for i, r in zip(chunk, out):
            comp = tok.decode(r[L:], skip_special_tokens=True)
            n += int(bool(R.VERIFIERS[idx[i]["metadata"]["family"]](idx[i], comp)))
    model.train()
    return n / max(1, len(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="WeiboAI/VibeThinker-3B")
    ap.add_argument("--data", required=True, help="comma-separated jsonl(s) with {prompt,response,verified}")
    ap.add_argument("--verified_only", type=int, default=1, help="use only verified=True traces (the good ones)")
    ap.add_argument("--eval_split", default="eval_data/selector_split_v0.1.json")
    ap.add_argument("--n_eval", type=int, default=48)
    ap.add_argument("--full_ft", action="store_true", help="full fine-tune (big GPU); default = LoRA")
    ap.add_argument("--rank", type=int, default=32); ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--bs", type=int, default=2); ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--out", default="checkpoints/sft")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    DT = torch.float16 if a.dtype == "fp16" else torch.bfloat16
    rng = random.Random(1337); torch.manual_seed(1337)

    # ---- data ----
    rows = []
    for p in a.data.split(","):
        for l in open(p):
            d = json.loads(l)
            if not d.get("response", "").strip(): continue
            if a.verified_only and not d.get("verified", True): continue
            rows.append((d["prompt"], d["response"], d.get("system_prompt")))   # v20 carries the Aurelius persona
    if a.smoke: rows = rows[:8]
    if not rows: print("  NO training rows (check --verified_only / --data)"); return

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # generation eval needs left pad; training builds its own padded tensors
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=DT).to(dev)
    if not a.full_ft:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
    model.train()

    enc = [chat_encode(tok, p, r, a.max_len, sysp) for p, r, sysp in rows]
    enc = [e for e in enc if len(e[0]) > len([x for x in e[1] if x == -100])]   # has >=1 response token
    print(f"SFT {a.base} on {dev}/{a.dtype}: {len(enc)} examples, {'LoRA r%d' % a.rank if not a.full_ft else 'FULL'}, lr {a.lr}, {a.epochs} ep", flush=True)

    # ---- eval split (held out from SFT data) ----
    eids = sorted(set(json.loads(Path(a.eval_split).read_text())["all"]))
    idx = {}
    for root, fam in [(R.RESEARCH / "gym-v0.1-FL", "F2_json"), (R.RESEARCH / "gym-v0.3", "F3_type")]:
        for sp in ["dev", "test", "smoke", "train"]:
            for f in (root / fam / sp).glob("*.json"):
                d = json.loads(f.read_text()); idx[d["instance_id"]] = d
    eids = [i for i in eids if i in idx][:(4 if a.smoke else a.n_eval)]
    before = gym_passrate(tok, model, idx, eids, dev, max_new=64 if a.smoke else 256)
    print(f"  gym pass-rate BEFORE: {before*100:.1f}% (n={len(eids)})", flush=True)

    # ---- train ----
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = max(1, int(len(enc) * a.epochs / a.bs)); order = list(range(len(enc)))
    pos = 0
    for step in range(steps):
        if pos + a.bs > len(order): rng.shuffle(order); pos = 0
        batch = [enc[order[pos + j]] for j in range(a.bs)]; pos += a.bs
        ml = max(len(x[0]) for x in batch)
        ids = torch.full((len(batch), ml), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(batch), ml), -100, dtype=torch.long)
        for bi, (fid, l) in enumerate(batch):
            ids[bi, :len(fid)] = torch.tensor(fid); lab[bi, :len(l)] = torch.tensor(l)
        ids, lab = ids.to(dev), lab.to(dev); am = (ids != tok.pad_token_id).long()
        loss = model(input_ids=ids, attention_mask=am, labels=lab).loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); opt.step()
        if step % 20 == 0 or step == steps - 1:
            print(f"  step {step:4d}/{steps}  loss {loss.item():.3f}", flush=True)

    after = gym_passrate(tok, model, idx, eids, dev, max_new=64 if a.smoke else 256)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print(f"\n  gym pass-rate AFTER: {after*100:.1f}%  (before {before*100:.1f}%, Δ {(after-before)*100:+.1f}pp)")
    print(f"  saved -> {a.out}/  (LoRA adapters; merge for release)" if not a.full_ft else f"  saved -> {a.out}/")
    print("  Δ>0 => the verified-trace SFT specialized the base toward the task. If Δ<=0, revisit data/lr/epochs.")


if __name__ == "__main__":
    main()
