#!/usr/bin/env python3
"""diagnose_sft.py — why did SFT change the gym pass-rate? Show BASE vs SFT completions side by side.

Loads base + the SFT adapter as ONE model (PeftModel + disable_adapter to toggle base/SFT — memory-cheap),
generates on a few held-out gym tasks, and prints the prompt, both completions, and whether each VERIFIES.
Distinguishes (A) genuine degradation / wrong format from (B) eval truncation: if SFT output is long and cut
off before the answer, raise --max_new; if it's a verbose template with no answer block, it's a data/dose issue.

Run (Colab):
  python docs/training/diagnose_sft.py --base WeiboAI/VibeThinker-3B --adapter /content/vibethinker-sft --n 4
"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "first_light"))
import first_light_runner_v4 as R


def gen(tok, model, prompt, dev, max_new):
    msg = (tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
           if getattr(tok, "chat_template", None) else prompt)
    enc = tok(msg, return_tensors="pt", add_special_tokens=False).to(dev)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="WeiboAI/VibeThinker-3B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--eval_split", default="eval_data/selector_split_v0.1.json")
    ap.add_argument("--exclude", default="/content/traces_r1.jsonl", help="held-out: skip traced ids")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--show", type=int, default=600, help="chars of each completion to print")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    idx = {}
    for root, fam in [(R.RESEARCH / "gym-v0.1-FL", "F2_json"), (R.RESEARCH / "gym-v0.3", "F3_type")]:
        for sp in ["dev", "test", "smoke", "train"]:
            for f in (root / fam / sp).glob("*.json"):
                d = json.loads(f.read_text()); idx[d["instance_id"]] = d
    excl = {json.loads(l).get("instance_id") for l in open(a.exclude)} if Path(a.exclude).exists() else set()
    ids = [i for i in json.loads(Path(a.eval_split).read_text())["all"] if i in idx and i not in excl][:a.n]

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).to(dev).eval()
    model = PeftModel.from_pretrained(base, a.adapter).eval()      # SFT = adapter on; disable_adapter() = base

    for i in ids:
        inst = idx[i]; fam = inst["metadata"]["family"]; vf = R.VERIFIERS[fam]
        with model.disable_adapter():
            b = gen(tok, model, inst["prompt_context"], dev, a.max_new)
        s = gen(tok, model, inst["prompt_context"], dev, a.max_new)
        bv, sv = bool(vf(inst, b)), bool(vf(inst, s))
        print("\n" + "=" * 90)
        print(f"TASK {i[-16:]}  family={fam}")
        print(f"PROMPT (tail): ...{inst['prompt_context'][-260:]}")
        print(f"\n--- BASE  [verify={bv}]  (len {len(b)} chars) ---\n{b[:a.show]}")
        print(f"\n--- SFT   [verify={sv}]  (len {len(s)} chars) ---\n{s[:a.show]}")
    print("\n" + "=" * 90)
    print("READ: SFT long & cut off before the answer => raise --eval_max_new (eval artifact).")
    print("      SFT verbose template w/ no answer block => data/dose issue (lower lr/epochs, fewer v20 rows, more traces).")


if __name__ == "__main__":
    main()
