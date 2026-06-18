#!/usr/bin/env python3
"""make_traces.py — generate VERIFIED reasoning/coding traces from a CLEAN, distillation-permitted teacher.

Release-track + legally clean: NO Claude / closed-model outputs. Runs task prompts (the repair gym, or any
verifiable task set) through a permissively-licensed teacher that explicitly allows distillation — default
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (MIT, DeepSeek released R1 for distillation); alternatives
Qwen/QwQ-32B, Qwen2.5-Coder-32B, CohereLabs/North-Mini-Code-1.0 (Apache). It captures the teacher's
reasoning+answer and KEEPS ONLY traces whose answer PASSES the gym verifier (rejection sampling / STaR-style)
-> clean, verified SFT data that teaches the *decisions and actions*, shippable.

Run (Colab Pro / A100):
  python docs/training/make_traces.py --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
     --split eval_data/selector_split_v0.1.json --samples 2 --max_new 2048 --out data/traces_r1.jsonl
"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "first_light"))
import first_light_runner_v4 as R


def to_input(tok, prompt):
    """Chat-format the prompt if the teacher has a chat template; else raw (base models)."""
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--split", default="eval_data/selector_split_v0.1.json")
    ap.add_argument("--n", type=int, default=0, help="cap #tasks (0=all)")
    ap.add_argument("--samples", type=int, default=1, help="generations/task; keep the first VERIFIED one (rejection sampling)")
    ap.add_argument("--max_new", type=int, default=2048, help="reasoning teachers need room (<think>...</think>+answer)")
    ap.add_argument("--gen_batch", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6, help="R1 recommends ~0.6; 0 = greedy")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--keep", choices=["verified", "all"], default="verified")
    ap.add_argument("--out", default="data/traces.jsonl")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    DT = torch.float16 if a.dtype == "fp16" else torch.bfloat16

    ids = sorted(set(json.loads(Path(a.split).read_text())["all"]))
    idx = {}
    for root, fam in [(R.RESEARCH / "gym-v0.1-FL", "F2_json"), (R.RESEARCH / "gym-v0.3", "F3_type")]:
        for sp in ["dev", "test", "smoke", "train"]:
            for f in (root / fam / sp).glob("*.json"):
                d = json.loads(f.read_text()); idx[d["instance_id"]] = d
    ids = [i for i in ids if i in idx]
    if a.n: ids = ids[:a.n]
    if a.smoke: ids, a.max_new, a.gen_batch = ids[:3], 64, 3

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.teacher)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=DT).to(dev).eval()
    print(f"teacher {a.teacher} on {dev}/{a.dtype}: {len(ids)} tasks x {a.samples} samples (verified rejection sampling)", flush=True)

    def gen(prompts):
        enc = tok([to_input(tok, p) for p in prompts], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        kw = dict(max_new_tokens=a.max_new, pad_token_id=tok.eos_token_id, do_sample=a.temperature > 0)
        if a.temperature > 0: kw.update(temperature=a.temperature, top_p=0.95)
        with torch.no_grad():
            out = model.generate(**enc, **kw)
        L = enc["input_ids"].shape[1]
        return [tok.decode(r[L:], skip_special_tokens=True) for r in out]

    recs, n_ver = [], 0
    for s in range(0, len(ids), a.gen_batch):
        chunk = ids[s:s + a.gen_batch]
        chosen = {i: (None, False) for i in chunk}     # iid -> (response, verified)
        for _ in range(a.samples):
            todo = [i for i in chunk if not chosen[i][1]]      # only re-sample tasks not yet verified
            if not todo: break
            comps = gen([idx[i]["prompt_context"] for i in todo])
            for i, comp in zip(todo, comps):
                passed = bool(R.VERIFIERS[idx[i]["metadata"]["family"]](idx[i], comp))
                if passed or chosen[i][0] is None:             # keep first verified, else latest as fallback
                    chosen[i] = (comp, passed)
        for i in chunk:
            comp, passed = chosen[i]
            if a.keep == "all" or passed:
                recs.append({"instance_id": i, "family": idx[i]["metadata"]["family"], "prompt": idx[i]["prompt_context"],
                             "response": comp, "verified": passed, "teacher": a.teacher})
                n_ver += int(passed)
        print(f"  [{min(s+a.gen_batch, len(ids))}/{len(ids)}] kept {len(recs)} ({n_ver} verified)", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    print(f"saved {len(recs)} traces ({n_ver} verified, {n_ver/max(1,len(ids))*100:.0f}% of tasks) -> {a.out}", flush=True)
    print("  next: sft_train.py --data this.jsonl --base WeiboAI/VibeThinker-3B", flush=True)


if __name__ == "__main__":
    main()
