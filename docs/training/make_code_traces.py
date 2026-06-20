#!/usr/bin/env python3
"""make_code_traces.py — VERIFIED general-coding traces from a clean teacher (MBPP rejection sampling).

Broadens the SFT data beyond the narrow F2/F3 repair gym: generates solutions to MBPP problems with a clean
teacher (DeepSeek-R1-Distill, MIT), EXECUTES each against MBPP's real unit tests, and keeps only the ones that
PASS. MBPP is disjoint from HumanEval, so training on these + evaluating on HumanEval is a clean held-out test
of whether broader verified-coding SFT improves GENERAL code (the thing the gym SFT didn't transfer to).
Response stored = the verified CODE only (no teacher reasoning) so it matches a no-think eval and avoids the
Qwen3 thinking-format conflict.

  python docs/training/make_code_traces.py --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
     --samples 2 --out /content/mbpp_traces.jsonl
"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_code_bench import extract_code, run_program, build_msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--n", type=int, default=0, help="cap #problems (0=all ~974)")
    ap.add_argument("--samples", type=int, default=2, help="tries/problem; keep first that PASSES the tests")
    ap.add_argument("--max_new", type=int, default=2048, help="reasoning teacher needs room")
    ap.add_argument("--gen_batch", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--timeout", type=int, default=12)
    ap.add_argument("--out", default="data/mbpp_traces.jsonl")
    ap.add_argument("--load_4bit", action="store_true", help="4-bit NF4 load -> fits a 32B teacher on one A100-40GB")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", split="train+validation+test")   # 964 problems
    probs = []
    for d in ds:
        ask = (f"Write a Python function for this task. Return ONLY the function in one ```python code block.\n\n"
               f"Task: {d['text']}\nIt must satisfy:\n{d['test_list'][0]}")
        probs.append((ask, (d.get("test_setup_code") or ""), "\n".join(d["test_list"])))
    if a.n: probs = probs[:a.n]
    if a.smoke: probs, a.max_new, a.gen_batch = probs[:3], 256, 3

    from transformers import AutoTokenizer
    from lm_load import load_causal_lm                            # robust: CausalLM or VLM (Qwen3.6-27B), +4bit
    tok = AutoTokenizer.from_pretrained(a.teacher, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model, used = load_causal_lm(a.teacher, a.load_4bit, dev)
    print(f"  loaded {a.teacher} via {used}", flush=True)
    print(f"teacher {a.teacher} on {dev}: {len(probs)} MBPP problems x {a.samples} samples (verified)", flush=True)

    def gen(asks):
        enc = tok([build_msg(tok, x, think=1) for x in asks], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        kw = dict(max_new_tokens=a.max_new, pad_token_id=tok.eos_token_id, do_sample=a.temperature > 0)
        if a.temperature > 0: kw.update(temperature=a.temperature, top_p=0.95)
        with torch.no_grad():
            out = model.generate(**enc, **kw)
        L = enc["input_ids"].shape[1]
        return [tok.decode(r[L:], skip_special_tokens=True) for r in out]

    recs = []
    for s in range(0, len(probs), a.gen_batch):
        chunk = probs[s:s + a.gen_batch]
        chosen = {i: None for i in range(len(chunk))}
        for _ in range(a.samples):
            todo = [i for i in range(len(chunk)) if chosen[i] is None]
            if not todo: break
            outs = gen([chunk[i][0] for i in todo])
            for i, o in zip(todo, outs):
                code = extract_code(o)
                setup, tests = chunk[i][1], chunk[i][2]
                if code.strip() and run_program(setup + "\n" + code + "\n" + tests + "\n", a.timeout):
                    chosen[i] = code
        for i in range(len(chunk)):
            if chosen[i]:
                recs.append({"prompt": chunk[i][0], "response": f"```python\n{chosen[i]}\n```",
                             "verified": True, "source": "mbpp", "teacher": a.teacher})
        print(f"  [{min(s+a.gen_batch, len(probs))}/{len(probs)}] kept {len(recs)} verified", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    print(f"saved {len(recs)} verified MBPP traces ({len(recs)/max(1,len(probs))*100:.0f}% of problems) -> {a.out}", flush=True)
    print("  next: sft_train.py --base Qwen/Qwen3-8B --data this.jsonl --skip_eval ; then eval_code_bench humaneval", flush=True)


if __name__ == "__main__":
    main()
