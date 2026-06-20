#!/usr/bin/env python3
"""eval_code_bench.py — pass@1 on HumanEval / MBPP (breadth check: does the gym SFT generalize?).

Generates a completion per problem from --model, extracts the code block, EXECUTES it against the benchmark's
unit tests in an isolated subprocess (with timeout), and reports pass@1. Run it on the BASE and on the SFT'd
model to see if the held-out gym gain (48->60%) shows up on standard code benchmarks too. If yes, the model
generalizes; if it's gym-only, you've got a specialist.

  python docs/training/eval_code_bench.py --model Qwen/Qwen3-8B            --bench humaneval   # base
  python docs/training/eval_code_bench.py --model /content/aurelius-v1-8b  --bench humaneval   # SFT'd

SECURITY: executes model-generated code. Run only in a disposable env (Colab VM / sandbox).
"""
import argparse, json, re, sys, subprocess, tempfile, os
from pathlib import Path
import torch


def extract_code(text):
    """Drop any <think>…</think>, then take the last ```python block; else raw text."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)   # strip reasoning if present
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)          # also handle an unclosed/leading think
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1]
    return text.strip()


def build_msg(tok, ask, think):
    """Chat-format; disable thinking for code pass@1 (Qwen3 otherwise spends the budget on <think>)."""
    if not getattr(tok, "chat_template", None):
        return ask
    kw = {} if think else {"enable_thinking": False}
    try:
        return tok.apply_chat_template([{"role": "user", "content": ask}], tokenize=False, add_generation_prompt=True, **kw)
    except TypeError:                                                  # template doesn't accept enable_thinking
        return tok.apply_chat_template([{"role": "user", "content": ask}], tokenize=False, add_generation_prompt=True)


def run_program(src, timeout=12):
    """Exec a self-contained program in a fresh subprocess. returncode 0 (no assert/err) == pass."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


def run_with_feedback(src, timeout=12):
    """Like run_program but returns (passed, feedback): on failure, the stderr/stdout tail for repair prompting."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0: return True, ""
        return False, (r.stderr or r.stdout or "non-zero exit").strip()[-800:]
    except subprocess.TimeoutExpired:
        return False, "timeout (likely an infinite loop)"
    finally:
        os.unlink(path)


def _norm_io(s):
    """Whitespace-lenient stdout comparison (trailing spaces / blank trailing lines vary harmlessly)."""
    if isinstance(s, list): s = "\n".join(map(str, s))
    return "\n".join(line.rstrip() for line in str(s).strip().splitlines())


def run_io_tests(code, inputs, outputs, timeout=8, max_cases=12):
    """Competitive-programming reward: run `code` as a script per (stdin, expected stdout); pass iff ALL match."""
    if not inputs or not outputs: return False
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); path = f.name
    try:
        for inp, exp in list(zip(inputs, outputs))[:max_cases]:
            try:
                r = subprocess.run([sys.executable, path], input=str(inp), capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                return False
            if _norm_io(r.stdout) != _norm_io(exp):
                return False
        return True
    finally:
        os.unlink(path)


def humaneval_items(n):
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    items = []
    for d in ds:
        items.append(dict(
            ask=f"Complete this Python function. Return ONLY the complete function in one ```python code block.\n\n{d['prompt']}",
            check=lambda code, d=d: code + "\n" + d["test"] + f"\ncheck({d['entry_point']})\n"))
    return items[:n] if n else items


def mbpp_items(n):
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", split="test")
    items = []
    for d in ds:
        tests = "\n".join(d["test_list"])
        items.append(dict(
            ask=f"Write a Python function for this task. Return ONLY the function in one ```python code block.\n\nTask: {d['text']}\nIt must satisfy:\n{d['test_list'][0]}",
            check=lambda code, tests=tests, d=d: d.get("test_setup_code", "") + "\n" + code + "\n" + tests + "\n"))
    return items[:n] if n else items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local path (base or SFT'd)")
    ap.add_argument("--bench", choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--n", type=int, default=0, help="cap #problems (0=all; HumanEval=164, MBPP=500)")
    ap.add_argument("--max_new", type=int, default=1024, help="thinking models need room")
    ap.add_argument("--gen_batch", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=12)
    ap.add_argument("--think", type=int, default=0, help="1=allow Qwen3 <think> (needs big --max_new); 0=direct code")
    ap.add_argument("--passk", type=int, default=0, help=">0: also sample K/problem -> oracle@K + selection_gap (SELECTION vs CAPABILITY diagnostic)")
    ap.add_argument("--passk_temp", type=float, default=0.8)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    items = humaneval_items(a.n) if a.bench == "humaneval" else mbpp_items(a.n)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    acfg = Path(a.model) / "adapter_config.json"
    if acfg.exists():                                                 # a LoRA adapter dir -> load base + merge
        from peft import PeftModel
        base_id = json.loads(acfg.read_text())["base_model_name_or_path"]
        base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16).to(dev)
        model = PeftModel.from_pretrained(base, a.model).merge_and_unload().eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    print(f"{a.bench} pass@1: {a.model} on {dev}, {len(items)} problems", flush=True)

    greedy = [False] * len(items)                                     # per-problem greedy pass (the failure ledger)
    for s in range(0, len(items), a.gen_batch):
        chunk = items[s:s + a.gen_batch]
        msgs = [build_msg(tok, it["ask"], a.think) for it in chunk]
        enc = tok(msgs, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.eos_token_id, use_cache=True)
        L = enc["input_ids"].shape[1]
        for j, (it, r) in enumerate(zip(chunk, out)):
            code = extract_code(tok.decode(r[L:], skip_special_tokens=True))
            greedy[s + j] = bool(run_program(it["check"](code), a.timeout))
        print(f"  [{min(s+a.gen_batch, len(items))}/{len(items)}] pass@1 so far: {sum(greedy)}", flush=True)
    npass = sum(greedy)
    print(f"\n=== {a.bench} pass@1 ({a.model}): {npass}/{len(items)} = {npass/len(items)*100:.1f}% ===", flush=True)

    if a.passk:                                                       # CAPABILITY-vs-SELECTION diagnostic
        oracle = list(greedy)                                         # oracle@K = greedy OR any of K samples passes
        for i, it in enumerate(items):
            if oracle[i]:                                             # already solved greedily -> skip sampling
                continue
            enc = tok(build_msg(tok, it["ask"], a.think), return_tensors="pt", add_special_tokens=False).to(dev)
            with torch.no_grad():
                out = model.generate(**enc, num_return_sequences=a.passk, do_sample=True, temperature=a.passk_temp,
                                     top_p=0.95, max_new_tokens=a.max_new, pad_token_id=tok.eos_token_id, use_cache=True)
            L = enc["input_ids"].shape[1]
            for r in out:
                code = extract_code(tok.decode(r[L:], skip_special_tokens=True))
                if run_program(it["check"](code), a.timeout):
                    oracle[i] = True; break
            if (i + 1) % 20 == 0: print(f"  passk [{i+1}/{len(items)}] oracle so far: {sum(oracle)}", flush=True)
        n_oracle = sum(oracle); gap = n_oracle - npass
        recovered = [i for i in range(len(items)) if oracle[i] and not greedy[i]]
        print(f"\n=== {a.bench} CAPABILITY MAP ({a.model}) ===")
        print(f"  pass@1(greedy) {npass}/{len(items)} = {npass/len(items)*100:.1f}%")
        print(f"  oracle@{a.passk}     {n_oracle}/{len(items)} = {n_oracle/len(items)*100:.1f}%  (T={a.passk_temp})")
        print(f"  selection_gap  +{gap}  ({len(recovered)} greedy-fails recovered by sampling)")
        print(f"  READ: big gap => SELECTION problem (best-of-N / reranker / distill from winners — cheap).")
        print(f"        small gap => CAPABILITY ceiling (bigger base / richer RL curriculum — expensive).")


if __name__ == "__main__":
    main()
