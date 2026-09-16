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


def run_asserts_fraction(setup, code, test_list, timeout=12, max_tests=10):
    """VeRPO dense signal for assert-style benches (MBPP): run each test separately -> (n_passed, n_total)."""
    tests = list(test_list)[:max_tests]
    if not tests: return 0, 0
    npass = sum(int(run_program((setup or "") + "\n" + code + "\n" + t + "\n", timeout)) for t in tests)
    return npass, len(tests)


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


def run_io_fraction(code, inputs, outputs, timeout=8, max_cases=10):
    """VeRPO dense signal for stdin/stdout benches (code_contests): -> (n_passed, n_total)."""
    cases = list(zip(inputs, outputs))[:max_cases]
    if not cases: return 0, 0
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); path = f.name
    try:
        npass = 0
        for inp, exp in cases:
            try:
                r = subprocess.run([sys.executable, path], input=str(inp), capture_output=True, text=True, timeout=timeout)
                npass += int(_norm_io(r.stdout) == _norm_io(exp))
            except subprocess.TimeoutExpired:
                pass
        return npass, len(cases)
    finally:
        os.unlink(path)


def humaneval_items(n):
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    items = []
    for d in ds:
        items.append(dict(
            tid=d["task_id"],                                         # stable id for cross-model flip alignment
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
            tid=f"mbpp/{d['task_id']}",                               # stable id for cross-model flip alignment
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
    ap.add_argument("--load_4bit", action="store_true", help="4-bit NF4 base -> fits Qwen3-8B inference on a 16GB T4 (keeps LoRA attached, no merge)")
    ap.add_argument("--dump", default="", help="write per-problem pass/fail (tid, greedy, oracle) to this JSON -> feed two of them to flip_analysis.py")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    items = humaneval_items(a.n) if a.bench == "humaneval" else mbpp_items(a.n)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    bnb = None
    if a.load_4bit:                                                   # T4 has no bf16 -> fp16 compute; device_map places weights (no .to(dev))
        from transformers import BitsAndBytesConfig
        cdt = torch.bfloat16 if (dev == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=cdt, bnb_4bit_use_double_quant=True)
    def _load(mid):
        if bnb is not None:
            return AutoModelForCausalLM.from_pretrained(mid, quantization_config=bnb, device_map={"": 0})
        return AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to(dev)
    acfg = Path(a.model) / "adapter_config.json"
    if acfg.exists():                                                 # a LoRA adapter dir -> load base + adapter
        from peft import PeftModel
        base_id = json.loads(acfg.read_text())["base_model_name_or_path"]
        m = PeftModel.from_pretrained(_load(base_id), a.model)
        model = (m if bnb is not None else m.merge_and_unload()).eval()   # can't merge into a 4-bit base -> keep adapter attached
    else:
        model = _load(a.model).eval()
    print(f"{a.bench} pass@1: {a.model} on {dev}{' (4bit)' if bnb else ''}, {len(items)} problems", flush=True)

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

    oracle = None
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

    if a.dump:                                                        # per-problem dump -> flip_analysis.py
        recs = [{"tid": it.get("tid", i), "greedy": bool(greedy[i]),
                 "oracle": (bool(oracle[i]) if oracle is not None else None)}
                for i, it in enumerate(items)]
        Path(a.dump).write_text(json.dumps(
            {"model": a.model, "bench": a.bench, "n": len(items), "think": a.think,
             "passk": a.passk, "pass1": npass, "results": recs}, indent=1))
        print(f"  dumped per-problem results -> {a.dump}  ({len(recs)} problems)", flush=True)


if __name__ == "__main__":
    main()
