"""Verification of the First Light readiness spike (2026-06-11).

Phase A: reproduce the receipt's headline — Qwen2.5-1.5B-Instruct, seed 42, 30 MBPP
         instances, chat pipeline with sig-in-prompt + markdown extraction. Expect 7/30.
Phase B: fair re-test of Qwen2.5-0.5B-Instruct with the SAME fixed pipeline
         (receipt eliminated it without the extraction fix — rule-order check).
Phase C: fair re-test of OLMo-2-0425-1B (base) with a completion-style fixed pipeline
         (sig in prompt, sig prepended to scored completion, top-level truncation).
ADR-1b order: OLMo -> Qwen 0.5B -> Qwen 1.5B. First in-band [20%,80%] wins the pin.
"""

import gc
import json
import random
import re
import sys

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/Users/christienantonio/aurelius")
from src.eval.mbpp_scorer import MBPPProblem, score_single  # noqa: E402

SEED = 42
tbl = pq.read_table("/tmp/mbpp_test.parquet")  # noqa: S108  # nosec B108 — lab spike script, fixed path
all_ids = [tbl.column("task_id")[i].as_py() for i in range(tbl.num_rows)]
random.seed(SEED)
SAMPLED = sorted(random.sample(all_ids, 30))
print("task_ids:", SAMPLED, flush=True)


def get_row(tid):
    idx = all_ids.index(tid)
    return {c: tbl.column(c)[idx].as_py() for c in tbl.column_names}


def extract_code_block(completion):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", completion, re.DOTALL)
    return m.group(1).strip() if m else completion.strip()


def truncate_completion(text):
    # cut base-model continuations at the first new top-level statement
    for pat in ["\ndef ", "\nassert", "\nprint(", "\nclass ", "\nif __name__", "\n#"]:
        i = text.find(pat)
        if i != -1:
            text = text[:i]
    return text


def run_model(repo_id, revision, mode):
    tok = AutoTokenizer.from_pretrained(repo_id, revision=revision)
    model = (
        AutoModelForCausalLM.from_pretrained(
            repo_id, revision=revision, dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        .to("mps")
        .eval()
    )
    passed, details = 0, []
    for tid in SAMPLED:
        row = get_row(tid)
        prob = MBPPProblem(
            task_id=tid,
            text=row["text"],
            code=row["code"],
            test_list=list(row["test_list"]),
            test_setup_code=row.get("test_setup_code", "") or "",
        )
        sig_m = re.match(r"(def \w+\([^)]*\))", row["code"])
        sig = sig_m.group(1) if sig_m else None
        if mode == "chat":
            content = f"{row['text']}\n\n{sig}:" if sig else row["text"]
            text_input = tok.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
            )
        else:  # base completion
            text_input = f"{row['text']}\n\n{sig}:" if sig else row["text"]
        inp = tok(text_input, return_tensors="pt").to("mps")
        with torch.no_grad():
            gen = model.generate(
                **inp, max_new_tokens=512, do_sample=False, pad_token_id=tok.eos_token_id
            )
        raw = tok.decode(gen[0], skip_special_tokens=True)[len(text_input) :]
        if mode == "chat":
            completion = extract_code_block(raw)
        else:
            completion = (sig + ":\n" if sig else "") + truncate_completion(raw)
        res = score_single(prob, completion)
        passed += int(res.passed)
        details.append((tid, res.passed))
        print(
            f"  {repo_id.split('/')[-1]} task {tid}: {'PASS' if res.passed else 'fail'}", flush=True
        )
    del model
    gc.collect()
    torch.mps.empty_cache()
    return passed, details


RESULTS = {}
for repo, rev, mode in [
    ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", "chat"),
    ("Qwen/Qwen2.5-0.5B-Instruct", None, "chat"),
    ("allenai/OLMo-2-0425-1B", "a1847dff35000b4271fa70afc5db10fd29fedbdf", "base"),
]:
    print(f"=== {repo} ({mode}) ===", flush=True)
    p, d = run_model(repo, rev, mode)
    RESULTS[repo] = {"passed": p, "rate": p / 30, "details": d}
    print(
        f"RESULT {repo}: {p}/30 = {p / 30 * 100:.1f}%  in_band={0.20 <= p / 30 <= 0.80}", flush=True
    )

print(
    json.dumps(
        {k: {"passed": v["passed"], "rate": v["rate"]} for k, v in RESULTS.items()}, indent=1
    )
)
