"""ADR-1b gap runs: the ratified candidate list named BASE Qwen2.5-0.5B/1.5B; the spike
only tested the Instruct variants. Same fixed pipeline the OLMo fair re-test used
(sig in prompt, sig prepended to scored completion, top-level truncation), seed 42,
same 30 MBPP instances."""
import gc
import json
import re
import random
import sys

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/Users/christienantonio/aurelius")
from src.eval.mbpp_scorer import MBPPProblem, score_single  # noqa: E402

SEED = 42
tbl = pq.read_table("/tmp/mbpp_test.parquet")
all_ids = [tbl.column("task_id")[i].as_py() for i in range(tbl.num_rows)]
random.seed(SEED)
SAMPLED = sorted(random.sample(all_ids, 30))


def truncate_completion(text):
    for pat in ["\ndef ", "\nassert", "\nprint(", "\nclass ", "\nif __name__", "\n#"]:
        i = text.find(pat)
        if i != -1:
            text = text[:i]
    return text


for repo in ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"]:
    print(f"=== {repo} (base completion pipeline) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, dtype=torch.bfloat16, low_cpu_mem_usage=True).to("mps").eval()
    passed = 0
    for tid in SAMPLED:
        idx = all_ids.index(tid)
        row = {c: tbl.column(c)[idx].as_py() for c in tbl.column_names}
        prob = MBPPProblem(task_id=tid, text=row["text"], code=row["code"],
                           test_list=list(row["test_list"]),
                           test_setup_code=row.get("test_setup_code", "") or "")
        sig_m = re.match(r"(def \w+\([^)]*\))", row["code"])
        sig = sig_m.group(1) if sig_m else None
        text_input = f'{row["text"]}\n\n{sig}:' if sig else row["text"]
        inp = tok(text_input, return_tensors="pt").to("mps")
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=512, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        raw = tok.decode(gen[0], skip_special_tokens=True)[len(text_input):]
        completion = (sig + ":\n" if sig else "") + truncate_completion(raw)
        res = score_single(prob, completion)
        passed += int(res.passed)
        print(f"  task {tid}: {'PASS' if res.passed else 'fail'}", flush=True)
    rate = passed / 30
    print(f"RESULT {repo}: {passed}/30 = {rate*100:.1f}%  in_band={0.20 <= rate <= 0.80}", flush=True)
    del model
    gc.collect()
    torch.mps.empty_cache()
