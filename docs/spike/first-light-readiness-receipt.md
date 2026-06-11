# First Light Readiness Spike — Receipt

**Date:** 2026-06-11
**Branch:** spike/first-light-readiness
**Operator:** Hermes Agent (qwen3.7-plus via opencode-go)

---

## Step 1: Model Loading + Forward Pass

| Item | Result |
|------|--------|
| OLMo-2-0425-1B (revision a1847dff) | Loaded on MPS bf16 ✓ |
| Forward pass | OK (logits [1,3,100352]) |
| Greedy generation | OK (512 tokens) |
| RSS after load | 9.32 GB (OVER 7.5 GB target) |
| RSS cause | Framework overhead (torch+transformers), not model weights |

**Note:** RSS violation is framework overhead on macOS. Model weights ~2 GB bf16.
The 7.5 GB target was not met but the model runs correctly.

## Step 2: 10-Instance Smoke Test

- **Model:** OLMo-2-0425-1B (base, not instruction-tuned)
- **Prompt:** raw task text (no function signature)
- **Result:** 0/10 passed (0%)
- **Failure mode:** Syntactically broken code (unterminated strings, invalid syntax)
- **Traces:** 10 rows emitted, schema v1.1.0 valid (FORCED_BASELINE, none_forced)
- **Trace file:** /tmp/smoke_10_traces.json

**Conclusion:** Tooling works. Base model + greedy + no code-context = garbage.
Schema compliance verified.

## Step 3: ADR-1b Base Smoke Pass (30 instances)

### Cascade results

| Model | Pass rate | In band [20%,80%]? |
|-------|-----------|---------------------|
| OLMo-2-0425-1B (base) | 0/30 (0.0%) | MISS |
| Qwen2.5-0.5B-Instruct (no sig, no extract) | 0/30 (0.0%) | MISS |
| Qwen2.5-0.5B-Instruct (sig, no extract) | 0/30 (0.0%) | MISS |
| Qwen2.5-1.5B-Instruct (no extract) | 0/30 (0.0%) | MISS |
| **Qwen2.5-1.5B-Instruct (sig + code extract)** | **7/30 (23.3%)** | **HIT** |

### Root cause of initial failures

Two bugs in the evaluation pipeline:
1. **Missing function signature in prompt:** Base/instruct models generate functions
   with different names than what MBPP tests expect. Fix: extract `def name(args)`
   from reference code and include in prompt.
2. **Markdown fence contamination:** Instruct models wrap code in ```python...```
   with explanation text. The scorer runs the entire completion as Python, causing
   SyntaxError. Fix: regex-extract code block from markdown fences before scoring.

### FROZEN-BASE-v1 Pin

```yaml
config_id: FROZEN-BASE-v1
repo: Qwen/Qwen2.5-1.5B-Instruct
revision: 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
tokenizer_hash: 86a13989cdf0acbe
config_hash: 98d2ff8cc47488d0
pass_rate_pilot: 0.233 (7/30)
seed: 42
task_ids: [23,24,26,27,55,58,63,68,82,112,122,125,130,136,151,225,227,269,290,298,313,319,338,343,357,370,377,388,390,467]
```

### Llama-3.2-1B-Instruct

Gated on HuggingFace (403). Not available without prior access approval.

## Step 4: Identity-Skip Wrapper Proof

| Item | Result |
|------|--------|
| Model | Qwen2.5-1.5B-Instruct (28 layers) |
| Routable window | Layers 7-20 (middle 50%, 14 layers) |
| k (skip count) | round(0.25 × 14) = 4 |
| Skip indices | [7, 8, 9, 10] |
| Wrapper | IdentitySkipLayer: returns hidden_states tensor unchanged |
| Forward pass | ✓ Runs |
| Outputs differ from dense | ✓ max_diff=14.75, mean_diff=1.37 |
| Identity (not zeros) | ✓ Hidden state passes through unchanged |

**Key implementation detail:** Qwen2 decoder layers return `torch.Tensor` directly
(not tuples). The wrapper must match this interface.

## Step 5: Canonical Operator Decision

### Survey of src/model/*.py

| File | Lines | Real torch | Custom blocks | Wraps HF? | Notes |
|------|-------|-----------|---------------|-----------|-------|
| mod.py | 175 | ✓ | ✓ | ✗ | Basic MoD, token router |
| **mod_v2.py** | **553** | **✓** | **✓** | **✗** | **Capacity tracking, aux loss, top-p routing** |
| mixture_of_depths.py | 223 | ✓ | ✓ | ✗ | Another MoD variant |
| dynamic_depth.py | 167 | ✓ | ✓ | ✗ | Adaptive layer depth |
| token_skip.py | 192 | ✓ | ✓ | ✗ | Per-token layer skipping |
| act_routing.py | 244 | ✓ | ✗ | ✗ | ACT halting, ponder cost |
| early_exit.py | 355 | ✓ | ✓ | ✗ | Intermediate exit points |
| remode.py | 167 | ✗ | ✗ | ✗ | **PURE PYTHON TOY — do not build on** |

### Recommendation

**Phase 1 operator: mod_v2.py**

Rationale:
- Most complete MoD implementation (553 lines)
- Real torch.nn.Module with capacity tracking (CapacityTracker)
- Auxiliary load-balancing loss + z-loss regularization
- Top-p routing in addition to top-k and threshold
- Routing visualization utilities

**Critical caveat:** mod_v2.py builds CUSTOM transformer blocks (uses its own
GroupedQueryAttention, SwiGLUFFN, RMSNorm). It does NOT wrap HuggingFace models.
To use it with FROZEN-BASE-v1 (Qwen2.5-1.5B-Instruct), we need to either:
1. Extract the routing logic from mod_v2 and adapt it as a wrapper around HF layers
2. Or train a from-scratch model using mod_v2's components

### ReMoDE disambiguation

- **repo src/model/remode.py:** Pure-Python toy. No torch. Do not build on.
- **plan "ReMoDE":** Reactive routing (SKIP/NORMAL/AMPLIFY + verified gain).
  This is a DESIGN, not the repo file. The repo file is a different thing that
  happens to share the name.

## Step 6: Governing Doc Updates

- This receipt: docs/spike/first-light-readiness-receipt.md
- Obligation ledger: OBL-037 discharged by this receipt
- Preregistration: execute-time pins updated (see separate commit)

---

## Commands and Seeds

```bash
# Environment
cd /Users/christienantonio/aurelius
source .venv/bin/activate
# torch=2.11.0, transformers=5.11.0, mps=True

# Model downloads
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')"

# 30-instance test (final, passing version)
python3 << 'PYEOF'
import re, random, pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.eval.mbpp_scorer import MBPPProblem, score_single

REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42

def extract_code_block(completion):
    pattern = r'```(?:python)?\s*\n(.*?)```'
    match = re.search(pattern, completion, re.DOTALL)
    return match.group(1).strip() if match else completion.strip()

tok = AutoTokenizer.from_pretrained(REPO_ID)
model = AutoModelForCausalLM.from_pretrained(REPO_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True).to("mps")
model.eval()

tbl = pq.read_table("/tmp/mbpp_test.parquet")
all_ids = [tbl.column("task_id")[i].as_py() for i in range(tbl.num_rows)]
random.seed(SEED)
sampled = sorted(random.sample(all_ids, 30))

passed = 0
for tid in sampled:
    idx = all_ids.index(tid)
    row = {c: tbl.column(c)[idx].as_py() for c in tbl.column_names}
    prob = MBPPProblem(task_id=tid, text=row["text"], code=row["code"],
                       test_list=row["test_list"], test_setup_code=row.get("test_setup_code","") or "")
    sig = re.match(r'(def \w+\([^)]*\))', row["code"])
    content = f'{row["text"]}\n\n{sig.group(1)}:' if sig else row["text"]
    messages = [{"role": "user", "content": content}]
    text_input = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tok(text_input, return_tensors="pt").to("mps")
    with torch.no_grad():
        gen = model.generate(**inp, max_new_tokens=512, do_sample=False, pad_token_id=tok.eos_token_id)
    completion = extract_code_block(tok.decode(gen[0], skip_special_tokens=True)[len(text_input):])
    res = score_single(prob, completion)
    if res.passed: passed += 1
print(f"{passed}/30 = {passed/30*100:.1f}%")
PYEOF
```

## Hard Rules Compliance

- [x] This spike is NOT First Light. No claims made.
- [x] Everything labeled pilot.
- [x] Greedy decoding only for band check.
- [x] Seeds, commands, versions recorded.
- [x] No history sections of master plan edited.
- [x] Work on spike branch only. No push. No other branches touched.
