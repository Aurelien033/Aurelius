#!/usr/bin/env python3
"""Generates aurelius_v1_colab.ipynb — the one-Run-all v1 build (setup -> traces -> SFT -> DPO -> merge/eval)."""
import json, nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = [
md("""# Aurelius v1 — clean build (Run all)
Clean base + clean verified data → SFT → DPO → merged model. **No external-model outputs except a distillation-permitted teacher (DeepSeek-R1, MIT).**
**Set Runtime → Change runtime type → A100** before running. End-to-end ≈ 2–2.5 h at the default knobs (shrink them in cell 2 for a faster first pass)."""),

code("""# 1. Setup
!git clone --depth 1 -b spike/first-light-readiness https://github.com/S3nna13/Aurelius.git
%cd Aurelius
!pip -q install -U transformers peft accelerate jsonschema pyyaml
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE — set Runtime>Change runtime type>A100")"""),

md("""## 2. Config — all the knobs in one place
Defaults target a real (not toy) v1 in ~2–2.5 h on A100. For a fast first pass: `TRACE_TASKS=60, SFT_ROWS=3000`, and skip cell 5 (DPO)."""),
code("""BASE        = "WeiboAI/VibeThinker-3B"                 # clean base (MIT). Swap -> "Qwen/Qwen3-8B" for a stronger model
TEACHER     = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" # clean teacher (MIT, distillation-permitted)
TRACE_TASKS = 150     # how many gym tasks to distill verified traces from (of 300; other 150 = held-out eval)
TRACE_SAMPLES = 3     # rejection-sampling tries per task (keep the first that PASSES the verifier)
SFT_ROWS    = 8000    # cap on v20 SFT rows (shuffled; full set is 105k). +traces are always included
SFT_EPOCHS  = 2
SFT_LR      = 1e-4
DPO_PAIRS   = 4000    # subset of the 16,749 clean preference pairs
N_EVAL      = 60      # held-out gym tasks for the before/after pass-rate
V20         = "data/aurelius_reasoning_sft_v20"
print("config set:", BASE, "| trace", TRACE_TASKS, "| sft_rows", SFT_ROWS, "| dpo", DPO_PAIRS)"""),

md("""## 3. Verified traces (quality layer)
Run `TRACE_TASKS` gym tasks through the clean teacher; keep only answers that **pass the verifier**. Watch `kept N (M verified)` — M is your usable data."""),
code("""!python docs/training/make_traces.py --teacher {TEACHER} \\
  --split eval_data/selector_split_v0.1.json --n {TRACE_TASKS} --samples {TRACE_SAMPLES} \\
  --max_new 2048 --gen_batch 8 --dtype bf16 --out /content/traces_r1.jsonl"""),

md("""## 4. SFT — scaffold (v20) + quality (traces)
Gentle LoRA, loss masked to responses. Eval is **held out** (`--eval_exclude` drops the traced tasks). Watch the **BEFORE → AFTER Δ**: Δ>0 = the SFT specialized the base."""),
code("""!python docs/training/sft_train.py --base {BASE} \\
  --data {V20}/sft/train.jsonl,/content/traces_r1.jsonl --verified_only 0 \\
  --max_rows {SFT_ROWS} --epochs {SFT_EPOCHS} --lr {SFT_LR} \\
  --eval_exclude /content/traces_r1.jsonl --n_eval {N_EVAL} \\
  --dtype bf16 --out /content/vibethinker-sft"""),

md("""## 5. DPO — sharpen (optional)
Continues from the SFT adapter on the clean preference pairs. Note: many v20 chosen/rejected pairs are close, so the margin can be small — DPO here mainly reinforces format/persona. Skip this cell for a faster first pass."""),
code("""!python docs/training/dpo_train.py --base {BASE} --adapter /content/vibethinker-sft \\
  --data {V20}/preferences.jsonl --n {DPO_PAIRS} --beta 0.1 --lr 5e-6 \\
  --dtype bf16 --out /content/vibethinker-dpo"""),

md("""## 6. Merge + final held-out eval + save
Merges the (DPO, else SFT) LoRA into the base, runs one held-out gym eval, saves a standalone model + a zip you can download from the Files panel."""),
code("""import sys, json, torch
from pathlib import Path
sys.path.insert(0, "docs/training"); sys.path.insert(0, "docs/first_light")
from sft_train import gym_passrate
import first_light_runner_v4 as R
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ADAPTER = "/content/vibethinker-dpo" if Path("/content/vibethinker-dpo").exists() else "/content/vibethinker-sft"
dev = "cuda"
tok = AutoTokenizer.from_pretrained(ADAPTER)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(dev)
m = PeftModel.from_pretrained(m, ADAPTER).merge_and_unload()   # bake LoRA into weights

idx = {}
for root, fam in [(R.RESEARCH/"gym-v0.1-FL","F2_json"), (R.RESEARCH/"gym-v0.3","F3_type")]:
    for sp in ["dev","test","smoke","train"]:
        for f in (root/fam/sp).glob("*.json"):
            d = json.loads(f.read_text()); idx[d["instance_id"]] = d
excl = {json.loads(l).get("instance_id") for l in open("/content/traces_r1.jsonl")}
allids = json.load(open("eval_data/selector_split_v0.1.json"))["all"]
eids = [i for i in allids if i in idx and i not in excl][:N_EVAL]
acc = gym_passrate(tok, m, idx, eids, dev, max_new=256)
print(f"\\n=== FINAL Aurelius-v1 (from {ADAPTER}) held-out gym pass-rate (n={len(eids)}): {acc*100:.1f}% ===")

m.save_pretrained("/content/aurelius-v1"); tok.save_pretrained("/content/aurelius-v1")
import subprocess; subprocess.run("cd /content && zip -qr aurelius-v1.zip aurelius-v1", shell=True)
print("saved -> /content/aurelius-v1/  + /content/aurelius-v1.zip (download from the Files panel)")"""),

md("""### Done.
You now have a clean, specialized, merged small model + the headline held-out pass-rate. Next levers (highest first): **bigger base** (`BASE="Qwen/Qwen3-8B"`), **more verified traces** (raise `TRACE_TASKS`/`TRACE_SAMPLES` + add other gym pools), then **LayerDelta** compression on the serving distribution."""),
]

nb["cells"] = cells
out = Path(__file__).resolve().parent / "aurelius_v1_colab.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
