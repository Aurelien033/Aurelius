#!/usr/bin/env python3
"""Generates aurelius_v1_colab.ipynb — the one-Run-all v1 build (setup -> traces -> SFT -> DPO -> merge/eval).
Self-contained cells (literal values, no cross-cell variables) so partial/out-of-order runs can't break."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = [
md("""# Aurelius v1 — clean build (Run all)
Clean base + clean verified data → SFT → DPO → merged model. **No external-model outputs except a distillation-permitted teacher (DeepSeek-R1, MIT).**

**HOW TO RUN:** Upload this file to **Google Colab** (File → Upload notebook), set **Runtime → Change runtime type → A100**, then **Runtime → Run all**. These are notebook cells — do **not** paste them into a terminal (they use Colab's `python` and the `!`/`%` magics). To change the base or sizes, edit the values marked `# EDIT` in each cell. End-to-end ≈ 2–2.5 h on A100."""),

code("""# 1. Setup
!git clone --depth 1 -b spike/first-light-readiness https://github.com/S3nna13/Aurelius.git
%cd Aurelius
!pip -q install -U transformers peft accelerate jsonschema pyyaml
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE — set Runtime>Change runtime type>A100")"""),

md("""## 2. Regenerate the clean SFT data
The 105k-example dataset isn't in the repo (its `train.jsonl` is 161 MB > GitHub's limit), but it's **deterministic synthetic** and the generator *is* in the repo — so this recreates it **byte-identically** in ~2 s. (Zero external-model outputs; fully clean.)"""),
code("""!python scripts/generate_aurelius_reasoning_data_v7.py \\
  --scale 150 --base-scale 400 --version aurelius-reasoning-sft-v20 --seed 20260614 \\
  --output data/aurelius_reasoning_sft_v20
!echo "train rows:" && wc -l < data/aurelius_reasoning_sft_v20/sft/train.jsonl"""),

md("""## 3. Verified traces (quality layer)
Run gym tasks through the clean teacher; keep only answers that **pass the verifier**. Watch `kept N (M verified)` — M is your usable data."""),
code("""# EDIT: teacher / #tasks / samples.  Of 300 tasks, the other 150 stay held-out for eval.
!python docs/training/make_traces.py \\
  --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\
  --split eval_data/selector_split_v0.1.json --n 150 --samples 3 \\
  --max_new 2048 --gen_batch 8 --dtype bf16 --out /content/traces_r1.jsonl"""),

md("""## 4. SFT — scaffold (v20) + quality (traces)
Gentle LoRA, loss masked to responses. Eval is **held out** (`--eval_exclude` drops the traced tasks). Watch the **BEFORE → AFTER Δ**: Δ>0 = the SFT specialized the base."""),
code("""# EDIT: --base (e.g. Qwen/Qwen3-8B for a stronger model), --max_rows, --epochs, --lr
!python docs/training/sft_train.py \\
  --base WeiboAI/VibeThinker-3B \\
  --data data/aurelius_reasoning_sft_v20/sft/train.jsonl,/content/traces_r1.jsonl --verified_only 0 \\
  --max_rows 8000 --epochs 2 --lr 1e-4 \\
  --eval_exclude /content/traces_r1.jsonl --n_eval 60 \\
  --dtype bf16 --out /content/vibethinker-sft"""),

md("""## 5. DPO — sharpen (optional)
Continues from the SFT adapter on the clean preference pairs. Many v20 chosen/rejected pairs are close, so the margin can be small — DPO here mainly reinforces format/persona. Skip this cell for a faster first pass (cell 6 falls back to the SFT model automatically)."""),
code("""# EDIT: keep --base identical to cell 4's; --n = #pairs
!python docs/training/dpo_train.py \\
  --base WeiboAI/VibeThinker-3B --adapter /content/vibethinker-sft \\
  --data data/aurelius_reasoning_sft_v20/preferences.jsonl --n 4000 --beta 0.1 --lr 5e-6 \\
  --dtype bf16 --out /content/vibethinker-dpo"""),

md("""## 6. Merge + final held-out eval + save
Merges the (DPO, else SFT) LoRA into the base, runs one held-out gym eval, saves a standalone model + a zip you can download from the Files panel."""),
code("""BASE = "WeiboAI/VibeThinker-3B"   # EDIT: must match cells 4-5
N_EVAL = 60
import sys, json, torch
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
A clean, specialized, merged small model + the headline held-out pass-rate. Next levers (highest first): **bigger base** (set `--base Qwen/Qwen3-8B` in cells 4–6), **more verified traces** (raise `--n`/`--samples` in cell 3), then **LayerDelta** compression."""),
]

nb["cells"] = cells
out = Path(__file__).resolve().parent / "aurelius_v1_colab.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
