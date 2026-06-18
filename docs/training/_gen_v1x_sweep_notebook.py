#!/usr/bin/env python3
"""Generates aurelius_v1x_sweep_colab.ipynb — base-ablation x trace-scaling sweep (v1.x iteration tooling)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

nb["cells"] = [
md("""# Aurelius v1.x — base-ablation × trace-scaling sweep
The two highest-ROI "make it better" experiments in one job: which clean **base** + our SFT wins, and how far
**more verified traces** keep helping. Held-out gym eval, tabled. **Runtime → A100.** Time scales with the grid
size — start small. (This is v1.x tooling; run it after v1 ships.)"""),

code("""# 1. Setup
!git clone --depth 1 -b spike/first-light-readiness https://github.com/S3nna13/Aurelius.git
%cd Aurelius
!pip -q install -U transformers peft accelerate jsonschema pyyaml
import torch; print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE — set A100")"""),

md("""## 2. Config — choose the sweep
- **Base ablation:** list bases, single trace size. - **Trace scaling:** single base, list trace sizes.
Each grid cell = one full SFT (~30–70 min on A100), so keep the grid small for a first pass."""),
code("""BASES       = "WeiboAI/VibeThinker-3B,Qwen/Qwen3-8B"   # base ablation (or one base for pure trace-scaling)
TRACE_SIZES = "150"                                     # trace scaling (e.g. "60,150,300") or one size
TEACHER     = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
SFT_ROWS    = 8000
N_EVAL      = 60
print("sweep:", BASES, "x traces", TRACE_SIZES)"""),

md("""## 3. Run the sweep
Generates traces once per size (reused across bases), SFTs each grid cell, evals held-out, and prints a ranked table + saves `sweep_results.json`."""),
code("""!python docs/training/sweep_v1x.py --bases {BASES} --trace_sizes {TRACE_SIZES} \\
  --teacher {TEACHER} --sft_rows {SFT_ROWS} --n_eval {N_EVAL} --out /content/sweep"""),

md("""### Read it
Top row = the winning base/trace-budget. Use that as the v1.x config: the base goes into the v1 notebook's `BASE=`,
the trace budget into `TRACE_TASKS=`. If the trace-scaling curve is still rising at the top, more traces are worth
generating; if it's flat, you've hit the data plateau and the next lever is a **bigger base** or **LayerDelta**."""),
]
out = Path(__file__).resolve().parent / "aurelius_v1x_sweep_colab.ipynb"
nbf.write(nb, str(out)); print("wrote", out)
