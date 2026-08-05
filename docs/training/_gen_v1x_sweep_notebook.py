#!/usr/bin/env python3
"""Generates aurelius_v1x_sweep_colab.ipynb — base-ablation x trace-scaling sweep (v1.x iteration tooling).
Self-contained cells (literal values, no cross-cell variables)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

nb["cells"] = [
md("""# Aurelius v1.x — base-ablation × trace-scaling sweep
Which clean **base** + our SFT wins, and how far **more verified traces** keep helping. Held-out gym eval, tabled.

**HOW TO RUN:** Upload to **Google Colab**, set **Runtime → A100**, **Run all**. These are notebook cells — don't paste into a terminal. Edit the `--bases` / `--trace_sizes` in the last cell. Each grid cell ≈ one SFT (~30–70 min), so keep the grid small. (Run this *after* v1 ships.)"""),

code("""# 1. Setup
!git clone --depth 1 -b spike/first-light-readiness https://github.com/S3nna13/Aurelius.git
%cd Aurelius
!pip -q install -U transformers peft accelerate jsonschema pyyaml
!pip -q uninstall -y torchao   # Colab ships torchao 0.10 which peft's LoRA dispatch rejects; we don't use it
import torch; print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE — set A100")"""),

md("""## 2. Regenerate the clean SFT data
Deterministic synthetic; recreates the 105k-example v20 dataset byte-identically in ~2 s (not shipped in the repo: 161 MB > GitHub limit)."""),
code("""!python scripts/generate_aurelius_reasoning_data_v7.py \\
  --scale 150 --base-scale 400 --version aurelius-reasoning-sft-v20 --seed 20260614 \\
  --output data/aurelius_reasoning_sft_v20"""),

md("""## 3. Run the sweep
- **Base ablation:** list bases in `--bases`, one `--trace_sizes`.
- **Trace scaling:** one base, list sizes (e.g. `--trace_sizes 60,150,300`).
Generates traces once per size (reused across bases), SFTs each grid cell, evals held-out, prints a ranked table + saves `sweep_results.json`."""),
code("""# EDIT --bases / --trace_sizes
!python docs/training/sweep_v1x.py \\
  --bases WeiboAI/VibeThinker-3B,Qwen/Qwen3-8B \\
  --trace_sizes 150 \\
  --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\
  --sft_rows 8000 --n_eval 60 --out /content/sweep"""),

md("""### Read it
Top row = the winning base/trace-budget → drop it into the v1 notebook (`--base` in cells 3–5, `--n` in cell 2). If the trace-scaling curve is still rising, more traces are worth it; if flat, the next lever is a **bigger base** or **LayerDelta**."""),
]
out = Path(__file__).resolve().parent / "aurelius_v1x_sweep_colab.ipynb"
nbf.write(nb, str(out)); print("wrote", out)
