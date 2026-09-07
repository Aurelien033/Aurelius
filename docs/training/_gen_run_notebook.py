#!/usr/bin/env python3
"""Generates aurelius_run_colab.ipynb — the full current run (Setup + Path A RLVR + Path B stronger-teacher SFT).
Self-contained cells (literal values), all session fixes baked in."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

nb["cells"] = [
md("""# Aurelius — full run (RLVR + stronger-teacher SFT)
**Upload to Colab, set Runtime → A100, run Cell 0 first.** Then **Path A** (RLVR — ⚠ the v1 recipe measured ≈ base on 2026-06-25; re-run only with a changed recipe, see note) and/or **Path B** (stronger-teacher SFT). Paths write to distinct `/content` names, so run either order. These are notebook cells — don't paste into a terminal."""),

md("""## Cell 0 — Setup (always run first)"""),
code("""!git clone --depth 1 -b spike/first-light-readiness https://github.com/Aurelien033/Aurelius.git
%cd Aurelius
!pip -q install -U transformers peft accelerate datasets bitsandbytes jsonschema pyyaml
!pip -q uninstall -y torchao   # Colab's torchao 0.10 breaks peft's LoRA dispatch
import torch; print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE — set Runtime>A100")"""),

md("""# Path A — RLVR (⚠ the v1 recipe measured ≈ base — fix the recipe before trusting a re-run)
The reward is ground-truth verifier pass-rate — no teacher ceiling *in principle*. But the v1 run (lr 1e-6, KL-anchored) measured ≈ base (2026-06-25): the policy barely moved. If re-running, raise lr (~1e-5/5e-5) and broaden the task distribution. Run A1 first; if `mean_reward` rises, do A2→A3."""),

md("""### A1 — smoke (~20 min): does `mean_reward` rise on the gym?"""),
code("""!python docs/training/grpo_train.py --base Qwen/Qwen2.5-1.5B --data gym \\
  --n_tasks 16 --group_size 4 --steps 60 --max_new 256 --lr 5e-6 --out /content/rlvr_smoke"""),

md("""### A2 — real RLVR on Qwen3-8B (session-sized: 150 steps ≈ 2–2.5 h, checkpoints every 50)
If it OOMs (8B + G=8 generation is the tight spot): drop `--group_size 4` and/or `--max_new 384`. On a disconnect, `/content/aurelius-rlvr` holds the latest checkpoint — just re-run A3."""),
code("""!python docs/training/grpo_train.py --base Qwen/Qwen3-8B --data both \\
  --group_size 8 --steps 150 --max_new 512 --lr 1e-6 --save_every 50 --out /content/aurelius-rlvr"""),

md("""### A3 — eval the RLVR model vs base (84.1% HumanEval)"""),
code("""!python docs/training/eval_code_bench.py --model /content/aurelius-rlvr --bench humaneval --think 0"""),

md("""# Path B — stronger-teacher SFT (Qwen3.6-27B → lift, not drag)
A *stronger* teacher (27B ≫ 8B) distills *up* — unlike R1-7B which dragged v2 down. (3.6-27B is a VLM; the loader handles it, but if B1 errors on **load**, paste it.)"""),

md("""### B1 — coding traces (MBPP) from Qwen3.6-27B (4-bit, fits A100)"""),
code("""!python docs/training/make_code_traces.py --teacher Qwen/Qwen3.6-27B --load_4bit \\
  --n 300 --samples 2 --out /content/mbpp_traces_q36.jsonl"""),

md("""### B2 — reasoning traces (gym) from Qwen3.6-27B"""),
code("""!python docs/training/make_traces.py --teacher Qwen/Qwen3.6-27B --load_4bit \\
  --n 240 --samples 2 --out /content/gym_traces_q36.jsonl"""),

md("""### B3 — SFT Qwen3-8B on the stronger-teacher traces → v3"""),
code("""!python docs/training/sft_train.py --base Qwen/Qwen3-8B \\
  --data /content/mbpp_traces_q36.jsonl,/content/gym_traces_q36.jsonl --verified_only 0 \\
  --epochs 2 --lr 2e-5 --skip_eval --dtype bf16 --out /content/aurelius-v3-8b"""),

md("""### B4 — eval v3 vs base (84.1% HumanEval)"""),
code("""!python docs/training/eval_code_bench.py --model /content/aurelius-v3-8b --bench humaneval --think 0"""),

md("""### Read it
- **A3 robustly > 84.1% across HumanEval *and* MBPP** → RLVR beat the base. (⚠ the v1 run was +2/164 on HumanEval = noise, MBPP 125=125 — require a multi-bench margin, not 2 problems on one bench.) **B4 > 84.1%** → the stronger teacher lifted it.
- Download a winner: `/content/aurelius-rlvr` or `/content/aurelius-v3-8b` (zip from the Files panel)."""),
]
out = Path(__file__).resolve().parent / "aurelius_run_colab.ipynb"
nbf.write(nb, str(out)); print("wrote", out)
