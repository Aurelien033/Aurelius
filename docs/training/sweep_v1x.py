#!/usr/bin/env python3
"""sweep_v1x.py — v1.x experiment harness: base-ablation x trace-scaling, tabled.

Runs the v1 pipeline (make_traces -> sft_train -> held-out gym eval) across a grid of {base x trace_tasks}
and tables the held-out pass-rate + Δ. Answers the two highest-ROI v1.x questions in one job:
  - base ablation:    which clean base + our SFT wins?      (vary --bases at a fixed trace size)
  - trace scaling:    does more verified trace data help?    (vary --trace_sizes at a fixed base)
Traces are generated ONCE per size and reused across bases (the expensive step isn't repeated). Each SFT run
writes result.json; the sweep collects them. Reuses the committed scripts via subprocess — no logic fork.

Run (Colab Pro / A100):
  python docs/training/sweep_v1x.py --bases WeiboAI/VibeThinker-3B,Qwen/Qwen3-8B --trace_sizes 150 \
     --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --sft_rows 8000 --out /content/sweep
  # trace-scaling: --bases WeiboAI/VibeThinker-3B --trace_sizes 60,150,300
"""
import argparse, json, subprocess, sys, re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
V20 = "data/aurelius_reasoning_sft_v20"


def sh(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(REPO))


def slug(s): return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default="WeiboAI/VibeThinker-3B")
    ap.add_argument("--trace_sizes", default="150")
    ap.add_argument("--teacher", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--sft_rows", type=int, default=8000)
    ap.add_argument("--epochs", type=float, default=2); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n_eval", type=int, default=60)
    ap.add_argument("--out", default="/content/sweep")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    bases = a.bases.split(","); sizes = [int(x) for x in a.trace_sizes.split(",")]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    smoke = ["--smoke"] if a.smoke else []

    # ---- generate traces once per size (reused across bases) ----
    trace_files = {}
    for ts in sizes:
        tf = out / f"traces_{ts}.jsonl"
        if not tf.exists():
            sh([PY, "docs/training/make_traces.py", "--teacher", a.teacher, "--n", ts, "--samples", a.samples,
                "--max_new", 2048, "--gen_batch", 8, "--dtype", "bf16", "--out", tf, *smoke])
        trace_files[ts] = tf
        kept = sum(1 for _ in open(tf)) if tf.exists() else 0
        print(f"  traces[{ts}] -> {tf} ({kept} rows)", flush=True)

    # ---- grid: base x trace_size ----
    results = []
    for base in bases:
        for ts in sizes:
            rundir = out / f"{slug(base)}_ts{ts}"
            sh([PY, "docs/training/sft_train.py", "--base", base,
                "--data", f"{V20}/sft/train.jsonl,{trace_files[ts]}", "--verified_only", 0,
                "--max_rows", a.sft_rows, "--epochs", a.epochs, "--lr", a.lr,
                "--eval_exclude", trace_files[ts], "--n_eval", a.n_eval, "--dtype", "bf16", "--out", rundir, *smoke])
            r = json.load(open(rundir / "result.json"))
            r["trace_tasks"] = ts
            results.append(r)

    # ---- table ----
    json.dump(results, open(out / "sweep_results.json", "w"), indent=1)
    print("\n" + "=" * 64 + "\nv1.x SWEEP RESULTS (held-out gym pass-rate)\n" + "=" * 64)
    print(f"{'base':32} {'traces':>7} {'before':>7} {'after':>7} {'Δpp':>6}")
    for r in sorted(results, key=lambda x: (-x["after"], x["base"])):
        print(f"{r['base'][:32]:32} {r['trace_tasks']:>7} {r['before']*100:>6.1f}% "
              f"{r['after']*100:>6.1f}% {r['delta']*100:>+6.1f}")
    best = max(results, key=lambda x: x["after"])
    print(f"\nBEST: {best['base']} @ {best['trace_tasks']} traces -> {best['after']*100:.1f}% "
          f"(Δ {best['delta']*100:+.1f}pp). Full json -> {out/'sweep_results.json'}")


if __name__ == "__main__":
    main()
