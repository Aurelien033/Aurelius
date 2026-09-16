# Skip-Robust Co-Train Recipe (the E87 continuation, the cheap fast-track to the payoff)

> Created 2026-06-14. The single cheapest experiment that turns the training phase into science.
> Needs cloud GPU + design ratification; NOT a from-scratch pretrain (no $15-25K, no weeks).

## Why this, before any 100M/1B pretrain

E87 found: on the **frozen** base, skipping 4 of 28 layers costs ~43pp (dense 73% → random 30%) and
informed per-layer selection can't beat random (layer non-additivity). But the frozen base **never
trained under the skip distribution it's asked to serve** (§135.3 — the missing load-bearing
experiment). The hypothesis: a model **co-trained to be skip-robust** makes skipping cheap, which is
the precondition for *any* routing win. This tests it for ~$50-200 and a day, on the existing base —
no pretrain.

## The experiment

```
base        = FROZEN-BASE-v1 (Qwen/Qwen2.5-1.5B, rev 8faed761)   # the teacher/start
adapters    = LoRA on attention + MLP of the routable layers [7..20] (and the router head)
skip_dist   = per training step, sample a skip-set: k ~ {0:0.4, 1:0.2, 2:0.2, 4:0.2} layers from [7..20]
              applied via the E87-verified IdentitySkip wrapper during the forward pass
objective   = next-token CE on a repair/code+general mix, computed UNDER the sampled skip
              (so the LoRA learns to produce good outputs even when layers are skipped)
data        = the gym families' source distribution (MBPP/code) + a general slice (fineweb-edu),
              tokenized per data_pipeline.py
budget      = ~50-200M tokens of LoRA fine-tuning, 1 GPU, hours
```

## Hypotheses (pre-register before running, like FL-PREREG / E87)

- **H-CT-0 (cost shrinks):** after co-train, dense−skip cost at k=4 is materially smaller than the
  frozen base's 43pp (the model recovers skipped-layer function). Falsified if the gap is unchanged.
- **H-CT-1 (routing becomes learnable):** on the co-trained model, a learned g_gain (or even entropy)
  controller at the best k from the E87 k-sweep BEATS random AND beats the frozen-base baseline at
  matched compute. (The E87 negative was on the frozen base; this re-tests on the skip-aware model.)
- **H-CT-2 (no capability loss):** co-trained dense pass-rate ≥ frozen dense (the LoRA doesn't degrade
  full-compute behavior).
- **KILL:** if cost doesn't shrink AND routing still can't beat random, layer-skip is the wrong compute
  lever for this base — pivot to width-sparsity/AMPLIFY or accept routing needs a from-scratch
  skip-native architecture.

## Eval = re-run the E87 chessboard + k-sweep on the co-trained model

Identical harness (`docs/first_light/e87_chessboard.py`, `e87_ksweep.py`), same held-out 60 test
instances, same independent re-score discipline (AGREE check). The comparison is co-trained-vs-frozen
at matched k. This makes the result directly continuous with E87.

## Why it's the fast-track

- Reuses everything: the base, the gym, the IdentitySkip wrapper, the chessboard, the re-score tools.
- ~$50-200 vs ~$15-25K for a from-scratch 100M/1B.
- Answers the load-bearing question (does training make skipping cheap?) BEFORE committing to scale.
- If it works → THEN a from-scratch skip-native 100M/1B is justified (and config_100m.yaml is ready).
- If it fails → you saved the pretrain budget and learned the lever is wrong.

## Blocks before running
1. design ratification (one line, like the ADRs) + a frozen pre-registration (hypotheses above).
2. cloud GPU (LoRA FT of 1.5B; ~1 GPU, hours).
3. a small mixed corpus tokenized (data_pipeline.py) + the gym source.
4. the skip-distribution training loop (extends trainer.py's loop with the IdentitySkip wrapper in the
   forward — small, launch-ready to write once ratified).

## Recommended order (the whole fast-track)
1. ✅ training readiness spike — DONE, PASS (the repo trains; loss 10.86→6.77).
2. **this skip-robust co-train** — the cheap science (does training make skipping cheap?).
3. 100M from-scratch (config_100m.yaml) — only if co-train shows routing is worth a skip-native model.
4. 1B via prune+distill from a teacher — only after the 100M rung wins (§141 gate).
