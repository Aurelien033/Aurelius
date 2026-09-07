# AMC Buildout — Pre-Implementation Briefing

**Read this entire document before starting T01.**
It replaces re-reading the four Part files and answers the questions
you'll have before you've had them.

---

## 1. Where We Are Right Now

### Done
- **T00** — `Mamba2Block` committed at `07dd177f` on branch
  `clean/amc-curation-20260521-101220`
- 30/31 tests pass (1 CUDA device test correctly skipped on CPU)
- `TRANCHE_STATUS.md` initialized with full 40-tranche tracker
- All four Part documents written (~380 KB total)

### Not Done
- Nothing else in `src/model/` except T00
- Nothing in `src/training/` exists yet
- No GPU training has run
- No data has been downloaded or tokenized

### One Known Divergence from the Part 1 Spec

T00 uses the **mathematically correct** Mamba-2 (ZOH discretization,
proper A/B/C SSM matrix semantics, state shape `(B, nheads, headdim, d_state)`).
The Part 1 doc had a simplified recurrence that would have broken
state continuation and gradient flow. The public API is identical —
T01 wraps the exact same interface the spec promised.

---

## 2. Environment Check — Do This Now

```bash
cd /Users/christienantonio/aurelius
.venv/bin/python --version           # must be 3.12+
.venv/bin/python -c "import torch; print(torch.__version__)"   # must be 2.1+
.venv/bin/python -c "import einops; print(einops.__version__)" # must be 0.8+
.venv/bin/python -c "import msgpack; print(msgpack.__version__)" # must be 1.0+

# All existing tests must still pass
.venv/bin/python -m pytest tests/ -q --tb=no 2>&1 | tail -3
```

If any import fails:
```bash
uv pip install einops>=0.8 msgpack>=1.0 tqdm pyyaml
```

If tests fail — **don't start building. Fix the baseline first.**

---

## 3. How the Tranche System Works

```
T00 → T01 → T02 → T04 → T15 → T17 → T18-T22 (train)
                                         ↓
T03 → T11 → T14 ─────────────────────T28 → T32 (paper)
         ↓
      T12 → T23 → T24-T27 ───────────┘
```

**Rules:**
1. One atomic commit per tranche. Exact commit message from the doc.
2. All acceptance criteria must pass before committing.
3. Never advance to Tn+1 while Tn has failing tests.
4. Never push to `main`. All work lives on the current branch until
   the full buildout is done, then a single curated merge.
5. Update `docs/prompts/TRANCHE_STATUS.md` in the same commit.

---

## 4. The Four Part Documents — What's in Each

| Part | File | Size | What's in it |
|------|------|------|--------------|
| Part 1 | `AMC_FULL_BUILDOUT_PROMPTS.md` | ~73 KB | T00-T04, T11, T15-T16, T17-T35 stubs, cost, timeline |
| Part 2 | `AMC_FULL_BUILDOUT_PROMPTS_PART2.md` | ~79 KB | T05-T10, T12-T14, T17-T22 detail, T24-T25, T28-T31 |
| Part 3 | `AMC_FULL_BUILDOUT_PROMPTS_PART3.md` | ~87 KB | T08-T10 full impl, T12-T14 full impl, T15-T16 full, T26-T27, S01-S05 |
| Part 4 | `AMC_FULL_BUILDOUT_PROMPTS_PART4.md` | ~92 KB | Full configs, training launcher, ConstitutionalMemory, ablation configs, shell scripts, LaTeX paper skeleton |
| README | `README.md` | ~10 KB | Navigation, getting-started, timeline |
| Tracker | `TRANCHE_STATUS.md` | ~6 KB | Live progress — update after each commit |

**Total: ~356 KB of buildout documentation.**

---

## 5. Tool Setup — Before T01

### Composer 2.5 (preferred for greens + yellows)
1. Open Aurelius in Cursor
2. Rebuild the index: `Cmd+Shift+P → "Cursor: Rebuild Index"`
3. Open `docs/prompts/README.md` in a tab (so it's in context)
4. For each tranche: open the Part file → find the tranche → Cmd+I → paste + add guardrails → apply diff → run tests → commit

### Claude Opus 4.7 (preferred for red tranches)
1. Have the API key ready
2. For each red tranche: paste the full tranche + "I will review every line before commit" → generate → read every line → patch → commit

### GLM 5.1 (optional, second coder)
- Use when Composer's output looks off: paste the same tranche + "explain what this code does line by line" → GLM's CoT will surface bugs Composer's diff review misses

### Me (Hermes)
- Use for: red tranche authorship, review of Composer/GLM output,
  phase-boundary validation gates, debugging stuck tranches

---

## 6. The Three Invariants — Green/Yellow Tranches

### Critical Path (minimum viable paper, ~16 weeks)

```
T00 ✅ → T01 → T02 → T04 → T17 → T18-T22 → T28 → T32
```

These are the tranches that must be clean before you touch training.

### YELLOW LIGHTS — Invariants to paste at the end of each prompt

#### T01 — AMCSSMLayer (state shape invariant)
```python
# Mamba2Block state shape is (B, nheads, headdim, d_state) — do NOT pool or
# reshape the state. Output of forward() is (B, L, d_model), not aggregated.
# AMCSSMLayer.get_state() returns the raw Mamba2Block state tensor with NO transforms.
```

#### T02 — Promotion gate (formula invariant)
```python
# GumbelSoftmax straight-through: hard - soft.detach() + soft
# The hard sample must not carry gradient, the soft sample must carry gradient.
# NOT: hard.detach_()  NOT: just 'soft'  NOT: just 'hard - soft'
```

#### T04 — AMCTransformer (attention/SSM alternation invariant)
```python
# Attention layers get RoPE. SSM layers do NOT get RoPE.
# SSM layers are at ODD indices: 1, 3, 5, ... (0-indexed).
# MLA layers at EVEN indices: 0, 2, 4, ...
# Do NOT apply RoPE to SSM hidden states at any point.
```

#### T05 — SurpriseHead (gradient boundary invariant)
```python
# surprise = self.head(x.detach()) — detach stopgrad on the input only.
# NOT torch.no_grad() — that suppresses all gradient in the whole block.
# NOT x.clone().detach() — clone isn't needed, detach alone is sufficient.
```

#### T06 — Gate networks (activation invariant)
```python
# All gate outputs (decay, erase, write) must be in [0, 1].
# Use nn.Sigmoid() as the final activation. NOT nn.ReLU, NOT nn.LeakyReLU.
```

#### T07 — MLA continuation (position invariant)
```python
# position_offset is ADDED to base_position, never subtracted.
# rotary_emb expects (B, L, D) not (B, 1, D) with broadcast dims.
```

#### T15 — AMC losses (sign invariant)
```python
# surprise_loss: BCELogitsLoss on surprise_scores vs importance_labels.
# consistency_loss: 1 - cosine_similarity(retrieved, current_hidden).
# promotion_loss: REINFORCE — baseline-subtracted log-prob of hard gate * reward.
# All three must have the CORRECT sign. Check the formula in Part 3 doc.
```

#### T17 — Trainer (optimizer LR invariant)
```python
# Three optimizer groups with DIFFERENT learning rates:
#   main model params:     lr = 3e-4
#   promotion gate params:  lr = 1e-4  (slower)
#   surprise head params:   lr = 1e-5  (slowest)
# All use linear-warmup-then-cosine-decay.
# Do NOT use a single optimizer with a single LR.
```

---

## 7. RED LIGHTS — What You Must Do Yourself (or Review Line by Line)

### T28 — Ablation runner

**You write the significance test.** Paste this verbatim. Do not let any model invent it:

```python
def bootstrap_paired_pvalue(
    scores_a: list[float], scores_b: list[float],
    *,
    n_bootstrap: int = 10000,
    random_seed: int = 42,
) -> float:
    """Paired bootstrap on per-benchmark score pairs.
    H0: full_amc and baseline have equal median improvement across benchmarks.
    Returns one-tailed p-value.
    """
    rng = np.random.default_rng(random_seed)
    diffs = np.array([a - b for a, b in zip(scores_a, scores_b)])
    observed = np.mean(diffs)
    count = 0
    for _ in range(n_bootstrap):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        if np.mean(sample) >= observed:
            count += 1
    return count / n_bootstrap
```

Without this verbatim block, Composer/Opus/GLM will all write a
two-sample t-test, which is wrong (paired scores ≠ independent samples).

### T29 — Adversarial audit (6 probes)

**After any model generates the probes, manually verify each one:**

```python
# For each probe class:
# [ ] There is a line that calls the AMC API with a MALICIOUS payload
#       e.g., amc.store(key=..., value=forged_content, trust=TRUSTED)
# [ ] AFTER that line, there is an assert that the failure mode was BLOCKED
#       e.g., assert result.status == "BLOCKED"
# If the assert comes first → probe is fake
# If there is no malicious payload call → probe is fake
```

Models (all of them) write tests that LOOK like probes but skip the
attack call because the "test passes anyway" and looks structurally correct.
No model reliably catches this. You are the only reliable check.

### T32-T35 — The paper

**These are YOUR tranches.** Models can format LaTeX, tighten prose,
and align tables. They cannot decide:
- What the abstract's first sentence says
- Which results go in Table 1 vs Table 2
- What "novel" means in this context
- Whether to claim "surprise prediction is differentiable" or "memory is differentiable"

You write the claims, you choose the framing, you write the abstract,
you write the introduction. Use models for polish, not authorship.

---

## 8. Phase-Boundary Validation Gates — Pause Here Before Advancing

### After Phase 1 (T10 complete) — before starting Phase 3
```
Run: .venv/bin/python -m pytest tests/model/ tests/memory/ -v
Expect: all tests pass, no failures

Verify T00-T10 compile together:
  .venv/bin/python -c "import src.model; import src.memory; print('import clean')"

If either fails: do NOT start Phase 3. Fix Phase 1 first.
```

### After Phase 2 (T14 complete) — before starting Phase 3
```
Verify the checkpoint serializer round-trips a dummy state:
  .venv/bin/python -c "
  from src.memory.amc_tier2 import AMCTier2Hook
  from src.memory.amc_checkpoint import save_checkpoint, load_checkpoint
  t2 = AMCTier2Hook()
  t2.promote('test', 'hello world', confidence=0.9)
  save_checkpoint(t2, '/tmp/test_t2.msgpack')
  t2_loaded = load_checkpoint(AMCTier2Hook, '/tmp/test_tier2.msgpack')
  assert t2_loaded.retrieve('test').value == 'hello world'
  print('checkpoint round-trip OK')
  "
```

### After Phase 3 (T22 complete) — before starting Phase 5
```
Check that training completed:
  - logs/<run>/checkpoint-final.pt exists and is > 1 GB
  - logs/<run>/training.jsonl has > 1000 lines
  - Training loss at final step < 2.5 (for a 1B model, tokenized 10M tokens)
If any of these fail: do not start Phase 5.
```

### After Phase 5 (T31 complete) — before starting Phase 6
```
Verify all ablation configs have been trained and evaluated:
  docs/reproducibility/results/ablation_scores_YYYYMMDD_HHMMSS.jsonl exists
  It contains entries for all 4 configs × 5 benchmarks = 20 rows
  plot_ablation.py ran successfully and produced a figure
If not: do not start Phase 6. Your paper can't claim results you don't have.
```

---

## 9. Git Discipline

```
On the current branch: clean/amc-curation-20260521-101220
NEVER push to main. EVER.
NEVER merge branch into main until ALL 40 tranches are done.
```

Per tranche:
```bash
git add <only the files listed in the tranche's "Files to stage">
git commit -m "<exact commit message from tranche doc>"
git push origin clean/amc-curation-20260521-101220   # personal fork, safe
```

When all 40 tranches are done:
1. Open a PR from `clean/amc-curation-20260521-101220` → `main`
2. I (your reviewer) will review each commit individually
3. Squash-merge to main after all 40 are verified

---

## 10. How to Ask for Help

**Ping me when:**
- A yellow tranche fails its tests and you can't see why
- A model produces code that looks right but produces wrong numbers
- A phase boundary check fails and you're unsure if it's a real problem or a test bug
- You're unsure whether to diverge from a tranche spec
- Phase 3 training starts and you want me to monitor the first 1000 steps

**Don't ping me for:**
- "What's the next tranche?" — read `TRANCHE_STATUS.md`
- "Where is T01?" — it's in `AMC_FULL_BUILDOUT_PROMPTS.md`
- "Can you fix this lint error?" — run `ruff check --fix` yourself

---

## 11. Cost/Runway Reality Check

| Item | Cost | When |
|---|---|---|
| Composer 2.5 Pro | ~$20/mo | Already running |
| Opus 4.7 (Anthropic API) | ~$15-30 for red tranches | T28, T29, T32-T35 |
| GLM 5.1 (if you use it) | check your plan | As needed |
| GPU training (4×A100) | ~$200 | T20, one 8-day run |
| **Total remaining** | **~$250-300** | December 2026 deadline |

You have runway. The bottleneck is your review time, not money.

---

## 12. What "Done" Looks Like

At the end of this buildout you will have:

- [ ] `aurelius-forge-1b-amc` trained and benchmarked (4 configs × 5 benchmarks)
- [ ] Ablation study shows `full_amc` significantly outperforms `baseline` on memory tasks
- [ ] Standard benchmarks (GSM8K, MMLU) NOT degraded
- [ ] 6 adversarial probes ALL PASS
- [ ] LaTeX paper with all sections complete and figures generated
- [ ] arXiv preprint submitted
- [ ] Code + weights on Hugging Face
- [ ] PR merged to `main`

That's the finish line. We're at 5% (T00 done, S05 initialized).
38 tranches to go.

---

*Last updated: 2026-05-23 | Current state: T00 ✅ | Branch: clean/amc-curation-20260521-101220*
