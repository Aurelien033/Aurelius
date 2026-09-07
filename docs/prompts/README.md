# AMC Full Buildout — Sequential Agent Prompts

This directory contains a complete, ordered set of prompts to build
the Aurelian Memory Core (AMC) from scratch. The goal: train a 1B
AMC model, run the ablation study, and prepare a research paper by
**December 2026**.

## Files

| File | Size | Contents |
|------|------|----------|
| `AMC_FULL_BUILDOUT_PROMPTS.md` | ~73 KB | Part 1: T00–T04, T11, T15–T16 partial, T17–T35 stubs. Master timeline, cost, dependency graph. |
| `AMC_FULL_BUILDOUT_PROMPTS_PART2.md` | ~79 KB | Part 2: Detailed T05–T10 (model completion), T12–T14 (runtime completion), T17–T22 training runbook, T24–T25, T28–T31. |
| `AMC_FULL_BUILDOUT_PROMPTS_PART3.md` | ~87 KB | Part 3: T08–T10 full implementations (RMSNorm, RoPE, param counter), T12–T14 full implementations, T15–T16 full loss and data pipelines, T26–T27, supporting infra (S01–S05). |
| `AMC_FULL_BUILDOUT_PROMPTS_PART4.md` | ~92 KB | Part 4: Full Aurelius-Forge 1B config, data prep scripts, training launcher, ConstitutionalMemory full impl, 3 ablation baseline configs, shell scripts (train/eval/ablation/plot), JSONL logger, full trainer, LaTeX paper skeleton. |
| `TRANCHE_STATUS.md` | live doc | Progress tracker (update as you go) |

**Total:** ~330 KB of executable buildout documentation + live tracker.

---

## Getting Started

### Prerequisites

```bash
cd /Users/christienantonio/aurelius
source .venv/bin/activate

# Verify dependencies
python -c "import torch, einops, numpy, pyyaml, msgpack, tqdm; print('deps OK')"

# Verify existing tests pass
python -m pytest tests/ -q --tb=no 2>&1 | tail -3
```

### Day 1: Start T00

Open `AMC_FULL_BUILDOUT_PROMPTS.md` and jump to **Tranche T00**.
T00 is the Mamba-2 selective state space block — the foundation
for everything else.

Read the full tranche, write the code (`src/model/mamba2_block.py`
and `tests/model/test_mamba2_block.py`), run validation, commit.

Then move to T01.

---

## How to Use

### Option A: Human-driven (recommended for first pass)

1. Open the relevant Part file
2. Read the tranche entirely. Understand the goal.
3. Write the code yourself using the spec as a guide.
4. Run the validation commands at the bottom.
5. If green, commit and move to next tranche.
6. Update `TRANCHE_STATUS.md` after each commit.

### Option B: Agent-driven (for parallel/accelerated work)

1. Copy the ENTIRE tranche (from "### 1. Create..." to the end of
   "### Acceptance criteria:").
2. Paste into Claude or another code agent with the Aurelius repo
   mounted as a codebase.
3. Agent writes code → you review validation → you commit.
4. Move to next tranche.

### Option C: Hybrid

- Use **Option A** for model layer (T00–T10, the novelty) — these
  need human judgment to shape correctly.
- Use **Option B** for infrastructure (T11–T14, T16–T17, scripts)
  — these are mechanical implementations.

---

## Tranche Dependencies

```
T00 → T01 → T02 → T04 → T15 → T17 → T18-T22 (train)
                                        ↓
T03 → T11 → T14 ────────────────────────T28 (ablation) → T32 (paper)
         ↓
      T12 → T23 (constitutional) → T24-T27 (agent)
```

**Parallelizable:**
- Phase 0 (T00–T03) can all be done in parallel across sessions
- Phase 1 (T04–T10) requires T01+T02
- Phase 2 (T11–T14) can run parallel with Phase 1
- Phase 3 (T15–T22) requires Phase 1 + Phase 2 complete
- Phase 4 (T23–T27) requires Phase 2 + Phase 3
- Phase 5–6 (T28–T35) strictly sequential at the end

---

## Critical Path (Minimum Viable Paper)

If you are time-constrained, this is the smallest set that produces
a publishable result:

```
T00 → T01 → T02 → T04 → T17 → T18-T22 → T28 → T32
```

This gives you:
- Mamba-2 block + SSM layer + promotion gate + AMCTransformer
- A trained 1B model (or scale down to 150M for proof of concept)
- An ablation study showing full_amc > baseline
- A paper outline + LaTeX skeleton

Everything else (T03, T11–T16, T23–T27, T29–T31) makes the paper
stronger but is not blocking for a first submission.

---

## Target Timeline

| Phase | Week Range | Duration |
|-------|------------|----------|
| Phase 0 (foundation)  | 1–2   | 2 weeks |
| Phase 1 (model layer) | 3–6   | 4 weeks |
| Phase 2 (runtime)     | 5–8   | 4 weeks (parallel w/ P1) |
| Phase 3 (training)    | 9–14  | 6 weeks |
| Phase 4 (agent)       | 13–18 | 6 weeks (parallel w/ P3) |
| Phase 5 (validation)  | 19–24 | 6 weeks |
| Phase 6 (paper)       | 25–30 | 6 weeks |
| **Total**             |       | **~30 weeks** |

Starting May 2026 → completion ~December 2026.

**Paper submission targets:**
- **ICLR 2027** (deadline typically late September)
- **NeurIPS 2027** (deadline typically mid-May)
- arXiv preprint any time (establishes priority)

---

## Cost Estimate

| Item | Cost |
|------|------|
| 4×A100 training (~8 days) | ~$200 |
| A100 inference/eval | ~$50 |
| Hugging Face hosting | $0 (free tier) |
| arXiv preprint | $0 |
| **Total** | **~$250** |

---

## When You Get Stuck

1. **A test fails.** Read the failure message carefully. Fix the
   specific test. Re-run the entire validation block.

2. **A dependency is missing.** Add it to `pyproject.toml`, run
   `uv sync` or `pip install`, re-run.

3. **A design decision.** Re-read the corresponding section in
   `docs/AMC_COMPLETE_BUILDOUT.md` (the master architecture plan) —
   every tranche references it.

4. **The paper needs framing.** Read the abstract in Part 4's
   LaTeX skeleton (`paper/main.tex`). The claims listed are exactly
   what the tranches produce. If you're missing a tranche, you're
   missing a claim.

5. **Agent produces wrong code.** The tranche may be ambiguous.
   Fix the ambiguity *in the tranche text itself* so future
   sessions don't repeat the problem.

---

## Final Goal

By December 2026:

- [ ] Aurelius-Forge-1B-AMC is trained and benchmarked
- [ ] Ablation study shows statistically significant improvement
      of `full_amc` over `baseline` on memory tasks
- [ ] Standard benchmarks (GSM8K, MMLU) are NOT degraded
- [ ] 6 adversarial security probes all PASS
- [ ] Paper written with LaTeX, figures, and full ablation tables
- [ ] arXiv preprint published (establishes priority)
- [ ] Code + weights released (Hugging Face + GitHub)
- [ ] Submitted to ICLR 2027 or NeurIPS 2027

---

## Quick-Start Checklist

Run these commands right now to confirm the repo is ready for T00:

```bash
cd /Users/christienantonio/aurelius
source .venv/bin/activate

# 1. Verify Python version
python --version    # Should be 3.12+

# 2. Install any missing deps
pip install einops>=0.8 msgpack>=1.0 tqdm pyyaml

# 3. Verify existing tests pass (baseline)
python -m pytest tests/ -q --tb=no 2>&1 | tail -3
# Expected: "... passed, ... skipped" with 0 failed

# 4. Create a working branch
git checkout -b feat/amc-buildout-2026

# 5. Open Part 1, Tranche T00
open docs/prompts/AMC_FULL_BUILDOUT_PROMPTS.md
```

---

## Tracking Progress

After each successful tranche:

1. ✅ **All acceptance criteria pass** (green pytest, green smoke test)
2. ✅ **Ruff/lint clean** on new code
3. ✅ **One clean commit** with the exact commit message from the tranche
4. ✅ **`TRANCHE_STATUS.md` updated** (mark ✅)
5. Push to your `feat/amc-buildout-2026` branch (not main). When
   the full buildout is done, do a curated merge to main.

---

## Notes on Novelty

For reference, `docs/AMC_NOVELTY_ASSESSMENT.md` (produced earlier
in this session) documents what is genuinely novel about AMC:

1. **Per-layer differentiable 3-tier hierarchy** — the transition
   between T1→T2→T3 is learnable. Not in Titanic, MemGPT, or
   Generative Agents.

2. **Trust-aware memory contract** with cache-identity binding —
   same content + different trust state = different cache key.
   Prevents silent poisoning.

3. **Constitutional memory alignment** — safety principles as
   permanent, non-evictable LTS entries always retrieved during
   generation. Alignment via retrieval, not via a separate
   classifier or reward term.

The novelty window is ~12–18 months. Meta's Titans team will likely
extend it in the next year. MemGPT, Zep/Graphiti, and RAG
frameworks will close it from below.

**Train the model. Publish the numbers. Establish priority.**

---

## Directory Structure

```
docs/
├── AMC_COMPLETE_BUILDOUT.md              # Master architecture plan
├── AMC_NOVELTY_ASSESSMENT.md             # (produced in this session)
└── prompts/
    ├── README.md                         # This file
    ├── TRANCHE_STATUS.md                 # Live progress tracker
    ├── AMC_FULL_BUILDOUT_PROMPTS.md      # Part 1 (~73 KB)
    ├── AMC_FULL_BUILDOUT_PROMPTS_PART2.md # Part 2 (~79 KB)
    ├── AMC_FULL_BUILDOUT_PROMPTS_PART3.md # Part 3 (~87 KB)
    └── AMC_FULL_BUILDOUT_PROMPTS_PART4.md # Part 4 (~92 KB)
```

---

## Credits

This buildout plan was drafted by Hermes Agent on 2026-05-23,
informed by:
- The existing Aurelius codebase (5,365 files, 572K LoC)
- 16 AMC-related Python modules already implemented (contracts only)
- `aurelius-repo-repair` skill with 38 reference documents
- Research papers: Mamba-2 (Dao & Gu 2024), Titans (Meta 2025),
  MemGPT (Packer 2023), Generative Agents (Park 2023),
  Constitutional AI (Bai 2022), RAG (Lewis 2020),
  DPO (Rafailov 2023), GRPO (Shao 2024)
- The user's target: **train a model and publish the paper by
  December 2026**

---

## Execute

This is a real research contribution. You're building something
that doesn't exist yet in the literature: a per-layer
differentiable 3-tier memory hierarchy with trust-aware contracts
and constitutional alignment, all inside the transformer forward
pass.

The novelty window is open. Train the model. Publish the results.

**Start at T00. Execute.**
