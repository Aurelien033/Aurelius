# Tranche Status Tracker

Builds on: `docs/AMC_COMPLETE_BUILDOUT.md` + the four Prompt Documents.
Target: working Aurelius-Forge 1B-AMC model + paper by **December 2026**.

Legend
------
- ⬜ Not started
- 🚧 In progress
- ✅ Done
- ❌ Blocked (notes required)
- ⏸️ Paused

Update this file after each green commit. One line per tranche.

---

## Phase 0 — Foundation  (weeks 1-2)

| # | Tranche | Status | Commit | Tests | Notes |
|---|---------|--------|--------|-------|-------|
| T00 | Mamba-2 selective SSM block | ✅ | `07dd177f` | 30/31 (1 CUDA skip on CPU) | Sequential scan; correct A/B/C/Δ math (note divergence from Part 1 spec in commit body) |
| T01 | AMC Tier-1 SSM layer (surprise + gates) | ⬜ | — | — | Depends on T00 |
| T02 | Differentiable promotion gate (Gumbel-softmax) | ⬜ | — | — | Depends on T01 |
| T03 | SQLite-backed SDB persistent event log | ⬜ | — | — | Parallel-safe with T00-T02 |

## Phase 1 — Model Layer  (weeks 3-6)  ← the paper

| # | Tranche | Status | Commit | Tests | Notes |
|---|---------|--------|--------|-------|-------|
| T04 | Hybrid AMCTransformer (MLA + SSM, promotion gates) | ⬜ | — | — | Depends on T01+T02 |
| T05 | Standalone SurpriseHead + focal-loss pretrainer | ⬜ | — | — | Depends on T01 |
| T06 | AMCGateController (decay/erase/write) | ⬜ | — | — | Depends on T01 |
| T07 | Multi-head Latent Attention (DeepSeek-MLA) | ⬜ | — | — | |
| T08 | RMSNorm + RoPE (AMC-aware, no RoPE on SSM state) | ⬜ | — | — | |
| T09 | Parameter counter + config validator | ⬜ | — | — | Requires T04-T08 |
| T10 | Full model smoke test (checkpoint round-trip) | ⬜ | — | — | Requires T04-T09 |

## Phase 2 — Durable Runtime  (weeks 5-8, parallel with Phase 1)

| # | Tranche | Status | Commit | Tests | Notes |
|---|---------|--------|--------|-------|-------|
| T11 | Tier-2/3 checkpoint serialization (msgpack) | ⬜ | — | — | |
| T12 | State reconstruction engine (replay from events) | ⬜ | — | — | Depends on T03 |
| T13 | Trust-aware KV cache (serving integration) | ⬜ | — | — | |
| T14 | Crash recovery (WAL mode) | ⬜ | — | — | Depends on T11, T12 |

## Phase 3 — Training Pipeline  (weeks 9-14)

| # | Tranche | Status | Commit | Tests | Notes |
|---|---------|--------|--------|-------|-------|
| T15 | Three AMC losses (surprise, consistency, promotion) | ⬜ | — | — | Depends on T04-T10 |
| T16 | Importance-annotated data pipeline | ⬜ | — | — | |
| T17 | AMCTrainer with 3 optimizers + schedulers | ⬜ | — | — | Depends on T15, T16 |
| T18 | Forge-1B YAML config | ⬜ | — | — | |
| T19 | Data preparation (RedPajama sample tokenize+pack) | ⬜ | — | — | |
| T20 | Launch training (4xA100, DeepSpeed ZeRO-2) | ⬜ | — | — | ~$200 |
| T21 | Mid-training monitor + checkpoint | ⬜ | — | — | |
| T22 | Final evaluation (AMC-Memory, GSM8K, MMLU) | ⬜ | — | — | |

## Phase 4 — Agent Deepening  (weeks 13-18, parallel with Phase 3)

| # | Tranche | Status | Commit | Tests | Notes |
|---|---------|--------|--------|-------|-------|
| T23 | ConstitutionalMemory (permanent safety LTS) | ⬜ | — | — | |
| T24 | Reflect-and-consolidate agent step | ⬜ | — | — | Depends on T23 |
| T25 | Skill crystallizer | ⬜ | — | — | |
| T26 | SLR end-to-end integration | ⬜ | — | — | |
| T27 | Full agent e2e integration test | ⬜ | — | — | Depends on T24-T26 |

## Phase 5 — Validation  (weeks 19-24)

| # | Tranche | Status | Commit | Tests | Notes |
|---|---------|--------|--------|-------|-------|
| T28 | Full ablation (4 configs × 5 benchmarks) | ⬜ | — | — | Requires T22 + T27 |
| T29 | Adversarial memory safety audit (6 probes) | ⬜ | — | — | |
| T30 | Reproducibility bundle (full README flow) | ⬜ | — | — | |
| T31 | External cross-validation on clean machine | ⬜ | — | — | |

## Phase 6 — Paper  (weeks 25-30)

| # | Tranche | Status | Commit | Tests | Notes |
|---|---------|--------|--------|-------|-------|
| T32 | Paper outline + abstract (LaTeX skeleton) | ⬜ | — | — | |
| T33 | Method section (math for surprise/gate/trust) | ⬜ | — | — | |
| T34 | Experiments section (ablation tables + figures) | ⬜ | — | — | Depends on T28 |
| T35 | Final assembly, arXiv preprint, HF weights | ⬜ | — | — | |

## Support infrastructure

| # | Tranche | Status | Commit | Tests | Notes |
|---|---------|--------|--------|-------|-------|
| S01 | DeepSpeed ZeRO-2 config | ⬜ | — | — | |
| S02 | `scripts/train.sh` launcher | ⬜ | — | n/a | |
| S03 | Hugging Face model card | ⬜ | — | — | |
| S04 | `scripts/monitor_training.py` | ⬜ | — | — | |
| S05 | This tracker document | ✅ | — | n/a | Initialized with T00 |

---

## Running totals

| Metric | Count |
|--------|-------|
| Total tranches | 40 (35 T + 5 S) |
| ✅ Done | 2 (T00, S05) |
| 🚧 In progress | 0 |
| ❌ Blocked | 0 |
| ⬜ Not started | 38 |
| Completion | 5 % |

Last updated: 2026-05-23 — T00 committed on branch `clean/amc-curation-20260521-101220`

---

## Divergence notes

- **T00 Mamba-2**: implemented with proper A/B/C/Δ ZOH discretization
  instead of the simplified recurrence in the Part 1 doc. State shape
  is `(B, nheads, headdim, d_state)` so T01 can wrap it as the per-layer
  Tier-1 working memory with no API changes. Sequential scan (not the
  associative parallel one from `mamba_ssm`) for clarity; swappable
  later without changing the public API.
- **Tests**: 31 written, 30 pass, 1 CUDA device test is correctly
  skipped on CPU-only runners.

---

## Policy

1. One atomic commit per tranche. Use the commit message from the
   tranche doc verbatim.
2. Never stage files outside the tranche's "Files to stage" list.
3. Never advance to Tn+1 while Tn has failing tests.
4. Never push to main. Branch work lives on
   `clean/amc-curation-20260521-101220` until the full buildout
   completes, then a curated merge.
5. Update this document in the same commit (for S05 init) or the
   tranche commit itself.
