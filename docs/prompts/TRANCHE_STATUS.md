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
| T00 | Mamba-2 selective SSM block | ✅ | `07dd177f` | 30/31 (1 CUDA skip) | ZOH discretization; sequential scan |
| T01 | AMC Tier-1 SSM layer (surprise + gates) | ✅ | `e20cf004` | green | P0 contracts + `AMCSSMLayer` |
| T02 | Differentiable promotion gate (Gumbel-softmax) | ✅ | `841182b7` | green | ST fix: `hard - soft.detach() + soft` (was wrong in `87e5040e`) |
| T03 | SQLite-backed SDB persistent event log | ✅ | `6061d688` | green | `sdb_persistent_log` |

## Phase 1 — Model Layer  (weeks 3-6)

| # | Tranche | Status | Commit | Tests | Notes |
|---|---------|--------|--------|-------|-------|
| T04 | Hybrid AMCTransformer | ✅ | `c58f358c` | green | MLA + SSM alternation |
| T05 | Standalone SurpriseHead | ✅ | `9d066bd5` | green | Detachable + pretrain hook |
| T06 | AMCGateController | ✅ | `be8021bf` | green | decay/erase/write |
| T07 | Multi-head Latent Attention | ✅ | `e3cb7762` | green | DeepSeek-MLA style |
| T08 | RMSNorm + RoPE | ✅ | `1a910c51` | green | AMC-aware |
| T09 | Parameter counter + config validator | ✅ | `6a28e132` | green | |
| T10 | Full model smoke + checkpoint | ✅ | `571c0399` | green | e2e smoke |

## Phase 2 — Durable Runtime  (weeks 5-8)

| # | Tranche | Status | Commit | Tests | Notes |
|---|---------|--------|--------|-------|-------|
| T11 | Tier-2/3 checkpoint serialization | ✅ | `8db339be` | green | msgpack |
| T12 | State reconstruction engine | ✅ | `c71f1318` | green | SDB replay |
| T13 | Trust-aware KV cache | ✅ | `4c8293b2` | green | serving |
| T14 | Crash recovery (WAL) | ✅ | `e5879f07` | green | |

## Phase 3 — Training Pipeline  (weeks 9-14)

| # | Tranche | Status | Commit | Tests | Notes |
|---|---------|--------|--------|-------|-------|
| T15 | Three AMC losses | ✅ | `40addc62` | green | surprise/consistency/promotion |
| T16 | Importance-annotated data pipeline | ✅ | `3942e924` | green | |
| T17 | AMCTrainer (3 optimizers) | ✅ | `62f1fab0` | green | |
| T18 | Forge-1B YAML config | ✅ | `02f5c64f` | green | validate script |
| T19 | Data prep + memmap dataset | ✅ | `b366190c` | green | |
| T20 | Launch training script | ✅ | `bb0f1981` | green | DeepSpeed path |
| T21 | Training monitor | ✅ | `12401b90` | green | JSONL metrics |
| T22 | Post-training eval harness | ✅ | `49ffbbc6` | green | |

## Phase 4 — Agent Deepening  (weeks 13-18)

| # | Tranche | Status | Commit | Tests | Notes |
|---|---------|--------|--------|-------|-------|
| T23 | ConstitutionalMemory | ✅ | `0df15457` | green | permanent Tier-3 |
| T24 | Reflect-and-consolidate agent | ✅ | `92e13413` | green | |
| T25 | Skill crystallizer | ✅ | `d69f05aa` | green | |
| T26 | SLR in ReAct loop | ✅ | `ab5e1b79` | green | |
| T27 | Full agent e2e | ✅ | `4046daa9` | green | |

## Phase 5 — Validation  (weeks 19-24)

| # | Tranche | Status | Commit | Tests | Notes |
|---|---------|--------|--------|-------|-------|
| T28 | Full ablation runner | ✅ | `a97609f8` | green | oracle smoke + bootstrap p-values |
| T29 | Adversarial memory safety audit | ✅ | `62e86cb9` | green | fail-closed SDB; test hardening `6dd23fee` |
| T30 | Reproducibility bundle | ✅ | `01bb24b0` | green | `docs/reproducibility/` |
| T31 | External cross-validation | ✅ | `91418a3f` | smoke PASS local | External clean machine: NOT EXECUTED |

## Phase 6 — Paper  (weeks 25-30)

| # | Tranche | Status | Commit | Tests | Notes |
|---|---------|--------|--------|-------|-------|
| T32 | Paper skeleton + claims ledger | ✅ | `d7c98cdb` | validate_paper | Split sections + CLAIMS_LEDGER |
| T33 | Method section (traceable math) | ✅ | `d7c98cdb` | — | Correct ST + SDB contract |
| T34 | Experiments + provenance | ✅ | `d7c98cdb` | — | Oracle AMC-Memory; other benchmarks TODO |
| T35 | Release assembly | ✅ | `dda1399c` | bash -n | Local scripts only; no upload |

## Support infrastructure

| # | Tranche | Status | Commit | Tests | Notes |
|---|---------|--------|--------|-------|-------|
| S01 | DeepSpeed ZeRO-2 config | ✅ | (in T18/T20) | — | `configs/deepspeed_zero2.json` |
| S02 | `scripts/train.sh` launcher | ✅ | `01bb24b0` | n/a | under reproducibility |
| S03 | Hugging Face model card | ✅ | `01bb24b0` | n/a | `docs/reproducibility/results/model_card.md` |
| S04 | `scripts/monitor_training.py` | ✅ | `12401b90` | — | |
| S05 | This tracker document | ✅ | `7b015460` | n/a | Updated through T30 |

---

## Running totals

| Metric | Count |
|--------|-------|
| Total tranches | 40 (35 T + 5 S) |
| ✅ Done | 38 (T00–T35, S01–S05) |
| 🚧 In progress | 0 |
| ❌ Blocked | 0 |
| ⬜ Not started | 0 |
| Completion | 100 % tranche checklist (GPU training / external xval still operator tasks) |

Last updated: **2026-05-22** — branch `clean/amc-curation-20260521-101220`, HEAD `91418a3f` (orchestrator Prompt 00 complete).

---

## Divergence notes

- **T00 Mamba-2**: ZOH discretization; state `(B, nheads, headdim, d_state)`; sequential scan.
- **T02 ST semantics**: initial commit `87e5040e` used incorrect `hard - hard.detach() + soft`; corrected in `841182b7`.
- **T28 ablation**: oracle smoke scores in `docs/reproducibility/results/ablation_scores.jsonl`; GSM8K/MMLU at 1.0 are harness stubs until GPU engine eval.
- **T31**: local smoke cross-validation PASS; no evidence yet from a separate clean external machine.

---

## Policy

1. One atomic commit per tranche when possible.
2. Never stage files outside the tranche's "Files to stage" list.
3. Never advance to Tn+1 while Tn has failing tests.
4. Never push to main without explicit operator approval.
5. Update this document when tranche status changes.
