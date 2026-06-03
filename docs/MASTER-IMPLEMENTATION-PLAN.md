# MASTER IMPLEMENTATION PLAN — Aurelius Research Platform

> **Version:** 1.0 — 2026-05-28  
> **Branch:** `clean/amc-curation-20260521-101220` (20+ commits ahead of `main`)  
> **Ground-truth baseline:** commit `3a759891` (DreamBank MVP hardened)

This document is the single source of truth for the Aurelius roadmap. Individual feature plans live in `docs/plans/` — link to them here rather than duplicating content. Agents executing tasks should read the linked plan file for the tranche specification, then use this document to understand merge rules and phase gates.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Current State — Ground Truth](#2-current-state--ground-truth)
3. [Merge Strategy](#3-merge-strategy)
4. [Active Tracks](#4-active-tracks)
5. [Phase Roadmap](#5-phase-roadmap)
6. [Agent Execution Rules](#6-agent-execution-rules)
7. [Done Criteria](#7-done-criteria)

---

## 1. Project Overview

**Aurelius** is a 1.395B decoder-only transformer built from scratch — no HuggingFace Transformers, no flash-attn runtime, no bitsandbytes at inference.

| Layer | Location | Language | Role |
|-------|----------|----------|------|
| Rust Engine | `crates/` | Rust 2024 | Tokenization, search, vector similarity, session management |
| Python Backend | `src/`, `agent/`, `gateway/` | Python 3.12 | Model, training, inference, alignment, API, CLI |
| Node.js BFF | `middle/` | TypeScript | Auth, rate limiting, WebSocket, SSE, cron |
| Frontend | `frontend/` | React 19 + TypeScript | Mission Control: dashboard, chat, analytics, admin |

**Model:** 24 layers, d_model=2048, 16 Q heads / 8 KV heads (GQA), SwiGLU FFN, RoPE θ=500k.

**Active research tracks:**
- Track A: Core model + inference (AMC transformer, memory systems)
- Track B: Alignment stack (DreamBank → CascadeBank → CB-07 → APEX)
- Track C: Serving + API surface
- Track D: HLM architecture exploration (separate repo)

---

## 2. Current State — Ground Truth

### 2.1 Branch State

```
Branch: clean/amc-curation-20260521-101220
HEAD:   46ee2f13  memory(debate): LLM voices for proposer/skeptic/judge
main:   5d95d445  Merge pull request #184 (vitest bump)
Delta:  20 commits ahead of main
```

The current branch is **not merged to main**. It contains all research work since the AMC curation session. All new development should branch from `HEAD`, not `main`.

### 2.2 Completed Work (on this branch)

| Commit | Tag | Description |
|--------|-----|-------------|
| `a02469d2` | DreamBank MVP | HLMPreferenceBank — preference-weighted memory injection |
| `3a759891` | DreamBank hardened | Gate, metadata, gradients, shape, restore contracts |
| `97ccc1dd` | CascadeBank | Zero-parameter routing on DreamBank gate signals |
| `72a99589` | CB-06 harness | Greedy decode + per-policy cost-proxy serving harness |
| `36f2a979` | Federated memory | FedAvg on DreamBank tensors (delta-level federation) |
| `2651b35b` | DP proof | (ε,δ)-DP Gaussian mechanism proof for federated deltas |
| `5a372fd1` | TrustRAG | Quarantine-aware, trust-bound retrieval controller |
| `fb4a3c61` | Memory debate | Adjudicated Tier-3 promotion via proposer/skeptic/judge |
| `2910a2a4` | Per-layer MLA | `hlm_bank_read_layers` wiring into AMCTransformer |
| `65e60019` | TrustRAG v2 | Semantic contradiction via injectable `contradiction_fn` |
| `46ee2f13` | Memory debate v2 | LLM voices for proposer/skeptic/judge via configurable API |
| `b70a0751` | Security | Fail-closed rate limiter + authenticated `/metrics` |
| `bf074470` | Security | Hardened code runner with regex-based sandbox denylist |
| `b2bb33b5` | Security | `check_torch_load` hook registered in pre-commit |

### 2.3 Dirty Working Tree (Uncommitted)

The working tree has ~20 modified files that are not yet committed. **These must be handled before any new branch is created or any merge is attempted.** Do not bundle them into a feature commit — audit each file first:

```bash
git status --short
```

Files modified include: `agent/__init__.py`, `agent/session_manager.py`, `gateway/aurelius_api.py`, `middle/src/config.ts`, `middle/src/provider_router.ts`, `middle/src/routes/auth.ts`, `middle/src/server.ts`, `src/alignment/simpo.py`, `src/memory/amc_tier2.py`, `src/model/__init__.py`, `src/model/moe.py`, `src/serving/aurelius_server.py`, `src/training/trainer.py`, `src/ui/session_manager.py`, `tests/integration/test_lambda_attention_integration.py`, `.github/workflows/ci.yml`.

**Protocol before any new commit:**
1. Run `make test` to confirm the test suite is still green.
2. For each dirty file, decide: belongs to a named in-flight feature → commit with that feature's message; or is pre-existing drift → stage separately with `chore: resolve working-tree drift`.
3. Never bundle drift files into a research feature commit.

---

## 3. Merge Strategy

### 3.1 Merging this Branch to Main

The branch is 20+ commits ahead of `main`. Main is at a trivial dependency bump (`5d95d445`). The correct merge path:

```bash
# 1. Resolve dirty tree (see §2.3)
git add <specific files>
git commit -m "chore: resolve working-tree drift pre-merge"

# 2. Rebase onto main to linearize history (preferred over merge commit)
git fetch origin
git rebase origin/main

# 3. Resolve any conflicts (main is trivially behind — conflicts unlikely)

# 4. Run full test suite
make test

# 5. Push and open PR
git push -u origin clean/amc-curation-20260521-101220
gh pr create --base main --title "feat: AMC curation — DreamBank + CascadeBank + TrustRAG + Federated Memory"
```

**Do not force-push to main.** Open a PR and let CI run.

### 3.2 Future Branch Protocol

All new features branch from the current HEAD of this branch (not from main until it's merged):

```bash
git checkout -b feat/<feature-name>
```

One feature per branch. Each branch must:
- Add only files listed in its plan's "Stage only" section.
- Leave pre-existing dirty/untracked files untouched.
- Pass the full DreamBank + CascadeBank focused test suite before merging.

### 3.3 Merge Order

Features must merge in this order to avoid conflicts:
1. `chore/resolve-dirty-tree` — unblock all subsequent merges
2. `feat/cb-07-gpu-serving` — depends on CB-06 harness (already committed)
3. `feat/apex-modality-router` — depends on existing alignment modules
4. `feat/apex-credit-engine` — depends on APEX modality router
5. `feat/apex-arch-optimizer` — depends on credit engine
6. `feat/serving-bff-routes` — depends on gateway API surface
7. `feat/hlm-novel-architecture` — independent, can run in parallel

---

## 4. Active Tracks

### Track A — AMC Model + Memory Stack

**Status:** Implemented and tested through the per-layer MLA bank.

**Key files:**
- `src/model/amc_transformer.py` — AMCTransformer (read-only consumer for downstream features)
- `src/model/hlm_bank_adapter.py` — HLMPreferenceAdapter (read-only)
- `src/memory/hlm_bank.py` — HLMPreferenceBank (read-only)
- `src/alignment/dreambank.py` — DreamBankController
- `src/inference/cascade_routing.py` — CascadeRouter + ComputePolicy
- `src/eval/serving_harness.py` — CB-06 greedy decode harness

**API surface that must not change** (frozen contracts):
- `AMCModelOutput.bank_alpha: torch.Tensor | None` — shape `(B, T, 1)`
- `AMCModelOutput.bank_confidence: torch.Tensor | None` — shape `(B, T, 1)`
- `AMCModelOutput.bank_telemetry: dict[str, float | int] | None`
- `AMCTransformerConfig.use_hlm_bank: bool`
- `AMCTransformerConfig.hlm_bank_inject_scale: float`
- `AMCTransformerConfig.hlm_bank_read_layers: list[int] | None`

**Next:** CB-07 (real MT-Bench/AlpacaEval + GPU measurement). Plan: `docs/plans/2026-05-27-cb06-serving-harness.md` §"Next Phase After MVP".

---

### Track B — Alignment Stack

**Status:** DreamBank → CascadeBank → CB-06 harness → Federated Memory → TrustRAG → Memory Debate all committed. APEX is **not yet started**.

**Implemented alignment modules (composable, do not rewrite):**
- `src/alignment/orpo.py`, `grpo_v3.py`, `reinforce_pp.py`, `prime.py`, `kto.py`, `spin.py`, `constitutional_ai_v3.py`
- `src/alignment/sapo.py`, `aem.py`, `tur_dpo.py`
- `src/alignment/praxis/mtah.py`, `praxis/steering_reward.py`, `praxis/expert_safety_affinity.py`
- `src/alignment/dapo.py`, `warp.py`
- `src/alignment/dreambank.py` — DreamBank controller (Tier-3 promotion)
- `src/alignment/trustrag/` — TrustRAG quarantine-aware retrieval

**Alignment scope (canonical):** See `docs/ALIGNMENT_SCOPE.md`. Five tracks: SFT, DPO, GRPO/RLVR, Constitutional memory, Red-team quarantine.

**APEX — not started:**
- Target path: `src/alignment/apex/`
- Design note: Three-tier (Modality Router → Credit Engine → Architecture-Aware Optimizer). Supersedes PRAXISTrainer and MOSAIC v2.
- All implementations must compose existing modules; no rewrites.
- Paper target: NeurIPS/ICML 2027.
- **Before starting APEX**: write `docs/plans/YYYY-MM-DD-apex-design.md` and get architecture validated.

---

### Track C — Serving + API Surface

**Status:** Gateway files exist (`gateway/`) but HTTP API surface is not unified. BFF routes in `middle/src/routes/` are partially implemented.

**Gap (from GAP_LEDGER.md):**
- `middle/src/routes/` — `chat.ts`, `registry.ts`, `brain.ts`, `provider_router.ts` missing or incomplete.
- No unified Docker/compose for production serving.
- No CI workflow in `.github/` that runs the full Python + Node.js stack together.

**Serving layer key files:**
- `gateway/aurelius_api.py` — main Python API server
- `gateway/router_pareto.py` — Pareto frontier router (done)
- `src/serving/aurelius_server.py` — server entrypoint
- `middle/src/server.ts` — Node.js BFF

**Next priority:** Resolve dirty-tree changes in `gateway/aurelius_api.py`, `middle/src/server.ts`, and `middle/src/routes/` first, then implement missing BFF routes.

---

### Track D — HLM Architecture Research

**Status:** Separate repo at `~/Desktop/HLM-Architecture-Exploration/`. Not in this repository. Canonical research doc: `architecture_research_v4.md`.

**This track is independent** and does not block any Aurelius merge. Novel proposals (H-MoE, GARB, CLX) are working concepts — not yet ready for Aurelius integration.

**Integration point:** When HLM architectural decisions mature, they will be incorporated into Aurelius as a new model architecture branch, separate from the AMCTransformer stack.

---

## 5. Phase Roadmap

### Phase 1 — Stabilization (Current priority)

**Goal:** Get the branch mergeable. No new features.

| Task | File(s) | Notes |
|------|---------|-------|
| Audit dirty tree | all modified files | Run `git status --short`, categorize each |
| Commit drift as `chore:` | see §2.3 | One commit per logical group |
| Run full test suite | `make test` | Must be green before PR |
| Open PR to main | — | Use `gh pr create` |

**Acceptance:** CI green, PR open, no force-push.

---

### Phase 2 — CB-07: Real Serving Metrics

**Goal:** Replace CB-06 proxy metrics with real MT-Bench/AlpacaEval scores and GPU latency.

**Depends on:** CB-06 harness at `72a99589` (done).

| Task | File(s) | Notes |
|------|---------|-------|
| Integrate MT-Bench runner | `src/eval/mt_bench.py` | Wire into serving harness |
| Integrate AlpacaEval scorer | `src/eval/alpaca_eval.py` | Prefer programmatic scoring |
| GPU FLOPs measurement | `src/eval/serving_harness.py` | Add `torch.cuda.profiler` |
| Real p50/p95 latency per policy class | `src/eval/serving_harness.py` | Timed with `time.perf_counter_ns` |
| Paper-grade threshold check | `tests/eval/test_serving_harness_gpu.py` | THOROUGH ≥ +3pp; FAST ≤ -1pp, -10% p95 |

**Plan doc:** Create `docs/plans/YYYY-MM-DD-cb07-gpu-serving.md` before starting.

**Success gate:**
- THOROUGH policy improves win rate vs BALANCED by ≥ 3pp on MT-Bench.
- FAST policy degrades win rate by ≤ 1pp but saves ≥ 10% p95 latency.
- If gate not met: pivot to "train bank for router-friendly signal distributions" before wiring more serving code.

---

### Phase 3 — APEX Alignment System

**Goal:** Grand-unified alignment trainer superseding PRAXISTrainer and MOSAIC v2.

**Depends on:** All Phase 1 + 2 gates green.

**Architecture (three tiers, compose not rewrite):**

```
Tier 1: Modality Router
  Input: preference pairs / scalar / binary / self-play / constitutional
  Route: ORPO / GRPO / KTO / SPIN / CAI
  Merge: PrecisionFusion (inverse-variance weighting)

Tier 2: Credit Engine
  SAPO segment decomposition + AEM entropy gating + TUR-DPO topology
  Output: hierarchical A(t) per token

Tier 3: Architecture-Aware Optimizer
  MTAH temporal extension + SRC + ESA + constitutional gate + WARP
  Output: final gradient signal
```

**Implementation path:**

| Task | Target file | Depends on |
|------|-------------|------------|
| Write design doc | `docs/plans/YYYY-MM-DD-apex-design.md` | Nothing |
| Config dataclass | `src/alignment/apex/config.py` | Design doc approved |
| Modality router | `src/alignment/apex/modality_router.py` | Config |
| Credit engine | `src/alignment/apex/credit_engine.py` | Modality router |
| Architecture optimizer | `src/alignment/apex/arch_optimizer.py` | Credit engine |
| APEX loss | `src/alignment/apex/apex_loss.py` | All tiers |
| Curriculum | `src/alignment/apex/curriculum.py` | APEX loss |
| SPIN generator | `src/alignment/apex/spin_generator.py` | Curriculum |
| Value head | `src/alignment/apex/value_head.py` | APEX loss |
| Trainer | `src/alignment/apex/trainer.py` | All above |
| Tests | `tests/alignment/test_apex_*.py` | Each module |

**Composable imports (no rewrites):**
- Tier 1: `orpo.py`, `grpo_v3.py`, `reinforce_pp.py`, `prime.py`, `kto.py`, `spin.py`, `constitutional_ai_v3.py`
- Tier 2: `sapo.py`, `aem.py`, `tur_dpo.py`
- Tier 3: `praxis/mtah.py`, `praxis/steering_reward.py`, `praxis/expert_safety_affinity.py`, `dapo.py`, `warp.py`

**Curriculum:** ARIA (0–1000 steps) → AURORA (1000–4000) → APEX (4000+).

**Paper evidence required:** Every prior method (DPO/GRPO/KTO/SPIN/ORPO/SAPO/TUR-DPO) must be a reproducible ablation of APEX. Add ablation harness at `src/eval/apex_ablation.py`.

---

### Phase 4 — Serving & BFF Surface

**Goal:** Unified production serving with complete BFF route coverage.

**Depends on:** Phase 1 (clean branch) + dirty-tree `middle/` files resolved.

| Task | File(s) | Notes |
|------|---------|-------|
| Add missing BFF chat route | `middle/src/routes/chat.ts` | See GAP_LEDGER §High Priority |
| Add BFF registry route | `middle/src/routes/registry.ts` | Agent/skill registry exposure |
| Add BFF brain route | `middle/src/routes/brain.ts` | BrainBridge integration |
| Wire provider router | `middle/src/provider_router.ts` | Route to local/cloud backends |
| Docker compose for full stack | `docker-compose.prod.yml` | Rust + Python + Node + Frontend |
| CI workflow for full stack | `.github/workflows/ci.yml` | Python tests + TypeScript tests + build |

---

### Phase 5 — HLM Architecture Integration

**Goal:** Incorporate novel architectural insights from HLM research into Aurelius.

**Depends on:** HLM design decisions finalized in `~/Desktop/HLM-Architecture-Exploration/architecture_research_v4.md`.

**This phase has no committed tasks yet.** When the user finalizes HLM architectural decisions, add tranches here following the same TDD + file-ownership protocol as prior phases.

**Coordination:** The HLM proposals (H-MoE, GARB, CLX) are the user's own working concepts. Treat them as first-party research, not external references.

---

## 6. Agent Execution Rules

These invariants apply to every agent executing tasks from this plan:

1. **Read before writing.** Run `git log -1 --oneline` to confirm HEAD before any commit.
2. **Stage only named files.** Each task's "Stage only" section is exhaustive. Do not `git add -A`.
3. **TDD required for all new production code.** Write failing test first. Run it. Implement minimum passing code. Re-run focused tests. Only then consider refactoring.
4. **Frozen contracts must not change.** The `AMCModelOutput`, `HLMPreferenceBank`, and `AMCTransformer` interfaces listed in §4 Track A are read-only consumers for all downstream code.
5. **DreamBank suite must remain green.** After every commit, run:
   ```bash
   .venv/bin/python -m pytest tests/memory/test_hlm_bank.py tests/model/test_hlm_bank_adapter.py tests/model/test_amc_transformer.py tests/alignment/test_dreambank.py -q
   ```
6. **Do not push.** Unless the task explicitly says "push and open PR."
7. **Do not modify dirty-tree files as part of a feature commit.** Pre-existing drift is a separate `chore:` commit.
8. **Compose, don't rewrite.** Existing alignment modules are the building blocks. Import them. Do not duplicate their logic.
9. **All commits require Co-Authored-By.** See commit template in §2.3 of individual plan files.

---

## 7. Done Criteria

### Phase 1 — Stabilization

```bash
git status --short  # no modified files outside of committed chore:
make test           # all tests pass
git log main..HEAD --oneline  # clean history, no stray commits
```

### Phase 2 — CB-07

```bash
.venv/bin/python -m pytest tests/eval/test_serving_harness_gpu.py -q
# THOROUGH win_rate delta >= 0.03
# FAST win_rate delta >= -0.01
# FAST p95_latency_ms savings >= 10%
```

### Phase 3 — APEX

```bash
.venv/bin/python -m pytest tests/alignment/test_apex_trainer.py tests/alignment/test_apex_ablation.py -q
# All prior method ablations reproduce within 2% of standalone trainer results
.venv/bin/python -c "from src.alignment.apex.trainer import APEXTrainer; print('apex imports OK')"
```

### Phase 4 — Serving

```bash
docker compose -f docker-compose.prod.yml up -d
curl http://localhost:3001/health  # BFF health check
curl http://localhost:3001/api/chat -d '{"message":"hello"}' -H "Content-Type: application/json"
npm test --workspace=middle  # TypeScript BFF test suite
```

---

## Cross-Reference Map

| Need | Document |
|------|----------|
| DreamBank API contracts | `docs/plans/2026-05-27-dreambank-implementation.md` |
| CascadeBank router spec | `docs/plans/2026-05-27-cascadebank-implementation.md` |
| CB-06 serving harness spec | `docs/plans/2026-05-27-cb06-serving-harness.md` |
| Federated memory deltas | `docs/plans/2026-05-27-federated-memory-deltas.md` |
| TrustRAG retrieval | `docs/plans/2026-05-27-trustrag.md` |
| Memory debate protocol | `docs/plans/2026-05-27-memory-debate.md` |
| Architecture overview | `docs/ARCHITECTURE.md` (v4.0) |
| Alignment scope | `docs/ALIGNMENT_SCOPE.md` |
| Gap ledger | `docs/GAP_LEDGER.md` |
| HLM research (external) | `~/Desktop/HLM-Architecture-Exploration/architecture_research_v4.md` |

---

**Last updated:** 2026-05-28  
**Maintained by:** Christien Antonio  
**Branch baseline:** `46ee2f13` (HEAD), `3a759891` (DreamBank hardened anchor)
