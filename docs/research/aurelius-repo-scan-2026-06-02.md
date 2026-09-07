# Aurelius Repository Scan — 2026-06-02

Generated: 2026-06-02 (post-6-month Arxiv sweep)
Scope: ~/aurelius (7.0 GB, 25+ branches, 5,633 source files)
Status: working tree on `feature/ring1-tranche5-20260531` with 22 uncommitted modifications
Purpose: ground-truth for the v3→v4 Plan update, the 8 Novel Contributions, and the safe-merge plan.

---

## 1. Repository Footprint

| Metric | Value |
| --- | --- |
| Disk size | 7.0 GB |
| Python files | 5,176 |
| TypeScript files | 146 |
| Rust files | 54 |
| Markdown files | 258 |
| YAML configs | 67 |
| Total source LOC (Python) | ~280k+ |

### Module size distribution

| Module | Size | Notes |
| --- | --- | --- |
| `frontend/` | 231 MB | Next.js app + node_modules |
| `src/` | 74 MB | Core model/memory/alignment code |
| `middle/` | 1.2 MB | BFF (TypeScript) |
| `crates/` | 1.4 MB | 12 Rust micro-services |
| `agent/` | 3.1 MB | Agent loop, skill registry |
| `aurelius/` | 296 KB | CLI entrypoint |
| `build/`, `dist/`, `__pycache__/` | (excluded) | Artifact dirs |

### Top-20 Python modules by LOC

| LOC | Path |
| --- | --- |
| 2,502 | `training_data/sft_generator.py` |
| 2,312 | `training_data/pretrain_generator.py` |
| 1,915 | `src/ui/aurelius_shell.py` |
| 1,509 | `src/agent/session_manager.py` |
| 1,429 | `src/eval/__init__.py` |
| 1,331 | `src/model/interface_framework.py` |
| 1,311 | `aurelius_cli/main.py` |
| 1,262 | `src/serving/aurelius_server.py` |
| 1,247 | `src/training/trainer.py` |
| 1,247 | `src/safety/topology_safety.py` |
| 1,243 | `src/agent/neuro_symbolic_skill.py` |

---

## 2. Branch Inventory

### 2.1 Local branches (25 total, ignoring dependabot remotes)

| Branch | State | Commits vs `main` | Divergence fingerprint |
| --- | --- | --- | --- |
| `main` | clean | — | (base) |
| `feature/ring1-tranche5-20260531` | **CURRENT** | +79 | A |
| `feature/ring1-tranche4-20260531` | idle | +79 | A |
| `feature/ring1-tranche3-20260531` | idle | +79 | A |
| `feature/ring1-tranche2-20260531` | idle | +79 | A |
| `feature/ring1-tranche1-20260531` | idle | +79 | A |
| `fix/security-p0-bff-identity-ws` | idle | +79 | A |
| `fix/security-p0-legacy-license` | idle | +79 | A |
| `fix/security-p1-boundaries` | idle | +79 | A |
| `fix/security-p1-sandbox-auth-deps` | idle | +79 | A |
| `fix/security-p1-session-auth` | idle | +79 | A |
| `fix/security-p2-ci-deploy-hardening` | idle | +79 | A |
| `fix/security-p2-runtime-proof` | idle | +79 | A |
| `fix/security-integration-stack` | idle | (ancestor) | — |
| `fix/security-integration-stack-2` | idle | (ancestor) | — |
| `fix/security-deferred-followups` | idle | (ancestor) | — |
| `fix/security-remediation-p0` | idle | (ancestor) | — |
| `consolidation-temp` | dirty | (ancestor) | — |
| `clean/amc-curation-20260521-101220` | idle | (ancestor) | — |
| `backup/local-main-dirty-push-20260521-091345` | archive | (ancestor) | — |
| `aurelius-v2-backup` | archive | (ancestor) | — |
| `feat/model-core` | idle | (ancestor) | — |

### 2.2 Critical finding: 12 branches share an identical 79-commit divergence

```text
git log main..<branch> --oneline  →  79 commits, all identical
```

**`fix/security-p0-legacy-license`, `fix/security-p1-session-auth`,
`fix/security-p2-runtime-proof`, `feature/ring1-tranche1-20260531`, … —
all twelve branches resolve to the SAME tip commit (or a tip within the
same 79-commit set) on the same divergence from `main`.**

The 12 branches are **labels on a single diverged line of work**, not
12 independent lines. This is the multi-tranche stacking pattern: the
user pre-created the labels as workspace slots, but the actual commit
content was added to whichever label was current at the time of commit.

### 2.3 What those 79 commits contain (head of log)

```
46ee2f13 memory(debate): LLM voices for proposer/skeptic/judge via configurable API
65e60019 retrieval(trustrag): semantic contradiction via injectable contradiction_fn
2910a2a4 model(per-layer): per-layer MLA bank wiring via hlm_bank_read_layers
2651b35b privacy(federated): (epsilon,delta)-DP Gaussian mechanism proof
fb4a3c61 memory(debate): adjudicated Tier-3 promotion via proposer/skeptic/judge
5a372fd1 memory(trustrag): quarantine-aware, trust-bound retrieval controller
36f2a979 memory(federated): FedAvg on DreamBank tensors (delta-level federation)
72a99589 eval(cb06): greedy decode + per-policy cost-proxy serving harness
97ccc1dd feat(cascadebank): compute routing on DreamBank alignment gate (zero new params)
3a759891 fix: harden DreamBank MVP contracts (gate, metadata, gradients, shape, restore)
a02469d2 feat: add DreamBank MVP
b2bb33b5 security(hooks): register check_torch_load in pre-commit config (M-02)
bf074470 security(tools): harden code runner with regex-based sandbox denylist (C-16)
b70a0751 security(gateway): fail-closed rate limiter + authenticated /metrics (C-03, C-04, M-06)
285ff32e fix(model): unify top-p nucleus sampling via shared _apply_top_p_filter (C-17, C-18, NEW-05)
2604d4d1 review(C-19): add chunksize to starmap for large-batch efficiency
7b688c73 fix(training): correct pool.map arity in tokenize pipeline (C-19)
65619cd2 docs: add Aurelius improvement plan
0abff91a docs: add AMC buildout and research synthesis artifacts
0cbb2fff docs: add AMC full buildout prompt pack continuation
```

(20 of 79 shown; the rest are security-p0/p1/p2 stack fixes, model
alignments, eval harness, and CLI/serving hardening.)

---

## 3. Working-Tree State (current branch: `feature/ring1-tranche5-20260531`)

22 tracked files modified, 6 untracked:

```
M  .github/workflows/ci.yml
M  agent/__init__.py
M  agent/session_manager.py
M  gateway/aurelius_api.py
M  middle/src/config.ts
M  middle/src/provider_router.ts
M  middle/src/routes/auth.ts
M  middle/src/routes/evaluation.ts
M  middle/src/routes/scheduler.ts
M  middle/src/server.ts
M  src/alignment/simpo.py
M  src/memory/amc_tier2.py
M  src/model/__init__.py
M  src/model/moe.py
M  src/serving/aurelius_server.py
M  src/serving/function_calling_api.py
M  src/serving/structured_output_decoder.py
M  src/training/trainer.py
M  src/ui/session_manager.py
M  tests/integration/test_lambda_attention_integration.py
M  tests/integration/test_parallel_attention_integration.py
M  tests/tools/test_web_tool.py
M  tools/web_tool.py
?? .github/workflows/nightly.yml
?? .hermes/plans/
?? TECHNICAL_DEBT.md
?? configs/ring1_tranche1.yaml
?? configs/ring1_tranche2.yaml
?? configs/ring1_tranche3.yaml
```

**Implication:** the in-flight Ring-1 work has at least 22 uncommitted
edits across the model core (`src/model/`), serving layer
(`src/serving/`), gateway (`gateway/`, `middle/`), agent loop
(`agent/`), and tests. These should be either:
  1. committed onto the current branch as a final Ring-1 wrap-up
     commit **before** any merge, or
  2. stashed and re-applied after the merge lands.

---

## 4. Plan/Research Artifacts (already on disk)

| File | Size | Notes |
| --- | --- | --- |
| `docs/research/aurelius-master-plan-v3-2026-05-29.md` | 13 KB | 5 phases, 5 contributions, 72 papers |
| `docs/research/aurelius-master-plan-v4-2026-06-02.md` | 38 KB | v3 + N6-N13 contributions, B23-B32 blind spots |
| `docs/research/aurelius-research-plan-2026-05-28.md` | 20 KB | Earlier research plan |
| `docs/research/aurelius-research-plan-2026-05-29.md` | 15 KB | Earlier research plan |
| `docs/research/research-to-code-mapping.md` | 4 KB | Research → code traceability |
| `~/Desktop/AI:ML Research/aurelius-historical-sweep-2026-h1.md` | 18 KB | Executive report |
| `~/Desktop/AI:ML Research/aurelius-improvement-map-2026-h1.md` | 27 KB | Per-paper improvement map |
| `~/research_loop/historical-sweep-2026-h1/synthesis.md` | 24 KB | Master synthesis |
| `~/research_loop/historical-sweep-2026-h1/IMPROVEMENT-MAP.md` | 27 KB | Copy of improvement map |
| `~/Obsidian Coding/Coding/AMC Buildout/AMC_FULL_BUILDOUT_PROMPTS.md` | 73 KB | Buildout prompt pack |
| `~/Obsidian Coding/Coding/AMC Buildout/MODEL_IMPLEMENTATION.md` | 18 KB | Model implementation notes |
| `~/Obsidian Coding/Coding/AMC Buildout/TRANCHE_STATUS.md` | 7 KB | Tranche tracker (T00 done, rest pending) |
| `~/Desktop/AI Plans/2026-05-09-aurelius-alignment-v2-design.md` | 29 KB | Alignment v2 design |
| `~/Desktop/AI Plans/2026-05-09-mosaic-v2-implementation.md` | 59 KB | Mosaic v2 impl |
| `~/Desktop/AI Plans/2026-05-09-praxis-implementation.md` | 45 KB | Praxis impl |

---

## 5. Cross-reference: Aurelius Pillars × Repo Modules

| Aurelius Pillar (v3) | Primary repo module | Secondary | Status |
| --- | --- | --- | --- |
| AMC surprise-driven memory | `src/memory/amc_tier2.py` | `src/memory/dreambank*` | partial |
| Hybrid MLA + SSM core | `src/model/moe.py`, `src/model/interface_framework.py` | `src/model/__init__.py` | scaffolded |
| Constitutional / alignment | `src/alignment/constitutional*.py`, `simpo.py` | `src/alignment/cpo.py` | extensive |
| Tiered trust + memory | `src/memory/*` | `src/agent/session_manager.py` | partial |
| Function-calling / agent loop | `src/serving/function_calling_api.py` | `src/agent/session_manager.py` | scaffolded |
| KV-cache / inference | `src/serving/aurelius_server.py` | `src/serving/structured_output_decoder.py` | present |
| Eval harness (cb06) | `src/eval/*` | `tests/integration/*` | present |
| Multi-agent benchmarks | `tests/integration/*` | `benchmarks/` | present |
| Privacy / federated | `src/privacy/*` (newly added in 79-commit bundle) | — | partial |
| Safety / policy | `src/safety/topology_safety.py` | `src/alignment/content_filter.py` | present |

---

## 6. Conclusions

1. **The 12 "stacked" branches are not really divergent.** They share
   an identical 79-commit fingerprint. Any merge strategy that
   cherry-picks from one branch will reproduce all twelve.

2. **The current branch (`ring1-tranche5`) has 22 uncommitted
   modifications.** These are likely the closing changes of the
   Ring-1 stack and should be committed (or stashed) before any
   merge.

3. **The 7.0 GB size is dominated by `node_modules` (frontend 231 MB)
   and `__pycache__` artifacts.** None of that needs to be in the
   merge diff; `.gitignore` already excludes them.

4. **The repo has the substrate to support all 8 v4 Novel
   Contributions** (N6-N13), but each requires new code in
   `src/{alignment,memory,privacy,serving,eval}/` plus tests.

5. **No credentials, no production secrets, no infra-as-code that
   would block a safe merge.** `.env.example` is committed; real
   `.env` is in `.gitignore`.

6. **The Master Plan v3 + v4 is the canonical source of truth** for
   what each contribution should be. The Tranche Status doc is the
   execution tracker. The two are now aligned.

See the companion document `aurelius-safe-merge-plan-2026-06-02.md`
for the concrete merge strategy.
