# Aurelius — Hybrid Linear-Attention LLM Research Platform

[![CI](https://github.com/Aurelien033/Aurelius/actions/workflows/ci.yml/badge.svg)](https://github.com/Aurelien033/Aurelius/actions/workflows/ci.yml)
[![Nightly](https://github.com/Aurelien033/Aurelius/actions/workflows/nightly.yml/badge.svg)](https://github.com/Aurelien033/Aurelius/actions/workflows/nightly.yml)
[![Rust](https://github.com/Aurelien033/Aurelius/actions/workflows/rust.yml/badge.svg)](https://github.com/Aurelien033/Aurelius/actions/workflows/rust.yml)
[![CodeQL](https://github.com/Aurelien033/Aurelius/actions/workflows/codeql/badge.svg)](https://github.com/Aurelien033/Aurelius/security/code-scanning)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Aurelius is an open-source platform and evidence-gated research program for hybrid
linear-attention LLMs (Gated DeltaNet / delta-rule + full-attention), post-training
(SFT → DPO → GRPO/RLVR), verifier-native inference, and mechanistic
interpretability. Falsification-first: what is published here is measured —
including the nulls.

**Active flagship: the Aurelius-9B release campaign** — a 9B-class hybrid student
(Qwen3.5-9B; 24 GDN + 8 full-attention layers, 3:1) with OPD-warm → GRPO/RLVR
post-training and a contamination-aware, verifier-gated eval harness.
Status: research + engineering active; **no tuned checkpoint published yet** — this
repo does not carry unverified benchmark claims.

## Contents

- [Research Program — Current Status](#research-program--current-status-2026-10-08)
- [Platform](#platform)
- [Repository Layout](#repository-layout)
- [Quick Start](#quick-start)
- [Testing & CI](#testing--ci)
- [Documents](#documents)
- [License](#license)

## Research Program — Current Status (2026-10-08)

### In flight

- **Aurelius-9B release campaign — final measured training rounds.** The single-GPU
  SFT arm has delivered; the unification arm is in flight. Release stays
  evidence-gated: no checkpoint ships without its pre-registered gate evidence, and
  no unverified benchmark claims are published here.
- **Training-data program — verified open-weight teachers.** OPD warm →
  GRPO/RLVR distillation from open-weight teacher models (DeepSeek / Kimi / GLM
  families). Teachers are qualified by measured strength before admission and
  verified against the source paper and the live registry — never asserted.
- **Agentic-trace research program (R1–R4).** 90+ agent-trace corpora cataloged
  from public sources and content-level gated for tool-use post-training —
  including Hermes-Agent harness traces (real tool-use trajectories), NVIDIA's
  Nemotron-Terminal-Corpus (Terminal-Bench trajectories, DeepSeek-V3.2 teacher),
  and AgentTrove. Admission gates: teacher class (open-weight only for the release
  lane), style fingerprints, self-checking density, tool-call validity,
  decontamination, and exclusion of eval-derived rows. Pilot mixes are sized in
  measured Qwen3.5 tokens.

### Measured results (published)

| Result | Finding |
|---|---|
| Self-improvement pilot (2026-09-10) | Four arms, one student, calibrated MATH-level pools: rejection-SFT sits **below the movement floor** — gated A **+0.8pp** (within noise), committee gate C **+1.2pp**, unfiltered B **−2.0pp**; round 2 self-regressed **−7.2pp** with acceptance collapsing 44.1% → 13.8%. **Capstone: swapping only the objective to RLVR (GRPO) produced the first movement on both axes — +2.4pp pass@1 / +6.0pp pass@8** — even with 60.6% of training groups carrying zero reward variance. The productive channel is RL-strength signal + a live verifier, not cheap self-SFT. |
| Falsification series E1–E10 | A decisive A/B exposed an architecture-gating bug: the default α range [−0.5, 0.5] caps GDN memory at ~2–3 tokens by construction (0.5¹⁷ ≈ 7.6e-6). Fixed in model defaults: **100% vs 2.3% recall on delay-17**. |
| 9B hybrid code audit | 10 bugs found, fixed, and re-verified in the 9B hybrid codebase (sequential-loop "parallel" scan → exact chunked scan; RoPE + KV-cache position corruption; dead beta write gate; OPD one-batch loop; stub eval harness → real harness with HF-baseline mode). 5/5 integration tests, 8/8 scan-correctness cases. |
| AIE v1 — Aurelius Improvement Engine | Evidence-gated closed-loop optimizer that measures the truth surface from eval artifacts and ranks interventions under damage guards. Every number is labeled measured / derived / prior; unmapped data is reported as NULL, never zeroed. |
| Repo engineering | Main + nightly CI green; Dependabot alerts 34 → 2; code-scanning findings 260 → 230 (including two real ReDoS rewrites + log key-material removal); the `rust_memory` crate is covered by CI; repo-wide simplification to −41.4% LOC (2026-09-16) with all gates green. |

### Prior frontier package (2026-08, kept for provenance)

SISA (Self-Indexed Sparse Attention — training-free KV indexer), AEX-KV
(bit-exact KV compression codec), MoK (Mixture of Kittens), and the adoption
matrix.

Canonical record of research notes: the lab's Obsidian vault (not this repo).

## Platform

| Layer | Location | Role |
|-------|----------|------|
| Python core | `src/`, `agent/`, `gateway/`, `aurelius_cli/`, `server/` | Model, training, inference, alignment, serving, CLI |
| Rust engine | `crates/`, `rust_memory/` | Tokenization, search, vector similarity, sessions, data engine |
| Node BFF | `middle/` | Auth, rate limiting, WebSocket, SSE, cron |
| Frontend | `frontend/` | Mission Control dashboard (React 19 + TypeScript) |
| Research assets | `paper/`, `release/`, `verification/`, `training_data/`, `docs/` | Paper (LaTeX + claims ledger), release manifests, ring-gate verification scripts, data pipelines, runbooks |
| Ops | `scripts/`, `configs/`, `deployment/`, `.github/` | Bootstrap + benchmarks + exporters, ablation and model configs, Docker + compose, CI/CD |

Highlights: handwritten transformer/GDN core with GQA + RoPE, 8 hot-swappable
KV-cache strategies, GRPO/DPO/PPO/SimPO trainers, execution-verifier tooling,
OpenAI-compatible serving API, ReAct agent loop with sandboxed tool execution,
14-guardrail safety system, production resilience primitives (circuit breaker /
bulkhead / retry / rate limiting).

## Repository Layout

```
Aurelius/
├── src/            # Model + training + inference (Python core)
├── agent/          # ReAct agent loop, sandboxed tool execution
├── gateway/        # OpenAI-compatible API, WebSocket, metrics
├── aurelius_cli/   # CLI entry points
├── crates/         # Rust engine (tokenizer, search, vectors, sessions)
├── rust_memory/    # Rust memory subsystem (CI-covered)
├── middle/         # Node BFF (auth, rate limiting, SSE, cron)
├── frontend/       # Mission Control dashboard (React 19 + TS)
├── paper/          # LaTeX paper + CLAIMS_LEDGER.md evidence ledger
├── release/        # ARTIFACT_MANIFEST.md + release packaging scripts
├── verification/   # Ring-gate verification scripts
├── training_data/  # Data pipeline (arxiv_pipeline.py)
├── docs/           # Plans, reports, reproducibility, training runbooks
├── configs/        # Ablation + model configs (100M → 14B)
├── scripts/        # bootstrap, benchmark, export, profile utilities
├── deployment/     # Docker + compose
└── tests/          # Pytest suite
```

## Quick Start

Prerequisites: Python 3.12+, Node 22+, Rust 1.81+, npm 10+.

```bash
git clone https://github.com/Aurelien033/Aurelius.git
cd Aurelius
bash scripts/bootstrap.sh         # full setup (Rust + Python + Node)
bash scripts/bootstrap.sh --fast  # skip Rust builds
```

```bash
aurelius                                       # interactive chat
aurelius chat --react --model-path <ckpt>      # ReAct tool-use loop
aurelius serve --port 8080                     # OpenAI-compatible API server
docker compose up                              # full stack (Docker)
```

OpenAI-compatible endpoints: `POST /v1/chat/completions`, `GET /v1/models`,
`GET /health` (liveness), `GET /health/ready` (readiness), `GET /metrics`,
`WebSocket /ws`.

## Testing & CI

```bash
make test           # Python backend
make frontend-test  # Vitest
make middle-test    # Node BFF
make rust-test      # Rust crates
make ci             # lint + typecheck + security + all tests
```

CI gates (green on `main` and the nightly lane): pytest focused suite
(3.12 + 3.13), ruff lint + format, py_compile, bandit (baseline-gated),
pip-audit, cargo audit, frontend + middle typecheck/lint/test, plus CodeQL
and DevSkim security scanning.

## Documents

| Document | Description |
|----------|-------------|
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [SECURITY.md](SECURITY.md) | Security policy and vulnerability reporting |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Code style, testing, branch strategy |
| [paper/CLAIMS_LEDGER.md](paper/CLAIMS_LEDGER.md) | Evidence ledger backing the paper's claims |
| [release/ARTIFACT_MANIFEST.md](release/ARTIFACT_MANIFEST.md) | Release artifact manifest |
| [docs/training/LIGHTNING_RUNBOOK.md](docs/training/LIGHTNING_RUNBOOK.md) | Training runbook |
| [docs/reports/](docs/reports/) | Traceability, validation, and provenance reports |

## License

[MIT License](LICENSE) — Copyright © 2025 Aurelius Systems, Inc.

**GitHub:** [https://github.com/Aurelien033/Aurelius](https://github.com/Aurelien033/Aurelius)
