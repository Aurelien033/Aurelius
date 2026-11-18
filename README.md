# Aurelius — Hybrid Linear-Attention LLM Research Platform

[![CI](https://github.com/Aurelien033/Aurelius/actions/workflows/ci.yml/badge.svg)](https://github.com/Aurelien033/Aurelius/actions/workflows/ci.yml)
[![Nightly](https://github.com/Aurelien033/Aurelius/actions/workflows/nightly.yml/badge.svg)](https://github.com/Aurelien033/Aurelius/actions/workflows/nightly.yml)
[![Rust](https://github.com/Aurelien033/Aurelius/actions/workflows/rust.yml/badge.svg)](https://github.com/Aurelien033/Aurelius/actions/workflows/rust.yml)
[![CodeQL](https://github.com/Aurelien033/Aurelius/actions/workflows/dynamic/github-code-scanning/codeql/badge.svg)](https://github.com/Aurelien033/Aurelius/security/code-scanning)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Aurelius is an open-source platform and evidence-gated research program for hybrid
linear-attention LLMs (Gated DeltaNet / delta-rule plus full attention), post-training
(SFT → DPO → GRPO/RLVR), verifier-native inference, and mechanistic
interpretability. Falsification first: published results include the nulls.

## Contents

- [Overview](#overview)
- [Project Direction](#project-direction)
- [Current Status](#current-status)
- [Architecture and Platform](#architecture-and-platform)
- [Repository Layout](#repository-layout)
- [Quick Start](#quick-start)
- [Usage and API](#usage-and-api)
- [Testing and CI](#testing-and-ci)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Overview

Aurelius separates two things that are often blended together: a research program
that tests which mechanisms actually help, and an engineering platform that makes
those mechanisms runnable. Every public result is tied to evidence; unverified
benchmark claims are not published here.

The flagship is [Aurelius-9B](https://huggingface.co/Aurelien33/Aurelius-9B), a
public, non-gated 9B-class hybrid model (Qwen3.5-9B; 24 GDN + 8 full-attention
layers, 3:1). The released artifact is merged bf16 under Apache-2.0. The public
card reports an observed-loss screen only — no accuracy-gain, general-capability,
or safety-certification claim.

## Project Direction

Three themes guide the work:

1. **Hybrid memory that earns its place.** Compare linear-attention state with
   full attention on equal footing. Keep the gains, publish the failures, and fix
   the measurement instruments when they disagree.
2. **Evidence-gated post-training.** SFT → DPO → GRPO/RLVR with live verifiers,
   decontamination, and pre-registered decisions. No checkpoint ships without its
   evidence; no score is claimed from an unmapped result.
3. **Verifier-native inference and interpretability.** Make tool use, long-context
   behavior, and failure modes measurable rather than asserted. Promote research
   prototypes into tested libraries only after their limits are written down.

Near-term work: consolidate the public 9B release evidence, turn research
prototypes into reproducible artifacts, and expand agentic tool-use evaluation.

## Current Status

*Snapshot: 2026-11-17. Full detail: [progress report](docs/reports/PROGRESS-2026-11-17.md).*
Anchor links in this section are stable across status-date updates.

### In flight

- **Aurelius-9B public state** — merged bf16 checkpoint is public and non-gated on
  Hugging Face under Apache-2.0. Claim scope stays limited to the observed-loss
  screen; quantized formats and adapters have no release date or parity claim.
- **Papers** — the SISA arXiv package is assembled but not submitted; a
  near-misses / instrument-fragility paper is in draft.
- **Agentic tooling** — the evaluation harness, audit logger, and head store are
  tested artifacts. The latter two are libraries or staged files, not wired into
  serving or training.
- **Security backlog** — 125 open Dependabot alerts on `main` at 2026-11-17
  (8 critical, 56 high, 52 moderate, 9 low). Remediation is open. The older
  "34 → 2" figure below is a 2026-09 snapshot, not current state.

### Measured results (published)

| Result | Finding |
|---|---|
| Self-improvement pilot (2026-09) | Rejection-SFT stayed within noise: gated **+0.8pp**, committee **+1.2pp**, unfiltered **−2.0pp**; round 2 self-regressed **−7.2pp**. Swapping only the objective to RLVR (GRPO) produced the first movement on both axes: **+2.4pp pass@1 / +6.0pp pass@8**, with 60.6% of groups carrying zero reward variance. |
| Falsification series E1–E10 | The default α range capped GDN memory at ~2–3 tokens (0.5¹⁷ ≈ 7.6e-6). Fixed in model defaults: **100% vs 2.3% recall at delay 17**. |
| 9B hybrid code audit | 10 bugs found, fixed, and re-verified (chunked scan, RoPE/KV-cache positions, beta gate, OPD loop, real eval harness). 5/5 integration tests, 8/8 scan cases. |
| AIE v1 — Aurelius Improvement Engine | Evidence-gated closed-loop optimizer. Every number is labeled measured / derived / prior; unmapped data is reported as NULL, never zeroed. |
| Repository engineering (2026-09 snapshot) | Dependabot alerts 34 → 2; code-scanning findings 260 → 230; repo-wide simplification to −41.4% LOC (2026-09-16) with gates green. Current alert count is listed in "In flight". |

### Prior packages (provenance)

SISA (Self-Indexed Sparse Attention, training-free KV indexer), AEX-KV (bit-exact
KV compression codec), MoK (Mixture of Kittens), and the adoption matrix — all
from 2026-08. The canonical research record is the lab's private vault; public
summaries live in [docs/reports/](docs/reports/).

## Architecture and Platform

| Layer | Location | Role |
|---|---|---|
| Python core | `src/`, `agent/`, `gateway/`, `aurelius_cli/`, `server/` | Model, training, inference, alignment, serving, CLI |
| Rust engine | `crates/`, `rust_memory/` | Tokenization, search, vector similarity, sessions, data engine |
| Node BFF | `middle/` | Auth, rate limiting, WebSocket, SSE, cron |
| Frontend | `frontend/` | Mission Control dashboard (React 19 + TypeScript) |
| Research assets | `paper/`, `release/`, `verification/`, `training_data/`, `docs/` | Paper (LaTeX + claims ledger), release manifests, ring-gate verification scripts, data pipelines, runbooks |
| Ops | `scripts/`, `configs/`, `deployment/`, `.github/` | Bootstrap, benchmarks, exporters, ablation and model configs, Docker + compose, CI/CD |

Highlights: handwritten transformer/GDN core with GQA + RoPE, 8 hot-swappable
KV-cache strategies, GRPO/DPO/PPO/SimPO trainers, execution-verifier tooling,
OpenAI-compatible serving API, ReAct agent loop with sandboxed tool execution,
14-guardrail safety system, and production resilience primitives (circuit breaker,
bulkhead, retry, rate limiting).

## Repository Layout

```text
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
├── scripts/        # Bootstrap, benchmark, export, profile utilities
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

## Usage and API

```bash
aurelius                                       # interactive chat
aurelius chat --react --model-path <ckpt>      # ReAct tool-use loop
aurelius serve --port 8080                     # OpenAI-compatible API server
docker compose up                              # full stack (Docker)
```

OpenAI-compatible endpoints: `POST /v1/chat/completions`, `GET /v1/models`,
`GET /health` (liveness), `GET /health/ready` (readiness), `GET /metrics`,
`WebSocket /ws`.

## Testing and CI

```bash
make test           # Python backend
make frontend-test  # Vitest
make middle-test    # Node BFF
make rust-test      # Rust crates
make ci             # lint + typecheck + security + all tests
```

CI gates: pytest focused suite (Python 3.12 + 3.13), ruff lint and format,
py_compile, bandit (baseline-gated), pip-audit, cargo audit, frontend and middle
typecheck/lint/test, plus CodeQL and DevSkim security scanning. Current gate status
is shown by the badges above and in the [Actions tab](https://github.com/Aurelien033/Aurelius/actions).

## Documentation

| Document | Description |
|---|---|
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [docs/reports/PROGRESS-2026-11-17.md](docs/reports/PROGRESS-2026-11-17.md) | Public-safe progress since 2026-10-08 |
| [paper/CLAIMS_LEDGER.md](paper/CLAIMS_LEDGER.md) | Evidence ledger backing the paper's claims |
| [release/ARTIFACT_MANIFEST.md](release/ARTIFACT_MANIFEST.md) | Release artifact manifest |
| [docs/training/LIGHTNING_RUNBOOK.md](docs/training/LIGHTNING_RUNBOOK.md) | Training runbook |
| [docs/reports/README.md](docs/reports/README.md) | Traceability, validation, and provenance reports |
| [SECURITY.md](SECURITY.md) | Security policy and vulnerability reporting |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Code style, testing, branch strategy |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for code style, testing, and branch strategy.
Research claims follow the evidence rules described above: measured results carry
their source, and null results are reported as nulls.

## License

[MIT License](LICENSE) — Copyright © 2025 Aurelius Systems, Inc.
