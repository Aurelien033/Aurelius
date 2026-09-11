# Aurelius — Hybrid Linear-Attention LLM Research Platform

Open-source platform and evidence-gated research program for hybrid linear-attention
LLMs (Gated DeltaNet / delta-rule + full-attention), post-training (SFT → DPO → GRPO/RLVR),
verifier-native inference, and mechanistic interpretability. Falsification-first: what is
published here is measured, including the nulls.

**Active flagship: the Aurelius-9B release campaign** — a 9B-class hybrid student
(Qwen3.5-9B, 24 GDN + 8 full-attention layers, 3:1) with OPD-warm → GRPO/RLVR
post-training and a contamination-aware, verifier-gated eval harness.
Status: research + engineering active; **no tuned checkpoint published yet** — this
repo does not carry unverified benchmark claims.

## Current Status — 2026-09-11

- **RSI pilot — first measured self-improvement loop at lab scale** (Kaggle P100,
  Qwen2.5-1.5B-Instruct, calibrated MATH-level pools, 7h51m run, every stage receipted):
  rejection-SFT self-training is **below the movement floor at 1.5B** — round 1 **+0.8pp**
  (within noise), round 2 **−7.2pp** with acceptance collapsing 44.1% → 13.8%, unfiltered
  control **−2.0pp**; pass@k flat (no boundary expansion). Gated-vs-unfiltered ordering
  replicates the literature in miniature (gates slow degradation). Conclusion: the first
  productive signal is RL-strength (RLVR/DAPO-class) + verifier/teacher upgrades — not
  cheap self-SFT. Artifacts (protocol, results, report, 11 primary-source verifications):
  `research_loop/universal/2026-09-10_rsi_kaggle_pilot/`.
- **AIE v1 — Aurelius Improvement Engine** — evidence-gated closed-loop optimizer that
  measures the truth surface from eval artifacts and ranks interventions under damage
  guards (`python -m aie.run`); every number labeled measured/derived/prior, unmapped
  data reported as NULL, never zeroed.
- **Repo hardening (09-10)** — main + nightly CI green; dependabot alerts 34 → 2;
  code-scanning 260 → 230 (11 fixed, including two real ReDoS rewrites + log
  key-material removal); `rust_memory` crate now covered by CI.

### Prior status — 2026-09-06

- **9B code audit** — 10 bugs found, fixed, and re-verified in the 9B hybrid codebase
  (sequential-loop "parallel" scan → exact chunked scan; RoPE + KV-cache position
  corruption; dead beta write gate; OPD one-batch loop; stub eval harness → real harness
  with HF-baseline mode). 5/5 integration tests, 8/8 scan-correctness cases.
- **Research cycle E1–E10 (falsification series)** — decisive A/B exposed an
  architecture-gating bug: the default α range [-0.5, 0.5] caps GDN memory at ~2–3 tokens
  by construction (0.5¹⁷ ≈ 7.6e-6). Fixed in model defaults: 100% vs 2.3% recall on delay-17.
- **RSI frontier sweep** — 244 scored sources, 21 deep-read, verified calculation battery,
  toy experiments (E1/E1b collapse laws, 100-seed verified), whitepaper draft.

Prior frontier package (kept for provenance) lives in [`research/2026-08-frontier/`](research/2026-08-frontier/README.md):
SISA (Self-Indexed Sparse Attention, training-free KV indexer), AEX-KV (bit-exact KV
compression codec), MoK (Mixture of Kittens), and the adoption matrix.

Canonical record of research notes: the lab's Obsidian vault (not this repo).

## What's in this repo

| Layer | Location | Role |
|-------|----------|------|
| Python core | `src/`, `agent/`, `gateway/`, `aurelius_cli/` | Model, training, inference, alignment, serving, CLI |
| Rust engine | `crates/` | Tokenization, search, vector similarity, sessions, data engine |
| Node BFF | `middle/` | Auth, rate limiting, WebSocket, SSE, cron |
| Frontend | `frontend/` | Mission Control dashboard (React 19 + TypeScript) |

Highlights: handwritten transformer/GDN core with GQA + RoPE, 8 hot-swappable KV-cache
strategies, GRPO/DPO/PPO/SimPO trainers, execution-verifier tooling, OpenAI-compatible
serving API, ReAct agent loop with sandboxed tool execution, 14-guardrail safety system,
production resilience primitives (circuit breaker / bulkhead / retry / rate limiting).

## Quick Start

Prerequisites: Python 3.12+, Node 22+, Rust 1.81+, npm 10+.

```bash
git clone https://github.com/Aurelien033/Aurelius.git
cd Aurelius
bash scripts/bootstrap.sh         # full setup (Rust + Python + Node)
bash scripts/bootstrap.sh --fast  # skip Rust builds
```

```bash
aurelius                              # interactive chat
aurelius chat --react --model-path <ckpt>   # ReAct tool-use loop
aurelius serve --port 8080            # OpenAI-compatible API server
docker compose up                     # full stack (Docker)
```

OpenAI-compatible endpoints: `POST /v1/chat/completions`, `GET /v1/models`,
`GET /health` (liveness), `GET /health/ready` (readiness), `GET /metrics`, `WebSocket /ws`.

## Testing & CI

```bash
make test           # Python backend
make frontend-test  # Vitest
make middle-test    # Node BFF
make rust-test      # Rust crates
make ci             # lint + typecheck + security + all tests
```

CI gates (all green on the release branch): pytest focused suite (3.12 + 3.13), ruff
lint + format, py_compile, bandit (baseline-gated), pip-audit, cargo audit,
frontend + middle typecheck/lint/test.

## Documents

| Document | Description |
|----------|-------------|
| [ROADMAP.md](ROADMAP.md) | Project history, measured results, forward plan |
| [SECURITY.md](SECURITY.md) | Security policy and vulnerability reporting |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Code style, testing, branch strategy |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | Architecture card |
| [docs/threat_model.md](docs/threat_model.md) | Security threat model |

[MIT License](LICENSE) — Copyright © 2025 Aurelius Systems, Inc.

**GitHub:** [https://github.com/Aurelien033/Aurelius](https://github.com/Aurelien033/Aurelius)
