# Claims Ledger — Aurelius AMC v1

**Date:** 2026-05-30
**Version:** v1.0
**Parent:** v5 Grand Unified Plan, PART VI (Paper 1 — AMC)
**Purpose:** Track evidence status for every claim that could appear in the AMC paper.

---

## EVIDENCE LEVELS

| Level | Definition | Paper Eligible |
|-------|-----------|----------------|
| L0 | Spec only (no code) | NO |
| L1 | Design documented | NO |
| L2 | Integration smoke test passes | NO (wiring proof only) |
| L3 | Benchmarked on real data | YES (with caveats) |
| L4 | Ablated (vs baseline) | YES |
| L5 | Reproducible (3+ seeds) | YES |
| L6 | Published (arXiv/conference) | YES |

---

## CLAIMS INVENTORY

### CATEGORY A: ARCHITECTURE (Core Novelty)

| # | Claim | Evidence Level | Source Files | Status |
|---|-------|---------------|--------------|--------|
| A1 | Per-layer differentiable memory exists as a Transformer component | L2 | `src/model/__init__.py` | BACKED |
| A2 | Per-layer SSM working memory provides short-term state | L1 | `docs/AMC_COMPLETE_BUILDOUT.md` | DESIGN ONLY |
| A3 | Stop-gradient surprise head detects novel inputs per-layer | L1 | `docs/AMC_COMPLETE_BUILDOUT.md` | DESIGN ONLY |
| A4 | Gumbel straight-through promotion gate enables differentiable Tier-1→Tier-2 writes | L1 | `docs/AMC_COMPLETE_BUILDOUT.md` | DESIGN ONLY |
| A5 | Three-tier hierarchy (SSM → episodic → constitutional) is implemented | L2 | `src/memory/amc_tier2.py`, `sdb_runtime.py`, `alignment/constitutional*.py` | PARTIAL |
| A6 | Surprise gate + three-tier hierarchy constitutes a live inference-time architecture search primitive | L1 | `docs/plans/*` | CLAIM ONLY |

### CATEGORY B: MEMORY MECHANISMS

| # | Claim | Evidence Level | Source Files | Status |
|---|-------|---------------|--------------|--------|
| B1 | Tier-2 episodic memory supports surprise-gated writes | L2 | `src/memory/amc_tier2.py:81-104` | BACKED |
| B2 | SDB propose-verify-commit contract prevents unverified memory writes | L3 | `src/memory/sdb_runtime.py:265-481` | BACKED |
| B3 | Memory retrieval supports semantic search + fallback to recent entries | L2 | `src/memory/amc_tier2.py:106-122` | BACKED |
| B4 | No-memory vs Tier-2 ablation shows context improvement | L2 | `src/memory/amc_tier2.py:136-165` | BACKED (wiring only) |
| B5 | Memory entries carry provenance (role, session_id, content hash) | L3 | `src/memory/amc_tier2.py:192-258` | BACKED |
| B6 | Trust-aware LTS (Tier-3) supports non-evictable constitutional principles | L1 | `src/alignment/constitutional*.py` | DESIGN ONLY |

### CATEGORY C: DREAMBANK + SLEEP

| # | Claim | Evidence Level | Source Files | Status |
|---|-------|---------------|--------------|--------|
| C1 | DreamBank controller performs multi-temperature self-play | L3 | `src/alignment/dreambank.py:74-169` | BACKED |
| C2 | Margin-filtered preference pair formation | L3 | `src/alignment/dreambank.py:108-125` | BACKED |
| C3 | HLMPreferenceBank upsert with decay | L2 | `src/memory/hlm_bank.py` | BACKED |
| C4 | Zero-weight invariant: backbone eval() + no_grad() during sleep | L2 | `scripts/run_dreambank_cycle.py` | BACKED (wiring only) |
| C5 | DreamBank sleep produces downstream task-success lift | L0 | N/A | NOT TESTED |
| C6 | Prompt hash-only storage (no raw prompts in metadata) | L3 | `src/alignment/dreambank.py:71-72,118` | BACKED |

### CATEGORY D: AGENT LOOP

| # | Claim | Evidence Level | Source Files | Status |
|---|-------|---------------|--------------|--------|
| D1 | Observe-Think-Act-Reflect loop exists | L1 | `src/agent/react_loop.py` | DESIGN ONLY |
| D2 | MCTS value head guides Think phase | L0 | N/A | NOT IMPLEMENTED |
| D3 | Tool cross-attention in Act phase | L0 | N/A | NOT IMPLEMENTED |
| D4 | Critic + self-correction in Reflect phase | L0 | N/A | NOT IMPLEMENTED |
| D5 | Memory writes/retrievals are auditable in trace | L0 | N/A | NOT TESTED |

### CATEGORY E: TRAINING + INFERENCE

| # | Claim | Evidence Level | Source Files | Status |
|---|-------|---------------|--------------|--------|
| E1 | 1.3B model trains with DeepSpeed ZeRO | L2 | `configs/deepspeed_zero1.json` | BACKED |
| E2 | Gradient checkpointing reduces activation memory | L2 | `configs/train_1b.yaml:51` | BACKED |
| E3 | Mixed precision (bf16) training is stable | L1 | `configs/train_1b.yaml:48` | DESIGN ONLY |
| E4 | engine_mode inference supports AMC memory | L1 | `src/serving/aurelius_server.py` | DESIGN ONLY |
| E5 | TST training accelerates pretraining by 2.5x | L0 | `src/training/tst_trainer.py` | NOT WIRED |

### CATEGORY F: SAFETY + GOVERNANCE

| # | Claim | Evidence Level | Source Files | Status |
|---|-------|---------------|--------------|--------|
| F1 | SDB runtime redacts secrets from memory payloads | L3 | `src/memory/sdb_runtime.py:97-108` | BACKED |
| F2 | Constitutional AI trainer supports multi-dimensional scoring | L2 | `src/alignment/constitutional_scoring.py` | BACKED |
| F3 | Differential privacy mechanism for episodic writes | L2 | `src/privacy/` | BACKED (proof) |
| F4 | Memory quarantine + revocation epochs | L2 | `src/memory/sdb_runtime.py` | BACKED |

---

## SUMMARY STATISTICS

| Category | Total Claims | L0 | L1 | L2 | L3 | L4 | L5 | L6 |
|----------|-------------|----|----|----|----|----|----|----|
| Architecture | 6 | 0 | 4 | 1 | 1 | 0 | 0 | 0 |
| Memory | 6 | 0 | 1 | 3 | 2 | 0 | 0 | 0 |
| DreamBank | 6 | 1 | 0 | 2 | 3 | 0 | 0 | 0 |
| Agent Loop | 5 | 4 | 1 | 0 | 0 | 0 | 0 | 0 |
| Training | 5 | 1 | 2 | 2 | 0 | 0 | 0 | 0 |
| Safety | 4 | 0 | 0 | 2 | 2 | 0 | 0 | 0 |
| **TOTAL** | **32** | **6** | **8** | **10** | **8** | **0** | **0** | **0** |

**Paper-ready claims (L3+):** 8/32 (25%)
**Ring 1 target (L4+):** 0/32 (0%) — this is the gap to close.

---

## RING 1 TARGET

To satisfy Gate R1-GA, we need these claims at L4+ (ablated):

| Claim | Current | Target | Path to Target |
|-------|---------|--------|---------------|
| B1 (surprise-gated writes) | L2 | L4 | Ring 1 traces with live writes |
| B4 (no-memory vs Tier-2 ablation) | L2 | L4 | Ablation A0 vs A2 |
| C1 (DreamBank self-play) | L3 | L4 | DreamBank on real traces |
| C5 (DreamBank lift) | L0 | L4 | Gate R1-GB |
| D5 (auditable traces) | L0 | L4 | Ring 1 trace format |

---

## VALIDATION PROTOCOL COMPLIANCE

Per PART I.2, six non-negotiable rules:

1. **Full invocation record:** Every L3+ claim must carry config, command, SHA, artifact path. ✅ for L3 claims.
2. **Oracle/smoke firewall:** L2 claims do not populate paper tables. ✅ enforced.
3. **Adapter validation before claim:** <=150M smoke first. ✅ planned.
4. **Ablation and failure symmetry:** Failures reported at same granularity. ✅ in trace spec.
5. **Seed + compute transparency:** Declare all seeds + GPU-hours. ✅ in trace spec.
6. **Reproducibility pack:** Minimal tarball replays in <4h. ✅ in Gate R1-GC.

---

**End of Claims Ledger v1.0**
