# Technical Debt Ledger — Aurelius v5

**Created:** 2026-05-30
**Updated:** 2026-05-30
**Owner:** Christien Antonio
**Update frequency:** After every Ring 1 milestone

---

## P0 — BLOCKS RING 1 CLAIM (Must Address Before Any Evidence)

| # | Debt | File/Location | Description | Due | Status |
|---|------|---------------|-------------|-----|--------|
| P0-1 | No trained 1B AMC checkpoint | `checkpoints/` | Largest existential gap. No checkpoint to run Ring 1 traces against. T34 training run family not yet executed. | Week 2 | OPEN |
| P0-2 | Missing per-layer SSM module | `src/memory/amc_transformer.py` | Referenced in plan as core AMC component. File does not exist. Per-layer working memory not implemented as differentiable module. | Week 1 | OPEN |
| P0-3 | Missing surprise head | `src/memory/amc_surprise.py` | Stop-gradient surprise head not implemented. Tier-2 uses threshold-based admission but not the full differentiable surprise mechanism from the AMC thesis. | Week 1 | OPEN |
| P0-4 | Missing promotion gate | `src/memory/amc_promotion.py` | Gumbel straight-through differentiable promotion gate not implemented. No per-layer gating from Tier-1 to Tier-2. | Week 1 | OPEN |
| P0-5 | No evaluation harness | `src/eval/ring1_*.py` | No harness for 4-12 step traces with memory introspection. Design exists (MINIMAL_EVALUATION_HARNESS_DESIGN.md) but not implemented. | Week 2 | OPEN |
| P0-6 | No Ring 1 trace templates | `traces/templates/` | Task templates for multi-hop QA, tool-use sequences, long-horizon planning not created. | Week 1 | OPEN |
| P0-7 | Missing CLAIMS_LEDGER.md | `docs/CLAIMS_LEDGER.md` | Referenced heavily in plan ("8/10 claims already BACKED"). File does not exist. Must create or locate. | Week 1 | OPEN |

---

## P1 — NEXT 4–8 WEEKS (Medium Priority)

| # | Debt | File/Location | Description | Due | Status |
|---|------|---------------|-------------|-----|--------|
| P1-1 | APEX/Praxis/Mosaic v2 integration surface-thin | `src/alignment/` | Design documents are excellent (apex-design.md, praxis-implementation.md) but only partial component imports implemented. | Week 4 | OPEN |
| P1-2 | No joint ablation of "all three tiers + DreamBank + MCTS" | `src/eval/` | Individual components exist but no integrated ablation. | Week 6 | OPEN |
| P1-3 | Serving/BFF/gateway fragmentation | `src/serving/`, `middle/src/routes/`, `gateway/` | Missing `chat.ts`, `registry.ts`, `brain.ts` per MASTER-IMPLEMENTATION-PLAN. Treat as Ring 2 only. | Ring 2 | OPEN |
| P1-4 | C9 adapter smoke validation incomplete | `src/eval/` | Per CLAIMS_LEDGER reference. Not smoke-green on <=150M model. | Week 3 | OPEN |
| P1-5 | Model Card 5-layer vision vs CORE_SURFACE.md reconciliation | `docs/MODEL_CARD.md`, `docs/CORE_SURFACE.md` | Boundary language inconsistency for external readers. | Week 4 | OPEN |
| P1-6 | MCTS value head not implemented | `src/agent/` | Referenced in MODEL_CARD.md as "Think" phase component. Not implemented. | Week 3 | OPEN |
| P1-7 | Observe-Think-Act-Reflect loop not wired | `src/agent/react_loop.py` | ReAct loop exists but not integrated with AMC memory layer. | Week 3 | OPEN |
| P1-8 | Tool cross-attention not implemented | `src/model/` | Referenced in MODEL_CARD.md as "Act" phase component. | Week 4 | OPEN |
| P1-9 | Critic + self-correction not implemented | `src/agent/` | Referenced in MODEL_CARD.md as "Reflect" phase. | Week 4 | OPEN |
| P1-10 | Reproducibility pack discipline never applied | N/A | Has never been applied to a real multi-step agent + DreamBank trace set. | Week 4 | OPEN |

---

## P2 — LOW PRIORITY (Cosmetic / Deferred)

| # | Debt | File/Location | Description | Due | Status |
|---|------|---------------|-------------|-----|--------|
| P2-1 | Cosmetic doc drift | `docs/` | Some docs reference old file paths or stale configs. | Ongoing | OPEN |
| P2-2 | Archive/ directories un-audited | `archive/`, `.aurelius/archive/` | Do not expand, do not delete without audit. | Ring 2 | OPEN |
| P2-3 | No Docker/compose for serving | `docker-compose.yml` | Per GAP_LEDGER.md. | Ring 2 | OPEN |
| P2-4 | CI workflow incomplete | `.github/workflows/ci.yml` | Modified but not fully functional. | Week 2 | OPEN |
| P2-5 | Rust bridge (`rust_bridge.py`) untested | `src/` | No tests exist. | Ring 2 | OPEN |
| P2-6 | `brain_layer.py` untested | `src/` | No tests. | Ring 2 | OPEN |

---

## COMPLETED (From GAP_LEDGER.md)

| Slice | Description | Tests Added | Date |
|-------|-------------|-------------|------|
| skills_registry.py | 7 registry entries, contract wrapper, verify_contract | 4 | Pre-v5 |
| agent_registry.py | 12 registry entries, 6 contract wrappers | 10 | Pre-v5 |
| api_registry.py | 14 registry entries, 5 contract wrappers | 4 | Pre-v5 |
| tool_schema_registry.py | 13 registry entries, verify_imports() | 4 | Pre-v5 |

**Bug fixes carried forward:**
- Missing `self.skill_dim` in `skills.py` — FIXED
- ValueHead 2D crash in `agent_core.py` — FIXED
- write_to_memory batch crash in `agent_loop.py` — FIXED
- reflect multi-batch crash in `agent_loop.py` — FIXED

**Test suite baseline:** 132/132 pass, 40 Python files syntax-OK.

---

## HOW TO USE THIS LEDGER

1. **Ring 1 gates are blocked by P0 items.** Clear P0s before any evidence claim.
2. **Update after every milestone.** Change status to `IN_PROGRESS` or `CLOSED`.
3. **P1 items are scoped to the 30-day plan.** Address in order.
4. **P2 items are cosmetic.** Do not touch during Ring 1 focus.
5. **Promote/demote as needed.** If a P1 becomes blocking, promote to P0.

---

**End of Technical Debt Ledger**
