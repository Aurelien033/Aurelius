# DreamBank MVP — Traceability Report

Generated: 2026-05-27
Branch: clean/amc-curation-20260521-101220

## Claim

A tiny runtime-writable preference bank in MLA latent space can be updated
during idle-time self-play to improve alignment/persona behavior without weight
updates, while preserving bounded compute and privacy-friendly storage.

## Not Claimed

- Generic memory bank novelty (crowded: Larimar, Titans, Memorizing Transformers, MSA, MoC)
- New preference loss function
- Federated sync mechanism

## Files Created

| File | Task | Purpose |
|---|---|---|
| `src/memory/hlm_bank.py` | DB-01 | Core bank: top-k read, no-grad upsert, decay, serialization |
| `tests/memory/test_hlm_bank.py` | DB-01 | Bank contract tests (10/10 pass) |
| `src/model/hlm_bank_adapter.py` | DB-02 | Differentiable adapter: gate + residual injection |
| `tests/model/test_hlm_bank_adapter.py` | DB-02 | Adapter invariants (6/6 pass) |
| `src/alignment/dreambank.py` | DB-04 | Dream cycle controller: self-play consolidation |
| `tests/alignment/test_dreambank.py` | DB-04 | Controller tests (7/7 pass) |
| `scripts/run_dreambank_cycle.py` | DB-05 | Dry-run CLI (JSON stdout, no model deps) |
| `tests/scripts/test_run_dreambank_cycle.py` | DB-05 | CLI smoke tests (5/5 pass) |
| `src/eval/dreambank_ablation.py` | DB-06 | Ablation harness: bank-on vs bank-off |
| `tests/eval/test_dreambank_ablation.py` | DB-06 | Ablation verification (3/3 pass) |

## Files Modified

| File | Task | Changes |
|---|---|---|
| `src/model/amc_transformer.py` | DB-03 | Config fields, output telemetry, optional adapter wiring |
| `tests/model/test_amc_transformer.py` | DB-03 | 5 new bank-related tests (+17 total pass) |
| `src/memory/__init__.py` | DB-01 | Export HLM bank symbols |
| `docs/research-brief.md` | DB-07 | DreamBank section added |

## Test Summary

```
tests/memory/test_hlm_bank.py ........... 10 passed
tests/model/test_hlm_bank_adapter.py .... 6 passed
tests/model/test_amc_transformer.py ..... 17 passed (12 baseline + 5 new)
tests/alignment/test_dreambank.py ....... 7 passed
tests/scripts/test_run_dreambank_cycle.py 5 passed
tests/eval/test_dreambank_ablation.py ... 3 passed
------------------------------------------
Total: 48 tests, 0 failures
```

## Evidence Produced

1. **Bank contract tests** — empty reads are zero, upserts fill slots, decay works, serialization round-trips
2. **Adapter invariants** — identity when bank none/empty, changes hidden when bank non-empty, gradients flow to adapter only
3. **Dream cycle determinism** — deterministic generate/score/embed produce reproducible bank updates
4. **Ablation delta proof** — non-empty bank measurably changes logits vs empty bank under identical weights
5. **Privacy** — no raw prompts stored; only SHA-256 hashes in metadata
6. **CLI dry-run** — zero-dependency JSON output for future agent smoke tests

## Next Evidence Needed

- Real preference benchmark (reward model or judge-based scoring)
- Sleep-cycle improvement curve (10+ cycles, measuring win rate delta)
- p95 latency impact on real inference (not tiny CPU model)

## Prior Art Boundary

- **Larimar, Titans, Memorizing Transformers** — generic attention-based memory banks
- **MSA (Mixture of Semantic Associators)** — associative memory in latent space
- **MoC (Memory of Concepts)** — episodic concept retrieval
- **DreamBank differs** by: MLA-latent-specific writes, sleep-time self-play consolidation, trust-labeled provenance, bounded slot count (14 default), privacy-first (hash-only metadata)
