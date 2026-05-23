# AMC Paper Claims Ledger

Every empirical claim must cite an artifact. Status: **BACKED** (code+tests), **PARTIAL** (structural/oracle only), **TODO** (not measured).

| Claim ID | Claim text | Evidence artifact | Status | Notes |
|----------|------------|-------------------|--------|-------|
| C1 | Per-layer three-tier memory in forward/runtime surfaces | `src/model/amc_transformer.py`, `src/memory/amc_tier2.py`, `src/memory/amc_tier3.py` | BACKED | Integration tests |
| C2 | Tier-1 uses Mamba-2/SSM ZOH discretization without mean-pooled state | `src/model/mamba2_block.py`, `src/model/amc_ssm_layer.py` | BACKED | `tests/model/test_mamba2_block.py` |
| C3 | Promotion gate is differentiable with correct straight-through semantics | `src/model/amc_promotion.py` | BACKED | `tests/model/test_amc_promotion.py`; fix `841182b7` |
| C4 | SDB propose/verify/commit is fail-closed and replayable | `src/memory/sdb_runtime.py` | BACKED | `tests/memory/test_sdb_memory_runtime.py`, `tests/security/test_amc_security_audit.py` |
| C5 | Trust-aware cache identity includes trust/quarantine/revocation | `src/serving/amc_kv_cache.py`, `src/memory/amc_runtime_cache.py` | BACKED | `tests/serving/test_amc_kv_cache.py` |
| C6 | Constitutional memory is permanent/non-evictable | `src/agent/constitutional_memory.py` | BACKED | agent e2e tests |
| C7 | AMC-Memory benchmark evaluates cross-session behavior | `src/eval/amc_memory_benchmark.py` | BACKED | harness exists |
| C8 | Ablation shows memory-task improvement with tiers | `docs/reproducibility/results/ablation_scores.jsonl` | PARTIAL | Oracle smoke only; not trained checkpoints |
| C9 | Standard benchmark parity (GSM8K, MMLU) | — | TODO | Engine eval on trained weights not run |
| C10 | Adversarial probes pass | `docs/reproducibility/results/security_audit.json` | BACKED | 6/6 PASS (T29) |
