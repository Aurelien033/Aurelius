# AMC Method Traceability (T33)

| Paper subsection | Primary source | Tests |
|------------------|----------------|-------|
| Tier-1 SSM | `src/model/mamba2_block.py`, `src/model/amc_ssm_layer.py` | `tests/model/test_mamba2_block.py` |
| Surprise | `src/model/amc_surprise.py` | `tests/model/test_amc_surprise.py` |
| Promotion ST | `src/model/amc_promotion.py` | `tests/model/test_amc_promotion.py` |
| SDB | `src/memory/sdb_runtime.py` | `tests/memory/test_sdb_memory_runtime.py` |
| Trust cache | `src/serving/amc_kv_cache.py` | `tests/serving/test_amc_kv_cache.py` |
| Constitutional | `src/agent/constitutional_memory.py` | `tests/agent/test_amc_agent_e2e.py` |

Straight-through identity in code: `store_hard - store_soft.detach() + store_soft` (`841182b7`).
