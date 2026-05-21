# Gap Ledger — Aurelius Integrations Loop v10

## Completed Slices (all sessions)

### Slice 1: react_loop memory integration (Current Session)
- **AMC Tier 2**: `react_loop.py` — observe fires on every assistant turn; Tier 2 hook called
- **AMC Tier 3**: Budget-exhaustion path promotes Tier 2 → LTM via `AMCTier3Hook.consolidate()`
- **Safety gate**: `SafetyAdmissionController` runs pre-LLM call; blocks recorded on `AgentTrace`
- **Counter fix**: `trace.tier2_calls` / `trace.tier2_writes` copied into success + budget paths
- **Tests**: `tests/agent/test_react_loop_memory_integration.py` (10 tests, 10 pass)
- **Tests**: `tests/agent/test_amc_safety_e2e.py` (8 tests, 8 pass)

### Slice 2: agent_mode_registry.refactor (Current Session)
- **Convenience API**: `get_mode()`, `build_system_prompt()`, `filter_tools()`, `CODE/ARCHITECT/ASK/DEBUG/CUSTOM_MODE` constants
- **Tests**: `tests/agent/test_agent_modes.py` (26 tests, 26 pass)

### Slice 3: src namespace port (Current Session)
- **Output compressor**: `src/cli/output_compressor.py` — fully ported, stdlib-only
- **Layered memory**: `src/memory/layered_memory.py` — fully ported, 5-layer TTL store
- **Progressive search**: `src/memory/progressive_search.py` — fully ported, 3-layer search
- **Tests**: `tests/cli/test_output_compressor.py` (21 tests, 21 pass)
- **Tests**: `tests/memory/test_layered_memory.py` (26 tests, 26 pass)
- **Tests**: `tests/memory/test_progressive_search.py` (34 tests, 34 pass)
- **Bugfix**: `progressive_search.py` — `search()` return sliced `[:top_k]` (not all candidates)

### Slice 4: legacy UTC cleanup (Current Session)
- **5 files fixed**: `episodic_memory.py`, `red_team.py`, `harness.py`, `transcript_viewer.py`, `amc_tier3.py`
- **Pattern**: `datetime.now(UTC)` → `datetime.now(timezone.utc)`

### Slice 5–9: Previous (from GAP_LEDGER history)
- skills_registry, agent_registry, api_registry, tool_schema_registry slices

## Test Suite

| Scope | Tests | Status |
|-------|------:|--------|
| agent/ (incl. react_loop, memory, e2e) | 2952 | pass |
| memory/ (incl. AMC, layered, progressive) | 278 | pass |
| safety/ (incl. admission controller) | 170 | pass |
| cli/ (output_compressor) | 21 | pass |
| serving/ (server, auth, guardrails…) | 1000+ | pass |
| **Total** | **4514** | **0 fail / 1 skip** |

## Remaining Gaps

### Completed this session
| Item | Status | Tests |
|------|--------|------:|
| `middle/src/routes/` (BFF stubs) | ✅ 25+ TS modules | 14 vitest |
| `deployment/` (compose + Helm) | ✅ 5 compose + chart | infra |
| `EpisodicMemory` dedup — session_id + step | ✅ store() dedup | +10 |
| `AgentMemoryBridgeContract._verify_write_shape` | ✅ multi-batch validator | +30 |

### Still open
| Priority | Item | Effort |
|-----------|------|--------:|
| Low | `rust_bridge.py` tests | small |
| Low | `brain_layer.py` tests | small–med |

## How to Continue
```
1. Pick next slice from "Remaining Gaps"
2. Run `pytest tests/agent tests/memory tests/safety tests/cli tests/serving -x -q`
3. Implement / port / fix
4. Run subset suite again; update this ledger
```
