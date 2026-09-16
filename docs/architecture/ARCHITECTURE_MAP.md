# Aurelius architecture map (import graph, generated)

**Generated:** 2026-09-16 · **Method:** static parse of every tracked Python import in `src/` and the root packages (`agent/`, `gateway/`, `cron/`, `tools/`, `middle/`, `aurelius_cli/`, `acp_adapter/`), relative imports resolved against the importing module's package. Regenerate with the command in §6.

## 1. Scale

- modules in scope: **2,473**
- modules with at least one resolved internal import: **665**
- resolved internal import edges: **1,807**
- **circular-dependency components: 17** (56 modules sit inside a cycle)

*Limits (stated so this map is not over-trusted):* this is a static parse. Imports constructed at runtime (`import_module(...)`, registry lookups, quoted module strings) are not edges here, and a handful of dynamic targets (~10) do not resolve. Treat the edge count as a **lower bound** and the cycle list as real but incomplete.

## 2. The most depended-upon modules (fan-in) — what everything else relies on

| dependents | module | lines | what it is |
|---|---|---|---|
| 162 | `src._compat` | 15 | compatibility shim — the single most connected module in the tree |
| 29 | `src.model.config` | 769 | model configuration objects |
| 22 | `src.model.transformer` | 549 | the transformer stack |
| 21 | `src.model.rms_norm` | 28 | normalisation primitive (tiny, widely reused — good) |
| 15 | `src.model.attention` | 332 | attention implementations |
| 15 | `src.persona.unified_persona` | 184 | persona registry |
| 13 | `src.model.interface_framework` | 1,332 | model interface layer (1.3k lines — a god file) |
| 12 | `src.eval` | 1,430 | eval package facade (1.4k lines — a god file) |
| 12 | `src.memory.amc_tier3` | 360 |  |
| 12 | `src.model.ffn` | 32 |  |
| 11 | `src.model.manifest` | 304 |  |
| 9 | `src.backends.base` | 151 |  |
| 9 | `src.memory.amc_tier2` | 263 |  |
| 9 | `src.memory.hlm_bank` | 292 |  |
| 8 | `src.computer_use.screen_parser` | 208 |  |
| 8 | `src.memory.sdb_runtime` | 584 |  |
| 8 | `src.memory.amc_runtime_cache` | 680 |  |
| 7 | `src.persona` | 83 |  |

## 3. The most coupled modules (fan-out) — what pulls in the most of the repo

| deps | module |
|---|---|
| 64 | `src.serving` |
| 63 | `src.agent` |
| 51 | `src.model` |
| 41 | `src.eval` |
| 31 | `src.training` |
| 28 | `src.safety` |
| 27 | `src.inference` |
| 26 | `src.tools` |
| 25 | `src.alignment` |
| 25 | `src.ui` |
| 24 | `src.longcontext` |
| 23 | `aurelius_cli.main` |

## 4. Circular dependencies (concrete refactor targets)

Each row is a strongly-connected component: every module in it can reach every other, so none can be imported in isolation. These are the highest-value elegance fixes because they force lazy imports and obscure the layering.

| size | modules |
|---|---|
| 6 | `src.training`, `src.training.ipo_trainer`, `src.training.mcts_rl_trainer`, `src.training.nce_objectives`, `src.training.offline_reward_modeling`, `src.training.token_credit_assignment` |
| 5 | `src.chat.security_personas`, `src.chat.threat_intel_persona`, `src.persona`, `src.persona.builtins`, `src.persona.persona_router` |
| 5 | `src.eval`, `src.eval.benchmark_runner`, `src.eval.multi_agent_debate_eval`, `src.eval.osworld_scorer`, `src.eval.reasoning_trace_eval` |
| 4 | `src.memory.amc_checkpoint`, `src.memory.sdb_persistent_log`, `src.memory.sdb_runtime`, `src.memory.state_reconstruction` |
| 4 | `src.serving.agentic_runtime`, `src.serving.api_server`, `src.serving.chat_session`, `src.serving.engine_loader` |
| 4 | `gateway.agentic_runtime`, `gateway.api_server`, `gateway.chat_session`, `gateway.engine_loader` |
| 4 | `src.inference`, `src.inference.adaptive_kv_eviction`, `src.inference.eagle3_decoding`, `src.inference.tree_of_thought` |
| 4 | `src.tools`, `src.tools.http_client`, `src.tools.shell_tool`, `tools` |
| 3 | `src.serving`, `src.serving.rate_limiter_v2`, `src.serving.task_api` |
| 3 | `src.alignment`, `src.alignment.kto_trainer`, `src.alignment.spin_trainer` |
| 2 | `agent.agent_persistence`, `agent.agent_runtime` |
| 2 | `agent.interface_runtime`, `agent.workflow_shell` |
| 2 | `aurelius_cli.debug_commands`, `aurelius_cli.main` |
| 2 | `src.agent.interface_runtime`, `src.agent.workflow_shell` |

### What the cycles mean here

- `src.serving.*` / `gateway.*` cycles are **mirror duplication showing up as structure**: the root `gateway/` package re-implements the `src/serving/` modules and they import each other. Consolidating the mirrors removes these cycles outright.
- `src.training.*`, `src.eval.*`, `src.inference.*`, `src.alignment.*`, `src.memory.*` cycles are self-inflicted: a package `__init__` importing its own submodules that import the package back. The fix is the standard one — keep `__init__` as a re-export facade with no logic, and have submodules import their siblings directly.
- `tools` ↔ `src.tools` is the same mirror problem in the tool layer.

## 5. How to navigate this codebase (the short version)

1. **Start at `src/_compat`.** 162 modules depend on it, so any behaviour question begins there.
2. **`src/model/` is the core stack** (`config` → `transformer` → `attention` → `rms_norm`/`ffn`); it is the most internally consistent, leaf-most part of the tree.
3. **`src/agent/` (63 outgoing package deps) and `src/serving/` (64)** are the two hubs that touch everything. They are where integration bugs live and where god files accumulate.
4. **The root packages are legacy mirrors**, not the canonical code: `agent/`, `gateway/`, `cron/`, `aurelius_cli/`, `tools/` duplicate `src/*` implementations (and `src.tools`/`src.tools.*` currently re-export *from* the root copy, so the dependency direction is inverted in places). Until they are consolidated, every search returns two hits for one concept — which is why this repo reads as larger than it is.

## 6. Regenerating this map

```bash
# from the repo root, with the project venv:
python - <<'PY'
# see tools/architecture_map.py — parses tracked imports, resolves relative imports,
# emits fan-in/fan-out tables and Tarjan SCCs for cycles.
PY
```
