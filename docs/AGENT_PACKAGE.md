# Agent package consolidation (H-03)

Aurelius historically shipped two Python agent trees:

| Package | Path | Status |
|---------|------|--------|
| Legacy import path | `agent/` | **Deprecated** — compatibility shims only |
| Canonical | `src/agent/` | **Preferred** for new code and security fixes |

## Shim policy

Modules that are fully migrated re-export from `src.agent` with a `DeprecationWarning`:

- `agent.react_loop` → `src.agent.react_loop`
- `agent.skill_library` → `src.agent.skill_library`
- `agent.absolute_zero` → `src.agent.absolute_zero`

Modules that still diverge (different line counts or behavior) remain in `agent/` until a parity review lands. **Do not copy security fixes to only one tree.**

Known divergent pairs (audit 2026-05-27):

- `interface_runtime.py`
- `plugin_sandbox.py`

`src/agent/code_execution_tool.py` is a shim **to** `agent.code_execution_tool` (implementation still lives under `agent/`).

## Import guidance

```python
# preferred
from src.agent.skill_library import VoyagerSkillLibrary

# legacy (emits DeprecationWarning)
from agent.skill_library import VoyagerSkillLibrary
```

## Completion criteria

- [ ] All duplicate modules are either deleted or thin shims
- [ ] `grep -R "from agent\."` limited to tests and explicit shims
- [ ] C-15 skill execution uses `_safe_exec` path exclusively
