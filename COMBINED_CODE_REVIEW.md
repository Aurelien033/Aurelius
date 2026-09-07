# Aurelius — Combined Code Review and Gate-Fix Report

Generated: 2026-05-18 11:47:22 EDT

## Sources combined

1. `/Users/christienantonio/Desktop/Aurelius/CODE_REVIEW.md`
   - Full multi-agent code review.
   - Reported scope: gateway/API, ML pipeline, training, Rust crates, CLI, tests, hooks, migrations, CI/CD, and infrastructure.
   - Reported issue count: ~145 total: 24 critical, 69 high, 52 medium.

2. Current Hermes implementation-loop report
   - Focused on current-branch release gates and immediate fixes.
   - Fixed: SinkCache absolute memory-token preservation, ruff format failures, Bandit B608 false positive.
   - Verified: ruff, format, compileall, focused pytest, CI-focused pytest, and Bandit baseline all passed after the fixes.

## Executive summary

The two reports describe different layers of risk and should be treated as complementary:

- The attached full review is the strategic remediation backlog. It identifies broad correctness, security, CI, infrastructure, and architecture risks across the repository. The highest-priority remaining work is still the production API/auth posture, generation correctness, training correctness, command execution surfaces, production secret defaults, and supply-chain hardening.
- The implementation-loop report is the current-branch stabilization delta. It fixed release-blocking validation failures and one long-context cache correctness bug found during review, but it does not close the large critical/high backlog from the full review.
- Current practical next step: commit the six-file gate-fix batch separately, then begin the P0 unresolved remediation sequence below.

## Current known repo state from implementation-loop pass

- Repo: `/Users/christienantonio/aurelius`
- Branch: `main`, ahead of `origin/main` by 4 commits when checked.
- Working tree after gate-fix loop: modified but uncommitted.
- Modified files:
  - `src/inference/sink_cache.py`
  - `tests/inference/test_sink_cache.py`
  - `src/eval/amc_memory_benchmark.py`
  - `src/alignment/preference_optimization.py`
  - `tests/runtime/test_memory_quarantine.py`
  - `tests/safety/test_admission_controller.py`

## Consolidated priority map

### P0-A · Already fixed in the implementation-loop pass

| ID | Status | Area | Issue | Risk | Effort | Payoff | Validation |
|---|---|---|---|---|---|---|---|
| F-01 | Fixed, uncommitted | Inference / long-context | SinkCache dropped absolute memory tokens after compression | High | Low/Medium | High | Sink-cache tests + focused release tests passed |
| F-02 | Fixed, uncommitted | CI / formatting | Ruff format gate failed on four files | Medium | Low | Medium | CI-equivalent ruff format --check passed |
| F-03 | Fixed, uncommitted | Security gate | Bandit B608 false positive on natural-language AMC prompt | Medium | Low | Medium | Bandit baseline exited 0 |

Immediate action:
1. Review the six-file diff.
2. Commit this as a focused gate-fix commit before mixing it with security/ML remediation.
3. Keep the SinkCache regression test in any future CI-focused suite that covers long-context/AMC behavior.

### P0-B · Unresolved critical fixes from the full review

These are the merged top priority items after combining both reports. They should be fixed before treating the repo as release-safe.

| Order | Source IDs | Area | Issue | Risk | Effort | Payoff | Concrete next step |
|---:|---|---|---|---|---|---|---|
| 1 | C-05, C-06, H-01, H-02, H-04, H-06 | Gateway/API auth | Production FastAPI/gateway paths are unauthenticated or auth is bypassed/dead-code; defaults bind broadly | Critical | Medium/High | Very high | Wire shared auth dependency/middleware into FastAPI and gateway-specific routes; default bind to 127.0.0.1 unless auth is configured; add negative tests for unauthenticated requests |
| 2 | C-02 | ML generation | Top-p nucleus sampling is inverted in generate/generate_stream | Critical | Low | Very high | Patch sort direction/masking, add deterministic distribution test that proves high-prob tokens are retained and low-prob tail is masked |
| 3 | C-01 | ML model | MoE FFN branch drops residual connection | Critical | Low | Very high | Add residual in MoE path; add unit test comparing residual-preserving block behavior |
| 4 | C-15, C-22 | Middle scheduler | Scheduler stores and executes raw user-supplied shell commands through self-calling HTTP dispatch | Critical | Medium | Very high | Replace raw command string with allowlisted task type/schema; add malicious command rejection tests |
| 5 | C-18, C-16, C-17, C-19 | Production secrets/CORS | Production compose/Docker/Helm allow empty auth secrets or wildcard CORS | Critical | Medium | Very high | Fail closed using required env expansions; remove wildcard CORS defaults; inject JWT_SECRET from Kubernetes Secret |
| 6 | C-21, H-51, H-52, H-53, H-54 | CI/CD supply chain | Unpinned privileged release actions and risky deployment credential handling | Critical/High | Medium | High | Pin release actions to SHAs; restrict permissions; avoid echoing kubeconfig; review autofix workflow triggers |
| 7 | C-20, C-24 | Containers/K8s | Raw manifests and deployment Dockerfile run as root / lack security context | Critical | Low/Medium | High | Add non-root USER and securityContext matching the safer Helm chart |
| 8 | H-11, H-12, H-13, H-14 | Alignment training | DPO/GRPO paths treat model tuple output as bare logits and crash | High | Medium | High | Normalize model-output adapter and add DPO/GRPO smoke tests |
| 9 | C-03 | Training config | TrainConfig vocab default 8,192 diverges from AureliusConfig 128,000 | Critical | Low | High | Set default to 128,000 or derive from AureliusConfig; test config parity |
| 10 | C-10, C-09, H-28, H-29, M-29, M-38 | CLI / agent execution | Deny-list command safety, shell interpolation, and file editing are bypassable | Critical/High | High | Very high | Move to argv-only execution, workspace containment, confirmation policy, and real sandboxing |

### P1 · High-value structural fixes after P0

| Source IDs | Area | Issue | Risk | Effort | Payoff | Concrete next step |
|---|---|---|---|---|---|---|
| Implementation-loop P1 + consolidation docs | Package structure | Deprecated import shims are still exercised by tests and runtime imports | Medium/High | Medium | High | Replace remaining `agent`, `plugins`, `tools`, `gateway`, `aurelius` legacy imports with canonical `src.*` imports; keep shims only for external compatibility |
| Structural observations + H/M gateway items | Server architecture | Two overlapping server stacks have different security postures | High | High | Very high | Pick one canonical production API path; port proven auth/rate/SSRF controls into it; mark the other as legacy/internal |
| Structural observations + M-17 | Checkpoints | Two incompatible checkpoint formats and naming schemes | Medium/High | Medium | High | Define one checkpoint manifest schema and add migration/discovery compatibility tests |
| M-18, M-19, M-20 | Dependencies | Invalid or platform-hostile dependencies (`torch>=2.11.0`, base CUDA-only deps, optional-but-required accelerate) | Medium/High | Low/Medium | High | Split CPU/Mac/CUDA extras; pin valid torch constraints; make training deps explicit |
| H-18–H-27, M-21–M-28 | Rust | Panics/expect/unwrap-like behavior and unbounded restore paths across FFI/service boundaries | High | Medium/High | High | Replace `expect()` on external/user data with typed errors; add malformed input and corrupted persistence tests |
| H-39, M-45, M-46, M-49, M-51 | Tests/CI | Several tests assert weak conditions or Makefile targets hide failures | Medium/High | Medium | High | Replace vacuous assertions, clean monkeypatch pollution, remove `|| true` from Rust test/lint targets |
| Bandit output observation | Security logs | Bandit exits 0 but emits noisy malformed `nosec` comment warnings | Low/Medium | Low/Medium | Medium | Normalize all nosec comments to exact IDs and concise rationales |

## Recommended remediation sequence

### Phase 0 — Land the verified gate fixes

Scope:
- `src/inference/sink_cache.py`
- `tests/inference/test_sink_cache.py`
- `src/eval/amc_memory_benchmark.py`
- format-only files from ruff

Acceptance:
- The validation commands listed in the implementation-loop report still pass.
- Commit message states why SinkCache absolute-token tracking matters for AMC/long-context correctness.

### Phase 1 — Close production-auth and command-execution risks

Fix first:
1. `gateway/aurelius_api.py` auth dependency/middleware on all non-probe routes.
2. `gateway/aurelius_server.py` fail-closed auth defaults and route ordering so `_require_auth()` is not bypassed.
3. `middle/src/routes/scheduler.ts` task schema/allowlist and remove self-calling command execution path.
4. `aurelius_cli/agent_engine.py` shell interpolation and deny-list bypasses.
5. Production compose/Docker/Helm secret and CORS fail-closed defaults.

Acceptance tests:
- Unauthenticated requests fail for all non-probe routes.
- Authenticated requests succeed with valid API keys/tokens.
- Malicious scheduler commands are rejected with 400.
- CLI execution uses argv-style APIs and rejects paths outside the workspace.
- Production manifests fail if required secrets are unset.

### Phase 2 — Fix generation/training correctness

Fix first:
1. Top-p sampling in `src/model/transformer.py`.
2. MoE residual connection in `src/model/transformer.py`.
3. `TrainConfig.model_vocab_size` default and config parity.
4. PPO clipped surrogate sign/min objective.
5. DPO/GRPO model-output tuple handling.
6. Tokenized loader view/copy issue.

Acceptance tests:
- Deterministic top-p distribution tests.
- MoE block preserves residual path.
- Training config default matches model config.
- PPO objective test against a known hand-computed example.
- DPO/GRPO forward smoke tests do not crash on tuple outputs.
- Tokenized loader labels/inputs cannot mutate each other accidentally.

### Phase 3 — Harden infra/CI and Rust boundaries

Fix first:
1. Pin release actions to full commit SHAs.
2. Add securityContext/USER to raw K8s and deployment Dockerfile.
3. Remove hidden `|| true` Rust lint/test failures from Makefile.
4. Replace Rust panics on external data with recoverable errors.
5. Add size/count limits to persistence load/restore paths.

Acceptance tests:
- CI fails if Rust lint/tests fail.
- Containers run non-root.
- Corrupted Rust persistence data returns an error, not a process abort.
- Release workflow permissions are minimal and actions are pinned.

### Phase 4 — Reduce architectural drift

Fix first:
1. Canonicalize package imports to `src.*` internally.
2. Consolidate dual server stacks or explicitly demote one to legacy.
3. Consolidate checkpoint schemas.
4. Split platform-specific dependency extras.
5. Clean Bandit `nosec` comment noise.

Acceptance tests:
- No DeprecationWarning from internal test imports.
- One documented production server path with auth/security tests.
- Checkpoints can be discovered and resumed by one canonical manager.
- CPU/Mac dev install works without CUDA-only dependencies.
- Bandit logs are low-noise and findings are actionable.

## Verification status matrix

| Area | Combined status |
|---|---|
| Gate-fix branch validation | Passed after implementation-loop fixes |
| Full attached report findings | Not re-verified one-by-one during this combine step; treat line numbers as review evidence requiring confirmation before patching |
| Security posture | Critical unresolved items remain, especially auth/defaults/command execution |
| ML correctness | Critical unresolved items remain, especially top-p, MoE residual, vocab default, PPO/DPO/GRPO |
| CI/release | Critical/high unresolved items remain around release workflow pinning and hidden failures |
| Architecture | Significant drift remains: dual server paths, shims still exercised, checkpoint split |

## Appendix A — Full attached code review report

The following is preserved from `/Users/christienantonio/Desktop/Aurelius/CODE_REVIEW.md`.

# Aurelius — Full Code Review Report

**Reviewed:** 2026-05-18  
**Scope:** All source code across gateway, ML pipeline, training, Rust crates, CLI, tests, hooks, migrations, CI/CD, and infrastructure  
**Agents:** 5 parallel reviewers (gateway/API, ML/training, Rust/CI/infra, core Python, tests/hooks/config)

**Total issues: ~145**
- CRITICAL: 24
- HIGH: 69
- MEDIUM: 52

---

## CRITICAL ISSUES

### C-01 · `src/model/transformer.py:111` · BUG · ML
**MoE branch missing residual connection**

The Sparse MoE path replaces `x` with raw FFN output instead of adding to it. Every MoE layer loses residual learning entirely. Training silently converges to a much worse solution.

```python
# WRONG — current code
x, aux_loss = self.ffn(self.ffn_norm(x))

# FIX
ffn_out, aux_loss = self.ffn(self.ffn_norm(x))
x = x + ffn_out
```

---

### C-02 · `src/model/transformer.py:325-330,385-390` · LOGIC_ERROR · ML
**Top-p nucleus sampling is completely inverted**

Sorts ascending instead of descending, then uses `(1 - top_p)` as the keep threshold. With `top_p=0.9`, the model samples from the bottom 10% of the distribution. Produces near-random garbage output. Bug is present in both `generate()` and `generate_stream()`.

```python
# WRONG — current code
sorted_logits, sorted_indices = torch.sort(next_logits, descending=False)
cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
sorted_mask = cumulative_probs <= (1.0 - top_p)

# FIX
sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
sorted_mask = cumulative_probs - sorted_logits.softmax(dim=-1) >= top_p
sorted_mask[..., 0] = False  # always keep at least 1 token
mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
next_logits = next_logits.masked_fill(mask, float("-inf"))
```

---

### C-03 · `src/training/trainer.py:226` · BUG · ML
**`TrainConfig.model_vocab_size` defaults to 8,192 vs `AureliusConfig` default of 128,000**

Any training run without explicit vocab config creates an `lm_head` of shape `[2048, 8192]` instead of `[2048, 128000]`. Any token ID >= 8,192 causes an `IndexError` or silent wrong prediction.

```python
# FIX
model_vocab_size: int = 128_000
```

---

### C-04 · `src/training/rlhf.py:247-249` · LOGIC_ERROR · ML
**PPO uses `torch.max` instead of `torch.min` for clipped surrogate**

Standard PPO objective is `E[min(ratio*A, clip(ratio)*A)]`. Using `torch.max` produces the wrong pessimistic bound when advantages are negative. Policy gradient update is incorrect.

```python
# WRONG — current code
policy_loss = torch.max(loss_unclipped, loss_clipped).mean()

# FIX
surr1 = ratio * advantages
surr2 = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages
policy_loss = -torch.min(surr1, surr2).mean()
```

---

### C-05 · `gateway/aurelius_api.py:579-585` · AUTH_BYPASS · Gateway
**FastAPI server binds `0.0.0.0` with zero authentication**

Every endpoint (`/v1/chat/completions`, `/workspaces`, `/sessions`, `/audit`, WebSocket `/ws`) has no auth. `auth_middleware.py` is well-written but only wired into the legacy stdlib `api_server.py`. The FastAPI server — the production entrypoint — uses none of it.

Fix: Apply `DEFAULT_AUTH_MIDDLEWARE` as a FastAPI dependency on all non-probe routes. Change default host to `127.0.0.1`.

---

### C-06 · `gateway/aurelius_server.py:1163-1221` · AUTH_BYPASS · Gateway
**`require_auth` defaults to `False`**

Server starts entirely unauthenticated by default. CLI `__main__` also defaults `--host 0.0.0.0`. Any host on the network can call all API endpoints.

Fix: Default `require_auth` to `True`. Never default to `0.0.0.0` without confirmed auth.

---

### C-07 · `gateway/aurelius_server.py:1025-1035` · AUTH_BYPASS / TOKEN_LEAK · Gateway
**License activation is unauthenticated and trivially bypassable**

`/api/license/activate` runs before `_require_auth()`. Any string starting with `"AURELIUS-"` and >= 32 chars passes validation. The last 16 chars of the key become the API key — fully predictable.

Attack: POST `/api/license/activate` with `{"key": "AURELIUS-" + "A" * 23}`. Server sets `api_key = "AAAAAAAAAAAAAAAA"` and `license_activated = True`.

Fix: Require a pre-shared bootstrap secret. Use a random, independently-generated API key.

---

### C-08 · `gateway/aurelius_api.py:300-304` · INJECTION · Gateway
**Arbitrary path injection via `/workspaces`**

`req.path` stored without validation, sanitization, or path confinement. Downstream code using this path is fully attacker-controlled. Includes `..` traversal, absolute paths, symlinks.

```python
# FIX — validate and confine to allowed root
resolved = Path(req.path).resolve()
if not str(resolved).startswith(str(ALLOWED_WORKSPACE_ROOT)):
    raise HTTPException(403, "Path not allowed")
```

---

### C-09 · `aurelius_cli/agent_engine.py:265-266` · INJECTION · Core Python
**Shell injection in `search_code()` via f-string interpolation**

User-supplied `pattern` and `path` directly interpolated into a shell command string. A crafted pattern value can escape the quote and inject arbitrary shell commands.

```python
# WRONG
return self.execute(f"grep -rn '{pattern}' {path}")

# FIX — pass as argv list, never interpolate user input
return self.execute_argv(["grep", "-rn", pattern, path])
```

---

### C-10 · `aurelius_cli/agent_engine.py:162-205` · SECURITY · Core Python
**DENY_LIST is trivially bypassed; Python execution explicitly allowed**

`DENY_LIST = {"rm -rf /", "dd if=", ...}` bypassed by double space, different case, or `bash -c` wrapper. The `_is_python_launcher` exception explicitly permits arbitrary Python one-liners via `python3 -c`.

Fix: Remove the DENY_LIST entirely — it provides false security. Remove the `_is_python_launcher` exception. Use a real sandbox (Docker, seccomp, bubblewrap).

---

### C-11 · `aurelius_cli/agent_engine.py:469-476` · SECURITY / INJECTION · Core Python
**Arbitrary session file loaded via `Message(**m)` from untrusted JSON**

`_load_session()` deserializes a JSON file from an arbitrary user-supplied path with no canonicalization. Passes all JSON keys as constructor arguments via `Message(**m)` — path traversal combined with type confusion.

```python
# FIX — validate every field explicitly
Message(
    role=str(m.get("role", "user"))[:50],
    content=str(m.get("content", "")),
    tool_calls=[],
)
```

---

### C-12 · `aurelius/api_registry.py:28-45` · BUG · Core Python
**`ImportError` fallback missing 4 names — raises `NameError` in fallback path**

The `except ImportError` block defines most names but omits `AGENT_CATEGORIES`, `SKILL_CATEGORIES`, `agent_to_dict`, `skill_to_dict`. Any call to `get_registry_snapshot()` in the fallback path immediately raises `NameError`.

Fix: Add the missing names to the fallback block.

---

### C-13 · `aurelius_cli/main.py:960-965` · LOGIC_ERROR · Core Python
**`chat` command routing inverted — `--model-path` silently ignored**

Logic bug: `args.command is None AND args.command != "chat"` — the second clause is always `True` when command is None, so `not model_path OR True` always launches the terminal, ignoring any provided model path.

Fix:
```python
if args.command is None:
    run_terminal(); return 0
if args.command == "chat":
    if not getattr(args, "model_path", None):
        run_terminal(); return 0
    # load model and run _run_chat
```

---

### C-14 · `alembic/versions/0001_initial_schema.py:127-135` · DATA_LOSS · Migrations
**`downgrade()` drops all 8 tables with wrong FK ordering and no data preservation**

`activity` has a FK to `agents`, but `agents` is dropped first. On strict PostgreSQL, this errors. No indexes dropped before tables. No data export step.

Fix: Drop child tables (FK holders) before parent tables. Drop all indexes before dropping tables.

---

### C-15 · `middle/src/routes/scheduler.ts:83-99` · INJECTION · Tests/Middleware
**Scheduler executes raw user-supplied shell commands via self-calling HTTP SSRF**

The scheduler stores `command` from the task creation API verbatim, then POSTs it to its own `/api/command` endpoint. Any `scheduler:admin` user can schedule an arbitrary OS command as a cron task.

Route handler only checks `if (!command)` — any non-empty string passes. The self-calling HTTP dispatch means the command executes with the server's OS privileges.

Fix: Add a command allowlist regex before storing. Eliminate the self-calling HTTP dispatch pattern entirely.

---

### C-16 · `Dockerfile:56` · SECRET_EXPOSURE · Infrastructure
**Wildcard CORS baked into Docker image as `ENV CORS_ORIGINS=*`**

```dockerfile
ENV CORS_ORIGINS=*  # baked into every container from this image
```

Fix: Remove from `ENV`. Require explicit operator configuration.

---

### C-17 · `docker-compose.yml:17` · SECRET_EXPOSURE · Infrastructure
**Wildcard CORS hardcoded in production compose**

```yaml
environment:
  CORS_ORIGINS: "*"
```

Fix: Use `${CORS_ORIGINS:?CORS_ORIGINS must be set}`.

---

### C-18 · `deployment/compose.production.yaml:47` · SECRET_EXPOSURE · Infrastructure
**Empty `JWT_SECRET` and `AURELIUS_API_KEY` defaults in production compose**

```yaml
- AURELIUS_API_KEY=${AURELIUS_API_KEY:-}   # empty string default = no auth
- JWT_SECRET=${JWT_SECRET:-}               # empty string default = no JWT auth
```

Fix: Use `${AURELIUS_API_KEY:?AURELIUS_API_KEY must be set for production}`.

---

### C-19 · `deployment/helm/templates/gateway-deployment.yaml` · SECRET_EXPOSURE · Infrastructure
**`JWT_SECRET` completely absent from Helm gateway deployment template**

The Rust gateway will panic at startup (`JWT_SECRET must be set`). No `JWT_SECRET` env var injected anywhere in the template, and no `gateway.jwtSecret` in `values.yaml`.

Fix: Add a `secretKeyRef` injection for `JWT_SECRET` in the template and a required value in `values.yaml`.

---

### C-20 · `k8s/aurelius-deployment.yaml` · PRIVILEGE_ESCALATION · Infrastructure
**No security context — runs as root with all capabilities**

The raw k8s manifests have zero `securityContext`. Runs as UID 0, `allowPrivilegeEscalation: true`, no capability drops, no seccomp. The Helm chart correctly sets `runAsNonRoot: true`, `allowPrivilegeEscalation: false`, `capabilities: drop: ["ALL"]` — the raw manifests have none of this.

Fix:
```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  capabilities:
    drop: ["ALL"]
```

---

### C-21 · `.github/workflows/release.yml:16-17,44-45` · SUPPLY_CHAIN · CI/CD
**Unpinned GitHub Actions with `contents: write` + PyPI publish permissions**

Every other workflow in this repo pins to full SHA. `release.yml` uses `@v4` / `@v5` tags for all actions including `pypa/gh-action-pypi-publish`. Tag hijacking enables supply chain compromise of the PyPI package.

Fix: Pin every action to its full commit SHA, as done in `ci.yml`, `cd.yml`, and `rust.yml`.

---

### C-22 · `middle/__tests__/scheduler.test.ts:31,41-43` · INJECTION · Tests
**Test validates that command injection works, not that it is blocked**

Tests create tasks with real shell command strings demonstrating the injection surface is functional. Zero tests verify that malicious commands are rejected with 400.

Fix: Add negative tests. Fix the underlying route validation first (see C-15).

---

### C-23 · `gateway/aurelius_server.py:171-185` · SECURITY · Gateway
**`X-API-Key` compared with `==` not `hmac.compare_digest()` — timing side-channel**

Plain `==` on secret strings leaks timing. `X-Session-Token` uses set `in` membership — also timing-leaky via hash collisions.

```python
# FIX
import hmac
if expected and hmac.compare_digest(api_key.encode(), expected.encode()):
    return True
```

---

### C-24 · `deployment/Dockerfile` · PRIVILEGE_ESCALATION · Infrastructure
**Production Python Dockerfile has no `USER` directive — runs as root UID 0**

The root `Dockerfile` creates `appuser` correctly. The `deployment/Dockerfile` (used by `compose.production.yaml` and `cd.yml`) has no `USER` instruction. Any RCE in the Python app has immediate root access.

Fix:
```dockerfile
RUN useradd --create-home --uid 1000 --no-log-init appuser
USER appuser
```

---

## HIGH ISSUES

### Authentication & Authorization

| # | File | Lines | Issue |
|---|------|-------|-------|
| H-01 | `gateway/aurelius_server.py` | 203–265 | All specific `/api/*` routes match and return before the catch-all `_require_auth()` check — auth is dead code for all specific routes |
| H-02 | `gateway/aurelius_api.py` | 362–379 | WebSocket `/ws` has no auth, no origin check, no message size limit |
| H-03 | `gateway/hermes_notifier.py` | 247–280 | SSRF: webhook URL is user-controlled, no IP blocklist, no TLS verification — can point at metadata endpoints |
| H-04 | `gateway/aurelius_server.py` | 1060–1074 | Authenticated users can POST `{"require_auth": false}` to `/api/config` — self-disabling auth |
| H-05 | `gateway/aurelius_server.py` | 130–136 | Path traversal: `".." in path` check bypassed by `%2e%2e`, Unicode variants, absolute paths |
| H-06 | `gateway/api_server.py` | 304–358 | `/metrics`, `/health`, `/openapi.json` served before auth check — exposes topology and Prometheus data |
| H-07 | `gateway/aurelius_api.py` | 494, 551 | `raise HTTPException(500, f"Generation error: {exc}")` leaks internal paths, CUDA device IDs, weight paths |
| H-08 | `gateway/cors_middleware.py` | 62 | Module-level `CORS = CORSMiddleware()` instantiated with `allowed_origins=""` — CORS broken by default with no warning |
| H-09 | `gateway/aurelius_server.py` | 1080–1087 | SSE endpoint hardcodes `Access-Control-Allow-Origin: *` regardless of configured CORS policy |

### ML / Training

| # | File | Lines | Issue |
|---|------|-------|-------|
| H-10 | `src/data/tokenized_loader.py` | 76–78 | `input_ids` and `labels` are views of the same numpy buffer — in-place ops corrupt both tensors simultaneously |
| H-11 | `src/training/dpo_trainer.py` | 198–201 | `policy_model()` expected to return bare logits tensor, actually returns `(loss, logits, kv)` tuple — crashes immediately on `.shape` |
| H-12 | `src/training/dpo_trainer.py` | 137 | Same tuple return bug in `ReferenceModelManager.compute_logps` |
| H-13 | `src/training/grpo_trainer.py` | 172–174 | `GroupSampler.sample_group` calls `self.model(cur_ids)` and uses result as tensor directly — same tuple crash |
| H-14 | `src/training/grpo_trainer.py` | 314–316 | `_sequence_log_prob` also calls model expecting tensor — same crash |
| H-15 | `src/model/mtp.py` | 114–118 | MTP cross-entropy missing `ignore_index=-100` — crashes or produces wrong gradients on any padding token |
| H-16 | `src/training/mixed_precision.py` | 149–157 | `.model.to(dtype)` called in-place on every forward pass — two full-model dtype casts per step, destroys tied embeddings, corrupts gradient state |
| H-17 | `src/training/trainer.py` | 887–900 | Multi-GPU resume loads into unwrapped model; Muon optimizer state key alignment risk on resume |

### Rust

| # | File | Lines | Issue |
|---|------|-------|-------|
| H-18 | `crates/usage-store/src/lib.rs` | 52 | `expect()` across PyO3 FFI boundary on serialization — panic is UB across FFI, aborts Python process |
| H-19 | `crates/usage-store/src/lib.rs` | 94 | `expect()` on deserialization of DB record across FFI — corrupted record panics and aborts process |
| H-20 | `crates/usage-store/src/lib.rs` | 190 | `expect()` on JSON parse of Python-supplied dict across FFI — `NaN`/`Infinity` from Python causes abort |
| H-21 | `crates/data-engine/src/engine.rs` | all lock sites (~13) | All `.expect("Failed to acquire ...lock")` — lock poisoning from any thread panic aborts Node.js process |
| H-22 | `crates/api-gateway/src/proxy.rs` | 81 | `.expect("Failed to build response")` on user-controlled upstream data — DoS via crafted response |
| H-23 | `crates/api-gateway/src/main.rs` | 21–35 | `Config::from_env()` called twice; `.expect("Invalid log level")` panics before JWT secret validated |
| H-24 | `crates/data-engine/src/persistence.rs` | 371–538 | `load()` restores data with no count limits — crafted save file exhausts all process memory |
| H-25 | `crates/api-gateway/src/auth.rs` | 63 | `value[7..]` byte-slice on original str after lowercase check — structural UTF-8 panic risk |
| H-26 | `crates/json-validator/src/lib.rs` | 382–395 | User-supplied regex compiled on every validation call — no cache, ReDoS vector |
| H-27 | `crates/search-index/src/lib.rs` | 96–161 | Check-then-act on DashMap without atomic lock — concurrent indexing produces inconsistent state |

### Core Python CLI

| # | File | Lines | Issue |
|---|------|-------|-------|
| H-28 | `aurelius_cli/agent_engine.py` | 139–153 | `verify_equivalence()` executes two arbitrary shell commands bypassing all safety checks |
| H-29 | `aurelius_cli/agent_engine.py` | 243–253 | `edit_file()` has no path containment — AI planner can overwrite sensitive files outside workspace |
| H-30 | `agent/self_upgrade.py` | 142–150 | Default `test_fn` and `safety_fn` return `True` — any injected `deploy_fn` deploys code with no validation |
| H-31 | `aurelius/api_registry.py` | 52–54, 96–98 | `app.request.args.get()` — wrong Flask API, raises `AttributeError` on every GET request |
| H-32 | `aurelius_cli/history_manager.py` | 36–55, 74–78 | `_load()`, `search()`, `__len__()` read shared list without holding lock — race condition |
| H-33 | `scripts/aurelius_state.py` | 21–29 | Singleton `get_state()` has no lock — check-then-set race on multi-threaded init |
| H-34 | `aurelius_cli/main.py` | 348–377 | Spinner thread catches `BaseException` and re-raises on main thread — `SystemExit` kills entire process |
| H-35 | `aurelius_cli/agent_engine.py` | 186–188 | `ToolExecutor.history` is unbounded list — ~15 MB per 1000 tool calls, never evicted |
| H-36 | `aurelius_cli/agent_engine.py` | 108–109 | `NL2SHEngine._cache` is unbounded dict — arbitrary memory growth over long sessions |

### Tests / Middleware

| # | File | Lines | Issue |
|---|------|-------|-------|
| H-37 | `middle/src/middleware/auth.ts` | 15–19 | API keys stored in plaintext in-memory Map — heap dump / `--inspect` exposes all active keys |
| H-38 | `middle/src/middleware/rate-limiter.ts` | 13 | Rate limit key uses `X-Forwarded-For` first — client-spoofable, enables bypass or DoS-ing other users |
| H-39 | `middle/__tests__/auth.test.ts` | 122–128 | `expect(res.status === 200 \|\| res.status === 401)` has no matcher — always passes, Bearer auth path untested |
| H-40 | `alembic/versions/0001_initial_schema.py` | 73 | `memory_entries.timestamp` is `String(32)` not `DateTime`/`Float` — SQL ordering and range queries broken |
| H-41 | `alembic/env.py` | 15 | `target_metadata = None` — Alembic autogenerate completely non-functional, schema drift undetectable |
| H-42 | `alembic.ini` | 3 | Database URL hardcoded — production credentials would be committed if this file is reused |
| H-43 | `configs/api_keys.example.json` | 4 | `.env.example` points at example key file; developer copying `.env.example` gets no-auth startup silently |
| H-44 | `.env.example` | 14 | `AURELIUS_API_KEY=` empty string — server starts with no valid API key, every request returns 401, no warning |
| H-45 | `Makefile` | 78, 177 | `$(checkpoint)`, `$(output)`, `$(message)` directly interpolated — shell injection via make variables |
| H-46 | `Makefile` | 35 | `serve` target binds `0.0.0.0` with no API key guard |
| H-47 | `tests/test_adversarial_permission.py` | 34–37 | `"override" not in inj` exempts `"SYSTEM OVERRIDE: ..."` injection from being checked |
| H-48 | `tests/test_security_audit.py` | 133–137 | `assert passed == 6` hardcoded — test passes for wrong reasons due to state reset ordering |
| H-49 | `tests/test_legacy_compatibility.py` | 12–15 | `len(snapshot["agents"]) == 22` hardcoded — adding/removing any agent breaks CI |
| H-50 | `tests/test_arxiv_modules.py` | 812–827 | `sys.modules["sklearn"] = MagicMock()` never cleaned up — corrupts all subsequent test process imports |

### CI/CD & Infrastructure

| # | File | Lines | Issue |
|---|------|-------|-------|
| H-51 | `.github/workflows/deploy.yml` | 18–30 | Registry credentials in `env:` block — visible to all subprocesses in job |
| H-52 | `.github/workflows/cd.yml` | 77–88 | Kubeconfig written as `echo "$KUBE_CONFIG" > ~/.kube/config` — exposed in debug step logs |
| H-53 | `.github/workflows/cd.yml` | 78 | `version: 'latest'` for Helm install — non-deterministic binary, CDN compromise risk |
| H-54 | `.github/workflows/ruff-autofix.yml` | 11–12 | Autofix workflow has `contents: write` + `pull-requests: write`, runs on any contributor's push |
| H-55 | `docker-compose.yml` (root) | entire | No CPU/memory resource limits on `aurelius-api` — runaway request exhausts host |
| H-56 | `deployment/compose.dev.yaml` | 37 | `AURELIUS_API_KEY=dev-key` hardcoded in version-controlled file |
| H-57 | `deployment/compose.production.yaml` + `values.yaml` | — | Images use mutable version tags, not digest-pinned — acknowledged TODO, unresolved |

---

## MEDIUM ISSUES

### Gateway / API

| # | File | Lines | Issue |
|---|------|-------|-------|
| M-01 | `gateway/rate_limiter.py` | 51–56 | Burst > 1 never refills tokens — `DEFAULT_RATE_LIMITER` (burst=200) effectively blocks forever after burst window |
| M-02 | `gateway/rate_limit.py` | 34–49 | `MemoryRateLimiter.allow()` has no lock — TOCTOU race, dict-mutation-during-iteration crash under load |
| M-03 | `gateway/hermes_notifier.py` | 205–216 | `_listeners` list mutated without lock during `notify()` iteration — `IndexError` or double/missed delivery |
| M-04 | `gateway/aurelius_server.py` | 472–485 | `limit=` parameter not bounded — `limit=999999999` forces serialization of millions of entries (DoS) |
| M-05 | `gateway/aurelius_server.py` | 984–1003 | Same unbounded `limit` in `/api/logs` — plus log entries may contain accidentally-logged secrets |
| M-06 | `gateway/acp_adapter/mcp_client.py` | 205–215 | `{"method": method, **params}` — `params` key named `"method"` silently overrides intended method |
| M-07 | `gateway/aurelius_api.py` | 263 | `audit_log` is uncapped list, exposed unauthenticated — unbounded memory + info disclosure |
| M-08 | `gateway/aurelius_api.py` | 263, 312–316 | `sessions` dict has no cap, TTL, or lock — unauthenticated endpoint creates unbounded sessions |
| M-09 | `gateway/aurelius_api.py` | 186–192 | Rate limiter allows all requests while `_rate_limiter is None` during model load (10–60 sec window) |
| M-10 | `gateway/aurelius_api.py` | 468–472 | `_sanitize_completion` drops all Unicode, no HTML escaping — not a security control |
| M-11 | `gateway/mission_control.py` | 329–333 | `escapeHtml` JS function has broken string literals — non-functional, server IDs injected raw to `innerHTML` (XSS) |
| M-12 | `gateway/engine_loader.py` | 94 | `trust_remote_code=True` on user-supplied model path — arbitrary code execution at tokenizer load time |
| M-13 | `gateway/web_ui.py` | 43–69 | SSRF IP blocklist checks only IP literals, not hostnames — DNS rebinding bypasses the guard |
| M-14 | `gateway/aurelius_api.py` | 441 | `print("[startup] Engine load failed: {exc}")` — not an f-string, literal `{exc}` printed, actual error swallowed |

### ML / Training

| # | File | Lines | Issue |
|---|------|-------|-------|
| M-15 | `src/model/moe.py` | 465–470 | `SoftMoELayer.forward` calls `TopKRouter._softmax` which does not exist — `AttributeError` at runtime |
| M-16 | `src/training/sequence_packing.py` | 373–380 | Attention mask wrong for sequences containing `token_id == pad_token_id` as legitimate content |
| M-17 | `src/training/trainer.py` + `checkpoint.py` | — | Two incompatible checkpoint systems: `step-{n:07d}` vs `checkpoint-{n:07d}`, different metadata formats |
| M-18 | `pyproject.toml` | 7 | `torch>=2.11.0` — this version does not exist (latest ~2.4.x), `pip install` fails |
| M-19 | `pyproject.toml` | 45 | `sageattention>=2.0.0` in base deps — requires CUDA, fails on CPU/Apple Silicon machines |
| M-20 | `pyproject.toml` | 15 | `accelerate>=1.0.0` in optional extras but required by core training loop |

### Rust

| # | File | Lines | Issue |
|---|------|-------|-------|
| M-21 | `crates/api-gateway/src/metrics.rs` | 101 | `partial_cmp(b).expect()` panics on `NaN` latency values — filter NaN before sort |
| M-22 | `crates/api-gateway/src/auth.rs` | 40–43 | `Validation::default()` may not enforce `exp` claim or restrict algorithms — use explicit `Validation::new(Algorithm::HS256)` |
| M-23 | `crates/api-gateway/src/rate_limit.rs` | 30–38 | Background cleanup task holds strong `Arc` reference — RateLimiter drop causes resource leak (no cancellation) |
| M-24 | `crates/api-gateway/src/routes/health.rs` | 50–53 | Readiness handler makes outbound HTTP call with no timeout — hangs under slow upstream, blocks worker pool |
| M-25 | `crates/data-engine/src/persistence.rs` | 365 | No file size limit on `fs::write` of JSON — can exhaust disk space |
| M-26 | `crates/json-validator/src/lib.rs` | 416 | `unwrap_or(Value::Null)` silently treats malformed JSON as null — misleading stats |
| M-27 | `crates/text-processor/src/lib.rs` | 407 | `Regex::new(r"\s+")` compiled on every call to `normalize_whitespace` — use `LazyLock<Regex>` |
| M-28 | `crates/token-counter/src/lib.rs` | 34–48 | All `as u32` casts silently truncate on inputs > 4 billion bytes |

### Core Python CLI

| # | File | Lines | Issue |
|---|------|-------|-------|
| M-29 | `aurelius_cli/agent_engine.py` | 325–378 | Keyword match (`"plan"`, `"run"`) triggers shell execution without confirmation — conversational input misclassified |
| M-30 | `aurelius_cli/config.py` | 27 | `int(os.environ.get("AURELIUS_MAX_HISTORY", "1000"))` raises `ValueError` on non-integer env var at import time |
| M-31 | `aurelius_cli/history_manager.py` | 25–29 | Path traversal check uses AND logic — absolute paths (e.g., `/etc/passwd`) bypass both conditions |
| M-32 | `aurelius_cli/main.py` | 256–266 | Byte-level tokenizer fallback silently used for 128K-token model — produces garbage output with no error |
| M-33 | `aurelius/api_registry.py` | 134–163 | `confidence` score is integer / capability count — not bounded to `[0, 1]`, callers treating as probability are wrong |
| M-34 | `aurelius/_self_upgrade_impl.py` | 33–36 | Improvement score formula breaks for zero and near-zero targets |
| M-35 | `scripts/batch_runner.py` | 6–8 | `run_batch()` always raises `NotImplementedError` — `__main__` block crashes with unhandled exception |
| M-36 | `aurelius_cli/main.py` | 279–297 | `strict=False` on `load_state_dict` silently loads mismatched checkpoint — randomly-initialized missing layers |
| M-37 | `agent/neural_brain.py` | 112–113 | Memory operation errors propagate to abort full cognitive cycle instead of degrading gracefully |
| M-38 | `aurelius_cli/agent_engine.py` | 325–378 | `execute_plan()` continues executing subsequent steps after a failed step — cascading state corruption |

### Tests / Hooks / Config

| # | File | Lines | Issue |
|---|------|-------|-------|
| M-39 | `hooks/check_torch_load.py` | 22 | `"tests" in path.parts` matches any directory named `tests` anywhere in the path — hook bypass |
| M-40 | `hooks/check_torch_load.py` | 26–27 | `except Exception: continue` silently swallows all read errors — unreadable files pass check |
| M-41 | `hooks/check_torch_load.py` | 14 | Hook checks for explicit `weights_only=False` but not the omission case (`torch.load(path)` with no arg) |
| M-42 | `configs/config.yaml` | 39–41, 65–67 | `False`/`True` (Python-style) instead of `false`/`true` — YAML 1.2 parsers treat as string, evaluates truthy |
| M-43 | `tests/test_cache.py` | 8–12 | `test_semantic_cache_hit` relies on implementation-specific similarity threshold — fragile |
| M-44 | `tests/test_cache.py` | 49–53 | `test_lru_cache_ttl` uses real `time.sleep(0.005)` — inherently flaky under CI load |
| M-45 | `tests/test_arxiv_modules.py` | 695–699 | `test_two_pathway_ood_detector_detect` body is entirely dead — no assertions, no method calls |
| M-46 | `tests/test_cli_integration.py` | 115–118 | `assert returncode == 0 or returncode == 1` — always true, tests nothing |
| M-47 | `tests/test_transformer_hybrid.py` | 22–26 | No assertion that generated token IDs are within vocab bounds `[0, vocab_size)` |
| M-48 | `middle/src/routes/auth.ts` | 22–31 | Session tokens from `/api/auth/login` are never validated by auth middleware — session infrastructure is dead code |
| M-49 | `middle/__tests__/registry_integration.test.ts` | 44–48 | Test asserts `plugins.length == 12` on a local constant — never queries the actual registry |
| M-50 | `middle/__tests__/chat_route.test.ts` | 63 | `JSON.parse(res.text)` instead of `await res.json()` — throws instead of assertion failure on format change |
| M-51 | `Makefile` | 119–127 | `rust-test` and `rust-lint` use `|| true` + `2>/dev/null` — all Rust test failures silently ignored, CI always exits 0 |
| M-52 | `Makefile` | 155–158 | `test` and `test-cov` have different `--ignore` lists — coverage report covers files the test run skips |

---

## TOP 10 PRIORITY FIXES

| Priority | Issue | File | Why |
|----------|-------|------|-----|
| 1 | Wire `DEFAULT_AUTH_MIDDLEWARE` into FastAPI | `gateway/aurelius_api.py` | Entire production API is publicly accessible |
| 2 | Fix top-p sampling (inverted sort + threshold) | `src/model/transformer.py:325-330` | All model generation output is wrong |
| 3 | Fix MoE residual connection | `src/model/transformer.py:111` | Residual learning disabled for MoE layers silently |
| 4 | Fix scheduler command injection | `middle/src/routes/scheduler.ts:83-99` | Authenticated RCE surface |
| 5 | Fix production compose empty auth defaults | `deployment/compose.production.yaml:47` | Production starts with no authentication |
| 6 | Pin `release.yml` Actions to SHA | `.github/workflows/release.yml:16,44` | Supply chain attack on PyPI package |
| 7 | Add securityContext to k8s deployment | `k8s/aurelius-deployment.yaml` | Containers run as root with all Linux capabilities |
| 8 | Fix DPO/GRPO model tuple unpack bug | `src/training/dpo_trainer.py:198-201` + `grpo_trainer.py:172` | Both crash on first forward pass |
| 9 | Fix `model_vocab_size` default | `src/training/trainer.py:226` | Wrong embedding shape on all default training runs |
| 10 | Replace DENY_LIST with real sandbox | `aurelius_cli/agent_engine.py:162-205` | All shell safety checks are trivially bypassed |

---

## STRUCTURAL OBSERVATIONS

**Dual-server problem (Gateway):** `api_server.py` (stdlib `HTTPServer`) and `aurelius_api.py` (FastAPI) serve overlapping functionality with wildly different security postures. `api_server.py` is comparatively hardened — it uses `DEFAULT_AUTH_MIDDLEWARE`, enforces fail-closed defaults, and rejects non-loopback binds without keys. `aurelius_api.py` has none of these controls and is the likely production entrypoint. All security work in `auth_middleware.py` is unused by the primary server.

**Good security code, wrong wiring:** `auth_middleware.py` uses `hmac.compare_digest`, SHA-256 hashing, and fail-closed defaults. The Helm chart has complete `securityContext`. The SSRF guard in `web_ui.py` is solid. These are well-implemented — they just are not connected to the code paths that matter.

**Two incompatible checkpoint systems:** `CheckpointManager` (inline in `trainer.py`) uses `step-{n:07d}` naming and `metadata.json`. The standalone `checkpoint.py` uses `checkpoint-{n:07d}` and `meta.json`. Resume logic using one system will never find checkpoints saved by the other.

**Dependency issues in `pyproject.toml`:** `torch>=2.11.0` does not exist (latest ~2.4.x) — `pip install` fails. `sageattention>=2.0.0` in base deps fails on CPU/Apple Silicon machines. `accelerate` is required by core training but only listed in optional extras.


---

## Appendix B — Implementation-loop gate-fix report

# Implementation-loop gate-fix report

Repo: /Users/christienantonio/aurelius
Branch state observed: main, ahead of origin/main by 4 commits
Working-tree files modified by the gate-fix loop:
- src/inference/sink_cache.py
- tests/inference/test_sink_cache.py
- src/eval/amc_memory_benchmark.py
- src/alignment/preference_optimization.py
- tests/runtime/test_memory_quarantine.py
- tests/safety/test_admission_controller.py

## Fixes completed

### F-01 · SinkCache absolute memory-token regression · FIXED
Root issue: SinkCache accepted memory_token_indices as absolute token positions, but after cache compression it reused those absolute positions as compressed tensor offsets. This could silently drop pinned memory tokens from the long-context/AMC cache.

Change made:
- SinkCache now tracks a token_indices sidecar in lockstep with the compressed K/V tensors.
- Absolute memory token IDs are remapped to current tensor positions before each compression pass.
- The returned selected_positions are applied to token_indices so tensor state and absolute-token provenance stay aligned.

Regression added:
- tests/inference/test_sink_cache.py::test_sink_cache_preserves_absolute_memory_tokens_after_compression

Risk/Effort/Payoff:
- Risk: High, because memory-preservation semantics could silently break.
- Effort: Low/Medium.
- Payoff: High, especially for AMC and long-context inference.

### F-02 · Ruff format gate · FIXED
Root issue: CI-format-equivalent ruff format --check failed on four files.

Change made:
- Ran ruff format on affected files.
- Format-only changes landed in alignment and safety/runtime tests.

Risk/Effort/Payoff:
- Risk: Medium; CI would reject otherwise valid work.
- Effort: Low.
- Payoff: Medium; deterministic style gate restored.

### F-03 · Bandit B608 false positive in AMC benchmark prompt · FIXED
Root issue: A natural-language benchmark prompt started with “Select…”, which Bandit classified as possible SQL construction.

Change made:
- Added an exact # nosec B608 annotation with an inline rationale in src/eval/amc_memory_benchmark.py.

Risk/Effort/Payoff:
- Risk: Medium; security gate failed on a false positive.
- Effort: Low.
- Payoff: Medium; CI security gate restored without weakening the baseline.

## Validation evidence from the implementation-loop pass

All of the following passed after the fixes:
- git diff --check
- .venv/bin/python -m ruff check src/ tests/ agent/ aurelius_cli/ gateway/ acp_adapter/ cron/ plugins/ tools/ scripts/ --ignore=I001 --output-format=concise
- .venv/bin/python -m ruff format --check src/ tests/ agent/ aurelius_cli/ gateway/ acp_adapter/ cron/ plugins/ tools/ scripts/
- .venv/bin/python -m compileall -q -j 8 src agent gateway aurelius_cli tools acp_adapter cron plugins tests
- .venv/bin/python -m pytest tests/inference/test_sink_cache.py -q --tb=short
- .venv/bin/python -m pytest tests/model/test_transformer.py tests/inference/test_sink_cache.py tests/longcontext/test_prefix_cache.py tests/eval/test_amc_memory_runner.py tests/eval/test_amc_memory_benchmark.py tests/runtime/test_memory_quarantine.py tests/safety/test_admission_controller.py tests/serving/test_api_server.py -q --tb=short
- .venv/bin/python -m pytest tests/eval/test_amc_memory_runner.py tests/eval/test_amc_memory_benchmark.py tests/eval/test_benchmark_config.py tests/memory/test_amc_tier2.py tests/test_package_consolidation.py tests/test_package_namespace.py tests/serving/test_web_ui.py tests/serving/test_engine_loader.py tests/agent/test_plugin_sandbox.py tests/inference/test_code_execution.py tests/multimodal/test_voice_engine.py -q --tb=short --maxfail=10
- .venv/bin/bandit -q -r src/ agent/ aurelius_cli/ gateway/ acp_adapter/ cron/ plugins/ tools/ scripts/ --baseline bandit-baseline.json --severity-level low

