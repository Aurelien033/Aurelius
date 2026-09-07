# Aurelius Code Review — Rescan Report
**Date:** 2026-05-18  
**Scope:** Full codebase rescan after 8 recent commits (~80+ changed files)  
**Baseline:** CODE_REVIEW.md (145 issues: 24 CRITICAL, 69 HIGH, 52 MEDIUM)

---

## Fix Rate Summary

| Domain | Prior Issues | Fixed | Partial | Still Open |
|--------|-------------|-------|---------|-----------|
| Gateway / Auth | 8 | 0 | 0 | 8 |
| ML / Training | 10 | 2 | 1 | 7 |
| Rust / FFI | 4 | 0 | 0 | 4 |
| CI / Infra | 8 | 0 | 1 | 7 |
| Tests / Hooks | 3 | 0 | 0 | 3 |
| Agent / CLI | 2 | 0 | 0 | 2 |
| **Total** | **35** | **2** | **2** | **31** |

**Overall fix rate: 6% (2 confirmed fixes, 2 partial)**

---

## Confirmed Fixes

### FIXED-01 — PPO Clipped Surrogate
**File:** `src/training/rlhf.py:249`  
**Was:** `torch.max(...)` used for clipped surrogate objective — wrong direction  
**Now:** Correct `torch.min(ratio * adv, torch.clamp(ratio, 1-eps, 1+eps) * adv)`  
Status: ✅ Fixed

### FIXED-02 — NumPy Shared Buffer in TokenizedDataLoader
**File:** `src/data/tokenized_loader.py:76-78`  
**Was:** In-place view assignment created aliased numpy buffers across samples  
**Now:** `.astype(np.int64)` forces a copy, breaking the aliasing  
Status: ✅ Fixed

---

## Partial Fixes

### PARTIAL-01 — Top-p Sampling Sort Direction
**File:** `src/model/transformer.py:325-330`  
**Was:** `descending=False` (ascending) — sampled from bottom of distribution  
**Now:** Sort direction corrected to `descending=True`  
**Remaining:** Threshold comparison at line 385-390 still inverted: `<= (1 - top_p)` should be `>= top_p`. Fix is incomplete; model still generates degenerate output under top-p.  
Status: ⚠️ Partial

### PARTIAL-02 — GitHub Actions Pinning
**File:** `.github/workflows/`  
**Was:** Multiple actions unpinned at `@v4` / `@v5`  
**Now:** Some workflows pinned, some still unpinned  
**Remaining:** `release.yml` lines 16, 21, 34, 45, 51 still use floating tags with PyPI publish scope.  
Status: ⚠️ Partial

---

## Still-Open Prior Issues (31)

### OPEN — Gateway (8 issues)

**GO-01** `gateway/aurelius_api.py` — FastAPI gateway has zero authentication. `auth_middleware.py` exists and is well-written but is not imported or applied. All endpoints remain publicly accessible.

**GO-02** `gateway/aurelius_server.py:1167` — `require_auth: False` default not changed.

**GO-03** `gateway/aurelius_server.py:273-275` — License activation handler runs before `_require_auth()` check.

**GO-04** `gateway/aurelius_server.py` — `hmac.compare_digest()` used correctly, but key loaded from `os.environ.get("API_KEY", "")` — empty string is a valid key. Server accepts all requests if the env var is unset.

**GO-05** `gateway/aurelius_api.py` — Binds `0.0.0.0:8080` by default with no network restriction. Combined with GO-01, the API is wide open on all interfaces.

**GO-06** `Dockerfile:58` — `ENV CORS_ORIGINS=*` baked into image. Overridable at runtime but the unsafe default will be used in any deployment that does not explicitly override it.

**GO-07** `deployment/compose.production.yaml:47` — `JWT_SECRET=` (empty default).

**GO-08** `deployment/compose.production.yaml:108` — `API_KEY=` (empty default).

### OPEN — ML / Training (7 issues)

**MO-01** `src/model/transformer.py:111` — **CRITICAL: MoE residual connection broken.**  
```python
# Current (wrong):
x, aux_loss = self.ffn(self.ffn_norm(x))   # x is REPLACED, not added

# Must be:
ffn_out, aux_loss = self.ffn(self.ffn_norm(x))
x = x + ffn_out
```
This means the MoE layer discards all prior representations. The entire expert routing system produces no net effect on the forward pass. This is a training-correctness CRITICAL.

**MO-02** `src/model/transformer.py:385-390` — Top-p threshold still inverted (see PARTIAL-01). Token probabilities are cumulative-summed correctly now, but the cutoff comparison remains wrong.

**MO-03** `src/training/trainer.py:226` — `model_vocab_size: int = 8_192`. The tokenizer uses 128k vocabulary. This mismatch causes an embedding dimension error at runtime unless the config is explicitly overridden. Should be `128_000`.

**MO-04** `src/training/dpo_trainer.py:137` — `policy_model(ids)` returns `(loss, logits, kv)` tuple; caller assigns result directly to a variable expected to be a logits tensor, then calls `.shape` on it — crashes at runtime.

**MO-05** `src/training/dpo_trainer.py:198-201` — Same tuple-vs-tensor crash at a second call site in the DPO loss computation.

**MO-06** `src/training/grpo_trainer.py:172` — `model(input_ids)` tuple crash — same pattern as MO-04/MO-05.

**MO-07** `src/training/grpo_trainer.py:310` — Second GRPO tuple crash site.

### OPEN — Rust / FFI (4 issues)

**RO-01** `src/rust/tokenizer/src/lib.rs` — Multiple `unwrap()` / `expect()` calls in PyO3 `#[pyfunction]` bodies. A panic in an FFI context is undefined behavior. Must use `PyResult<T>` with `map_err()` throughout.

**RO-02** `src/rust/tokenizer/src/lib.rs` — String slice from Python passed directly into Rust without lifetime validation. If the Python GC collects the object before Rust finishes using the slice, use-after-free.

**RO-03** `src/rust/tokenizer/src/lib.rs` — No `#[pyo3(text_signature = "...")]` annotations; breaks Python introspection and type stubs.

**RO-04** `src/rust/` — `Cargo.lock` not committed. For a binary/extension used in production, the lock file must be committed to ensure reproducible builds.

### OPEN — CI / Infra (7 issues)

**IO-01** `.github/workflows/release.yml:16,21,34,45,51` — Still unpinned (see PARTIAL-02). These steps have `id-token: write` and `packages: write` permissions, making supply chain compromise high-impact.

**IO-02** `k8s/aurelius-deployment.yaml` — No `securityContext`. Pod runs as root with all Linux capabilities. Helm templates (`k8s/helm/`) are correctly configured; the raw manifest is not.

**IO-03** `k8s/aurelius-deployment.yaml` — No resource `limits` block. Unbounded CPU/memory usage can cause node evictions.

**IO-04** `k8s/aurelius-deployment.yaml` — No `readinessProbe` or `livenessProbe`. K8s routes traffic to pods before the server is ready.

**IO-05** `deployment/compose.production.yaml` — Secrets passed as plain env vars rather than Docker secrets or a secrets manager integration.

**IO-06** `deployment/compose.production.yaml:47,108` — Empty secret defaults (see GO-07, GO-08).

**IO-07** Alembic migration files — No `downgrade()` implementations in recent migrations. Rollbacks are destructive (data loss) with no safety net.

### OPEN — Tests / Hooks (3 issues)

**TO-01** No integration tests for gateway authentication. The auth bypass (GO-01) has no test coverage, so automated CI will not catch a regression if auth is later partially implemented.

**TO-02** Pre-commit hooks in `.pre-commit-config.yaml` do not run the test suite. Broken commits can land on main.

**TO-03** No smoke test for inference path. The MoE residual bug (MO-01) and top-p bug (MO-02) would be caught by a single end-to-end generation test.

### OPEN — Agent / CLI (2 issues)

**AO-01** `agent/task_scheduler.py:12-26` — `subprocess` and `sys` are used in `_make_runner()` but never imported. First call raises `NameError` at runtime. (Detailed below as NC-01 since the module is new in this rescan.)

**AO-02** CLI argument parsing in several commands does not validate input length before passing to model inference, enabling prompt injection through the CLI.

---

## New Issues Introduced (57 total)

### New CRITICAL Issues (NC)

---

#### NC-01 — Missing Imports in Task Scheduler
**File:** `agent/task_scheduler.py:12-26`  
**Severity:** CRITICAL  
**Category:** Runtime crash

`_make_runner()` uses `subprocess.Popen` and `sys.executable` at lines 18 and 23 respectively. Neither `subprocess` nor `sys` is imported at the top of the file. The first call to schedule any task raises `NameError: name 'subprocess' is not defined`. This is a complete feature outage for the task scheduling system.

**Fix:** Add `import subprocess` and `import sys` to the module imports.

---

#### NC-02 — CLI Eval on User Input (RCE)
**File:** `aurelius_cli/pipeline_commands.py:69-83`  
**Severity:** CRITICAL  
**Category:** Remote code execution

The `filter` subcommand accepts a `--expr` argument and passes it directly to `eval()`:
```python
result = eval(expr, {"__builtins__": {}}, {"record": record})
```
The intent to sandbox with an empty builtins dict is incorrect. Python's `eval()` with `{"__builtins__": {}}` does NOT prevent access to builtins — a caller can traverse the object hierarchy through any object's `__class__` attribute chain to reach arbitrary subclasses including those with file system and subprocess access. Any user of the CLI can execute arbitrary code by crafting a filter expression that walks the class hierarchy to reach OS-level primitives. The fix is to replace `eval()` with a proper AST-based safe evaluator (e.g., `ast.literal_eval` for simple filters, or a whitelist visitor using `ast.NodeVisitor`).

---

#### NC-03 — Skill Library Executes Arbitrary Code (RCE)
**File:** `src/agent/skill_library.py:136, 142, 154, 187`  
**Severity:** CRITICAL  
**Category:** Remote code execution

Skills loaded from the skill library are executed via `exec(skill.code, {}, namespace)`. The same builtins sandbox bypass described in NC-02 applies here. An empty `{}` globals dict passed to `exec()` does not prevent access to Python builtins or OS primitives when skill code uses the object class hierarchy. Any skill stored in the library — including user-defined or network-fetched skills — can escape the sandbox and run arbitrary OS commands. Skills should be executed in a subprocess with no inherited environment, or skills should be validated against a strict AST whitelist before execution.

---

#### NC-04 — Plugin Sandbox Bypassed by functools.partial
**File:** `src/agent/plugin_sandbox.py:239-245`  
**Severity:** CRITICAL  
**Category:** Security sandbox bypass

`_check_callable_globals()` inspects `callable.__globals__` to detect dangerous imports. This check is trivially bypassed because:
1. `functools.partial` objects have no `__globals__` attribute — the check silently skips them.
2. Lambda functions that call `__import__()` directly bypass the globals inspection entirely because the import happens at call time, not at definition time.

A plugin can wrap a dangerous callable in `functools.partial` to pass the sandbox check while still executing prohibited operations. The sandbox provides false security. Fix requires either disallowing `functools.partial` and lambda in plugin code (AST check), or switching to a subprocess isolation model.

---

#### NC-05 — Bulkhead Race Condition Exceeds max_concurrent
**File:** `src/resilience/bulkhead.py:115-122`  
**Severity:** CRITICAL  
**Category:** Race condition / correctness

The `_release()` method releases the semaphore before signaling the queue waiter:
```python
async def _release(self):
    self._semaphore.release()        # ← semaphore released here
    if self._queue:
        waiter = self._queue.popleft()
        waiter.set_result(None)      # ← waiter unblocked after
```
Between the semaphore release and the waiter signal, a different coroutine can acquire the semaphore concurrently with the waiter. This allows `max_concurrent + 1` tasks to run simultaneously. The fix is to signal the waiter first, then release the semaphore (or to atomically hand the semaphore slot directly to the waiter without releasing it).

---

#### NC-06 — EventBus Subscriber List Overwritten During Publish
**File:** `src/observability/event_bus.py:72-96`  
**Severity:** CRITICAL  
**Category:** Concurrency / correctness

During `publish()`, the dead-reference pruning step rebuilds the subscriber list:
```python
alive = [s for s in subs if s() is not None]
self._subs[event_type] = alive
```
If a subscriber added its callback during the current `publish()` call (a re-entrant subscription), the rebuilt `alive` list — computed at the start of pruning — does not include the newly-added subscriber. The overwrite silently drops the new subscriber. Additionally, if `publish()` is called from multiple async tasks, both will overwrite `_subs[event_type]` with their own snapshot, losing each other's updates. Fix: use a lock around the subscriber list mutation, or collect new subscribers in a pending set and merge after publish completes.

---

#### NC-07 — Scheduler Jobs Never Execute
**File:** `aurelius_cli/scheduler_commands.py:132-157`  
**Severity:** CRITICAL  
**Category:** Logic / functionality

The `run` subcommand starts the scheduler and immediately shuts it down:
```python
sched.start()
# ... no blocking wait ...
finally:
    sched.shutdown()
```
`sched.start()` is non-blocking — it launches a background thread. Without a blocking call (e.g., `Event().wait()` or a signal handler loop) in the main thread, the `finally` block executes immediately and calls `sched.shutdown()`, terminating the scheduler before any jobs fire. The entire scheduling system is non-functional in CLI mode. Fix: add a blocking loop (e.g., `threading.Event().wait()`) between `start()` and the `finally` block, with a `SIGINT` / `SIGTERM` handler to break the wait cleanly.

---

#### NC-08 — ContextVar / threading.local Collision Corrupts Async Traces
**File:** `src/observability/trace_context.py:57-60`  
**Severity:** CRITICAL  
**Category:** Async correctness

Span propagation uses `ContextVar` for async context, but the fallback path writes the same span to `threading.local`:
```python
_span_local = threading.local()

def get_current_span():
    try:
        return _span_ctx_var.get()
    except LookupError:
        return getattr(_span_local, "span", None)
```
In an async application, multiple coroutines share the same OS thread. When coroutines interleave on a thread, `threading.local` holds the span from whichever coroutine last wrote it — every subsequent coroutine on that thread gets the wrong span until it sets its own. This produces corrupted traces where spans appear under the wrong parent. The `threading.local` fallback path must be removed entirely; `ContextVar` is the correct primitive for async-safe per-task context.

---

### New HIGH Issues (NH)

**NH-01** `src/resilience/circuit_breaker.py:88` — `half_open_calls` counter not reset when circuit transitions from HALF_OPEN back to OPEN after a failure. Subsequent HALF_OPEN probes use stale counts.

**NH-02** `src/resilience/circuit_breaker.py:134` — `datetime.now()` (naive, local time) used for cooldown windows. DST transitions can cause a 1-hour gap or immediate re-trip. Use `datetime.now(timezone.utc)`.

**NH-03** `src/resilience/retry.py:67` — `asyncio.sleep(delay)` called from a synchronous function. If the caller is a sync context, this will raise a `RuntimeError: no running event loop`. The retry decorator is used on both sync and async functions — it needs separate code paths.

**NH-04** `src/resilience/retry.py:89-93` — Jitter is `random.uniform(0, delay)`. Under high concurrency this produces correlated retry storms because all retriers start at the same time and add the same average jitter. Use full jitter: `random.uniform(0, cap)` where `cap` is the exponential backoff cap.

**NH-05** `src/resilience/timeout.py:45` — `asyncio.wait_for()` called without propagating `CancelledError` correctly. If the caller catches `asyncio.TimeoutError` and re-enters the coroutine, the cancelled internal task may still be running. Wrap with proper cancellation guard.

**NH-06** `src/caching/response_cache.py:112` — LRU eviction acquires the write lock while holding the read lock. Python's `threading.RLock` allows re-entrant locking from the same thread, but under async this can deadlock if two coroutines reach the lock boundary simultaneously.

**NH-07** `src/caching/semantic_cache.py:78` — Cosine similarity computed without normalizing embeddings first. For embeddings not guaranteed to be unit vectors, this produces incorrect similarity scores. Normalize before dot product or use `F.cosine_similarity`.

**NH-08** `src/caching/semantic_cache.py:204` — Cache keys include the full prompt text. For prompts longer than a few tokens this creates unbounded memory growth in the key index. Hash the prompt instead.

**NH-09** `src/agent/memory_manager.py:156` — `json.loads(row["content"])` without try/except. Corrupted or manually edited memory entries crash the entire memory retrieval path, not just the affected record.

**NH-10** `src/agent/memory_manager.py:201` — Memory search returns top-k by timestamp, not by relevance. The semantic search index is built but never consulted in the default retrieval path.

**NH-11** `src/agent/tool_registry.py:88-92` — Tool schemas stored in a plain dict with no thread safety. Concurrent tool registration from multiple async tasks can produce partially-written schema entries.

**NH-12** `src/agent/tool_registry.py:134` — `importlib.import_module(tool.module_path)` with user-controlled `tool.module_path` enables path traversal to load arbitrary Python modules from the filesystem.

**NH-13** `src/agent/agent_executor.py:67` — `max_iterations` defaults to 100 with no cost tracking. An agent can make 100 LLM calls before hitting the limit; at typical token counts this could exhaust budget silently.

**NH-14** `src/agent/agent_executor.py:189` — Agent loop appends messages without pruning context window. Once the context exceeds model `max_seq_len`, the model call fails with an out-of-bounds error that is not caught.

**NH-15** `src/agent/plugin_sandbox.py:78` — Allowed-module list includes `requests` and `httpx`. Plugins can make arbitrary outbound HTTP requests, enabling data exfiltration.

**NH-16** `src/observability/metrics_collector.py:45` — Prometheus counters created inside a request handler. `Counter("name", ...)` raises `ValueError` if a counter with the same name already exists (Prometheus registry is global). Under concurrent requests the second registration raises.

**NH-17** `src/observability/metrics_collector.py:112` — Histogram bucket boundaries `[0.01, 0.1, 0.5, 1.0]` (seconds) are too coarse for LLM inference latency which often spans 5-60 seconds. Misleading SLO dashboards.

**NH-18** `src/observability/structured_logger.py:67` — Log records include `user_input` field without sanitization. Newlines in user input create log injection vectors (fake log entries, SIEM alert bypass).

**NH-19** `src/observability/trace_context.py:89` — `TraceID` generated with `random.random()` (Mersenne Twister, not cryptographically random). Trace IDs are guessable. Use `secrets.token_hex(16)` for 128-bit cryptographic trace IDs.

**NH-20** `deployment/compose.production.yaml:23` — Model weights volume mounted read-write. The inference server only needs read access. A compromised container can overwrite model weights.

**NH-21** `deployment/compose.production.yaml:89` — `network_mode: host` for the monitoring container exposes all host ports, not just the metrics scrape port. Scope to a bridge network with explicit port mapping.

**NH-22** `src/data/preprocessing/text_cleaner.py:134` — Regex `re.compile(pattern)` inside a loop body. Pattern is recompiled on every iteration. For large datasets this is a significant performance regression (10-100x slower than pre-compiling once).

---

### New MEDIUM Issues (NM)

**NM-01** `src/resilience/bulkhead.py:45` — `max_concurrent` validated only at construction time. Dynamic reconfiguration at runtime can set it below the current active count, causing negative semaphore state.

**NM-02** `src/resilience/circuit_breaker.py:201` — State transitions not logged. Debugging circuit breaker behavior in production requires adding observability instrumentation retroactively.

**NM-03** `src/caching/response_cache.py:67` — Cache `max_size` defaults to 1000 entries with no size-based eviction. Each cached LLM response can be 4-64KB. At 1000 entries this is 4MB-64MB of unbounded cache memory.

**NM-04** `src/caching/semantic_cache.py:156` — Embedding model loaded synchronously at first cache lookup. The first request after startup has 2-10 second latency spike. Should warm the embedding model at startup.

**NM-05** `src/agent/memory_manager.py:89` — `sqlite3.connect(db_path)` with no `check_same_thread=False`. SQLite connections are not thread-safe by default. Async code accessing memory from multiple coroutines will raise `ProgrammingError`.

**NM-06** `src/agent/memory_manager.py:245` — No TTL on memory entries. The memory store grows without bound. Long-running agent sessions accumulate stale memories that degrade retrieval quality.

**NM-07** `src/agent/tool_registry.py:56` — Tool registration logs `tool.name` and `tool.description` at DEBUG level. If tool descriptions contain user-controlled content, this creates a log injection path.

**NM-08** `src/agent/agent_executor.py:123` — System prompt prepended on every iteration of the agent loop rather than once at the start. At iteration 50 the context has 50 copies of the system prompt, consuming ~10% of the context budget.

**NM-09** `src/agent/plugin_sandbox.py:167` — Plugin timeout enforced with `threading.Timer`. Under CPython's GIL, a compute-bound plugin may not yield the GIL to the timer callback for several seconds past the timeout.

**NM-10** `src/observability/event_bus.py:134` — No maximum subscriber count. A bug or test that registers handlers without deregistering will grow the subscriber list without bound.

**NM-11** `src/observability/metrics_collector.py:189` — `gc.collect()` called after every request to clean up metric objects. Manual GC invocation in a hot path causes latency spikes (stop-the-world pauses). Remove; Python's generational GC handles this.

**NM-12** `src/observability/structured_logger.py:45` — Logger initialized without a handler check. If the root logger has a handler configured upstream, log records are processed twice (double-logging in production).

**NM-13** `src/observability/trace_context.py:112` — Spans not finalized (`.end()` not called) in error paths that short-circuit via exception. Dangling open spans accumulate in the trace exporter buffer.

**NM-14** `aurelius_cli/scheduler_commands.py:89` — `argparse` help text for `--cron` accepts any string without validation. Invalid cron expressions cause an unhandled `ValueError` from the scheduler library rather than a clean error message.

**NM-15** `aurelius_cli/pipeline_commands.py:45` — `--batch-size` has no upper bound validation. Passing an extremely large value exhausts memory before the pipeline produces an error.

**NM-16** `agent/task_scheduler.py:67` — No retry logic for failed tasks. A transient error permanently marks the task as failed with no recovery path.

**NM-17** `agent/task_scheduler.py:89` — Task results written to a plain file without atomic rename. A crash mid-write produces a truncated result file that subsequent reads interpret as a complete but corrupted result.

**NM-18** `src/training/trainer.py:345` — Checkpoint saved every N steps but the old checkpoint is deleted before the new one is fully flushed to disk. A crash during the save window loses the last N steps of training with no recovery.

**NM-19** `src/training/trainer.py:412` — `torch.save(model.state_dict(), path)` saves without `_use_new_zipfile_serialization=False`. PyTorch's zipfile format is slower to load and can cause issues with very large checkpoints. Use `map_location` parameter on load.

**NM-20** `src/training/dpo_trainer.py:67` — No gradient clipping in DPO training loop. Combined with the tuple crash bugs (MO-04/MO-05), if those bugs are patched, unclipped gradients will cause training instability.

**NM-21** `src/training/grpo_trainer.py:45` — KL coefficient `beta` defaults to 0.0 — effectively disabling the KL penalty. With no KL constraint, GRPO reward hacking is unconstrained. Should default to 0.1.

**NM-22** `src/model/transformer.py:267` — KV-cache is pre-allocated using `max_seq_len` regardless of actual input length. For short inputs (typical chat turns) this wastes ~95% of allocated VRAM.

**NM-23** `src/model/transformer.py:401` — Temperature scaling applied before softmax. Standard practice is to divide logits by temperature before softmax, not after. Post-softmax division changes the probability distribution in a non-equivalent way.

**NM-24** `src/data/preprocessing/text_cleaner.py:89` — Unicode normalization uses `NFC` for some fields and `NFKC` for others. Inconsistent normalization means semantically identical strings may not match in downstream deduplication.

**NM-25** `src/data/preprocessing/text_cleaner.py:201` — Parallelized with `multiprocessing.Pool` but spawns a new pool for every file in a loop. Pool creation overhead dominates for small files.

**NM-26** `k8s/aurelius-deployment.yaml` — `imagePullPolicy: Always` not set. If the image tag is `latest`, the cluster may run a cached stale image after a new push.

**NM-27** `src/resilience/timeout.py:78` — Timeout value accepted as a float parameter with no validation. A timeout of 0 or negative causes `asyncio.wait_for()` to immediately cancel without executing the coroutine body.

---

## Priority Table — Top 20 Fixes

| Priority | ID | File | Impact |
|----------|----|------|--------|
| 1 | GO-01 | `gateway/aurelius_api.py` | API is fully unauthenticated |
| 2 | MO-01 | `src/model/transformer.py:111` | MoE layer produces no effect |
| 3 | NC-02 | `aurelius_cli/pipeline_commands.py:69` | RCE via CLI filter expression |
| 4 | NC-03 | `src/agent/skill_library.py:136` | RCE via skill execution |
| 5 | NC-04 | `src/agent/plugin_sandbox.py:239` | Sandbox bypass via functools.partial |
| 6 | MO-04+05 | `src/training/dpo_trainer.py:137,201` | DPO trainer crashes at runtime |
| 7 | MO-06+07 | `src/training/grpo_trainer.py:172,310` | GRPO trainer crashes at runtime |
| 8 | NC-05 | `src/resilience/bulkhead.py:115` | max_concurrent race condition |
| 9 | NC-06 | `src/observability/event_bus.py:72` | Subscriber loss during publish |
| 10 | NC-07 | `aurelius_cli/scheduler_commands.py:132` | Scheduler jobs never run |
| 11 | NC-08 | `src/observability/trace_context.py:57` | Async trace corruption |
| 12 | NC-01 | `agent/task_scheduler.py:12` | NameError crashes task scheduling |
| 13 | MO-02 | `src/model/transformer.py:385` | Top-p sampling still broken |
| 14 | MO-03 | `src/training/trainer.py:226` | Vocab size mismatch |
| 15 | IO-01 | `.github/workflows/release.yml` | Supply chain attack surface |
| 16 | IO-02 | `k8s/aurelius-deployment.yaml` | Pod runs as root |
| 17 | NH-12 | `src/agent/tool_registry.py:134` | Arbitrary module load via tool path |
| 18 | NH-16 | `src/observability/metrics_collector.py:45` | Prometheus counter collision crash |
| 19 | NH-18 | `src/observability/structured_logger.py:67` | Log injection via user input |
| 20 | RO-01 | `src/rust/tokenizer/src/lib.rs` | FFI panics = undefined behavior |

---

## Structural Observations

### Authentication Gap Persists
Despite multiple gateway commits, the FastAPI server (`aurelius_api.py`) still has no authentication. The auth middleware file exists and works correctly in isolation. The missing step is a single `app.add_middleware()` call and ensuring `API_KEY` / `JWT_SECRET` are set to non-empty values in all deployment configurations. Until this is done, the entire API surface is public.

### New RCE Surfaces Added in Recent Commits
Three independent code execution paths were added in recent commits: CLI eval (NC-02), skill execution (NC-03), and plugin sandbox (NC-04). All three attempt a Python sandbox using empty builtins dicts — a well-known insufficient mitigation. Any of these three paths can be exploited to run arbitrary OS commands. The correct fix is the same for all three: replace `eval()`/`exec()` with an AST-based whitelist validator that rejects any node that could resolve to OS access, or execute untrusted code in an isolated subprocess with no inherited environment.

### Training Pipeline Has Multiple Runtime Crashes Pending
The DPO and GRPO trainers both crash when they call the model and expect a logits tensor but receive a `(loss, logits, kv)` tuple. These bugs exist at 4 call sites and will prevent any DPO or GRPO training run from completing. The pattern is consistent: `output = model(ids)` followed by `output.shape` or `output[:, -1, :]`. The fix at each site is to unpack the tuple: `_, output, _ = model(ids)`.

### MoE Architecture Broken at the Core
The MoE residual connection bug (MO-01) means the Mixture of Experts routing, load balancing, and auxiliary loss are all computed but their outputs are thrown away. Every MoE layer replaces `x` with the raw expert output instead of adding it to the residual stream. This is a fundamental training correctness bug that affects every forward pass through the model. It must be fixed before any meaningful training results can be interpreted.

---

## Issue Count Summary

| Category | New Issues |
|----------|-----------|
| New CRITICAL | 8 |
| New HIGH | 22 |
| New MEDIUM | 27 |
| **Total New** | **57** |
| Prior issues fixed | 2 |
| Prior issues partially fixed | 2 |
| Prior issues still open | 31 |
| **Grand total open** | **88** |

---

*Generated by multi-agent rescan. Cross-reference with CODE_REVIEW.md for full baseline issue descriptions.*
