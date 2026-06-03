---
title: Aurelius Security Remediation Plan
date: 2026-06-01
author: Hermes Agent, active model note acknowledged as minimax-m3-free via OpenCode Zen
status: draft-for-review
source_audit: /Users/christienantonio/Desktop/AI:ML Research/aurelius-security-code-review-2026-06-01.md
repo: /Users/christienantonio/aurelius
branch_at_planning_time: feature/ring1-tranche5-20260531
commit_at_planning_time: 46ee2f13
boundary: plan-only; no source implementation in this turn
---

# Aurelius Security Remediation Plan

> For a future implementation agent: use `subagent-driven-development` only with strict path allowlists and immediate scope-audit after every delegated task. Do not implement from this document without first creating a clean remediation worktree and re-running the pre-flight checks below.

## 0. Goal

Bring Aurelius from the audit state described in `aurelius-security-code-review-2026-06-01.md` to a production-defensible security baseline:

1. No deprecated or legacy gateway can be accidentally built, run, exposed, or used as an auth bypass.
2. No user-supplied license or local runtime config can mint privileged credentials.
3. Browser, BFF, gateway, WebSocket, RAG, file, scheduler, and sandbox boundaries are explicitly authenticated, authorized, rate-limited, size-limited, logged safely, and fail-closed.
4. Privileged service credentials never substitute for user identity except on narrowly allowlisted internal calls.
5. Dependency, lint, audit, and regression gates run in CI and block reintroduction of the same classes of bugs.

This is intentionally risk-first, not architecture-beauty-first. Fix the trust boundaries before polishing style debt.

---

## 1. Current verified truth surface

Verified during planning, not assumed from memory:

- Audit artifact read: `/Users/christienantonio/Desktop/AI:ML Research/aurelius-security-code-review-2026-06-01.md`, 536 lines, 27,708 bytes.
- Repo path: `/Users/christienantonio/aurelius`.
- Current branch: `feature/ring1-tranche5-20260531`.
- Current HEAD: `46ee2f13`.
- Current working tree is dirty: many modified and untracked files exist across `.github/`, `agent/`, `gateway/`, `middle/`, `src/`, `tests/`, `docs/`, `server/`, `skills/`, and more.
- Root JS workspace scripts:
  - `npm run build -w frontend -w middle -w server`
  - `npm run test -w frontend -w middle`
  - `npm run lint -w frontend -w middle`
- `middle/package.json` scripts:
  - `npm run build` -> `tsc`
  - `npm run test` -> `vitest run`
  - `npm run lint` -> `eslint src/`
- `frontend/package.json` scripts:
  - `npm run build` -> `tsc -b && vite build`
  - `npm run test` -> `vitest run`
  - `npm run lint` -> `eslint .`
- `server/package.json` has no tests and still has build/start/dev scripts, making decommissioning/hard-disable important.
- Python project uses Python `>=3.12`, pytest, ruff, and security lint rules `E/F/I/UP/S` with global ignores for `S105`, `S110`, `S311`, `UP038`.
- Existing relevant TS tests:
  - `middle/__tests__/auth.test.ts`
  - `middle/__tests__/provider_router.test.ts`
  - `middle/__tests__/scheduler.test.ts`
  - `middle/__tests__/models.test.ts`
  - `middle/__tests__/persistence.test.ts`
  - `middle/__tests__/scopes.test.ts`
  - frontend page tests under `frontend/src/test/pages/`.
- Existing relevant Python security tests:
  - `tests/security/test_gateway_perimeter.py`
  - `tests/security/test_safe_archive.py`
  - `tests/security/test_checkpoint_signer.py`
  - `tests/security/run_broad_security_audit.py`
  - `tests/security/run_security_audit.py`
  - `tests/gateway/test_t3_gateway_fail_closed.py`

Implication: do not execute security remediation inside the current dirty working tree unless the user explicitly says these uncommitted changes are part of the remediation baseline. Use a clean worktree from `HEAD` or from the agreed canonical branch.

---

## 2. Non-negotiable remediation invariants

These are the invariants every task must preserve. Any task that violates one is not done even if tests pass.

### 2.1 Trust-boundary invariants

- Fail closed on missing credentials, malformed credentials, missing origin, missing tenant, missing scope, invalid upstream URL, or unknown room/channel.
- Authenticate before business logic and before emitting any WebSocket data.
- Authorize at the resource boundary, not just at route entry.
- User identity must remain user identity across the BFF -> gateway boundary. A service key cannot silently become the user's upstream principal.
- Long-lived secrets must never be stored in browser-readable storage.
- Untrusted text is data, never code, never a shell command, never raw HTML, never a log-line prefix.
- Runtime config can adjust non-security tuning only. Security-critical config changes require explicit allowlists, scoped admin/break-glass auth, audit diff, and restart or signed deployment artifact.

### 2.2 Implementation-process invariants

- No broad staging: never use `git add -A`, `git add .`, or broad globs in this repo while many unrelated untracked files exist.
- Every security behavior change starts with a failing regression test.
- Every implementation tranche has an independent spec review and independent security/code-quality review.
- Every dependency upgrade has lockfile diff review, build/test/audit proof, and rollback note.
- Every legacy-removal change proves both absence from production paths and absence from package/build scripts.
- Every accepted risk gets a dated waiver with owner, scope, expiry, compensating controls, and a test preventing accidental scope expansion.

---

## 3. Clean execution setup before any code change

### Task 3.1: Create a clean remediation worktree

Objective: isolate security remediation from the current dirty Ring1/research work.

Recommended command sequence:

```bash
cd /Users/christienantonio/aurelius
git status --short --branch
git rev-parse --short HEAD

# Create a clean worktree from the current audited HEAD.
git worktree add ../aurelius-security-remediation 46ee2f13 -b fix/security-remediation-p0
cd ../aurelius-security-remediation
git status --short --branch
```

Expected result:

- New worktree at `/Users/christienantonio/aurelius-security-remediation`.
- Branch `fix/security-remediation-p0`.
- Clean working tree.

If the user wants to base remediation on the current dirty branch content, stop and ask for explicit scope. Do not guess.

### Task 3.2: Capture baseline proof before modifying anything

Run and save baseline output:

```bash
cd /Users/christienantonio/aurelius-security-remediation
mkdir -p docs/remediation/security-2026-06-01/baseline

git status --short --branch > docs/remediation/security-2026-06-01/baseline/git-status.txt
git rev-parse HEAD > docs/remediation/security-2026-06-01/baseline/head.txt

python -m pytest tests/security/test_gateway_perimeter.py tests/gateway/test_t3_gateway_fail_closed.py -q \
  > docs/remediation/security-2026-06-01/baseline/python-security-tests.txt 2>&1 || true

npm run test -w middle \
  > docs/remediation/security-2026-06-01/baseline/middle-tests.txt 2>&1 || true

npm run test -w frontend \
  > docs/remediation/security-2026-06-01/baseline/frontend-tests.txt 2>&1 || true

cargo clippy --workspace --all-targets -- -D warnings \
  > docs/remediation/security-2026-06-01/baseline/cargo-clippy.txt 2>&1 || true

cargo audit --json \
  > docs/remediation/security-2026-06-01/baseline/cargo-audit.json 2>&1 || true

npm audit --omit=dev --json \
  > docs/remediation/security-2026-06-01/baseline/npm-audit-root.json 2>&1 || true
npm audit --omit=dev --json -w middle \
  > docs/remediation/security-2026-06-01/baseline/npm-audit-middle.json 2>&1 || true
npm audit --omit=dev --json -w server \
  > docs/remediation/security-2026-06-01/baseline/npm-audit-server.json 2>&1 || true

ruff check . \
  > docs/remediation/security-2026-06-01/baseline/ruff.txt 2>&1 || true
bandit -q -f json -o docs/remediation/security-2026-06-01/baseline/bandit.json -r gateway agent src tests \
  || true
pip-audit -f json -o docs/remediation/security-2026-06-01/baseline/pip-audit.json \
  || true
```

Do not treat these baseline commands as success criteria yet; they classify pre-existing failures. Later tranches must show no new failures and preferably reduce the baseline.

### Task 3.3: Establish remediation issue register

Create a tracked register file:

- `docs/remediation/security-2026-06-01/register.md`

Columns:

| ID | Audit finding | Severity | Owner area | Branch | Status | Tests added | Audit proof | Residual risk |
|---|---|---:|---|---|---|---|---|---|
| SEC-P0-01 | C1 legacy server | P0 | deploy/gateway | TBD | open | TBD | TBD | TBD |
| SEC-P0-02 | C2 license-derived keys | P0 | auth/license | TBD | open | TBD | TBD | TBD |
| SEC-P0-03 | C3 WebSocket auth/origin/channel | P0 | BFF/frontend/server | TBD | open | TBD | TBD | TBD |
| SEC-P0-04 | C4 BFF service-key forwarding | P0 | BFF/provider | TBD | open | TBD | TBD | TBD |
| SEC-P1-01 | H1 upstream SSRF/credential exfil | P1 | config/provider | TBD | open | TBD | TBD | TBD |
| SEC-P1-02 | H2 command route | P1 | command/provider/logs | TBD | open | TBD | TBD | TBD |
| SEC-P1-03 | H3 scheduler replay | P1 | scheduler | TBD | open | TBD | TBD | TBD |
| SEC-P1-04 | H4 RAG tenant isolation | P1 | RAG/memory | TBD | open | TBD | TBD | TBD |
| SEC-P1-05 | H5 file uploads | P1 | files/storage | TBD | open | TBD | TBD | TBD |
| SEC-P1-06 | H6 sandbox isolation | P1 | agent/sandbox | TBD | open | TBD | TBD | TBD |
| SEC-P1-07 | H7 auth parity | P1 | gateway/auth | TBD | open | TBD | TBD | TBD |
| SEC-P1-08 | H8 frontend credential storage | P1 | frontend/BFF | TBD | open | TBD | TBD | TBD |
| SEC-P1-09 | H9 dependencies | P1 | supply chain | TBD | open | TBD | TBD | TBD |
| SEC-P2-01..13 | M1-M13 | P2 | mixed | TBD | open | TBD | TBD | TBD |

The register is boring, but it prevents duplicate fixes and makes CI evidence reviewable.

---

## 4. Remediation architecture target

The target architecture should converge on one security control plane:

```text
Browser UI
  |  HTTPS only
  |  HttpOnly Secure SameSite session cookie OR short-lived bearer minted by BFF
  v
Middle BFF
  - canonical browser session/auth endpoint
  - validates CSRF for cookie-auth unsafe methods
  - per-user rate limits and audit IDs
  - never uses service key for user-triggered provider calls except allowlisted exchange
  - explicit upstream origin allowlist at startup
  - WS token minting and room authorization
  v
Python Gateway / Model API
  - one canonical FastAPI entrypoint
  - shared auth middleware only
  - hashed API keys / signed internal tokens / constant-time comparisons
  - no license-derived credential mutation
  - strict request-size and schema validation
  v
Agent / RAG / Sandbox / Rust Memory
  - tenant-scoped document and memory namespaces
  - streaming uploads to temp storage
  - real hostile-code sandbox when untrusted execution is enabled
  - unsafe Rust blocks documented, fuzzed, and Miri-checked where possible
```

Legacy `server/` must be either removed from production paths or explicitly marked as a non-production compatibility shell that cannot start without `AURELIUS_ENABLE_LEGACY_SERVER=true` and cannot mint/proxy/authenticate independently.

---

# PHASE P0 — block deployment until these are fixed

## P0.1 — Decommission or hard-disable the legacy Node server surface

Audit finding: C1. Also supports H7, M7, M13.

Primary files:

- `package.json`
- `server/package.json`
- `server/src/main.ts`
- `server/src/app.ts`
- `server/src/routes/index.ts`
- `server/src/routes/license.ts`
- `server/src/routes/proxy.ts`
- `server/src/ws/index.ts`
- deployment files: `Dockerfile*`, `docker-compose*`, `deployment/**`, `k8s/**`, `.github/workflows/**`
- docs mentioning server startup

Preferred decision: decommission from build/deploy. If the code must stay for historical reference, it should not be in the workspace build/test/deploy path.

### P0.1.1: Inventory every production path that can run `server/`

Read-only discovery command:

```bash
git grep -nE "server|aurelius-gateway|dev:server|build:server|start|node dist/main|tsx watch src/main" \
  -- package.json server deployment .github Dockerfile* docker-compose* k8s helm 2>/dev/null || true
```

Expected output: a complete list of all paths that reference legacy server.

### P0.1.2: Add decommissioning regression tests before changing build paths

Create tests that fail on the current state.

Python test path:

- Create: `tests/security/test_legacy_server_decommissioned.py`

Behaviors to test:

- Root `package.json` must not include `server` in default `build`, `test`, `lint`, or production scripts.
- No Dockerfile, compose, k8s, or deployment manifest may reference `server` as a production service unless the file name is explicitly dev-only and contains `AURELIUS_ENABLE_LEGACY_SERVER=true`.
- `server/src/routes/license.ts` and `server/src/routes/proxy.ts` must not be mounted by a production entrypoint.
- `server/package.json` must include an unmistakable deprecation/start guard if it remains.

Suggested assertions:

```python
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[2]

def test_root_package_does_not_build_legacy_server_by_default():
    pkg = json.loads((ROOT / "package.json").read_text())
    scripts = pkg.get("scripts", {})
    forbidden = ["build", "test", "lint"]
    for key in forbidden:
        assert "server" not in scripts.get(key, "")


def test_production_manifests_do_not_reference_legacy_server():
    candidates = []
    for pattern in ["Dockerfile*", "docker-compose*", "deployment/**/*", ".github/workflows/**/*", "k8s/**/*", "helm/**/*"]:
        candidates.extend(ROOT.glob(pattern))
    offenders = []
    for path in candidates:
        if path.is_file() and "dev" not in path.name.lower():
            text = path.read_text(errors="ignore")
            if re.search(r"\b(server|aurelius-gateway|dev:server|build:server)\b", text):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
```

### P0.1.3: Remove legacy server from default scripts/builds

Modify root `package.json`:

- Remove `dev:server`, `build:server`, and `server` from default `build` command.
- If `server/` must remain available for local comparison, rename scripts to explicit dev-only names, for example:
  - `dev:legacy-server:unsafe`
  - `build:legacy-server:unsafe`
- The default `build` should become `npm run build -w frontend -w middle` only.
- The default `test` should remain `frontend` and `middle`, plus any newly added tests.

Modify `server/package.json`:

- Make `dev/start/build` fail closed unless `AURELIUS_ENABLE_LEGACY_SERVER=true` is present, or rename all scripts to `dev:legacy`, `build:legacy`, `start:legacy`.
- Add a prestart guard script if retaining scripts.

Add/modify:

- Create: `server/DEPRECATED.md` or strengthen existing `server/DEPRECATED.md`.
- Create: `server/src/deprecation_guard.ts` if server remains buildable.

Acceptance criteria:

- `npm run build` does not compile/build `server`.
- `npm run dev:server` no longer exists, or exits with explicit decommissioning text.
- `tests/security/test_legacy_server_decommissioned.py` passes.

Validation:

```bash
python -m pytest tests/security/test_legacy_server_decommissioned.py -q
npm run build
npm run test -w middle
npm run test -w frontend
```

### P0.1.4: If legacy must remain, harden it to parity instead of pretending deprecation is enough

Only do this if the user chooses retention.

Required hardening tasks:

- `server/src/middleware/auth.ts`: fail-closed auth, hashed keys, constant-time comparison, scopes.
- `server/src/routes/license.ts`: delete credential-minting endpoint or make it return 410 Gone.
- `server/src/routes/proxy.ts`: delete or enforce the same URL policy as P1.1.
- `server/src/ws/index.ts`: require WS token and Origin allowlist before any message is accepted.
- `server/src/routes/index.ts`: ensure every mutating route has auth + scope middleware.
- Add tests under `server/__tests__/` if server continues to exist. If no test harness exists, either create one or keep server excluded from production.

Do not compromise here. A deprecated-but-runnable auth gateway is usually worse than no gateway because operators trust the new gateway while attackers find the old one.

---

## P0.2 — Replace license-derived API keys with signed licenses and random scoped credentials

Audit finding: C2.

Primary files:

- `server/src/routes/license.ts`
- `gateway/aurelius_server.py`
- `gateway/auth_middleware.py`
- `gateway/aurelius_api.py`
- `frontend/src/components/LicenseGate.tsx`
- `frontend/src/pages/Login.tsx`
- `middle/src/routes/auth.ts`
- tests: `tests/security/test_gateway_perimeter.py`, `tests/gateway/test_t3_gateway_fail_closed.py`, plus new tests below

Security target:

- License activation verifies an issuer signature. It never derives or sets an API key from license material.
- API keys are random, generated server-side, scoped, shown once, hashed at rest, revocable, and auditable.
- Legacy license-derived keys are rejected or migrated through a one-time explicit flow.

### P0.2.1: Write failing tests for credential self-issuance rejection

Create/extend:

- `tests/security/test_license_activation_security.py`
- `middle/__tests__/auth.test.ts` if auth route remains in BFF
- If `server/` retained: `server/__tests__/license.test.ts`

Behaviors:

1. A syntactically valid license string cannot become an API key.
2. License activation without a valid signature fails closed.
3. License activation cannot mutate `require_auth`, `api_key`, `metrics_key`, or other runtime auth config.
4. API key creation returns a random credential not derivable from the license payload.
5. Stored key material is hashed, not raw.

Python-oriented expected tests:

```python
def test_license_suffix_is_not_accepted_as_api_key(client):
    license_key = "AURELIUS-" + "A" * 64
    response = client.post("/license/activate", json={"license_key": license_key})
    assert response.status_code in {400, 401, 403, 410}
    assert "api_key" not in response.text.lower()


def test_license_activation_cannot_change_auth_runtime_config(client, admin_headers):
    before = client.get("/config", headers=admin_headers).json()
    response = client.post("/license/activate", json={"license_key": "AURELIUS-" + "B" * 64})
    assert response.status_code in {400, 401, 403, 410}
    after = client.get("/config", headers=admin_headers).json()
    assert after.get("require_auth") == before.get("require_auth")
```

### P0.2.2: Choose the license model

Recommended model:

- License token is a compact signed document:
  - `license_id`
  - `subject/account_id`
  - `plan/tier`
  - `issued_at`
  - `expires_at`
  - `features`
  - `deployment_fingerprint` if applicable
  - `nonce`
- Signature algorithm: Ed25519 preferred for implementation simplicity and deterministic signatures, or P-256 if ecosystem requires.
- Public verification key may be embedded/configured in the app.
- Private issuer key never ships with Aurelius.

License verification result:

```text
LicenseVerification {
  license_id: string
  subject: string
  features: string[]
  expires_at: datetime
  verified: true
}
```

API key generation result:

```text
ApiKeyCreated {
  key_id: string
  plaintext_key: shown_once_only
  key_hash: sha256/argon2id/hmac-sha256-with-server-secret
  scopes: string[]
  owner: subject/user
  created_at: datetime
  expires_at: optional
}
```

### P0.2.3: Remove or neutralize vulnerable flows

Modify:

- `server/src/routes/license.ts`: if server decommissioned, route is irrelevant; if retained, return 410 or validate signed license only.
- `gateway/aurelius_server.py`: remove `AURELIUS-*` suffix acceptance and any `runtime_config["api_key"] = key[-16:]` pattern.
- Any frontend/BFF flow that expects a license activation to return an API key must be changed to expect either a session or a prompt to create a scoped key.

Acceptance criteria:

- `git grep -n "aurelius-api-key\|key\[-16:\]\|license_key.*api" server gateway middle frontend` returns no vulnerable derivation path.
- Valid signed license verifies, but does not by itself create admin API credentials.
- Invalid license is rejected with generic error.
- No raw API key is persisted.

Validation:

```bash
python -m pytest tests/security/test_license_activation_security.py tests/security/test_gateway_perimeter.py -q
npm run test -w middle -- auth
```

---

## P0.3 — Preserve user identity through BFF upstream calls; eliminate default service-key forwarding

Audit finding: C4. Also intersects H1, H7, H9, observability.

Primary files:

- `middle/src/config.ts`
- `middle/src/provider_router.ts`
- `middle/src/server.ts`
- `middle/src/routes/models.ts`
- `middle/src/middleware/auth.ts`
- `middle/src/store/types.ts`
- `middle/__tests__/provider_router.test.ts`
- `middle/__tests__/models.test.ts`
- `middle/__tests__/scopes.test.ts`

Security target:

- User-triggered calls carry user identity and user scopes to upstream authorization logic.
- A service credential can exist only for a small allowlist of internal routes and cannot be used with arbitrary upstream URLs or arbitrary request bodies.
- Logs and metrics attribute requests to the user/session, not just the service.

### P0.3.1: Define BFF auth subject contract

Create/modify a type similar to:

```ts
export interface AuthSubject {
  subjectId: string;
  tenantId: string;
  scopes: string[];
  keyId?: string;
  sessionId?: string;
  authMethod: 'session' | 'api_key' | 'internal_service';
}
```

Location options:

- `middle/src/store/types.ts` if shared with persistence.
- Better: create `middle/src/auth/types.ts` if auth grows.

### P0.3.2: Write failing tests proving service key is not attached to user-provider calls

Extend:

- `middle/__tests__/provider_router.test.ts`
- `middle/__tests__/models.test.ts`
- `middle/__tests__/scopes.test.ts`

Test cases:

- A user request with `read:model` or `chat:complete` scope forwards a user-bound token or subject header, not `config.serviceApiKey`.
- A user with read-only scope cannot trigger an upstream admin operation through BFF.
- If `config.serviceApiKey` is configured and upstream URL is malicious/unallowlisted, no credential is attached.
- Audit log includes `subjectId` and `tenantId`.

Pseudo-test shape:

```ts
it('does not use service key as user principal for provider completion', async () => {
  const calls: Array<{ headers: Record<string, string> }> = [];
  const fetchStub = async (_url: string, init: any) => {
    calls.push({ headers: init.headers });
    return new Response(JSON.stringify({ ok: true }), { status: 200 });
  };

  await providerRouter.complete({ userPrompt: 'hi' }, {
    subject: { subjectId: 'u1', tenantId: 't1', scopes: ['chat:complete'], authMethod: 'api_key' },
    fetch: fetchStub,
  });

  expect(calls[0].headers['X-Api-Key']).not.toBe(process.env.AURELIUS_SERVICE_KEY);
  expect(calls[0].headers['X-Aurelius-Subject']).toBe('u1');
});
```

### P0.3.3: Implement explicit upstream auth modes

Design modes:

1. `user_forwarded`: forward user key/session-derived upstream token.
2. `token_exchange`: BFF exchanges browser session for a short-lived upstream token bound to subject/scopes/audience.
3. `internal_service`: service key used only for explicitly allowlisted administrative internal endpoints.

Configuration should fail closed:

```ts
type UpstreamAuthMode = 'user_forwarded' | 'token_exchange' | 'internal_service';

interface UpstreamRoutePolicy {
  routeName: string;
  allowedAuthModes: UpstreamAuthMode[];
  requiredScopes: string[];
  allowedMethods: string[];
  bodyAllowlistSchema: unknown;
}
```

Do not let arbitrary routes choose auth mode dynamically from request input.

### P0.3.4: Replace direct `config.serviceApiKey` attachment

Modify `middle/src/provider_router.ts`, `middle/src/server.ts`, and `middle/src/routes/models.ts` so calls go through one helper:

```ts
buildUpstreamAuthHeaders(policy, authSubject, upstreamUrl)
```

The helper must:

- Verify upstream URL passed allowlist validation from P1.1.
- Check subject scopes.
- Attach user token or exchanged token by default.
- Attach service key only if policy allows `internal_service` and route is not user-driven.
- Never attach any secret to an unvalidated URL.

Acceptance criteria:

- `git grep -n "serviceApiKey\|AURELIUS_SERVICE_KEY" middle/src` shows usage only in config loading and the central auth-header helper.
- User-triggered route tests prove service key is absent.
- Admin/internal tests prove service key, if used, is route-allowlisted and audited.

Validation:

```bash
npm run test -w middle -- provider_router models scopes auth
npm run build -w middle
```

---

## P0.4 — Fix WebSocket authentication, Origin validation, and room authorization

Audit finding: C3. Also supports H8, H7, M4.

Primary files:

- `middle/src/ws/handler.ts`
- `middle/src/ws/rooms.ts`
- `middle/src/config.ts`
- `middle/src/middleware/auth.ts`
- `middle/src/routes/auth.ts`
- `frontend/src/hooks/useWebSocket.ts`
- `frontend/src/hooks/useAuth.ts`
- `frontend/src/stores/apiStore.ts`
- `server/src/ws/index.ts` if server retained
- `server/src/ws/hub.ts` if server retained
- Tests: create `middle/__tests__/websocket_auth.test.ts`; update frontend tests as needed

Security target:

- Server rejects cross-origin WebSocket attempts before any business event.
- WS auth uses a short-lived token, cookie-backed session, or `Sec-WebSocket-Protocol`, not long-lived localStorage API key.
- Subscribe messages use a single schema (`room`, not drift between `channel` and `room`).
- Room policy maps rooms to scopes and tenant ownership.

### P0.4.1: Define the WS protocol contract

Canonical client -> server messages:

```json
{ "type": "subscribe", "room": "tenant:t1:notifications" }
{ "type": "unsubscribe", "room": "tenant:t1:notifications" }
{ "type": "ping" }
```

Canonical server -> client messages:

```json
{ "type": "connected", "connectionId": "...", "subjectId": "...", "tenantId": "..." }
{ "type": "subscribed", "room": "..." }
{ "type": "error", "code": "ROOM_FORBIDDEN", "message": "Forbidden" }
```

Room classes:

- `tenant:{tenantId}:notifications` requires same tenant and `notifications:read`.
- `tenant:{tenantId}:agents:{agentId}` requires same tenant and `agents:read` plus resource authorization.
- `system:health` requires authenticated subject and `system:read`.
- `admin:*` requires `admin` or explicit admin scope.

### P0.4.2: Write failing WS tests

Create `middle/__tests__/websocket_auth.test.ts`.

Test cases:

- Rejects connection with missing token/session.
- Rejects connection with bad Origin.
- Accepts connection with allowed Origin and valid short-lived token/session.
- Sends no `connected` payload before auth passes.
- Rejects subscribe to room with mismatched tenant.
- Rejects subscribe to admin room without admin scope.
- Accepts allowed tenant room.
- Rejects old `{ channel: ... }` schema or maps it deliberately with deprecation warning; do not silently drift.

Implementation note for tests:

- Use `ws` package test client if available.
- If hard to run full server, unit-test `validateWsHandshake()` and `authorizeRoom()` first, then add one integration test.

### P0.4.3: Implement handshake validation helpers

Recommended functions:

- `getAllowedOrigins(config): Set<string>`
- `validateWsOrigin(origin: string | undefined, allowed: Set<string>): boolean`
- `extractWsToken(req): string | null`
- `verifyWsToken(token): AuthSubject`
- `authorizeRoom(subject: AuthSubject, room: string): boolean`
- `parseWsMessage(raw): WsMessage | ValidationError`

Design decisions:

- Prefer cookie-based session auth if P1.8 lands first.
- Otherwise add an endpoint like `POST /api/auth/ws-token` that returns a short-lived token with audience `ws`, expiration <= 60 seconds, and subject/scopes/tenant claims.
- Do not put long-lived API keys in query parameters. Query tokens are logged too often; if query is used, token must be short-lived and audience-bound.

### P0.4.4: Align frontend hook with protocol

Modify `frontend/src/hooks/useWebSocket.ts`:

- Use `wss://` when page uses `https:`.
- Get token from BFF `POST /api/auth/ws-token` or rely on cookie/session.
- Set `Sec-WebSocket-Protocol` if token transport chosen.
- Send `{ type: 'subscribe', room }`, not `{ channel }`.
- Enforce finite reconnect backoff and stop after auth failures.

Test updates:

- `frontend/src/test/pages/Notifications.test.tsx` if WS is exercised.
- Create `frontend/src/test/hooks/useWebSocket.test.tsx` if hook tests are supported.

Acceptance criteria:

- Cross-origin WS is rejected.
- Missing/bad token is rejected.
- Tenant-mismatched room is rejected.
- Frontend and middle agree on schema.
- No raw long-lived key is sent in WS URL.

Validation:

```bash
npm run test -w middle -- websocket
npm run test -w frontend -- useWebSocket Notifications
npm run build -w middle
npm run build -w frontend
```

---

# PHASE P1 — high-risk hardening this week

## P1.1 — Lock down upstream URL configuration and SSRF/credential exfiltration risk

Audit finding: H1, M7. Supports P0.3.

Primary files:

- `middle/src/config.ts`
- `middle/src/engine.ts`
- `middle/src/provider_router.ts`
- `middle/src/server.ts`
- `middle/src/routes/models.ts`
- `server/src/routes/proxy.ts` if retained
- `server/src/config.ts` if retained
- Tests: `middle/__tests__/provider_router.test.ts`, create `middle/__tests__/url_policy.test.ts`

Security target:

- Upstream origins are validated once at startup and represented as typed, already-approved origins.
- Non-dev production disallows private/link-local/metadata IP ranges, non-HTTPS, credentials in URL, and runtime mutations.
- Service credentials are attached only after URL policy passes.

### P1.1.1: Write URL policy tests

Create `middle/__tests__/url_policy.test.ts`.

Cases:

- Accept `https://api.allowed.example` when configured in allowlist.
- Reject `http://api.allowed.example` in production.
- Allow `http://127.0.0.1:<port>` only in explicit dev/test mode.
- Reject `http://169.254.169.254/latest/meta-data`.
- Reject `http://localhost.evil.com` if only localhost is allowed.
- Reject userinfo URLs like `https://user:pass@example.com`.
- Reject runtime config updates that change upstream URL without break-glass scope.
- Ensure auth headers are not attached when policy rejects URL.

### P1.1.2: Implement URL policy module

Create:

- `middle/src/security/url_policy.ts`

Functions:

- `parseAndValidateUpstreamUrl(raw: string, env: 'development' | 'test' | 'production', allowlist: string[]): ValidatedUpstreamUrl`
- `isPrivateOrMetadataAddress(hostOrIp: string): Promise<boolean>` or synchronous for literal IP + async DNS in startup path.
- `assertCanAttachCredential(validatedUrl: ValidatedUpstreamUrl): void`

Important rules:

- No wildcard allowlist in production.
- Validate both hostname and resolved IPs where feasible.
- Re-resolve periodically only if you can detect DNS rebinding; otherwise pin at startup.
- Never validate a string and then use a different string to call fetch.

### P1.1.3: Remove runtime mutable upstream URLs

Modify:

- `middle/src/engine.ts`: runtime config cannot update upstream URL except via break-glass path.
- `middle/src/routes/config.ts`: security-sensitive keys are denylisted or allowed only with explicit `config:security:write` scope and audit event.
- `gateway/aurelius_server.py` runtime config route: same denylist for Python legacy.

Acceptance criteria:

- `git grep -n "pythonUrl\|upstreamUrl\|provider.*url" middle/src server/src gateway` shows all uses flow through validation.
- Attempting to set provider/upstream URL at runtime without break-glass fails.
- Credentials are never attached to invalid URL.

Validation:

```bash
npm run test -w middle -- url_policy provider_router config
python -m pytest tests/security/test_gateway_perimeter.py -q
```

---

## P1.2 — Replace frontend localStorage API keys with secure session design

Audit finding: H8, M4.

Primary files:

- `frontend/src/stores/apiStore.ts`
- `frontend/src/stores/persist.ts`
- `frontend/src/services/api.ts`
- `frontend/src/hooks/useAuth.ts`
- `frontend/src/hooks/useWebSocket.ts`
- `frontend/src/pages/Login.tsx`
- `middle/src/routes/auth.ts`
- `middle/src/middleware/auth.ts`
- Tests: `frontend/src/test/pages/Login.test.tsx`, `frontend/src/test/pages/Settings.test.tsx`, `middle/__tests__/auth.test.ts`

Security target:

- Browser stores no long-lived API keys in `localStorage`.
- Preferred mode: BFF issues `HttpOnly; Secure; SameSite=Lax/Strict` session cookies.
- Unsafe methods use CSRF token or SameSite+double-submit depending on deployment.
- API clients use `credentials: 'include'` and never read the secret value.

### P1.2.1: Decide BYOK vs server-side account/session model

Two acceptable modes:

1. Production session mode:
   - User authenticates once through BFF.
   - BFF stores or verifies credential server-side.
   - Browser receives only session cookie and CSRF token.
2. Local developer BYOK mode:
   - If user manually enters API key, keep it in memory or `sessionStorage` with TTL and explicit warning.
   - Never put BYOK in localStorage by default.

Do not mix modes silently. Add a visible config flag:

- `AURELIUS_AUTH_MODE=session|local_byok_dev`

### P1.2.2: Write failing credential-storage tests

Frontend tests:

- Login should not call `localStorage.setItem('apiKey', ...)`.
- Generic persist layer must deny persistence of fields named `apiKey`, `token`, `secret`, `password` unless explicitly waived.
- On logout, session state clears and no credential is left in localStorage/sessionStorage.

Middle tests:

- Login sets cookie with `HttpOnly`, `Secure` in production, `SameSite=Lax` or stricter.
- Unsafe method without CSRF fails if cookie-authenticated.
- Auth middleware accepts session cookie and sets `AuthSubject`.

### P1.2.3: Implement cookie/session auth in BFF

Modify/create:

- `middle/src/routes/auth.ts`: login/session/logout/ws-token endpoints.
- `middle/src/middleware/auth.ts`: parse session cookie, verify session, set subject.
- `middle/src/store/types.ts`: session model.
- `middle/src/store/sqlite-store.ts` or memory store: session persistence if appropriate.

Cookie rules:

- `HttpOnly=true`
- `Secure=true` in production
- `SameSite=Lax` minimum, `Strict` if flows allow
- Short idle timeout and absolute timeout
- Rotate session ID on login and privilege change

### P1.2.4: Update frontend API client

Modify:

- `frontend/src/services/api.ts`: use `credentials: 'include'`; remove key header assembly from browser state.
- `frontend/src/stores/apiStore.ts`: store auth status and display metadata, not secret.
- `frontend/src/stores/persist.ts`: denylist secret-like keys.

Acceptance criteria:

- `git grep -n "localStorage.*api\|apiKey.*localStorage\|setItem(.*token\|setItem(.*secret" frontend/src` has no production credential persistence.
- E2E-ish frontend tests prove login/logout behavior without localStorage key.
- BFF auth tests prove cookie flags.

Validation:

```bash
npm run test -w frontend -- Login Settings
npm run test -w middle -- auth scopes
npm run build -w frontend
npm run build -w middle
```

---

## P1.3 — Add tenant/owner isolation and quotas to RAG documents and memory insertion

Audit finding: H4, M6, M9.

Primary files:

- `middle/src/routes/rag.ts`
- `middle/src/store/types.ts`
- `middle/src/store/memory-store.ts`
- `middle/src/store/sqlite-store.ts`
- `middle/src/engine.ts` if RAG memory layer is there
- `middle/__tests__/rag.test.ts` new
- `middle/__tests__/persistence.test.ts` maybe update

Security target:

- Every document has `tenantId`, `ownerSubjectId`, `createdBy`, `visibility`, `sizeBytes`, and `sourceMetadata`.
- Every list/read/search/delete filters by tenant and authorization.
- Shared memory insertion uses tenant-scoped namespaces.
- Quotas bound content length, chunk count, document count, total bytes, and search fanout.

### P1.3.1: Write failing RAG isolation tests

Create `middle/__tests__/rag.test.ts`.

Cases:

- User A cannot list/search/read/delete User B's document.
- Tenant A cannot retrieve Tenant B chunks via search.
- Admin can access cross-tenant only with explicit admin scope and audit event.
- Oversized document is rejected before chunking.
- IDs are `crypto.randomUUID()` format or equivalent, not `Date.now()+Math.random`.
- Deleting document removes or tombstones corresponding memory-layer chunks.

### P1.3.2: Define document model

Suggested TS type:

```ts
interface RagDocument {
  id: string;
  tenantId: string;
  ownerSubjectId: string;
  visibility: 'private' | 'tenant' | 'admin';
  title: string;
  sanitizedSourceName?: string;
  contentHash: string;
  sizeBytes: number;
  chunkCount: number;
  createdAt: string;
  updatedAt: string;
}
```

Chunk namespace:

```text
rag:{tenantId}:{visibility}:{documentId}:{chunkIndex}
```

### P1.3.3: Replace global process array with scoped store

Avoid process-global arrays for multi-user state. Use existing store abstraction if feasible:

- `middle/src/store/types.ts`: add RAG methods.
- `middle/src/store/memory-store.ts`: test/dev in-memory scoped store.
- `middle/src/store/sqlite-store.ts`: durable store if already in use.

Acceptance criteria:

- No route operates on all docs without tenant filter.
- Search receives subject/tenant and restricts namespace.
- Quotas enforced before memory insertion.

Validation:

```bash
npm run test -w middle -- rag persistence scopes
npm run build -w middle
```

---

## P1.4 — Stream file uploads, sniff content server-side, sanitize filenames once

Audit finding: H5, M9, M10.

Primary files:

- `middle/src/routes/files.ts`
- `middle/src/store/types.ts` if file metadata persisted
- `middle/__tests__/files.test.ts` new
- `middle/src/middleware/rate-limiter.ts`

Security target:

- No 50MB per-request in-memory buffer.
- Uploads stream to temp files with hard byte counter and atomic move.
- Server-side content sniffing for allowlisted types.
- Sanitized filename stored and used everywhere; original name never used as log prefix or path.
- Per-user concurrent upload and storage quotas.

### P1.4.1: Write failing upload tests

Create `middle/__tests__/files.test.ts`.

Cases:

- Upload over limit aborts early and removes temp file.
- MIME type header mismatch is rejected.
- Filename with CR/LF/path separators/control characters is sanitized once.
- Concurrent upload limit returns 429 or 503.
- Activity logs contain structured sanitized filename field, not concatenated raw name.

### P1.4.2: Implement streaming upload pipeline

Recommended pipeline:

```text
request stream
  -> byte-counting transform
  -> temp file under non-public temp dir
  -> content sniff first N bytes
  -> hash while streaming
  -> atomic move to content-addressed storage
  -> persist metadata
```

Rules:

- Use `crypto.randomUUID()` for temp names.
- Never derive filesystem path directly from client filename.
- Use `fs.open` with exclusive flags where possible.
- Delete temp file on any error.
- Put upload root outside web-static directory.

Acceptance criteria:

- Memory usage no longer scales with full file size per request.
- Oversized requests terminate early.
- Raw filename cannot inject logs or paths.

Validation:

```bash
npm run test -w middle -- files
npm run build -w middle
```

---

## P1.5 — Convert admin command and scheduler from raw text execution to typed, auditable command workflows

Audit findings: H2, H3, M10.

Primary files:

- `middle/src/server.ts`
- `middle/src/routes/scheduler.ts`
- `middle/src/store/types.ts`
- `middle/src/store/sqlite-store.ts`
- `middle/src/store/memory-store.ts`
- `frontend/src/pages/ScheduledTasks.tsx`
- `frontend/src/components/CommandPalette.tsx`
- Tests: `middle/__tests__/scheduler.test.ts`, create/extend `middle/__tests__/command.test.ts`

Security target:

- `/api/command` accepts a typed command object, not arbitrary privileged text.
- Allowed command types are explicit and scoped.
- External-provider execution requires explicit mode/disclosure.
- Scheduler can only replay approved typed commands, with owner, scope, max runs, expiry, and emergency disable.
- Logs are structured and redacted.

### P1.5.1: Define command schema

Example command envelope:

```ts
interface CommandRequest {
  type: 'agent.start' | 'agent.stop' | 'model.reload' | 'rag.reindex' | 'system.health_check';
  targetId?: string;
  params: Record<string, unknown>;
  reason: string;
  dryRun?: boolean;
}
```

Policy table:

```ts
const COMMAND_POLICIES = {
  'agent.start': { scopes: ['agents:write'], externalProviderAllowed: false },
  'model.reload': { scopes: ['admin:model'], externalProviderAllowed: false },
  'rag.reindex': { scopes: ['rag:write'], externalProviderAllowed: false },
  'system.health_check': { scopes: ['system:read'], externalProviderAllowed: false },
};
```

### P1.5.2: Write failing tests

Cases:

- Raw string command rejected.
- Unknown command type rejected.
- Command requiring admin scope rejected for non-admin.
- Scheduled command cannot be created without approval metadata.
- Scheduler refuses disabled-all switch.
- Scheduler refuses expired/max-run-exceeded tasks.
- Logs redact params marked sensitive.

### P1.5.3: Separate chat/LLM from privileged tool invocation

If the product needs an admin natural-language command UI, split it into two phases:

1. LLM translates text into proposed `CommandRequest` with no execution.
2. Human/admin reviews structured command and approves.
3. Executor runs only the schema-valid, policy-allowed command.

Never let a provider response directly decide privileged execution.

Acceptance criteria:

- `/api/command` cannot be used as a raw LLM prompt-to-privileged-action bridge.
- Scheduler persists structured commands with owner/approval/max-run/expiry.
- Emergency disable stops all scheduled execution.

Validation:

```bash
npm run test -w middle -- scheduler command
npm run build -w middle
```

---

## P1.6 — Replace subprocess-only sandbox with explicit trusted vs hostile execution modes

Audit finding: H6.

Primary files:

- `agent/code_execution_tool.py`
- `agent/code_execution_sandbox.py`
- `tests/tools/test_code_runner_exploits.py`
- possibly `src/security/sandbox_executor.py`
- deployment files for sandbox runner if added

Security target:

- Hostile/untrusted code is disabled by default unless a real isolation backend is configured.
- Local subprocess sandbox is labeled trusted-dev only.
- Production hostile-code execution uses gVisor, Firecracker, nsjail, bubblewrap, Docker-with-hardening, or Wasm, with no network by default and disposable filesystem.

### P1.6.1: Clarify threat model in code and config

Add config:

```text
AURELIUS_CODE_EXECUTION_MODE=disabled|trusted_subprocess|isolated
AURELIUS_SANDBOX_BACKEND=none|gvisor|firecracker|nsjail|bubblewrap|wasm
AURELIUS_SANDBOX_NETWORK=none|allowlisted
```

Production default must be `disabled` unless isolated backend is configured.

### P1.6.2: Write failing tests

Extend `tests/tools/test_code_runner_exploits.py`.

Cases:

- Default config refuses untrusted code execution.
- `trusted_subprocess` requires explicit local/dev flag.
- Isolated mode fails closed if backend binary/socket is missing.
- Network access is denied by default.
- Filesystem outside run directory is not readable.
- Timeout kills whole isolated unit, not just waiting future.

### P1.6.3: Add `SandboxBackend` abstraction

Suggested Python interface:

```python
class SandboxBackend(Protocol):
    def run(self, code: str, limits: SandboxLimits, context: SandboxContext) -> SandboxResult: ...
```

Implementations:

- `DisabledSandboxBackend`
- `TrustedSubprocessSandboxBackend`
- `IsolatedSandboxBackend` wrapper around selected runtime

The first production implementation can be minimal, but the contract must make isolation explicit.

Acceptance criteria:

- Running untrusted code in production config without isolation returns clear refusal.
- Existing local developer tests can opt into trusted subprocess.
- Exploit tests demonstrate denied network/filesystem access for isolated backend, or are skipped with clear reason if backend unavailable while production remains disabled.

Validation:

```bash
python -m pytest tests/tools/test_code_runner_exploits.py -q
python -m pytest tests/security/test_gateway_perimeter.py -q
```

---

## P1.7 — Normalize gateway auth behavior across Python gateway, BFF, and retained server paths

Audit finding: H7, M11, M12.

Primary files:

- `gateway/auth_middleware.py`
- `gateway/aurelius_api.py`
- `gateway/aurelius_server.py`
- `middle/src/middleware/auth.ts`
- retained `server/src/middleware/auth.ts`
- `tests/security/test_gateway_perimeter.py`
- `tests/gateway/test_t3_gateway_fail_closed.py`

Security target:

- One canonical Python gateway entrypoint.
- Shared auth semantics: no key -> fail closed, wrong key -> 401/403, correct key -> scoped access, metrics key constant-time, config mutations scoped.
- Legacy `aurelius_server.py` cannot weaken auth or mutate auth config.

### P1.7.1: Add cross-entrypoint auth contract tests

Extend `tests/security/test_gateway_perimeter.py`.

Cases for every executable entrypoint:

- No API key in env/config -> protected route rejects in production mode.
- Wrong API key rejects.
- Correct key accepts only allowed scope.
- Metrics key uses constant-time helper path.
- Config endpoint cannot mutate `require_auth`, `api_key`, `metrics_key`, CORS origins, upstream URLs, or sandbox mode without break-glass scope.

### P1.7.2: Create central secret comparison helper

Use `hmac.compare_digest` everywhere.

Python:

```python
def constant_time_equal(provided: str | None, expected: str | None) -> bool:
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided.encode(), expected.encode())
```

TS equivalent for server if retained:

```ts
crypto.timingSafeEqual(Buffer.from(a), Buffer.from(b))
```

Handle unequal lengths safely by comparing fixed hashes.

### P1.7.3: Lock runtime config mutability

Mutable config allowlist should contain only safe operational values, for example:

- UI theme
- non-security model sampling defaults with bounds
- feature flags not affecting auth/security

Denylist:

- `require_auth`
- `api_key`
- `serviceApiKey`
- `metrics_key`
- `cors_origin(s)`
- `upstream_url`
- `sandbox_mode`
- `allowed_origins`
- `rate_limit_disabled`

Acceptance criteria:

- Cross-entrypoint tests pass.
- `git grep -n "!=.*key\|==.*key" gateway middle server` finds no raw secret equality in production paths.
- Runtime config cannot alter auth perimeter.

Validation:

```bash
python -m pytest tests/security/test_gateway_perimeter.py tests/gateway/test_t3_gateway_fail_closed.py -q
npm run test -w middle -- auth scopes
```

---

## P1.8 — Upgrade vulnerable dependencies and make audits complete

Audit finding: H9, M13.

Primary files:

- `pyproject.toml`
- lockfiles if present: `uv.lock`, `requirements*.txt`, etc.
- `package-lock.json`
- `middle/package-lock.json`
- `server/package-lock.json`
- create/commit `frontend/package-lock.json` or migrate workspace lock strategy
- `.github/workflows/ci.yml`

Security target:

- npm `qs` and `ws` moderate vulns fixed.
- pip-audit vulnerabilities fixed or explicitly waived with no-fixed-version rationale.
- Frontend has lockfile coverage or is fully covered by root workspace lock.
- CI runs audits for every package root.

### P1.8.1: Prepare dependency plan per ecosystem

npm:

- Ensure `ws >= 8.20.1` wherever used.
- Update `qs` through direct or transitive dependency resolution.
- Run `npm audit fix --omit=dev` only in a clean branch and review lockfile diff.
- If workspace lock already covers frontend, document that; if not, generate/commit frontend lockfile.

Python:

- Ensure environment uses:
  - `idna >= 3.15`
  - `langchain-core >= 1.3.3` or fixed compatible line; current `pyproject.toml` comments indicate langchain-core/langsmith removed, verify real env.
  - `langsmith >= 0.8.0` or removed.
  - `pip >= 26.1` at environment level.
  - `pytest >= 9.0.3`
  - `python-multipart >= 0.0.27`
  - `starlette >= 1.0.1` if compatible with FastAPI pin.
  - `urllib3 >= 2.7.0`
  - `chromadb` CVE-2026-45829: no fixed version reported; remove, isolate, or waive with compensating controls.

### P1.8.2: Write dependency CI tests/gates

CI should run:

```bash
npm audit --workspaces --omit=dev --audit-level=high
npm audit --omit=dev --audit-level=high -w middle
npm audit --omit=dev --audit-level=high -w frontend
npm audit --omit=dev --audit-level=high -w server || true  # only if server retained; otherwise no server workspace
pip-audit --strict
cargo audit --deny warnings
```

Use `--audit-level=moderate` after current moderate vulns are fixed.

Acceptance criteria:

- No known high/critical vulnerabilities.
- Moderate vulnerabilities either fixed or waived with owner/expiry.
- Frontend audit no longer skipped due to missing lock coverage.

Validation:

```bash
npm install
npm audit --omit=dev
npm audit --omit=dev -w middle
npm audit --omit=dev -w frontend
python -m pip install --upgrade pip
pip-audit
cargo audit
npm run build
python -m pytest tests/security -q
```

---

# PHASE P2 — medium findings, CI proof, and production hardening

## P2.1 — Rust unsafe serialization: SAFETY proofs, Miri, fuzzing

Audit finding: M1.

Primary files:

- `rust_memory/src/lib.rs`
- `rust_memory/src/checkpoint.rs`
- `rust_memory/Cargo.toml`
- `rust_memory/tests/**`
- maybe `rust_memory/fuzz/**`

Target:

- Every `unsafe` block has a nearby `// SAFETY:` comment documenting invariants.
- Checkpoint serialization validates alignment, lengths, endian, overflow, and malformed input.
- Miri tests cover core unsafe paths where possible.
- Fuzz target feeds malformed checkpoints.

Tasks:

1. Add tests for malformed header length, truncated f32 slice, huge length overflow, misaligned bytes if applicable, endian mismatch.
2. Add `SAFETY` comments before every `unsafe` block.
3. Prefer explicit byte encoding or `bytemuck`/`zerocopy` checked types if it reduces unsafe surface.
4. Add fuzz target with `cargo fuzz` if accepted; otherwise add property tests with `proptest`.
5. Add optional CI job:

```bash
cargo +nightly miri test -p rust_memory
cargo test -p rust_memory
cargo clippy --workspace --all-targets -- -D warnings
```

Acceptance criteria:

- `git grep -n "unsafe" rust_memory/src` shows every block has a SAFETY comment.
- Malformed checkpoint tests pass.
- cargo audit/clippy remain clean.

---

## P2.2 — Replace unsafe PyTorch checkpoint loading

Audit finding: M2.

Primary file:

- `src/training/amc_trainer.py:212`

Target:

- Prefer `safetensors` for untrusted/external checkpoints.
- If PyTorch checkpoint is required, use `torch.load(..., weights_only=True)` where supported.
- Load only from trusted artifact stores with checksum/signature verification.

Tasks:

1. Add regression test with a fake checkpoint path requiring safe-load helper.
2. Create helper like `load_trusted_weights(path, expected_sha256=None, allow_pickle=False)`.
3. Replace direct `torch.load` call with helper.
4. Default `allow_pickle=False`.
5. If backwards compatibility requires pickle, gate it behind explicit CLI/config flag and warning.

Validation:

```bash
python -m pytest tests/training/test_amc_trainer.py -q
bandit -q -r src/training -x tests
```

---

## P2.3 — Structured redaction and log-injection resistance

Audit finding: M10, H2, H5.

Primary files:

- `middle/src/server.ts`
- `middle/src/routes/files.ts`
- `middle/src/routes/rag.ts`
- `gateway/aurelius_server.py`
- logging middleware files
- frontend log display pages if any: `frontend/src/pages/Logs.tsx`, `frontend/src/pages/Activity` if exists

Target:

- Structured logs only.
- No concatenation of untrusted strings into log message prefixes.
- Redact secrets, URLs with credentials, Authorization headers, API keys, tokens, cookies, and control characters.
- Render logs with text escaping.

Tasks:

1. Create `middle/src/security/redaction.ts`.
2. Create Python equivalent if needed: `gateway/redaction.py`.
3. Add tests for control chars, ANSI escapes, `<script>`, newline log forging, bearer tokens, query secrets.
4. Replace activity append calls with structured fields:
   - `eventType`
   - `actorSubjectId`
   - `tenantId`
   - `resourceType`
   - `resourceId`
   - `safeSummary`
   - `redactedFields`
5. Ensure frontend renders message fields as text, not HTML.

Validation:

```bash
npm run test -w middle -- logs files rag command
npm run test -w frontend -- Logs Activity
python -m pytest tests/security -q
```

---

## P2.4 — CORS, CSP, request-size, and security headers parity

Audit findings: M4, M8, M9.

Primary files:

- `gateway/aurelius_api.py`
- `middle/src/server.ts`
- `middle/src/config.ts`
- `frontend/vite.config.*` if CSP affects dev
- deployment manifests
- tests: `tests/security/test_gateway_perimeter.py`, `middle/__tests__/security_headers.test.ts` new

Target:

- Wildcard CORS forbidden in production.
- Effective CORS tested for dev/staging/prod profiles.
- CSP forbids inline scripts and restricts `connect-src` / `ws-src`.
- Request size limits centralized and route-specific.

Tasks:

1. Add config validator that refuses `*` CORS in production.
2. Add middleware for security headers in BFF if not present:
   - `Content-Security-Policy`
   - `X-Content-Type-Options: nosniff`
   - `Referrer-Policy`
   - `Frame-Options` or CSP `frame-ancestors`
   - HSTS in TLS-terminated prod path if applicable.
3. Add route size policy table.
4. Enforce RAG/doc/file/chat body caps consistently.
5. Add tests for headers and limits.

Validation:

```bash
npm run test -w middle -- security_headers validation rag files
python -m pytest tests/security/test_gateway_perimeter.py -q
```

---

## P2.5 — Plugin trust model before dynamic loading exists

Audit finding: M5.

Primary files:

- `middle/src/routes/plugins.ts`
- `middle/__tests__/plugins.test.ts`
- docs: `docs/security/plugin-trust-model.md`

Target:

- Static plugin metadata remains safe.
- Any future dynamic plugin capability is blocked until signed manifests, capabilities, sandbox, and approval workflow exist.

Tasks:

1. Add explicit schema for plugin metadata.
2. Reject arbitrary entrypoint paths or URLs.
3. Add `dynamicLoadingEnabled=false` default and route tests proving dynamic loading endpoints do not exist.
4. Document required future trust model.

Validation:

```bash
npm run test -w middle -- plugins
```

---

## P2.6 — Lint/security baseline cleanup and CI gates

Audit findings: M3 and low/hygiene.

Primary files:

- `pyproject.toml`
- `.github/workflows/ci.yml`
- possibly `.ruff.toml` if split
- docs: `docs/remediation/security-2026-06-01/lint-baseline.md`

Target:

- Security lint is actionable.
- Global suppressions are replaced with local, justified suppressions over time.
- CI fails on new medium/high security findings.

Tasks:

1. Fix the one `F821` immediately because undefined names indicate broken paths.
2. Separate auto-fixable import/style cleanup from security remediation to avoid noisy PRs.
3. Run `ruff check --fix` only on intentionally scoped files or in a dedicated style PR.
4. Replace global `S105`, `S110`, `S311` ignores with local ignores where feasible.
5. Create Bandit baseline file reviewed by owner.
6. CI policy:
   - fail on new Bandit MEDIUM/HIGH
   - fail on new ruff `S` findings outside baseline
   - report LOW findings but do not block initially.

Validation:

```bash
ruff check .
bandit -q -r gateway agent src -x tests
python -m pytest tests/security -q
```

---

## P2.7 — Deployment manifest hardening and drift checks

Audit low/hygiene item.

Primary files:

- `deployment/**`
- `docker-compose*`
- `k8s/**`
- `helm/**`
- `.github/workflows/**`

Target:

- Non-root users, read-only rootfs where possible, dropped capabilities, seccomp, resource limits, pinned images, no dev compose in prod.
- Helm/raw manifests do not drift on security posture.

Tasks:

1. Inventory all deployment manifests.
2. Add a static policy test script under `tests/security/test_deployment_manifests.py`.
3. Required checks:
   - no wildcard CORS in prod env
   - no hardcoded API keys/secrets
   - image tags pinned or digest-pinned for prod
   - `runAsNonRoot: true`
   - `readOnlyRootFilesystem: true` where service allows
   - `allowPrivilegeEscalation: false`
   - capabilities dropped
   - resource limits present
   - no legacy server production service
4. Add CI job to run the manifest test.

Validation:

```bash
python -m pytest tests/security/test_deployment_manifests.py -q
```

---

# 5. Recommended branch and PR decomposition

Because this codebase is dirty and security-sensitive, do not ship one giant PR unless the user explicitly wants a monolithic remediation. Use risk-aligned PRs with crisp proof.

## PR 1: `fix/security-p0-legacy-license`

Includes:

- P0.1 legacy decommission/hard-disable.
- P0.2 license-derived API key removal.
- Tests proving no legacy production build and no self-issued credentials.

Why first:

- Eliminates the easiest catastrophic bypass before touching BFF internals.

Validation:

```bash
python -m pytest tests/security/test_legacy_server_decommissioned.py tests/security/test_license_activation_security.py tests/security/test_gateway_perimeter.py -q
npm run build
```

## PR 2: `fix/security-p0-bff-identity-ws`

Includes:

- P0.3 BFF identity preservation.
- P0.4 WS origin/auth/room policy.
- Frontend WS protocol alignment.

Why second:

- Fixes live credential blast radius and realtime exfiltration.

Validation:

```bash
npm run test -w middle -- provider_router models scopes auth websocket
npm run test -w frontend -- useWebSocket Login Notifications
npm run build -w middle
npm run build -w frontend
```

## PR 3: `fix/security-p1-boundaries`

Includes:

- P1.1 URL allowlists/DNS pinning/no runtime upstream mutation.
- P1.3 RAG tenant isolation.
- P1.4 streaming file upload.
- P1.5 command/scheduler schema.

Why third:

- These share boundary validation, quota, store, and audit patterns.

Validation:

```bash
npm run test -w middle -- url_policy provider_router rag files scheduler command persistence scopes
npm run build -w middle
```

## PR 4: `fix/security-p1-auth-session-sandbox-deps`

Includes:

- P1.2 frontend credential storage/session auth.
- P1.6 sandbox mode hardening.
- P1.7 auth parity.
- P1.8 dependencies.

This may be split further if auth/session refactor grows.

Validation:

```bash
npm run test -w frontend -- Login Settings
npm run test -w middle -- auth scopes
python -m pytest tests/tools/test_code_runner_exploits.py tests/security tests/gateway -q
npm audit --omit=dev
pip-audit
cargo audit
```

## PR 5: `chore/security-p2-proof-ci`

Includes:

- Rust unsafe comments/Miri/fuzz.
- torch.load replacement.
- structured redaction.
- CORS/CSP/security headers.
- plugin trust model.
- lint/audit CI gates.
- deployment manifest hardening.

Validation:

```bash
cargo clippy --workspace --all-targets -- -D warnings
cargo audit
python -m pytest tests/security tests/training/test_amc_trainer.py -q
npm run test -w middle
npm run test -w frontend
ruff check .
bandit -q -r gateway agent src -x tests
```

---

# 6. Definition of done for the entire remediation campaign

The campaign is complete only when all of these are true:

## 6.1 P0/P1 closure

- C1 legacy server cannot be built/deployed/run through production/default paths, or is fully hardened and tested.
- C2 license-derived credential flow is gone.
- C3 WebSockets require valid Origin + auth + per-room authorization.
- C4 BFF no longer forwards service key as default user principal.
- H1 upstream URL policy blocks SSRF and credential exfiltration.
- H2/H3 command/scheduler raw privileged text replay is replaced with typed command policy.
- H4 RAG is tenant/owner scoped.
- H5 files stream and sniff safely.
- H6 sandbox distinguishes trusted local subprocess from hostile isolated mode.
- H7 auth behavior is consistent across entrypoints.
- H8 browser no longer stores long-lived API key in localStorage.
- H9 dependencies fixed or waived with compensating controls.

## 6.2 Automated proof

Required commands must pass or have documented pre-existing non-security failures:

```bash
python -m pytest tests/security tests/gateway/test_t3_gateway_fail_closed.py tests/tools/test_code_runner_exploits.py -q
npm run test -w middle
npm run test -w frontend
npm run build -w middle
npm run build -w frontend
cargo clippy --workspace --all-targets -- -D warnings
cargo audit
npm audit --omit=dev
npm audit --omit=dev -w middle
npm audit --omit=dev -w frontend
pip-audit
ruff check .
bandit -q -r gateway agent src -x tests
```

## 6.3 Manual proof

Manual smoke checklist:

- Start BFF + Python gateway in dev with auth required.
- Login through UI; verify no API key in localStorage.
- Attempt direct protected API call without auth -> rejected.
- Attempt bad API key -> rejected.
- Attempt cross-origin WS from disallowed Origin -> rejected.
- Attempt tenant A to subscribe/read tenant B room/doc -> rejected.
- Upload oversized file -> rejected early; temp file cleaned.
- Schedule command without approval -> rejected.
- Try changing upstream URL through runtime config -> rejected.
- Verify logs contain request IDs and redacted fields, not raw secrets/control chars.

---

# 7. Subagent execution model if using AI implementers

Use subagents only when a tranche has isolated file paths. Because previous Aurelius work had scope-drift, every subagent task must include this guard:

```text
You may ONLY modify these exact paths: [...]. If any other file needs modification, stop and report it. Do not change unrelated tests, __init__.py files, scripts, or docs. Do not use git add -A. After finishing, report exact files changed and commands run.
```

After every subagent:

```bash
git diff --name-only HEAD
```

If any file outside the allowlist changed:

```bash
git checkout HEAD -- <drifted-file>
```

Two-stage review per task:

1. Spec compliance review: did it implement exactly the planned behavior and no extra behavior?
2. Security/code-quality review: does it preserve trust boundaries, tests, style, and fail-closed behavior?

Do not let implementer self-review substitute for independent review.

---

# 8. Risk register and fallback decisions

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| Auth/session refactor breaks dev UX | Medium | Medium | Feature flag session mode, keep dev BYOK in sessionStorage only, document migration |
| Decommissioning server breaks hidden workflow | Medium | High | Inventory scripts/deployments first; provide explicit legacy dev-only escape hatch |
| Dependency upgrades break TypeScript or FastAPI compatibility | Medium | Medium | Upgrade in dedicated branch, inspect lockfile, run build/test/audit, pin compatible versions |
| Sandbox real isolation is too large for first sprint | High | High | Default hostile execution to disabled; keep trusted subprocess dev mode only; implement backend later |
| URL DNS pinning too complex initially | Medium | High | Start with strict origin allowlist + private IP literal rejection; add DNS resolution/pinning before prod |
| RAG persistent store migration conflicts with current memory architecture | Medium | High | Implement store abstraction and memory-store first; SQLite migration second; add tenant namespace tests before persistence |
| CI becomes noisy due to existing lint debt | High | Medium | Baseline current findings; fail on new security findings first; style cleanup separate PR |

---

# 9. Suggested implementation order inside each task

For each finding, use strict TDD:

1. Write the smallest failing test that proves the vulnerability.
2. Run only that test and confirm it fails for the expected reason.
3. Implement minimal fix.
4. Run the specific test and confirm pass.
5. Run nearby test file/suite.
6. Run relevant scanner/build command.
7. Scope-audit changed files.
8. Commit exact files only.
9. Update remediation register with test and proof command.

Commit style:

```bash
git add exact/file1 exact/file2 exact/test_file
git commit -m "fix(security): reject legacy license-derived api keys"
```

Never use broad staging in this repo.

---

# 10. Reviewer checklist for final security PRs

For every PR, reviewer must answer:

- Does this remove the vulnerability or only hide the symptom?
- Does every new failure mode fail closed?
- Does any user-triggered route use a service credential?
- Can a malicious tenant/user access another tenant/user resource?
- Are all new IDs cryptographically random where enumeration matters?
- Are all new tokens audience-bound, short-lived, and not logged?
- Are all untrusted strings structured/redacted before logging?
- Are browser secrets protected from XSS by storage design, not just by absence of current sinks?
- Are deployment defaults safe, or only local defaults?
- Are tests adversarial enough to fail on the old bug?
- Did the PR modify unrelated files?
- Are all dependency and lockfile changes explainable?

---

# 11. Immediate next action recommendation

Do not start with dependency upgrades or lint cleanup. Start with this sequence:

1. Create clean worktree from `46ee2f13`.
2. Write and pass the legacy decommission tests.
3. Remove legacy server from default build/deploy paths.
4. Delete/neutralize license-derived API key issuance.
5. Fix BFF service-key forwarding.
6. Fix WebSocket Origin/auth/room policy.

That gets the catastrophic credential and realtime-exfiltration risks out first. Everything else can then land behind clearer boundaries.
