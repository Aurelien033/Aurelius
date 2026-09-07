# 01 — P0 Legacy Server + License Credential Implementer

    UNIVERSAL GUARDRAILS — NON-NEGOTIABLE

You are implementing security remediation for Aurelius.

Source materials:
- Audit report: /Users/christienantonio/Desktop/AI:ML Research/aurelius-security-code-review-2026-06-01.md
- Remediation plan: /Users/christienantonio/Desktop/AI Plans/aurelius-security-remediation-plan-2026-06-01.md
- Original repo: /Users/christienantonio/aurelius
- Required clean implementation worktree: /Users/christienantonio/aurelius-security-remediation
- Audited base commit: 46ee2f13

Operating mode:
- This is implementation, not research prose.
- Follow strict TDD: write failing regression tests first, run them and confirm they fail for the expected security reason, then implement the minimum fix, then rerun targeted and nearby tests.
- Fail closed on missing credentials, malformed credentials, missing origin, missing tenant, missing scope, invalid upstream URL, unknown room/channel, and missing sandbox backend.
- Preserve user identity across Browser -> BFF -> Gateway. Never collapse user-triggered operations into a shared service/admin principal.
- Never store long-lived API keys, tokens, secrets, passwords, cookies, or service credentials in browser-readable persistent storage.
- Treat all untrusted input as data: never raw shell, never eval, never HTML, never unstructured log prefix.
- Runtime config must not mutate security-critical values unless an explicit break-glass path exists and is tested.
- Do not print, copy, invent, or persist real secrets. If a secret-like value appears, write `[REDACTED]` in notes.

Git discipline:
- Work only in the clean worktree: /Users/christienantonio/aurelius-security-remediation
- If the worktree does not exist, create it from 46ee2f13; do not modify the dirty original repo.
- Never use `git add -A`, `git add .`, or broad globs.
- Stage exact files only.
- After every change and before every final answer, run: `git diff --name-only HEAD`
- If any modified file is outside the allowlist in this prompt, revert it immediately with `git checkout HEAD -- <path>` and explain the scope drift.
- Do not commit unless this prompt explicitly says to commit. If committing, commit only exact allowed paths.

Required implementation loop:
1. Confirm clean worktree and branch.
2. Read the exact files listed in the task.
3. Write failing tests first.
4. Run targeted tests and verify RED.
5. Implement the minimal fix.
6. Run targeted tests and verify GREEN.
7. Run nearby build/lint/audit commands listed in this prompt.
8. Scope-audit changed files.
9. Update `docs/remediation/security-2026-06-01/register.md` with findings closed, tests added, commands run, and residual risk.
10. Return exact files changed, commands run, test output summary, and remaining risks.

If a needed file is not in the allowlist:
- STOP.
- Do not modify it.
- Report the exact additional path and why it is required.

    ---

    BRANCH / PR TARGET
    - Branch name: `fix/security-p0-legacy-license`
    - Commit message, if you are explicitly asked to commit: `fix(security): decommission legacy server and reject license-derived api keys`

    MISSION
    Fix P0 findings C1 and C2.

C1: the legacy Node `server/` surface must not be built, run, or deployed by default. Prefer decommissioning from default root scripts. If retained for local dev, it must be explicitly named unsafe/deprecated and fail closed unless `AURELIUS_ENABLE_LEGACY_SERVER=true`.

C2: license activation must never mint or mutate API keys from user-supplied license material. Replace deterministic license-derived API key behavior with either 410 Gone for legacy activation or signed-license verification that does not create privileged credentials.

    STRICT PATH ALLOWLIST
    You may ONLY modify these paths. If any other path is needed, stop and report it.

    - `package.json`
- `server/package.json`
- `server/DEPRECATED.md`
- `server/src/deprecation_guard.ts`
- `server/src/main.ts`
- `server/src/app.ts`
- `server/src/routes/index.ts`
- `server/src/routes/license.ts`
- `server/src/routes/proxy.ts`
- `server/src/ws/index.ts`
- `server/src/ws/hub.ts`
- `gateway/aurelius_server.py`
- `gateway/auth_middleware.py`
- `gateway/aurelius_api.py`
- `middle/src/routes/auth.ts`
- `frontend/src/components/LicenseGate.tsx`
- `frontend/src/pages/Login.tsx`
- `tests/security/test_legacy_server_decommissioned.py`
- `tests/security/test_license_activation_security.py`
- `tests/security/test_gateway_perimeter.py`
- `tests/gateway/test_t3_gateway_fail_closed.py`
- `docs/remediation/security-2026-06-01/register.md`

    REQUIRED RED TESTS FIRST
    Add or extend tests for these behaviors before implementation. You must run them and verify they fail before changing production code.

    - Create `tests/security/test_legacy_server_decommissioned.py` proving root default scripts do not build/test/lint/run `server`, and production manifests/scripts do not reference legacy server without an explicit dev-only guard.
- Create `tests/security/test_license_activation_security.py` proving a syntactically valid license string cannot become an API key, invalid license activation fails closed, and license activation cannot mutate `require_auth`, `api_key`, metrics key, CORS, upstream URL, or sandbox mode.
- Extend `tests/security/test_gateway_perimeter.py` or `tests/gateway/test_t3_gateway_fail_closed.py` only if needed to lock the auth perimeter.

    IMPLEMENTATION REQUIREMENTS
    - Implement the minimum safe change that satisfies the tests.
    - Prefer central helpers for auth, URL policy, redaction, and validation instead of repeated ad hoc checks.
    - Do not weaken an existing test to make it pass.
    - Do not skip tests because unrelated baseline failures exist; run the targeted tests and document any unrelated baseline failures separately.
    - Do not add broad compatibility fallbacks that silently accept insecure legacy behavior.

    ACCEPTANCE CRITERIA
    - `npm run build` no longer builds `server` by default.
- No default `dev:server`/`build:server` path remains unless renamed as explicit unsafe legacy and gated.
- Any retained legacy server startup fails closed without `AURELIUS_ENABLE_LEGACY_SERVER=true`.
- License activation does not return, set, derive, or persist API keys.
- Raw or deterministic license-derived key patterns are gone.
- Register rows SEC-P0-01 and SEC-P0-02 are updated with evidence.

    VALIDATION COMMANDS
    Run these commands from `/Users/christienantonio/aurelius-security-remediation` unless a command says otherwise.

    ```bash
    git status --short --branch
python -m pytest tests/security/test_legacy_server_decommissioned.py tests/security/test_license_activation_security.py tests/security/test_gateway_perimeter.py tests/gateway/test_t3_gateway_fail_closed.py -q
npm run build
npm run test -w middle -- auth
npm run test -w frontend -- Login
git grep -n "license.*api\|api.*license\|key\[-16:\]\|AURELIUS-" server gateway middle frontend || true
git diff --name-only HEAD
    ```

    FINAL RESPONSE FORMAT
    Return exactly this structure:

    ```text
    RESULT: PASS | NEEDS_FOLLOWUP | BLOCKED
    Branch: <branch>
    Files changed:
    - <path>
    Tests written/updated:
    - <path>::<test or behavior>
    Commands run:
    - <command> => <PASS/FAIL and short output summary>
    Security invariants proven:
    - <invariant>
    Scope audit:
    - git diff --name-only HEAD => <list>
    Register updates:
    - <register rows touched>
    Remaining risks / follow-up:
    - <risk or none>
    ```
