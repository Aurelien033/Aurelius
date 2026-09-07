# 02 — P0 BFF Identity + WebSocket Implementer

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
    - Branch name: `fix/security-p0-bff-identity-ws`
    - Commit message, if you are explicitly asked to commit: `fix(security): preserve bff identity and enforce websocket room auth`

    MISSION
    Fix P0 findings C3 and C4.

C4: BFF must preserve user identity and scopes in upstream calls. Service credentials may only be used for narrowly allowlisted internal-service policies and never for arbitrary user-triggered provider/model calls.

C3: WebSockets must validate Origin, authenticate before emitting data, use a short-lived/cookie/session-compatible token flow, and authorize per-room subscription by tenant and scope. Frontend and BFF must agree on `{ type: 'subscribe', room: '...' }` schema.

    STRICT PATH ALLOWLIST
    You may ONLY modify these paths. If any other path is needed, stop and report it.

    - `middle/src/config.ts`
- `middle/src/provider_router.ts`
- `middle/src/server.ts`
- `middle/src/routes/models.ts`
- `middle/src/routes/auth.ts`
- `middle/src/middleware/auth.ts`
- `middle/src/store/types.ts`
- `middle/src/auth/types.ts`
- `middle/src/auth/ws_token.ts`
- `middle/src/ws/handler.ts`
- `middle/src/ws/rooms.ts`
- `frontend/src/hooks/useWebSocket.ts`
- `frontend/src/hooks/useAuth.ts`
- `frontend/src/stores/apiStore.ts`
- `frontend/src/services/api.ts`
- `middle/__tests__/provider_router.test.ts`
- `middle/__tests__/models.test.ts`
- `middle/__tests__/scopes.test.ts`
- `middle/__tests__/auth.test.ts`
- `middle/__tests__/websocket_auth.test.ts`
- `frontend/src/test/pages/Notifications.test.tsx`
- `frontend/src/test/hooks/useWebSocket.test.tsx`
- `docs/remediation/security-2026-06-01/register.md`

    REQUIRED RED TESTS FIRST
    Add or extend tests for these behaviors before implementation. You must run them and verify they fail before changing production code.

    - Extend provider/model/scopes/auth tests proving user-triggered provider calls do not attach service credentials and do carry user subject/tenant/scope.
- Add `middle/__tests__/websocket_auth.test.ts` proving missing token/session rejects, bad Origin rejects, allowed Origin + valid auth accepts, no connected payload appears before auth, tenant-mismatched room rejects, admin room without admin scope rejects, allowed tenant room accepts, and old `{ channel }` schema is either rejected or explicitly deprecated.
- Add/update frontend hook/page tests proving the frontend sends canonical `{ type: 'subscribe', room }` and does not put long-lived API keys in the WS URL.

    IMPLEMENTATION REQUIREMENTS
    - Implement the minimum safe change that satisfies the tests.
    - Prefer central helpers for auth, URL policy, redaction, and validation instead of repeated ad hoc checks.
    - Do not weaken an existing test to make it pass.
    - Do not skip tests because unrelated baseline failures exist; run the targeted tests and document any unrelated baseline failures separately.
    - Do not add broad compatibility fallbacks that silently accept insecure legacy behavior.

    ACCEPTANCE CRITERIA
    - `serviceApiKey` / `AURELIUS_SERVICE_KEY` are used only in config loading and a central policy helper.
- User-triggered upstream requests use user-bound subject/token/exchange, not shared service principal.
- WS Origin allowlist is enforced before auth success or events.
- WS auth is required and audience/expiry-bound if token based.
- WS room authorization is tenant/scope aware.
- Frontend uses the same room schema as BFF.
- Register rows SEC-P0-03 and SEC-P0-04 are updated with evidence.

    VALIDATION COMMANDS
    Run these commands from `/Users/christienantonio/aurelius-security-remediation` unless a command says otherwise.

    ```bash
    git status --short --branch
npm run test -w middle -- provider_router models scopes auth websocket
npm run test -w frontend -- useWebSocket Notifications
npm run build -w middle
npm run build -w frontend
git grep -n "serviceApiKey\|AURELIUS_SERVICE_KEY" middle/src || true
git grep -n "channel" frontend/src/hooks/useWebSocket.ts middle/src/ws || true
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
