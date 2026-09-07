# 02 — P0 BFF/WS Spec Review

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

    ROLE
    You are the independent SPEC-COMPLIANCE reviewer for `P0 BFF identity + WebSocket auth remediation`. You did not implement the code. Review only whether the implementation satisfies the remediation plan and the task prompt. Do not fix code unless explicitly asked in a later prompt.

    EXPECTED SPEC
    - `serviceApiKey` / `AURELIUS_SERVICE_KEY` are used only in config loading and a central policy helper.
- User-triggered upstream requests use user-bound subject/token/exchange, not shared service principal.
- WS Origin allowlist is enforced before auth success or events.
- WS auth is required and audience/expiry-bound if token based.
- WS room authorization is tenant/scope aware.
- Frontend uses the same room schema as BFF.
- Register rows SEC-P0-03 and SEC-P0-04 are updated with evidence.

    PATH SCOPE TO REVIEW
    Review only these paths unless the diff shows unexpected drift. If unexpected files changed, mark REQUEST_CHANGES.

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

    REVIEW STEPS
    ```bash
    cd /Users/christienantonio/aurelius-security-remediation
    git diff --name-only HEAD
    git diff --stat HEAD
    git diff HEAD -- middle/src/config.ts middle/src/provider_router.ts middle/src/server.ts middle/src/routes/models.ts middle/src/routes/auth.ts middle/src/middleware/auth.ts middle/src/store/types.ts middle/src/auth/types.ts middle/src/auth/ws_token.ts middle/src/ws/handler.ts middle/src/ws/rooms.ts frontend/src/hooks/useWebSocket.ts frontend/src/hooks/useAuth.ts frontend/src/stores/apiStore.ts frontend/src/services/api.ts middle/__tests__/provider_router.test.ts middle/__tests__/models.test.ts middle/__tests__/scopes.test.ts middle/__tests__/auth.test.ts middle/__tests__/websocket_auth.test.ts
    git status --short --branch
npm run test -w middle -- provider_router models scopes auth websocket
npm run test -w frontend -- useWebSocket Notifications
npm run build -w middle
npm run build -w frontend
git grep -n "serviceApiKey\|AURELIUS_SERVICE_KEY" middle/src || true
git grep -n "channel" frontend/src/hooks/useWebSocket.ts middle/src/ws || true
git diff --name-only HEAD
    ```

    SPEC REVIEW CHECKLIST
    - All required tests were added before or with implementation.
    - Tests actually target the audit finding, not just happy path behavior.
    - All required code paths from the prompt were addressed.
    - No implementation-only behavior contradicts the remediation plan.
    - No extra features, broad rewrites, or unrelated cleanup were added.
    - Register was updated with tests/commands/residual risk.
    - Changed files are inside the allowlist.

    RETURN FORMAT
    ```text
    SPEC_VERDICT: PASS | REQUEST_CHANGES | BLOCKED
    Scope drift: yes/no; details
    Missing requirements:
    - ...
    Overbuilt / unrelated changes:
    - ...
    Required fixes before security review:
    - ...
    Evidence reviewed:
    - commands/files
    ```
