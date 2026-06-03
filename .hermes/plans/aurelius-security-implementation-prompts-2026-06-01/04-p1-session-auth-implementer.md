# 04 — P1 Browser Session Auth Implementer

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
    - Branch name: `fix/security-p1-session-auth`
    - Commit message, if you are explicitly asked to commit: `fix(security): replace browser api key persistence with session auth`

    MISSION
    Fix P1 finding H8 and support H7/C3.

Browser auth must move away from long-lived API keys in localStorage. Preferred production mode is BFF session cookies with HttpOnly/Secure/SameSite flags plus CSRF protection for unsafe cookie-authenticated methods. Local developer BYOK mode may exist only as explicit `local_byok_dev`, with memory/sessionStorage TTL and visible warning, never localStorage by default.

    STRICT PATH ALLOWLIST
    You may ONLY modify these paths. If any other path is needed, stop and report it.

    - `middle/src/routes/auth.ts`
- `middle/src/middleware/auth.ts`
- `middle/src/store/types.ts`
- `middle/src/store/memory-store.ts`
- `middle/src/store/sqlite-store.ts`
- `middle/src/config.ts`
- `middle/src/security/csrf.ts`
- `middle/src/security/session.ts`
- `frontend/src/stores/apiStore.ts`
- `frontend/src/stores/persist.ts`
- `frontend/src/services/api.ts`
- `frontend/src/hooks/useAuth.ts`
- `frontend/src/hooks/useWebSocket.ts`
- `frontend/src/pages/Login.tsx`
- `frontend/src/pages/Settings.tsx`
- `frontend/src/test/pages/Login.test.tsx`
- `frontend/src/test/pages/Settings.test.tsx`
- `frontend/src/test/hooks/useAuth.test.tsx`
- `frontend/src/test/hooks/useWebSocket.test.tsx`
- `middle/__tests__/auth.test.ts`
- `middle/__tests__/scopes.test.ts`
- `docs/remediation/security-2026-06-01/register.md`

    REQUIRED RED TESTS FIRST
    Add or extend tests for these behaviors before implementation. You must run them and verify they fail before changing production code.

    - Frontend tests proving login does not call `localStorage.setItem` for api keys/tokens/secrets and logout clears session-visible state.
- Persist-layer tests proving fields matching apiKey/token/secret/password are denied unless explicitly waived.
- Middle auth tests proving login sets HttpOnly/Secure-in-production/SameSite cookies, cookie auth sets AuthSubject, unsafe methods without CSRF fail, logout invalidates session, and WS token endpoint is short-lived/audience-bound if token mode is used.

    IMPLEMENTATION REQUIREMENTS
    - Implement the minimum safe change that satisfies the tests.
    - Prefer central helpers for auth, URL policy, redaction, and validation instead of repeated ad hoc checks.
    - Do not weaken an existing test to make it pass.
    - Do not skip tests because unrelated baseline failures exist; run the targeted tests and document any unrelated baseline failures separately.
    - Do not add broad compatibility fallbacks that silently accept insecure legacy behavior.

    ACCEPTANCE CRITERIA
    - `frontend/src` no longer persists long-lived API keys/tokens/secrets/passwords in localStorage.
- API client uses cookie/session mode with `credentials: 'include'` where appropriate.
- `AURELIUS_AUTH_MODE=session|local_byok_dev` or equivalent explicit mode exists.
- CSRF defense is active for unsafe cookie-auth methods.
- WS auth integrates with the session/token flow without long-lived URL secrets.
- Register row SEC-P1-08 is updated with evidence.

    VALIDATION COMMANDS
    Run these commands from `/Users/christienantonio/aurelius-security-remediation` unless a command says otherwise.

    ```bash
    git status --short --branch
npm run test -w frontend -- Login Settings useAuth useWebSocket
npm run test -w middle -- auth scopes
npm run build -w frontend
npm run build -w middle
git grep -n "localStorage.*api\|apiKey.*localStorage\|setItem(.*token\|setItem(.*secret\|setItem(.*password" frontend/src || true
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
