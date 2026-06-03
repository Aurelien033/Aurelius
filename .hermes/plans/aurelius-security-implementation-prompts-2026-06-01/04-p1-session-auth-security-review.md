# 04 — P1 Session Auth Security Review

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
    You are the independent SECURITY/CODE-QUALITY reviewer for `P1 browser credential/session remediation`. Assume the implementer may have made plausible-looking but unsafe changes. Your job is to break the fix conceptually and with tests/greps where practical. Do not implement fixes unless explicitly asked in a later prompt.

    THREAT MODEL TO VERIFY
    Attackers use XSS/localStorage reads, CSRF, stale sessions, WS URL leakage, logout bypass, and developer BYOK mode confusion to recover long-lived credentials.

    PATH SCOPE TO REVIEW
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

    REVIEW COMMANDS
    ```bash
    cd /Users/christienantonio/aurelius-security-remediation
    git diff --name-only HEAD
    git diff --stat HEAD
    git diff HEAD -- middle/src/routes/auth.ts middle/src/middleware/auth.ts middle/src/store/types.ts middle/src/store/memory-store.ts middle/src/store/sqlite-store.ts middle/src/config.ts middle/src/security/csrf.ts middle/src/security/session.ts frontend/src/stores/apiStore.ts frontend/src/stores/persist.ts frontend/src/services/api.ts frontend/src/hooks/useAuth.ts frontend/src/hooks/useWebSocket.ts frontend/src/pages/Login.tsx frontend/src/pages/Settings.tsx frontend/src/test/pages/Login.test.tsx frontend/src/test/pages/Settings.test.tsx frontend/src/test/hooks/useAuth.test.tsx frontend/src/test/hooks/useWebSocket.test.tsx middle/__tests__/auth.test.ts
    git status --short --branch
npm run test -w frontend -- Login Settings useAuth useWebSocket
npm run test -w middle -- auth scopes
npm run build -w frontend
npm run build -w middle
git grep -n "localStorage.*api\|apiKey.*localStorage\|setItem(.*token\|setItem(.*secret\|setItem(.*password" frontend/src || true
git diff --name-only HEAD
    ```

    SECURITY REVIEW CHECKLIST
    - Fail-closed behavior on missing/malformed/unauthorized input.
    - Auth before business logic.
    - Authorization at resource boundary.
    - No service/admin credential used for user-triggered operations.
    - No long-lived secret in browser-readable persistent storage.
    - No raw untrusted text in shell/eval/HTML/log-prefix contexts.
    - No runtime mutation of auth/CORS/upstream/sandbox/rate-limit critical config unless break-glass is explicitly tested.
    - No new secret equality with `==`/`!=` where constant-time compare is required.
    - No dependency lockfile or package drift that is not explainable.
    - No tests weakened, skipped, or xfailed to hide failures.

    RETURN FORMAT
    ```text
    SECURITY_VERDICT: APPROVED | REQUEST_CHANGES | BLOCKED
    Critical issues:
    - ...
    Important issues:
    - ...
    Minor issues:
    - ...
    Commands run:
    - ...
    Residual risk decision:
    - acceptable/not acceptable and why
    ```
