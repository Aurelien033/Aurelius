# 03 — P1 URL/RAG/File/Command Boundaries Implementer

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
    - Branch name: `fix/security-p1-boundaries`
    - Commit message, if you are explicitly asked to commit: `fix(security): enforce upstream rag file and command boundaries`

    MISSION
    Fix P1 findings H1, H2, H3, H4, and H5.

H1: upstream URL config must be validated through an allowlist and must never receive credentials unless policy-valid.
H4: RAG documents/chunks must be tenant/owner scoped, quota bounded, and non-enumerable.
H5: file uploads must stream with hard byte limits, server-side content sniffing, sanitized filenames, temp cleanup, and upload quotas.
H2/H3: `/api/command` and scheduler must use typed command policies, scopes, approval metadata, expiry/max-runs, and safe structured logs. No raw privileged text bridge.

    STRICT PATH ALLOWLIST
    You may ONLY modify these paths. If any other path is needed, stop and report it.

    - `middle/src/security/url_policy.ts`
- `middle/src/security/validation.ts`
- `middle/src/config.ts`
- `middle/src/engine.ts`
- `middle/src/provider_router.ts`
- `middle/src/server.ts`
- `middle/src/routes/models.ts`
- `middle/src/routes/config.ts`
- `middle/src/routes/rag.ts`
- `middle/src/routes/files.ts`
- `middle/src/routes/scheduler.ts`
- `middle/src/store/types.ts`
- `middle/src/store/memory-store.ts`
- `middle/src/store/sqlite-store.ts`
- `middle/src/middleware/rate-limiter.ts`
- `frontend/src/pages/ScheduledTasks.tsx`
- `frontend/src/components/CommandPalette.tsx`
- `middle/__tests__/url_policy.test.ts`
- `middle/__tests__/provider_router.test.ts`
- `middle/__tests__/config.test.ts`
- `middle/__tests__/rag.test.ts`
- `middle/__tests__/files.test.ts`
- `middle/__tests__/scheduler.test.ts`
- `middle/__tests__/command.test.ts`
- `middle/__tests__/persistence.test.ts`
- `middle/__tests__/scopes.test.ts`
- `docs/remediation/security-2026-06-01/register.md`

    REQUIRED RED TESTS FIRST
    Add or extend tests for these behaviors before implementation. You must run them and verify they fail before changing production code.

    - Add `middle/__tests__/url_policy.test.ts` for HTTPS prod allowlist, dev localhost exception, metadata IP rejection, private IP rejection in production, userinfo URL rejection, and no auth header attachment on invalid URL.
- Add `middle/__tests__/rag.test.ts` for cross-user/cross-tenant list/search/read/delete rejection, admin scope audit, oversized doc rejection, UUID IDs, and chunk deletion/tombstone behavior.
- Add `middle/__tests__/files.test.ts` for oversize early abort/temp cleanup, MIME mismatch rejection, filename sanitization, concurrent upload quota, and log-injection-safe metadata.
- Add/extend scheduler/command tests proving raw string commands reject, unknown command rejects, non-admin admin command rejects, scheduled commands require approval/max-run/expiry, emergency disable works, and sensitive params redact.

    IMPLEMENTATION REQUIREMENTS
    - Implement the minimum safe change that satisfies the tests.
    - Prefer central helpers for auth, URL policy, redaction, and validation instead of repeated ad hoc checks.
    - Do not weaken an existing test to make it pass.
    - Do not skip tests because unrelated baseline failures exist; run the targeted tests and document any unrelated baseline failures separately.
    - Do not add broad compatibility fallbacks that silently accept insecure legacy behavior.

    ACCEPTANCE CRITERIA
    - All provider/upstream calls use validated URL objects or central URL policy helper.
- Runtime config cannot mutate upstream/auth/CORS/sandbox/rate-limit security keys without explicit tested break-glass policy.
- RAG routes filter every operation by tenant and authorization.
- Upload path does not buffer whole request in memory and never trusts client filename/MIME alone.
- Scheduler stores typed command envelopes with owner, scope, approval, max runs, and expiry.
- Raw text `/api/command` privileged execution path is removed or turned into proposal-only flow.
- Register rows SEC-P1-01 through SEC-P1-05 are updated with evidence.

    VALIDATION COMMANDS
    Run these commands from `/Users/christienantonio/aurelius-security-remediation` unless a command says otherwise.

    ```bash
    git status --short --branch
npm run test -w middle -- url_policy provider_router config rag files scheduler command persistence scopes
npm run build -w middle
git grep -n "upstreamUrl\|pythonUrl\|provider.*url" middle/src || true
git grep -n "Date.now().*Math.random\|Math.random().*Date.now" middle/src/routes middle/src/store || true
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
