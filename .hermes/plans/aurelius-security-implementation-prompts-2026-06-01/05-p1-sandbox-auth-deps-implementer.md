# 05 — P1 Sandbox/Auth Parity/Dependencies Implementer

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
    - Branch name: `fix/security-p1-sandbox-auth-deps`
    - Commit message, if you are explicitly asked to commit: `fix(security): fail closed sandbox auth parity and dependency audits`

    MISSION
    Fix P1 findings H6, H7, and H9.

H6: code execution sandbox must distinguish disabled, trusted local subprocess, and real isolated hostile-code modes. Production default must fail closed without an isolation backend.
H7: auth behavior must be consistent across Python gateway, BFF, and any retained server path; constant-time comparison for secrets; runtime config cannot weaken auth perimeter.
H9: dependency vulnerabilities must be upgraded or explicitly waived with owner/expiry/compensating controls, and audit coverage must include frontend.

    STRICT PATH ALLOWLIST
    You may ONLY modify these paths. If any other path is needed, stop and report it.

    - `agent/code_execution_tool.py`
- `agent/code_execution_sandbox.py`
- `src/security/sandbox_executor.py`
- `tests/tools/test_code_runner_exploits.py`
- `gateway/auth_middleware.py`
- `gateway/aurelius_api.py`
- `gateway/aurelius_server.py`
- `tests/security/test_gateway_perimeter.py`
- `tests/gateway/test_t3_gateway_fail_closed.py`
- `middle/src/middleware/auth.ts`
- `middle/src/routes/config.ts`
- `middle/__tests__/auth.test.ts`
- `middle/__tests__/config.test.ts`
- `pyproject.toml`
- `requirements.txt`
- `requirements-dev.txt`
- `package.json`
- `package-lock.json`
- `middle/package.json`
- `middle/package-lock.json`
- `frontend/package.json`
- `frontend/package-lock.json`
- `server/package.json`
- `server/package-lock.json`
- `.github/workflows/ci.yml`
- `docs/remediation/security-2026-06-01/register.md`

    REQUIRED RED TESTS FIRST
    Add or extend tests for these behaviors before implementation. You must run them and verify they fail before changing production code.

    - Extend `tests/tools/test_code_runner_exploits.py`: default denies untrusted code, trusted_subprocess requires dev flag, isolated mode fails closed if backend missing, network/filesystem denied where backend exists, timeout kills execution unit.
- Extend gateway perimeter tests: protected routes fail closed on missing/wrong keys, correct scoped access works, metrics key constant-time path, runtime config cannot mutate security-critical keys.
- Middle auth/config tests mirror the same denylist behavior.
- Dependency audit proof must show npm/pip/cargo status and frontend audit coverage or explicit reason.

    IMPLEMENTATION REQUIREMENTS
    - Implement the minimum safe change that satisfies the tests.
    - Prefer central helpers for auth, URL policy, redaction, and validation instead of repeated ad hoc checks.
    - Do not weaken an existing test to make it pass.
    - Do not skip tests because unrelated baseline failures exist; run the targeted tests and document any unrelated baseline failures separately.
    - Do not add broad compatibility fallbacks that silently accept insecure legacy behavior.

    ACCEPTANCE CRITERIA
    - `AURELIUS_CODE_EXECUTION_MODE=disabled|trusted_subprocess|isolated` or equivalent explicit mode exists.
- Production/untrusted execution refuses unless isolated backend configured.
- Auth secret comparison is constant-time or fixed-hash equivalent.
- Runtime config cannot mutate auth, CORS, upstream URL, sandbox mode, metrics key, or rate-limit-disable keys.
- Known dependency vulnerabilities are fixed or waived with dated owner/expiry and compensating controls.
- Register rows SEC-P1-06, SEC-P1-07, SEC-P1-09 are updated with evidence.

    VALIDATION COMMANDS
    Run these commands from `/Users/christienantonio/aurelius-security-remediation` unless a command says otherwise.

    ```bash
    git status --short --branch
python -m pytest tests/tools/test_code_runner_exploits.py tests/security/test_gateway_perimeter.py tests/gateway/test_t3_gateway_fail_closed.py -q
npm run test -w middle -- auth config
npm run build -w middle
npm run build -w frontend
cargo audit
npm audit --omit=dev || true
npm audit --omit=dev -w middle || true
npm audit --omit=dev -w frontend || true
pip-audit || true
git grep -n "==.*key\|!=.*key\|compare_digest\|timingSafeEqual" gateway middle/src server/src || true
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
