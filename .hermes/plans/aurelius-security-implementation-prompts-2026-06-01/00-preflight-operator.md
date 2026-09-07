# 00 — Preflight Operator Prompt

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

    MISSION
    Prepare the clean remediation worktree and baseline evidence. Do not implement any vulnerability fix in this prompt.

    ALLOWED PATHS TO CREATE/MODIFY
    - `docs/remediation/security-2026-06-01/baseline/git-status.txt`
    - `docs/remediation/security-2026-06-01/baseline/head.txt`
    - `docs/remediation/security-2026-06-01/baseline/python-security-tests.txt`
    - `docs/remediation/security-2026-06-01/baseline/middle-tests.txt`
    - `docs/remediation/security-2026-06-01/baseline/frontend-tests.txt`
    - `docs/remediation/security-2026-06-01/baseline/cargo-clippy.txt`
    - `docs/remediation/security-2026-06-01/baseline/cargo-audit.json`
    - `docs/remediation/security-2026-06-01/baseline/npm-audit-root.json`
    - `docs/remediation/security-2026-06-01/baseline/npm-audit-middle.json`
    - `docs/remediation/security-2026-06-01/baseline/npm-audit-server.json`
    - `docs/remediation/security-2026-06-01/baseline/ruff.txt`
    - `docs/remediation/security-2026-06-01/baseline/bandit.json`
    - `docs/remediation/security-2026-06-01/baseline/pip-audit.json`
    - `docs/remediation/security-2026-06-01/register.md`

    COMMANDS
    ```bash
    cd /Users/christienantonio/aurelius
    git status --short --branch
    git rev-parse --short HEAD

    if [ ! -d /Users/christienantonio/aurelius-security-remediation ]; then
      git worktree add ../aurelius-security-remediation 46ee2f13 -b fix/security-remediation-p0
    fi

    cd /Users/christienantonio/aurelius-security-remediation
    git status --short --branch
    mkdir -p docs/remediation/security-2026-06-01/baseline

    git status --short --branch > docs/remediation/security-2026-06-01/baseline/git-status.txt
    git rev-parse HEAD > docs/remediation/security-2026-06-01/baseline/head.txt

    python -m pytest tests/security/test_gateway_perimeter.py tests/gateway/test_t3_gateway_fail_closed.py -q > docs/remediation/security-2026-06-01/baseline/python-security-tests.txt 2>&1 || true
    npm run test -w middle > docs/remediation/security-2026-06-01/baseline/middle-tests.txt 2>&1 || true
    npm run test -w frontend > docs/remediation/security-2026-06-01/baseline/frontend-tests.txt 2>&1 || true
    cargo clippy --workspace --all-targets -- -D warnings > docs/remediation/security-2026-06-01/baseline/cargo-clippy.txt 2>&1 || true
    cargo audit --json > docs/remediation/security-2026-06-01/baseline/cargo-audit.json 2>&1 || true
    npm audit --omit=dev --json > docs/remediation/security-2026-06-01/baseline/npm-audit-root.json 2>&1 || true
    npm audit --omit=dev --json -w middle > docs/remediation/security-2026-06-01/baseline/npm-audit-middle.json 2>&1 || true
    npm audit --omit=dev --json -w server > docs/remediation/security-2026-06-01/baseline/npm-audit-server.json 2>&1 || true
    ruff check . > docs/remediation/security-2026-06-01/baseline/ruff.txt 2>&1 || true
    bandit -q -f json -o docs/remediation/security-2026-06-01/baseline/bandit.json -r gateway agent src tests || true
    pip-audit -f json -o docs/remediation/security-2026-06-01/baseline/pip-audit.json || true
    ```

    REGISTER CONTENT
    If `docs/remediation/security-2026-06-01/register.md` does not exist, create it with the issue rows from the remediation plan. Do not mark any finding closed in preflight.

    FINAL RESPONSE FORMAT
    ```text
    RESULT: PASS | BLOCKED
    Worktree status:
    Baseline files created:
    Commands run:
    Pre-existing blockers:
    Next prompt to run: 01-p0-legacy-license-implementer.md
    ```
