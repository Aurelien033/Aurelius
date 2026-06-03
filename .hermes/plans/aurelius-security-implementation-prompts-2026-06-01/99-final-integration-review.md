# 99 — Final Integration Review Prompt

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

    ROLE
    You are the final integration reviewer. Do not implement. Verify that all remediation tranches compose into one coherent security posture and that no earlier fix was weakened by a later tranche.

    REVIEW INPUTS
    - Audit report: /Users/christienantonio/Desktop/AI:ML Research/aurelius-security-code-review-2026-06-01.md
    - Remediation plan: /Users/christienantonio/Desktop/AI Plans/aurelius-security-remediation-plan-2026-06-01.md
    - Register: `docs/remediation/security-2026-06-01/register.md`
    - Current worktree: /Users/christienantonio/aurelius-security-remediation

    REQUIRED COMMANDS
    ```bash
    cd /Users/christienantonio/aurelius-security-remediation
    git status --short --branch
    git diff --name-only HEAD
    git log --oneline --decorate -20
    python -m pytest tests/security tests/gateway/test_t3_gateway_fail_closed.py tests/tools/test_code_runner_exploits.py -q
    npm run test -w middle
    npm run test -w frontend
    npm run build -w middle
    npm run build -w frontend
    cargo clippy --workspace --all-targets -- -D warnings
    cargo audit
    npm audit --omit=dev || true
    npm audit --omit=dev -w middle || true
    npm audit --omit=dev -w frontend || true
    pip-audit || true
    ruff check . || true
    bandit -q -r gateway agent src -x tests || true
    ```

    INTEGRATION CHECKLIST
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
    - P2 hardening has CI/deployment/test evidence.

    RETURN FORMAT
    ```text
    INTEGRATION_VERDICT: READY_FOR_PR | REQUEST_CHANGES | BLOCKED
    Findings closed:
    - ...
    Commands run:
    - ...
    Failed commands / baseline issues:
    - ...
    Cross-tranche regressions:
    - ...
    Release blockers:
    - ...
    Recommended PR body:
    <draft body>
    ```
