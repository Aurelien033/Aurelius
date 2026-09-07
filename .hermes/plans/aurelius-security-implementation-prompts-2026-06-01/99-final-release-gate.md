# 99 — Final Release Gate Prompt

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
    You are the release gatekeeper. Do not implement. Decide whether the remediation branch is safe to push/open as a PR.

    REQUIRED PRE-PUSH CHECKS
    ```bash
    cd /Users/christienantonio/aurelius-security-remediation
    git status --short --branch
    git diff --name-only HEAD
    git diff --stat HEAD
    git log --oneline origin/main..HEAD || true
    git diff --name-only origin/main..HEAD || true
    grep -E '^(skills/|llm-skills/|projects/)' <(git diff --name-only origin/main..HEAD || true) || true
    ```

    BLOCKING CONDITIONS
    - Dirty worktree with uncommitted source changes.
    - Any changed file outside the intended remediation path set.
    - Any broad accidental path family: `skills/`, `llm-skills/`, `projects/`, unrelated `.github/workflows/`.
    - No register evidence for a closed finding.
    - Any P0/P1 row still open without explicit waiver.
    - Any failing targeted test introduced by the remediation.
    - Any audit high/critical vulnerability not fixed or waived.
    - Any secret-like value in diff.

    OPTIONAL PR CREATE COMMANDS
    Only run these if the release gate says READY and the human explicitly wants a PR opened.

    ```bash
    git push -u origin HEAD
    gh pr create --draft       --title "fix(security): remediate Aurelius gateway and boundary vulnerabilities"       --body-file docs/remediation/security-2026-06-01/pr-body.md
    ```

    RETURN FORMAT
    ```text
    RELEASE_VERDICT: READY_TO_PUSH | REQUEST_CHANGES | BLOCKED
    Dirty state:
    Branch diff summary:
    Forbidden path check:
    P0/P1 register status:
    Security audit status:
    Exact next command for human:
    ```
