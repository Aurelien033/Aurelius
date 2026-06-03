# 07 — P2 CI/Deploy Hardening Implementer

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
    - Branch name: `fix/security-p2-ci-deploy-hardening`
    - Commit message, if you are explicitly asked to commit: `fix(security): harden headers plugins ci and deployment policy`

    MISSION
    Fix remaining P2 hardening items: CORS/CSP/security headers/request-size parity, plugin trust model, lint/security CI gates, and deployment manifest hardening.

This prompt may touch deployment globs listed in the allowlist only if those files exist. Do not create broad generated manifests. Do not rewrite workflows unrelated to security gates.

    STRICT PATH ALLOWLIST
    You may ONLY modify these paths. If any other path is needed, stop and report it.

    - `middle/src/server.ts`
- `middle/src/config.ts`
- `middle/src/middleware/security-headers.ts`
- `middle/src/middleware/validation.ts`
- `middle/src/routes/plugins.ts`
- `middle/__tests__/security_headers.test.ts`
- `middle/__tests__/plugins.test.ts`
- `tests/security/test_deployment_manifests.py`
- `docs/security/plugin-trust-model.md`
- `docs/remediation/security-2026-06-01/lint-baseline.md`
- `pyproject.toml`
- `.github/workflows/ci.yml`
- `.github/workflows/security.yml`
- `deployment/**`
- `docker-compose.yml`
- `docker-compose.dev.yml`
- `Dockerfile`
- `Dockerfile.*`
- `k8s/**`
- `helm/**`
- `docs/remediation/security-2026-06-01/register.md`

    REQUIRED RED TESTS FIRST
    Add or extend tests for these behaviors before implementation. You must run them and verify they fail before changing production code.

    - Add `middle/__tests__/security_headers.test.ts` for CSP, nosniff, referrer policy, frame-ancestors/X-Frame-Options, CORS production wildcard rejection, and request-size caps.
- Extend `middle/__tests__/plugins.test.ts` proving plugin metadata schema rejects arbitrary entrypoint paths/URLs and dynamic loading is disabled by default.
- Add `tests/security/test_deployment_manifests.py` checking no prod wildcard CORS, no hardcoded secrets, no legacy server prod service, non-root/read-only/drop-capabilities/resource limits where applicable.
- CI workflow changes must run targeted security tests, npm audit, pip-audit, cargo audit, ruff, and bandit with baseline policy.

    IMPLEMENTATION REQUIREMENTS
    - Implement the minimum safe change that satisfies the tests.
    - Prefer central helpers for auth, URL policy, redaction, and validation instead of repeated ad hoc checks.
    - Do not weaken an existing test to make it pass.
    - Do not skip tests because unrelated baseline failures exist; run the targeted tests and document any unrelated baseline failures separately.
    - Do not add broad compatibility fallbacks that silently accept insecure legacy behavior.

    ACCEPTANCE CRITERIA
    - Production wildcard CORS is impossible through config validation.
- BFF emits security headers and route-specific request size limits.
- Plugin trust model is documented and dynamic loading disabled until signed manifests/capabilities/sandbox exist.
- CI/security workflow has audit gates without masking failures via unconditional `|| true` except documented nonblocking baseline collection.
- Deployment manifest test passes or clearly skips non-existent manifest families.
- Register P2 rows are updated with evidence.

    VALIDATION COMMANDS
    Run these commands from `/Users/christienantonio/aurelius-security-remediation` unless a command says otherwise.

    ```bash
    git status --short --branch
npm run test -w middle -- security_headers validation plugins
npm run build -w middle
python -m pytest tests/security/test_deployment_manifests.py -q
ruff check . || true
bandit -q -r gateway agent src -x tests || true
npm audit --omit=dev || true
npm audit --omit=dev -w middle || true
npm audit --omit=dev -w frontend || true
pip-audit || true
cargo audit
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
