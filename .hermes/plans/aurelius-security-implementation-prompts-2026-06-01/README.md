# Aurelius Security Implementation Prompt Pack — 2026-06-01

    This directory contains Cursor/OpenCode-ready prompts for implementing the Aurelius security remediation plan.

    HOW TO RUN THESE PROMPTS

Cursor CLI / Composer 2.5:
```bash
cd /Users/christienantonio/aurelius-security-remediation
cursor composer -f /Users/christienantonio/Desktop/AI\ Plans/aurelius-security-implementation-prompts-2026-06-01/<PROMPT_FILE>.md
```

OpenCode one-shot:
```bash
cd /Users/christienantonio/aurelius-security-remediation
opencode run -f /Users/christienantonio/Desktop/AI\ Plans/aurelius-security-implementation-prompts-2026-06-01/<PROMPT_FILE>.md "Follow the attached prompt exactly."
```

Recommended order:
1. 00-preflight-operator.md
2. 01-p0-legacy-license-implementer.md
3. 01-p0-legacy-license-spec-review.md
4. 01-p0-legacy-license-security-review.md
5. 02-p0-bff-identity-ws-implementer.md
6. 02-p0-bff-identity-ws-spec-review.md
7. 02-p0-bff-identity-ws-security-review.md
8. 03-p1-boundaries-implementer.md
9. 03-p1-boundaries-spec-review.md
10. 03-p1-boundaries-security-review.md
11. 04-p1-session-auth-implementer.md
12. 04-p1-session-auth-spec-review.md
13. 04-p1-session-auth-security-review.md
14. 05-p1-sandbox-auth-deps-implementer.md
15. 05-p1-sandbox-auth-deps-spec-review.md
16. 05-p1-sandbox-auth-deps-security-review.md
17. 06-p2-runtime-proof-implementer.md
18. 06-p2-runtime-proof-spec-review.md
19. 06-p2-runtime-proof-security-review.md
20. 07-p2-ci-deploy-hardening-implementer.md
21. 07-p2-ci-deploy-hardening-spec-review.md
22. 07-p2-ci-deploy-hardening-security-review.md
23. 99-final-integration-review.md
24. 99-final-release-gate.md

    Source plan:
    - /Users/christienantonio/Desktop/AI Plans/aurelius-security-remediation-plan-2026-06-01.md

    Source audit:
    - /Users/christienantonio/Desktop/AI:ML Research/aurelius-security-code-review-2026-06-01.md

    Critical rule:
    - Use the clean worktree `/Users/christienantonio/aurelius-security-remediation`, not the dirty original repo.
    - Never use `git add -A`, `git add .`, or broad globs.
    - Every implementation prompt has a strict path allowlist.
    - Every implementation tranche must pass its spec review before its security review.

    Prompt roles:
    - `*-implementer.md`: writes tests and code.
    - `*-spec-review.md`: checks plan/task compliance and scope drift.
    - `*-security-review.md`: adversarial security/code-quality review.
    - `99-final-*`: final integration and release gates.

    Model note:
    - Generated after user noted active model switch to `minimax-m3-free via OpenCode Zen`.
    - These prompts are intentionally explicit to work with cheaper/faster implementation models.
