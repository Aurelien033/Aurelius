# 06 — P2 Runtime Proof Implementer

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
    - Branch name: `fix/security-p2-runtime-proof`
    - Commit message, if you are explicitly asked to commit: `fix(security): add runtime safety proofs and redacted logging`

    MISSION
    Fix key P2 runtime-proof findings M1, M2, and M10.

M1: Rust unsafe serialization must have SAFETY comments, malformed checkpoint tests, and optional fuzz/Miri hooks.
M2: unsafe PyTorch checkpoint loading must use safetensors or `weights_only=True` helper with explicit trusted-pickle opt-in.
M10: logs/activity must be structured and redacted; untrusted text cannot forge log lines or inject control sequences.

    STRICT PATH ALLOWLIST
    You may ONLY modify these paths. If any other path is needed, stop and report it.

    - `rust_memory/src/lib.rs`
- `rust_memory/src/checkpoint.rs`
- `rust_memory/Cargo.toml`
- `rust_memory/tests/checkpoint_malformed.rs`
- `rust_memory/tests/unsafe_invariants.rs`
- `rust_memory/fuzz/Cargo.toml`
- `rust_memory/fuzz/fuzz_targets/checkpoint.rs`
- `src/training/amc_trainer.py`
- `tests/training/test_amc_trainer.py`
- `middle/src/security/redaction.ts`
- `gateway/redaction.py`
- `middle/src/server.ts`
- `middle/src/routes/files.ts`
- `middle/src/routes/rag.ts`
- `middle/src/routes/scheduler.ts`
- `middle/__tests__/redaction.test.ts`
- `middle/__tests__/logs.test.ts`
- `middle/__tests__/files.test.ts`
- `middle/__tests__/rag.test.ts`
- `middle/__tests__/command.test.ts`
- `tests/security/test_log_redaction.py`
- `docs/remediation/security-2026-06-01/register.md`

    REQUIRED RED TESTS FIRST
    Add or extend tests for these behaviors before implementation. You must run them and verify they fail before changing production code.

    - Add Rust malformed checkpoint tests for truncated payload, oversized length, bad alignment/shape if applicable, and endian/header mismatch.
- Add/extend training test proving checkpoint load uses safe helper and rejects untrusted pickle unless explicit opt-in.
- Add middle/Python redaction tests for Authorization headers, bearer tokens, query secrets, cookies, CR/LF, ANSI escapes, `<script>`, and raw filename/control char cases.

    IMPLEMENTATION REQUIREMENTS
    - Implement the minimum safe change that satisfies the tests.
    - Prefer central helpers for auth, URL policy, redaction, and validation instead of repeated ad hoc checks.
    - Do not weaken an existing test to make it pass.
    - Do not skip tests because unrelated baseline failures exist; run the targeted tests and document any unrelated baseline failures separately.
    - Do not add broad compatibility fallbacks that silently accept insecure legacy behavior.

    ACCEPTANCE CRITERIA
    - Every Rust unsafe block touched has a nearby `SAFETY:` comment documenting invariants.
- Malformed checkpoint tests pass.
- Direct unsafe `torch.load` is replaced with a safe helper or explicit trusted opt-in path.
- Logs use structured fields and redaction helpers for untrusted/sensitive fields.
- Register rows SEC-P2 runtime-proof items are updated with evidence.

    VALIDATION COMMANDS
    Run these commands from `/Users/christienantonio/aurelius-security-remediation` unless a command says otherwise.

    ```bash
    git status --short --branch
cargo test -p rust_memory || cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
python -m pytest tests/training/test_amc_trainer.py tests/security/test_log_redaction.py -q
bandit -q -r src/training gateway -x tests || true
npm run test -w middle -- redaction logs files rag command
npm run build -w middle
git grep -n "unsafe" rust_memory/src || true
git grep -n "torch.load" src/training tests/training || true
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
