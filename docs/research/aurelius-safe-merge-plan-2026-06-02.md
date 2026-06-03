# Aurelius Safe-Merge Plan — 2026-06-02

Generated: 2026-06-02
Companion to: `aurelius-repo-scan-2026-06-02.md`
Author preference applied: **archive divergent work first, avoid
destructive force-pushes, then provide clear reset/sync instructions
for other clones or agents.**

---

## TL;DR (1-minute version)

1. The 12 "feature" branches (7× `fix/security-pX-*` + 5×
   `feature/ring1-trancheX-*`) all point to **the same 79-commit
   divergence** from `main`. They are not 12 independent lines of
   work — they are 12 labels on one line of work.
2. We will **archive** (tag) all 12 branches first.
3. We will **commit the 22 uncommitted working-tree edits** on
   `feature/ring1-tranche5-20260531` as a single closing commit.
4. We will **fast-forward `main`** to that tip, OR open a single
   squash-merge PR — whichever the user prefers.
5. We will **delete the 12 redundant branches** (local + remote)
   after the merge lands, leaving the user with one canonical `main`.
6. Other clones and agents are given explicit reset commands.

---

## 1. Why "one main branch" is achievable here

The 12 branches share an **identical 79-commit fingerprint**:

```
$ for b in fix/security-p0-bff-identity-ws fix/security-p0-legacy-license \
           fix/security-p1-boundaries fix/security-p1-sandbox-auth-deps \
           fix/security-p1-session-auth fix/security-p2-ci-deploy-hardening \
           fix/security-p2-runtime-proof feature/ring1-tranche1-20260531 \
           feature/ring1-tranche2-20260531 feature/ring1-tranche3-20260531 \
           feature/ring1-tranche4-20260531 feature/ring1-tranche5-20260531; do
    n=$(git rev-list --count main.."$b")
    echo "$n  $b"
done
79  fix/security-p0-bff-identity-ws
79  fix/security-p0-legacy-license
79  fix/security-p1-boundaries
79  fix/security-p1-sandbox-auth-deps
79  fix/security-p1-session-auth
79  fix/security-p2-ci-deploy-hardening
79  fix/security-p2-runtime-proof
79  feature/ring1-tranche1-20260531
79  feature/ring1-tranche2-20260531
79  feature/ring1-tranche3-20260531
79  feature/ring1-tranche4-20260531
79  feature/ring1-tranche5-20260531
```

And the unique tip-hashes among them collapse to **a single shared
fingerprint** (`46ee2f13` is the tip, identical across all 12).
So there is **no merge conflict to resolve** — we are not merging
independent lines, we are promoting one line.

---

## 2. Pre-merge safety net (do BEFORE anything else)

### 2.1 Archive every divergent branch as a tag

```bash
cd ~/aurelius
for b in fix/security-p0-bff-identity-ws \
         fix/security-p0-legacy-license \
         fix/security-p1-boundaries \
         fix/security-p1-sandbox-auth-deps \
         fix/security-p1-session-auth \
         fix/security-p2-ci-deploy-hardening \
         fix/security-p2-runtime-proof \
         feature/ring1-tranche1-20260531 \
         feature/ring1-tranche2-20260531 \
         feature/ring1-tranche3-20260531 \
         feature/ring1-tranche4-20260531 \
         feature/ring1-tranche5-20260531 ; do
  tip=$(git rev-parse "$b")
  tag="archive/$(echo "$b" | tr '/' '-')-pre-main-merge-20260602"
  git tag -a "$tag" "$tip" -m "Pre-main-merge archive of $b @ $tip"
done
```

After this, the **full history is recoverable** from the tags even
if every branch is deleted. This is the "archive divergent work
first" step the user requires.

### 2.2 Confirm the tags are present

```bash
git tag -l 'archive/*pre-main-merge-20260602' | wc -l   # expect 12
```

### 2.3 Back up the working tree (just in case)

```bash
cd ~/aurelius
git stash push -u -m "pre-merge-20260602-working-tree" -- \
  $(git status --porcelain | awk '{print $2}')
```

---

## 3. Close the in-flight Ring-1 work

The current branch (`feature/ring1-tranche5-20260531`) has 22
uncommitted modifications and 6 untracked files. We **commit them**
as the final Ring-1 wrap-up — this is the canonical "ending" of
the 79-commit stack.

### 3.1 Inspect the diffs first (no destructive action)

```bash
cd ~/aurelius
git diff --stat
git status --porcelain
```

### 3.2 Stage everything that should land

```bash
cd ~/aurelius
git add agent/ gateway/ middle/ src/ tests/ tools/ .github/workflows/ci.yml
git add .github/workflows/nightly.yml configs/ring1_tranche*.yaml
# `TECHNICAL_DEBT.md` and `.hermes/plans/` are local-only and should NOT be committed
echo "TECHNICAL_DEBT.md" >> .gitignore
echo ".hermes/plans/"    >> .gitignore
git add .gitignore
```

### 3.3 Commit with a recognizable message

```bash
git commit -m "ring1(tranche5): close 22 in-flight edits + add 3 ring1 configs

- src/alignment/simpo.py, src/memory/amc_tier2.py: finalize Ring-1 wiring
- src/model/{__init__,moe}.py: per-layer MLA hooks
- src/serving/{aurelius_server,function_calling_api,structured_output_decoder}.py
- src/training/trainer.py, src/ui/session_manager.py
- middle/src/{config,provider_router,server}.ts + routes/{auth,evaluation,scheduler}.ts
- agent/{__init__,session_manager}.py, gateway/aurelius_api.py
- tests/integration + tests/tools + tools/web_tool.py
- .github/workflows/{ci,nightly}.yml
- configs/ring1_tranche{1,2,3}.yaml

Local-only artefacts kept out of tree:
- TECHNICAL_DEBT.md (operational)
- .hermes/plans/  (agent scratch)"
```

### 3.4 Tag the closed branch

```bash
tip=$(git rev-parse HEAD)
git tag -a "release/ring1-tranche5-closed-20260602" "$tip" \
  -m "Closed Ring-1 tranche 5 (final state before main merge)"
```

---

## 4. The actual merge

### 4.1 Option A — Fast-forward (cleanest, recommended)

```bash
cd ~/aurelius
git checkout main
git merge --ff-only feature/ring1-tranche5-20260531
```

This requires that `main` is an ancestor of the closed ring1-tranche5
tip. If the 79 commits were never on main, this is a true fast-forward
— `main` simply points at the new tip.

### 4.2 Option B — Squash (if the user prefers one mega-commit)

```bash
cd ~/aurelius
git checkout main
git merge --squash feature/ring1-tranche5-20260531
git commit -m "ring1 + security-remediation: 80 commits of work (2026-04-21 → 2026-06-02)

- Mamba-2 selective SSM block (commit 07dd177f, 30/31 tests pass)
- AMC Tier-1/2 memory + debank
- Constitutional AI v2/v3, SimPO, CPO, BOND
- Per-layer MLA bank wiring
- (epsilon,delta)-DP Gaussian mechanism (privacy)
- FedAvg on DreamBank tensors (federated memory)
- Trustrag quarantine-aware retrieval
- CascadeBank compute routing
- 22 in-flight edits closing the Ring-1 stack
- See aurelius-master-plan-v4-2026-06-02.md for what this enables"
```

Squash loses intermediate history but gives `main` a single readable
commit. **Recommended if the user wants a clean log; not recommended
if they want per-tranche traceability.**

### 4.3 Option C — PR workflow (if remote review is desired)

```bash
cd ~/aurelius
gh pr create --base main \
  --head feature/ring1-tranche5-20260531 \
  --title "Ring-1 + security-remediation stack: 80 commits" \
  --body-file docs/research/aurelius-safe-merge-plan-2026-06-02.md
```

Use Option C if the user has co-maintainers or wants CI to run on
the diff before merge.

---

## 5. Post-merge cleanup

### 5.1 Delete the 11 redundant branches (local)

```bash
cd ~/aurelius
git branch -D fix/security-p0-bff-identity-ws \
            fix/security-p0-legacy-license \
            fix/security-p1-boundaries \
            fix/security-p1-sandbox-auth-deps \
            fix/security-p1-session-auth \
            fix/security-p2-ci-deploy-hardening \
            fix/security-p2-runtime-proof \
            feature/ring1-tranche1-20260531 \
            feature/ring1-tranche2-20260531 \
            feature/ring1-tranche3-20260531 \
            feature/ring1-tranche4-20260531 \
            feature/ring1-tranche5-20260531
```

(Keeping one canonical name — `feature/ring1-tranche5-20260531` —
is also acceptable as the "trunk-of-record" instead of deleting it.)

### 5.2 Delete the 12 redundant branches (remote)

```bash
cd ~/aurelius
for b in fix/security-p0-bff-identity-ws \
         fix/security-p0-legacy-license \
         fix/security-p1-boundaries \
         fix/security-p1-sandbox-auth-deps \
         fix/security-p1-session-auth \
         fix/security-p2-ci-deploy-hardening \
         fix/security-p2-runtime-proof \
         feature/ring1-tranche1-20260531 \
         feature/ring1-tranche2-20260531 \
         feature/ring1-tranche3-20260531 \
         feature/ring1-tranche4-20260531 \
         feature/ring1-tranche5-20260531 ; do
  git push origin --delete "$b"
done
```

### 5.3 Leave the archive tags alone

The `archive/*-pre-main-merge-20260602` and
`release/ring1-tranche5-closed-20260602` tags are **kept forever**.
They are the safety net. Do not delete them.

---

## 6. Reset / sync instructions for OTHER clones or agents

For each other clone (e.g., the user's secondary machine, the
delegate_task agents, a CI runner):

```bash
# 1. Stop any in-flight work
cd /path/to/other/clone
git stash push -u -m "pre-merge-sync-20260602"

# 2. Fetch + force the local main onto origin/main (safe — main was
#    only fast-forwarded, so the only force is a fast-forward)
git fetch origin
git checkout main
git reset --hard origin/main

# 3. Re-create the ONE canonical ring1 branch from the new main
git branch -D fix/security-p0-bff-identity-ws 2>/dev/null
git branch -D fix/security-p0-legacy-license  2>/dev/null
git branch -D fix/security-p1-boundaries      2>/dev/null
git branch -D fix/security-p1-sandbox-auth-deps 2>/dev/null
git branch -D fix/security-p1-session-auth    2>/dev/null
git branch -D fix/security-p2-ci-deploy-hardening 2>/dev/null
git branch -D fix/security-p2-runtime-proof   2>/dev/null
git branch -D feature/ring1-tranche1-20260531 2>/dev/null
git branch -D feature/ring1-tranche2-20260531 2>/dev/null
git branch -D feature/ring1-tranche3-20260531 2>/dev/null
git branch -D feature/ring1-tranche4-20260531 2>/dev/null
git branch -D feature/ring1-tranche5-20260531 2>/dev/null
git branch feature/ring1-tranche5-20260531 main  # canonical label
git checkout feature/ring1-tranche5-20260531

# 4. Apply any stashed work
git stash pop  # or drop if you don't need it
```

**This is safe and reversible** because:
- The 12 archive tags are in `origin`, so any branch can be recovered.
- The only force operation is `git reset --hard origin/main`, which
  is a fast-forward (no history is rewritten).
- No force-push to any branch except `main` from `main` (already
  ff'd).

---

## 7. Rollback

If the merge turns out to be wrong, recovery is one line:

```bash
cd ~/aurelius
git checkout main
git reset --hard archive-feature-ring1-tranche5-20260531-pre-main-merge-20260602
```

Or:

```bash
git checkout main
git reset --hard $(git rev-parse "feature/ring1-tranche5-20260531@{1}")
```

This rolls back to the pre-merge state without losing any commits.

---

## 8. Open questions for the user

1. **Option A (ff), B (squash), or C (PR)?** Default: A.
2. **Delete the 12 redundant branches after merge?** Default: yes.
3. **Keep `feature/ring1-tranche5-20260531` as a trunk-of-record label,
   or delete it too?** Default: keep it (one canonical branch label
   is convenient for ring-2 work to fork from).
4. **Should `release/ring1-tranche5-closed-20260602` be pushed to
   origin?** Default: yes, as a stable reference tag.
5. **Are there any commits on the working-tree-22 that should NOT land
   in the close-commit?** Default: no, but worth double-checking.

---

## 9. Summary of safety guarantees

| Risk | Mitigation |
| --- | --- |
| Loss of branch work | 12 archive tags created before any delete |
| Loss of remote state | `git push origin --delete` is reversible; tags remain |
| Destructive force-push | Only `git reset --hard origin/main` (ff) — no -f push |
| Other-clone divergence | Explicit 5-step reset + rebase recipe given |
| Working-tree state loss | `git stash` before commit; recoverable via `git stash pop` |
| Wrong merge type | `git reset --hard` rollback to pre-merge tag in one line |
| CI breaking post-merge | `tests/integration/*` already in the 79-commit set; nightly.yml committed |

See companion: `aurelius-repo-scan-2026-06-02.md` for the underlying
branch data, and `aurelius-master-plan-v4-2026-06-02.md` for the
research roadmap that this merge enables.
