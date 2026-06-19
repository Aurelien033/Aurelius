# Triage: SFT Batch-3 + 4D-Elasticity docs (2026-06-19)
### honest reconciliation against what we've actually PROVEN

> Inputs: `aurelius-sft-training-experiments-batch3-deep-research-2026-06-19.md` (1017 ln) and the 6-part
> `aurelius-4d-elasticity-expansiveness-*` series (~7100 ln). Both are *handoff specs, no code run*. The job here
> isn't to catalogue them — it's to separate what extends our evidence from what re-treads falsified ground.

---

## The reconciliation lens (what we now KNOW, which these docs mostly precede/ignore)

1. **Dynamic layer routing is dead on a frozen base** — 3 powered nulls (E87 per-layer; exact k=2 pair matrix; per-task selector at k=2 *and* k=4). Non-additivity is the *diagnosis*, not an open door.
2. **SFT specializes, doesn't generalize, and can't beat a strong base** (Qwen3-8B: +11.5pp on held-out repair, ≈flat on HumanEval).
3. **The lever that can beat the base is RLVR** (reward = ground-truth correctness). The RL code already exists (`grpo_v3.py`/`rlvr.py`, 69/69 tests).
4. **Base moved to Qwen3-8B**; FROZEN-BASE-v1 (Qwen2.5-1.5B) is a *research* base, not the release base.

Both docs are still largely written for the 1.5B routing world. That's the gap.

---

## A. SFT Batch-3 — high quality, but split it

**Verdict: genuinely well-disciplined** (verifiable tasks, required baselines, falsifiers, the state→action→check→verdict→commit grammar). But its 12 experiments fall into two halves with opposite value:

### ✅ ADOPT — the verifier-aligned subset (extends our verification-native menu)
| ID | what | maps to |
|---|---|---|
| **CFT-SFT** (counterfactual failure-trace) | turn dense-vs-cheap (or fail-vs-pass) deltas into repair supervision | our E-3 (reasoning-trace SFT) + RLVR negative mining |
| **CID-DPO** (cheap-invariant / correctness-per-compute) | preference = *cheapest verifier-passing* answer | a principled upgrade of our "real DPO pairs" (E-3); the `savings = p−c` admission law |
| **VCSS** (verifier-causal step SFT) | checkable causal micro-steps instead of long-CoT mimicry | **E-7** (the existing `cot_verifier.py` / ProcessRewardModel) |
| **the training grammar** | `state→action→check→state→verdict→commit/repair/fallback` as a unifying data schema | the **data format for RLVR + future SFT** — worth standardizing on |

### ⛔ SKIP — the routing subset (re-treads our powered nulls)
- **RCSFT** (route-conditioned SFT), **JRM** (joint route masks), **TLA** (token-level arbitration). The doc frames JRM as "the direct response to E87 — learn joint masks, not per-layer labels." **But we already ran exactly that**: the exhaustive k=2 pair matrix *and* the per-task prompt-conditioned selector at k=2 and k=4 — all powered nulls. JRM/TLA are the same hypothesis at a different granularity, on the same dead frozen-base regime. Don't spend 20–80 GPU-h re-falsifying. (If routing is ever revived, it's via *skip-native pretraining*, not route-conditioned SFT.)
- **Base update:** run the adopted ones on **Qwen3-8B**, not 1.5B; and read "scale route-aware SFT" as **"do RLVR"** — that's the lever the doc's own grammar points at but stops short of.

---

## B. 4D-Elasticity × Expansiveness (6 parts) — speculative reservoir, 2 kernels worth keeping

**Verdict: mostly the scope-dilution failure mode.** VECS / NEVS / elastic-dimensionality / Collapsometer / StabilityShield / MUD-LDP-CCAE-… is a **15+-mechanism framework built on the ACDT SKIP/AMPLIFY routing thesis we've falsified.** The series itself concedes it: "Research-II is a stale fork," "no VECS/4D implementation in the live repo," filed by the canonical plan as a "future mechanism reservoir." Its discipline *framing* is good (falsifiers, budget-matched baselines, isolated packets) — but the content is exactly the 18-mechanism dilution the external reviewer warned against, on a thesis that's empirically closed. **Do not build the framework.**

### Extract these 2 grounded kernels (and drop the rest)
1. **Proof-carrying / certificate-gated training data (Part 5: "no certificate, no training").** Stripped of the 4DEE machinery, this is a *stronger verified-data filter*: a trace is eligible only if it carries provenance + no-reward-hacking + dense-comparison + freshness certificates — not merely "passed a verifier once." This **directly upgrades our trace pipeline** (`make_traces`/`make_code_traces`) and matters a lot for RLVR (reward-hacking guard). Cheap, grounded, adopt the *principle*.
2. **The falsifier/baseline discipline** (must beat a budget-matched baseline; ledger/verifier integrity). Already how we work — good to see it reinforced.

Everything else (elastic dims, expansive substrates, collapse shields, the named-mechanism zoo) = reservoir. Revisit only if a *specific* piece earns a falsifier + a need + a budget, per our promotion gate.

---

## Net actions
- **Fold into the experiment menu:** CFT-SFT, CID-DPO, VCSS under the verification-native banner (they reinforce E-3/E-7/RLVR); adopt the **state→verdict training grammar** as the data schema; adopt **certificate-gated trace eligibility** as a `make_*_traces` upgrade (reward-hacking guard for RLVR).
- **Explicitly skip:** RCSFT/JRM/TLA (re-falsify nulls) and the full 4DEE framework (scope-dilution on a closed thesis).
- **No change to the current plan:** still gated on the v2 HumanEval number → then RLVR. These docs add *data-quality* and *preference-design* refinements to that path, not a new direction.
