# AMC Buildout — Model Assignment Per Tranche

Updated: 2026-05-23
Scope: 40 tranches (T00-T35 + S01-S05) across 6 phases

Legend
------
🟢  Composer 2.5 (Sonnet, $20/mo flat) — cheap, adequate, you review at commit time
🟡  Composer 2.5 + SPECIFIC INVARIANT pasted at end of prompt
🟡🔶 Composer 2.5 + invariant + extra line-by-line math review
🟠  Claude Opus 4.7 direct (Anthropic API — write, must review before commit)
🔴  YOU write — model can help with formatting/LaTeX only, you are the author
⚡  Hermes (me) — I write from spec, no back-and-forth, you review once

---

## Phase 0 — Foundation  (weeks 1-2)

| # | Tranche | Model | Why |
|---|---------|-------|-----|
| T00 | Mamba-2 SSM block | ✅ ALREADY DONE (Hermes) | Math correctness is non-negotiable — I wrote it |
| T01 | AMCSSMLayer | 🟡 Composer + T01 invariant | State shape trap; Composer gets it right with the 3-liner |
| T02 | Promotion gate (Gumbel-straight-through) | 🟡 Composer + T02 invariant | Misses straight-through ~40% of time without the guard |
| T03 | SQLite SDB persistent event log | 🟢 Composer | Pure mechanical CRUD + opstore pattern; low risk |

---

## Phase 1 — Model Layer  (weeks 3-6) ← the paper

| # | Tranche | Model | Why |
|---|---------|-------|-----|
| T04 | AMCTransformer (MLA+SSM hybrid) | 🟡🔶 Composer + T04 invariant + attention/SSM alternation review | RoPE on SSM is the silent killer; review the alternation list explicitly |
| T05 | SurpriseHead + pretrainer | 🟡🔶 Composer + T05 invariant | The `.detach()` trap is invisible until T22 produces garbage; check one line |
| T06 | AMCGateController (decay/erase/write) | 🟡 Composer + T06 invariant | Sigmoid output range; easy to verify in diff |
| T07 | DeepSeek-MLA KV attention | 🟡 Composer + T07 invariant | position_offset direction; review two lines |
| T08 | RMSNorm + RoPE (AMC-aware) | 🟢 Composer | Mechanical wrapper; low novelty risk |
| T09 | Parameter counter + config validator | 🟢 Composer | Pure arithmetic + file IO |
| T10 | Full model smoke test | 🟢 Composer | Straightforward round-trip test |

---

## Phase 2 — Durable Runtime  (weeks 5-8, parallel with Phase 1)

| # | Tranche | Model | Why |
|---|---------|-------|-----|
| T11 | Tier-2/3 checkpoint (msgpack) | 🟢 Composer | Serialization; mechanical |
| T12 | State reconstruction engine | 🟡 Composer + verify replay matches original | Tests must replay events and assert bit-identical output; verify the assert is there |
| T13 | Trust-aware KV cache | 🟡 Composer + T13 invariant | Cache key = hash(content ‖ tier ‖ trust ‖ epoch); verify all 4 fields are in the hash |
| T14 | Crash recovery (WAL) | 🟢 Composer | WAL replay is mechanical |

---

## Phase 3 — Training Pipeline  (weeks 9-14)

| # | Tranche | Model | Why |
|---|---------|-------|-----|
| T15 | Three AMC losses | 🟡🔶 Composer + T15 invariant + sign verification | surprise/consistency/promotion signs matter; verify formulas in Part 3 doc match diff |
| T16 | Importance-annotated data pipeline | 🟢 Composer | Tokenization + packing; mechanical file IO |
| T17 | AMCTrainer + 3 optimizers | 🟡 Composer + T17 invariant | Single optimizer with single LR breaks the whole training dynamic; verify 3-group structure |
| T18 | Forge-1B YAML config | 🟢 Composer | YAML; mechanical |
| T19 | Data prep scripts | 🟢 Composer | Mechanical |
| T20 | Training launcher + DeepSpeed | 🟡 Composer | DeepSpeed API traps; verify the ZeRO-2 json matches S01 |
| T21 | Monitor script | 🟢 Composer | Mechanical |
| T22 | Eval (AMC-Memory, GSM8K, MMLU) | 🟢 Composer | eval harness calls existing runners; mechanical |

---

## Phase 4 — Agent Deepening  (weeks 13-18, parallel with Phase 3)

| # | Tranche | Model | Why |
|---|---------|-------|-----|
| T23 | ConstitutionalMemory | ⚡ Hermes write, Composer review | I already wrote the full implementation in Part 4; let Composer generate the tests from the spec |
| T24 | Reflect-and-consolidate agent step | 🟢 Composer | Orchestration logic; relatively mechanical |
| T25 | Skill crystallizer | 🟢 Composer | Merge/summarize skill blobs; mechanical |
| T26 | SLR end-to-end integration | 🟡 Composer | Wiring the layers together; one subtlety: verify AMCPrefixCompiler is called, not manually constructed |
| T27 | Full agent e2e integration test | 🟡 Composer + verify it actually calls the full stack | Many e2e tests assert without exercising the path; check each test calls the full pipeline |

---

## Phase 5 — Validation  (weeks 19-24)

| # | Tranche | Model | Why |
|---|---------|-------|-----|
| T28 | Ablation runner (4×5 benchmarks) | 🔴 YOU (paste the bootstrap formula verbatim) | The significance test is a one-line mistake that invalidates every p-value in the paper |
| T29 | Adversarial audit (6 probes) | 🔴 YOU (verify each probe body manually) | All models write fake probes; this is the only reliable check |
| T30 | Reproducibility bundle | 🟢 Composer | README, flow doc, environment.yml — mechanical |
| T31 | External cross-validation | 🟢 Composer + you run the commands | Scripted; just needs to run clean on a fresh machine |

---

## Phase 6 — Paper  (weeks 25-30)

| # | Tranche | Model | Why |
|---|---------|-------|-----|
| T32 | Paper outline + abstract (LaTeX) | 🔴 YOU write abstract and claims; Composer LaTeX-polish | The framing is 100% you; Composer is a typesetter |
| T33 | Method section (math prose) | 🔴 YOU write claims; Composer formats equations | You decide what's novel; Composer TeX-ifies it |
| T34 | Experiments (tables + figures) | 🟠 Opus 4.7 for LaTeX tables; YOU choose which results | Table formatting is mechanical, result selection is not |
| T35 | Final assembly, arXiv, HF release | 🟢 Composer + you approve | Shell scripts, upload scripts, HF model card — mechanical |

---

## Support Infrastructure

| # | Tranche | Model | Why |
|---|---------|-------|-----|
| S01 | DeepSpeed ZeRO-2 config | 🟢 Composer | Straight JSON |
| S02 | `scripts/train.sh` launcher | 🟢 Composer | Shell boilerplate |
| S03 | Hugging Face model card | 🟢 Composer | Markdown template |
| S04 | `scripts/monitor_training.py` | 🟢 Composer | Matplotlib plots; mechanical |
| S05 | TRANCHE_STATUS.md | ✅ Already done (Hermes) | Docs |

---

## One-Page Summary

| Your model combo | Covers | Remaining you-do |
|---|---|---|
| Composer 2.5 ($20/mo) + invariants | 28/40 tranches | Review diffs, verify 3 trap lines per yellow tranche |
| Opus 4.7 direct (~$30) | 5 extra tranches (T28, T29, T32-T34 quality boost) | Same review burden, slightly lower |
| Hermes (me) | 2 tranches (T00 ✅, T23 ready-to-write) | Zero — I write the code, you verify and commit |
| YOU directly | T28, T29, T32 abstract + claims | No model replaces your judgment here |

**Total model cost for remaining work: ~$50-80** (Composer current month + Opus API for red tranches).
GPU training cost: ~$200 (separate, Lambda/RunPod).
Total: ~$250-280. Within your budget.

---

## Quick Reference: The Eight Invariants

Copy-paste these into Composer for each yellow tranche.

**T01 — state shape:**
```
# state: (B, nheads, headdim, d_state) — do NOT pool/reshape.
# forward output: (B, L, d_model), not aggregated.
```

**T02 — promotion gate:**
```
# GumbelSoftmax straight-through: hard - soft.detach() + soft
```

**T04 — attention/SSM alternation:**
```
# Attention at EVEN indices (0,2,4,...), SSM at ODD (1,3,5,...).
# RoPE only on attention layers. Never on SSM hidden states.
```

**T05 — surprise head:**
```
# surprise = self.head(x.detach()) — NOT torch.no_grad()
```

**T06 — gate activation:**
```
# All gates (decay/erase/write) end with nn.Sigmoid().
```

**T07 — MLA continuation:**
```
# position_offset is ADDED to base_position, never subtracted.
```

**T15 — loss signs:**
```
# surprise: BCE on surprise_scores vs importance_labels.
# consistency: 1 - cosine_similarity.
# promotion: REINFORCE — log_prob * reward with baseline subtraction.
```

**T17 — 3 optimizers:**
```
# Three ADAMW groups: main=3e-4, gate=1e-4, surprise=1e-5.
# Each has its own linear-warmup-then-cosine scheduler.
```

---

*See `PRE_IMPLEMENTATION_BRIEFING.md` for environment check, phase-boundary gates, git discipline, and cost reality.*
