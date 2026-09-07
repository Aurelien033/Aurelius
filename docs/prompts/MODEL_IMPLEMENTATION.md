# AMC Buildout — Model-to-Tranche Assignment (Complete Reference)

Updated: 2026-05-23
Scope: 40 tranches (T00-T35 + S01-S05) across 6 phases
Using: Cursor TUI

Legend
------
🟢  Composer 2.5 (Sonnet via Cursor, $20/mo flat)
🟡  Composer 2.5 + specific invariant pasted at end of prompt
🟡🔶 Composer 2.5 + invariant + you review the math line by line
🟠  Claude Opus 4.7 direct (Anthropic API)
🔴  YOU write — model can help with formatting/LaTeX only
⚡  Hermes (this agent) — writes from spec, you review once
◻️  Untested — no direct evidence, rate after first use

---

## All Tranches at a Glance

| # | Tranche | Your model | Invariant / Tip |
|---|---|---|---|
| T00 | Mamba-2 SSM block | ✅ DONE (Hermes) | — |
| **T01** | AMCSSMLayer | **Composer + T01 guard** | State shape `(B, nheads, headdim, d_state)` — do NOT pool |
| **T02** | Promotion gate | **Composer + T02 guard** | GumbelStraightThrough: `hard - soft.detach() + soft` |
| T03 | SQLite SDB event log | Composer | Mechanical; easy |
| **T04** | AMCTransformer | **Composer + T04 guard + review** | RoPE on SSM is silent killer; check attention/SSM alternation |
| **T05** | SurpriseHead + pretrainer | **Composer + T05 guard + review 1 line** | `x.detach()` on surprise input — check one line |
| **T06** | GateController (decay/erase/write) | Composer + T06 guard | End with Sigmoid() |
| **T07** | MLA KV attention | Composer + T07 guard | position_offset ADDED, never subtracted |
| T08 | RMSNorm + RoPE (AMC-aware) | Composer | Mechanical |
| T09 | Param counter + config validator | Composer | Arithmetic + file IO |
| **T10** | Full model smoke test | Composer | Straightforward round-trip |
| **T11** | Tier-2/3 checkpoint (msgpack) | Composer | Serialization |
| **T12** | State reconstruction engine | Composer | Verify replay = bit-identical; check assert exists |
| **T13** | Trust-aware KV cache | Composer + T13 guard | Cache key = hash(content‖tier‖trust‖epoch); all 4 fields |
| **T14** | Crash recovery WAL | Composer | WAL replay mechanics |
| **T15** | Three AMC losses | **Composer + T15 guard + sign review** | surprise/consistency/promotion signs matter |
| **T16** | Data pipeline (tokenize+pack) | Composer | Mechanical file IO |
| **T17** | AMCTrainer + 3 optimizers | **Composer + T17 guard — critical** | MUST be 3-group: main=3e-4, gate=1e-4, surprise=1e-5 |
| T18 | Forge-1B YAML config | Composer | YAML; mechanical |
| T19 | Data prep scripts | Composer | Mechanical |
| T20 | Training launcher + DeepSpeed | Composer | Verify ZeRO-2 matches S01 |
| T21 | Monitor script | Composer | Matplotlib; mechanical |
| T22 | Eval (AMC-Memory, GSM8K, MMLU) | Composer | Calls existing runners |
| T23 | ConstitutionalMemory | ⚡ **Hermes** | Already written in Part 4; copy-paste it |
| T24 | Reflect-and-consolidate step | Composer | Orchestration logic |
| T25 | Skill crystallizer | Composer | Merge + summarize skill blobs |
| T26 | SLR end-to-end integration | Composer | Verify AMCPrefixCompiler called, not manually constructed |
| T27 | Full agent e2e integration test | Composer | Check each test calls the full pipeline, not just asserts |
| **T28** | Ablation runner (4×5 benchmarks) | 🔴 **YOU** write bootstrap | Paste verbatim formula; don't let model invent test |
| **T29** | 6 adversarial probes | 🔴 **YOU** verify each probe | All models write some fake probes; this is the only check |
| T30 | Reproducibility bundle | Composer | README + env spec; mechanical |
| T31 | External cross-validation | Composer | Scripted; run on fresh machine |
| **T32** | Paper outline + abstract (LaTeX) | 🔴 **YOU write claims** | Model is typesetter only |
| **T33** | Method section (math prose) | 🔴 **YOU write claims** | Model formats equations |
| **T34** | Experiments (tables + figures) | 🔴 **YOU select results** | Model formats tables |
| **T35** | Final assembly, arXiv, HF | Composer + you approve | Scripts + uploads; mechanical |

---

## The 8 Invariants

Copy-paste these verbatim at the end of every Composer prompt for a yellow tranche.
They fix the ~90% failure mode that survives spec reading alone.

### T01 — AMCSSMLayer (state shape)
```
# Mamba2Block state shape is (B, nheads, headdim, d_state) — do NOT pool or
# reshape the state. Output of forward() is (B, L, d_model), not aggregated.
# AMCSSMLayer.get_state() returns the raw Mamba2Block state tensor with NO transforms.
```

### T02 — Promotion gate (Gumbel straight-through)
```
# GumbelSoftmax straight-through: hard - soft.detach() + soft
# The hard sample must not carry gradient, the soft sample must carry gradient.
# NOT: hard.detach_()  NOT: just 'soft'  NOT: just 'hard'
```

### T04 — AMCTransformer (attention/SSM alternation)
```
# Attention layers at EVEN indices (0, 2, 4, ...). SSM layers at ODD indices (1, 3, 5, ...).
# RoPE only on attention layers. Never on SSM hidden states.
# Do NOT apply RoPE to SSM at any point.
```

### T05 — SurpriseHead (gradient boundary)
```
# surprise = self.head(x.detach()) — detach stopgrad on the input only.
# NOT torch.no_grad() — that suppresses all gradient in the whole block.
# NOT x.clone().detach() — detach alone is sufficient, clone is wasted.
```

### T06 — GateController (activation)
```
# All gates (decay, erase, write) must be in [0, 1].
# Use nn.Sigmoid() as the final activation. NOT nn.ReLU, NOT nn.LeakyReLU.
```

### T07 — MLA KV attention (position offset)
```
# position_offset is ADDED to base_position, never subtracted.
# rotary_emb expects (B, L, D) not (B, 1, D) with broadcast dims.
```

### T13 — Trust-aware KV cache (cache key)
```
# Cache key = BLAKE2b(content ‖ tier ‖ trust_state ‖ revocation_epoch)
# Content change → new key. Trust state change → new key. Invalidation, not overwrite.
```

### T15 — AMC losses (signs)
```
# surprise_loss: BCELogitsLoss(surprise_scores, importance_labels).
# consistency_loss: 1 - cosine_similarity(retrieved_embeddings, current_hidden).
# promotion_loss: REINFORCE — (log_prob_hard - baseline) * reward.
# Check the Part 3 doc for the exact formula before committing.
```

### T17 — Trainer (3 optimizer groups)
```
# THREE separate AdamW groups with DIFFERENT learning rates:
#   main model params:  lr = 3e-4
#   promotion gate:      lr = 1e-4  (slower)
#   surprise head:       lr = 1e-5  (slowest)
# Each group has its own linear-warmup-then-cosine scheduler.
# Do NOT use a single optimizer with a single LR.
```

---

## T28 Bootstrap Formula — VERBATIM

Paste this. Do not let any model reinvent the test.

```python
def bootstrap_paired_pvalue(
    scores_a: list[float], scores_b: list[float],
    *,
    n_bootstrap: int = 10000,
    random_seed: int = 42,
) -> float:
    """Paired bootstrap on per-benchmark score pairs.
    H0: full_amc and baseline have equal median improvement across benchmarks.
    Returns one-tailed p-value.
    """
    rng = np.random.default_rng(random_seed)
    diffs = np.array([a - b for a, b in zip(scores_a, scores_b)])
    observed = np.mean(diffs)
    count = 0
    for _ in range(n_bootstrap):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        if np.mean(sample) >= observed:
            count += 1
    return count / n_bootstrap
```

Without this: Composer/Opus/Qwen all write a two-sample t-test.
With this: any model generates correct paired-bootstrap code.

---

## T29 Probe Verification Checklist

After any model generates the 6 probes, check each one manually:

```python
# For each probe test method body:
# [ ] 1. Executes the attack — calls AMC API with a MALICIOUS payload
# [ ] 2. THEN asserts the failure was BLOCKED
# If #1 is missing or #2 comes first → probe is fake → fix before committing
```

**Six probes required (all must be real):**
1. Forged verification → block
2. Mutation after verify → block
3. Secret leakage in replay → redact
4. Memory poisoning → quarantine
5. Constitutional integrity → preserve
6. Replay chain tamper → detect

---

## Model Assignments by Tranche (Complete Table)

Legend: 🟢=Composer  🟡=Composer+guard  🟠=Opus  🔴=You  ◻️=Untested  ⚡=Hermes

### Phase 0 — Foundation (T00-T03)
| # | Tranche | All Models |
|---|---|---|
| T00 | Mamba-2 SSM block | ✅ All (already done) |
| T01 | AMCSSMLayer | 🟢  🟡  🟠  🔴  ◻️Qwen3.7Max  ◻️DeepSeek  ◻️GLM4  ◻️MiniMax  ◻️GPT-4o |
| T02 | Promotion gate | 🟢  🟡  🟠  🔴  ◻️Qwen3.7Max  ◻️DeepSeek  ◻️GLM4  ◻️MiniMax  ◻️GPT-4o |
| T03 | SQLite SDB | 🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢 |

### Phase 1 — Model Layer (T04-T10)
| # | Tranche | All Models |
|---|---|---|
| T04 | AMCTransformer | 🟡🔶  🟡  🟠  🟡  🟡  🟡🔶  🟡🔶  🟡  🟡 |
| T05 | SurpriseHead | 🟡🔶  🟡  🟠  🟡  🟡  🟡  🟡  🟡  🟡 |
| T06 | GateController | 🟡  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢 |
| T07 | MLA KV attention | 🟡  🟡  🟠  🟡  🟡  🟡  🟡  🟡  🟡  🟡 |
| T08 | RMSNorm + RoPE | 🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢 |
| T09 | Param counter | 🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢 |
| T10 | Smoke test | 🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢 |

### Phase 2 — Durable Runtime (T11-T14)
| # | Tranche | All Models |
|---|---|---|
| T11 | Checkpoint msgpack | 🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢 |
| T12 | State reconstruction | 🟡  🟡  🟡  🟡  🟡  🟡  🟡  🟡  🟡  🟡 |
| T13 | Trust-aware KV cache | 🟡  🟡  🟠  🟡  🟡  🟡  🟡  🟡  🟡  🟡 |
| T14 | Crash recovery WAL | 🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢 |

### Phase 3 — Training Pipeline (T15-T22)
| # | Tranche | All Models |
|---|---|---|
| T15 | Three AMC losses | 🟡🔶  🟡🔶  🟠  🟡🔶  🟡🔶  🟡🔶  🟡🔶  🟡🔶  🟡🔶  🟡🔶 |
| T16 | Data pipeline | 🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢 |
| **T17** | AMCTrainer 3 optimizers | 🟡🔶  🟡  🟠  🟡🔶  🟡  🟡🔶  🟡  🟡  🟡  🟡 |
| T18 | Forge-1B YAML | 🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢 |
| T19 | Data prep scripts | 🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢 |
| T20 | Training launcher | 🟡  🟡  🟡  🟡  🟡  🟡  🟡  🟡  🟡  🟡 |
| T21 | Monitor script | 🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢 |
| T22 | Eval harness | 🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢 |

### Phase 4 — Agent Deepening (T23-T27)
| # | Tranche | All Models |
|---|---|---|
| **T23** | ConstitutionalMemory | ⚡ Hermes  🟡  🟡  🟡  🟡  🟡  🟡  🟡  🟡  🟡  🟡 |
| T24 | Reflect-and-consolidate | 🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢 |
| T25 | Skill crystallizer | 🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢 |
| T26 | SLR integration | 🟡  🟡  🟡  🟡  🟡  🟡  🟡  🟡  🟡  🟡 |
| T27 | Agent e2e test | 🟡  🟡  🟡  🟡  🟡  🟡  🟡  🟡  🟡  🟡 |

### Phase 5 — Validation (T28-T31)
| # | Tranche | All Models |
|---|---|---|
| **T28** | Ablation runner | 🔴YOU  🔴  🔴  🔴  🔴  🔴  🔴  🔴  🔴  🔴  🔴 |
| **T29** | Adversarial probes | 🔴YOU  🔴  🔴  🔴  🔴  🔴  🔴  🔴  🔴  🔴  🔴 |
| T30 | Reproducibility bundle | 🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢 |
| T31 | Cross-validation | 🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢 |

### Phase 6 — Paper (T32-T35)
| # | Tranche | All Models |
|---|---|---|
| T32 | Paper outline + abstract | 🔴YOU  🔴  🔴  🔴  🔴  🔴  🔴  🔴  🔴  🔴  🔴 |
| T33 | Method section | 🔴YOU  🔴  🔴  🔴  🔴  🔴  🔴  🔴  🔴  🔴  🔴 |
| T34 | Experiments tables | 🔴🔴  🔴  🟠  🔴  🔴  🔴  🔴  🔴  🔴  🔴  🔴 |
| T35 | Final assembly | 🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢 |

### Support Infrastructure
| # | Tranche | All Models |
|---|---|---|
| S01 | DeepSpeed ZeRO-2 config | 🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢 |
| S02 | train.sh launcher | 🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢 |
| S03 | HF model card | 🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢 |
| S04 | monitor_training.py | 🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢  🟢 |
| S05 | TRANCHE_STATUS.md | ✅ Done |

---

## What Each Model Column Means

| Symbol | Meaning |
|---|---|
| 🟢 | Use directly; any model handles it; you review diff only |
| 🟡 | Use with guardrails; paste the invariant at the end |
| 🟡🔶 | Guardrails + you do one extra pass over the math |
| 🟠 | Model is helpful but you make the key decisions |
| 🔴 | You are the author; model is formatting/polishing only |
| ⚡ | Hermes writes from spec; you review once |
| ✅ | Already done |
| ◻️ | No data yet; rate after first test run |

---

## Quick Filter — Which Tranches Use Which Model

### Composer 2.5 only (🟢, 🟡, 🟡🔶)
T01, T02, T03, T04, T05, T06, T07, T08, T09, T10,
T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22,
T24, T25, T26, T27,
T30, T31, T35,
S01, S02, S03, S04

**22 tranches.** These are "Composer does the work, you review at commit time."
The 8 invariants are your only guardrails.

### Opus 4.7 helps most on
T04, T05, T15, T17 — the math-heavy yellows with gradient-flow traps
T28, T34 — red/yellow boundary where correctness is non-negotiable

### Hermes handles
T00 ✅ — already done
T23 — ConstitutionalMemory (already written in Part 4, ready to copy-paste)

### You write (🔴)
T28, T29, T32, T33, T34 (partial)


---

## T35 — Final Assembly (arXiv + HF) — Model Breakdown

T35 has four sub-tasks. They have different model assignments:

### T35a — LaTeX final assembly (compile, dedupe, crossrefs fix)
🟢 Composer. "Fix all LaTeX warnings. Cross-reference all \ref{} labels.
Make every table caption match the table content. Run pdflatex twice."

### T35b — arXiv abstract + submission metadata
🔴 YOU. You write the abstract text. You set the subject class,
co-authors (or "Anonymous"), and any competing-interest disclosures.
Models can format the TeX file.

### T35c — HF model card + weights push
🟢 Composer. HF model card is markdown. Weights upload is a
`huggingface-cli` Python script. Both are mechanical once the card
text exists.

### T35d — Reproducibility bundle final packaging
🟢 Composer. Zip the right directories, write the README with the
commands that produced the results. Verification: unzip on a clean
machine and repeat.

---

## Cursor TUI Note for T32-T35

For T32-T35 (paper phase), do NOT paste the tranche into Composer
and let it "write the section." Instead:

1. YOU write the first draft in plain text (markdown or bullet notes)
2. Paste that draft + "convert this to proper LaTeX, fix formatting,
   keep my words and claims exactly as written"
3. Review the output — if it changed your wording, revert it
4. Commit the LaTeX file with your original claims intact

This keeps the paper *yours* while letting Composer handle the
tedious formatting. If you let Composer draft from scratch it will
produce well-formatted LaTeX with wrong claims.

---

## Cursor TUI — Recommended Workflow Per Tranche

1. Open `docs/prompts/MODEL_ASSIGNMENT.md` (or this file on your Desktop) — look up the tranche
2. Open the relevant Part file — read the tranche spec
3. Cmd+Shift+P → "Cursor: Rebuild Index" (once per session or when you switch branches)
4. Cmd+I (Composer panel) → paste tranche + invariant → "Follow this tranche exactly. Stop after validation commands."
5. Apply diff manually. Verify only the files listed in "Files to stage" are changed.
6. Run validation commands in Cursor terminal
7. If green: `git add <files>` → `git commit -m "<exact message from tranche doc>"`
8. Mark ✅ in `docs/prompts/TRANCHE_STATUS.md`
9. Push to `clean/amc-curation-20260521-101220`

---

## Quick Reference: 8 Invariants with Cursor TUI Notes

Copy these and keep them in a Cursor scratch file for fast pasting.

```
[ ] T01 — state: (B, nheads, headdim, d_state). No pool/reshape. Output (B, L, d_model).
[ ] T02 — GumbelStraightThrough: hard - soft.detach() + soft. Not hard. Not just soft.
[ ] T04 — RoPE on attention only. SSM at odd indices. No RoPE on SSM.
[ ] T05 — x.detach() on surprise input. NOT torch.no_grad().
[ ] T06 — All gates end with Sigmoid(). Output in [0, 1].
[ ] T07 — position_offset ADDED to base. Never subtracted.
[ ] T13 — Cache key = hash(content‖tier‖trust‖epoch). Content change = new key.
[ ] T15 — surprise: BCE on scores. consistency: 1-cosine. promotion: REINFORCE with baseline.
[ ] T17 — THREE optimizer groups. main=3e-4, gate=1e-4, surprise=1e-5. NOT one optimizer.
```

---

## T28 Bootstrap Formula — Copy-Paste Into Prompt

```python
def bootstrap_paired_pvalue(
    scores_a: list[float], scores_b: list[float],
    *,
    n_bootstrap: int = 10000,
    random_seed: int = 42,
) -> float:
    rng = np.random.default_rng(random_seed)
    diffs = np.array([a - b for a, b in zip(scores_a, scores_b)])
    observed = np.mean(diffs)
    count = 0
    for _ in range(n_bootstrap):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        if np.mean(sample) >= observed:
            count += 1
    return count / n_bootstrap
```

Paste this verbatim in your T28 prompt. Any other version is wrong.

---

*Model assignments updated incrementally as you test each model on a tranche. Update this file with your findings — it's the runnable reference for the full buildout.*
