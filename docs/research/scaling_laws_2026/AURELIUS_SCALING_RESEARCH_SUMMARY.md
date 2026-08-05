# Aurelius Scaling-Law Research Summary

Date: 2026-06-26

This file captures the closer scaling-law research pass and the concrete implications for Aurelius.

## Completed artifacts

Files created or updated:

- `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/inventory.md`
- `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/CLOSER_RESEARCH_TO_AURELIUS.md`
- `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/ACDT_SCALING_MECHANISMS.md`
- `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/aurelius_config_scaling_audit.json`
- `/Users/christienantonio/aurelius/src/model/tapered_transformer.py`
- `/Users/christienantonio/aurelius/tests/model/test_tapered_transformer.py`

Verification performed:

- Processed PDFs: 21
- Missing PDFs: 0
- Lilian Weng scaling-laws page fetched: 35,125 chars
- Test command: `pytest tests/model/test_tapered_transformer.py -q`
- Test result: `3 passed`

## Important correction

Tapered Language Models do not automatically give a 40% total FLOP reduction when implemented correctly.

The core claim is matched-budget capacity reallocation: move MLP width from later layers to earlier layers while preserving average `d_ff`, total parameter budget, and roughly total FFN compute.

An earlier working note overclaimed “40% FLOP reduction.” The closer synthesis corrects this, and correction notices were prepended to older draft docs that contained speculative or over-strong language.

## Updated thesis

A ≤10B Aurelius can beat many 100B+ systems in practical use, but not by pretending parameter scale is irrelevant.

The viable path is:

1. Data-rich training under the local deployment cap.
2. Sparse capacity for rare/complex tasks.
3. Verifier-backed inference-time search.
4. RLVR for backtracking and repair behavior.
5. Learned memory/resource management.
6. Agent/CUA data flywheels with strong verifiers.
7. Component-level AMC/memory benchmarks.

Large models still have a real advantage: they learn rare/complex task features because they suffer less gradient interference. Aurelius must compensate through routing, replay, MoE, memory, and verifier-driven RL, not just longer next-token training.

## Highest-leverage findings for Aurelius

### 1. Taper only `d_ff`, not `d_model`

From Tapered Language Models, the correct implementation is to keep the residual stream fixed and taper the MLP intermediate width.

Formula:

```text
d_ff(l) = d_end + (d_start - d_end) * (1 + cos(pi*l/(L-1))) / 2
```

Default schedule:

```text
1.5x d_ff at early layers -> 0.5x d_ff at late layers
```

Implemented in:

```text
src/model/tapered_transformer.py
```

Validated 7B schedule:

```text
layer 0:  21504
layer 39: 7168
average:  14336.0
```

This preserves the config_7b baseline average `d_ff = 14336`.

### 2. Aurelius configs are data-rich relative to Chinchilla

Validated config audit:

```text
config_1b.yaml:  approx 0.98B params, 200B tokens, D/N=203x
config_3b.yaml:  approx 3.48B params, 500B tokens, D/N=143x
config_7b.yaml:  approx 8.39B params, 1T tokens, D/N=119x
config_14b.yaml: approx 22.49B params, 2T tokens, D/N=89x
config_32b.yaml: approx 66.95B params, 4T tokens, D/N=60x
```

Interpretation:

Aurelius is intentionally data-heavy if the deployment cap is ≤10B. This is not compute-optimal in the pure Chinchilla sense, but it can be strategically right if the extra data is used to teach rare-task behavior, tool use, memory behavior, CUA behavior, and verifier-backed reasoning.

### 3. “Why Larger Models Learn More” gives Aurelius a warning

The paper’s key mechanism: small models allocate limited neurons to high-frequency/simple features first. Rare/complex tasks get displaced.

Aurelius response:

- Rare-task replay buckets.
- Hard-example oversampling.
- Gradient-interference telemetry.
- MoE experts specialized for rare/complex tasks.
- Eval slices for low-frequency capability, not just average loss.

### 4. MoE should be framed as rare-task capacity, not hype

The Shazeer MoE paper matters because it shows conditional computation can add capacity without proportional active compute. But it also warns about router collapse.

Existing relevant code:

```text
src/model/moe.py
```

Production gates needed:

```text
max/min expert load ratio < 2.0
token drop rate < 0.1%
rare-task suite improves at matched active FLOPs
router aux/z-loss stable
```

### 5. Neural Garbage Collection is directly relevant to AMC/KV memory

NGC’s key idea: treat KV eviction as a learned RL action, not a hand-coded heuristic.

Aurelius action:

Add a `MemoryEvictionHead` or resource-action head:

```text
reward = task_success - lambda * peak_kv_fraction
```

This is stronger than static KIVI/PackKV alone because the model learns what to forget based on task reward.

### 6. SPIRAL should upgrade Speculative Thinking

SPIRAL’s core point: models are usually trained on single sequential traces, but deployed with sequential + parallel + aggregation compute. That mismatch wastes inference compute.

Aurelius action:

Train the Speculative Thinking coordinator as a candidate-set RL system:

```text
prompt -> N traces -> subset scoring -> aggregation -> final verifier reward
```

This should not be just runtime best-of-N. The generator should learn to produce traces useful to the aggregator.

### 7. Fara-1.5 is the CUA recipe Aurelius should copy

Fara’s pipeline:

```text
environments + solver + user simulator + verifiers
```

Verifier families:

```text
correctness
efficiency
critical-point/user-interaction adherence
```

Aurelius CUA v2 should use this shape: synthetic websites, stateful backends, teacher rollouts, user simulator, and three independent verifier gates before any trajectory enters SFT.

### 8. RLVR over SFT is not optional for reasoning

The RLVR paper’s strongest claim: SFT on golden paths does not teach backtracking. RLVR learns from failed rollouts and teaches recovery.

Aurelius training implication:

```text
SFT = imitation prior
RLVR = repair/backtracking/search competence
```

For code, tools, CUA, and reasoning, Aurelius needs failed trajectories and verifier-backed recovery, not only clean demonstrations.

### 9. FORGE and MD Decoupling are training-feasibility upgrades

FORGE:

- Eliminates materialized gradient tensors by consuming gradient tiles in-register.
- Extracted numbers: 75.0 -> 35.4 GiB peak memory on Llama-3.1-8B.
- Reported speedup: about 1.5x.
- This is not a quick Python patch, but should enter the CUDA/kernel roadmap.

MD Decoupling:

- Separates magnitude and direction dynamics in optimizers.
- Reported to improve Adam/Muon and help MoE scaling.
- Aurelius already uses Muon in configs; this is worth a future optimizer experiment.

## Recommended implementation order

1. Keep the corrected tapered FFN utility and wire it behind a config flag.
2. Add proxy eval: uniform FFN vs tapered FFN at 100M/1B.
3. Add MoE telemetry before adding more MoE complexity.
4. Build rare-task eval buckets from the cleaned dataset.
5. Add SPIRAL-style candidate-set evaluator for reasoning/code tasks.
6. Add NGC-style learned KV eviction as a small RLVR experiment.
7. Build Fara-style CUA synthetic env + verifier pipeline.
8. Add AMC component benchmarks: representation fidelity, retrieval precision, update correctness, long-horizon stability, latency/cost.

## Bottom line

Aurelius should not chase 100B dense scale. The better local-first architecture is:

```text
≤10B dense/sparse core
+ cosine tapered FFN
+ rare-task MoE/replay
+ FlashMLA/KV compression
+ learned KV eviction
+ SPIRAL candidate-set RL
+ RLVR backtracking
+ Fara-style CUA data flywheel
+ AMC component benchmarks
```

That is the path where a smaller model can beat larger models in actual agentic use: not by magic, but by spending capacity and inference compute where large dense models waste it.
