# Closer Scaling-Law Research for Aurelius

Date: 2026-06-26
Sources extracted under: `docs/research/scaling_laws_2026/`

## 0. Corrected thesis

Small models can beat larger models only under specific mechanisms. The stronger statement is:

A ≤10B Aurelius can beat many 100B+ systems in use when it combines data-rich training, verifier-backed inference-time search, sparse capacity, memory, and systems efficiency. But parameter scale still buys rare-task capacity by reducing gradient interference. So Aurelius must replace raw dense scale with routing, replay, verifiers, and learned resource control.

This is the important correction: size is not destiny, but capacity bottlenecks are real.

## 1. Evidence extracted

### Tapered Language Models — arXiv:2606.23670
Source: `2606.23670v1.txt:180-196`, `:227-267`.

Mechanism: later MLP outputs increasingly align with the residual stream, so later layers refine rather than compute new features. Reallocate MLP width toward early layers while preserving average `d_ff`.

Formula:

```text
d_ff(l) = d_end + (d_start - d_end) * (1 + cos(pi*l/(L-1))) / 2
```

Best reported pattern in extracted text: cosine taper, approximately `1.5x -> 0.5x`, with fixed total parameter/FLOP budget.

Aurelius action: taper `d_ff`, not `d_model`. I corrected `src/model/tapered_transformer.py` to implement `TaperedSwiGLUFFN` and a verified cosine schedule.

### Why Larger Models Learn More — arXiv:2605.29548
Source: `2605.29548v1.txt:110-150`, `:206-235`.

Mechanism: larger models learn rare/complex tasks because small models allocate neurons to high-frequency or low-complexity features. The theorem ranks features by utility `u_k,j = pi_k * lambda_k,j`; rare features have low utility and are displaced in small models.

Aurelius action: do not rely on data volume alone. Add rare-task protection:
- task-balanced replay
- hard-example oversampling
- gradient-interference telemetry
- MoE experts specialized for rare/complex tasks

### Sparsely-Gated MoE — arXiv:1701.06538
Source: `1701.06538v1.txt:26-38`, `:188-218`, `:288-305`.

Mechanism: conditional computation gives large capacity with smaller active compute. Noisy top-k gating uses:

```text
G(x) = softmax(KeepTopK(H(x), k))
H_i(x) = (x W_g)_i + Normal() * Softplus((x W_noise)_i)
```

Failure mode: router collapse. The paper uses an importance load-balancing loss based on coefficient of variation.

Aurelius action: existing `src/model/moe.py` is relevant, but production MoE needs utilization gates: max/min expert load ratio, token-drop rate, and auxiliary/z-loss monitoring.

### Neural Garbage Collection — arXiv:2604.18002
Source: `2604.18002v1.txt:13-27`, `:120-129`, `:212-240`.

Mechanism: train the model to evict KV entries as an RL action using outcome reward. Reported extracted result: strong accuracy with 2-3x peak KV compression; on Countdown, 49.6% vs 21.2% next-best baseline at 2.4x cache reduction.

Aurelius action: current KV stack is static/compressive. Add a learned `MemoryEvictionHead` trained with RLVR reward + memory cost penalty.

### SPIRAL — arXiv:2606.23595
Source: `2606.23595v1.txt:17-31`, `:149-200`, `:202-240`.

Mechanism: train sequential, parallel, and aggregation inference together. Set RL gives shared reward to a candidate set, then marginal set advantage credits candidates based on usefulness in successful sets.

Aurelius action: the Speculative Thinking coordinator should not be only a runtime scaffold. Train Aurelius to generate diverse candidates that aggregate well.

### Fara-1.5 — arXiv:2606.20785
Source: `2606.20785v1.txt:19-34`, `:96-106`, `:117-137`.

Mechanism: scalable CUA data flywheel = environments + solvers + verifiers. Three verifier families: correctness, efficiency, critical-point/user-interaction adherence. 9B model reaches 63.4% Online-Mind2Web and 86.6% WebVoyager in extracted abstract/figure.

Aurelius action: copy the pipeline shape for CUA v2: synthetic websites, teacher solver, user simulator, and three independent verifiers.

### RLVR over SFT — arXiv:2606.22938
Source: `2606.22938v1.txt:24-48`, `:98-113`, `:116-138`.

Mechanism: SFT on golden shortest paths does not teach backtracking; RLVR learns to backtrack from dead ends with outcome reward. Extracted claim: exponential inference-time compute separation: RLVR `Theta(WK)` vs SFT `Theta(WL^K)`.

Aurelius action: for reasoning/tool/CUA tasks, SFT is only phase 1. Phase 2 must include verifier-backed failed rollouts and backtracking traces.

### FORGE — arXiv:2606.22932
Source: `2606.22932v1.txt:17-36`, `:41-56`, `:91-113`.

Mechanism: fuse optimizer into backward pass so gradient tiles are consumed in registers and never materialized. Extracted numbers: peak memory 75.0 -> 35.4 GiB on Llama-3.1-8B; about 1.5x faster; 8B training at 4x micro-batch.

Aurelius action: not a quick Python patch, but it changes feasibility of 8-10B continued pretraining. Add to kernel roadmap after current CUDA path stabilizes.

### Agent-native memory — arXiv:2606.24775
Source: `2606.24775v1.txt:76-107`, `:122-140`.

Mechanism: memory must be decomposed into representation/storage, extraction, retrieval/routing, and maintenance. No single architecture dominates; localized maintenance is more cost-efficient than global reorganization.

Aurelius action: AMC needs component metrics, not only end-task success: representation fidelity, retrieval precision, update correctness, long-horizon stability, latency/cost.

## 2. Aurelius config audit

Validated with script: `aurelius_config_scaling_audit.json`.

| Config | Approx params | Pretrain tokens | D/N | Chinchilla 20x target |
|---|---:|---:|---:|---:|
| 1B | 0.98B | 200B | 203x | 19.7B |
| 3B | 3.48B | 500B | 143x | 69.7B |
| 7B | 8.39B | 1T | 119x | 167.8B |
| 14B | 22.49B | 2T | 89x | 449.8B |
| 32B | 66.95B | 4T | 60x | 1.339T |

Interpretation: Aurelius configs are data-rich relative to Chinchilla. That is acceptable only if the goal is a local-capped specialist/student that absorbs broad data via curriculum, MoE, and RLVR. If pure compute-optimal pretraining is the goal, the 7B config is too small for 1T tokens; if deployment <=10B is non-negotiable, use the extra data to train rare-task behavior and verifiers rather than just next-token loss.

## 3. Highest-leverage changes

1. Replace uniform FFN with cosine tapered FFN schedule.
2. Add rare-task replay and gradient-interference telemetry.
3. Promote MoE from optional architecture flag to rare-task capacity strategy.
4. Train early-exit and memory eviction as learned resource actions, not static heuristics.
5. Build SPIRAL-style candidate-set RL for reasoning/code/CUA tasks.
6. Use Fara-style CUA data flywheel with three verifiers.
7. Add AMC component benchmark suite.
8. Track FORGE + MD Decoupling for training feasibility once kernels become implementable.

## 4. Falsifier gates

- Tapered FFN is rejected if matched-parameter proxy loses >1% validation loss vs dense baseline after equal tokens.
- MoE is rejected if max/min expert load ratio >2.0 for more than 1K steps, token drop >0.1%, or rare-task eval does not improve.
- SPIRAL is rejected if candidate diversity rises but verifier pass@k does not improve at equal inference FLOPs.
- Neural Garbage Collection is rejected if compression saves memory but answer accuracy drops beyond full-cache baseline tolerance.
- CUA data flywheel is rejected if any generated trajectory lacks correctness, efficiency, or critical-point verifier approval.
