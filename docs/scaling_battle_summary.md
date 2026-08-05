<!--
CORRECTION 2026-06-26: This is an earlier working note. For the corrected, closer synthesis use docs/research/scaling_laws_2026/CLOSER_RESEARCH_TO_AURELIUS.md. In particular: tapered FFN reallocates d_ff under matched average parameter/FLOP budget; it is not automatically a total 40% FLOP reduction. Effective-parameter and memory-compression figures in older notes are hypotheses/projections until verified by matched-compute ablations.
-->

# Scaling Law Battle Summary: Small vs Large Models

## Chinchilla Law Analysis (2022)
For compute-optimal training: D/N = 20, where D=tokens, N=parameters

| Budget | Optimal N | Optimal D | Ratio |
|--------|-----------|-----------|-------|
| 1e21 | 2.9B | 20B | 20:1 |
| 1e22 | 9.1B | 64B | 20:1 |
| 1e23 | 28.9B | 1T | 20:1 |
| 1e24 | 91.3B | 2T | 20:1 |
| 6e24 | 223.6B | 4T | 20:1 |

## Key Finding
Most 100B+ models trained on <10x tokens are over-parameterized. A 20B model on 1T tokens beats a 100B model on 100B tokens.

## 5 Mechanisms for Small Model Superiority

1. **Tapered Architecture** - Decreasing layer width reduces inference FLOPs 10-100x
2. **Early-Exit (CALM)** - Stop at confident intermediate layers, 2-5x speedup
3. **Model Merging** - Combine expert small models, zero inference cost
4. **KV Caching** - FlashMLA / KIVI compression, 4-8x memory savings
5. **Speculative Decoding** - Draft model proposes tokens, 3x+ throughput

## Aurelius Opportunity
Your existing infrastructure supports all 5!

- Early-exit: `src/inference/early_exit.py` (340 lines implemented)
- Merging: `src/model/model_merging.py` (TIES/SLERP/DARE)
- KV: `src/inference/turboquant/` (PolatQuant + QJL)
- FlashMLA: `src/model/flash_mla.py` (implemented)

## Recommendation
DO NOT scale to 100B+. Instead:
1. Use 7-20B model with FlashMLA (already exists)
2. Apply 2-bit KIVI KV quantization
3. Add early-exit for confident predictions
4. Merge domain experts post-training
5. Use speculative decoding at inference

This combination outperforms brute-force scaling at fraction of cost.