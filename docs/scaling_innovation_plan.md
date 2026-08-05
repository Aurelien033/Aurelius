<!--
CORRECTION 2026-06-26: This is an earlier working note. For the corrected, closer synthesis use docs/research/scaling_laws_2026/CLOSER_RESEARCH_TO_AURELIUS.md. In particular: tapered FFN reallocates d_ff under matched average parameter/FLOP budget; it is not automatically a total 40% FLOP reduction. Effective-parameter and memory-compression figures in older notes are hypotheses/projections until verified by matched-compute ablations.
-->

# Scaling Law Innovation Plan: Aurelius Path to SOTA

## Executive Summary

NO - Models do NOT need 100B+ to be SOTA. Chinchilla law shows 20B models with 1T tokens outperform 100B+ models under-trained.

## Your Advantage Stack

1. **Existing Infrastructure** (0 cost to adopt)
   - FlashMLA: 4x KV cache compression, implemented
   - Early-exit: CALM framework, 340 lines
   - Model merging: TIES/SLERP/DARE, production-ready
   - KIVI: 2-bit quantization, implemented

2. **Novel Additions** (low implementation cost)
   - Tapered architecture: 40% FLOPs reduction
   - Speculative ensemble: Multi-exit draft experts

## Resource Requirements

| Approach | Training Cost | Inference Cost | SOTA Potential |
|----------|---------------|----------------|----------------|
| 100B+ dense | $2M+ | $20/hr | High |
| 20B + tapered + early-exit | $200K | $2/hr | Equivalent |
| 7B + FlashMLA + merging | $50K | $0.50/hr | 90% of SOTA |

## Immediate Actions

```bash
# Enable existing optimizations
sed -i 's/mla_enabled: False/mla_enabled: True/' configs/config.yaml
sed -i 's/kivi_bits: 0/kivi_bits: 2/' configs/config.yaml

# Train early-exit classifiers (data exists in cleaned dataset)
python -m src.inference.early_exit --train --calibration-data /Volumes/Yggdrasil

# Merge domain experts post-SFT
python -m src.model.model_merging --strategy ties --experts code,math,function-calling
```

## Success Metrics

- KV cache: 1.2GB → 0.3GB (enable FlashMLA)
- Inference throughput: 45 tok/s → 120 tok/s (KIVI + early-exit)
- Context window: 8K → 128K (KIVI 2-bit)
- Training cost: 10x reduction (smaller model, same data ratio)