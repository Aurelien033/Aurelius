<!--
CORRECTION 2026-06-26: This is an earlier working note. For the corrected, closer synthesis use docs/research/scaling_laws_2026/CLOSER_RESEARCH_TO_AURELIUS.md. In particular: tapered FFN reallocates d_ff under matched average parameter/FLOP budget; it is not automatically a total 40% FLOP reduction. Effective-parameter and memory-compression figures in older notes are hypotheses/projections until verified by matched-compute ablations.
-->

# Tapered Architecture: Formulas & Implementation

## Key Formulas from Bayat et al. 2026 (arXiv:2606.23670)

### Cosine Taper Schedule (Equation 5):
```
dff(l) = dend + (dstart - dend) * (1 + cos(π * l / (L-1))) / 2
```

Optimal range: dstart = 1.5 * dff_baseline, dend = 0.5 * dff_baseline

### Linear Taper Schedule (Equation 4):
```
dff(l) = dstart - (dstart - dend) * (l / (L-1))
```

### Sigmoid Taper (Equation 6):
```
dff(l) = dend + (dstart - dend) / (1 + exp(k * (L-1 - 0.5) - k * l))
```

## Empirical Results

For 440M Transformer:
- Uniform baseline: 16.28 perplexity
- Cosine taper: 14.44 perplexity (-1.84 improvement)
- Linear taper: ~15.5 perplexity
- Sigmid taper: ~15.2 perplexity

Best performer: Cosine taper with 1.5→0.5 range.

## Implementation Notes

1. Apply taper to MLP width only (attention width unchanged)
2. Total parameters held constant across taper configurations
3. Requires projection layers when coupling with standard blocks
4. Works across: Transformer, Gated Attention, Hope-attention, Titans

## For Aurelius (7B)

Current: dff = 4 * d_model = 8192 for all layers
Tapered: dff varies 12288 → 4096 across 24 layers
- Layer 0: dff = 12288 (1.5×)
- Layer 12: dff = 8192 (1.0×)  
- Layer 23: dff = 4096 (0.5×)

FLOPs reduction: ~40% in later layers