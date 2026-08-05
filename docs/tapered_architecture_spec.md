<!--
CORRECTION 2026-06-26: This is an earlier working note. For the corrected, closer synthesis use docs/research/scaling_laws_2026/CLOSER_RESEARCH_TO_AURELIUS.md. In particular: tapered FFN reallocates d_ff under matched average parameter/FLOP budget; it is not automatically a total 40% FLOP reduction. Effective-parameter and memory-compression figures in older notes are hypotheses/projections until verified by matched-compute ablations.
-->

# Tapered Architecture for Aurelius

## Novel Mechanism: Progressive Layer Width Reduction

Key insight: Deeper layers don't need full width. Taper from d_model → d_model/4 toward output.

## Implementation Plan

```python
# src/model/tapered_block.py
class TaperedTransformerBlock(nn.Module):
    """Transformer block with progressive width reduction."""
    
    def __init__(self, config, layer_idx, n_layers):
        # Calculate reduced width for this layer
        taper_ratio = 1.0 - 0.5 * (layer_idx / n_layers)  # Linear taper
        d_model = int(config.d_model * taper_ratio)
        
        self.attn = GroupedQueryAttention(d_model)
        self.ffn = SwiGLU(d_model, int(config.d_model * 4 * taper_ratio))
        self.norm = RMSNorm(d_model)
```

## Performance Projection

For 7B model (24 layers, d_model=2048):
- Layer 0: 2048 width
- Layer 12: ~1500 width  
- Layer 23: ~1000 width

FLOPs reduction: ~40% at inference vs uniform width.

Projected speedup: 1.6x on same hardware.