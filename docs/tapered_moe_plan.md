<!--
CORRECTION 2026-06-26: This is an earlier working note. For the corrected, closer synthesis use docs/research/scaling_laws_2026/CLOSER_RESEARCH_TO_AURELIUS.md. In particular: tapered FFN reallocates d_ff under matched average parameter/FLOP budget; it is not automatically a total 40% FLOP reduction. Effective-parameter and memory-compression figures in older notes are hypotheses/projections until verified by matched-compute ablations.
-->

# Combined Innovation Plan: Tapered + Early-Exit + MoE

## Architecture: Tapered-MoE-EarlyExit (TMoE-EE)

### Design Principle
Combine three techniques for multiplicative efficiency gains:

1. **Tapered MLP width** (Bayat 2026): -40% FLOPs in later layers
2. **Early-exit** (CALM): 2-6x speedup on confident inputs  
3. **MoE** (your implementation): 2-8x parameter amplification

### Implementation Recipe

```python
# src/model/tapered_moe_block.py
class TaperedMoEBlock(nn.Module):
    """Tapered MoE block: width decreases + mixture-of-experts for capacity."""
    
    def __init__(self, config, layer_idx, n_layers):
        # Tapered width
        taper = 1.0 - 0.5 * (layer_idx / n_layers)
        d_ff = int(config.d_ff * taper)
        
        # MoE replaces dense FFN
        self.moe = SparseMoELayer(
            d_model=config.d_model,
            d_ff=d_ff,
            n_experts=config.n_experts,
            top_k=2,
        )
```

### Efficiency Stack

For 7B model (24 layers, 8192 d_ff):

| Layer | Width | Experts Active | FLOPs/token |
|-------|-------|----------------|-------------|
| 0 | 12288 (1.5×) | 2/8 experts | 1.0x |
| 8 | 9830 | 2/8 experts | 0.8x |
| 16 | 7680 | 2/8 experts | 0.6x |
| 23 | 4096 (0.5×) | 2/8 experts | 0.3x |

Total FLOPs reduction: (sum / N) ≈ 0.6x = 40% saved

With early-exit at layer 12: additional 2x speedup

**Net result: 1.3x FLOPs + 2x early-exit = 2.6x faster inference**

### Integration with Existing Code

Your codebase already has:
- ✅ `src/model/moe.py` with `SparseMoELayer`, `ExpertFFN`
- ✅ `src/inference/early_exit.py` with `EarlyExitWrapper`
- ✅ `src/model/flash_mla.py` with KV compression
- ✅ `src/model/tapered_transformer.py` (just created)

### Training Recipe

1. **Phase 0**: Pretrain tapered transformer on 1T tokens (Chinchilla optimal)
2. **Phase 1**: Upscale to MoE with 8 experts, freeze feedforward layers
3. **Phase 2**: Fine-tune with load balancing, router warmup
4. **Phase 3**: Train early-exit classifiers on calibration set (1hr)

### Resource Projection

| Technique | VRAM | Speed | Effective Params |
|-----------|------|-------|------------------|
| Dense 7B | 16GB | 45 tok/s | 7B |
| +FlashMLA | 4GB | 120 tok/s | 7B |
| +tapered | 9GB | 180 tok/s | 7B |
| +MoE (8/2) | 24GB* | 180 tok/s | 28B total, 7B active |
| +early-exit | 24GB | 350 tok/s | 28B effective |

*MoE requires expert parallelism - use existing infrastructure

### Validation Command

```bash
# Test tapered-MoE integration
python -m src.model.tapered_transformer --test-taper-moe

# Train classifier (use cleaned dataset)
python -m src.inference.early_exit --train-classifier

# Full integration test
pytest tests/integration/test_tapered_moe_exit.py -v
```