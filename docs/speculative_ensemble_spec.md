# Speculative Ensemble Architecture

## Novel Mechanism: Multi-Early-Exit Speculative Decoding

Instead of single draft model, use ensemble of early-exit checkpoints.

## Architecture

```
Input → Layer 4 (confident?) → If yes: exit + propose tokens
      → Layer 8 (confident?) → If yes: exit + propose tokens  
      → Layer 12 (confident?) → If yes: exit + propose tokens
      → Full model (Layer 24)
```

Each exit point acts as a "draft expert" for different confidence levels.

## Implementation

```python
# src/inference/speculative_ensemble.py
class SpeculativeEnsembleDecoder:
    """Multiple early-exit points as speculative draft experts."""
    
    def __init__(self, model):
        self.exits = {
            4: model.layer_4_highway,
            8: model.layer_8_highway,
            12: model.layer_12_highway,
        }
        
    def generate(self, input_ids, n_speculate=4):
        # Try earliest exit first
        for exit_layer, classifier in sorted(self.exits.items()):
            logits, exited = self.try_exit(input_ids, classifier, n_speculate)
            if exited:
                return self.verify_and_append(logits)
        # Fall back to full model
        return self.full_model_generate(input_ids, n_speculate)
```

## Performance Projection

- Low-confidence inputs (40%): use exit at layer 4 (6x speedup)
- Medium (30%): layer 8 (3x speedup) 
- High-confidence (20%): layer 12 (1.5x speedup)
- Remaining: full model

Weighted average: 3.6x overall speedup vs no speculation.

## Resource Cost

- Training: Train highway classifiers on calibration set (<1hr)
- Memory: Negligible (classifiers << main model)
- Integration: Use existing early_exit.py as base