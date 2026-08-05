# CUDA Optimization Implementation Stubs for Aurelius

This directory contains concrete CUDA optimization implementations for the vectors identified in CUDA_OPTIMIZATION_ANALYSIS.md.

## Files

### optimized_attention.py
Drop-in replacement for attention module using SageAttention-2 + KIVI quantization.

### cuda_kernel_bench.py
Benchmark harness comparing FP16 SDPA vs INT8 SageAttention.

### memory_layout.py
Optimal tensor layouts for KV cache (transposed for memory coalescing).

## Usage

```python
# Enable optimizations in config.yaml:
# use_sage_attention: True
# kivi_bits: 2

# Then use optimized block:
from src.model.optimized_attention import OptimizedGQA

block = OptimizedGQA(config)
out, kv = block(x, freqs_cis, cache)
```

## Performance Targets

| Model | Current VRAM (16K seq) | Target VRAM | Current tok/s | Target tok/s |
|-------|------------------------|-------------|---------------|--------------|
| 7B   | 6.5 GB                | 4.0 GB      | 45            | 75           |
| 14B  | 12 GB                 | 7.5 GB      | 32            | 55           |
| 32B  | 28 GB                 | 18 GB       | 18            | 32           |