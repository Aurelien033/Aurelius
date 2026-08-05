#!/usr/bin/env python3
"""Smoke test for CUDA optimization analysis - run without actual CUDA to verify code structure."""

import sys

# Verify SageAttention import works (graceful fallback)
try:
    import sageattention
    SAGE_AVAILABLE = True
except ImportError:
    SAGE_AVAILABLE = False
    print("sageattention not installed (expected on non-CUDA systems)")

# Verify KIVI import works
try:
    from src.inference.kivi_quant import KIVIQuantizer
    KIVI_AVAILABLE = True
except ImportError as e:
    KIVI_AVAILABLE = False
    print(f"KIVI import: {e}")

print(f"\nCUDA Optimization Analysis Smoke Test")
print(f"=" * 40)
print(f"SageAttention: {'OK' if SAGE_AVAILABLE else 'NOT AVAILABLE'}")
print(f"KIVI Quantizer: {'OK' if KIVI_AVAILABLE else 'NOT AVAILABLE'}")

# Check config values
import yaml
try:
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    print(f"\nCurrent config values:")
    print(f"  use_sage_attention: {cfg.get('use_sage_attention', False)}")
    print(f"  kivi_bits: {cfg.get('kivi_bits', 0)}")
    print(f"  use_liger_kernels: {cfg.get('use_liger_kernels', False)}")
except Exception as e:
    print(f"Config check failed: {e}")

# Check benchmark file exists
import os
benchmark_path = "benchmarks/cuda_optimization_benchmark.py"
if os.path.exists(benchmark_path):
    print(f"\nBenchmark script: {benchmark_path} -> EXISTS ({os.path.getsize(benchmark_path)} bytes)")
else:
    print(f"\nBenchmark script: {benchmark_path} -> MISSING")

print("\nSmoke test complete.")