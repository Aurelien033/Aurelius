#!/usr/bin/env python3
"""CUDA optimization benchmark for Aurelius attention layers.

Compares:
1. Standard PyTorch SDPA (FP16)
2. SageAttention-2 (INT8 quantized)
3. KIVI KV cache compression
4. Liger kernel fused FFN

Usage:
    python benchmarks/cuda_optimization_benchmark.py --model 7b --seq_len 4096
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class BenchmarkResult:
    name: str
    time_ms: float
    memory_mb: float
    tokens_per_sec: float


def benchmark_attention_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n_iters: int = 100) -> BenchmarkResult:
    """Benchmark standard PyTorch SDPA attention."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    start = time.perf_counter()
    for _ in range(n_iters):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    
    elapsed_ms = (time.perf_counter() - start) / n_iters * 1000
    memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    seq_len = q.shape[2]
    
    return BenchmarkResult(
        name="SDPA FP16",
        time_ms=elapsed_ms,
        memory_mb=memory_mb,
        tokens_per_sec=seq_len / (elapsed_ms / 1000),
    )


def benchmark_sage_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n_iters: int = 100) -> BenchmarkResult | None:
    """Benchmark SageAttention-2 if available."""
    try:
        import sageattention  # type: ignore
    except ImportError:
        return None
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    start = time.perf_counter()
    for _ in range(n_iters):
        out = sageattention.sageattn(q, k, v, tensor_layout="HND", is_causal=True)
    torch.cuda.synchronize()
    
    elapsed_ms = (time.perf_counter() - start) / n_iters * 1000
    memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    seq_len = q.shape[2]
    
    return BenchmarkResult(
        name="SageAttention-2 INT8",
        time_ms=elapsed_ms,
        memory_mb=memory_mb,
        tokens_per_sec=seq_len / (elapsed_ms / 1000),
    )


def benchmark_kivi_compression(k_cache: torch.Tensor, v_cache: torch.Tensor, bits: int = 2, n_iters: int = 100) -> BenchmarkResult | None:
    """Benchmark KIVI quantization compression speed and ratio."""
    try:
        from src.inference.kivi_quant import KIVIQuantizer
    except ImportError:
        return None
    
    quantizer = KIVIQuantizer(bits=bits, residual_length=128)
    
    torch.cuda.synchronize()
    original_bytes = k_cache.numel() * 2 + v_cache.numel() * 2  # FP16
    
    start = time.perf_counter()
    compressed = None
    for _ in range(n_iters):
        compressed = quantizer.compress_kv_cache(k_cache, v_cache)
    torch.cuda.synchronize()
    
    elapsed_ms = (time.perf_counter() - start) / n_iters * 1000
    
    if compressed is None:
        return None
    
    # Calculate compressed size
    k_q = compressed["k_q"]
    v_q = compressed["v_q"]
    compressed_bytes = k_q.numel() + v_q.numel() + 2 * (compressed["k_scale"].numel() + compressed["v_scale"].numel()) * 2  # int8 + fp16 scales
    
    compression_ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 0
    
    return BenchmarkResult(
        name=f"KIVI {bits}-bit compression",
        time_ms=elapsed_ms,
        memory_mb=compressed_bytes / (1024**2),
        tokens_per_sec=compression_ratio,
    )


def benchmark_ffn_swiglu(x: torch.Tensor, d_model: int, d_ff: int, n_iters: int = 100) -> BenchmarkResult:
    """Benchmark SwiGLU FFN with standard vs Liger kernels."""
    
    # Standard implementation
    gate = nn.Linear(d_model, d_ff, bias=False).cuda()
    up = nn.Linear(d_model, d_ff, bias=False).cuda()
    down = nn.Linear(d_ff, d_model, bias=False).cuda()
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    start = time.perf_counter()
    for _ in range(n_iters):
        g = gate(x)
        u = up(x)
        h = torch.nn.functional.silu(g) * u
        out = down(h)
    torch.cuda.synchronize()
    
    elapsed_std_ms = (time.perf_counter() - start) / n_iters * 1000
    memory_std_mb = torch.cuda.max_memory_allocated() / (1024**2)
    
    return BenchmarkResult(
        name="SwiGLU Standard",
        time_ms=elapsed_std_ms,
        memory_mb=memory_std_mb,
        tokens_per_sec=x.shape[1] / (elapsed_std_ms / 1000),
    )


def main():
    parser = argparse.ArgumentParser(description="CUDA optimization benchmark for Aurelius")
    parser.add_argument("--model", choices=["150m", "1b", "3b", "7b", "14b", "32b"], default="7b")
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_iters", type=int, default=100)
    args = parser.parse_args()
    
    # Model configs
    configs = {
        "150m": {"d_model": 768, "n_heads": 12, "d_ff": 3072},
        "1b": {"d_model": 1536, "n_heads": 16, "d_ff": 6144},
        "3b": {"d_model": 2048, "n_heads": 20, "d_ff": 8192},
        "7b": {"d_model": 3584, "n_heads": 40, "d_ff": 14336},
        "14b": {"d_model": 5120, "n_heads": 40, "d_ff": 20480},
        "32b": {"d_model": 7168, "n_heads": 56, "d_ff": 28672},
    }
    
    cfg = configs[args.model]
    head_dim = cfg["d_model"] // cfg["n_heads"]
    n_kv_heads = cfg["n_heads"] // 4  # GQA 4:1 ratio
    
    print(f"\n{'='*60}")
    print(f"Aurelius CUDA Optimization Benchmark - {args.model.upper()}")
    print(f"{'='*60}\n")
    
    # Create test tensors
    q = torch.randn(args.batch_size, cfg["n_heads"], args.seq_len, head_dim, device="cuda", dtype=torch.half)
    k = torch.randn(args.batch_size, n_kv_heads, args.seq_len, head_dim, device="cuda", dtype=torch.half)
    v = torch.randn(args.batch_size, n_kv_heads, args.seq_len, head_dim, device="cuda", dtype=torch.half)
    
    x = torch.randn(args.batch_size, args.seq_len, cfg["d_model"], device="cuda", dtype=torch.half)
    k_cache = torch.randn(args.batch_size, args.seq_len, n_kv_heads, head_dim, device="cuda", dtype=torch.half)
    v_cache = torch.randn(args.batch_size, args.seq_len, n_kv_heads, head_dim, device="cuda", dtype=torch.half)
    
    results = []
    
    # Attention benchmarks
    print("Attention Kernels:")
    print("-" * 40)
    
    r_std = benchmark_attention_sdpa(q, k, v, args.n_iters)
    results.append(r_std)
    print(f"  SDPA FP16:       {r_std.time_ms:.2f} ms, {r_std.memory_mb:.1f} MB")
    
    r_sage = benchmark_sage_attention(q, k, v, args.n_iters)
    if r_sage:
        results.append(r_sage)
        speedup = r_std.time_ms / r_sage.time_ms
        print(f"  SageAttention-2: {r_sage.time_ms:.2f} ms, {r_sage.memory_mb:.1f} MB, {speedup:.2f}x speedup")
    else:
        print("  SageAttention-2: Not installed")
    
    # KV cache benchmarks
    print("\nKV Cache Quantization:")
    print("-" * 40)
    
    r_kv = benchmark_kivi_compression(k_cache, v_cache, bits=2, n_iters=args.n_iters)
    if r_kv:
        results.append(r_kv)
        print(f"  KIVI 2-bit:      {r_kv.time_ms:.2f} ms, {r_kv.memory_mb:.1f} MB compressed, {r_kv.tokens_per_sec:.1f}x compression")
    
    # FFN benchmarks
    print("\nFFN Layers:")
    print("-" * 40)
    
    r_ffn = benchmark_ffn_swiglu(x, cfg["d_model"], cfg["d_ff"], args.n_iters)
    results.append(r_ffn)
    print(f"  SwiGLU Standard:  {r_ffn.time_ms:.2f} ms, {r_ffn.memory_mb:.1f} MB")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    
    if r_sage:
        print(f"  Attention speedup (SageAttn vs SDPA): {r_std.time_ms / r_sage.time_ms:.2f}x")
    print(f"  Memory needed for {args.seq_len} tokens attention: {r_std.memory_mb:.1f} MB (FP16) → {r_sage.memory_mb if r_sage else r_std.memory_mb:.1f} MB optimized")
    
    # Export results
    import json
    payload = {
        "model": args.model,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "results": [
            {"name": r.name, "time_ms": r.time_ms, "memory_mb": r.memory_mb, "tokens_per_sec": r.tokens_per_sec}
            for r in results
        ]
    }
    print(f"\nJSON results: {json.dumps(payload, indent=2)}")


if __name__ == "__main__":
    main()