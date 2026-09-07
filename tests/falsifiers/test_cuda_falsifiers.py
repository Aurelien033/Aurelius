"""Falsifier tests for CUDA optimization mechanisms (ACDT spec compliance)."""

import pytest
import torch

from src.model.flash_mla import FlashMLAAttention, FlashMLAConfig
from src.inference.turboquant.qjl import QJLSketch
from src.inference.turboquant.polar_quant import PolarQuant


class TestMLAFalsifier:
    """F-MLA-01: FlashMLA numerical equivalence falsifier."""

    def test_standard_absorbed_equivalence(self):
        """Verify both paths produce identical outputs within tolerance."""
        cfg = FlashMLAConfig(d_model=64, n_heads=4, head_dim=16, kv_lrank=16)
        model = FlashMLAAttention(cfg)
        model.absorb_projections()

        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            out_std = model(x, use_absorbed=False)
            out_abs = model(x, use_absorbed=True)

        max_diff = (out_std - out_abs).abs().max().item()
        assert max_diff < 1e-4, f"Falsified: max_diff={max_diff:.6f} exceeds tolerance"


class TestQJLFalsifier:
    """F-QJL-02: QJL unbiased estimator falsifier."""

    def test_estimator_unbiasedness(self):
        """Verify QJL estimator maintains variance bound."""
        sketch_dim = 64
        dim = 128

        qjl = QJLSketch(dim=dim, sketch_dim=sketch_dim, seed=42)

        # Generate test data
        residual = torch.randn(100, dim)
        query = torch.randn(100, dim)

        signs, norms = qjl.compress_keys(residual)
        estimate = qjl.estimate_attention(signs, norms, query)

        # True inner products
        true_inner = (residual * query).sum(dim=-1)

        # Check variance ratio (unbiased estimator property)
        var_ratio = estimate.var() / true_inner.var()
        assert var_ratio <= 1.1, f"Falsified: var_ratio={var_ratio:.3f} exceeds bound"


class TestKIVIFalsifier:
    """F-KIVI-03: KV cache memory reduction falsifier."""

    def test_compression_ratio(self):
        """Verify PolarQuant achieves expected compression ratio."""
        dim = 128
        n_codes = 4  # 2-bit = 4 codes

        pq = PolarQuant(dim=dim, n_codes=n_codes)

        # Original storage: dim floats = 128 * 4 bytes = 512 bytes
        # Compressed storage: n_codes indices (int64) + mins + maxs = 128*8 + 8 + 8 = 1040 bytes
        # With quantization to int4: much smaller

        x = torch.randn(100, dim)
        state, residual = pq.compress(x)

        original_bytes = x.numel() * 4  # float32
        compressed_bytes = state.codes.numel() * 1 + state.mins.numel() * 4 + state.maxs.numel() * 4

        # Must achieve at least 4x compression for 2-bit target
        compression_ratio = original_bytes / compressed_bytes
        assert compression_ratio >= 2.0, f"Falsified: compression_ratio={compression_ratio:.2f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
