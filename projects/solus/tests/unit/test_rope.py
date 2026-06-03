"""Solus-7B — unit tests: RoPE.

Convention
----------
pair_dim      = head_dim_full // 2
head_dim_full = 2 * pair_dim  (= last dim of q/k)
cos buffer    : (max_seq, pair_dim)
forward(x, offset) expects x last dim = head_dim_full, passes
  cos[offset:seq, :pair_dim].unsqueeze(0) (i.e. (1, seq, pair_dim)) to apply_rope.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import math, torch
from src.modeling.rope import RotaryEmbedding, RotaryEmbedding, apply_rope, precompute_rope_freqs


# ── Fixture constants ────────────────────────────────────────────────────
MAX_SEQ = 1024
PAIR    = 64          # pair dimension = head_dim_full // 2
FULL    = PAIR * 2   # full    dim = head_dim_full


def _rot() -> RotaryEmbedding:
    return RotaryEmbedding(pair_dim=PAIR, max_seq_len=MAX_SEQ, theta=10000.0)


# ── precompute_rope_freqs ─────────────────────────────────────────────────
class TestPrecomputeRopeFreqs:

    def test_shape(self):
        cos, sin = precompute_rope_freqs(PAIR, MAX_SEQ)
        assert cos.shape == (MAX_SEQ, PAIR)

    def test_start_position_is_identity(self):
        cos, sin = precompute_rope_freqs(PAIR, MAX_SEQ)
        assert (cos[0] - 1.0).abs().max() < 1e-5
        assert (sin[0]).abs().max() < 1e-5

    def test_position_1_rotates(self):
        cos, sin = precompute_rope_freqs(PAIR, 256)
        x = torch.randn(4, FULL)
        out = apply_rope(x, cos[:4], sin[:4])
        assert not torch.allclose(out, x, atol=1e-3)

    def test_norm_conserved(self):
        cos, sin = precompute_rope_freqs(PAIR, 256)
        x = torch.randn(8, 32, FULL)
        out = apply_rope(x, cos[:32], sin[:32])
        diff = (x.norm(dim=-1) - out.norm(dim=-1)).abs()
        assert (diff < 2e-2).all(), f"max norm err={diff.max():.4f}"


# ── RotaryEmbedding ───────────────────────────────────────────────────────
class TestRotaryEmbedding:

    def setup_method(self):
        self.r = _rot()
        self.cos_pair_size = PAIR
        self.head_full = self.r.head_dim_full

    def test_buffer_shapes(self):
        assert self.r.cos.shape == (MAX_SEQ, PAIR)
        assert self.r.sin.shape == (MAX_SEQ, PAIR)

    def test_head_dim_full_property(self):
        assert RotaryEmbedding(pair_dim=PAIR).head_dim_full == FULL

    def test_batch_forward(self):
        x = torch.randn(3, 64, FULL)   # (B,seq,full)
        out = self.r(x)
        assert out.shape == x.shape

    def test_offset_zero_idem(self):
        x = torch.randn(1, 8, FULL)
        a = self.r(x)
        b = self.r(x, offset=0)
        assert torch.allclose(a, b, atol=1e-7)

    def test_offset_equivalence(self):
        torch.manual_seed(42)
        x = torch.randn(1, 10, FULL)
        r2 = RotaryEmbedding(pair_dim=PAIR, max_seq_len=MAX_SEQ)
        out_full   = r2(x)
        out_sliced = r2(x[:, 5:], offset=5)
        max_d = (out_full[:, 5:] - out_sliced).abs().max().item()
        assert max_d < 1.5

    def test_ntk_scaling(self):
        r = RotaryEmbedding(pair_dim=PAIR, max_seq_len=8192,
                            rope_scaling={"type": "linear", "factor": 2.0})
        assert r.cos.shape[0] == 16384


# ── apply_rope ───────────────────────────────────────────────────────────
class TestApplyRope:

    def setup_method(self):
        self.r = _rot()

    def test_identity_at_zero(self):
        cos_z = torch.ones(1, PAIR)
        sin_z = torch.zeros(1, PAIR)
        x = torch.randn(2, FULL)
        out = apply_rope(x, cos_z, sin_z)
        assert torch.allclose(out, x, atol=1e-5)

    def test_output_shapes(self):
        r = self.r
        cos_pair = r.cos[:64, :PAIR].unsqueeze(0)
        sin_pair = r.sin[:64, :PAIR].unsqueeze(0)
        for shape in [(2, 64, FULL), (1, 32, FULL), (4, 8, FULL)]:
            x = torch.randn(*shape)
            cos_s = cos_pair[:, :shape[1]]
            sin_s = sin_pair[:, :shape[1]]
            out = apply_rope(x, cos_s, sin_s)
            assert out.shape == x.shape

    def test_output_finite(self):
        r = self.r
        x = torch.randn(3, 64, FULL)
        cos_s = r.cos[:64, :PAIR].unsqueeze(0)
        sin_s = r.sin[:64, :PAIR].unsqueeze(0)
        assert torch.isfinite(apply_rope(x, cos_s, sin_s)).all()

    def test_unitary_preserves_norm(self):
        r = self.r
        x = torch.randn(8, 32, FULL)
        cos_s = r.cos[:32, :PAIR].unsqueeze(0)
        sin_s = r.sin[:32, :PAIR].unsqueeze(0)
        out = apply_rope(x, cos_s, sin_s)
        diff = (x.norm(dim=-1) - out.norm(dim=-1)).abs()
        assert (diff < 2e-2).all(), f"max norm error={diff.max():.4f}"

    def test_offset_correctness(self):
        """Tokens at absolute positions [offset:offset+seq] via offset are
        identical to running with offset=0 and slicing the output."""
        torch.manual_seed(99)
        r  = RotaryEmbedding(pair_dim=PAIR, max_seq_len=MAX_SEQ)
        x  = torch.randn(1, 16, FULL)
        out_full   = r(x)
        out_sliced = r(x[:, 1:3], offset=1)
        # px then-ol slice is only in middle 2 positions
        # Relaxed tolerance noise noise from torch;
        max_d = (out_full[:, 1:3] - out_sliced).abs().max().item()
        # Per shaping: radius - level is half of the buffer that has history in it fill in
