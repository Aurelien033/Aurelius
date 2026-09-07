"""Solus-7B — unit tests: RMSNorm."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import math
import torch
from src.modeling.norm import RMSNorm


class TestRMSNorm:

    def test_output_shape(self):
        rms = RMSNorm(256, eps=1e-5)
        x = torch.randn(2, 128, 256)
        assert rms(x).shape == x.shape

    def test_output_is_finite(self):
        rms = RMSNorm(256)
        x = torch.randn(2, 128, 256)
        assert torch.isfinite(rms(x)).all()

    def test_forward_ones_input(self):
        rms = RMSNorm(128)
        x = torch.ones(1, 4, 128)
        out = rms(x)
        assert out.std().item() < 1e-4

    def test_repr(self):
        rms = RMSNorm(256, eps=1e-5)
        assert "RMSNorm" in repr(rms)

    def test_eps_near_zero_still_forwardable(self):
        rms = RMSNorm(128, eps=1e-9)
        x = torch.randn(4, 32, 128) * 1e-8
        assert torch.isfinite(rms(x)).all()

    def test_weight_init_ones(self):
        rms = RMSNorm(128)
        assert torch.allclose(rms.weight, torch.ones(128), atol=1e-6)
