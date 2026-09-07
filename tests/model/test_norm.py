"""Tests for src.model.norm.RMSNorm."""

from __future__ import annotations

import torch

from src.model.norm import RMSNorm


def test_rmsnorm_output_shape() -> None:
    norm = RMSNorm(64)
    x = torch.randn(2, 8, 64)
    assert norm(x).shape == x.shape


def test_rmsnorm_unit_rms_on_ones() -> None:
    norm = RMSNorm(16)
    x = torch.ones(1, 1, 16)
    out = norm(x)
    assert torch.allclose(out, torch.ones(1, 1, 16), atol=1e-5)


def test_rmsnorm_zero_mean_input() -> None:
    norm = RMSNorm(16)
    x = torch.randn(1, 1, 16) * 10 + 5
    out = norm(x)
    assert out.mean().item() > 0


def test_rmsnorm_bias_off() -> None:
    norm = RMSNorm(16, bias=False)
    assert norm.bias is None


def test_rmsnorm_bias_on() -> None:
    norm = RMSNorm(16, bias=True)
    assert norm.bias is not None


def test_rmsnorm_gradient_flow() -> None:
    norm = RMSNorm(64)
    x = torch.randn(2, 4, 64, requires_grad=True)
    out = norm(x)
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_rmsnorm_nan_safe() -> None:
    norm = RMSNorm(8)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0]]])
    out = norm(x)
    assert torch.isfinite(out).all()


def test_rmsnorm_dtype_preservation() -> None:
    x_bf16 = torch.randn(1, 2, 8, dtype=torch.bfloat16)
    norm = RMSNorm(8).to(torch.bfloat16)
    out = norm(x_bf16)
    assert out.dtype == torch.bfloat16
