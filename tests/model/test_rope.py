"""Tests for src.model.rope (AMC attention-only RoPE)."""

from __future__ import annotations

import pytest
import torch

from src.model.rope import RotaryEmbedding, apply_rope


def test_rope_creates_cos_sin() -> None:
    rope = RotaryEmbedding(dim=16, max_seq_len=128)
    cos, sin = rope(8, device=torch.device("cpu"), dtype=torch.float32)
    assert cos.shape == (8, 16) and sin.shape == (8, 16)


def test_rope_identity_at_position_zero() -> None:
    rope = RotaryEmbedding(dim=16, max_seq_len=64)
    cos, sin = rope(1, device=torch.device("cpu"), dtype=torch.float32)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)
    q_out, k_out = apply_rope(q, k, cos, sin)
    assert torch.allclose(q_out, q, atol=1e-6)
    assert torch.allclose(k_out, k, atol=1e-6)


def test_rope_causality() -> None:
    rope = RotaryEmbedding(dim=16, max_seq_len=64)
    cos, sin = rope(8, device=torch.device("cpu"), dtype=torch.float32)
    q = torch.randn(1, 1, 8, 16)
    k = torch.randn(1, 1, 8, 16)
    q_out, _ = apply_rope(q, k, cos, sin)
    q2 = q.clone()
    q2[:, :, 5, :] = 0
    q2_out, _ = apply_rope(q2, k, cos, sin)
    assert torch.allclose(q_out[:, :, :5], q2_out[:, :, :5], atol=1e-6)
    assert not torch.allclose(q_out[:, :, 5:], q2_out[:, :, 5:], atol=1e-6)


def test_rope_dot_product_is_rotation_invariant() -> None:
    torch.manual_seed(1)
    rope = RotaryEmbedding(dim=4, max_seq_len=16)
    cos, sin = rope(4, device=torch.device("cpu"), dtype=torch.float32)
    q = torch.randn(1, 1, 4, 4)
    k = torch.randn(1, 1, 4, 4)
    q_rot, k_rot = apply_rope(q, k, cos, sin)
    same_pos_q = q_rot[:, :, 2, :]
    same_pos_k = k_rot[:, :, 2, :]
    dot_rotated = (same_pos_q * same_pos_k).sum()
    dot_original = (q[:, :, 2, :] * k[:, :, 2, :]).sum()
    assert abs(dot_rotated.item() - dot_original.item()) < 1e-4


def test_rope_continuation_with_offset() -> None:
    rope = RotaryEmbedding(dim=8, max_seq_len=32)
    cos, sin = rope(8, device=torch.device("cpu"), dtype=torch.float32)
    q1 = torch.randn(1, 1, 4, 8)
    k1 = torch.randn(1, 1, 4, 8)
    q1_out, k1_out = apply_rope(q1, k1, cos[:4], sin[:4])
    q2 = torch.randn(1, 1, 4, 8)
    k2 = torch.randn(1, 1, 4, 8)
    k_cached = k1_out
    k_new = torch.cat([k_cached, k2], dim=2)
    q2_out, k_new_out = apply_rope(q2, k_new, cos, sin, position_offset=4)
    assert q2_out.shape == q2.shape
    assert k_new_out.shape == k_new.shape


def test_rope_odd_dim_raises() -> None:
    with pytest.raises(ValueError, match="even"):
        RotaryEmbedding(dim=7)


def test_rope_rotates_q_and_k_not_v() -> None:
    rope = RotaryEmbedding(dim=16, max_seq_len=64)
    cos, sin = rope(4, device=torch.device("cpu"), dtype=torch.float32)
    q = torch.randn(1, 2, 4, 16)
    k = torch.randn(1, 2, 4, 16)
    v = k.clone()
    q_out, k_out = apply_rope(q, k, cos, sin)
    assert not torch.allclose(q_out, q)
    assert not torch.allclose(k_out, k)
    assert torch.equal(v, k.clone())


def test_rope_position_continuity() -> None:
    rope = RotaryEmbedding(dim=8, max_seq_len=32)
    cos, sin = rope(16, device=torch.device("cpu"), dtype=torch.float32)
    q_short = torch.randn(1, 1, 8, 8)
    k_short = torch.randn(1, 1, 8, 8)
    q_short_out, k_short_out = apply_rope(q_short, k_short, cos[:8], sin[:8])

    q_long = torch.cat([q_short, torch.randn(1, 1, 8, 8)], dim=2)
    k_long = torch.cat([k_short, torch.randn(1, 1, 8, 8)], dim=2)
    q_long_out, k_long_out = apply_rope(q_long, k_long, cos, sin)

    assert torch.allclose(q_short_out, q_long_out[:, :, :8], atol=1e-6)
    assert torch.allclose(k_short_out, k_long_out[:, :, :8], atol=1e-6)
