"""Solus-7B — attention tests.
Convention: head_dim_full = hidden_size // num_heads; pair_dim = head_dim_full // 2.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import pytest

import math
import torch
from src.modeling.config import SolusConfig
from src.modeling.attention import SolusAttention
from src.modeling.layers   import SolusDecoderLayer


@pytest.fixture(scope="session")
def cfg():
    return SolusConfig(
        hidden_size=1024,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=1,
        intermediate_size=4096,
        max_position_embeddings=2048,
        vocab_size=50000,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_dropout=0.0,
    )


# ── TestSolusAttentionForward ─────────────────────────────────────────────
class TestSolusAttentionForward:

    @pytest.fixture(autouse=True)
    def init(self, cfg):
        self.attn    = SolusAttention(cfg)
        self.B, self.S = 2, 64
        self.hidden  = torch.randn(self.B, self.S, cfg.hidden_size)

    def test_output_shape(self):
        out, _ = self.attn(self.hidden)
        assert out.shape == (self.B, self.S, self.hidden.shape[-1])

    def test_output_dtype(self):
        out, _ = self.attn(self.hidden)
        assert out.dtype == self.hidden.dtype

    def test_no_nan_or_inf(self):
        out, _ = self.attn(self.hidden)
        assert torch.isfinite(out).all()

    def test_deterministic_with_seed(self):
        torch.manual_seed(0)
        a1, _ = self.attn(self.hidden)
        torch.manual_seed(0)
        a2, _ = self.attn(self.hidden)
        assert torch.allclose(a1, a2, atol=1e-6)

    def test_batch_sizes(self):
        attn = self.attn
        for b in [1, 3, 7]:
            h = torch.randn(b, self.S, attn.cfg.hidden_size)
            out, _ = attn(h)
            assert out.shape[:2] == (b, self.S)

    def test_seq_len_1(self):
        h = torch.randn(self.B, 1, self.hidden.shape[-1])
        out, _ = self.attn(h)
        assert out.shape[:2] == (self.B, 1)

    def test_output_differs_from_input(self):
        out, _ = self.attn(self.hidden)
        assert (out - self.hidden).abs().max() > 1e-5

    def test_gradient_flows(self):
        h = self.hidden.clone().requires_grad_()
        out, _ = self.attn(h)
        out.sum().backward()
        assert h.grad is not None and h.grad.abs().max().item() > 0


# ── TestSolusDecoderLayer ──────────────────────────────────────────────────
@pytest.fixture(scope="function")
def layer(cfg):
    return SolusDecoderLayer(cfg)

@pytest.fixture(scope="function")
def xs(cfg):
    return torch.randn(2, 64, cfg.hidden_size)


def test_forward_shape(layer, xs):
    out = layer(xs)
    assert out.shape == xs.shape


def test_output_differs_from_input(layer, xs):
    out = layer(xs)
    assert (out - xs).abs().max() > 1e-5


def test_decoder_no_nan(layer, xs):
    out = layer(xs)
    assert torch.isfinite(out).all()


def test_decoder_gradient_flows(cfg):
    xs = torch.randn(2, 64, cfg.hidden_size).requires_grad_()
    SolusDecoderLayer(cfg)(xs).sum().backward()
    assert xs.grad is not None and xs.grad.abs().max().item() > 0
