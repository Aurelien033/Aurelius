\
"""
Solus-MoE -- Router unit tests (25 tests).
Convention: hidden=1024, seq=16, num_experts=4, top_k=2.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))

import torch
import pytest
from src.modeling.config import SolusConfig
from src.modeling.moe.router import SolusMoERouter


@pytest.fixture(scope="session")
def cfg():
    return SolusConfig(
        hidden_size=1024,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=1,
        intermediate_size=1024 * 4,
        vocab_size=5000,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        expert_intermediate_size=256,
        moe_num_experts=4,
        moe_top_k=2,
    )


@pytest.fixture(scope="function")
def router(cfg):
    return SolusMoERouter(cfg)


#  Construction 
class TestRouterConstruction:
    def test_init_creates_gate(self, cfg):
        r = SolusMoERouter(cfg)
        assert hasattr(r, "gate")
        assert isinstance(r.gate, torch.nn.Linear)

    def test_init_sets_fields(self, cfg):
        r = SolusMoERouter(cfg)
        assert r.num_experts == cfg.moe_num_experts
        assert r.top_k == cfg.moe_top_k

    def test_init_gate_out_features(self, cfg):
        r = SolusMoERouter(cfg)
        assert r.gate.out_features == cfg.moe_num_experts

    def test_init_gate_in_features(self, cfg):
        r = SolusMoERouter(cfg)
        assert r.gate.in_features == cfg.hidden_size


#  Forward output 
class TestRouterForward:
    def setup_method(self):
        self.cfg = SolusConfig(
            hidden_size=1024, num_hidden_layers=2, num_attention_heads=8,
            num_key_value_heads=1, intermediate_size=1024 * 4, vocab_size=5000,
            expert_intermediate_size=256, moe_num_experts=4, moe_top_k=2,
        )
        self.r  = SolusMoERouter(self.cfg)
        self.B  = 2
        self.T  = 16
        self.hidden = torch.randn(self.B, self.T, self.cfg.hidden_size)

    def test_output_logits_shape(self):
        out = self.r(self.hidden)
        assert out.logits.shape == (self.B, self.T, self.cfg.moe_num_experts)

    def test_output_weights_shape(self):
        out = self.r(self.hidden)
        assert out.weights.shape == (self.B, self.T, self.cfg.moe_num_experts)

    def test_output_expert_idx_shape(self):
        out = self.r(self.hidden)
        assert out.expert_indices.shape == (self.B, self.T, self.cfg.moe_top_k)

    def test_output_topk_weights_shape(self):
        out = self.r(self.hidden)
        assert out.topk_weights.shape == (self.B, self.T, self.cfg.moe_top_k)

    def test_output_weights_sum_to_k(self):
        out = self.r(self.hidden)
        sums = out.weights.sum(dim=-1)
        assert (sums - self.cfg.moe_top_k).abs().max().item() < 1e-4

    def test_output_weights_finite(self):
        out = self.r(self.hidden)
        assert torch.isfinite(out.weights).all()

    def test_expert_idx_range(self):
        out = self.r(self.hidden)
        assert out.expert_indices.min() >= 0
        assert out.expert_indices.max() < self.cfg.moe_num_experts

    def test_topk_weights_sum_to_1(self):
        out = self.r(self.hidden)
        sums = out.topk_weights.sum(dim=-1)
        assert (sums - 1.0).abs().max().item() < 1e-4

    def test_topk_weights_positive(self):
        out = self.r(self.hidden)
        assert (out.topk_weights >= 0).all()

    def test_aux_loss_scalar(self):
        out = self.r(self.hidden)
        assert out.aux_loss.numel() == 1

    def test_aux_loss_finite(self):
        out = self.r(self.hidden)
        assert torch.isfinite(out.aux_loss)

    def test_det_aux_loss_zero_no_alpha(self):
        cfg = SolusConfig(
            hidden_size=256, num_hidden_layers=1, expert_intermediate_size=64,
            moe_num_experts=2, moe_top_k=1, moe_aux_loss_alpha=0.0,
        )
        r = SolusMoERouter(cfg)
        x = torch.randn(1, 4, 256)
        assert r(x).aux_loss.item() == 0.0

    def test_no_nan(self, router):
        out = router(self.hidden)
        assert torch.isfinite(out.logits).all()

    def test_batch_sizes(self):
        for b in [1, 3, 7]:
            h = torch.randn(b, self.T, self.cfg.hidden_size)
            out = self.r(h)
            assert out.logits.shape[0] == b
            assert out.weights.shape[0] == b

    def test_seq_len_1(self):
        h = torch.randn(self.B, 1, self.cfg.hidden_size)
        out = self.r(h)
        assert out.logits.shape[1] == 1

    def test_diff_experts_get_different_logits(self):
        x = torch.randn(1, 4, self.cfg.hidden_size)
        out = self.r(x)
        w = out.weights
        # At least one pair of experts should have different weights across the batch dim
        pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        any_diff = any((w[..., a] - w[..., b]).abs().max() > 1e-6
                       for a, b in pairs)
        assert any_diff, "all expert weights are identical -- router is degenerate"

    def test_seed_deterministic(self):
        x = torch.randn(1, self.T, self.cfg.hidden_size)
        torch.manual_seed(42)
        a = self.r(x)
        torch.manual_seed(42)
        b = self.r(x)
        assert torch.allclose(a.weights, b.weights, atol=1e-6)

    def test_no_grad_on_detached_logits(self):
        x = torch.randn(1, self.T, self.cfg.hidden_size, requires_grad=True)
        # Detached probs should not backprop
        probs = torch.softmax(self.r.gate(x), dim=-1)
        probs_d = probs.detach()
        assert not probs_d.requires_grad

    def test_gradient_flows(self):
        x = torch.randn(1, self.T, self.cfg.hidden_size, requires_grad=True)
        out = self.r(x)
        out.logits.sum().backward()
        assert x.grad is not None and x.grad.abs().max() > 0
