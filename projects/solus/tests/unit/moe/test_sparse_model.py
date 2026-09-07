"""
Solus-MoE -- SolusMoEForCausalLM unit tests.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))

import torch
import pytest
from src.modeling.config import SolusConfig
from src.modeling.moe.sparse_model import SolusMoEForCausalLM


@pytest.fixture(scope="session")
def moe_cfg():
    return SolusConfig(
        name="solus-moe-8b",
        architecture="solus-moe-transformer",
        hidden_size=1024,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=1,
        intermediate_size=1024 * 4,
        expert_intermediate_size=256,
        moe_num_experts=4,
        moe_top_k=2,
        moe_layer_indices=[0],
        vocab_size=5000,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )


@pytest.fixture(scope="session")
def moe_model(moe_cfg):
    return SolusMoEForCausalLM(moe_cfg)  # session-scoped moe_model fixture


def _inp(vocab=5000, batch=2, seq=8):
    return torch.randint(0, vocab, (batch, seq))


# ── Forward ──────────────────────────────────────────────────────────────
def test_forward_logits_shape(moe_model):
    inp = _inp()
    out = moe_model(inp)
    assert out["logits"].shape == (inp.shape[0], inp.shape[1], 5000)


def test_forward_loss_is_scalar(moe_model):
    inp = _inp()
    lbl = torch.randint(0, 5000, inp.shape)
    out = moe_model(inp, labels=lbl)
    assert out["loss"].numel() == 1


def test_forward_loss_positive(moe_model):
    inp = _inp()
    lbl = torch.randint(0, 5000, inp.shape)
    out = moe_model(inp, labels=lbl)
    assert out["loss"].item() > 0


def test_forward_no_labels_no_loss(moe_model):
    inp = _inp()
    out = moe_model(inp)
    assert out.get("loss") is None


def test_forward_no_nan(moe_model):
    inp = _inp()
    out = moe_model(inp)
    assert torch.isfinite(out["logits"]).all()


# ── Architecture ─────────────────────────────────────────────────────────
def test_moe_layer_count(moe_model, moe_cfg):
    assert moe_model.num_moe_layers == len(moe_cfg.moe_layer_indices)


def test_dense_layer_count(moe_model, moe_cfg):
    assert moe_model.num_dense_layers == moe_cfg.num_hidden_layers - len(moe_cfg.moe_layer_indices)


def test_tied_embeddings(moe_model):
    assert moe_model.lm_head.weight is moe_model.embed_tokens.weight


def test_total_layers_match_cfg(moe_model, moe_cfg):
    n = len(moe_model.dense_layers) + len(moe_model.moe_layers)
    assert n == moe_cfg.num_hidden_layers


# ── Gradient flow ────────────────────────────────────────────────────────
def test_gradient_flow(moe_model):
    inp = _inp(batch=1, seq=4)
    moe_model(input_ids=inp)["logits"].sum().backward()
    any_grad = any(
        p.grad is not None and p.grad.abs().max().item() > 0
        for p in moe_model.parameters()
    )
    assert any_grad, "no gradients flowed"


def test_loss_grad_flow(moe_model):
    inp  = _inp(batch=1, seq=4)
    lbl  = torch.randint(0, 5000, inp.shape)
    out  = moe_model(inp, labels=lbl)
    out["loss"].backward()
    any_grad = any(
        p.grad is not None and p.grad.abs().max().item() > 0
        for p in moe_model.parameters()
    )
    assert any_grad, "loss gradient did not flow"


# ── Generate (deterministic at temp=0) ───────────────────────────────────
def test_generate_output_longer_than_input(moe_model):
    inp = _inp(batch=1, seq=4)
    out = moe_model.generate(inp, max_new_tokens=3, temperature=0.0, top_p=1.0)
    assert out.shape[-1] >= 7


def test_generate_max_new_tokens_respected(moe_model):
    inp = _inp(batch=1, seq=4)
    out = moe_model.generate(inp, max_new_tokens=5, temperature=0.0, top_p=1.0)
    assert out.shape[-1] == 9


def test_generate_modes_char(moe_model):
    inp = _inp(batch=1, seq=2)
    out = moe_model.generate(inp, max_new_tokens=4, temperature=0.0, top_p=1.0)
    assert out.shape[-1] == 6


def test_forward_mixed_dense_moe_layers(moe_model):
    """Tokens see different computation graphs in MoE vs dense layers."""
    inp   = _inp(batch=2, seq=8)
    lbl   = torch.randint(0, 5000, inp.shape)
    out   = moe_model(inp, labels=lbl)
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["logits"]).all()
    assert out["logits"].shape[:-1] == inp.shape