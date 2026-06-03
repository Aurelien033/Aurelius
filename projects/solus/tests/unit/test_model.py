"""Solus-7B — unit tests: SolusForCausalLM."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import math
import pytest
import torch
from src.modeling.config import SolusConfig
from src.modeling.model import SolusForCausalLM


@pytest.fixture(scope="session")
def cfg():
    return SolusConfig(
        hidden_size=1024,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=1,
        intermediate_size=4096,
        max_position_embeddings=2048,
        vocab_size=10000,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_dropout=0.0,
    )


@pytest.fixture(scope="session")
def model(cfg):
    return SolusForCausalLM(cfg)


def _inp(vocab=10000, batch=2, seq=8):
    return torch.randint(0, vocab, (batch, seq))


def test_forward_logits_shape(model):
    inp = _inp()
    out = model(inp)
    assert out["logits"].shape == (inp.shape[0], inp.shape[1], model.cfg.vocab_size)


def test_forward_loss_is_scalar(model):
    inp = _inp(); lbl = torch.randint(0, 10000, inp.shape)
    out = model(inp, labels=lbl)
    assert out["loss"].numel() == 1


def test_forward_loss_positive(model):
    inp = _inp(); lbl = torch.randint(0, 10000, inp.shape)
    out = model(inp, labels=lbl)
    assert out["loss"].item() > 0


def test_forward_no_labels_no_loss(model):
    inp = _inp()
    out = model(inp)
    assert out.get("loss") is None


def test_generate_output_longer_than_input(model):
    inp = _inp(batch=1, seq=4)
    out = model.generate(inp, max_new_tokens=3, temperature=0.0, top_p=1.0)
    assert out.shape[-1] >= 4 + 3


def test_generate_max_new_tokens_respected(model):
    inp = _inp(batch=1, seq=4)
    out = model.generate(inp, max_new_tokens=5, temperature=0.0, top_p=1.0)
    assert out.shape[-1] == 4 + 5


def test_gradient_flow(model):
    inp = _inp(batch=2, seq=8)
    model(input_ids=inp)["logits"].sum().backward()
    any_grad = any(
        p.grad is not None and p.grad.abs().max().item() > 0
        for p in model.parameters()
    )
    assert any_grad, "no gradients flowed"
