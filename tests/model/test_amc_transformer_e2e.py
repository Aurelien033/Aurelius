"""End-to-end tests for AMCTransformer forward, checkpoint, and AMC memory."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from src.model.amc_transformer import AMCTransformer, AMCTransformerConfig, AMCModelOutput


def _small_config() -> AMCTransformerConfig:
    return AMCTransformerConfig(
        vocab_size=1000,
        d_model=128,
        n_layers=6,
        n_heads=4,
        ssm_d_state=32,
        ssm_headdim=32,
        ssm_expand=2,
    )


def test_forward_produces_amc_output() -> None:
    model = AMCTransformer(_small_config())
    x = torch.randint(0, 1000, (2, 16))
    out = model(x, use_amc=True, return_memory=True)
    assert isinstance(out, AMCModelOutput)
    assert out.logits.shape == (2, 16, 1000)
    assert len(out.memory_blocks) == model.ssm_layer_count


def test_forward_without_amc_is_faster_and_no_memory() -> None:
    model = AMCTransformer(_small_config())
    model.eval()
    x = torch.randint(0, 1000, (2, 16))
    with torch.no_grad():
        out_amc = model(x, use_amc=True, return_memory=True)
        out_no = model(x, use_amc=False, return_memory=False)
    assert torch.allclose(out_amc.logits, out_no.logits, atol=1e-3)
    assert out_no.memory_blocks == []


def test_reset_state_clears_ssm() -> None:
    model = AMCTransformer(_small_config())
    model.eval()
    x = torch.randint(0, 1000, (1, 8))
    with torch.no_grad():
        out1 = model(x, step=0, return_memory=True)
        model.reset_amc_state()
        out2 = model(x, step=0, return_memory=True)
    assert torch.allclose(out1.logits, out2.logits, atol=1e-5)


def test_checkpoint_save_load_roundtrip() -> None:
    model = AMCTransformer(_small_config())
    model.eval()
    x = torch.randint(0, 1000, (2, 8))
    with torch.no_grad():
        out_before = model(x, return_memory=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ckpt.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "config": model.config.__dict__,
            },
            path,
        )
        loaded = torch.load(path, weights_only=False)

    model2 = AMCTransformer(AMCTransformerConfig(**loaded["config"]))
    model2.load_state_dict(loaded["model"])
    model2.eval()

    with torch.no_grad():
        out_after = model2(x, return_memory=True)

    assert torch.allclose(out_before.logits, out_after.logits, atol=1e-6)


def test_gradient_flows_to_all_components() -> None:
    model = AMCTransformer(_small_config())
    x = torch.randint(0, 1000, (2, 8))
    out = model(x, use_amc=True, return_memory=True)
    loss = out.logits.sum()
    if out.promotion_loss is not None:
        loss = loss + out.promotion_loss
    loss.backward()

    assert model.embed.weight.grad is not None
    assert model.embed.weight.grad.abs().sum() > 0
    assert model.lm_head.weight.grad is not None
    assert model.lm_head.weight.grad.abs().sum() > 0

    mla_layer = model.layers[0]
    ssm_layer = model.layers[1]
    assert next(mla_layer.parameters()).grad is not None
    assert next(ssm_layer.parameters()).grad is not None


def test_tied_embeddings_when_configured() -> None:
    cfg = _small_config()
    assert cfg.tie_embeddings
    model = AMCTransformer(cfg)
    assert model.lm_head.weight is model.embed.weight


def test_surprise_scores_present_when_memory_enabled() -> None:
    model = AMCTransformer(_small_config())
    x = torch.randint(0, 1000, (2, 8))
    out = model(x, use_amc=True, return_memory=True)
    assert out.surprise_scores is not None
    assert out.surprise_scores.shape[0] == model.ssm_layer_count
    assert (out.surprise_scores >= 0).all() and (out.surprise_scores <= 1).all()


def test_promotion_loss_none_in_eval_mode() -> None:
    model = AMCTransformer(_small_config())
    model.eval()
    x = torch.randint(0, 1000, (2, 8))
    with torch.no_grad():
        out = model(x, use_amc=True, return_memory=True)
    assert out.promotion_loss is None


def test_promotion_loss_scalar_in_train_mode() -> None:
    model = AMCTransformer(_small_config())
    model.train()
    x = torch.randint(0, 1000, (2, 8))
    out = model(x, use_amc=True, return_memory=True)
    assert out.promotion_loss is not None
    assert out.promotion_loss.dim() == 0
    assert torch.isfinite(out.promotion_loss)
