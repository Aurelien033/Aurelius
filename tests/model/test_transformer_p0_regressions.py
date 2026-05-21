from __future__ import annotations

import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.moe import SparseMoELayer
from src.model.transformer import TransformerBlock, _apply_top_p_filter
from src.training.trainer import TrainConfig


class _ZeroAttention(nn.Module):
    def forward(self, x, freqs_cis, mask=None, past_kv=None):
        return torch.zeros_like(x), None


class _ConstantMoE(SparseMoELayer):
    def __init__(self, value: float) -> None:
        nn.Module.__init__(self)
        self.value = value

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.full_like(x, self.value), x.new_tensor(0.25)


def test_sparse_moe_block_preserves_residual_stream() -> None:
    cfg = AureliusConfig(
        d_model=4,
        n_layers=1,
        n_heads=1,
        n_kv_heads=1,
        head_dim=4,
        d_ff=8,
        vocab_size=16,
        max_seq_len=8,
        moe_enabled=True,
        moe_every_n_layers=1,
        moe_num_experts=2,
        moe_top_k=1,
    )
    block = TransformerBlock(cfg, layer_idx=0, n_layers=1)
    block.attn = _ZeroAttention()
    block.attn_norm = nn.Identity()
    block.ffn_norm = nn.Identity()
    block.ffn = _ConstantMoE(2.0)

    x = torch.full((1, 3, cfg.d_model), 3.0)
    out, _kv, aux_loss = block(x, freqs_cis=torch.empty(0), mask=None, past_kv=None)

    assert torch.allclose(out, torch.full_like(x, 5.0))
    assert aux_loss.item() == 0.25


def test_top_p_filter_keeps_highest_probability_nucleus() -> None:
    logits = torch.log(torch.tensor([[0.40, 0.35, 0.15, 0.10]], dtype=torch.float32))

    filtered = _apply_top_p_filter(logits, top_p=0.75)

    assert torch.isfinite(filtered[0, 0])
    assert torch.isfinite(filtered[0, 1])
    assert torch.isneginf(filtered[0, 2])
    assert torch.isneginf(filtered[0, 3])


def test_train_config_vocab_default_matches_model_config() -> None:
    assert TrainConfig().model_vocab_size == AureliusConfig().vocab_size
