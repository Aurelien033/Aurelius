"""Tests for is_router_param — verifies prefix-based MoE routing classification."""

from __future__ import annotations

from src.training.trainer import TrainConfig, is_router_param


def _cfg(moe_enabled: bool = True, n_layers: int = 4) -> TrainConfig:
    return TrainConfig(model_moe_enabled=moe_enabled, model_n_layers=n_layers)


class TestIsRouterParam:
    def test_attention_proj_not_router_when_moe_enabled(self):
        cfg = _cfg(moe_enabled=True)
        assert not is_router_param("model.layers.0.attn.out_proj.weight", cfg)

    def test_attention_proj_not_router_when_moe_disabled(self):
        cfg = _cfg(moe_enabled=False)
        assert not is_router_param("model.layers.0.attn.out_proj.weight", cfg)

    def test_ffn_router_weight_classified_when_moe_enabled(self):
        cfg = _cfg(moe_enabled=True, n_layers=4)
        assert is_router_param("model.layers.2.ffn.router.weight", cfg)

    def test_ffn_router_bias_classified_when_moe_enabled(self):
        cfg = _cfg(moe_enabled=True, n_layers=4)
        assert is_router_param("model.layers.0.ffn.router.bias", cfg)

    def test_ffn_router_not_classified_when_moe_disabled(self):
        cfg = _cfg(moe_enabled=False)
        assert not is_router_param("model.layers.0.ffn.router.weight", cfg)

    def test_out_of_range_layer_not_classified(self):
        cfg = _cfg(moe_enabled=True, n_layers=2)
        assert not is_router_param("model.layers.5.ffn.router.weight", cfg)
