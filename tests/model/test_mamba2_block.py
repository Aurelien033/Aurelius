"""Tests for :class:`src.model.mamba2_block.Mamba2Block`.

Covers:
  - Initialization & config math
  - Forward shapes for several (B, L, D) configurations
  - Determinism under a fixed seed
  - return_state=True path
  - State continuation: chunked forward == single forward
  - reset_state + get_state/set_state
  - Gradient flow to A_log / dt_bias / D / conv1d
  - Various batch sizes
  - Causality: zeroing future tokens does not affect earlier outputs
  - Runs on CPU (and CUDA when available)
"""
from __future__ import annotations

import random

import pytest
import torch
from torch import nn

from src.model.mamba2_block import Mamba2Block, Mamba2Config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_config(**overrides) -> Mamba2Config:
    defaults = dict(
        d_model=64,
        d_state=32,
        d_conv=4,
        expand=2,
        headdim=32,
        ngroups=1,
    )
    defaults.update(overrides)
    return Mamba2Config(**defaults)


def _seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestMamba2Config:
    def test_d_inner_and_nheads_defaults(self) -> None:
        cfg = _small_config()
        assert cfg.d_inner == cfg.d_model * cfg.expand == 128
        assert cfg.nheads == cfg.d_inner // cfg.headdim == 4

    def test_d_inner_custom_expand(self) -> None:
        cfg = _small_config(d_model=128, expand=4)
        assert cfg.d_inner == 512

    def test_invalid_d_model_rejected(self) -> None:
        with pytest.raises(ValueError):
            _small_config(d_model=0)

    def test_invalid_headdim_rejected(self) -> None:
        # d_inner=128, headdim=7 does not divide evenly.
        with pytest.raises(ValueError, match="divisible"):
            Mamba2Config(d_model=64, d_state=16, expand=2, headdim=7)

    def test_invalid_dt_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            _small_config(dt_min=0.5, dt_max=0.1)

    def test_invalid_A_init_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            _small_config(A_init_range=(5.0, 1.0))


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_construction(self) -> None:
        _seed(0)
        model = Mamba2Block(_small_config())
        assert isinstance(model.in_proj, nn.Linear)
        assert isinstance(model.conv1d, nn.Conv1d)
        assert isinstance(model.out_proj, nn.Linear)
        assert model.A_log.shape == (4,)
        assert model.D.shape == (4,)
        assert model.dt_bias.shape == (4,)

    def test_A_log_initialized_positive_before_negation(self) -> None:
        _seed(1)
        model = Mamba2Block(_small_config(A_init_range=(1.0, 16.0)))
        # A = -exp(A_log) should be negative.
        A = -torch.exp(model.A_log)
        assert (A < 0).all(), f"A should be negative, got {A}"


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

class TestForwardShape:
    @pytest.mark.parametrize("B,L,D", [(1, 1, 64), (2, 8, 64),
                                        (1, 64, 64), (4, 16, 128)])
    def test_output_shape(self, B: int, L: int, D: int) -> None:
        _seed(0)
        cfg = _small_config(d_model=D)
        model = Mamba2Block(cfg)
        x = torch.randn(B, L, D)
        out = model(x)
        assert out.shape == (B, L, D)

    def test_forward_deterministic_with_seed(self) -> None:
        _seed(42)
        cfg = _small_config()
        model = Mamba2Block(cfg)
        x = torch.randn(2, 8, cfg.d_model)
        a = model(x)
        b = model(x)
        assert torch.allclose(a, b, atol=1e-6), "two calls on same input differ"


class TestReturnState:
    def test_return_state_is_tuple_with_correct_keys(self) -> None:
        _seed(0)
        cfg = _small_config()
        model = Mamba2Block(cfg)
        x = torch.randn(2, 8, cfg.d_model)
        out = model(x, return_state=True)
        assert isinstance(out, tuple) and len(out) == 2
        tensor, state = out
        assert tensor.shape == (2, 8, cfg.d_model)
        assert "ssm_state" in state
        assert "step" in state
        assert state["ssm_state"].shape == (2, cfg.nheads, cfg.headdim, cfg.d_state)

    def test_step_advances_by_seq_len(self) -> None:
        _seed(0)
        cfg = _small_config()
        model = Mamba2Block(cfg)
        x = torch.randn(1, 12, cfg.d_model)
        _, state = model(x, step=3, return_state=True)
        assert state["step"] == 3 + 12


# ---------------------------------------------------------------------------
# State continuation — the key correctness check
# ---------------------------------------------------------------------------

class TestStateContinuation:
    def test_chunked_forward_matches_full(self) -> None:
        """Run (B=1, L=16) full. Then run L=8, take state, run L=8 with
        ``prev_state``. The concatenated output must match the full
        output within tolerance."""
        _seed(7)
        cfg = _small_config()
        model = Mamba2Block(cfg)
        model.eval()

        x = torch.randn(1, 16, cfg.d_model)
        with torch.no_grad():
            full_out = model(x)

            # First half.
            first_half, state = model(x[:, :8], step=0, return_state=True)
            # Second half using the cached state.
            second_half = model(x[:, 8:], step=8,
                                prev_state=state["ssm_state"])

        combined = torch.cat([first_half, second_half], dim=1)
        assert combined.shape == full_out.shape
        # Sequential scan should reproduce exactly (no parallel-scan
        # numerical error). Allow a small tolerance for float ops.
        torch.testing.assert_close(combined, full_out,
                                   rtol=1e-4, atol=1e-5,
                                   msg="chunked != full — state continuity broken")


# ---------------------------------------------------------------------------
# Reset / introspection
# ---------------------------------------------------------------------------

class TestStateIntrospection:
    def test_get_state_initially_none(self) -> None:
        model = Mamba2Block(_small_config())
        assert model.get_state() is None

    def test_set_then_get_roundtrip(self) -> None:
        cfg = _small_config()
        model = Mamba2Block(cfg)
        state = torch.zeros(2, cfg.nheads, cfg.headdim, cfg.d_state)
        model.set_state(state)
        out = model.get_state()
        assert out is not None
        assert torch.equal(out, state)

    def test_set_state_wrong_shape_raises(self) -> None:
        cfg = _small_config()
        model = Mamba2Block(cfg)
        bad = torch.zeros(2, 1, 1, 1)
        with pytest.raises(ValueError):
            model.set_state(bad)

    def test_reset_clears_cached_state(self) -> None:
        _seed(0)
        cfg = _small_config()
        model = Mamba2Block(cfg)
        x = torch.randn(1, 4, cfg.d_model)
        model(x)  # populates self._state
        assert model.get_state() is not None
        model.reset_state()
        assert model.get_state() is None

    def test_state_continues_across_steps_when_fed_back(self) -> None:
        _seed(0)
        cfg = _small_config()
        model = Mamba2Block(cfg)

        x = torch.randn(1, 4, cfg.d_model)
        _, state1 = model(x, step=0, return_state=True)
        _, state2 = model(x, step=4,
                          prev_state=state1["ssm_state"], return_state=True)

        s1 = state1["ssm_state"]
        s2 = state2["ssm_state"]
        assert not torch.allclose(s1, s2), "state did not evolve across steps"


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_gradients_reach_all_parameters(self) -> None:
        _seed(0)
        cfg = _small_config()
        model = Mamba2Block(cfg)
        x = torch.randn(2, 8, cfg.d_model)
        out = model(x)
        loss = out.sum()
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"no gradient on {name}"
            assert p.grad.abs().sum() > 0, f"zero gradient on {name}"

    def test_conv1d_has_gradient(self) -> None:
        _seed(0)
        cfg = _small_config()
        model = Mamba2Block(cfg)
        x = torch.randn(2, 8, cfg.d_model)
        loss = model(x).sum()
        loss.backward()
        assert model.conv1d.weight.grad is not None
        assert model.conv1d.weight.grad.norm() > 0


# ---------------------------------------------------------------------------
# Various batch sizes
# ---------------------------------------------------------------------------

class TestBatchSizes:
    @pytest.mark.parametrize("B", [1, 2, 4, 8])
    def test_batch_sizes_forward(self, B: int) -> None:
        _seed(0)
        cfg = _small_config()
        model = Mamba2Block(cfg)
        x = torch.randn(B, 8, cfg.d_model)
        out = model(x)
        assert out.shape == (B, 8, cfg.d_model)


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------

class TestCausality:
    def test_future_tokens_do_not_affect_past_outputs(self) -> None:
        """Zeroing future tokens must leave earlier outputs unchanged."""
        _seed(0)
        cfg = _small_config()
        model = Mamba2Block(cfg)
        model.eval()

        x = torch.randn(1, 16, cfg.d_model)
        with torch.no_grad():
            out_full = model(x)

            x_modified = x.clone()
            x_modified[:, 10:, :] = 0.0
            out_modified = model(x_modified)

        # Outputs at positions 0..9 (inclusive) must be identical because
        # those positions never see tokens 10+.
        torch.testing.assert_close(
            out_full[:, :10, :], out_modified[:, :10, :],
            rtol=1e-5, atol=1e-6,
            msg="causality violation: future tokens affected past outputs",
        )
        # Position 10+ should differ.
        assert not torch.allclose(out_full[:, 10:, :], out_modified[:, 10:, :],
                                  atol=1e-6), \
            "future positions unchanged despite zeroed input — scan bug"


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

class TestDevice:
    def test_runs_on_cpu(self) -> None:
        _seed(0)
        cfg = _small_config()
        model = Mamba2Block(cfg)
        x = torch.randn(1, 4, cfg.d_model, device="cpu")
        out = model(x)
        assert out.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
    def test_runs_on_cuda(self) -> None:
        _seed(0)
        cfg = _small_config()
        model = Mamba2Block(cfg).cuda()
        x = torch.randn(1, 4, cfg.d_model, device="cuda")
        out = model(x)
        assert out.device.type == "cuda"


# ---------------------------------------------------------------------------
# Smoke (the exact command from Tranched T00)
# ---------------------------------------------------------------------------

def test_tranched_smoke() -> None:
    """Reproduce the smoke snippet from the tranche doc."""
    _seed(0)
    cfg = Mamba2Config(d_model=64, d_state=32, d_conv=4, expand=2, headdim=32)
    model = Mamba2Block(cfg)
    x = torch.randn(2, 16, 64)
    out, state = model(x, return_state=True)
    assert out.shape == (2, 16, 64)
    assert list(state.keys()) == ["ssm_state", "step"]
    # Gradient check.
    loss = out.sum()
    loss.backward()
    assert model.conv1d.weight.grad is not None
    assert model.conv1d.weight.grad.norm() > 0
