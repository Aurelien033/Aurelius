"""Tests for AMC decay/erase/write gate networks."""

from __future__ import annotations

import torch

from src.memory.amc_update_contract import validate_gate
from src.model.amc_gates import AMCGateController, GateNetwork


def test_gate_network_output_range() -> None:
    net = GateNetwork(d_model=64, d_out=32)
    x = torch.randn(4, 64)
    out = net(x)
    assert out.shape == (4, 32)
    assert (out >= 0).all() and (out <= 1).all()


def test_gate_controller_returns_three() -> None:
    ctrl = AMCGateController(d_model=64, d_state=32)
    x = torch.randn(4, 64)
    gates = ctrl(x)
    assert set(gates.keys()) == {"decay", "erase", "write"}
    for gate in gates.values():
        assert gate.shape == (4, 32)


def test_gate_telemetry() -> None:
    ctrl = AMCGateController(d_model=64, d_state=32)
    x = torch.randn(4, 64)
    tel = ctrl.telemetry(x)
    assert all(
        k in tel
        for k in ["decay_mean", "erase_mean", "write_mean", "erase_write_correlation"]
    )


def test_apply_to_state_no_input() -> None:
    ctrl = AMCGateController(d_model=64, d_state=32)
    x = torch.randn(4, 64)
    gates = ctrl(x)
    state = torch.randn(4, 32)
    new_state = ctrl.apply_to_state(state, gates)
    assert new_state.shape == state.shape


def test_apply_to_state_with_input() -> None:
    ctrl = AMCGateController(d_model=64, d_state=32)
    x = torch.randn(4, 64)
    gates = ctrl(x)
    state = torch.randn(4, 32)
    new_input = torch.randn(4, 32)
    new_state = ctrl.apply_to_state(state, gates, new_input)
    assert new_state.shape == state.shape


def test_apply_to_state_broadcasts_to_3d() -> None:
    ctrl = AMCGateController(d_model=64, d_state=8)
    x = torch.randn(2, 64)
    gates = ctrl(x)
    state = torch.randn(2, 8, 16)
    new_state = ctrl.apply_to_state(state, gates)
    assert new_state.shape == (2, 8, 16)


def test_ewm_compatibility() -> None:
    ctrl = AMCGateController(d_model=64, d_state=32)
    x = torch.randn(2, 64)
    gates = ctrl(x)
    for name, gate in gates.items():
        validate_gate(gate.mean().item(), name=name)


def test_gates_have_gradients() -> None:
    ctrl = AMCGateController(d_model=64, d_state=32)
    x = torch.randn(2, 64, requires_grad=True)
    gates = ctrl(x)
    loss = gates["decay"].sum() + gates["erase"].sum() + gates["write"].sum()
    loss.backward()
    for name, param in ctrl.named_parameters():
        if "weight" in name:
            assert param.grad is not None and param.grad.abs().sum() > 0, f"no grad on {name}"
