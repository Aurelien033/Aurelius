"""Differentiable gate networks for AMC memory updates."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.memory.amc_update_contract import gate_correlation


class GateNetwork(nn.Module):
    """Single gate network: d_model → d_out with sigmoid output in [0, 1]."""

    def __init__(self, d_model: int, d_out: int, hidden: int = 0) -> None:
        super().__init__()
        hidden = hidden or max(32, d_model // 2)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_out),
            nn.Sigmoid(),
        )
        self.d_out = d_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AMCGateController(nn.Module):
    """Joint controller for decay, erase, and write gates."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        hidden: int = 0,
    ) -> None:
        super().__init__()
        self.decay = GateNetwork(d_model, d_state, hidden)
        self.erase = GateNetwork(d_model, d_state, hidden)
        self.write = GateNetwork(d_model, d_state, hidden)
        self.d_state = d_state

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "decay": self.decay(x),
            "erase": self.erase(x),
            "write": self.write(x),
        }

    @torch.no_grad()
    def telemetry(self, x: torch.Tensor) -> dict[str, float | None]:
        gates = self.forward(x)
        decay_mean = gates["decay"].mean().item()
        erase_mean = gates["erase"].mean().item()
        write_mean = gates["write"].mean().item()
        ew_corr = gate_correlation(erase_mean, write_mean)
        return {
            "decay_mean": decay_mean,
            "erase_mean": erase_mean,
            "write_mean": write_mean,
            "erase_write_correlation": ew_corr,
        }

    @staticmethod
    def _broadcast_gate(gate: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if gate.shape == state.shape:
            return gate
        if gate.dim() == 2 and state.dim() >= 2 and gate.shape[-1] == state.shape[-1]:
            view_shape = [gate.shape[0], *([1] * (state.dim() - 2)), gate.shape[1]]
            return gate.view(*view_shape)
        while gate.dim() < state.dim():
            gate = gate.unsqueeze(-1)
        return gate

    def apply_to_state(
        self,
        state: torch.Tensor,
        gates: dict[str, torch.Tensor],
        new_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        decay = self._broadcast_gate(gates["decay"], state)
        erase = self._broadcast_gate(gates["erase"], state)
        write = self._broadcast_gate(gates["write"], state)

        result = decay * state - erase * state
        if new_input is not None:
            write = self._broadcast_gate(write, result)
            result = result + write * new_input
        return result


__all__ = ["AMCGateController", "GateNetwork"]
