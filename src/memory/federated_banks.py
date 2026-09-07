"""Federated Memory Deltas — simulation harness for federated DreamBank deltas.

Demonstrates that federating DreamBank tensors (keys/values/strengths) at
the memory level is a tractable alternative to federated LLM fine-tuning,
with orders of magnitude lower communication cost.

This is an in-process simulator (no real network, no real devices). Real
DP privacy analysis and real device/communication infrastructure come later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.alignment.dreambank import DreamBankConfig, DreamBankController, DreamSeed
from src.memory.hlm_bank import HLMPreferenceBank, HLMPreferenceBankConfig


@dataclass(frozen=True)
class FederatedConfig:
    num_devices: int = 8
    rounds: int = 3
    local_cycles_per_round: int = 2
    dp_sigma: float = 0.0  # 0 = no DP noise; >0 adds Gaussian noise pre-aggregation
    seeds_per_device_per_cycle: int = 4
    max_writes_per_cycle: int = 8
    bank_size: int = 8
    bank_dim: int = 16
    seed: int = 0
    min_margin_for_write: float = 0.01

    def __post_init__(self) -> None:
        if self.num_devices <= 0:
            raise ValueError(f"num_devices must be > 0, got {self.num_devices}")
        if self.rounds <= 0:
            raise ValueError(f"rounds must be > 0, got {self.rounds}")
        if self.bank_size <= 0:
            raise ValueError(f"bank_size must be > 0, got {self.bank_size}")
        if self.bank_dim <= 0:
            raise ValueError(f"bank_dim must be > 0, got {self.bank_dim}")
        if self.dp_sigma < 0:
            raise ValueError(f"dp_sigma must be >= 0, got {self.dp_sigma}")


@dataclass
class FederationReport:
    num_devices: int
    rounds: int
    per_device_final_fill: list[int] = field(default_factory=list)
    per_device_final_mean_strength: list[float] = field(default_factory=list)
    isolated_mean_fill: float = 0.0
    isolated_mean_strength: float = 0.0
    federated_mean_fill: float = 0.0
    federated_mean_strength: float = 0.0
    delta_fill: float = 0.0
    delta_strength: float = 0.0
    total_comm_bytes_proxy: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "num_devices": self.num_devices,
            "rounds": self.rounds,
            "per_device_final_fill": list(self.per_device_final_fill),
            "per_device_final_mean_strength": list(self.per_device_final_mean_strength),
            "isolated_mean_fill": float(self.isolated_mean_fill),
            "isolated_mean_strength": float(self.isolated_mean_strength),
            "federated_mean_fill": float(self.federated_mean_fill),
            "federated_mean_strength": float(self.federated_mean_strength),
            "delta_fill": float(self.delta_fill),
            "delta_strength": float(self.delta_strength),
            "total_comm_bytes_proxy": self.total_comm_bytes_proxy,
        }


def _comm_bytes_per_round(num_devices: int, bank_size: int, bank_dim: int) -> int:
    """Approximate per-round communication: each device uploads keys, values, strengths.

    Uses fp32 sizing for simplicity. Real deployments would use fp16/bf16.
    """
    bytes_per_device = 4 * (2 * bank_size * bank_dim + bank_size)
    # Upload + broadcast-of-merged result back
    return num_devices * bytes_per_device * 2


def _device_generate_fn(device_seed: int):
    """Deterministic generate_fn seeded by device + prompt + temperature."""
    import hashlib

    def gen(prompt: str, temperature: float) -> str:
        h = hashlib.sha256(f"{device_seed}:{prompt}:{temperature}".encode()).hexdigest()[:16]
        return f"device{device_seed}_{h}_t{temperature}"

    return gen


def _device_score_fn(device_seed: int):
    """Deterministic score_fn that gives each device a slightly different taste."""
    import hashlib

    def score(prompt: str, response: str) -> float:
        h = int(hashlib.sha256(f"{device_seed}:{response}".encode()).hexdigest()[:8], 16)
        # Bias by device seed so each device has slightly different preference distribution
        bias = (device_seed % 17) / 100.0
        return min(1.0, (h % 1000) / 1000.0 + bias)

    return score


def _device_embed_fn(dim: int):
    def embed(text: str) -> torch.Tensor:
        seed = sum(ord(c) for c in text) % (2**31)
        torch.manual_seed(seed)
        return torch.randn(dim)

    return embed


class FederatedBankSimulator:
    """In-process simulator for federated DreamBank aggregation."""

    def __init__(self, config: FederatedConfig | None = None) -> None:
        self.cfg = config or FederatedConfig()

    def _make_local_bank(self) -> HLMPreferenceBank:
        return HLMPreferenceBank(
            HLMPreferenceBankConfig(
                bank_size=self.cfg.bank_size,
                bank_dim=self.cfg.bank_dim,
            )
        )

    def _run_local_cycles(self, device_id: int, bank: HLMPreferenceBank) -> None:
        ctrl = DreamBankController(
            bank,
            DreamBankConfig(
                min_margin=self.cfg.min_margin_for_write,
                max_writes_per_cycle=self.cfg.max_writes_per_cycle,
            ),
        )
        generator = _device_generate_fn(device_id)
        scorer = _device_score_fn(device_id)
        embedder = _device_embed_fn(self.cfg.bank_dim)
        for cycle in range(self.cfg.local_cycles_per_round):
            seeds = [
                DreamSeed(f"d{device_id}_c{cycle}_s{i}")
                for i in range(self.cfg.seeds_per_device_per_cycle)
            ]
            ctrl.run_cycle(
                seeds=seeds,
                generate_fn=generator,
                score_fn=scorer,
                embed_fn=embedder,
            )

    def _aggregate_fedavg(
        self, banks: list[HLMPreferenceBank]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """FedAvg aggregation: element-wise mean of each device's tensor."""
        keys = torch.stack([b.keys.clone() for b in banks]).mean(dim=0)
        values = torch.stack([b.values.clone() for b in banks]).mean(dim=0)
        strengths = torch.stack([b.strengths.clone() for b in banks]).mean(dim=0)
        return keys, values, strengths

    def _add_dp_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.cfg.dp_sigma <= 0:
            return tensor
        with torch.no_grad():
            noise = torch.randn_like(tensor) * self.cfg.dp_sigma
            return tensor + noise

    def _download_aggregated(
        self,
        bank: HLMPreferenceBank,
        keys: torch.Tensor,
        values: torch.Tensor,
        strengths: torch.Tensor,
    ) -> None:
        """Overwrite local bank tensors with aggregated version."""
        with torch.no_grad():
            bank.keys.copy_(keys)
            bank.values.copy_(values)
            bank.strengths.copy_(strengths)

    def _run_once(self, *, federate: bool) -> list[HLMPreferenceBank]:
        """Run the simulation; if federate=False, devices run in isolation."""
        torch.manual_seed(self.cfg.seed)
        banks = [self._make_local_bank() for _ in range(self.cfg.num_devices)]

        for round_idx in range(self.cfg.rounds):
            # Local DreamBank cycles
            for d, bank in enumerate(banks):
                self._run_local_cycles(d, bank)

            if federate and self.cfg.num_devices >= 2:
                # "Upload" (with optional DP noise)
                uploaded_keys = torch.stack([self._add_dp_noise(b.keys.clone()) for b in banks])
                uploaded_values = torch.stack([self._add_dp_noise(b.values.clone()) for b in banks])
                uploaded_strengths = torch.stack(
                    [self._add_dp_noise(b.strengths.clone()) for b in banks]
                )
                # Server-side FedAvg over uploaded tensors
                merged_keys = uploaded_keys.mean(dim=0)
                merged_values = uploaded_values.mean(dim=0)
                merged_strengths = uploaded_strengths.mean(dim=0)
                # "Download"
                for bank in banks:
                    self._download_aggregated(bank, merged_keys, merged_values, merged_strengths)

        return banks

    def run(self) -> FederationReport:
        """Run federated and isolated baselines; report delta."""
        # --- ISOLATED baseline (no federation) ---
        isolated_banks = self._run_once(federate=False)
        isolated_fills = [int(b.strengths.gt(0).sum().item()) for b in isolated_banks]
        isolated_strengths = [float(b.strengths.mean().item()) for b in isolated_banks]
        isolated_mean_fill = sum(isolated_fills) / len(isolated_fills)
        isolated_mean_strength = sum(isolated_strengths) / len(isolated_strengths)

        # --- FEDERATED ---
        fed_banks = self._run_once(federate=True)
        fed_fills = [int(b.strengths.gt(0).sum().item()) for b in fed_banks]
        fed_strengths = [float(b.strengths.mean().item()) for b in fed_banks]
        fed_mean_fill = sum(fed_fills) / len(fed_fills)
        fed_mean_strength = sum(fed_strengths) / len(fed_strengths)

        return FederationReport(
            num_devices=self.cfg.num_devices,
            rounds=self.cfg.rounds,
            per_device_final_fill=fed_fills,
            per_device_final_mean_strength=fed_strengths,
            isolated_mean_fill=isolated_mean_fill,
            isolated_mean_strength=isolated_mean_strength,
            federated_mean_fill=fed_mean_fill,
            federated_mean_strength=fed_mean_strength,
            delta_fill=fed_mean_fill - isolated_mean_fill,
            delta_strength=fed_mean_strength - isolated_mean_strength,
            total_comm_bytes_proxy=_comm_bytes_per_round(
                self.cfg.num_devices, self.cfg.bank_size, self.cfg.bank_dim
            )
            * self.cfg.rounds,
        )
