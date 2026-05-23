"""AMC Tier-1 SSM working memory layer (Mamba-2 + surprise + gates).

Wraps :class:`Mamba2Block` with surprise scoring (detached from model gradient),
differentiable decay/erase/write gates, and :class:`AMCMemoryBlock` emission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.memory.amc_runtime_cache import AMCMemoryBlock, TrustState
from src.memory.amc_tensor_api import (
    AdmissionAction,
    AMCReadResult,
    AMCTensorState,
    AMCWriteDecision,
    MemoryTier,
)
from src.model.amc_gates import AMCGateController
from src.model.amc_surprise import SurpriseHead, SurpriseHeadConfig
from src.model.mamba2_block import Mamba2Block, Mamba2Config


@dataclass
class AMCSSMConfig:
    """Configuration for :class:`AMCSSMLayer`."""

    d_model: int
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    surprise_head_hidden: int | None = None
    gate_hidden: int | None = None
    surprise_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.surprise_head_hidden is None:
            self.surprise_head_hidden = max(1, self.d_model // 4)
        if self.gate_hidden is None:
            self.gate_hidden = self.d_state

    def to_mamba_config(self) -> Mamba2Config:
        return Mamba2Config(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            headdim=self.headdim,
        )


@dataclass
class AMCForwardOutput:
    """Forward bundle from :class:`AMCSSMLayer`."""

    hidden: torch.Tensor
    tensor_state: AMCTensorState
    memory_block: AMCMemoryBlock
    surprise_scores: torch.Tensor


class AMCSSMLayer(nn.Module):
    """Per-layer SSM block implementing AMC Tier-1 working memory."""

    def __init__(self, config: AMCSSMConfig, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.ssm = Mamba2Block(config.to_mamba_config())

        self.surprise_head = SurpriseHead(
            SurpriseHeadConfig(
                d_model=config.d_model,
                hidden_dim=config.surprise_head_hidden or 0,
            )
        )

        gate_out = config.gate_hidden
        if gate_out is None:
            raise ValueError("gate_hidden must be set")
        self.gates = AMCGateController(
            d_model=config.d_model,
            d_state=gate_out,
        )

        self._last_tensor_state: AMCTensorState | None = None
        self._last_block: AMCMemoryBlock | None = None
        self._observed_events = 0
        self._stored_events = 0

    def forward(
        self,
        x: torch.Tensor,
        *,
        step: int = 0,
        prev_state: torch.Tensor | None = None,
        return_state: bool = False,
    ):
        """Run SSM forward, apply gates, emit AMC memory artifacts."""
        ssm_out, ssm_state = self.ssm(
            x,
            step=step,
            prev_state=prev_state,
            return_state=True,
        )

        surprise = self.surprise_head(x.detach())
        mean_surprise = float(surprise[:, -1].mean().item()) if surprise.numel() else 0.0

        x_last = x[:, -1, :]
        gate_dict = self.gates(x_last)
        decay = gate_dict["decay"]
        erase = gate_dict["erase"]
        write = gate_dict["write"]

        h = ssm_state["ssm_state"]
        gated_state = self.gates.apply_to_state(h, gate_dict)
        write_g = self.gates._broadcast_gate(write, h)
        gated_state = gated_state + write_g
        ssm_state["ssm_state"] = gated_state
        self.ssm.set_state(gated_state.detach())

        # Differentiable bridge: gate-modulated state feeds back into hidden (training path).
        state_delta = gated_state - h
        gate_feedback = state_delta.mean(dim=(1, 2, 3)).view(-1, 1, 1)
        ssm_out = ssm_out + gate_feedback * 1e-2

        block = AMCMemoryBlock(
            block_id=f"amc_l{self.layer_index}_s{step}",
            tokens=tuple(range(8)),
            tier=1,
            trust_state=TrustState.UNVERIFIED,
            provenance=f"ssm_layer_{self.layer_index}",
            salience=mean_surprise,
            surprise_score=mean_surprise,
            kv_ref=ssm_state["ssm_state"].detach(),
            metadata={
                "gates": {
                    "decay_mean": float(decay.mean().item()),
                    "erase_mean": float(erase.mean().item()),
                    "write_mean": float(write.mean().item()),
                },
                "step": step,
                "layer_index": self.layer_index,
            },
        )

        tensor_state = AMCTensorState(
            layer_index=self.layer_index,
            token_count=step + x.shape[1],
            kvs=(ssm_state["ssm_state"].detach(),),
            rms_norm_stats=None,
            dtype=x.dtype,
            device_index=x.device.index if x.device.type == "cuda" else None,
            metadata={
                "gates": {
                    "decay": decay.detach(),
                    "erase": erase.detach(),
                    "write": write.detach(),
                },
            },
        )

        amc_out = AMCForwardOutput(
            hidden=ssm_out,
            tensor_state=tensor_state,
            memory_block=block,
            surprise_scores=surprise,
        )
        self._last_tensor_state = tensor_state
        self._last_block = block

        if return_state:
            return amc_out, ssm_state
        return amc_out

    def reset_state(self) -> None:
        self.ssm.reset_state()
        self._last_tensor_state = None
        self._last_block = None

    def get_state(self) -> torch.Tensor | None:
        """Raw Mamba-2 recurrent state ``(B, nheads, headdim, d_state)`` — no transforms."""
        return self.ssm.get_state()

    def set_state(self, state: torch.Tensor) -> None:
        self.ssm.set_state(state)

    # ── AMCLayerMemory (Tier-1 protocol surface) ───────────────────────────

    def read(
        self,
        layer_index: int,
        step: int,
        *,
        seq_len: int,
    ) -> AMCReadResult:
        if layer_index != self.layer_index:
            raise ValueError(
                f"layer_index {layer_index} != this layer {self.layer_index}"
            )
        if self._last_tensor_state is None:
            raise RuntimeError("no tensor state; run forward first")
        return AMCReadResult(
            token_count=self._last_tensor_state.token_count,
            state=self._last_tensor_state,
            metadata={"seq_len": seq_len, "step": step},
        )

    def write(
        self,
        state: AMCTensorState,
        *,
        step: int,
        surprise: float = 0.0,
    ) -> AMCWriteDecision:
        self._observed_events += 1
        admitted = surprise >= self.config.surprise_threshold
        if admitted:
            self._stored_events += 1
            self._last_tensor_state = state
        return AMCWriteDecision(
            admitted=admitted,
            action=AdmissionAction.ALLOW if admitted else AdmissionAction.SKIP,
            reason="surprise gate" if admitted else "below surprise threshold",
            surprise_score=surprise,
            surprise_adjusted=surprise if admitted else 0.0,
            tier2_write_promoted=False,
        )

    def write_observation(
        self,
        layer_index: int,
        step: int,
        content: str,
        *,
        surprise: float,
        importance: float | None = None,
    ) -> AMCWriteDecision:
        if layer_index != self.layer_index:
            raise ValueError(
                f"layer_index {layer_index} != this layer {self.layer_index}"
            )
        _ = (step, content, importance)
        return self.write(
            self._last_tensor_state
            if self._last_tensor_state is not None
            else AMCTensorState(
                layer_index=self.layer_index,
                token_count=0,
                kvs=(),
            ),
            step=step,
            surprise=surprise,
        )

    def read_memory(
        self,
        query: str,
        *,
        tier: MemoryTier = MemoryTier.TIER_2,
        limit: int = 5,
        layer_index: int | None = None,
        step: int | None = None,
    ) -> list[dict[str, Any]]:
        _ = (query, tier, limit, layer_index, step)
        if self._last_block is None:
            return []
        return [
            {
                "content": self._last_block.provenance,
                "source": self._last_block.block_id,
                "trust_level": str(self._last_block.trust_state),
            }
        ]

    def consolidate(
        self,
        *,
        tiers: tuple[MemoryTier, ...] = (MemoryTier.TIER_3,),
        max_entries: int | None = None,
    ) -> dict[str, Any]:
        _ = (tiers, max_entries)
        return {
            "promoted": 0,
            "quarantined": 0,
            "expired_pruned": 0,
            "errors": [],
        }

    def reset(self, layer_index: int | None = None) -> None:
        if layer_index is not None and layer_index != self.layer_index:
            return
        self.reset_state()
        self._observed_events = 0
        self._stored_events = 0

    def stats(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "episodic_entries": 0,
            "surprise_threshold": self.config.surprise_threshold,
            "observed_events": self._observed_events,
            "stored_events": self._stored_events,
        }


__all__ = [
    "AMCSSMConfig",
    "AMCSSMLayer",
    "AMCForwardOutput",
]
