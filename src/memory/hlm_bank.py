"""HLM Preference Bank — runtime-writable preference storage in MLA latent space.

A tiny bank of key/value pairs with strength decay, top-k cosine similarity read,
and no-grad upserts.  Stores keys/values/strength as registered buffers (not
trainable parameters).  Never stores raw prompts — only hashes and provenance tags.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class HLMPreferenceBankConfig:
    """Configuration for the preference bank."""

    bank_size: int = 14
    bank_dim: int = 64
    top_k: int = 4
    read_temperature: float = 8.0
    decay: float = 0.995
    min_strength: float = 1e-4
    write_momentum: float = 0.2
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.bank_size <= 0:
            raise ValueError(f"bank_size must be > 0, got {self.bank_size}")
        if self.bank_dim <= 0:
            raise ValueError(f"bank_dim must be > 0, got {self.bank_dim}")
        if self.top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {self.top_k}")
        if not 0.0 < self.decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {self.decay}")
        if self.min_strength < 0:
            raise ValueError(f"min_strength must be >= 0, got {self.min_strength}")


@dataclass(frozen=True)
class HLMPreferenceWrite:
    """A single bank write request."""

    key: torch.Tensor  # (..., bank_dim) or (bank_dim,)
    value: torch.Tensor  # same shape contract as key
    strength: float
    provenance: str = "dreambank"
    trust: str = "unverified"
    metadata_hash: str = ""


@dataclass(frozen=True)
class HLMPreferenceRead:
    """Result of a top-k bank read."""

    context: torch.Tensor  # (..., bank_dim)
    weights: torch.Tensor  # (..., top_k)
    indices: torch.Tensor  # (..., top_k)
    confidence: torch.Tensor  # (..., 1)


class HLMPreferenceBank(nn.Module):
    """In-process preference bank with deterministic top-k read and no-grad upsert.

    Bank slots are stored as registered buffers — not trainable parameters.
    Empty reads return zero context with zero confidence (never NaN).
    Upserts choose the first empty slot, or the lowest-strength / LRU slot when full.
    """

    def __init__(self, config: HLMPreferenceBankConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or HLMPreferenceBankConfig()
        dim = self.cfg.bank_dim
        size = self.cfg.bank_size

        # Buffers (not parameters) — persisted with state_dict but no gradients
        self.register_buffer("keys", torch.zeros(size, dim))
        self.register_buffer("values", torch.zeros(size, dim))
        self.register_buffer("strengths", torch.zeros(size))
        self.register_buffer(
            "access_time", torch.zeros(size)
        )  # monotonically-increasing LRU counter

        self._slot_counter: int = 0

        # Per-slot metadata (parallel lists, not tensors since strings aren't tensor-safe)
        self._provenances: list[str] = [""] * size
        self._trusts: list[str] = [""] * size
        self._metadata_hashes: list[str] = [""] * size

    # ── helpers ──────────────────────────────────────────────────────────────

    def _filled_mask(self) -> torch.Tensor:
        return self.strengths > 0.0

    def _normalize(self, t: torch.Tensor) -> torch.Tensor:
        """L2-normalise along last dimension, clamping denom to eps."""
        return t / (t.norm(dim=-1, keepdim=True) + self.cfg.eps)

    # ── public API ───────────────────────────────────────────────────────────

    def is_empty(self) -> bool:
        return not bool(self.strengths.gt(0).any().item())

    @torch.no_grad()
    def _next_access(self) -> float:
        self._slot_counter += 1
        return float(self._slot_counter)

    def read(self, query: torch.Tensor, *, top_k: int | None = None) -> HLMPreferenceRead:
        """Top-k cosine-similarity read; returns weighted context.

        Args:
            query: (..., bank_dim) — any batch shape, final dim must match bank_dim.
            top_k: override per-read; defaults to config.top_k.

        Returns:
            HLMPreferenceRead with context, weights, indices, confidence.
        """
        k = top_k or self.cfg.top_k
        filled = self._filled_mask()  # (size,)
        filled_count = filled.sum().item()

        if filled_count == 0:
            flat_shape = query.shape[:-1]
            return HLMPreferenceRead(
                context=torch.zeros(
                    *flat_shape, self.cfg.bank_dim, device=query.device, dtype=query.dtype
                ),
                weights=torch.zeros(*flat_shape, k, device=query.device, dtype=query.dtype),
                indices=torch.full((*flat_shape, k), -1, device=query.device, dtype=torch.long),
                confidence=torch.zeros(*flat_shape, 1, device=query.device, dtype=query.dtype),
            )

        # Flatten query for similarity: (B, D)
        orig_shape = query.shape
        q_flat = query.view(-1, self.cfg.bank_dim)  # (B, D)
        q_norm = self._normalize(q_flat)  # (B, D)

        # Only compare against filled slots
        k_keys = self.keys[filled]  # (filled, D)
        k_norm = self._normalize(k_keys)  # (filled, D)

        # Cosine sim: (B, filled)
        sim = torch.mm(q_norm, k_norm.t())
        sim = sim / self.cfg.read_temperature

        # Softmax weights per query
        weights = torch.softmax(sim, dim=-1)  # (B, filled)

        # Weighted value aggregation
        filled_vals = self.values[filled]  # (filled, D)
        context = torch.mm(weights, filled_vals)  # (B, D)

        # Top-k indices into filled slots
        actual_k = min(k, filled_count)
        top_weights, top_local_idx = torch.topk(weights, actual_k, dim=-1)
        # Map local indices back to global bank indices
        filled_global_idx = torch.where(filled)[0]  # (filled,)
        global_idx = filled_global_idx[top_local_idx]  # (B, actual_k)

        # Pad to exactly k if filled_count < k
        if actual_k < k:
            B = q_flat.shape[0]
            pad_w = torch.zeros(B, k - actual_k, device=query.device, dtype=query.dtype)
            pad_i = torch.full((B, k - actual_k), -1, device=query.device, dtype=torch.long)
            top_weights = torch.cat([top_weights, pad_w], dim=-1)
            global_idx = torch.cat([global_idx, pad_i], dim=-1)

        # Confidence: mean of top-k weights (excluding padding zeros)
        confidence = top_weights[:, :actual_k].mean(dim=-1, keepdim=True)  # (B, 1)

        # Reshape to original batch shape
        out_shape = (*orig_shape[:-1], self.cfg.bank_dim)
        ctx_out = context.view(*out_shape)
        w_out = top_weights.view(*orig_shape[:-1], -1)
        idx_out = global_idx.view(*orig_shape[:-1], -1)
        conf_out = confidence.view(*orig_shape[:-1], 1)

        return HLMPreferenceRead(
            context=ctx_out,
            weights=w_out,
            indices=idx_out,
            confidence=conf_out,
        )

    @torch.no_grad()
    def upsert(self, write: HLMPreferenceWrite) -> int:
        """Insert or update a bank slot. Returns the slot index written.

        Chooses first empty slot; if full, evicts lowest-strength / LRU slot.
        """
        key = write.key.view(self.cfg.bank_dim)
        value = write.value.view(self.cfg.bank_dim)

        filled = self._filled_mask()
        empty_slots = (~filled).nonzero(as_tuple=True)[0]

        if empty_slots.numel() > 0:
            slot = empty_slots[0].item()
        else:
            # Find lowest-strength slot (ties broken by oldest access_time = LRU)
            min_str_idx = self.strengths.argmin().item()
            slot = min_str_idx

        # Blend with momentum if slot already occupied
        if self.keys[slot].norm() > 0:
            mom = self.cfg.write_momentum
            self.keys[slot] = mom * key + (1 - mom) * self.keys[slot]
            self.values[slot] = mom * value + (1 - mom) * self.values[slot]
            self.strengths[slot] = mom * max(write.strength, 0.0) + (1 - mom) * self.strengths[slot]
        else:
            self.keys[slot] = key
            self.values[slot] = value
            self.strengths[slot] = max(write.strength, 0.0)

        self.access_time[slot] = self._next_access()
        self._provenances[slot] = write.provenance
        self._trusts[slot] = write.trust
        self._metadata_hashes[slot] = write.metadata_hash
        return slot

    @torch.no_grad
    def decay_(self, steps: int = 1) -> None:
        """Exponentially decay strengths; entries below min_strength are cleared."""
        factor = self.cfg.decay**steps
        self.strengths.mul_(factor)
        below = self.strengths < self.cfg.min_strength
        self.strengths[below] = 0.0

    @torch.no_grad()
    def consolidate_(self) -> None:
        """Remove entries below min_strength threshold."""
        weak = self.strengths < self.cfg.min_strength
        if weak.any():
            self.keys[weak] = 0.0
            self.values[weak] = 0.0
            self.strengths[weak] = 0.0

    def telemetry(self) -> dict[str, float | int]:
        return {
            "bank_size": self.cfg.bank_size,
            "bank_dim": self.cfg.bank_dim,
            "filled_slots": int(self.strengths.gt(0).sum().item()),
            "mean_strength": float(self.strengths.mean().item()),
            "max_strength": float(self.strengths.max().item()),
        }

    def export_state(self) -> dict[str, torch.Tensor | list[str] | int]:
        """Export state for serialization (no raw prompts, only tensors + hashes)."""
        cfg_dict = {
            "bank_size": self.cfg.bank_size,
            "bank_dim": self.cfg.bank_dim,
            "top_k": self.cfg.top_k,
            "read_temperature": self.cfg.read_temperature,
            "decay": self.cfg.decay,
            "min_strength": self.cfg.min_strength,
            "write_momentum": self.cfg.write_momentum,
            "eps": self.cfg.eps,
        }
        return {
            "keys": self.keys.clone(),
            "values": self.values.clone(),
            "strengths": self.strengths.clone(),
            "access_time": self.access_time.clone(),
            "config": cfg_dict,
            "metadata_hashes": list(self._metadata_hashes),
            "trusts": list(self._trusts),
            "provenances": list(self._provenances),
        }

    @classmethod
    def from_state(
        cls, state: dict[str, object], config: HLMPreferenceBankConfig | None = None
    ) -> HLMPreferenceBank:
        # Infer config from state if not provided
        if config is None and "config" in state:
            fields = HLMPreferenceBankConfig.__dataclass_fields__
            config = HLMPreferenceBankConfig(**{k: state["config"][k] for k in fields})
        bank = cls(config=config)
        bank.load_state_dict({k: v for k, v in state.items() if isinstance(v, torch.Tensor)})
        # Restore metadata
        if "metadata_hashes" in state:
            bank._metadata_hashes = list(state["metadata_hashes"])  # type: ignore[arg-type]
        if "trusts" in state:
            bank._trusts = list(state["trusts"])  # type: ignore[arg-type]
        if "provenances" in state:
            bank._provenances = list(state["provenances"])  # type: ignore[arg-type]
        return bank
