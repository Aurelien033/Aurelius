"""DreamBank ablation harness — proves bank-on vs bank-off is measurable."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.memory.hlm_bank import HLMPreferenceBank, HLMPreferenceBankConfig, HLMPreferenceWrite
from src.model.amc_transformer import AMCTransformer, AMCTransformerConfig


@dataclass
class AblationResult:
    mean_logit_delta: float
    alpha_mean: float
    alpha_std: float
    confidence_mean: float
    bank_fill: int
    empty_bank: bool

    def to_dict(self) -> dict:
        return {
            "mean_logit_delta": self.mean_logit_delta,
            "alpha_mean": self.alpha_mean,
            "alpha_std": self.alpha_std,
            "confidence_mean": self.confidence_mean,
            "bank_fill": self.bank_fill,
            "empty_bank": self.empty_bank,
        }


def run_ablation(
    model: AMCTransformer,
    input_ids: torch.Tensor,
    bank: HLMPreferenceBank | None = None,
) -> AblationResult:
    """Run the same input with and without the bank, report delta."""
    out_base = model(input_ids)

    if bank is None:
        bank_empty = HLMPreferenceBank(HLMPreferenceBankConfig(
            bank_dim=model.config.hlm_bank_dim or model.config.kv_lrank,
        ))
        bank = bank_empty

    out_bank = model(input_ids, preference_bank=bank)

    delta = (out_base.logits - out_bank.logits).abs().mean().item()

    alpha_mean = 0.0
    alpha_std = 0.0
    conf_mean = 0.0
    if out_bank.bank_alpha is not None:
        alpha_mean = float(out_bank.bank_alpha.mean().item())
        alpha_std = float(out_bank.bank_alpha.std().item())
    if out_bank.bank_confidence is not None:
        conf_mean = float(out_bank.bank_confidence.mean().item())

    filled = int(bank.strengths.gt(0).sum().item()) if hasattr(bank, "strengths") else 0

    return AblationResult(
        mean_logit_delta=delta,
        alpha_mean=alpha_mean,
        alpha_std=alpha_std,
        confidence_mean=conf_mean,
        bank_fill=filled,
        empty_bank=bank.is_empty(),
    )
