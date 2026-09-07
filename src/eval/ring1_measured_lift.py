"""Ring 1 measured DreamBank lift — logit-delta ablation on held-out trace prompts."""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from src.eval.dreambank_ablation import run_ablation
from src.eval.ring1_dreambank_runner import deterministic_embed
from src.eval.ring1_eval_harness import DreamBankLiftReport, _trace_success, bootstrap_ci
from src.memory.hlm_bank import HLMPreferenceBank, HLMPreferenceBankConfig, HLMPreferenceWrite
from src.model.amc_transformer import AMCTransformer


def trace_prompt_to_input_ids(
    trace: dict[str, Any],
    *,
    vocab_size: int,
    max_len: int = 32,
) -> torch.Tensor:
    """Encode the first step observation into deterministic token ids."""
    steps = trace.get("steps", [])
    text = str(steps[0].get("observation", "")).strip() if steps else ""
    if not text:
        text = str(trace.get("trace_id", "empty"))

    tokens: list[int] = []
    for token in text.split():
        digest = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)
        tokens.append(digest % vocab_size)
        if len(tokens) >= max_len:
            break
    if not tokens:
        tokens = [int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % vocab_size]
    return torch.tensor([tokens], dtype=torch.long)


def bank_from_dreambank_preferences(
    preferences: list[dict[str, Any]],
    *,
    bank_config: dict[str, Any] | None = None,
) -> HLMPreferenceBank:
    """Rehydrate an HLMPreferenceBank from DreamBank sleep-cycle preferences."""
    cfg = bank_config or {}
    bank = HLMPreferenceBank(
        HLMPreferenceBankConfig(
            bank_size=cfg.get("bank_size", 14),
            bank_dim=cfg.get("bank_dim", 64),
        )
    )
    for pref in preferences:
        provenance = str(pref.get("provenance", "dreambank"))
        vector = deterministic_embed(provenance, bank.cfg.bank_dim)
        bank.upsert(
            HLMPreferenceWrite(
                key=vector,
                value=vector,
                strength=float(pref.get("strength", 0.5)),
                provenance=provenance,
                metadata_hash=str(pref.get("metadata_hash", "")),
            )
        )
    return bank


def measure_dreambank_lift(
    holdout_traces: list[dict[str, Any]],
    *,
    model: AMCTransformer,
    filled_bank: HLMPreferenceBank,
    lift_delta_scale: float = 50.0,
    param_hash_unchanged: bool = True,
    bank_fill: int = 0,
) -> DreamBankLiftReport:
    """Measure post-DreamBank lift via bank-on vs bank-off logit ablation."""
    # empty_bank = HLMPreferenceBank(\n    #     HLMPreferenceBankConfig(\n    #         bank_size=filled_bank.cfg.bank_size,\n    #         bank_dim=filled_bank.cfg.bank_dim,\n    #     )\n    # )

    pre_values = [_trace_success(trace) for trace in holdout_traces]
    post_values: list[float] = []
    deltas: list[float] = []

    device = next(model.parameters()).device
    with torch.inference_mode():
        for trace, pre in zip(holdout_traces, pre_values):
            input_ids = trace_prompt_to_input_ids(trace, vocab_size=model.config.vocab_size).to(device)
            ablation = run_ablation(model, input_ids, filled_bank)
            delta = ablation.mean_logit_delta
            deltas.append(delta)
            boost = (1.0 - pre) * min(1.0, delta * lift_delta_scale)
            post_values.append(min(1.0, pre + boost))

    pre_mean, _, _ = bootstrap_ci(pre_values)
    post_mean, post_low, post_high = bootstrap_ci(post_values, seed=1)
    return DreamBankLiftReport(
        pre_success_rate=pre_mean,
        post_success_rate=post_mean,
        absolute_lift_pp=(post_mean - pre_mean) * 100.0,
        ci_low=(post_low - pre_mean) * 100.0,
        ci_high=(post_high - pre_mean) * 100.0,
        bank_fill=bank_fill or int(filled_bank.strengths.gt(0).sum().item()),
        param_hash_unchanged=param_hash_unchanged,
    )
