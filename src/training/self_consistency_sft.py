"""
src/training/self_consistency_sft.py — Self-consistency training for SFT.

Trains the model so that multiple samples at temperature > 0 converge to
the same correct answer. Uses verifier-agreed or majority-vote consensus
as the training target, with a KL divergence bonus encouraging agreement.

Reference: "Self-Consistency as a Training Signal" (Aurelius, 2026)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class SelfConsistencyConfig:
    """Configuration for self-consistency training.

    Attributes:
        n_samples: Number of samples to generate per prompt.
        temperature: Sampling temperature for generating multiple outputs.
        max_new_tokens: Maximum new tokens per generation.
        lambda_consistency: Weight of consistency KL loss.
        consensus_method: How to determine consensus ('majority', 'verifier', 'entropy').
        sample_every_n_steps: How often to run consistency sampling.
    """

    n_samples: int = 5
    temperature: float = 0.8
    max_new_tokens: int = 128
    lambda_consistency: float = 0.2
    consensus_method: str = "majority"
    sample_every_n_steps: int = 50


class SelfConsistencyLoss:
    """Self-consistency loss for SFT training.

    For each training prompt:
    1. Generate N samples at temperature > 0.
    2. Determine consensus output (by majority vote or verifier).
    3. Compute KL divergence between each sample's logits and the
       consensus distribution.
    4. Add consistency_loss = mean KL as regularization.

    Usage:
        loss_fn = SelfConsistencyLoss(model, tokenizer, config)
        total_loss = ce_loss + lambda * consistency_loss
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        config: SelfConsistencyConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SelfConsistencyConfig()
        self._device = next(model.parameters()).device

    def compute_consistency_loss(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute self-consistency KL loss for a batch.

        Args:
            input_ids: (B, T) input token IDs.
            attention_mask: (B, T) optional attention mask.

        Returns:
            Scalar consistency loss.
        """
        batch_size = input_ids.size(0)
        total_kl = torch.tensor(0.0, device=self._device)
        n_valid = 0

        for b in range(batch_size):
            single_input = input_ids[b : b + 1]
            single_mask = attention_mask[b : b + 1] if attention_mask is not None else None

            # Generate N samples
            with torch.no_grad():
                outputs = self.model.generate(
                    single_input,
                    attention_mask=single_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    num_return_sequences=self.config.n_samples,
                    pad_token_id=self.tokenizer.eos_token_id,
                    output_logits=True,
                    return_dict_in_generate=True,
                )

            # Compute consensus distribution
            # Stack logits across samples at each generation step
            logits_list = outputs.logits  # list of (N, V) per step
            n_steps = len(logits_list)
            if n_steps < 2:
                continue

            # Average logits across samples at each step
            stacked = torch.stack(logits_list)  # (n_steps, N, V)
            consensus = F.softmax(stacked.mean(dim=1), dim=-1)  # (n_steps, V)

            # Compute mean KL across samples and steps
            kl_sum = 0.0
            for step_logits in logits_list:  # each (N, V)
                probs = F.softmax(step_logits, dim=-1)
                kl = F.kl_div(
                    probs.log(),
                    consensus[: probs.size(0)],
                    reduction="batchmean",
                    log_target=False,
                )
                kl_sum += kl

            avg_kl = kl_sum / len(logits_list)
            total_kl = total_kl + avg_kl
            n_valid += 1

        if n_valid == 0:
            return torch.tensor(0.0, device=self._device, requires_grad=True)

        return total_kl / n_valid


class MajorityVoteFilter:
    """Filters outputs by majority vote across multiple samples.

    Used to determine which samples are 'correct enough' to train on.
    """

    def __init__(self, config: SelfConsistencyConfig | None = None) -> None:
        self.config = config or SelfConsistencyConfig()

    def __call__(
        self,
        samples: list[dict[str, Any]],
        verifier_fn: Callable | None = None,
    ) -> list[dict[str, Any]]:
        """Return samples that agree with the consensus.

        Args:
            samples: List of dicts with 'prompt', 'response', 'logprobs'.
            verifier_fn: Optional callable(response) -> bool.

        Returns:
            Filtered list of samples that match consensus.
        """
        if verifier_fn is not None:
            # Keep only verifier-passing samples
            verified = [s for s in samples if verifier_fn(s["prompt"], s["response"])]
            if verified:
                return verified

        # Majority vote: group identical/similar responses
        if not samples:
            return []

        response_groups: dict[str, list[dict[str, Any]]] = {}
        for s in samples:
            norm_response = s["response"].strip().lower()
            response_groups.setdefault(norm_response, []).append(s)

        # Find the largest group
        largest_group = max(response_groups.values(), key=len)

        # Return all samples in the consensus group
        return largest_group


# Registry entry
SELF_CONSISTENCY_REGISTRY: dict[str, type] = {
    "self_consistency_loss": SelfConsistencyLoss,
    "majority_vote_filter": MajorityVoteFilter,
}
