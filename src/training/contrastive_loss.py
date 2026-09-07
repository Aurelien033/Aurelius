"""
src/training/contrastive_loss.py — Contrastive Tool-Call SFT Loss.

Adds near-miss negative examples to tool-call SFT training. For each
positive tool-call example, generates negative variants and trains with
a margin-based contrastive loss on top of standard cross-entropy.

Reference: "Contrastive Training for Reliable Tool Use" (Aurelius, 2026)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ContrastiveSFTConfig:
    """Configuration for contrastive tool-call SFT.

    Attributes:
        margin: Minimum margin between positive and negative log-probabilities.
        lambda_c: Weight of contrastive loss relative to CE loss.
        n_negatives: Number of negative variants to generate per positive.
        contrastive_frequency: Fraction of batches to apply contrastive loss.
        error_types: List of error types to generate.
    """

    margin: float = 0.5
    lambda_c: float = 0.3
    n_negatives: int = 3
    contrastive_frequency: float = 0.5
    error_types: tuple[str, ...] = (
        "parameter_name_typo",
        "type_violation",
        "wrong_tool_name",
        "missing_parameter",
    )


class NegativeGenerator:
    """Generates near-miss negative variants of tool-call examples.

    For a valid tool call, produces variants with:
    - Parameter name typos
    - Type violations
    - Wrong tool names
    - Missing required parameters
    """

    def __init__(self, config: ContrastiveSFTConfig | None = None) -> None:
        self.config = config or ContrastiveSFTConfig()

    def generate(
        self,
        tool_call: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate negative variants of a tool call.

        Args:
            tool_call: Dictionary with 'name' and 'arguments' keys.

        Returns:
            List of negative variant dicts with 'name', 'arguments',
            and 'error_type' keys.
        """
        name = tool_call.get("name", "")
        args = tool_call.get("arguments", tool_call.get("parameters", {}))
        if not isinstance(args, dict):
            args = {}

        negatives: list[dict[str, Any]] = []
        used_error_types = set()

        # Negative 1: Parameter name typo
        if (
            args
            and "parameter_name_typo" in self.config.error_types
        ):
            for key in args:
                if len(key) > 3 and "parameter_name_typo" not in used_error_types:
                    typo_key = self._make_typo(key)
                    neg_args = dict(args)
                    neg_args[typo_key] = neg_args.pop(key)
                    negatives.append({
                        "name": name,
                        "arguments": neg_args,
                        "error_type": "parameter_name_typo",
                    })
                    used_error_types.add("parameter_name_typo")
                    break

        # Negative 2: Type violation
        if (
            args
            and "type_violation" in self.config.error_types
        ):
            for key, val in args.items():
                if isinstance(val, str) and "type_violation" not in used_error_types:
                    neg_args = dict(args)
                    neg_args[key] = 12345  # Replace string with integer
                    negatives.append({
                        "name": name,
                        "arguments": neg_args,
                        "error_type": "type_violation",
                    })
                    used_error_types.add("type_violation")
                    break
                elif isinstance(val, (int, float)) and "type_violation" not in used_error_types:
                    neg_args = dict(args)
                    neg_args[key] = "not_a_number"
                    negatives.append({
                        "name": name,
                        "arguments": neg_args,
                        "error_type": "type_violation",
                    })
                    used_error_types.add("type_violation")
                    break

        # Negative 3: Wrong tool name
        if len(name) > 3 and "wrong_tool_name" in self.config.error_types:
            wrong_name = self._make_tool_name_typo(name)
            negatives.append({
                "name": wrong_name,
                "arguments": dict(args),
                "error_type": "wrong_tool_name",
            })

        # Negative 4: Missing parameter
        if (
            args
            and len(args) >= 2
            and "missing_parameter" in self.config.error_types
        ):
            neg_args = dict(args)
            last_key = list(neg_args.keys())[-1]
            del neg_args[last_key]
            negatives.append({
                "name": name,
                "arguments": neg_args,
                "error_type": "missing_parameter",
            })

        return negatives[: self.config.n_negatives]

    def _make_typo(self, key: str) -> str:
        """Introduce a plausible typo in a parameter name."""
        if len(key) <= 3:
            return key + "2"
        # Swap two adjacent characters
        idx = len(key) // 2
        chars = list(key)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)

    def _make_tool_name_typo(self, name: str) -> str:
        """Introduce a plausible typo in a tool name."""
        if len(name) > 6:
            # Drop a middle character
            return name[: len(name) // 2] + name[len(name) // 2 + 1 :]
        return name + "_v2"


class ContrastiveToolLoss:
    """Contrastive loss for tool-call SFT.

    Combines standard cross-entropy loss with a contrastive margin loss
    that pushes positive tool-call log-probabilities above negative ones.

    Usage:
        loss_fn = ContrastiveToolLoss(config)
        total_loss, components = loss_fn(model, pos_batch, neg_batch)
    """

    def __init__(self, config: ContrastiveSFTConfig | None = None) -> None:
        self.config = config or ContrastiveSFTConfig()
        self.negative_generator = NegativeGenerator(config)

    def __call__(
        self,
        pos_logits: Tensor,
        pos_labels: Tensor,
        neg_logits: Tensor,
        neg_labels: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute CE + contrastive loss.

        Args:
            pos_logits: (B, T, V) logits for positive examples.
            pos_labels: (B, T) labels for positive examples.
            neg_logits: (B, T, V) logits for negative examples.
            neg_labels: (B, T) labels for negative examples.

        Returns:
            (total_loss, components_dict) where components_dict contains
            individual loss terms for monitoring.
        """
        # Cross-entropy on positive examples
        ce_loss = F.cross_entropy(
            pos_logits.view(-1, pos_logits.size(-1)),
            pos_labels.view(-1),
            ignore_index=-100,
        )

        # Cross-entropy on negative examples (they should also be learnable)
        neg_ce = F.cross_entropy(
            neg_logits.view(-1, neg_logits.size(-1)),
            neg_labels.view(-1),
            ignore_index=-100,
        )

        # Positive log-probability (mean over valid tokens)
        pos_logprob = -F.cross_entropy(
            pos_logits.view(-1, pos_logits.size(-1)),
            pos_labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )

        # Negative log-probability
        neg_logprob = -F.cross_entropy(
            neg_logits.view(-1, neg_logits.size(-1)),
            neg_labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )

        # Margin-based contrastive loss
        margin_loss = F.relu(
            self.config.margin - (pos_logprob - neg_logprob)
        )

        total = ce_loss + neg_ce + self.config.lambda_c * margin_loss

        return total, {
            "ce_loss": ce_loss.item() if hasattr(ce_loss, "item") else float(ce_loss),
            "neg_ce_loss": neg_ce.item() if hasattr(neg_ce, "item") else float(neg_ce),
            "margin_loss": margin_loss.item() if hasattr(margin_loss, "item") else float(margin_loss),
            "pos_logprob": pos_logprob.item() if hasattr(pos_logprob, "item") else float(pos_logprob),
            "neg_logprob": neg_logprob.item() if hasattr(neg_logprob, "item") else float(neg_logprob),
            "total_loss": total.item() if hasattr(total, "item") else float(total),
        }


# Registry entry
CONTRASTIVE_LOSS_REGISTRY: dict[str, type] = {
    "contrastive_tool": ContrastiveToolLoss,
    "negative_generator": NegativeGenerator,
}
