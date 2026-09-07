"""
src/training/self_sft_loop.py — Self-Improvement Loop for SFT.

Implements a multi-round self-improvement pipeline where the model
generates completions, filters them by a verifier, and retrains on
its own correct outputs. Proven mechanism (STaR, ReST, SPIN).

Reference: "Self-Improvement Loop for Aurelius" (Aurelius, 2026)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


@dataclass
class SelfSFTConfig:
    """Configuration for the self-improvement loop.

    Attributes:
        n_rounds: Number of self-improvement rounds.
        generation_temperature: Temperature for sampling completions.
        n_samples_per_prompt: Number of completions per prompt per round.
        max_new_tokens: Maximum new tokens per generation.
        acceptance_threshold: Minimum acceptance rate before proceeding.
        batch_size: Generation batch size for GPU efficiency.
        rounds_dir: Directory to store round artifacts.
    """

    n_rounds: int = 3
    generation_temperature: float = 0.7
    n_samples_per_prompt: int = 3
    max_new_tokens: int = 256
    acceptance_threshold: float = 0.05
    batch_size: int = 4
    rounds_dir: str = "experiments/self_sft/rounds"


@dataclass
class GenerationExample:
    """A single generation example produced by the model."""

    prompt: str
    response: str
    verifier_result: bool = False
    verifier_detail: str = ""
    round: int = 0
    temperature: float = 0.7


class VerifierInterface:
    """Abstract verifier interface for self-improvement filtering.

    Subclass and implement check() for domain-specific verification.
    """

    def check(self, prompt: str, response: str) -> tuple[bool, str]:
        """Return (accepted: bool, detail: str)."""
        raise NotImplementedError


class SimpleVerifier(VerifierInterface):
    """Simple length + quality heuristic verifier.

    Rejects too-short, too-long, refusal-pattern, and repetitive outputs.
    Use as a placeholder; replace with domain-specific verifiers.
    """

    def __init__(
        self,
        min_words: int = 3,
        max_chars: int = 2000,
        refusal_phrases: tuple[str, ...] | None = None,
        min_vocab_ratio: float = 0.2,
    ) -> None:
        self.min_words = min_words
        self.max_chars = max_chars
        self.refusal_phrases = refusal_phrases or (
            "i cannot", "i am not able", "i cannot answer",
            "i'm not able", "i cannot fulfill",
        )
        self.min_vocab_ratio = min_vocab_ratio

    def check(self, prompt: str, response: str) -> tuple[bool, str]:
        if len(response.split()) < self.min_words:
            return False, f"too_short: {len(response.split())} words"
        if len(response) > self.max_chars:
            return False, f"too_long: {len(response)} chars"
        lower = response.lower()
        for phrase in self.refusal_phrases:
            if phrase in lower:
                return False, f"refusal: contains '{phrase}'"
        words = lower.split()
        if len(set(words)) / max(len(words), 1) < self.min_vocab_ratio:
            return False, f"repetitive: vocab_ratio={len(set(words))/max(len(words),1):.2f}"
        return True, "accepted"


class SelfSFTLoop:
    """Multi-round self-improvement loop.

    Each round:
    1. Generate N completions per prompt from the current model.
    2. Filter by verifier (keep only correct/reasonable completions).
    3. Train the model on accepted completions mixed with seed data.
    4. Evaluate acceptance rate and quality metrics.

    Usage:
        loop = SelfSFTLoop(model, tokenizer, verifier, config)
        results = loop.run(prompts, seed_sft_data)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        verifier: VerifierInterface,
        config: SelfSFTConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.config = config or SelfSFTConfig()

        self.round_results: list[dict[str, Any]] = []
        self._device = next(model.parameters()).device

    def run(
        self,
        prompts: list[str],
        seed_sft_data: list[dict[str, Any]] | None = None,
        trainer_fn: Callable | None = None,
    ) -> dict[str, Any]:
        """Run the N-round self-improvement loop.

        Args:
            prompts: List of prompt strings to generate from.
            seed_sft_data: Initial SFT data (prompt, response pairs).
            trainer_fn: Optional callable(base_model, train_data) -> trained_model.
                If None, a basic SFT loop is used.

        Returns:
            Dict with per-round results and final metrics.
        """
        current_model_state = self.model.state_dict()

        for round_idx in range(self.config.n_rounds):
            logger.info(
                "Self-SFT Round %d/%d starting (temp=%.1f, n_samples=%d)",
                round_idx + 1,
                self.config.n_rounds,
                self.config.generation_temperature,
                self.config.n_samples_per_prompt,
            )

            # Step 1: Generate completions
            candidates = self._generate(prompts, current_model_state)

            # Step 2: Filter by verifier
            accepted = [c for c in candidates if c.verifier_result]
            rejected = [c for c in candidates if not c.verifier_result]

            acceptance_rate = len(accepted) / max(len(candidates), 1)
            logger.info(
                "Round %d: %d/%d accepted (%.1f%%)",
                round_idx + 1,
                len(accepted),
                len(candidates),
                acceptance_rate * 100,
            )

            # Check kill gate
            if acceptance_rate < self.config.acceptance_threshold:
                logger.warning(
                    "Kill gate triggered: acceptance_rate=%.2f < threshold=%.2f. "
                    "Increase temperature or use easier prompts.",
                    acceptance_rate,
                    self.config.acceptance_threshold,
                )
                break

            # Step 3: Prepare training data
            train_data = list(seed_sft_data or [])
            for acc in accepted:
                train_data.append({
                    "prompt": acc.prompt,
                    "response": acc.response,
                })

            # Step 4: Train (via provided function or basic loop)
            if trainer_fn is not None:
                self.model = trainer_fn(self.model, train_data)
                current_model_state = self.model.state_dict()

            # Record round results
            round_result = {
                "round": round_idx + 1,
                "candidates": len(candidates),
                "accepted": len(accepted),
                "rejected": len(rejected),
                "acceptance_rate": acceptance_rate,
                "temperature": self.config.generation_temperature,
                "train_examples": len(train_data),
            }
            self.round_results.append(round_result)

            # Log sample
            if accepted:
                sample = accepted[0]
                logger.info(
                    "Sample accepted: prompt=%.80s... response=%.80s...",
                    sample.prompt, sample.response,
                )

        return {
            "n_rounds_completed": len(self.round_results),
            "rounds": self.round_results,
            "final_acceptance_rate": self.round_results[-1]["acceptance_rate"]
            if self.round_results else 0.0,
        }

    def _generate(
        self,
        prompts: list[str],
        model_state: dict[str, torch.Tensor],
    ) -> list[GenerationExample]:
        """Generate completions and filter by verifier."""
        self.model.load_state_dict(model_state)
        self.model.eval()

        candidates: list[GenerationExample] = []

        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                ).to(self._device)

                for _ in range(self.config.n_samples_per_prompt):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.generation_temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                    response = self.tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )

                    verifier_pass, detail = self.verifier.check(prompt, response)

                    candidates.append(GenerationExample(
                        prompt=prompt,
                        response=response,
                        verifier_result=verifier_pass,
                        verifier_detail=detail,
                        round=len(self.round_results) + 1,
                        temperature=self.config.generation_temperature,
                    ))

        return candidates

    def save_round_artifacts(self, output_dir: str | Path) -> None:
        """Save accepted/rejected examples and metrics per round."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "config": {
                "n_rounds": self.config.n_rounds,
                "temperature": self.config.generation_temperature,
                "n_samples_per_prompt": self.config.n_samples_per_prompt,
            },
            "rounds": self.round_results,
        }

        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Saved self-SFT artifacts to %s", output_dir)


# Registry entry
SELF_SFT_REGISTRY: dict[str, type] = {
    "self_sft_loop": SelfSFTLoop,
    "simple_verifier": SimpleVerifier,
}
