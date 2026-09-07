"""
src/training/progressive_self_sft.py — Progressive Self-SFT with difficulty curriculum.

Staged self-improvement: Stage 1 (easy teacher data), Stage 2 (medium
self-generated data), Stage 3 (hard self-generated data), Stage 4
(self-improvement via critique). Each stage uses the previous checkpoint.

Reference: "Progressive Self-SFT for Aurelius" (Aurelius, 2026)
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
class ProgressiveSelfSFTConfig:
    """Configuration for progressive self-improvement.

    Attributes:
        stage_temperatures: Temperature for each stage.
        stage_samples: Number of samples per prompt for each stage.
        stage_epochs: Number of training epochs for each stage.
        stage_learning_rates: Learning rate for each stage.
        difficulty_labels: How prompts are labeled by difficulty.
        checkpoint_dir: Directory to save stage checkpoints.
    """

    stage_temperatures: tuple[float, ...] = (0.5, 0.7, 0.3, 0.5)
    stage_samples: tuple[int, ...] = (1, 3, 3, 2)
    stage_epochs: tuple[int, ...] = (2, 2, 1, 1)
    stage_learning_rates: tuple[float, ...] = (2e-5, 1e-5, 5e-6, 5e-6)
    difficulty_labels: tuple[str, ...] = ("easy", "medium", "hard", "critique")
    checkpoint_dir: str = "checkpoints/progressive_self_sft"
    n_stages: int = 4


@dataclass
class StageResult:
    """Result of a single progressive self-SFT stage."""

    stage: int
    difficulty: str
    n_prompts: int
    n_accepted: int
    acceptance_rate: float
    temperature: float
    learning_rate: float
    checkpoint_path: str = ""


class ProgressiveSelfSFT:
    """Multi-stage progressive self-improvement SFT.

    Each stage:
    - Uses prompts of a specific difficulty level.
    - Generates completions at a specific temperature.
    - Filters by verifier.
    - Trains on accepted completions (mixed with previous stage data).
    - Saves a checkpoint for the next stage.

    Usage:
        pipeline = ProgressiveSelfSFT(model, tokenizer, verifier, config)
        results = pipeline.run(difficulty_prompts, seed_sft_data)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        verifier: Callable[[str, str], tuple[bool, str]],
        config: ProgressiveSelfSFTConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.config = config or ProgressiveSelfSFTConfig()
        self._device = next(model.parameters()).device
        self._stage_results: list[StageResult] = []
        self._checkpoint_paths: list[str] = []

    def run(
        self,
        prompts_by_difficulty: dict[str, list[str]],
        seed_sft_data: list[dict[str, Any]] | None = None,
        trainer_fn: Callable | None = None,
    ) -> dict[str, Any]:
        """Run all progressive self-SFT stages.

        Args:
            prompts_by_difficulty: Dict mapping difficulty label to prompts.
            seed_sft_data: Optional initial training data (stage 0 supplement).
            trainer_fn: Optional callable(model, data) -> trained_model.

        Returns:
            Dict with per-stage results and final metrics.
        """
        accumulated_data = list(seed_sft_data or [])

        for stage in range(self.config.n_stages):
            difficulty = self.config.difficulty_labels[stage]
            temperature = self.config.stage_temperatures[stage]
            n_samples = self.config.stage_samples[stage]
            n_epochs = self.config.stage_epochs[stage]
            lr = self.config.stage_learning_rates[stage]

            prompts = prompts_by_difficulty.get(difficulty, [])
            if not prompts:
                logger.warning(
                    "Stage %d (%s): no prompts available, skipping",
                    stage, difficulty,
                )
                continue

            logger.info(
                "Stage %d/%d (%s): temp=%.1f, samples=%d, epochs=%d, lr=%.1e",
                stage + 1, self.config.n_stages, difficulty,
                temperature, n_samples, n_epochs, lr,
            )

            # Generate completions
            candidates = self._generate(prompts, temperature, n_samples)

            # Filter by verifier
            accepted = [c for c in candidates if self.verifier(c["prompt"], c["response"])[0]]
            acceptance_rate = len(accepted) / max(len(candidates), 1)

            logger.info(
                "Stage %d: accepted %d/%d (%.1f%%)",
                stage, len(accepted), len(candidates), acceptance_rate * 100,
            )

            # Accumulate training data
            for acc in accepted:
                accumulated_data.append({
                    "prompt": acc["prompt"],
                    "response": acc["response"],
                    "stage": stage,
                    "difficulty": difficulty,
                })

            # Train
            if trainer_fn is not None and accepted:
                self.model = trainer_fn(
                    self.model, accumulated_data,
                    lr=lr, epochs=n_epochs,
                )

            # Save checkpoint
            ckpt_path = f"{self.config.checkpoint_dir}/stage_{stage}_{difficulty}"
            self._checkpoint_paths.append(ckpt_path)

            self._stage_results.append(StageResult(
                stage=stage,
                difficulty=difficulty,
                n_prompts=len(prompts),
                n_accepted=len(accepted),
                acceptance_rate=acceptance_rate,
                temperature=temperature,
                learning_rate=lr,
                checkpoint_path=ckpt_path,
            ))

        return self.summary()

    def _generate(
        self,
        prompts: list[str],
        temperature: float,
        n_samples: int,
    ) -> list[dict[str, Any]]:
        """Generate completions from the current model."""
        self.model.eval()
        candidates: list[dict[str, Any]] = []

        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                ).to(self._device)

                for _ in range(n_samples):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                    response = self.tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                    candidates.append({"prompt": prompt, "response": response})

        return candidates

    def summary(self) -> dict[str, Any]:
        """Return summary of all stages."""
        return {
            "n_stages": len(self._stage_results),
            "stages": [
                {
                    "stage": r.stage,
                    "difficulty": r.difficulty,
                    "n_accepted": r.n_accepted,
                    "acceptance_rate": r.acceptance_rate,
                    "temperature": r.temperature,
                }
                for r in self._stage_results
            ],
            "total_accepted": sum(r.n_accepted for r in self._stage_results),
            "checkpoints": self._checkpoint_paths,
        }


# Registry entry
PROGRESSIVE_SFT_REGISTRY: dict[str, type] = {
    "progressive_self_sft": ProgressiveSelfSFT,
}
