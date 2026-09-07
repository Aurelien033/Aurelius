"""DreamBank controller — sleep-time preference consolidation via self-play.

For each DreamSeed:
  1. Generate responses at configured temperatures via injectable generate_fn.
  2. Score each with injectable score_fn.
  3. Form (chosen, rejected) pairs when margin >= min_margin.
  4. Embed chosen responses, upsert into HLMPreferenceBank.
  5. Apply bank decay at end of cycle.

No raw prompts are stored in bank metadata — only prompt hashes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from src.memory.hlm_bank import HLMPreferenceBank, HLMPreferenceWrite


@dataclass(frozen=True)
class DreamBankConfig:
    max_candidates_per_seed: int = 4
    temperatures: tuple[float, ...] = (0.2, 0.7, 1.0, 1.3)
    min_margin: float = 0.05
    max_writes_per_cycle: int = 8
    decay_steps_per_cycle: int = 1
    metadata_salt: str = "dreambank-v1"


@dataclass(frozen=True)
class DreamSeed:
    prompt: str
    source: str = "recent_query"


@dataclass(frozen=True)
class DreamPreferencePair:
    prompt_hash: str
    chosen: str
    rejected: str
    margin: float
    provenance: str


@dataclass(frozen=True)
class DreamCycleResult:
    seeds: int
    candidates: int
    pairs: int
    writes: int
    mean_margin: float
    bank_fill: int


class DreamBankController:
    """Sleep-time dream-cycle controller for HLMPreferenceBank."""

    def __init__(
        self,
        bank: HLMPreferenceBank,
        config: DreamBankConfig | None = None,
    ) -> None:
        self.bank = bank
        self.cfg = config or DreamBankConfig()

    @staticmethod
    def _hash_prompt(prompt: str, salt: str) -> str:
        return hashlib.shake_256((salt + prompt).encode("utf-8")).hexdigest(16)

    def run_cycle(
        self,
        seeds: Sequence[DreamSeed],
        *,
        generate_fn: Callable[[str, float], str],
        score_fn: Callable[[str, str], float],
        embed_fn: Callable[[str], torch.Tensor],
    ) -> DreamCycleResult:
        """Run one dream cycle.

        Args:
            seeds: Dream seeds (prompts to re-score).
            generate_fn: (prompt, temperature) -> response text.
            score_fn: (prompt, response) -> scalar score.
            embed_fn: response text -> (bank_dim,) tensor embedding.

        Returns:
            DreamCycleResult with cycle statistics.
        """
        cfg = self.cfg
        pairs: list[DreamPreferencePair] = []
        total_candidates = 0
        writes = 0

        for seed in seeds:
            temps = cfg.temperatures[: cfg.max_candidates_per_seed]
            responses: list[tuple[str, float]] = []

            for temp in temps:
                text = generate_fn(seed.prompt, temp)
                score = score_fn(seed.prompt, text)
                responses.append((text, score))
                total_candidates += 1

            if len(responses) < 2:
                continue

            chosen_text, chosen_score = max(responses, key=lambda x: x[1])
            rejected_text, rejected_score = min(responses, key=lambda x: x[1])

            margin = chosen_score - rejected_score
            if margin < cfg.min_margin:
                continue

            prompt_hash = self._hash_prompt(seed.prompt, cfg.metadata_salt)
            pairs.append(DreamPreferencePair(
                prompt_hash=prompt_hash,
                chosen=chosen_text,
                rejected=rejected_text,
                margin=margin,
                provenance=f"dream_{seed.source}",
            ))

        # Upsert chosen embeddings into bank
        bank_dim = self.bank.cfg.bank_dim
        for pair in pairs:
            if writes >= cfg.max_writes_per_cycle:
                break

            embedding = embed_fn(pair.chosen)
            # Flatten embedding to 1D
            embedding = embedding.flatten()
            # Pad or truncate to bank_dim
            if embedding.shape[0] != bank_dim:
                if embedding.shape[0] > bank_dim:
                    embedding = embedding[:bank_dim]
                else:
                    pad = torch.zeros(bank_dim, device=embedding.device, dtype=embedding.dtype)
                    pad[: embedding.shape[0]] = embedding
                    embedding = pad
            embedding = embedding.to(self.bank.keys.device, self.bank.keys.dtype)

            self.bank.upsert(HLMPreferenceWrite(
                key=embedding,
                value=embedding,
                strength=min(max(pair.margin, 0.0), 1.0),
                provenance=pair.provenance,
                trust="unverified",
                metadata_hash=pair.prompt_hash,
            ))
            writes += 1

        # Decay at end of cycle
        if cfg.decay_steps_per_cycle > 0:
            self.bank.decay_(cfg.decay_steps_per_cycle)

        mean_margin = float(torch.tensor([p.margin for p in pairs]).mean()) if pairs else 0.0

        return DreamCycleResult(
            seeds=len(seeds),
            candidates=total_candidates,
            pairs=len(pairs),
            writes=writes,
            mean_margin=mean_margin,
            bank_fill=int(self.bank.strengths.gt(0).sum().item()),
        )
