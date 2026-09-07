"""Ring 1 DreamBank runner — trace → sleep cycles with zero-gradient proof.

Tranche 3 glue only; does not modify src/alignment/dreambank.py or core model code.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.alignment.dreambank import (
    DreamBankConfig,
    DreamBankController,
    DreamCycleResult,
    DreamSeed,
)
from src.eval.ring1_trace_logger import load_traces
from src.memory.hlm_bank import HLMPreferenceBank, HLMPreferenceBankConfig


def hash_model_params(model: nn.Module) -> str:
    """SHA-256 digest of model state_dict tensors (zero-weight mutation proof)."""
    digest = hashlib.sha256(usedforsecurity=False)
    state = model.state_dict()
    for key in sorted(state.keys()):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class ZeroGradProof:
    pre_cycle_param_hash: str
    post_cycle_param_hash: str
    param_hash_unchanged: bool
    inference_mode: bool
    model_eval_mode: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DreamBankRunResult:
    cycles: list[DreamCycleResult]
    total_writes: int
    bank_fill: int
    sleep_cycle_log: list[dict[str, Any]] = field(default_factory=list)
    zero_grad_proof: ZeroGradProof | None = None
    preferences: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "cycles": [asdict(cycle) for cycle in self.cycles],
            "total_writes": self.total_writes,
            "bank_fill": self.bank_fill,
            "sleep_cycle_log": self.sleep_cycle_log,
            "preferences": self.preferences,
        }
        if self.zero_grad_proof is not None:
            payload["zero_grad_proof"] = self.zero_grad_proof.to_dict()
        return payload


def extract_seeds_from_traces(
    traces: Sequence[dict[str, Any]],
    *,
    max_seeds: int | None = None,
    fields: tuple[str, ...] = ("observation", "reflection"),
) -> list[DreamSeed]:
    """Build DreamBank seeds from trace step observations and reflections."""
    seeds: list[DreamSeed] = []
    for trace in traces:
        trace_id = trace.get("trace_id", "unknown")
        for step in trace.get("steps", []):
            for field_name in fields:
                if field_name == "observation":
                    content = step.get("observation", "")
                else:
                    reflection = step.get("reflection", {})
                    content = reflection.get("summary", "") if isinstance(reflection, dict) else ""
                content = str(content).strip()
                if not content:
                    continue
                seeds.append(
                    DreamSeed(
                        prompt=content[:512],
                        source=f"{trace_id}:step_{step.get('step_id', 0)}:{field_name}",
                    )
                )
                if max_seeds is not None and len(seeds) >= max_seeds:
                    return seeds
    return seeds


def deterministic_generate(prompt: str, temperature: float) -> str:
    digest = hashlib.sha256(f"{prompt}@{temperature}".encode()).hexdigest()
    return f"gen[{digest[:8]}]_t{temperature}"


def deterministic_score(prompt: str, response: str) -> float:
    digest = int(hashlib.sha256(f"{prompt}:{response}".encode()).hexdigest()[:8], 16)
    return (digest % 1000) / 1000.0


def deterministic_embed(text: str, dim: int = 64) -> torch.Tensor:
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randn(dim, generator=generator)


def run_sleep_cycles(
    seeds: Sequence[DreamSeed],
    *,
    bank: HLMPreferenceBank,
    dreambank_config: DreamBankConfig | None = None,
    num_cycles: int = 1,
    generate_fn: Callable[[str, float], str] | None = None,
    score_fn: Callable[[str, str], float] | None = None,
    embed_fn: Callable[[str], torch.Tensor] | None = None,
    model: nn.Module | None = None,
    enforce_zero_grad: bool = True,
) -> DreamBankRunResult:
    """Run DreamBank sleep cycles inside inference_mode with optional param-hash proof."""
    controller = DreamBankController(bank, dreambank_config)
    gen = generate_fn or deterministic_generate
    score = score_fn or deterministic_score
    embed = embed_fn or (lambda text: deterministic_embed(text, bank.cfg.bank_dim))

    pre_hash = hash_model_params(model) if model is not None else "no-model-attached"
    if model is not None:
        model.eval()

    cycles: list[DreamCycleResult] = []
    sleep_log: list[dict[str, Any]] = []
    preferences: list[dict[str, Any]] = []

    with torch.inference_mode():
        for cycle_idx in range(num_cycles):
            result = controller.run_cycle(
                seeds,
                generate_fn=gen,
                score_fn=score,
                embed_fn=embed,
            )
            cycles.append(result)
            sleep_log.append(
                {
                    "cycle": cycle_idx,
                    "seeds": result.seeds,
                    "candidates": result.candidates,
                    "pairs": result.pairs,
                    "writes": result.writes,
                    "mean_margin": result.mean_margin,
                    "bank_fill": result.bank_fill,
                }
            )

    post_hash = hash_model_params(model) if model is not None else "no-model-attached"
    proof = ZeroGradProof(
        pre_cycle_param_hash=pre_hash,
        post_cycle_param_hash=post_hash,
        param_hash_unchanged=pre_hash == post_hash,
        inference_mode=True,
        model_eval_mode=model.training is False if model is not None else None,
    )
    if enforce_zero_grad and model is not None and not proof.param_hash_unchanged:
        raise RuntimeError(
            f"Zero-gradient violation: param hash changed ({pre_hash[:12]} -> {post_hash[:12]})"
        )

    total_writes = sum(cycle.writes for cycle in cycles)
    bank_fill = int(bank.strengths.gt(0).sum().item())
    exported = bank.export_state()
    for index, strength in enumerate(exported["strengths"].tolist()):
        if strength > 0:
            preferences.append(
                {
                    "slot": index,
                    "strength": strength,
                    "provenance": exported["provenances"][index],
                    "metadata_hash": exported["metadata_hashes"][index],
                }
            )

    return DreamBankRunResult(
        cycles=cycles,
        total_writes=total_writes,
        bank_fill=bank_fill,
        sleep_cycle_log=sleep_log,
        zero_grad_proof=proof,
        preferences=preferences,
    )


def run_shuffled_control_cycles(
    seeds: Sequence[DreamSeed],
    *,
    bank: HLMPreferenceBank,
    dreambank_config: DreamBankConfig | None = None,
    num_cycles: int = 1,
    shuffle_seed: int = 999,
) -> DreamBankRunResult:
    """Run DreamBank with shuffled seed prompts as placebo control (R1-GB Step 5)."""
    shuffled = list(seeds)
    rng = random.Random(shuffle_seed)
    rng.shuffle(shuffled)
    shuffled_seeds = [
        DreamSeed(prompt=seed.prompt, source=f"shuffled:{seed.source}") for seed in shuffled
    ]
    return run_sleep_cycles(
        shuffled_seeds,
        bank=bank,
        dreambank_config=dreambank_config,
        num_cycles=num_cycles,
        enforce_zero_grad=False,
    )


def run_sleep_on_trace_file(
    traces_path: Path,
    *,
    bank_config: dict[str, Any] | None = None,
    dreambank_config: dict[str, Any] | None = None,
    num_cycles: int = 1,
    max_seeds: int = 32,
    model: nn.Module | None = None,
) -> DreamBankRunResult:
    """Load traces from JSONL and run DreamBank sleep cycles."""
    traces = load_traces(traces_path)
    seeds = extract_seeds_from_traces(traces, max_seeds=max_seeds)
    bank_cfg = bank_config or {}
    bank = HLMPreferenceBank(
        HLMPreferenceBankConfig(
            bank_size=bank_cfg.get("bank_size", 14),
            bank_dim=bank_cfg.get("bank_dim", 64),
        )
    )
    db_cfg = dreambank_config or {}
    config = DreamBankConfig(
        max_candidates_per_seed=db_cfg.get("max_candidates_per_seed", 4),
        min_margin=db_cfg.get("min_margin", 0.05),
        max_writes_per_cycle=db_cfg.get("max_writes_per_cycle", 8),
        decay_steps_per_cycle=db_cfg.get("decay_steps_per_cycle", 1),
        temperatures=tuple(db_cfg.get("temperatures", [0.2, 0.7, 1.0, 1.3])),
    )
    return run_sleep_cycles(
        seeds,
        bank=bank,
        dreambank_config=config,
        num_cycles=num_cycles,
        model=model,
        enforce_zero_grad=db_cfg.get("enforce_zero_grad", True),
    )


def write_dreambank_artifacts(result: DreamBankRunResult, output_dir: Path) -> None:
    """Write sleep logs and hash proof files for reproducibility packs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sleep_cycle_log.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in result.sleep_cycle_log) + "\n",
        encoding="utf-8",
    )
    (output_dir / "dreambank_result.json").write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    if result.zero_grad_proof is not None:
        (output_dir / "pre_cycle_model_hash.txt").write_text(
            result.zero_grad_proof.pre_cycle_param_hash + "\n",
            encoding="utf-8",
        )
        (output_dir / "post_cycle_model_hash.txt").write_text(
            result.zero_grad_proof.post_cycle_param_hash + "\n",
            encoding="utf-8",
        )
    if result.preferences:
        (output_dir / "preferences.json").write_text(
            json.dumps(result.preferences, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
