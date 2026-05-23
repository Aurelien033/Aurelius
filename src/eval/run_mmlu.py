"""MMLU-57 evaluation runner for Forge checkpoints.

Uses canonical exemplars for smoke/CI profiles. Exposes :func:`run_benchmark`
for ablation scripts.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

from src.eval.mmlu_scorer import CANONICAL_EXEMPLARS, MMLUProblem, MMLUScorer

_CHOICE_LETTERS = ("A", "B", "C", "D")

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _problems_for_profile(profile: str) -> list[MMLUProblem]:
    if profile == "smoke":
        return list(CANONICAL_EXEMPLARS[:3])
    return list(CANONICAL_EXEMPLARS)


def build_generate_fn(
    *,
    mode: str,
    checkpoint: str,
    backend: str = "mock",
    problems: list[MMLUProblem],
) -> Any:
    if mode == "oracle":

        def _oracle(prompt: str) -> str:
            for problem in problems:
                if problem.question in prompt:
                    return f"({_CHOICE_LETTERS[problem.correct_index]})"
            return "(A)"

        return _oracle

    if mode == "mock":
        return lambda _prompt: "(A)"

    if mode == "engine":
        from src.eval.amc_memory_runner import build_engine_generate_fn

        return build_engine_generate_fn(backend=backend, model_path=checkpoint)

    raise ValueError(f"unknown mode {mode!r}; expected oracle, mock, or engine")


def run_benchmark(
    checkpoint: str = "",
    config: str = "",
    n_samples: int = 5,
    seed: int = 42,
    *,
    profile: str = "smoke",
    mode: str = "oracle",
    backend: str = "mock",
) -> list[float]:
    """Return per-seed overall accuracy scores."""
    _ = config
    problems = _problems_for_profile(profile)
    scorer = MMLUScorer(generate_fn=None, n_shots=0)
    rng = random.Random(seed)
    scores: list[float] = []

    for sample_idx in range(n_samples):
        order = list(problems)
        rng.shuffle(order)
        generate_fn = build_generate_fn(
            mode=mode,
            checkpoint=checkpoint,
            backend=backend,
            problems=order,
        )
        responses = [generate_fn(scorer.format_prompt(problem)) for problem in order]
        metrics = scorer.score(order, responses)
        scores.append(float(metrics["overall_accuracy"]))
        rng.seed(seed + sample_idx + 1)
    return scores


def run_eval(
    *,
    checkpoint: str = "",
    config: str = "",
    profile: str = "smoke",
    mode: str | None = None,
    backend: str = "mock",
    n_samples: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    resolved_mode = mode or ("oracle" if profile == "smoke" else "engine")
    sample_count = 1 if profile == "smoke" else n_samples
    scores = run_benchmark(
        checkpoint=checkpoint,
        config=config,
        n_samples=sample_count,
        seed=seed,
        profile=profile,
        mode=resolved_mode,
        backend=backend,
    )
    accuracy = statistics.mean(scores) if scores else 0.0
    stderr = 0.0
    if len(scores) > 1:
        stderr = statistics.stdev(scores) / (len(scores) ** 0.5)
    return {
        "suite": "mmlu_57",
        "profile": profile,
        "mode": resolved_mode,
        "checkpoint": checkpoint or None,
        "config": config or None,
        "n_problems": len(_problems_for_profile(profile)),
        "n_samples": len(scores),
        "accuracy": accuracy,
        "score": accuracy,
        "stderr": stderr,
        "per_seed_scores": scores,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MMLU evaluation")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path for engine mode")
    parser.add_argument("--config", default="", help="Optional YAML config path (recorded only)")
    parser.add_argument("--profile", choices=("smoke", "ci"), default="smoke")
    parser.add_argument("--mode", choices=("oracle", "mock", "engine"), default=None)
    parser.add_argument("--backend", default="mock", help="Serving backend for engine mode")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None, help="Write JSON results here")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_eval(
        checkpoint=args.checkpoint,
        config=args.config,
        profile=args.profile,
        mode=args.mode,
        backend=args.backend,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    raise SystemExit(main())
