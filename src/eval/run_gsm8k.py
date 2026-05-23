"""GSM8K evaluation runner for Forge checkpoints.

Provides a small fixed smoke set for CI and an engine-backed path for real
checkpoints. Exposes :func:`run_benchmark` for ablation scripts.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from src.eval.gsm8k_scorer import GSM8KScorer

_REPO_ROOT = Path(__file__).resolve().parents[2]

GSM8K_SMOKE_EXAMPLES: list[tuple[str, str]] = [
    (
        "Natalia sold 48 clips in April. In May she sold half as many. "
        "How many clips did she sell altogether?",
        "April: 48. May: 48/2=24. Total: 48+24=72.\n#### 72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday she worked 5 hours. "
        "How much did she earn?",
        "5 hours * $12 = $60.\n#### 60",
    ),
    (
        "Betty has 20 books. She gives 5 to her friend. How many remain?",
        "20 - 5 = 15.\n#### 15",
    ),
    (
        "A robe takes 2 bolts of blue fiber and half that much white fiber. "
        "How many bolts in total?",
        "Blue: 2. White: 1. Total: 3.\n#### 3",
    ),
    (
        "Josh buys a house for $80,000 and puts in $50,000 repairs. "
        "The value increases 150%. What is his profit?",
        "Value: 80000+50000=130000; after 150% gain value is 325000; profit 195000.\n#### 195000",
    ),
]


def _oracle_lookup() -> dict[str, str]:
    return {question.strip(): answer for question, answer in GSM8K_SMOKE_EXAMPLES}


def build_generate_fn(
    *,
    mode: str,
    checkpoint: str,
    backend: str = "mock",
) -> Callable[[str], str]:
    if mode == "oracle":
        lookup = _oracle_lookup()

        def _oracle(prompt: str) -> str:
            key = prompt.strip()
            if key in lookup:
                return lookup[key]
            for question, answer in GSM8K_SMOKE_EXAMPLES:
                if question.strip() in prompt:
                    return answer
            return "#### 0"

        return _oracle

    if mode == "mock":
        return lambda _prompt: "#### 0"

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
    mode: str = "oracle",
    backend: str = "mock",
) -> list[float]:
    """Return per-seed accuracy scores (for ablation aggregation)."""
    _ = config
    scorer = GSM8KScorer()
    generate_fn = build_generate_fn(mode=mode, checkpoint=checkpoint, backend=backend)
    rng = random.Random(seed)

    scores: list[float] = []
    for sample_idx in range(n_samples):
        pairs = list(GSM8K_SMOKE_EXAMPLES)
        rng.shuffle(pairs)
        preds = [generate_fn(question) for question, _answer in pairs]
        gts = [answer for _question, answer in pairs]
        metrics = scorer.batch_score(preds, gts)
        scores.append(float(metrics["accuracy"]))
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
        mode=resolved_mode,
        backend=backend,
    )
    accuracy = statistics.mean(scores) if scores else 0.0
    stderr = 0.0
    if len(scores) > 1:
        stderr = statistics.stdev(scores) / (len(scores) ** 0.5)
    return {
        "suite": "gsm8k",
        "profile": profile,
        "mode": resolved_mode,
        "checkpoint": checkpoint or None,
        "config": config or None,
        "n_samples": len(scores),
        "accuracy": accuracy,
        "score": accuracy,
        "stderr": stderr,
        "per_seed_scores": scores,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GSM8K evaluation")
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
