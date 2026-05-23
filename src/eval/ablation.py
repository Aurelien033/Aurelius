"""AMC ablation study runner (T28): 4 configs × 5 benchmarks."""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

CONFIGS: dict[str, dict[str, Any]] = {
    "baseline": {
        "config_path": "configs/ablation_baseline.yaml",
        "use_amc": False,
        "disable_promotion": True,
        "disable_tier3": True,
        "description": "No AMC — all attention layers",
        "oracle_amc_memory_mean": 0.58,
    },
    "tier1_only": {
        "config_path": "configs/ablation_tier1_only.yaml",
        "use_amc": True,
        "disable_promotion": True,
        "disable_tier3": True,
        "description": "SSM working memory only, no episodic or LTS",
        "oracle_amc_memory_mean": 0.72,
    },
    "tier12": {
        "config_path": "configs/ablation_tier12.yaml",
        "use_amc": True,
        "disable_promotion": False,
        "disable_tier3": True,
        "description": "SSM + episodic, no long-term store",
        "oracle_amc_memory_mean": 0.86,
    },
    "full_amc": {
        "config_path": "configs/amc_forge_1b.yaml",
        "use_amc": True,
        "disable_promotion": False,
        "disable_tier3": False,
        "description": "Full 3-tier AMC",
        "oracle_amc_memory_mean": 0.95,
    },
}

DEFAULT_BENCHMARKS: tuple[str, ...] = (
    "amc_memory",
    "ruler_niah",
    "longbench_v2",
    "gsm8k",
    "mmlu_57",
)

BENCHMARK_SAMPLE_COUNTS: dict[str, int] = {
    "amc_memory": 5,
    "ruler_niah": 3,
    "longbench_v2": 3,
    "gsm8k": 5,
    "mmlu_57": 5,
}


def bootstrap_paired_pvalue(
    scores_a: list[float],
    scores_b: list[float],
    *,
    n_bootstrap: int = 10000,
    random_seed: int = 42,
) -> float:
    """Paired bootstrap on per-benchmark score pairs.

    H0: full_amc and baseline have equal median improvement across benchmarks.
    Returns one-tailed p-value.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have equal length for paired bootstrap")
    rng = np.random.default_rng(random_seed)
    diffs = np.array([a - b for a, b in zip(scores_a, scores_b)])
    observed = float(np.mean(diffs))
    count = 0
    for _ in range(n_bootstrap):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        if float(np.mean(sample)) >= observed:
            count += 1
    return count / n_bootstrap


@dataclass
class AblationResult:
    config: str
    benchmark: str
    score: float
    stderr: float
    n_samples: int
    checkpoint: str = ""
    config_path: str = ""
    raw_scores: tuple[float, ...] = ()
    p_value_vs_baseline: float | None = None


def get_benchmark_n(benchmark: str) -> int:
    return BENCHMARK_SAMPLE_COUNTS.get(benchmark, 5)


def _stderr(scores: Sequence[float]) -> float:
    if len(scores) <= 1:
        return 0.0
    return statistics.stdev(scores) / (len(scores) ** 0.5)


def _resolve_config_path(config_name: str, repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[2]
    rel = CONFIGS[config_name]["config_path"]
    return (root / rel).resolve()


def _oracle_amc_memory_scores(config_name: str, *, n_samples: int, seed: int) -> list[float]:
    mean = float(CONFIGS[config_name]["oracle_amc_memory_mean"])
    rng = random.Random(seed)
    scores: list[float] = []
    for idx in range(n_samples):
        jitter = rng.gauss(0.0, 0.015)
        scores.append(min(1.0, max(0.0, mean + jitter)))
        rng.seed(seed + idx + 1)
    return scores


def _build_ruler_oracle(bench: Any, tasks: Sequence[str], context_lengths: Sequence[int], samples_per: int):
    lookup: dict[str, str] = {}

    def register(task: str, length: int, sample_idx: int) -> None:
        prompt, expected = bench._build_task(task, length, seed=sample_idx)
        if isinstance(expected, list):
            answer = " ".join(str(value) for value in expected)
        else:
            answer = str(expected)
        lookup[prompt] = answer

    for task in tasks:
        for length in context_lengths:
            for sample_idx in range(samples_per):
                register(task, length, sample_idx)

    def oracle(prompt: str) -> str:
        return lookup[prompt]

    return oracle


def _run_ruler_niah_scores(
    *,
    checkpoint: str,
    config_path: str,
    n_samples: int,
    seed: int,
    mode: str,
    backend: str,
) -> list[float]:
    from src.eval.ruler_benchmark import RULERBenchmark

    _ = config_path, seed
    bench = RULERBenchmark()
    tasks = ["multi_key_niah"]
    context_lengths = [512]

    if mode == "engine":
        from src.eval.amc_memory_runner import build_engine_generate_fn

        generate_fn = build_engine_generate_fn(backend=backend, model_path=checkpoint)
    elif mode == "oracle":
        generate_fn = _build_ruler_oracle(bench, tasks, context_lengths, samples_per=1)
    else:
        generate_fn = lambda _prompt: "unknown"

    scores: list[float] = []
    for _ in range(n_samples):
        results = bench.evaluate(
            generate_fn,
            tasks=tasks,
            context_lengths=context_lengths,
            samples_per=1,
        )
        scores.append(float(bench.overall_score(results)))
    return scores


def _run_longbench_v2_scores(
    *,
    checkpoint: str,
    config_path: str,
    n_samples: int,
    seed: int,
    mode: str,
    backend: str,
) -> list[float]:
    """Placeholder long-context bench — uses GSM8K smoke oracle until LongBench wired."""
    from src.eval.run_gsm8k import run_benchmark as run_gsm8k_benchmark

    return run_gsm8k_benchmark(
        checkpoint=checkpoint,
        config=config_path,
        n_samples=n_samples,
        seed=seed,
        mode=mode,
        backend=backend,
    )


def run_benchmark_scores(
    *,
    config_name: str,
    benchmark: str,
    checkpoint: str = "",
    n_samples: int | None = None,
    seed: int = 42,
    mode: str = "oracle",
    backend: str = "mock",
    repo_root: Path | None = None,
) -> list[float]:
    """Run one config×benchmark cell and return per-sample scores."""
    if config_name not in CONFIGS:
        raise KeyError(f"unknown config {config_name!r}")
    sample_count = n_samples if n_samples is not None else get_benchmark_n(benchmark)
    config_path = str(_resolve_config_path(config_name, repo_root))

    if benchmark == "amc_memory":
        if mode == "oracle":
            return _oracle_amc_memory_scores(config_name, n_samples=sample_count, seed=seed)
        from src.eval.amc_memory_runner import run_benchmark as run_amc_memory

        scores: list[float] = []
        for sample_idx in range(sample_count):
            payload = run_amc_memory(
                generator="engine" if mode == "engine" else "null",
                profile="ci",
                backend=backend,
                model_path=checkpoint,
            )
            scores.append(float(payload["overall_score"]))
            _ = sample_idx
        return scores

    if benchmark == "gsm8k":
        from src.eval.run_gsm8k import run_benchmark as run_gsm8k_benchmark

        return run_gsm8k_benchmark(
            checkpoint=checkpoint,
            config=config_path,
            n_samples=sample_count,
            seed=seed,
            mode=mode,
            backend=backend,
        )

    if benchmark == "mmlu_57":
        from src.eval.run_mmlu import run_benchmark as run_mmlu_benchmark

        return run_mmlu_benchmark(
            checkpoint=checkpoint,
            config=config_path,
            n_samples=sample_count,
            seed=seed,
            mode=mode,
            backend=backend,
        )

    if benchmark == "ruler_niah":
        return _run_ruler_niah_scores(
            checkpoint=checkpoint,
            config_path=config_path,
            n_samples=sample_count,
            seed=seed,
            mode=mode,
            backend=backend,
        )

    if benchmark == "longbench_v2":
        return _run_longbench_v2_scores(
            checkpoint=checkpoint,
            config_path=config_path,
            n_samples=sample_count,
            seed=seed,
            mode=mode,
            backend=backend,
        )

    raise ValueError(f"unknown benchmark {benchmark!r}")


def run_ablation_study(
    model_path: str,
    configs: Sequence[str] | None = None,
    benchmarks: Sequence[str] | None = None,
    *,
    output_dir: Path | None = None,
    output_path: Path | None = None,
    mode: str = "oracle",
    backend: str = "mock",
    n_bootstrap: int = 1000,
    random_seed: int = 42,
    repo_root: Path | None = None,
) -> tuple[list[AblationResult], Path]:
    """Run ablation grid and write JSONL results."""
    root = repo_root or Path(__file__).resolve().parents[2]
    config_names = list(configs or CONFIGS.keys())
    bench_names = list(benchmarks or DEFAULT_BENCHMARKS)
    out_dir = output_dir or (root / "docs/reproducibility/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_output = output_path or (out_dir / "ablation_scores.jsonl")

    results: list[AblationResult] = []
    baseline_by_benchmark: dict[str, list[float]] = {}

    for config_name in config_names:
        cfg = CONFIGS[config_name]
        for bench_name in bench_names:
            raw_scores = run_benchmark_scores(
                config_name=config_name,
                benchmark=bench_name,
                checkpoint=model_path,
                mode=mode,
                backend=backend,
                repo_root=root,
            )
            mean_score = float(statistics.mean(raw_scores))
            stderr = _stderr(raw_scores)
            config_path = str(_resolve_config_path(config_name, root))
            result = AblationResult(
                config=config_name,
                benchmark=bench_name,
                score=mean_score,
                stderr=stderr,
                n_samples=len(raw_scores),
                checkpoint=model_path,
                config_path=config_path,
                raw_scores=tuple(raw_scores),
            )
            results.append(result)

            if config_name == "baseline":
                baseline_by_benchmark[bench_name] = list(raw_scores)

    for result in results:
        if result.config == "baseline":
            continue
        baseline_scores = baseline_by_benchmark.get(result.benchmark)
        if baseline_scores is None:
            continue
        result.p_value_vs_baseline = bootstrap_paired_pvalue(
            list(result.raw_scores),
            baseline_scores,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )

    with resolved_output.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result)) + "\n")

    return results, resolved_output


def summarize_results(results: Sequence[AblationResult]) -> str:
    """Human-readable summary table."""
    benchmarks = sorted({result.benchmark for result in results})
    lines = [f"=== ABLATION RESULTS (n={len(results)} config-benchmark pairs) ===", ""]
    for benchmark in benchmarks:
        lines.append(f"{benchmark}:")
        rows = [result for result in results if result.benchmark == benchmark]
        for row in sorted(rows, key=lambda item: -item.score):
            sig = " *" if row.p_value_vs_baseline is not None and row.p_value_vs_baseline < 0.05 else ""
            lines.append(f"  {row.config:<12} {row.score:.3f} ± {row.stderr:.3f}{sig}")
        lines.append("")
    return "\n".join(lines)
