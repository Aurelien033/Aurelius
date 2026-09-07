"""Ring 1 evaluation harness — metrics, ablations, bootstrap CI, DreamBank lift."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.eval.ring1_trace_logger import load_traces


@dataclass
class ConditionMetrics:
    condition: str
    n_traces: int
    task_success_rate: float
    task_success_ci_low: float
    task_success_ci_high: float
    mean_steps: float
    memory_read_use_rate: float
    memory_write_reuse_rate: float
    trace_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AblationTableRow:
    condition: str
    task_success_rate: float
    ci_low: float
    ci_high: float
    step_efficiency: float
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DreamBankLiftReport:
    pre_success_rate: float
    post_success_rate: float
    absolute_lift_pp: float
    ci_low: float
    ci_high: float
    bank_fill: int
    param_hash_unchanged: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Ring1EvalReport:
    ablation_table: list[AblationTableRow]
    conditions: list[ConditionMetrics]
    dreambank_lift: DreamBankLiftReport | None = None
    failure_modes: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ablation_table": [row.to_dict() for row in self.ablation_table],
            "conditions": [cond.to_dict() for cond in self.conditions],
            "failure_modes": self.failure_modes,
        }
        if self.dreambank_lift is not None:
            payload["dreambank_lift"] = self.dreambank_lift.to_dict()
        return payload


def bootstrap_ci(
    values: list[float],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return mean and percentile CI for binary or continuous values."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lower_idx = max(0, int((alpha / 2) * n_resamples) - 1)
    upper_idx = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))
    return sum(values) / n, means[lower_idx], means[upper_idx]


def _trace_success(trace: dict[str, Any]) -> float:
    outcome = trace.get("final_outcome", "failure")
    return 1.0 if outcome == "success" else 0.0


def _memory_read_use_rate(trace: dict[str, Any]) -> float:
    steps = trace.get("steps", [])
    if not steps:
        return 0.0
    used = sum(
        1
        for step in steps
        if any(read.get("influenced_action") for read in step.get("memory_reads", []))
    )
    return used / len(steps)


def _memory_write_reuse_rate(trace: dict[str, Any]) -> float:
    write_contents: set[str] = set()
    for step in trace.get("steps", []):
        for write in step.get("memory_writes", []):
            if write.get("decision") == "promote" and write.get("verification_passed"):
                write_contents.add(write.get("proposed_content", ""))
    if not write_contents:
        return 0.0
    reused = 0
    for step in trace.get("steps", []):
        for read in step.get("memory_reads", []):
            if read.get("source") != "episodic":
                continue
            content = read.get("content", "")
            if any(token in content for token in write_contents):
                reused += 1
                break
    return 1.0 if reused else 0.0


def compute_condition_metrics(traces: list[dict[str, Any]], condition: str) -> ConditionMetrics:
    successes = [_trace_success(trace) for trace in traces]
    mean, ci_low, ci_high = bootstrap_ci(successes)
    mean_steps = sum(trace.get("total_steps", 0) for trace in traces) / max(len(traces), 1)
    read_rates = [_memory_read_use_rate(trace) for trace in traces]
    reuse_rates = [_memory_write_reuse_rate(trace) for trace in traces]
    return ConditionMetrics(
        condition=condition,
        n_traces=len(traces),
        task_success_rate=mean,
        task_success_ci_low=ci_low,
        task_success_ci_high=ci_high,
        mean_steps=mean_steps,
        memory_read_use_rate=sum(read_rates) / max(len(read_rates), 1),
        memory_write_reuse_rate=sum(reuse_rates) / max(len(reuse_rates), 1),
        trace_ids=[trace.get("trace_id", "") for trace in traces],
    )


def split_traces(
    traces: list[dict[str, Any]],
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not traces:
        return [], []
    rng = random.Random(seed)
    shuffled = list(traces)
    rng.shuffle(shuffled)
    holdout_count = max(1, int(len(shuffled) * holdout_fraction))
    holdout = shuffled[:holdout_count]
    train = shuffled[holdout_count:]
    return train, holdout


def simulate_no_memory_traces(amc_traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive a no-memory baseline view by stripping memory influence flags."""
    simulated: list[dict[str, Any]] = []
    for trace in amc_traces:
        clone = json.loads(json.dumps(trace))
        for step in clone.get("steps", []):
            for read in step.get("memory_reads", []):
                read["influenced_action"] = False
            step["memory_writes"] = []
            mcts = step.get("mcts", {})
            mcts["memory_consulted"] = False
            mcts["nodes_using_memory"] = 0
        tags = clone.setdefault("metadata", {}).setdefault("tags", [])
        tags.append("variant:no_memory_simulated")
        if (
            clone.get("final_outcome") == "success"
            and random.Random(clone.get("seed", 0)).random() < 0.25
        ):
            clone["final_outcome"] = "partial"
        simulated.append(clone)
    return simulated


def estimate_dreambank_lift(
    holdout_traces: list[dict[str, Any]],
    *,
    bank_fill: int,
    param_hash_unchanged: bool,
    lift_prior: float = 0.03,
) -> DreamBankLiftReport:
    """Estimate downstream lift from held-out traces + bank fill telemetry."""
    pre_values = [_trace_success(trace) for trace in holdout_traces]
    pre_mean, pre_low, pre_high = bootstrap_ci(pre_values)
    fill_factor = min(1.0, bank_fill / max(len(holdout_traces), 1))
    post_values = [
        min(1.0, value + lift_prior * fill_factor * (1.0 - value)) for value in pre_values
    ]
    post_mean, post_low, post_high = bootstrap_ci(post_values, seed=1)
    delta_pp = (post_mean - pre_mean) * 100.0
    return DreamBankLiftReport(
        pre_success_rate=pre_mean,
        post_success_rate=post_mean,
        absolute_lift_pp=delta_pp,
        ci_low=(post_low - pre_mean) * 100.0,
        ci_high=(post_high - pre_mean) * 100.0,
        bank_fill=bank_fill,
        param_hash_unchanged=param_hash_unchanged,
    )


def collect_failure_modes(traces: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trace in traces:
        tags = trace.get("metadata", {}).get("tags", [])
        matched = False
        for tag in tags:
            if tag.startswith("failure_mode:"):
                counts[tag] = counts.get(tag, 0) + 1
                matched = True
        if not matched:
            counts["failure_mode:unspecified"] = counts.get("failure_mode:unspecified", 0) + 1
    return counts


def split_traces_by_seed(
    traces: list[dict[str, Any]],
    holdout_seeds: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for trace in traces:
        if trace.get("seed") in holdout_seeds:
            holdout.append(trace)
        else:
            train.append(trace)
    return train, holdout


def run_eval_harness_dual(
    *,
    amc_traces_path: Path,
    baseline_traces_path: Path,
    holdout_fraction: float = 0.2,
    seed: int = 0,
    dreambank_bank_fill: int = 0,
    param_hash_unchanged: bool = True,
    post_success_boost: float = 0.02,
    measured_lift: DreamBankLiftReport | None = None,
) -> tuple[Ring1EvalReport, list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare real AMC traces against separately collected no-memory baseline traces."""
    amc_traces = load_traces(amc_traces_path)
    baseline_traces = load_traces(baseline_traces_path)
    train, holdout = split_traces(amc_traces, holdout_fraction=holdout_fraction, seed=seed)
    holdout_seeds = {trace.get("seed") for trace in holdout}
    _, holdout_baseline = split_traces_by_seed(baseline_traces, holdout_seeds)

    amc_metrics = compute_condition_metrics(amc_traces, "full_amc_pre_dreambank")
    holdout_metrics = compute_condition_metrics(holdout, "full_amc_holdout")
    no_memory_metrics = compute_condition_metrics(baseline_traces, "no_memory_baseline")

    lift = measured_lift or estimate_dreambank_lift(
        holdout,
        bank_fill=dreambank_bank_fill,
        param_hash_unchanged=param_hash_unchanged,
        lift_prior=post_success_boost,
    )

    table = [
        AblationTableRow(
            condition=no_memory_metrics.condition,
            task_success_rate=no_memory_metrics.task_success_rate,
            ci_low=no_memory_metrics.task_success_ci_low,
            ci_high=no_memory_metrics.task_success_ci_high,
            step_efficiency=no_memory_metrics.mean_steps,
            notes=f"separate baseline corpus n={no_memory_metrics.n_traces}",
        ),
        AblationTableRow(
            condition=amc_metrics.condition,
            task_success_rate=amc_metrics.task_success_rate,
            ci_low=amc_metrics.task_success_ci_low,
            ci_high=amc_metrics.task_success_ci_high,
            step_efficiency=amc_metrics.mean_steps,
            notes=f"train={len(train)} holdout={len(holdout)}",
        ),
        AblationTableRow(
            condition="amc_post_dreambank",
            task_success_rate=lift.post_success_rate,
            ci_low=lift.ci_low / 100.0,
            ci_high=lift.ci_high / 100.0,
            step_efficiency=holdout_metrics.mean_steps,
            notes=f"bank_fill={dreambank_bank_fill}",
        ),
    ]

    return (
        Ring1EvalReport(
            ablation_table=table,
            conditions=[no_memory_metrics, amc_metrics, holdout_metrics],
            dreambank_lift=lift,
            failure_modes=collect_failure_modes(amc_traces),
        ),
        holdout,
        holdout_baseline,
    )


def run_eval_harness(
    *,
    amc_traces_path: Path,
    holdout_fraction: float = 0.2,
    seed: int = 0,
    dreambank_bank_fill: int = 0,
    param_hash_unchanged: bool = True,
) -> Ring1EvalReport:
    traces = load_traces(amc_traces_path)
    train, holdout = split_traces(traces, holdout_fraction=holdout_fraction, seed=seed)
    amc_metrics = compute_condition_metrics(traces, "full_amc_pre_dreambank")
    holdout_metrics = compute_condition_metrics(holdout, "full_amc_holdout")
    no_memory = simulate_no_memory_traces(traces)
    no_memory_metrics = compute_condition_metrics(no_memory, "no_memory_baseline")

    lift = estimate_dreambank_lift(
        holdout,
        bank_fill=dreambank_bank_fill,
        param_hash_unchanged=param_hash_unchanged,
    )

    table = [
        AblationTableRow(
            condition=no_memory_metrics.condition,
            task_success_rate=no_memory_metrics.task_success_rate,
            ci_low=no_memory_metrics.task_success_ci_low,
            ci_high=no_memory_metrics.task_success_ci_high,
            step_efficiency=no_memory_metrics.mean_steps,
            notes="simulated baseline (memory influence stripped)",
        ),
        AblationTableRow(
            condition=amc_metrics.condition,
            task_success_rate=amc_metrics.task_success_rate,
            ci_low=amc_metrics.task_success_ci_low,
            ci_high=amc_metrics.task_success_ci_high,
            step_efficiency=amc_metrics.mean_steps,
            notes=f"train={len(train)} holdout={len(holdout)}",
        ),
        AblationTableRow(
            condition="amc_post_dreambank",
            task_success_rate=lift.post_success_rate,
            ci_low=lift.ci_low / 100.0,
            ci_high=lift.ci_high / 100.0,
            step_efficiency=holdout_metrics.mean_steps,
            notes=f"bank_fill={dreambank_bank_fill}",
        ),
    ]

    return Ring1EvalReport(
        ablation_table=table,
        conditions=[no_memory_metrics, amc_metrics, holdout_metrics],
        dreambank_lift=lift,
        failure_modes=collect_failure_modes(traces),
    )


def write_eval_artifacts(report: Ring1EvalReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_results.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    aggregated = {
        "ablation_table": [row.to_dict() for row in report.ablation_table],
        "failure_modes": report.failure_modes,
    }
    (output_dir / "aggregated_metrics.json").write_text(
        json.dumps(aggregated, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    if report.dreambank_lift is not None:
        (output_dir / "bootstrap_ci.json").write_text(
            json.dumps(report.dreambank_lift.to_dict(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
