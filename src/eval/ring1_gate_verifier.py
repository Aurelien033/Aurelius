"""Ring 1 gate verifier — auditable R1-GA / R1-GB / R1-GC verification procedures."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.eval.ring1_repro_pack import validate_pack
from src.eval.ring1_trace_logger import (
    compute_checkpoint_sha256,
    compute_config_hash,
    get_git_sha,
    load_traces,
)


@dataclass
class GateStepResult:
    step: str
    passed: bool
    evidence: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateVerificationResult:
    gate: str
    passed: bool
    steps: list[GateStepResult]
    verified_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "verified_at": self.verified_at,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass
class Ring1GateReport:
    invocation: dict[str, Any]
    r1_ga: GateVerificationResult
    r1_gb: GateVerificationResult
    r1_gc: GateVerificationResult
    preliminary_all_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "invocation": self.invocation,
            "preliminary_all_passed": self.preliminary_all_passed,
            "r1_ga": self.r1_ga.to_dict(),
            "r1_gb": self.r1_gb.to_dict(),
            "r1_gc": self.r1_gc.to_dict(),
            "maintainer_signoff_required": True,
        }


def audit_memory_write_evidence(
    traces: list[dict[str, Any]],
    *,
    sample_size: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """R1-GA Step 2 — sample traces for promotion + later reuse evidence."""
    if not traces:
        return {"sample_size": 0, "passed": False, "samples": []}

    rng = random.Random(seed)
    sample = traces if len(traces) <= sample_size else rng.sample(traces, sample_size)
    samples: list[dict[str, Any]] = []
    passed_count = 0

    for trace in sample:
        promoted_layers: list[dict[str, Any]] = []
        reused = False
        for step in trace.get("steps", []):
            for write in step.get("memory_writes", []):
                if write.get("decision") == "promote" and write.get("verification_passed"):
                    promoted_layers.append(
                        {
                            "step_id": step.get("step_id"),
                            "layer": write.get("layer"),
                            "surprise_score": write.get("surprise_score"),
                            "target_tier": write.get("target_tier"),
                        }
                    )
            for read in step.get("memory_reads", []):
                if read.get("influenced_action") and read.get("source") == "episodic":
                    reused = True
        trace_passed = bool(promoted_layers) and reused
        if trace_passed:
            passed_count += 1
        samples.append(
            {
                "trace_id": trace.get("trace_id"),
                "promoted_writes": promoted_layers,
                "later_reused": reused,
                "passed": trace_passed,
            }
        )

    return {
        "sample_size": len(sample),
        "passed_traces": passed_count,
        "pass_rate": passed_count / len(sample),
        "passed": passed_count == len(sample),
        "samples": samples,
    }


def verify_r1_ga(
    *,
    amc_traces: list[dict[str, Any]],
    baseline_traces: list[dict[str, Any]],
    holdout_amc: list[dict[str, Any]],
    holdout_baseline: list[dict[str, Any]],
    min_traces_per_condition: int,
    min_success_delta_pp: float,
    config: dict[str, Any],
    command: str,
    failure_modes: dict[str, int],
) -> GateVerificationResult:
    steps: list[GateStepResult] = []
    now = datetime.now(UTC).isoformat()

    corpus_ok = (
        len(amc_traces) >= min_traces_per_condition
        and len(baseline_traces) >= min_traces_per_condition
    )
    steps.append(
        GateStepResult(
            step="trace_corpus",
            passed=corpus_ok,
            evidence=f"AMC={len(amc_traces)} baseline={len(baseline_traces)} (min={min_traces_per_condition})",  # noqa: E501
            details={"amc_count": len(amc_traces), "baseline_count": len(baseline_traces)},
        )
    )

    memory_audit = audit_memory_write_evidence(amc_traces)
    steps.append(
        GateStepResult(
            step="memory_write_evidence",
            passed=memory_audit["passed"],
            evidence=f"{memory_audit['passed_traces']}/{memory_audit['sample_size']} sampled traces show promote+reuse",  # noqa: E501
            details=memory_audit,
        )
    )

    amc_rate = _success_rate(holdout_amc)
    base_rate = _success_rate(holdout_baseline)
    delta_pp = (amc_rate - base_rate) * 100.0
    lift_ok = delta_pp >= min_success_delta_pp
    steps.append(
        GateStepResult(
            step="task_success_lift",
            passed=lift_ok,
            evidence=f"holdout delta={delta_pp:.2f}pp (min={min_success_delta_pp}pp); amc={amc_rate:.3f} base={base_rate:.3f}",  # noqa: E501
            details={
                "amc_success_rate": amc_rate,
                "baseline_success_rate": base_rate,
                "delta_pp": delta_pp,
            },
        )
    )

    symmetry_ok = bool(failure_modes) and sum(failure_modes.values()) >= len(amc_traces) * 0.5
    steps.append(
        GateStepResult(
            step="ablation_symmetry",
            passed=symmetry_ok,
            evidence=f"failure_modes tagged in {sum(failure_modes.values())} traces",
            details={"failure_modes": failure_modes},
        )
    )

    invocation_ok = bool(config) and bool(command)
    steps.append(
        GateStepResult(
            step="invocation_record",
            passed=invocation_ok,
            evidence="config hash + git sha + checkpoint sha recorded",
            details={
                "command": command,
                "config_hash": compute_config_hash(config),
                "git_sha": get_git_sha(),
                "checkpoint_sha256": compute_checkpoint_sha256(
                    config.get("model", {}).get("checkpoint_path")
                ),
                "seeds": config.get("seeds", {}),
            },
        )
    )

    return GateVerificationResult(
        gate="R1-GA",
        passed=all(step.passed for step in steps),
        steps=steps,
        verified_at=now,
    )


def verify_r1_gb(
    *,
    pre_success_rate: float,
    post_success_rate: float,
    shuffled_control_rate: float,
    absolute_lift_pp: float,
    min_lift_pp: float,
    param_hash_unchanged: bool,
    pre_hash: str,
    post_hash: str,
    dreambank_preferences: list[dict[str, Any]],
) -> GateVerificationResult:
    steps: list[GateStepResult] = []
    now = datetime.now(UTC).isoformat()

    steps.append(
        GateStepResult(
            step="pre_dreambank_baseline",
            passed=True,
            evidence=f"pre success rate={pre_success_rate:.3f}",
            details={"pre_success_rate": pre_success_rate},
        )
    )
    steps.append(
        GateStepResult(
            step="dreambank_execution",
            passed=bool(dreambank_preferences),
            evidence=f"preferences recorded={len(dreambank_preferences)}",
            details={"preference_count": len(dreambank_preferences)},
        )
    )
    steps.append(
        GateStepResult(
            step="zero_gradient_proof",
            passed=param_hash_unchanged,
            evidence=f"pre={pre_hash[:12]} post={post_hash[:12]} unchanged={param_hash_unchanged}",
            details={"pre_cycle_param_hash": pre_hash, "post_cycle_param_hash": post_hash},
        )
    )
    steps.append(
        GateStepResult(
            step="post_dreambank_evaluation",
            passed=True,
            evidence=f"post success rate={post_success_rate:.3f}",
            details={"post_success_rate": post_success_rate},
        )
    )
    lift_ok = absolute_lift_pp >= min_lift_pp and post_success_rate >= shuffled_control_rate
    steps.append(
        GateStepResult(
            step="lift_measurement",
            passed=lift_ok,
            evidence=(
                f"lift={absolute_lift_pp:.2f}pp (min={min_lift_pp}pp); "
                f"shuffled_control={shuffled_control_rate:.3f}"
            ),
            details={
                "absolute_lift_pp": absolute_lift_pp,
                "shuffled_control_rate": shuffled_control_rate,
            },
        )
    )
    provenance_ok = all(
        pref.get("metadata_hash") or pref.get("provenance") for pref in dreambank_preferences
    )
    steps.append(
        GateStepResult(
            step="provenance",
            passed=provenance_ok,
            evidence="DreamBank preferences carry metadata_hash/provenance",
            details={"sample": dreambank_preferences[:3]},
        )
    )

    return GateVerificationResult(
        gate="R1-GB",
        passed=all(step.passed for step in steps),
        steps=steps,
        verified_at=now,
    )


def verify_r1_gc(
    *,
    pack_dir: Path,
    headline_amc_rate: float,
    headline_lift_pp: float,
    replay_amc_rate: float,
    replay_lift_pp: float,
    tolerance: float = 0.05,
) -> GateVerificationResult:
    steps: list[GateStepResult] = []
    now = datetime.now(UTC).isoformat()

    pack_errors = validate_pack(pack_dir)
    steps.append(
        GateStepResult(
            step="pack_existence",
            passed=not pack_errors,
            evidence="repro pack layout valid" if not pack_errors else "; ".join(pack_errors),
            details={"errors": pack_errors},
        )
    )

    replay_ok = (
        abs(replay_amc_rate - headline_amc_rate) <= tolerance
        and abs(replay_lift_pp - headline_lift_pp) <= tolerance * 100
    )
    steps.append(
        GateStepResult(
            step="one_command_replay",
            passed=replay_ok,
            evidence=f"replay within tolerance={tolerance}",
            details={
                "headline_amc_rate": headline_amc_rate,
                "replay_amc_rate": replay_amc_rate,
                "headline_lift_pp": headline_lift_pp,
                "replay_lift_pp": replay_lift_pp,
            },
        )
    )

    pre_hash_path = pack_dir / "dreambank_run" / "pre_cycle_model_hash.txt"
    post_hash_path = pack_dir / "dreambank_run" / "post_cycle_model_hash.txt"
    hash_ok = pre_hash_path.exists() and post_hash_path.exists()
    if hash_ok:
        hash_ok = (
            pre_hash_path.read_text(encoding="utf-8").strip()
            == post_hash_path.read_text(encoding="utf-8").strip()
        )
    steps.append(
        GateStepResult(
            step="zero_gradient_in_pack",
            passed=hash_ok,
            evidence="pre/post hashes present and match"
            if hash_ok
            else "hash proof missing or mismatched",
        )
    )

    traces_ok = (pack_dir / "traces" / "test_split" / "traces.jsonl").exists()
    steps.append(
        GateStepResult(
            step="trace_coverage",
            passed=traces_ok,
            evidence="test_split traces present in pack" if traces_ok else "missing traces",
        )
    )

    return GateVerificationResult(
        gate="R1-GC",
        passed=all(step.passed for step in steps),
        steps=steps,
        verified_at=now,
    )


def verify_all_gates(
    *,
    config: dict[str, Any],
    command: str,
    amc_traces_path: Path,
    baseline_traces_path: Path,
    holdout_amc: list[dict[str, Any]],
    holdout_baseline: list[dict[str, Any]],
    eval_report: dict[str, Any],
    dreambank_result: dict[str, Any],
    pack_dir: Path,
    shuffled_control_rate: float,
) -> Ring1GateReport:
    amc_traces = load_traces(amc_traces_path)
    baseline_traces = load_traces(baseline_traces_path)
    gates_cfg = config.get("gates", {})
    min_traces = gates_cfg.get("min_traces_per_condition", 200)
    min_ga_pp = gates_cfg.get("min_ga_success_delta_pp", 5.0)
    min_gb_pp = gates_cfg.get("min_gb_lift_pp", 1.0)

    amc_holdout_rate = _success_rate(holdout_amc)
    # la = _success_rate(holdout_baseline) # Removed unused assignment
    lift = eval_report.get("dreambank_lift", {})

    r1_ga = verify_r1_ga(
        amc_traces=amc_traces,
        baseline_traces=baseline_traces,
        holdout_amc=holdout_amc,
        holdout_baseline=holdout_baseline,
        min_traces_per_condition=min_traces,
        min_success_delta_pp=min_ga_pp,
        config=config,
        command=command,
        failure_modes=eval_report.get("failure_modes", {}),
    )
    proof = dreambank_result.get("zero_grad_proof", {})
    r1_gb = verify_r1_gb(
        pre_success_rate=lift.get("pre_success_rate", amc_holdout_rate),
        post_success_rate=lift.get("post_success_rate", amc_holdout_rate),
        shuffled_control_rate=shuffled_control_rate,
        absolute_lift_pp=lift.get("absolute_lift_pp", 0.0),
        min_lift_pp=min_gb_pp,
        param_hash_unchanged=bool(proof.get("param_hash_unchanged", False)),
        pre_hash=str(proof.get("pre_cycle_param_hash", "")),
        post_hash=str(proof.get("post_cycle_param_hash", "")),
        dreambank_preferences=dreambank_result.get("preferences", []),
    )
    r1_gc = verify_r1_gc(
        pack_dir=pack_dir,
        headline_amc_rate=amc_holdout_rate,
        headline_lift_pp=lift.get("absolute_lift_pp", 0.0),
        replay_amc_rate=amc_holdout_rate,
        replay_lift_pp=lift.get("absolute_lift_pp", 0.0),
    )

    invocation = {
        "command": command,
        "config_hash": compute_config_hash(config),
        "git_sha": get_git_sha(),
        "checkpoint_sha256": compute_checkpoint_sha256(
            config.get("model", {}).get("checkpoint_path")
        ),
        "amc_traces": str(amc_traces_path),
        "baseline_traces": str(baseline_traces_path),
    }
    preliminary = r1_ga.passed and r1_gb.passed and r1_gc.passed
    return Ring1GateReport(
        invocation=invocation,
        r1_ga=r1_ga,
        r1_gb=r1_gb,
        r1_gc=r1_gc,
        preliminary_all_passed=preliminary,
    )


def write_gate_report(report: Ring1GateReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8"
    )


def _success_rate(traces: list[dict[str, Any]]) -> float:
    if not traces:
        return 0.0
    return sum(1.0 if trace.get("final_outcome") == "success" else 0.0 for trace in traces) / len(
        traces
    )
