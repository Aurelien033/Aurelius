#!/usr/bin/env python3
"""Scenario calculations for Aurelius Composer intelligence roadmap.

This script deliberately separates:
  - measured repo facts gathered from the live Aurelius checkout
  - exact formula outputs under explicitly named assumptions

No benchmark claims are made here. These are design scenarios.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.composer import CodebaseIndexer, ContextAssembler, ContextBudget, ContextRequest


def pass_at_k(p: float, k: int) -> float:
    return 1.0 - (1.0 - p) ** k


def required_p_for_target(target: float, k: int, selection_accuracy: float = 1.0) -> float | None:
    if target > selection_accuracy:
        return None
    oracle_target = target / selection_accuracy
    return 1.0 - (1.0 - oracle_target) ** (1.0 / k)


def cost_per_solved(
    success: float, rollout_cost: float, verifier_cost: float, k: int = 1
) -> float | None:
    if success <= 0:
        return None
    return (k * (rollout_cost + verifier_cost)) / success


def main() -> None:
    repo = Path("/Users/christienantonio/aurelius")
    out_dir = Path(
        "/Users/christienantonio/Desktop/AI Plans/aurelius-composer-fable5-scenarios-2026-07-01"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    indexer = CodebaseIndexer(repo)
    stats = indexer.build(force=True)
    assembler = ContextAssembler(repo, indexer)
    packet = assembler.assemble(
        ContextRequest(
            task="harden composer diff edit verification and apply-model training",
            explicit_files=["src/composer/composer_agent.py"],
            queries=["DiffEngine", "EditVerifier", "CheckpointRollback", "ContextAssembler"],
        ),
        ContextBudget(max_tokens=32_000, reserve_tokens=4_000, max_file_tokens=4_000),
    )

    chars_per_token = 4.0
    full_repo_est_tokens = math.ceil(stats.total_size_bytes / chars_per_token)
    usable_context_tokens = packet.budget.usable_tokens
    full_to_budget_ratio = full_repo_est_tokens / usable_context_tokens
    full_to_selected_ratio = full_repo_est_tokens / packet.total_estimated_tokens
    quadratic_prefill_ratio = (full_repo_est_tokens / max(packet.total_estimated_tokens, 1)) ** 2

    pass_scenarios = []
    for p in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
        for k in [1, 2, 4, 8, 16, 32, 64]:
            oracle = pass_at_k(p, k)
            pass_scenarios.append(
                {
                    "pass1_assumption": p,
                    "k": k,
                    "oracle_at_k": oracle,
                    "final_if_selection_0_70": oracle * 0.70,
                    "final_if_selection_0_85": oracle * 0.85,
                    "final_if_selection_0_95": oracle * 0.95,
                    "cost_per_solved_units_verifier_0_10": cost_per_solved(oracle, 1.0, 0.10, k),
                }
            )

    required_for_90 = []
    for selection in [0.70, 0.80, 0.85, 0.90, 0.95, 0.99, 1.00]:
        for k in [1, 4, 8, 16, 32, 64, 128]:
            req = required_p_for_target(0.90, k, selection)
            required_for_90.append(
                {"selection_accuracy": selection, "k": k, "required_pass1_for_90_final": req}
            )

    repair_scenarios = []
    for base_p in [0.10, 0.20, 0.30, 0.40]:
        for repair_success_on_fail in [0.10, 0.20, 0.35, 0.50]:
            success = base_p + (1 - base_p) * repair_success_on_fail
            expected_cost = 1.0 + (1 - base_p) * 1.2 + 0.2
            repair_scenarios.append(
                {
                    "base_pass1": base_p,
                    "repair_success_given_initial_fail": repair_success_on_fail,
                    "final_success": success,
                    "expected_cost_units": expected_cost,
                    "cost_per_solved_units": expected_cost / success,
                }
            )

    subagent_scenarios = []
    for base_p in [0.20, 0.30, 0.40]:
        for delta in [0.02, 0.05, 0.10, 0.15]:
            for subagent_calls in [1, 3, 5]:
                success = min(1.0, base_p + delta)
                cost = 1.0 + subagent_calls * 0.15 + 0.1
                subagent_scenarios.append(
                    {
                        "base_pass1": base_p,
                        "absolute_success_delta": delta,
                        "subagent_calls": subagent_calls,
                        "final_success": success,
                        "expected_cost_units": cost,
                        "cost_per_solved_units": cost / success,
                    }
                )

    training_scenarios = []
    for params_b in [7, 10, 14, 30]:
        for traces in [50_000, 100_000, 250_000, 1_000_000]:
            context_tokens = 24_000
            diff_tokens = 1_000
            tokens = traces * (context_tokens + diff_tokens)
            flops = 6 * params_b * 1_000_000_000 * tokens
            training_scenarios.append(
                {
                    "params_b": params_b,
                    "traces": traces,
                    "tokens_per_trace_assumption": context_tokens + diff_tokens,
                    "total_tokens": tokens,
                    "dense_training_flops_6N_rule": flops,
                    "exaFLOP": flops / 1e18,
                }
            )

    data = {
        "measured_repo": {
            "indexed_files": stats.total_files,
            "indexed_symbols": stats.total_symbols,
            "indexed_bytes": stats.total_size_bytes,
            "languages": stats.languages,
            "context_items_selected": len(packet.items),
            "context_selected_tokens_est": packet.total_estimated_tokens,
            "context_usable_tokens": usable_context_tokens,
            "omitted_files_due_to_budget": len(packet.omitted_files),
            "full_repo_tokens_est_chars_per_4": full_repo_est_tokens,
            "full_repo_to_usable_context_ratio": full_to_budget_ratio,
            "full_repo_to_selected_context_ratio": full_to_selected_ratio,
            "quadratic_prefill_full_vs_selected_ratio": quadratic_prefill_ratio,
        },
        "pass_at_k_scenarios": pass_scenarios,
        "required_pass1_for_90_final": required_for_90,
        "repair_loop_scenarios": repair_scenarios,
        "subagent_scenarios": subagent_scenarios,
        "training_compute_scenarios": training_scenarios,
    }

    json_path = out_dir / "scenario_calculations.json"
    json_path.write_text(json.dumps(data, indent=2))

    # Human-readable markdown summary.
    lines = []
    lines.append("# Aurelius Composer Fable-5 Roadmap — Scenario Calculations")
    lines.append("")
    lines.append("Generated: 2026-07-01")
    lines.append("")
    lines.append("## Measured live repo facts")
    m = data["measured_repo"]
    for key, value in m.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Key formulas")
    lines.append("- oracle@K = 1 - (1 - pass@1)^K")
    lines.append("- final_success ≈ oracle@K × selection_accuracy")
    lines.append("- required pass@1 for target = 1 - (1 - target/selection_accuracy)^(1/K)")
    lines.append("- dense training FLOPs ≈ 6 × parameters × tokens")
    lines.append("")
    lines.append("## Required pass@1 to reach 90% final success")
    lines.append("| selection | K | required pass@1 |")
    lines.append("|---:|---:|---:|")
    for row in required_for_90:
        req = row["required_pass1_for_90_final"]
        req_s = "impossible" if req is None else f"{req:.6f}"
        if row["k"] in [1, 8, 32, 128]:
            lines.append(f"| {row['selection_accuracy']:.2f} | {row['k']} | {req_s} |")
    lines.append("")
    lines.append("## Pass@K scenario subset")
    lines.append("| pass@1 | K | oracle@K | final@0.85 selection | cost/solve units |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in pass_scenarios:
        if row["pass1_assumption"] in [0.10, 0.30, 0.50] and row["k"] in [1, 8, 32]:
            lines.append(
                f"| {row['pass1_assumption']:.2f} | {row['k']} | {row['oracle_at_k']:.6f} | "
                f"{row['final_if_selection_0_85']:.6f} | {row['cost_per_solved_units_verifier_0_10']:.3f} |"
            )
    lines.append("")
    lines.append("## Training compute subset")
    lines.append("| params B | traces | tokens | exaFLOP |")
    lines.append("|---:|---:|---:|---:|")
    for row in training_scenarios:
        if row["params_b"] in [10, 14] and row["traces"] in [100_000, 1_000_000]:
            lines.append(
                f"| {row['params_b']} | {row['traces']} | {row['total_tokens']} | {row['exaFLOP']:.3f} |"
            )

    md_path = out_dir / "AURELIUS_COMPOSER_FABLE5_SCENARIO_REPORT.md"
    md_path.write_text("\n".join(lines) + "\n")

    print(json_path)
    print(md_path)
    print(json.dumps(data["measured_repo"], indent=2))


if __name__ == "__main__":
    main()
