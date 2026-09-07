#!/usr/bin/env python3
"""Plot ablation study results.

Usage:
    python scripts/plot_ablation.py docs/reproducibility/results/ablation_scores.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_results(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def plot_ascii_table(results: list[dict]) -> None:
    benchmarks = sorted({row["benchmark"] for row in results})
    configs = ["baseline", "tier1_only", "tier12", "full_amc"]

    header = f"{'config':<14}" + "".join(f"{bench:<14}" for bench in benchmarks)
    print()
    print("=" * len(header))
    print("  ABLATION RESULTS (ASCII)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for cfg in configs:
        row = f"{cfg:<14}"
        for bench in benchmarks:
            matches = [item for item in results if item["config"] == cfg and item["benchmark"] == bench]
            if matches:
                match = matches[0]
                cell = f"{match['score']:.3f}±{match.get('stderr', 0.0):.3f}"
                row += f"{cell:<14}"
            else:
                row += f"{'---':<14}"
        print(row)
    print("=" * len(header))
    print()


def plot_bar_chart(results: list[dict], output: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; outputting ASCII table only.")
        plot_ascii_table(results)
        return

    benchmarks = sorted({row["benchmark"] for row in results})
    configs_ordered = ["baseline", "tier1_only", "tier12", "full_amc"]
    colors = {
        "baseline": "#888888",
        "tier1_only": "#2196F3",
        "tier12": "#4CAF50",
        "full_amc": "#FF9800",
    }
    config_labels = {
        "baseline": "Baseline (no AMC)",
        "tier1_only": "Tier-1 only (SSM)",
        "tier12": "Tier-1+2 (SSM+episodic)",
        "full_amc": "Full AMC (3-tier)",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    n_configs = len(configs_ordered)
    width = 0.8 / n_configs
    x_positions = range(len(benchmarks))

    for index, cfg in enumerate(configs_ordered):
        scores: list[float] = []
        errs: list[float] = []
        for bench in benchmarks:
            matches = [row for row in results if row["config"] == cfg and row["benchmark"] == bench]
            if matches:
                scores.append(float(matches[0]["score"]))
                errs.append(float(matches[0].get("stderr", 0.0)))
            else:
                scores.append(0.0)
                errs.append(0.0)

        ax.bar(
            [pos + index * width for pos in x_positions],
            scores,
            width,
            yerr=errs,
            capsize=3,
            label=config_labels[cfg],
            color=colors[cfg],
            edgecolor="white",
        )

        for bench_index, bench in enumerate(benchmarks):
            match = next(
                (row for row in results if row["config"] == cfg and row["benchmark"] == bench),
                None,
            )
            if match and match.get("p_value_vs_baseline") is not None:
                if float(match["p_value_vs_baseline"]) < 0.05:
                    ax.text(
                        bench_index + index * width,
                        scores[bench_index] + errs[bench_index] + 0.02,
                        "*",
                        ha="center",
                        fontsize=12,
                        weight="bold",
                    )

    ax.set_xticks([pos + 0.4 - width / 2 for pos in x_positions])
    ax.set_xticklabels(benchmarks, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Aurelius AMC — Ablation Study: Effect of Each Memory Tier")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(output, dpi=150)
    if output.suffix == ".pdf":
        fig.savefig(output.with_suffix(".png"), dpi=150)
    print(f"Figure saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_jsonl", help="Ablation results JSONL")
    parser.add_argument(
        "--output",
        default=None,
        help="Output figure path (default: results_jsonl with .pdf extension)",
    )
    args = parser.parse_args()

    results_path = Path(args.results_jsonl)
    output = Path(args.output) if args.output else results_path.with_suffix(".pdf")

    results = load_results(results_path)
    print(f"Loaded {len(results)} results from {results_path}")
    plot_bar_chart(results, output)
    plot_ascii_table(results)


if __name__ == "__main__":
    main()
