#!/usr/bin/env python3
"""Ring 1 trace collector CLI — generates valid 4–12 step traces with memory logging."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.ring1_agent import Ring1Agent
from src.eval.ring1_dummy_agent import Ring1DummyAgent
from src.eval.ring1_trace_logger import Ring1TraceLogger, load_traces, validate_trace


def resolve_agent(config: dict):
    agent_type = config.get("agent", {}).get("type", "dummy")
    if agent_type == "integrated":
        return Ring1Agent(config)
    return Ring1DummyAgent(config)


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def collect_traces(
    *,
    config: dict,
    num_traces: int,
    max_steps: int,
    output_dir: Path,
    base_seed: int,
) -> Path:
    checkpoint_path = config.get("model", {}).get("checkpoint_path")
    logger = Ring1TraceLogger(output_dir=output_dir, config=config, checkpoint_path=checkpoint_path)
    agent = resolve_agent(config)

    domains = config.get("domains", ["multi_hop_qa", "tool_use", "closed_world_planning"])
    for index in range(num_traces):
        seed = base_seed + index
        domain = domains[index % len(domains)]
        trace = agent.generate_trace(seed=seed, logger=logger, max_steps=max_steps, domain=domain)
        logger.write_trace(trace)

    return logger._jsonl_path  # noqa: SLF001 — intentional for CLI return path


def validate_output(jsonl_path: Path, sidecar_dir: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    traces = load_traces(jsonl_path)
    for index, payload in enumerate(traces):
        try:
            from src.eval.ring1_trace_logger import Ring1Trace, TraceStep, MCTSStats, MemoryReadEvent, MemoryWriteEvent

            steps = []
            for step_data in payload["steps"]:
                steps.append(
                    TraceStep(
                        step_id=step_data["step_id"],
                        observation=step_data["observation"],
                        mcts=MCTSStats(**step_data["mcts"]),
                        memory_reads=[MemoryReadEvent(**read) for read in step_data["memory_reads"]],
                        memory_writes=[MemoryWriteEvent(**write) for write in step_data["memory_writes"]],
                        action=step_data["action"],
                        reflection=step_data["reflection"],
                        memory_delta_summary=step_data.get("memory_delta_summary", ""),
                        timestamp=step_data.get("timestamp", ""),
                    )
                )
            trace = Ring1Trace(
                trace_id=payload["trace_id"],
                model_checkpoint_sha256=payload["model_checkpoint_sha256"],
                config_hash=payload["config_hash"],
                seed=payload["seed"],
                total_steps=payload["total_steps"],
                domain=payload["domain"],
                steps=steps,
                final_outcome=payload["final_outcome"],
                metadata=payload.get("metadata", {}),
            )
            validate_trace(trace)
            sidecar = sidecar_dir / f"{trace.trace_id}_memory_events.ndjson"
            if not sidecar.exists():
                errors.append(f"Trace {index}: missing sidecar {sidecar.name}")
        except Exception as exc:  # noqa: BLE001 — aggregate validation errors for CLI
            errors.append(f"Trace {index}: {exc}")
    return len(traces), errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Ring 1 traces (Tranche 1/2)")
    parser.add_argument("--config", type=Path, default=Path("configs/ring1_tranche2.yaml"))
    parser.add_argument("--num_traces", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override config base seed")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    base_seed = args.seed if args.seed is not None else config.get("seeds", {}).get("base_seed", 42)
    output_dir = args.output_dir or Path(
        config.get("collection", {}).get("output_dir", "data/ring1_traces/tranche2")
    )

    if args.validate_only:
        jsonl_path = output_dir / "traces.jsonl"
        count, errors = validate_output(jsonl_path, output_dir / "sidecars")
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        print(f"Validated {count} traces successfully.")
        return 0

    jsonl_path = collect_traces(
        config=config,
        num_traces=args.num_traces,
        max_steps=args.max_steps,
        output_dir=output_dir,
        base_seed=base_seed,
    )
    count, errors = validate_output(jsonl_path, output_dir / "sidecars")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"Collected {count} valid traces -> {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
