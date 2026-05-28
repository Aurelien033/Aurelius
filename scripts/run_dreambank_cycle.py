#!/usr/bin/env python
"""DreamBank dry-run cycle runner.

Usage:
    python scripts/run_dreambank_cycle.py --dry-run --cycles 2 --seed "test"

Outputs JSON to stdout with cycle statistics. Has no network calls or external
model dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

# Ensure project root is on sys.path when invoked as `python scripts/...`
_script_root = Path(__file__).resolve().parent.parent
if str(_script_root) not in sys.path:
    sys.path.insert(0, str(_script_root))

import torch

from src.alignment.dreambank import DreamBankConfig, DreamBankController, DreamSeed
from src.memory.hlm_bank import HLMPreferenceBank, HLMPreferenceBankConfig


def deterministic_generate(prompt: str, temperature: float) -> str:
    """Deterministic fake generator for dry runs."""
    h = hashlib.md5(f"{prompt}@{temperature}".encode()).hexdigest()
    return f"gen[{h[:8]}]_t{temperature}_response"


def deterministic_score(prompt: str, response: str) -> float:
    """Deterministic fake scorer for dry runs."""
    h = int(hashlib.md5(f"{prompt}:{response}".encode()).hexdigest()[:8], 16)
    return (h % 1000) / 1000.0


def deterministic_embed(text: str, dim: int = 64) -> torch.Tensor:
    """Deterministic fake embedding for dry runs."""
    h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    torch.manual_seed(h)
    return torch.randn(dim)


def main() -> None:
    parser = argparse.ArgumentParser(description="DreamBank dry-run cycle runner")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Dry run mode (no real model)")
    parser.add_argument("--cycles", type=int, default=1, help="Number of dream cycles")
    parser.add_argument("--seed", type=str, default="test", nargs="+", help="Seed prompts")
    parser.add_argument("--bank-dim", type=int, default=64, help="Bank dimension")
    parser.add_argument("--bank-size", type=int, default=14, help="Bank slot count")

    args = parser.parse_args()

    if args.cycles < 0:
        print(f"Error: --cycles must be >= 0, got {args.cycles}", file=sys.stderr)
        sys.exit(1)

    dim = args.bank_dim
    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=args.bank_size, bank_dim=dim))
    ctrl = DreamBankController(bank)

    total_writes = 0
    all_args = args.seed if hasattr(args.seed, "__iter__") and not isinstance(args.seed, str) else [args.seed]
    result = None

    for _ in range(args.cycles):
        seeds = [DreamSeed(prompt=s) for s in all_args]
        result = ctrl.run_cycle(
            seeds=seeds,
            generate_fn=deterministic_generate,
            score_fn=deterministic_score,
            embed_fn=lambda t: deterministic_embed(t, dim),
        )
        total_writes += result.writes

    output = {
        "dry_run": True,
        "cycles": args.cycles,
        "seeds": all_args,
        "bank_dim": dim,
        "bank_size": args.bank_size,
        "total_writes": total_writes,
        "bank_fill": int(bank.strengths.gt(0).sum().item()),
        "mean_margin": round(result.mean_margin, 4) if result else 0.0,
        "telemetry": bank.telemetry(),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
