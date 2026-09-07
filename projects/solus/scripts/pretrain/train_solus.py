#!/usr/bin/env python
"""Solus pretrain / SFT single-command runner.

Usage
-----
Full pretrain (100B tokens, 8× A100):
    python scripts/pretrain/train_solus.py --config configs/solus-7b.yaml

SFT on a JSONL chat dataset:
    python scripts/pretrain/train_solus.py --config configs/solus-7b.yaml --mode sft --jsonl data/sft/train.jsonl

Resume from checkpoint:
    python scripts/pretrain/train_solus.py --config configs/solus-7b.yaml --resume checkpoints/step020000

Quit early smoke test (5 steps on tiny dummy batch to verify pipeline):
    python scripts/pretrain/train_solus.py --config configs/solus-7b.yaml --quick
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import torch

# Add project root to sys.path so src.* imports work when run from scripts/
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from src.modeling.config import SolusConfig
from src.modeling.model import SolusForCausalLM
from src.training.pretrain_trainer import train_solus


def main():
    parser = argparse.ArgumentParser(description="Train Solus-7B")
    parser.add_argument("--config",   type=str, default=str(BASE_DIR / "configs/solus-7b.yaml"),
                        help="Path to YAML config file")
    parser.add_argument("--resume",   type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--jsonl",    type=str, default=None,
                        help="JSONL path for SFT mode")
    parser.add_argument("--output",   type=str, default=str(BASE_DIR / "checkpoints"),
                        help="Output directory for checkpoints + logs")
    parser.add_argument("--quick",    action="store_true",
                        help="Run a 5-step smoke test (dummy batch, no real data)")

    args = parser.parse_args()
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    args.output = str(Path(args.output))

    # Print summary, warn if no GPU
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available()  else 1
    print(f"\n{'='*55}")
    print(f"  SOLUS-7B Training")
    print(f"  Config : {args.config}")
    print(f"  Output : {args.output}")
    print(f"  GPUs   : {n_gpus}")
    print(f"  Mode   : {'QUICK SMOKE' if args.quick else 'FULL'}")
    print(f"{'='*55}\n")

    train_solus(args.config, args.output, resume_from=args.resume, quick=args.quick)


if __name__ == "__main__":
    main()
