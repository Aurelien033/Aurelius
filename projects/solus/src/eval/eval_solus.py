#!/usr/bin/env python
"""Solus-7B — evaluation harness.

Benchmarks
----------
  • Perplexity on held-out validation split
  • HellaSwag / WinoGrande  (needs datasets or lm-eval-harness)
  • HumanEval / MBPP (code — needs evaluate repo + OJ data)
  • MMLU (strong broad benchmark; needs methodtool)

Minimal run (perplexity + generation quality):
    python -m src.eval.eval_solus --checkpoint checkpoints/final/ --data data/pretrain/val/
"""
from __future__ import annotations
import argparse
import torch
from pathlib import Path
from typing import Optional

from src.modeling.config import SolusConfig
from src.modeling.model import SolusForCausalLM


def compute_perplexity(model, val_dataloader, accelerator) -> float:
    """Compute corpus-level perplexity on a validation data loader."""
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids  = batch["input_ids"].to(accelerator.device)
            labels     = batch["labels"].to(accelerator.device)
            loss = model(input_ids=input_ids, labels=labels)["loss"]
            losses.append(loss.item())
    return math.exp(sum(losses) / len(losses))


@torch.no_grad()
def run_generation_bench(model, tokenizer, prompts: list[str], **gen_kwargs) -> list[str]:
    """Run generation on a list of prompted input strings."""
    model.eval()
    outputs = []
    for prompt in prompts:
        toks = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        out  = model.generate(toks, **gen_kwargs)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        outputs.append(text.replace(prompt, "").strip())
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Evaluate Solus-7B checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="path to saved checkpoint directory")
    parser.add_argument("--data", type=str, required=True,
                        help="path to validation token shards (.npy)")
    parser.add_argument("--prompts", type=str, default=None,
                        help="optional JSONL of prompts for generation benchmark")
    args = parser.parse_args()

    # Load model
    cfg     = SolusConfig.from_pretrained(args.checkpoint)
    model   = SolusForCausalLM(cfg)
    state   = torch.load(args.checkpoint / "pytorch_model.bin",
                         map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().cuda()

    # Validate perplexity
    from src.data.dataset_builder import build_pretrain_dataloader
    val_loader = build_pretrain_dataloader(args.data, seq_len=2048, batch_size=64)
    ppl = compute_perplexity(model, val_loader)
    print(f"  Perplexity : {ppl:.2f}")

    # Generation benchmark
    if args.prompts:
        import json
        from tqdm import tqdm
        with open(args.prompts) as f:
            prompts = [json.loads(l)["prompt"] for l in f]
        outputs = run_generation_bench(
            model, tokenizer=None,
            prompts=prompts[:20],  # limit for speed
            max_new_tokens=256, temperature=0.7, top_p=0.9,
        )
        for p, o in zip(prompts[:20], outputs):
            print(f"\n  [Q] {p}")
            print(f"  [A] {o[:400]}")
    print("\n  ✓ Eval complete.")


if __name__ == "__main__":
    main()
