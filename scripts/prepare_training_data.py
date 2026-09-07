#!/usr/bin/env python3
"""Prepare AMC training data.

Pipeline:
1. Stream documents (RedPajama JSONL or synthetic fallback)
2. Tokenize
3. Pack fixed-length sequences
4. Annotate per-token importance (heuristic)
5. Train/eval split on disk (numpy memmap + manifest.json)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable


TOPIC_WEIGHTS: dict[str, float] = {
    "Aurelius": 0.7,
    "AMC": 0.7,
    "Tier-1": 0.8,
    "Tier-2": 0.8,
    "Tier-3": 0.8,
    "differentiable": 0.75,
    "Gumbel": 0.7,
    "RoPE": 0.6,
    "Constitutional": 0.85,
    "surprise": 0.7,
    "decay": 0.6,
    "erase": 0.6,
    "write": 0.6,
    "MLA": 0.65,
    "PackKV": 0.6,
    "Yggdrasil": 0.6,
    "Nightjar": 0.6,
}


def get_tokenizer(model_name: str = "NousResearch/Llama-2-7b-hf"):
    """Lazy-load tokenizer; fall back to word-hash tokenizer for tests."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to load HF tokenizer ({exc}). Using fallback.", file=sys.stderr)
        return _fallback_tokenizer()


def _fallback_tokenizer():
    class FallbackTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        bos_token_id = 2

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            _ = add_special_tokens
            return [abs(hash(word)) % 120000 + 3 for word in text.split()]

        def decode(self, ids: list[int]) -> str:
            return " ".join(f"<{token_id}>" for token_id in ids)

    return FallbackTokenizer()


def stream_redpajama_sample(*, n_docs: int = 50_000, seed: int = 42) -> Iterator[str]:
    """Stream documents from REDPAJAMA_PATH or synthetic fallback."""
    rng = random.Random(seed)
    rp_path = os.environ.get("REDPAJAMA_PATH", "")
    if rp_path and Path(rp_path).is_file():
        with Path(rp_path).open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= n_docs:
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text", "")
                if isinstance(text, str) and text.strip():
                    yield text
        return

    print(
        "REDPAJAMA_PATH not set. Generating synthetic AMC corpus for testing.",
        file=sys.stderr,
    )
    topics = [
        "The Aurelius project uses a per-layer differentiable memory hierarchy.",
        "AMC Tier-1 is the SSM working memory at each transformer layer.",
        "Tier-2 is surprise-gated episodic storage across turns.",
        "Tier-3 is trust-aware durable long-term storage.",
        "The promotion gate is differentiable via Gumbel-softmax.",
        "RoPE is applied to attention Q/K but NOT to SSM state.",
        "Constitutional memory encodes safety principles as permanent LTS entries.",
        "The surprise head predicts which turns are memory-worthy.",
    ]
    for _ in range(n_docs):
        doc = " ".join(rng.choice(topics) for _ in range(rng.randint(20, 60)))
        yield doc


def annotate_importance_per_token(
    text: str,
    token_ids: list[int],
    tokenizer: Any,
    *,
    base_weight: float = 0.05,
) -> list[float]:
    """Assign per-token importance aligned to source words/sentences."""
    _ = tokenizer
    words = re.split(r"\s+", text.strip())
    word_weights: list[float] = []
    for word in words:
        weight = base_weight
        lowered = word.lower()
        for term, term_weight in TOPIC_WEIGHTS.items():
            if term.lower() in lowered:
                weight = max(weight, term_weight)
        word_weights.append(weight)

    if len(token_ids) == len(word_weights):
        return word_weights

    importances: list[float] = []
    for index in range(len(token_ids)):
        word_index = min(index, len(word_weights) - 1) if word_weights else 0
        importances.append(word_weights[word_index] if word_weights else base_weight)
    return importances


def _sha256_prefix(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()[:16]


def pack_sequences(
    tokenizer: Any,
    doc_stream: Iterator[str],
    *,
    max_seq_len: int,
    n_sequences: int,
    output_prefix: Path,
    annotate: bool = True,
) -> dict[str, Any]:
    """Pack documents into memmap arrays and return split manifest metadata."""
    input_ids_path = output_prefix.parent / f"{output_prefix.name}_input_ids.bin"
    importance_path = output_prefix.parent / f"{output_prefix.name}_importance.bin"
    input_ids_path.parent.mkdir(parents=True, exist_ok=True)

    input_ids_mmap = np.memmap(
        input_ids_path,
        dtype=np.uint32,
        mode="w+",
        shape=(n_sequences, max_seq_len),
    )
    importance_mmap = np.memmap(
        importance_path,
        dtype=np.float32,
        mode="w+",
        shape=(n_sequences, max_seq_len),
    )

    token_buffer: list[int] = []
    importance_buffer: list[float] = []
    packed_count = 0

    for doc in tqdm(doc_stream, desc=f"Packing {output_prefix.name}"):
        if packed_count >= n_sequences:
            break
        if not doc.strip():
            continue
        try:
            tokens = tokenizer.encode(doc, add_special_tokens=False)
        except Exception:  # noqa: S112 - tokenizer errors on malformed docs are skipped by design during data prep
            continue
        if not tokens:
            continue

        importances = (
            annotate_importance_per_token(doc, tokens, tokenizer)
            if annotate
            else [0.05] * len(tokens)
        )
        token_buffer.extend(tokens)
        importance_buffer.extend(importances)

        while len(token_buffer) >= max_seq_len and packed_count < n_sequences:
            seq_tokens = token_buffer[:max_seq_len]
            seq_importance = importance_buffer[:max_seq_len]
            token_buffer = token_buffer[max_seq_len:]
            importance_buffer = importance_buffer[max_seq_len:]

            input_ids_mmap[packed_count] = np.asarray(seq_tokens, dtype=np.uint32)
            importance_mmap[packed_count] = np.asarray(seq_importance, dtype=np.float32)
            packed_count += 1

    del input_ids_mmap
    del importance_mmap

    return {
        "n_sequences": packed_count,
        "max_seq_len": max_seq_len,
        "input_ids_path": str(input_ids_path),
        "importance_path": str(importance_path),
        "input_ids_sha256_prefix": _sha256_prefix(input_ids_path),
        "importance_sha256_prefix": _sha256_prefix(importance_path),
        "total_tokens": packed_count * max_seq_len,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AMC training data")
    parser.add_argument("--output-dir", default="data/tokenized/amc_forge_1b")
    parser.add_argument("--train-tokens", type=int, default=10_000_000)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--n-docs", type=int, default=100_000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_total_sequences = max(1, args.train_tokens // args.max_seq_len)
    n_train = max(1, int(n_total_sequences * args.train_split))
    n_eval = max(1, n_total_sequences - n_train)

    print(
        f"Target: {n_total_sequences} sequences × {args.max_seq_len} tokens = "
        f"{n_total_sequences * args.max_seq_len:,} tokens"
    )
    print(f"  train: {n_train} sequences")
    print(f"  eval:  {n_eval} sequences")

    tokenizer = get_tokenizer(args.tokenizer)

    train_manifest = pack_sequences(
        tokenizer,
        stream_redpajama_sample(n_docs=int(args.n_docs * args.train_split), seed=args.seed),
        max_seq_len=args.max_seq_len,
        n_sequences=n_train,
        output_prefix=out_dir / "train",
    )
    eval_manifest = pack_sequences(
        tokenizer,
        stream_redpajama_sample(
            n_docs=int(args.n_docs * (1 - args.train_split)),
            seed=args.seed + 1,
        ),
        max_seq_len=args.max_seq_len,
        n_sequences=n_eval,
        output_prefix=out_dir / "eval",
    )

    manifest = {
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "max_seq_len": args.max_seq_len,
        "train": train_manifest,
        "eval": eval_manifest,
        "total_tokens": train_manifest["total_tokens"] + eval_manifest["total_tokens"],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nData prepared at: {out_dir}")
    print(f"  Train: {train_manifest['total_tokens']:,} tokens ({train_manifest['n_sequences']} seq)")
    print(f"  Eval:  {eval_manifest['total_tokens']:,} tokens ({eval_manifest['n_sequences']} seq)")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
