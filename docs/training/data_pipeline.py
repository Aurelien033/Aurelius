#!/usr/bin/env python3
"""Data pipeline — stream a HF text corpus → uint16 .npy token shards (the format the trainer needs).

Validated format: the training readiness spike proved AureliusTransformer + TokenizedShardDataset train
on exactly these shards. This extends the spike's make_shard() to STREAM a real corpus (fineweb-edu,
the standard high-quality English source the corpus digests recommend) without a full download, writing
fixed-size shards. Tokenizer = GPT-2 (vocab 50257, matches config_100m).

Pin (config_100m): ~2.1B tokens train (ADR-3 20:1 @ 92M) + ~10M val.
Run (needs network; CPU-only, no GPU):
  python docs/training/data_pipeline.py --out data/pretrain-100m --train_tokens 2_100_000_000 --val_tokens 10_000_000
"""
import argparse, json
from pathlib import Path
import numpy as np

SHARD_TOKENS = 100_000_000   # 100M tokens/shard (~200MB uint16); resumable per shard


def stream_tokenize(dataset, split, tokenizer, target_tokens, out_dir, tag):
    out_dir.mkdir(parents=True, exist_ok=True)
    buf, written, shard_i = [], 0, 0
    eot = tokenizer.eos_token_id or 50256
    for ex in dataset:
        ids = tokenizer(ex["text"])["input_ids"] + [eot]
        buf.extend(ids)
        while len(buf) >= SHARD_TOKENS:
            arr = np.array(buf[:SHARD_TOKENS], dtype=np.uint16)
            np.save(out_dir / f"{tag}_{shard_i:04d}.npy", arr)
            buf = buf[SHARD_TOKENS:]; written += SHARD_TOKENS; shard_i += 1
            print(f"  [{tag}] shard {shard_i} | {written:,}/{target_tokens:,} tokens", flush=True)
            if written >= target_tokens: return written
    if buf:  # flush remainder
        np.save(out_dir / f"{tag}_{shard_i:04d}.npy", np.array(buf, dtype=np.uint16)); written += len(buf)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pretrain-100m")
    ap.add_argument("--train_tokens", type=int, default=2_100_000_000)
    ap.add_argument("--val_tokens", type=int, default=10_000_000)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--subset", default="sample-10BT")
    a = ap.parse_args()
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    out = Path(a.out)
    ds = load_dataset(a.dataset, name=a.subset, split="train", streaming=True)
    it = iter(ds)
    nval = stream_tokenize((next(it) for _ in iter(int, 1)), "val", tok, a.val_tokens, out/"val", "val")
    ntr = stream_tokenize((next(it) for _ in iter(int, 1)), "train", tok, a.train_tokens, out/"train", "train")
    card = {"dataset": a.dataset, "subset": a.subset, "tokenizer": "gpt2/50257",
            "format": "uint16 .npy shards (TokenizedShardDataset)", "shard_tokens": SHARD_TOKENS,
            "train_tokens": ntr, "val_tokens": nval, "ratio_at_92M": round(ntr/92e6, 1)}
    (out/"data_card.json").write_text(json.dumps(card, indent=2))
    print(f"DONE: train {ntr:,} val {nval:,} -> {out}/  (ratio {card['ratio_at_92M']}:1)")


if __name__ == "__main__":
    main()
