#!/usr/bin/env python3
"""Training readiness spike — proves the repo training path actually TRAINS (reduces loss).

Motivation: the repo has a deep trainer suite but ZERO real checkpoints (only a loss-0.0, step-2
smoke). Before any 100M/1B spend, verify end-to-end that: (1) AureliusConfig + AureliusTransformer
build, (2) the uint16 .npy + TokenizedShardDataset data path works, (3) a real training loop drives
the loss DOWN. Uses the REPO's own components (not a reimplementation). Runs on CPU (tiny model) so it
needs no GPU and won't contend with other jobs. PASS = final loss << initial loss.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/Users/christienantonio/aurelius")
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.data.tokenized_loader import TokenizedShardDataset
from torch.utils.data import DataLoader

torch.set_num_threads(4); torch.manual_seed(0)
OUT = Path("/Users/christienantonio/aurelius/docs/training/spike_artifacts"); OUT.mkdir(parents=True, exist_ok=True)
SEQ, BATCH, STEPS, LOG = 128, 4, 120, 15


def make_shard(text_path, n_chars=2_000_000):
    """Tokenize a slice of real text -> uint16 .npy shard (validates the data format the trainer needs)."""
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    text = Path(text_path).read_text()[:n_chars]
    ids = tok(text)["input_ids"]
    arr = np.array(ids, dtype=np.uint16)   # the exact format TokenizedShardDataset expects
    p = OUT / "wiki_shard.npy"; np.save(p, arr)
    print(f"  tokenized {len(text):,} chars -> {len(arr):,} tokens (uint16) -> {p.name}")
    return p


def main():
    print("=" * 60); print("TRAINING READINESS SPIKE"); print("=" * 60)
    # ---- data path ----
    shard = make_shard("/tmp/wiki_train.txt")
    ds = TokenizedShardDataset([shard], seq_len=SEQ)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)
    print(f"  TokenizedShardDataset: {len(ds):,} windows OK")

    # ---- model (repo's own) ----
    cfg = AureliusConfig(d_model=256, n_layers=4, n_heads=4, n_kv_heads=2, head_dim=64,
                         d_ff=688, vocab_size=50257, max_seq_len=SEQ, tie_embeddings=True)
    model = AureliusTransformer(cfg)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"  AureliusTransformer built: {nparams/1e6:.1f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
    model.train()
    losses = []; t0 = time.time(); it = iter(dl)
    for step in range(STEPS):
        try: xb, yb = next(it)
        except StopIteration: it = iter(dl); xb, yb = next(it)
        loss, logits, _ = model(xb, labels=yb) if _accepts_labels(model) else (None, model(xb)[1], None)
        if loss is None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), yb[:, :-1].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        losses.append(loss.item())
        if step % LOG == 0 or step == STEPS - 1:
            print(f"  step {step:3d}  loss {loss.item():.3f}  ({(time.time()-t0)/(step+1):.2f}s/step)")

    init = np.mean(losses[:5]); final = np.mean(losses[-5:])
    drop = init - final
    print(f"\n  initial loss ~{init:.3f}  ->  final ~{final:.3f}  (drop {drop:.3f})")
    verdict = "PASS — trainer reduces loss; the training path WORKS" if drop > 1.0 else \
              "FAIL — loss did not drop; training path BROKEN, do not scale"
    print(f"  VERDICT: {verdict}")
    (OUT / "spike_result.txt").write_text(f"params={nparams}\ninit={init:.3f}\nfinal={final:.3f}\ndrop={drop:.3f}\nverdict={verdict}\n")
    return 0 if drop > 1.0 else 1


def _accepts_labels(model):
    import inspect
    return "labels" in inspect.signature(model.forward).parameters


if __name__ == "__main__":
    sys.exit(main())
