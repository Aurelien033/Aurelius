#!/usr/bin/env python3
"""Skip-native pretraining (OBL-063 / skipnative_preregistration.yaml): train a from-scratch model to be
skip-tolerant via SANDWICH + self-DISTILLATION + a skip-probability CURRICULUM, using the transformer.py
`skip_layers` hook. Tests whether routability can be TRAINED IN (vs the three frozen-base nulls). Includes a
DENSE-TWIN control (--no_skip) — the contrast is the result.

Recipe (designed to avoid the co-train's 'uniform mediocrity' level-down):
  per step: DENSE forward (skip=[]) -> CE loss AND serves as the teacher; SKIP forward (a curriculum-sampled
  routable set) -> CE loss + KL(dense_teacher || skip). Loss = CE_dense + CE_skip + lambda*KL. The dense path
  keeps full quality (sandwich); the skip path is anchored to dense (distill) but DIFFERENT inputs get
  DIFFERENT sampled sets -> trains for SELECTABLE, not uniform, robustness.

Run (GPU, after tokenizing a corpus to uint16 .npy):
  python docs/training/skipnative_pretrain.py --config configs/config_100m.yaml --corpus data/corpus.npy \
     --steps 4000 --out checkpoints/skipnative
  dense-twin control:  ... --no_skip --out checkpoints/dense_twin
Then: a routability analysis loads the checkpoint and tests the cheap per-input selector (H-SN-0).
"""
import argparse, sys, random
from pathlib import Path
from dataclasses import fields
import numpy as np, torch, torch.nn.functional as F, yaml
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.model.transformer import AureliusTransformer
from src.model.config import AureliusConfig


def build_model(cfg_path):
    m = yaml.safe_load(open(cfg_path))["model"]
    accepted = {f.name for f in fields(AureliusConfig)}
    cfg = AureliusConfig(**{k: v for k, v in m.items() if k in accepted})
    return AureliusTransformer(cfg), cfg


def sample_skip(rng, routable, dist):
    ks, ws = zip(*dist.items()); k = rng.choices(ks, weights=ws, k=1)[0]
    return sorted(rng.sample(routable, k)) if k else []


def batches(arr, seq, bs, dev, rng):
    n = bs * (seq + 1)
    while True:
        s = rng.randint(0, len(arr) - n - 1)
        t = torch.tensor(arr[s:s + n].astype("int64")).view(bs, seq + 1).to(dev)
        yield t[:, :-1], t[:, 1:]


@torch.no_grad()
def routability_headroom(model, data, routable, dist, dev, n_batches=8, n_cand=8):
    """On held-out batches: dense loss vs random-skip vs ORACLE-per-batch (best of n_cand sampled sets).
    oracle << random gap = there IS per-input structure to route (the precondition for a cheap selector)."""
    model.eval(); rng = random.Random(99)
    d, r, o = [], [], []
    for _ in range(n_batches):
        xb, yb = next(data)
        ld, _, _ = model(input_ids=xb, labels=yb); d.append(ld.item())
        cand = [sample_skip(rng, routable, {k: v for k, v in dist.items() if k}) for _ in range(n_cand)]
        losses = [model(input_ids=xb, labels=yb, skip_layers=s)[0].item() for s in cand]
        r.append(float(np.mean(losses))); o.append(float(np.min(losses)))
    model.train()
    return float(np.mean(d)), float(np.mean(r)), float(np.mean(o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/config_100m.yaml"))
    ap.add_argument("--corpus", default=None, help="uint16 .npy token ids; omitted => RANDOM (smoke only)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--no_skip", action="store_true", help="DENSE-TWIN control: standard pretraining")
    ap.add_argument("--skip_dist", default="0:0.5,1:0.2,2:0.2,4:0.1")
    ap.add_argument("--distill", type=float, default=1.0, help="lambda for KL(dense||skip)")
    ap.add_argument("--curriculum", type=int, default=1000, help="steps to ramp skip prob 0->1")
    ap.add_argument("--routable", default=None, help="comma layer idxs; default = middle band")
    ap.add_argument("--out", default="checkpoints/skipnative")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    rng = random.Random(1337); torch.manual_seed(1337)
    if a.smoke: a.steps, a.bs, a.seq, a.curriculum = 12, 2, 64, 4

    model, cfg = build_model(a.config); model = model.to(dev).train()
    nL, V = cfg.n_layers, cfg.vocab_size
    routable = [int(x) for x in a.routable.split(",")] if a.routable else list(range(nL // 4, nL - nL // 4))
    dist = {int(k): float(v) for k, v in (kv.split(":") for kv in a.skip_dist.split(","))}
    print(f"skipnative {dev}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, n_layers={nL}, "
          f"routable={routable}, mode={'DENSE-TWIN (no skip)' if a.no_skip else dist}", flush=True)

    if a.corpus and Path(a.corpus).exists():
        arr = np.load(a.corpus)
    else:
        if not a.smoke: print("  WARN: no corpus -> RANDOM data (smoke/debug only; not a real run)", flush=True)
        arr = np.random.randint(0, V, size=max(a.bs * (a.seq + 1) * 6, 100_000)).astype(np.uint16)
    data = batches(arr, a.seq, a.bs, dev, random.Random(7))
    heldout = batches(arr, a.seq, a.bs, dev, random.Random(54321))   # disjoint RNG => held-out windows
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95))
    Path(a.out).mkdir(parents=True, exist_ok=True)

    for step in range(a.steps):
        xb, yb = next(data)
        loss_dense, logits_dense, _ = model(input_ids=xb, labels=yb)
        loss = loss_dense
        if not a.no_skip:
            p_skip = min(1.0, step / max(1, a.curriculum))                 # curriculum ramp
            sk = sample_skip(rng, routable, dist) if rng.random() < p_skip else []
            if sk:
                loss_skip, logits_skip, _ = model(input_ids=xb, labels=yb, skip_layers=sk)
                kl = F.kl_div(F.log_softmax(logits_skip.reshape(-1, V).float(), -1),
                              F.log_softmax(logits_dense.detach().reshape(-1, V).float(), -1),
                              log_target=True, reduction="batchmean")
                loss = loss_dense + loss_skip + a.distill * kl              # SANDWICH + DISTILL
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 50 == 0 or step == a.steps - 1:
            print(f"  step {step:5d}  loss {loss.item():.3f}  dense {loss_dense.item():.3f}", flush=True)

    dn, ra, orc = routability_headroom(model, heldout, routable, dist, dev)
    print(f"  ROUTABILITY HEADROOM (held-out): dense {dn:.3f} | random-skip {ra:.3f} | oracle-per-batch {orc:.3f} "
          f"(oracle<<random gap = per-input structure exists)", flush=True)
    torch.save({"model": model.state_dict(), "config_path": str(a.config), "routable": routable,
                "no_skip": a.no_skip, "headroom": {"dense": dn, "random": ra, "oracle": orc}}, Path(a.out) / "ckpt.pt")
    print(f"  saved {a.out}/ckpt.pt — next: per-input cheap-selector routability test (H-SN-0) on this + the dense twin.", flush=True)


if __name__ == "__main__":
    main()
