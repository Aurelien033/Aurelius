"""
Round F — R6 cold-start collapse surrogate (the training-dynamics pre-build gate).
Spec: experiment-2-spec.md v2 §7 conditional cap; risk R6 (notes §10.3).

R6 ("the silent killer"): the architecture trains with a compute-budget regularizer that
fights the accuracy loss (without it, adaptivity never emerges). But the CHEAPEST way for
the optimizer to cut compute may be to COLLAPSE the surprise signal itself to a constant
("surprise becomes a learned constant") — so the gate never fires and you spend uniform
compute anyway. The proposed fix (notes §10.4): STOP-GRADIENT the surprise signal on the
compute-budget path, and anneal the cost term in from ~0.

THREE-ARM EXPERIMENT (the point of this script):
  A. no cost term            -> reference: whatever surprise variance training yields.
  B. cost term, NO stop-grad -> expected FAILURE: surprise variance collapses as lambda rises.
  C. cost term, WITH stop-grad on surprise -> the FIX: variance + predictive power preserved
                                              while compute is still cut.

Per-token soft halting (PonderNet-flavored): continue-prob p_k = sigmoid(w*S_k + b), where
S_k = residual magnitude at loop k (the surprise). Expected loops E[L] = sum_k prod_{j<k}p_j.
Loss = task_loss(expected-halt readout) + lambda(t) * E[L].  In arm C, S_k is detached where
it feeds p_k, so the cost term cannot reduce E[L] by shrinking S.

Tracks across training, per arm: surprise CV (collapse?), corr(surprise, realized benefit)
(predictive power retained?), mean compute E[L] (did it actually cut compute?), task loss.
VERDICT: R6 mitigated iff arm C keeps surprise CV and corr(S,benefit) high (near arm A) AND
cuts compute, while arm B collapses.

Authored without execution — smoke-test with --n 16 --K 6 --steps 300 first.
"""
import argparse, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

class TiedLoop(nn.Module):
    def __init__(self, vocab, dim=64, window=1, drop=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim); self.pos = nn.Embedding(512, dim)
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim); self.v = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))
        self.alpha = nn.Parameter(torch.tensor(0.5)); self.head = nn.Linear(dim, vocab)
        self.halt_w = nn.Parameter(torch.tensor(-1.0)); self.halt_b = nn.Parameter(torch.tensor(2.0))
        self.window, self.dim = window, dim
    def block(self, z):
        n = z.shape[1]; i = torch.arange(n, device=z.device)
        mask = torch.where((i[None]-i[:,None]).abs() <= self.window, 0.0, float("-inf"))
        a = torch.softmax(self.q(z) @ self.k(z).transpose(-1,-2)/self.dim**0.5 + mask, -1) @ self.v(z)
        return a + self.mlp(z)
    def forward(self, x, K, stop_grad_surprise):
        z = self.emb(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        logits_k, S_k = [], []
        for _ in range(K):
            upd = self.block(z); resid = z - upd                       # subtractive residual
            S = resid.norm(dim=-1)                                     # surprise = residual magnitude
            S_k.append(S); logits_k.append(self.head(z))
            z = z - self.alpha * resid
        # continue-probabilities from surprise (detached on the budget path in arm C)
        Suse = [s.detach() if stop_grad_surprise else s for s in S_k]
        p = [torch.sigmoid(self.halt_w * s + self.halt_b) for s in Suse]   # P(continue past loop k)
        return logits_k, S_k, p

def expected_loops(p):                                   # E[L] = sum_k prod_{j<k} p_j
    cum = torch.ones_like(p[0]); E = torch.zeros_like(p[0])
    for pk in p:
        E = E + cum; cum = cum * pk
    return E

def make_batch(b, n, P, g, dev):
    x = torch.randint(0, P, (b, n), generator=g); y = torch.cumsum(x, 1) % P
    return x.to(dev), y.to(dev)

def spearman(a, b):
    ra = a.argsort().argsort().astype(float); rb = b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra*rb).sum() / (math.sqrt((ra*ra).sum()*(rb*rb).sum()) + 1e-12))

def train_arm(name, cost, stop_grad, args, g, dev):
    net = TiedLoop(args.P, args.dim, args.window).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    for step in range(args.steps):
        lam = (0.0 if not cost else args.lam * min(1.0, step / max(1, args.anneal)))   # annealed cost
        x, y = make_batch(args.bs, args.n, args.P, g, dev)
        logits_k, S_k, p = net(x, args.K, stop_grad)
        # expected-halt readout: weight each loop's logits by its halting distribution
        cum = torch.ones(x.shape[:2], device=dev); w = []
        for pk in p:
            w.append(cum * (1 - pk)); cum = cum * pk
        w.append(cum)                                                 # remainder -> last loop
        wsum = sum(w) + 1e-9
        mix = sum(wi.unsqueeze(-1) * lg for wi, lg in zip(w, logits_k + [logits_k[-1]])) / wsum.unsqueeze(-1)
        task = F.cross_entropy(mix.reshape(-1, args.P), y.reshape(-1))
        loss = task + lam * expected_loops(p).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    # ---- diagnostics on a fresh eval batch ----
    with torch.no_grad():
        x, y = make_batch(args.eval_bs, args.n, args.P, g, dev)
        logits_k, S_k, p = net(x, args.K, stop_grad)
        ks = args.k_scout
        S = S_k[ks].reshape(-1).cpu().numpy()
        nll = [F.cross_entropy(logits_k[k].reshape(-1, args.P), y.reshape(-1), reduction="none") for k in range(args.K)]
        B = (nll[ks] - nll[ks+1]).cpu().numpy()                       # realized loop-benefit
        EL = expected_loops(p).mean().item()
        cv = float(S.std() / (S.mean() + 1e-9))
        corr = spearman(S, B)
    print(f"[{name:<22}] surprise CV={cv:.3f}  corr(S,benefit)={corr:.3f}  mean E[loops]={EL:.2f}  (K={args.K})")
    return cv, corr, EL

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=32); p.add_argument("--P", type=int, default=7)
    p.add_argument("--dim", type=int, default=64); p.add_argument("--window", type=int, default=1)
    p.add_argument("--K", type=int, default=10); p.add_argument("--k_scout", type=int, default=2)
    p.add_argument("--steps", type=int, default=2500); p.add_argument("--anneal", type=int, default=1000)
    p.add_argument("--bs", type=int, default=128); p.add_argument("--eval_bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3); p.add_argument("--lam", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(); g = torch.Generator().manual_seed(args.seed); dev = args.device
    assert 0 <= args.k_scout < args.K - 1, f"k_scout must be < K-1 (got {args.k_scout}, K={args.K})"
    print("Three-arm R6 cold-start test (higher CV + corr = signal survived; lower E[loops] = compute cut):\n")
    a = train_arm("A: no cost term",        cost=False, stop_grad=False, args=args, g=g, dev=dev)
    b = train_arm("B: cost, NO stop-grad",  cost=True,  stop_grad=False, args=args, g=g, dev=dev)
    c = train_arm("C: cost, stop-grad fix", cost=True,  stop_grad=True,  args=args, g=g, dev=dev)
    print("\nVERDICT:")
    collapsed_B = b[0] < 0.5 * a[0] or b[1] < 0.5 * a[1]
    fixed_C = c[0] >= 0.8 * a[0] and c[1] >= 0.8 * a[1] and c[2] < 0.9 * args.K
    if collapsed_B and fixed_C:
        print("  R6 reproduced in arm B and MITIGATED by stop-grad in arm C -> conditional-GREEN can be lifted.")
    elif not collapsed_B:
        print("  Arm B did not collapse at this lambda/anneal -> push lambda up / anneal faster to stress R6 properly.")
    else:
        print("  Arm C did NOT preserve the signal while cutting compute -> R6 is a real pre-build blocker; rethink.")
    print("  (Pre-register CV/corr collapse thresholds and the lambda that achieves a target compute cut.)")

if __name__ == "__main__":
    main()
