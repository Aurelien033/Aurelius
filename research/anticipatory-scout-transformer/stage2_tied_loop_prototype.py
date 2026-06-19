"""
Stage 2 — tied-loop prototype (the BUILD GATE, signal side).
Spec: experiment-2-spec.md v2.

Trains a small WEIGHT-TIED loop with a subtractive/residual update and LOCAL attention
(so information propagates a bounded distance per loop -> later positions provably need
more loops -> per-token loop-benefit variance by construction). Then asks: does a cheap
truncated-loop signal predict where MORE LOOPS reduce loss, BEYOND trivial baselines and
WITHOUT the residual tautology?

Signals:
  S1 = ||subtractive residual|| at scout loop k_scout   (the architecture's native signal)
  S2 = truncated-loop committee (MC-dropout) disagreement at k_scout (JSD)
Benefit:
  B(t) = NLL(loop k) - NLL(loop k+1)   (marginal, signed)
Tautology controls (HARD GATES, spec §4):
  C_cur  = current-NLL at k_scout      |  C_trunc = loop-k_scout readout NLL to gold
  S must predict B AFTER partialling out BOTH; and beat the CALM single-pass entropy
  and current-NLL on the tail metric, with instance-level bootstrap CIs.

Authored without execution — smoke-test with --n 16 --K 6 --steps 200 first.
Task here is a stand-in (local-attention prefix-sum mod P); swap in a real algorithmic
suite per spec §2 for the actual gate.
"""
import argparse, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# ------------------------------- model -------------------------------
class TiedLoopBlock(nn.Module):
    def __init__(self, dim, window, drop=0.1):
        super().__init__()
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim); self.v = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))
        self.alpha = nn.Parameter(torch.tensor(0.5)); self.window = window; self.drop = nn.Dropout(drop)
        self.dim = dim
    def local_mask(self, n, dev):
        i = torch.arange(n, device=dev)
        d = (i[None, :] - i[:, None]).abs()
        return torch.where(d <= self.window, 0.0, float("-inf"))
    def block(self, z):                                  # one application of the tied operator
        n = z.shape[1]
        att = self.q(z) @ self.k(z).transpose(-1, -2) / self.dim**0.5 + self.local_mask(n, z.device)
        a = torch.softmax(att, -1) @ self.v(z)
        return self.drop(a + self.mlp(z))
    def forward(self, z):                                # subtractive/residual update
        return z - self.alpha * (z - self.block(z))

class TiedLoopNet(nn.Module):
    def __init__(self, vocab, dim=64, window=1, K=8, drop=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim); self.pos = nn.Embedding(512, dim)
        self.loop = TiedLoopBlock(dim, window, drop); self.head = nn.Linear(dim, vocab); self.K = K
    def run(self, x, K=None):                            # returns per-loop logits + states
        K = K or self.K
        z = self.emb(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        logits_per_loop, states = [], []
        for _ in range(K):
            states.append(z)
            z = self.loop(z)
            logits_per_loop.append(self.head(z))
        return logits_per_loop, states

# ------------------------------- task (stand-in) -------------------------------
def make_batch(b, n, P, g, dev):
    x = torch.randint(0, P, (b, n), generator=g)
    y = torch.cumsum(x, 1) % P                            # prefix-sum mod P: pos i needs info from 0..i
    return x.to(dev), y.to(dev)

# ------------------------------- stats -------------------------------
def rankf(a): return a.argsort().argsort().astype(float)
def spearman(a, b):
    ra, rb = rankf(a), rankf(b); ra -= ra.mean(); rb -= rb.mean()
    return float((ra*rb).sum() / (math.sqrt((ra*ra).sum()*(rb*rb).sum()) + 1e-12))
def partial_spearman(a, b, c):                            # Spearman(a,b | c), rank-linear residualization
    ra, rb, rc = rankf(a), rankf(b), rankf(c)
    def resid(y, x):
        x = x - x.mean(); beta = (x*(y-y.mean())).sum()/((x*x).sum()+1e-12); return (y-y.mean()) - beta*x
    return spearman(resid(ra, rc), resid(rb, rc))
def boot_ci(fn, arrs, inst, nb=800, seed=0):
    rng = np.random.default_rng(seed); insts = np.unique(inst); vals = []
    for _ in range(nb):
        pick = rng.choice(insts, len(insts), replace=True)
        idx = np.concatenate([np.where(inst == d)[0] for d in pick])
        vals.append(fn(*[a[idx] for a in arrs]))
    return fn(*arrs), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
def tail_capture(S, B, frac=0.2):
    k = max(1, int(frac*len(S)))
    num = B[np.argsort(-S)[:k]].clip(min=0).sum()          # positive benefit captured by top-k by S
    den = max(B[np.argsort(-B)[:k]].clip(min=0).sum(), 1e-9)   # oracle ceiling (positive mass only)
    return float(num / den)

# ------------------------------- train + analyze -------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=32); p.add_argument("--P", type=int, default=7)
    p.add_argument("--dim", type=int, default=64); p.add_argument("--window", type=int, default=1)
    p.add_argument("--K", type=int, default=10); p.add_argument("--k_scout", type=int, default=2)
    p.add_argument("--steps", type=int, default=2000); p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3); p.add_argument("--T", type=int, default=12)
    p.add_argument("--eval_inst", type=int, default=400); p.add_argument("--tail_frac", type=float, default=0.2)
    p.add_argument("--deep_sup", type=int, default=1, help="1=deep supervision (note: can manufacture monotone help)")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    g = torch.Generator().manual_seed(args.seed); dev = args.device
    net = TiedLoopNet(args.P, args.dim, args.window, args.K).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    for step in range(args.steps):
        x, y = make_batch(args.bs, args.n, args.P, g, dev)
        logits, _ = net.run(x)
        if args.deep_sup:
            loss = sum(F.cross_entropy(l.reshape(-1, args.P), y.reshape(-1)) for l in logits) / len(logits)
        else:
            loss = F.cross_entropy(logits[-1].reshape(-1, args.P), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, args.steps//5) == 0:
            acc = (logits[-1].argmax(-1) == y).float().mean().item()
            print(f"  step {step:5d}  loss {loss.item():.4f}  final-loop acc {acc:.3f}")

    # ---- evaluation: per-(instance,position) S1,S2,B and the tautology controls ----
    net.eval()
    S1, S2, B, Ccur, Ctrunc, Ent, Inst = [], [], [], [], [], [], []
    with torch.no_grad():
        for di in range(args.eval_inst):
            x, y = make_batch(1, args.n, args.P, g, dev)
            logits, states = net.run(x)                                  # deterministic pass
            nll_k = [F.cross_entropy(logits[k].reshape(-1, args.P), y.reshape(-1), reduction="none")
                     for k in range(args.K)]
            ks = args.k_scout
            b = (nll_k[ks] - nll_k[ks+1]).reshape(args.n).cpu().numpy()   # marginal loop-benefit (signed)
            # S1: subtractive residual magnitude at scout loop
            z = states[ks]; resid = z - net.loop.block(z)
            s1 = (net.loop.alpha.abs() * resid).norm(dim=-1)[0].cpu().numpy()
            ccur = nll_k[ks].reshape(args.n).cpu().numpy()                # current-NLL at scout (control)
            # NOTE: with a single shared readout, the "truncated-loop readout NLL" control
            # coincides with current-NLL; add a SECOND independent readout to separate them (spec §4.2/C8).
            ctr = ccur
            ent = (-(F.softmax(logits[ks], -1) * F.log_softmax(logits[ks], -1)).sum(-1))[0].cpu().numpy()
            # S2: dropout(m=1) disagreement at scout loop (JSD). NOTE: this is dropout-only on a
            # SINGLE weight set -- NOT the spec's m>=2 between-mode committee (which needs >=2
            # independently-seeded trained nets, Fort 2019). Treat S2 here as a diagnostic.
            net.train()                                                   # dropout on
            ps = []
            for _ in range(args.T):
                lg, _ = net.run(x, K=ks+1); ps.append(F.softmax(lg[ks], -1))
            net.eval()
            Pm = torch.stack(ps).mean(0)
            jsd = (-(Pm * Pm.clamp_min(1e-12).log()).sum(-1)
                   + torch.stack([(pp*pp.clamp_min(1e-12).log()).sum(-1) for pp in ps]).mean(0))
            s2 = jsd.clamp_min(0)[0].cpu().numpy()
            for arr, val in [(S1,s1),(S2,s2),(B,b),(Ccur,ccur),(Ctrunc,ctr),(Ent,ent)]:
                arr.append(val)
            Inst.append(np.full(args.n, di))
    S1,S2,B,Ccur,Ctrunc,Ent,Inst = map(np.concatenate, (S1,S2,B,Ccur,Ctrunc,Ent,Inst))

    print(f"\nscored (instance,position) points: {len(B)}  instances: {len(np.unique(Inst))}")
    def show(name, S):
        rho, lo, hi = boot_ci(spearman, (S, B), Inst)
        pc, plo, phi = boot_ci(partial_spearman, (S, B, Ccur), Inst)   # control resampled with S,B (was unaligned -> crash)
        cap = tail_capture(S, B, args.tail_frac)
        print(f"[{name}] Spearman(S,B)={rho:.3f} CI[{lo:.3f},{hi:.3f}] | "
              f"partial|currentNLL={pc:.3f} CI[{plo:.3f},{phi:.3f}] | tail-capture={cap:.2%}")
        return lo, plo, cap
    lo1, plo1, cap1 = show("S1 residual", S1)
    lo2, plo2, cap2 = show("S2 dropout(m=1)", S2)
    capN = tail_capture(Ccur, B, args.tail_frac); capE = tail_capture(Ent, B, args.tail_frac)
    print(f"[baselines] current-NLL tail-capture={capN:.2%}  entropy tail-capture={capE:.2%}")
    # GREEN (signal side): a cheap signal's partial-corr CI > 0 AND beats both baselines on the tail
    s1_green = (plo1 > 0) and (cap1 > capN) and (cap1 > capE)
    print(f"\nVERDICT (signal side, S1 primary): {'GREEN-candidate' if s1_green else 'YELLOW/RED'}")
    print("  Still required for a real build gate: anisotropic operator (Round A1), NL arm,")
    print("  the iso-wall-clock R1 gate, untrained-scout->converged-benefit (R7), R6 cold-start surrogate.")

if __name__ == "__main__":
    main()
