"""F3c — CORRECTED distance-isolated binding probe (registered re-run, 2026-08-05).

Fixes the three flaws found in second-round review (mok_f3b_probe.py):
1. VALUES WITHOUT REPLACEMENT (NV=64 pool, 4 distinct) — kills the
   count-only modal-value shortcut (old floor 0.337; new floor 1/4=0.25
   because pairing is unrecoverable from counts).
2. KEY-VALUE SEPARATION: layout is k1 [gap] v1 ... k4 [gap] v4 q [label]
   — the key and its value are Dgap tokens apart. The old layout kept
   (k,v) adjacent (gap only separated irrelevant pairs) so ANY local
   bigram/trigram model solved it with zero long-range memory. Now the
   binding itself must survive the gap.
3. FP64 GATED ARM — the old fp32 parallel scan underflowed
   (0.88^137 = 2.5e-8 < 1e-6 clamp at gap 32; c=0 at 256/512). fp64
   keeps 0.88^2057 = 1e-115 representable (fp64 min ~2e-308). The
   gated arm is the ONLY order-sensitive arm from the original family;
   it was numerically dead, not mechanistically bad.
Plus: count-Bayes control computed by simulation, bigram-conv control
(local-order-sensitive, should NOT solve gap>0), 5 seeds per arm-gap,
eval n=1024, steps=1000 B=32.

Hypothesis: if the fp64 gated arm beats 0.25 by >2 SE at gap >= 128,
state memory is REAL and MoK's thesis is resurrected with the fix. If
nothing beats the floor, honest headline: state cells don't bind at
this task; MoK unresolved (the review's predicted outcome).
"""
import math, sys, time, warnings
warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

NK = 64      # key pool (distinct per sequence)
NV = 64      # value pool (distinct per sequence)
FILL = 0
KEY0, VAL0 = 1, 1 + NK          # keys [1,65), values [65,129)
VOCAB = 1 + NK + NV             # 129
NP = 4                          # pairs per sequence
D = 128
GAPS = [64, 128, 256, 512]
STEPS = 1000
B = 32
SEEDS = 5
EVAL_N = 1024                   # 4 x 256 eval batches
T0 = time.time()

def batch(Dgap, B):
    """k1 [gap] v1 ... k4 [gap] v4 q [label]; keys and values DISTINCT."""
    L = NP * (2 + Dgap) + 2
    seq = torch.full((B, L), FILL, dtype=torch.long)
    keys = torch.stack([torch.randperm(NK)[:NP] for _ in range(B)]) + KEY0
    vals = torch.stack([torch.randperm(NV)[:NP] for _ in range(B)]) + VAL0
    qidx = torch.randint(0, NP, (B,))
    pos = 0
    for p in range(NP):
        seq[:, pos] = keys[:, p]
        seq[:, pos + 1 + Dgap] = vals[:, p]
        pos += 2 + Dgap
    q = keys.gather(1, qidx.unsqueeze(1)).squeeze(1)
    target = vals.gather(1, qidx.unsqueeze(1)).squeeze(1)
    seq[:, pos] = q
    seq[:, pos + 1] = target
    return seq, target

def count_bayes_floor(Dgap, n=2000):
    """Simulated count-only optimal accuracy: with distinct values the
    pairing is unrecoverable from the multiset -> uniform 1/4 exactly."""
    torch.manual_seed(123)
    accs = []
    for _ in range(n // 256):
        seq, target = batch(Dgap, 256)
        # count-only predictor: value multiset = present values (all distinct);
        # target uniform among them -> 1/4. Verified by simulation below.
        for i in range(256):
            vals_present = set(seq[i].tolist()) & set(range(VAL0, VAL0 + NV))
            accs.append(1.0 / len(vals_present) if vals_present else 0.0)
    return sum(accs) / len(accs)

class GatedKitten(nn.Module):
    """Order-sensitive gated recurrence, fp64, numerically stable."""
    def __init__(self, d=D):
        super().__init__()
        self.d = d
        self.embed = nn.Linear(VOCAB, d)
        self.gate = nn.Linear(d, d)
        self.inp = nn.Linear(d, d)
        self.out = nn.Linear(d, VOCAB)
        with torch.no_grad():
            self.gate.bias.fill_(2.0)
    def forward(self, x):
        x = self.embed(x.double())
        g = torch.sigmoid(self.gate(x))
        u = torch.tanh(self.inp(x))
        # chunkwise parallel scan (chunk=64), fp64: within a chunk use the
        # cumprod/cumsum identity; carry the true state s between chunks.
        T = x.shape[1]
        s = torch.zeros(x.shape[0], self.d, dtype=torch.float64)
        outs = []
        for t0 in range(0, T, 64):
            gt = g[:, t0:t0 + 64]          # (B, C, d)
            ut = u[:, t0:t0 + 64]
            c = torch.cumprod(gt, dim=1)   # (B, C, d)
            z = torch.cumsum((1 - gt) * ut / c.clamp_min(1e-12), dim=1)
            s_local = s.unsqueeze(1) * c + c * z
            s = s_local[:, -1]
            outs.append(s_local)
        s = torch.cat(outs, dim=1)
        return self.out(s)

class LinearKitten(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.d = d
        self.embed = nn.Linear(VOCAB, d)
        self.inp = nn.Linear(d, d)
        self.out = nn.Linear(d, VOCAB)
    def forward(self, x):
        x = self.embed(x.double())
        s = torch.cumsum(torch.tanh(self.inp(x)), dim=1)
        return self.out(s)

class LDK(nn.Module):
    """Accumulate-only with adaptive write (fp64)."""
    def __init__(self, d=D):
        super().__init__()
        self.d = d
        self.embed = nn.Linear(VOCAB, d)
        self.inp = nn.Linear(d, d)
        self.alpha = nn.Linear(d, d)
        self.out = nn.Linear(d, VOCAB)
    def forward(self, x):
        x = self.embed(x.double())
        u = torch.tanh(self.inp(x))
        a = torch.sigmoid(self.alpha(x))
        s = torch.cumsum(a * u, dim=1)
        return self.out(s)

class BigramConv(nn.Module):
    """Local-order control: kernel-2 causal conv. Solves (k,v) adjacency
    ONLY if they are adjacent; at gap>0 it is also at the count floor."""
    def __init__(self, d=D):
        super().__init__()
        self.d = d
        self.embed = nn.Linear(VOCAB, d)
        self.conv = nn.Conv1d(d, d, kernel_size=2, padding=1)
        self.out = nn.Linear(d, VOCAB)
    def forward(self, x):
        x = self.embed(x.double())
        cv = self.conv(x.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        return self.out(torch.tanh(cv))

class NoMem(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.d = d
        self.embed = nn.Linear(VOCAB, d)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.Tanh())
        self.out = nn.Linear(d, VOCAB)
    def forward(self, x):
        x = self.embed(x.double())
        s = torch.tanh(self.mlp(x)).mean(dim=1).unsqueeze(1)
        return self.out(s.expand(x.shape[0], x.shape[1], self.d))

def run(model, Dgap, steps=STEPS, lr=1e-3, B=B):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for i in range(steps):
        seq, target = batch(Dgap, B)
        logits = model(F.one_hot(seq[:, :-1], VOCAB).float())
        loss = F.cross_entropy(logits[:, -1], target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # eval: EVAL_N samples
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(EVAL_N // 256):
            seq, target = batch(Dgap, 256)
            logits = model(F.one_hot(seq[:, :-1], VOCAB).float())
            correct += (logits[:, -1].argmax(-1) == target).sum().item()
            total += 256
    return correct / total

print(f"F3c | d={D} | NP={NP} | keys/values distinct (NK=NV={NK}) | "
      f"k-[gap]-v layout | fp64 gated | seeds={SEEDS} | steps={STEPS} | B={B}")
for Dgap in GAPS:
    floor = count_bayes_floor(Dgap)
    print(f"  gap={Dgap:4d} count-Bayes floor = {floor:.4f} (expect 0.2500)")
print()

results = {}
for Dgap in GAPS:
    row = {}
    for name, cls in [("gated64", GatedKitten), ("linear64", LinearKitten),
                      ("ldk64", LDK), ("bigram", BigramConv), ("nomem", NoMem)]:
        accs = []
        for seed in range(SEEDS):
            torch.manual_seed(1000 * seed + Dgap + 7)
            m = cls().double()
            accs.append(run(m, Dgap))
        mean = sum(accs) / SEEDS
        se = (sum((a - mean) ** 2 for a in accs) / (SEEDS - 1)) ** 0.5 / SEEDS ** 0.5
        row[name] = (mean, se, accs)
        print(f"  gap={Dgap:4d} {name:9s} mean={mean:.4f} se={se:.4f} seeds={[round(a,3) for a in accs]}")
    results[Dgap] = row
    print()

print("=== F3c VERDICT (vs count-Bayes floor 0.25) ===")
floor = 0.25
for Dgap in GAPS:
    for name, (mean, se, accs) in results[Dgap].items():
        z = (mean - floor) / se if se > 0 else 0
        flag = "BEATS FLOOR" if z > 2 else ("at floor" if abs(z) <= 2 else "below floor")
        if name in ("gated64", "linear64", "ldk64"):
            print(f"  gap={Dgap:4d} {name:9s} {mean:.4f} +/- {se:.4f} z={z:+.2f} [{flag}]")
best = {}
for Dgap in GAPS:
    for name, (mean, se, _) in results[Dgap].items():
        z = (mean - floor) / se
        best.setdefault(name, []).append(z)
for name, zs in best.items():
    n_big = sum(1 for z in zs if z > 2)
    print(f"{name}: beats floor at {n_big}/{len(GAPS)} gaps (z>2)")
print(f"TOTAL {time.time() - T0:.0f}s")
