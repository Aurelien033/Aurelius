"""MoK F3b — distance-isolated state survival (the clean F3 test).

F3 lesson: T-pair recall conflates CAPACITY (how many bindings fit in the
state) with DISTANCE (does a binding survive N tokens). F3b isolates distance:
  4 key-value pairs, filler gaps of D tokens between pairs and before query.
  D in {32, 64, 128, 256, 512} — the window-law comparison: a sliding-window
  attention layer with W=32 structurally fails at D>32; with W=128 fails at
  D>128. The kitten has NO window — if state survives D=256+, the claim
  "state memory ≠ attention window" is demonstrated.

Fixes vs F3: state d=128 (capacity no longer the bottleneck), gate bias init
+2 (forget gate starts closed -> slow forgetting; the standard gated-delta
trick), 6000 steps, linear control retained.
"""
import math, time, torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(1)
t0 = time.time()
DEV = "cpu"
VOCAB, D, NV, NK, NP = 64, 128, 32, 16, 4   # 4 pairs
FILL = 3

def batch(Dgap, B, device=DEV):
    keys = torch.randint(0, NK, (B, NP))
    vals = torch.randint(NV, NV + NK, (B, NP))
    qidx = torch.randint(0, NP, (B,))
    q = keys.gather(1, qidx.unsqueeze(1)).squeeze(1)
    target = vals.gather(1, qidx.unsqueeze(1)).squeeze(1)
    # layout: k1 v1 [gap] k2 v2 [gap] k3 v3 [gap] k4 v4 [gap] q [label]
    L = 2 * NP * (Dgap + 1) + 2
    seq = torch.full((B, L), FILL, dtype=torch.long)
    pos = 0
    for p in range(NP):
        seq[:, pos] = keys[:, p]; seq[:, pos + 1] = vals[:, p]
        pos += Dgap + 2
    seq[:, pos] = q
    seq[:, pos + 1] = target
    return seq.to(device)

class Kitten(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.d = d
        self.embed = nn.Linear(VOCAB, d)
        self.inp = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.gate.bias.data.fill_(2.0)      # start with forget gate CLOSED
        self.out = nn.Linear(d, NV + NK)
    def forward(self, x):
        x = self.embed(x)
        B, T, _ = x.shape
        # exact parallel scan for the diagonal linear recurrence:
        #   s_t = g_t ⊙ s_{t-1} + (1-g_t) ⊙ u_t
        #   c_t = cumprod(g) ;  z_t = cumsum((1-g)⊙u / c) ;  s_t = c ⊙ (s_0 + z_t)
        u = torch.tanh(self.inp(x))                 # (B,T,d)
        g = torch.sigmoid(self.gate(x))             # (B,T,d)
        c = torch.cumprod(g, dim=1)                 # (B,T,d)
        z = torch.cumsum((1 - g) * u / c.clamp_min(1e-6), dim=1)
        s = c * z                                   # s_0 = 0
        return self.out(s)

class LinearKitten(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.d = d
        self.embed = nn.Linear(VOCAB, d)
        self.inp = nn.Linear(d, d)
        self.out = nn.Linear(d, NV + NK)
    def forward(self, x):
        x = self.embed(x)
        # s += u_t  ->  cumsum over time, vectorized
        s = torch.cumsum(torch.tanh(self.inp(x)), dim=1)
        return self.out(s)

def run(model, Dgap, steps=2000, lr=1e-3, B=32):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for i in range(steps):
        seq = batch(Dgap, B)
        emb = F.one_hot(seq[:, :-1], VOCAB).float()
        logits = model(emb)
        loss = F.cross_entropy(logits[:, -1], seq[:, -1])
        opt.zero_grad(); loss.backward(); opt.step()
    seq = batch(Dgap, 256)
    emb = F.one_hot(seq[:, :-1], VOCAB).float()
    with torch.no_grad():
        logits = model(emb)
    acc = (logits[:, -1].argmax(-1) == seq[:, -1]).float().mean().item()
    return acc

print(f"MoK F3b | d={D} | 4 pairs | gap-isolated | {time.strftime('%H:%M:%S')}")
results = {}
for Dgap in [32, 64, 128, 256, 512]:
    row = {}
    for name, cls in [("kitten", Kitten), ("linear", LinearKitten)]:
        m = cls()
        n = sum(p.numel() for p in m.parameters())
        acc = run(m, Dgap)
        row[name] = round(acc, 4)
        print(f"  gap={Dgap:4d} {name:7s} params={n:7d} acc={acc:.4f}", flush=True)
    results[Dgap] = row

print("\n=== F3b VERDICT ===")
chance = 1 / 32
for Dgap, r in results.items():
    print(f"  gap={Dgap:4d}: kitten={r['kitten']:.3f} linear={r['linear']:.3f} (chance={chance:.3f})")
k256 = results.get(256, {}).get("kitten", 0)
k512 = results.get(512, {}).get("kitten", 0)
ok = k256 >= 0.8 and k512 >= 0.4
print(f"  kitten @256 = {k256:.3f} (>=0.8?) | @512 = {k512:.3f} (>=0.4?)")
print(f"VERDICT: {'F3b CLEARED — state memory survives far beyond W=32/128 windows' if ok else 'F3b NOT CLEARED — state decays (capacity or recurrence limits)'}")
print(f"TOTAL {time.time()-t0:.0f}s")
