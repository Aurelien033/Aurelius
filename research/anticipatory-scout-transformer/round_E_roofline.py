"""
Round E — R1 wall-clock roofline gate (the systems half of the build gate).
Spec: experiment-2-spec.md v2 §7a; risk R1 (notes §10.3).

THE PREMISE UNDER TEST: adaptive per-token compute saves FLOPs on paper, but on a GPU the
block is a dense GEMM whose cost is set by tensor SHAPE, not by which tokens "need" work.
This benchmark asks the only question that matters: does tiered+scout beat a matched dense
baseline on MEASURED tokens/sec, with the scout's OWN compute charged?

It measures THREE regimes so you can see exactly where wall-clock savings do/don't appear:
  (1) DENSE            : every sequence runs K_max loops (the baseline).
  (2) MASKED-TOKEN     : every sequence still runs K_max loops, but "finished" tokens are
                         masked out -> SAME GEMM shape -> demonstrates FLOP-savings != wall-clock
                         (the "counter artifact" R1 warns about).
  (3) BUCKETED-SEQUENCE: sequences are assigned to tiers (by simulated hardness) and each tier
                         runs its own loop count as a dense batch -> CAN save wall-clock IF hard
                         sequences are rare. Token-level bucketing WITHIN a sequence is NOT done
                         because attention couples tokens (that is the whole R1 difficulty).
Scout overhead (m committee passes x k_scout loops over all sequences) is added to (2) and (3).

PASS LINE (pre-register): bucketed-sequence tokens/sec >= 1.15x dense AND
  scout_time + bucket/scatter + sync < saved_main_time (all MEASURED), swept over hard-fraction.
Synthetic weights — systems cost is weight-independent; this needs no trained model.

Authored without execution — smoke-test with --seq 64 --batch 8 --reps 5 first.
"""
import argparse, time
import torch, torch.nn as nn

class Block(nn.Module):                                  # representative transformer block
    def __init__(self, d, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
    def forward(self, x, key_padding_mask=None):
        a, _ = self.attn(self.n1(x), self.n1(x), self.n1(x), key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))

def sync(dev):
    if dev.type == "cuda": torch.cuda.synchronize()

def timed(fn, dev, reps, warmup):
    for _ in range(warmup): fn()
    sync(dev); t0 = time.perf_counter()
    for _ in range(reps): fn()
    sync(dev); return (time.perf_counter() - t0) / reps

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=512); p.add_argument("--heads", type=int, default=8)
    p.add_argument("--seq", type=int, default=256); p.add_argument("--batch", type=int, default=32)
    p.add_argument("--tiers", default="1,3,6,10", help="loop counts per tier; last = K_max")
    p.add_argument("--hard_frac", default="0.05,0.1,0.2,0.4", help="fraction of sequences sent to the deepest tier")
    p.add_argument("--m_committee", type=int, default=3); p.add_argument("--k_scout", type=int, default=2)
    p.add_argument("--reps", type=int, default=20); p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--pass_speedup", type=float, default=1.15)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    tiers = [int(t) for t in args.tiers.split(",")]; Kmax = max(tiers)
    block = Block(args.d, args.heads).to(dev).eval()
    x = torch.randn(args.batch, args.seq, args.d, device=dev)

    def run_loops(z, K):
        for _ in range(K): z = block(z)
        return z

    dense_t = timed(lambda: run_loops(x, Kmax), dev, args.reps, args.warmup)
    masked_t = timed(lambda: run_loops(x, Kmax), dev, args.reps, args.warmup)  # same shape; mask is free-but-useless
    scout_t = timed(lambda: [run_loops(x, args.k_scout) for _ in range(args.m_committee)], dev, args.reps, args.warmup)
    toks = args.batch * args.seq
    print(f"device={dev}  tokens/iter={toks}  K_max={Kmax}  tiers={tiers}")
    print(f"[DENSE]  {dense_t*1e3:.2f} ms  ({toks/dense_t:,.0f} tok/s)")
    print(f"[MASKED-TOKEN] {masked_t*1e3:.2f} ms  -> same as dense by construction "
          f"(FLOP 'savings' do NOT reduce wall-clock; this is the R1 counter artifact)")
    print(f"[SCOUT overhead] {scout_t*1e3:.2f} ms  ({100*scout_t/dense_t:.1f}% of dense)\n")

    print(f"{'hard_frac':>10} {'bucketed ms':>12} {'+scout ms':>10} {'tok/s':>12} {'speedup':>8} {'verdict':>8}")
    for hf in [float(h) for h in args.hard_frac.split(",")]:
        # assign sequences to tiers: hard_frac to deepest, rest spread across cheaper tiers
        n_hard = max(1, int(hf * args.batch)); n_easy = args.batch - n_hard
        ntiers = len(tiers)
        base = (n_easy // (ntiers - 1)) if ntiers > 1 else 0       # guard single-tier (--tiers 10)
        counts = [base] * (ntiers - 1) + [n_hard]
        counts[0] += args.batch - sum(counts)                     # absorb remainder into tier 0
        def bucketed():
            off = 0
            for K, c in zip(tiers, counts):
                if c <= 0: continue
                _ = run_loops(x[off:off+c], K); off += c
        buck_t = timed(bucketed, dev, args.reps, args.warmup)
        total_t = buck_t + scout_t                                   # charge the scout's own compute
        speed = dense_t / total_t
        ok = speed >= args.pass_speedup and (scout_t + buck_t) < dense_t
        print(f"{hf:>10.2f} {buck_t*1e3:>12.2f} {total_t*1e3:>10.2f} {toks/total_t:>12,.0f} "
              f"{speed:>8.2f} {'PASS' if ok else 'FAIL':>8}")
    print(f"\nPASS line: speedup >= {args.pass_speedup}x AND scout+bucket < dense (measured). "
          f"FLOP counts NEVER satisfy this gate. Token-level (within-sequence) adaptivity is NOT "
          f"benchmarked because attention couples tokens -> bucket at the sequence/request level.")
    if dev.type != "cuda":
        print("WARNING: on CPU these numbers are not representative. Re-run on the target GPU; "
              "also report sustained GPU utilization (nvidia-smi) and check the memory-vs-compute roofline.")

if __name__ == "__main__":
    main()
