# Aurelius Research Status — 2026-08-04 (Frontier Package)

This directory vendors the current research arc: measured mechanisms,
submission artifacts, and integration code. Every claim below is backed by
a runnable script in this folder or a record in the Obsidian vault
(~/Obsidian Coding/Aurelius/).

## 1. SISA — Self-Indexed Sparse Attention (paper-ready)
Training-free indexer: the model's OWN early attention selects which
structured-context records stay in the attention window.
- `sisa/sisa_paper.pdf` — submission draft (LaTeX source in the lab vault,
  tectonic-compiled, 90+ KB, all numbers measured — no fabricated rows).
- `sisa/sisa_battery.py` — full evidence battery (budget envelope, natural
  keys, 6-qid, ablations; SISA beats uniform 9/9, oracle-Q 8/9; t=-4.62,
  p=0.006 at B=135).
- `sisa/sisa_circuit_probe.py` — the binding circuit: heads {3,11} at the
  calibrated layer; window law W_req ≈ 1.1-1.3×horizon (measured across
  R=15/30/60 and Qwen2.5-1.5B); H2 2-head sufficiency (corr 0.964);
  H4/H4b ablation (indexer is read-only; value readout decoupled);
  H7 family scan (early-quarter binding, never late quarter, all 5 models).
- Transfer law: binding layer is model-specific; full-layer aggregate robust
  for Qwen ≥1.5B; TinyLlama = family boundary (0/3).
- Production indexer = parse-filtered candidates + attention ranking
  (substring attraction on colliding keys is a documented, reported
  phenomenon, not a bug).

## 2. AEX-KV — Adaptive Bit-Exact KV Codec (integration probe PASSED)
Lossless KV storage tier: decode(encode(X)) == X as uint16 (preserves
+0/-0, subnormals, infinities, NaN payloads). 21 reversible modes, per-block
adaptive selection, CRC32, real binary streams.
- `aexkv/src/aexkv.py` — the codec (vendored from the research repo,
  tests 9/9 + fuzz 500/500 + corruption detection; evidence audit on file).
- `aexkv/ALGORITHM_SPEC.md` — AEXKV002 stream spec (blocking, mode
  portfolio, float-flip, Lorenzo residuals, bit-planes, dictionary/RLE).
- `aexkv/aexkv_integration_probe.py` — LIVE WIRING PROOF: real Qwen3-1.7B
  KV (28 layers, N=903): 103.6 MB → 70.6 MB = 1.468×, verify_exact TRUE on
  all 56 tensors, 30s on M1 Pro CPU.
- ROLE: storage tier of the three-tier memory — HOT (SISA-admitted) /
  WARM (AEX-compressed DRAM) / COLD (AEX-compressed SSD). SISA eviction
  becomes REVERSIBLE: window law = cache policy, not hard retrieval bound.
- Full design: vault record aexkv-aurelius-integration-2026-08-04.

## 3. MoK — Mixture of Kittens (our algorithm, F1 in flight)
Attention-routed recurrent processors: small gated-delta kitten blocks
replacing the dense MLP, routed training-free by the model's binding circuit.
- `mok/mok_f3b_probe.py` — DISTANCE LAW CLEARED: kitten state holds
  associative bindings at 1.000 accuracy through 512-token gaps (window law
  says W=128 layers fail >128). Capacity law (F3): bindings ≈ O(state dim).
- F1 (nanochat d6, MoK layer vs relu2 MLP at matched budget) RUNNING —
  verdict gates on val/bpb ≤ 1.164608 (relu2 baseline).
- Related: DART 2608.02032 (decoded attention over recurrent states) is the
  published concurrent direction; our training-free router is the
  differentiator.

## 4. Current architecture verdicts (adoption matrix, vault record)
- GQA base ADOPT · MLA EVALUATE on GB10 · DSA REJECT for structured (SISA
  replaces the indexer) · KDA HOLD · SiTU-GLU NO-ADVANTAGE (measured,
  relu2 wins 1.164608 vs 1.165957) · AttnRes/Muon EVALUATE queued ·
  MXFP4 HOLD · DeepSeek-V4-Flash ADOPT as post-training substrate ·
  mid-training ADOPT before RLVR · Keyless Attention/FIPO EVALUATE.

## 5. How to reproduce
- SISA: `python3 sisa_battery.py` (needs Qwen3-1.7B via transformers, CPU
  ok, ~10-20 min; attn_implementation="eager" required).
- AEX-KV: `cd aexkv && python3 -m pytest tests/ -q` then
  `python3 aexkv_integration_probe.py` (CPU, ~30 s + model load).
- MoK: `python3 mok_f3b_probe.py` (CPU, ~15 min, vectorized parallel scan).
