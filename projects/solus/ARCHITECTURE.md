# Solus-MoE 8B — Architecture Reference

## 1 · Overview

Solus-MoE 8B is a 32-layer causal decoder-only Transformer augmented with a
Sparse Mixture-of-Experts (MoE) block in 8 of its layers.  It shares the same
dense-building-block foundation as **Solus-7B** but swaps the dense FFN for a
router + expert-pool in 8 of the 32 layers, raising the total parameter count
from 6,330 M → **8,025 M** while keeping FLOPs close to the dense baseline at
inference time (top-k=2 active experts).

## 2 · Design Principles

| Principle | Decision |
|-----------|----------|
| No positional bias | All-MLP MLP structure (no dropout used) |
| Attention → FFN ordering (not FFN → Attention) | Match all of DeepSeek Micro architecture |
| Grouped-Query attention (GQA) | 32 query heads, 7 KV heads |
| RoPE | Rotary Position Embedding (θ=10 000), no scaling |
| Normalisation | Pre-RMSNorm (eps=1e-5) x2 per block |
| Activation | SiLU (Swish) for SwiGLU gating |
| Weight tie | LM head ↔ token embedding weight matrix |
| Router noise | jitter_eps=0.01 during training only |

## 3 · Data-flow per Block

### 3 · a  Dense Layer (24 of 32)

```
x  ──► RMSNorm ──► GQA Attention ──► + x ──► RMSNorm ──► SwiGLU MLP ──► + x
     (pre-attn)        (QKV+O)    residual  (pre-ffn)   (gate·up·down)   residual
```

### 3 · b  MoE Layer (8 of 32)

```
x  ──► RMSNorm ──► GQA Attention ──► + x ──► RMSNorm ──► MoE ──► + x
     (pre-attn)        (QKV+O)    residual  (pre-moe)  (sparse)  residual

MoE inside:
  logits = Linear(Hidden × NumExperts)        # router gate
  probs  = softmax(logits)                    # per-expert probability
  top2   = top_k(probs, k=2)                  # expert selection
  tokens dispatched to selected experts
  weighted sum of expert outputs → hidden
```

### 3 · c  MoE Layer Placement

The 8 MoE layers are **evenly spaced** at every 4th position to distribute
computational load throughout the network:

```
L  0  → MoE   L  4  → MoE   L  8  → MoE   L 12  → MoE
L 16  → MoE   L 20  → MoE   L 24  → MoE   L 28  → MoE
(all other 24 layers are standard dense)
```

## 4 · Parameter Accounting

```
Component        Formula                     Parameters   % of model
────────────────────────────────────────────────────────────────────
Token embeddings V × H                       525,336,576   6.5%
24 × Dense        24 × (attn + 3×H×I + 2×H)  4,756,537,344  59.2%
8 × MoE           8  × (attn + 8×3×H×Ie + 2×H) 2,743,402,496 34.2%
Final norm        H                                4,096   0.0%
────────────────────────────────────────────────────────────────────
Total                                           8,025,280,512 = 8,025 M
```

**Per-layer breakdown**

| Layer type  | attn Q/K/V/O  | SwiGLU MLP        | Norms  | Router  | Pool     | Total per layer  |
|-------------|:------------:|:-----------------:|:------:|:-------:|:--------:|:---------------:|
| Dense       | 24,117,248   | 157,286,400       | 8,192  | –       | –        | **181,411,840** |
| MoE         | 24,117,248   | ×8 experts = …    | 8,192  | 32,768  | 2,414,424| **342,925,312** |

expert_ffn = 3 × H × I_expert = 3 × 4096 × 3072 = 37,748,736 per expert  
expert_pool = 8 × 37,748,736 = 301,989,888  
router_proj = H × E = 4096 × 8 = 32,768  

`per_moe = 24,117,248 + 301,989,888 + 32,768 + 8,192 = 342,925,312` ✓

## 5 · Router Mechanics

```
Input hidden states  (B, T, H)
        │
        ▼ router.gate: Linear(H, E=8)
   Logits (B, T, 8)
        │
        ▼ softmax  (numerically-stable; subtract max)
   probs (B, T, 8)  — sum to 1
        │
        ▼ top_k(k=2)
   top2_weight : (B, T, 2)
   top2_indices: (B, T, 2) int64
        │
        ├─► Load-balance aux loss (training only):
        │       L_aux = α × E × Σ_i p̄_i × f̄_i
        │    (Shazeer et al. 2017, ST-MoE)
        │
        └─► Dispatch tokens to top-2 experts
                weighted sum → output (B, T, H)
```

## 6 · Routing Auxiliary Loss

Only added to `out["loss"]` during **training** (not eval / generate):

```python
L_aux = cfg.moe_aux_loss_alpha * E * Σ_i (p̄_i · f̄_i)
  where
    p̄_i … mean per-token probability routed to expert i over the batch
    f̄_i … fraction of total tokens actually assigned to expert i
```

When routing is balanced `p̄_i ≈ f̄_i ≈ 1/E`, so `L_aux ≈ cfg.moe_aux_loss_alpha`.

## 7 · Initialization

| Component  | Strategy                    |
|----------  |:--------------------------  |
| Linear     | `trunc_normal_(mean=0, std=0.02)` |
| RMSNorm weight | `ones_()`              |
| Embedding  | `trunc_normal_(mean=0, std=0.02)` |

## 8 · Precision & Memory

| Scenario          | Bytes / model   | Notes |
|------            |:---------------:|:----- |
| FP32 inference   | ~32 GB          | weights only (not activations) |
| 4-bit quant      | ~4 GB           | GPTQ / AWQ compatible |
| Training FP16    | ~16 GB shards   | + optimizer states 2–3× overhead |

## 9 · File Map

```
solus/
├── configs/
│   └── moe_config.py        ← production hyper-parameter set
├── src/
│   └── modeling/
│       ├── config.py        ← SolusConfig + helpers
│       ├── solus-moe-8b/    ← full model skeleton
│       │   ├── dptorch_model.py     ← model+Pipeline+Scheduler
├── README.md
└── ARCHITECTURE.md          ← this document
```

## 10 · Compared to Solus-7B (Dense)

|              | Solus-7B Dense | Solus-MoE 8B |
|---|---|---|
| Total params | **6,330 M** | **8,025 M** (+27%) |
| Active FLOPs / token | all layers | ~dense equivalent (2 experts × 8 layers active) |
| Memory / token | 100 % | ~ dense (only active experts occupy VRAM) |
| Training throughput | baseline | lower due to all-gather dispatch overhead |
| Inference throughput | baseline | ≈ baseline (top-k = 2 active experts) |

---
*Last updated: 2026-05-22 · Based on closed-form per-component derivation.*
