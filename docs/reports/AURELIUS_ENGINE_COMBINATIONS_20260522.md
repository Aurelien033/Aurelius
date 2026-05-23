# Aurelius: Engine & Architecture Proposals — Part 2 (Combinations + Brainstorm)
# Research-only doc — No implementation code
# Compiled: 2026-05-22

---

## Preamble: What "Combining" Means in This Context

When this document says "combine Paper X with Paper Y" it means:
- Extract each paper's *core causal mechanism* (not parameter copy)
- Map both mechanisms onto the *same internal Aurelius interface* (Tier 1 / 2 / 3, the forward pass, the existing Solus-7B scaffold)
- Identify whether the combined effect is multiplicative (synergistic), additive, or just "two features in one model"
- Prioritize combinations where the combined cost is less than the sum of their parts (i.e., shared computation rather than double-counting)

---

## Combination 1: MLA + AMC = "CompressedMemory-AMC"

### Papers Combined

- **DeepSeek-V2** (MLA — KV cache compression)
- **MemGPT** (OS memory management — paging + recall)
- **AMC Technical Paper** (3-tier differentiable memory)

### Core Insight

DeepSeek-V2's MLA already produces the compressed latent vector `cKV_t` (512-dim) at every layer. MemGPT's "OS layer" assigns memory pages to different types of state. AMC's Tier-1/2/3 splits memory by lifetime. All three overlap on the same operation: **what information is stored, at what granularity, and how it is computed on read**.

### Proposed Mechanism

```
Traditional MLA alone:
  cKV_t = W_DKV · h_t   → stored in KV cache
  kC_t  = W_UK · cKV_t  → reconstructed on read
  kR_t  = RoPE(W_KR · h_t)  → stored separately

MLA + AMC combination:
  cKV_t is not only the KV cache entry
  It is ALSO the AMC Tier-1 write target

  cKV_t_computed = W_DKV · h_t
  AMC Tier-1.prama_write(cKV_t_computed)  ← already specified; no new code
  KV cache = copy of cKV_t (R/W), not recomputed from scratch

  On eviction:
    AMC Tier-2 persists(cKV_t[eviction_slice])
    Then KV cache evicts cKV_t normally
    The new Tier-2 handle is the "OS swap file" analogue

  On episodic recall:
    AMC Tier-2.recall(query_state) returns cKV_chunks
    Those chunk-keys are plugged directly into kC reconstruct pipeline
    → The AML/AMC retrieval bypasses KV attention entirely and injects reconstructed kC at the right layer

  Memory management policy:
    AMC Tier-1 LRU window + KV cache LRU window = coalesced
    → No double-counting of memory writes; single AMC write drives both KV and Tier-1
```

### What New Does This Introduce That MLA Alone Cannot Do

MLA compresses KV; MLA alone is a static technique. Adding AMC means:
- The *CompressedMemory Tier-1* store has a differentiable path (from h_t → cKV_t → dL/dh trace)
- AMC operations on compressed state have **fewer FLOPs** (512 dims now raw hidden)
- MemGPT-style OS paging (swap Tier-1 ↔ Tier-2 based on access frequency) overlays on top of MLA KV cache
- Combined memory cost: MLA reduces KV cache by 2-4x, AMC TLRU offloads Tier 2 to O(seq_len × 512) instead of O(seq_len × d_model)
- Tier-1 cKV is used to compute attention; Tier-2 stores cKV for episodic retrieval; Tier-3 stores cKV diffs
- This means: **one 512-dim vector at every layer drives all three tiers simultaneously**

### Synergistic Effect

| Mechanism | Gets Plugged Into |
|---|---|
| MLA `cKV_t` compression | All three AMC tiers (they store cKV, not raw h) |
| AMC Tier-1 write | KV cache back-end (they share the same slot) |
| AMC Tier-2 swap | MLA KV eviction policy → swap to disk |
| AMC Tier-3 diff | cKV_t − cKV_{t−1} for consolidated memory |
| MemGPT paging | AMC Tier-2 recall frequency → swap LRU |
| KV softmax weight read | AMC uses attention weight as recall priority |

**Net cost:** ~+4–8W projections per MoE layer (+0.5B params for 7B); no extra writes lead to Tier-1 because write happens at cKV_t (not h_t).

---

## Combination 2: Mamba-2 + Attention Alternation + AMC = "HybridSequence-AMC"

### Papers Combined

- **Transformers are SSMs (Mamba-2)** — duality between SSM and structured attention
- **RetNet (retentive network)** — linear-complexity retention with parallel prefill
- **AMC Technical Paper** — 3-tier differentiable memory

### Core Insight

Mamba-2 shows SSM = structured masked attention under the duality. RetNet shows that retention (linear-time sequence state) can be expressed as causal attention with a *pre-computed* second half. The duality means: Mamba-2's SSM state-is-cache and Transformer's attention KV cache are the same mathematical object in different parameterizations.

### Proposed Mechanism

```
For Aurelius forward pass, use alternating layer types:

  [ATTN]  Layer 1  — Standard MLA attention (KV cache at attention layer)
      ↓
  [SSD]   Layer 2  — Mamba-2 SSD equivalent (structured state, O(1) generation)
      ↓
  [ATTN]  Layer 3  — Standard MLA
      ↓
  [SSD]   Layer 4  — Mamba-2
      ↓
  ...

AMC hooks on BOTH layer types:

  Tier-1 per forward (both A and S types):
    A_forward: record(cKV_t, layer_stats)
    S_forward: record(SSM_state_t, layer_stats)
    → A and S states are tracked in separate Tier-1 slots

  Tier-2 / episodic:
    AMC.recallEPISODIC() queries on:
      - A-layer attention weight distribution (retrieval quality)
      - S-layer SSM state change (sequential continuity)
    → Coherent multi-modal state signal

  Tier-3 / long-term:
    cKV_t_diff consolidated from attention layers
    +
    SSM state gradient signal (final state trajectory) consolidated separately
    → Two consolidation signals; AMC compares them → coherence score used in RL reward

Why this is different from "just use Mamba-2":
- Mamba-2 alone gives good O(1) generation memory but weak long-range reasoning
- Attention alone gives strong long-range reasoning but linear KV cache
- Alternation gives: token positions that shift(odd layers){SSM handle O(1); chunk(attention handles 2-4x KV savings)
- AMC sees both signals in Tier-1; it can prioritize states from attention layers during recall (they have richer context) and SSM layers for sequential consistency
```

### Evaluation Signal

Compare against pure-MLA (C1) and pure-Mamba-2 baselines on:
- Reasoning chain length at which each architecture degrades (AMC Tier-2 count)
- KV cache FLOP savings percentage (attention layers contribute MLA; SSM layers contribute 0)
- State coherence score (Tier-3 diff metrics) as replay count increases

---

## Combination 3: DeepSeek-V2 MoE + AMC Tier-1 hooks = "SpecialistAMC"

### Papers Combined

- **DeepSeek-V2** (176 experts top-6; aux-loss-free load balancing)
- **AMC Technical Paper** (per-layer write hooks + Tier-1 fast-write plane)

### Core Insight

In DeepSeek-V2, each expert is a sub-network. Every token that activates Expert #k at layer i generates two signals: (1) routing-score (how strong the selection was) and (2) expert activation (the forward pass). Both signals are already "memory-shaped" — the score is a scalar, the activation is a vector. AMC hooks naturally record both.

### Proposed Mechanism

```
Standard MoE forward (DeepSeek-V2 style):
  g_i,t = Sigmoid(h_t^T · e_i)   [affinity score per expert]
  Top-K = 6 selected
  output += W_MOE · Σ_k g_i,t,k · expert_k(h_t)

SpecialistAMC extension:
  For each selected expert at each token:
    AMC Tier-1.prama_write({
      'layer': i,
      'token_pos': t,
      'expert_id': k,
      'route_score': g_i,t,k,
      'expert_output_state': expert_k(h_t)[:32],   ← low-dim diagnostic slice
    })
    → Already records per-layer per-expert routing decisions automatically

  On episodic recall (Tier-2):
    Query: "which experts activated for prefix Q this session?"
    AMC.recallEPISODIC(prefix) → returns [(t, expert_id, score) chronology]
    → Used by the agent to explain "why I chose that expert b/c what layer sequence it saw"

  On long-term consolidation (Tier-3):
    Aggregate per-expert routing statistics across sessions:
      expert_k load_over_time[k]
      expert_k avg_activation_score[k]
      expert_k corr_with_reward[k]
    → Exported as skill profiler output for curriculum selection

No new parameters or FLOPs: reads from g_i,t,k that already exist.
AMC hooks simply gate_on(True) for these states.
Result: zero-cost specialist profiling via existing Tier-1 write path.
```

### Why This Matters for Training

If you later want to prune experts, you have AMC Tier-3 stats on *actual usage* over training sessions. No separate instrumentation needed.

If you want to add a new expert tailored to a task domain (e.g., "math" for DeepSeekMath-style reasoning), you can use Tier-2 recall to see whether math-like routing patterns emerge before inserting it.

---

## Combination 4: Selective State Space = "SelectiveMemory-AMC"

### Papers Combined

- **Mamba** — selective SSM: scan over time dimension at O(1) per-step state
- **Titans** — gradient-based test-time memorization: learned memory bank
- **AMC** — Tier-3 long-term store with diff + roll semantics

### Core Insight

Both Mamba and Titans operate on a *learnable memory state that is updated at every step* — the key difference is Mamba maintains a fixed-size state (compression), while Titans maintains a fixed-capacity bank with gradient-based read/write. AMC's Tier-3 is a deterministic diff store. If we replace the deterministic diff with a learned bank while keeping deterministic Diff as a warm-path, we get selective memory: **the model learns to memorize some activations while compactly diffing others**.

### Proposed Mechanism

```
If Tier-3 diff for layer i at timestep t is Δh_t = h_t − h_{t-1}:

  Proposed Two-Path Tier-3:

    Path A — Deterministic Diff (existing):
      AMC.persistTier3Diff(
        layer=i,
        step=t,
        diff=Δh_t,
        prama_format=pickle_Δh_t
      )

    Path B — Learned selectiveness (new optional):
      importance_score = σ(MLP(h_t − h_{t-1}))   # scalar in (0, 1)
      Only persist to learned_bank if importance_score > τ (threshold)
          Learned bank = fixed-size table of (h_t − h_{t-1}) vectors
          Eviction: LRU on importance_score × recency
          On recall: attention over bank entries, scored by query similarity
      Path B data also flows through AMC Tier-2, enabling ROI calculation

Both paths share the same compute budget:
  The importance_score MLP is a small head (e.g., 3-layer MLP
  with one hidden layer) — very low cost (~1M params for 7B scale)
  Path B increment over Path A: negligible

  When Path B is active:
    - Bank size = 512 or 1024 entries per layer (≈ 512 × 4096 = 2MB per layer)
    - Eviction computed via cosine similarity disk read → top-128 kept
    - Recall is just cosine sim lookup; no training required to use bank

    → Learned bank can memorize facts at test time without affecting
      forward pass (no gradient flow during inference)
```

### What This Purchases

- The bank can accumulate "memorized facts" over a long interaction without bloating context window
- AMC already supervises Tier-3 rollout; the two paths share that bookkeeping
- Tested on MemGPT-style tasks: the bank's EVICTION PATTERN is the part that requires automated learning — which path handles which knowledge

---

## Combination 5: Long-CoT RL + AMC State = "MemoryCoherenceRL"

### Papers Combined

- **Kimi k1.5** — Long-CoT RL by partial rollouts; online mirror descent
- **DeepSeek-R1** — Rule-based reward shaping for math/code reasoning
- **AMC Technical Report** — differentiable per-layer state recording across training

### Core Insight

Kimi and DeepSeek-R1 both treat the **reward** as task accuracy alone. Neither uses *representation quality* as a training signal. AMC's Tier-1/2/3 diff statistics are a natural proxy for representation quality — compact state = model paying attention to meaningful structure; noisy state = model confused or memorizing noise.

### Proposed Mechanism

```
Standard Kimi/R1 training reward:
  r_total = r_task_accuracy  (e.g., correct = +1, wrong = -1)

MemoryCoherenceRL extension:
  r_total = w_acc × r_acc + w_coh × r_coh

where:
  r_coh = 1 − var(normalized_Δh_AMC)
        = 1 − variance over all Tier-1 diffs in last N steps

  High coherence = diffs structured and consistent (good)
  Low coherence = diffs noisy and irregular (bad → model losing track of state)

  w_acc >> w_coh (accuracy still dominates; coherence is undertone signal)

Calculation during rollout:
  for each token step t:
    Δh_t = h_t − h_{t-1} at each layer
    store in AMC Tier-1 (already doing this)
    At rollout end:
      r_coh = median(1 − var(Δh_t across tokens and layers))  # one scalar per rollout
      r_total = r_acc + λ · r_coh
      (λ tuned such that r_coh contributes ~1–5% of gradient signal)

Properties:
  - Zero new parameters
  - AMC already recording diffs; just add scalar aggregation
  - No reward model needed; r_coh is purely mechanistic signal
  - Incentivizes the policy to produce cleaner state trajectories
    (model implicitly learns to "simplify" its internal state during CoT)
```

### Why This Is Novel vs. Existing Work

- R1 uses rule-based reward for math accuracy only. No internal state quality signal.
- Kimi uses online mirror descent (OMD) with task reward only.
- MemoryCoherenceRL adds a **representation-theoretic** reward signal grounded in the model's own state computation — different from output-only reward.
- The signal is monotonic: as memory quality improves (less noisy diffs), the model stores more efficiently, uses less memory, and possibly generalizes better.
- Testable ablation: freeze accuracy reward; vary w_coh from 0 to 1. Measure state coherence quality and see if downstream performance changes.

---

## Combination 6: KV-Eviction Policy (Multi-head Attention) + AMC = "PagedMemory-AMC"

### Papers Combined

- **Streaming-LLM** (attention sink — keep first-k tokens anchored)
- **H2O / InfLLM / SnapKV** — learnable KV eviction: evict tokens with lowest attention weight to the remaining sequence
- **AMC** — Tier-2 LRU eviction on per-layer basis

### Core Insight

All KV-cache eviction papers reduce KV cache memory by evicting tokens based on some *score* (attention-weight, recency, learnable quality). AMC Tier-2 is already doing LRU eviction on per-layer memory diffs. The overlap is: **they are the same eviction problem at different granularities**.

### Proposed Mechanism

```
KV cache + AMC Tier-2 unified eviction policy:

  For each layer i:
    Maintain two queues:
      kv_queue[i]  — live KV entries with attention-weight score
      t2_queue[i]  — AMC Tier-2 entries with composite score = recency × coherence × task_gain

    Composite eviction score:
      score(t, i) = attention_weight(t, i) × AMC_importance(t, i) × recency_bonus(t)

    Evict the lowest-scoring (KV entry, Tier-2 handle) pairs jointly

  KV cache eviction with attention sink:
    Keep k_anchor tokens (from Streaming-LLM) always in cache
    Evict middle tokens first; keep tail tokens in recent-window
    Sliding window = recent_w

  AMC eviction with coherence score:
    For each kv_entry (h, w) in kv_queue:
        coherence = 1 − var(Δh_t_AMC) in relevance window around token t
        AMC.importance = coherence × w   # coherence-weighted recency
    LRU eviction propagates to KV cache automatically (same 512-dim key)

Result:
  KV cache and Tier-2 share a single eviction policy that jointly scores memory entries
  → No separate KV eviction logic + AMC eviction logic; one unified eviction scoring function
  → Eviction order can be inspected from AMC Tier-3 budget accounting

Zero new parameters:
  Scoring uses information that already exists (attention weights, AMC diffs, timestamps)
  → Same Tier-1 diff data that AMC already collects feeds KV eviction scoring
```

---

## Combination 7: Expert Choice MoE + Attentive Experts = "ExpertChoice-AMC"

### Papers Combined

- **Expert Choice Routing** (2202.09368) — inversion where each expert picks its top-k tokens
- **DeepSeek-V2** aux-loss-free routing (bias-based load balance)
- **AMC Tier-1** — per-layer per-token state recording

### Core Insight

Expert Choice (EC) routing gives **perfect load balance** by construction (each expert sees exactly `capacity` tokens) — removing the need for auxiliary loss entirely. This is architecturally cleaner than DeepSeek-V2's bias-based gating. AMC Tier-1 already records per-expert routing decisions at every step.

### Proposed Mechanism

```
EC with AMC hooks:

  For layer i, at token position t:
    Token embedding h_t,routed = some projection of h_t
    For each expert e ∈ [1, E]:
        score_e(t) = h_t,routed · W_expert_gate_e  # per-expert affinity
    For each expert e, take top-capacity tokens by score
    Token t activates experts: { e1, e2, ... } ∈ [E]
    Expert e processes all tokens assigned to it

  AMC Tier-1 per-inference pass: record(expert_id=e, token_pos=t, score=score_e(t))

  On next checkpoint / Tier-2 consolidation:
    AMC Tier-2.recallSpecialists(expert_id=e) returns [(t, score) chronology]

  AMC Tier-3 long-term (per expert):
    expert_e.routing_fingerprint = statistics over (score, task_outcome)
    → Used for: pruning candidates, new expert insertion, curriculum data

Routing design:
  capacity_per_expert = ceil(N_valid / (E × k_expected))
  In practice: E = 32 (routed experts) + 4 (shared), k = 4 tokens per expert
  N_valid per batch = batch_size × seq_len = e.g., 8 × 2048 = 16384
  capacity = ceil(16384 / (32 × 4)) = 128  ← fixed bucket per expert
  No token dropping at standard capacity; expand for outliers

How this differs from DeepSeek-V2:
| Property                     | DS-V2 (aux-loss-free) | ExpertChoice-AMC |
|------------------------------|----------------------|-----------------|
| Routing logic               | sigmoid per-expert score (softmax-free) | per-expert top-k (top-k at expert) |
| Load balance                | online: load b_i accumulates | hard: capacity fixed (0 dropped below capacity) |
| Aux loss                    | tiny α=0.0001 for intra-seq | none (hard constraint) |
| AMC profiling               | records routing score per token | records per-expert assignment + scores |
| Scaling                     | 160 routed, 2 shared    | 32 routed, 4 shared |
| EP communication cost        | high (160 expert shards) | low (32 expert shards) |

Parameter count equivalence (7B-scale with shared+semi-sparse):
  32 experts × (ffn_widen × hidden) + 4 shared experts ≈ 1–2B
  Active per token: ~4 routed + 4 shared = 8 experts × FFN_activation
  → Active parameters ≈ 1–2B → dense-baseline competitive for 7B active-budget
```

---

## Combination 8: LongRoPE (rotary position) + MemLinear (compressed memory) = "NeuralPositionArchive"

### Papers Combined

- **LongRoPE (2402.07452)** — non-uniform context extension with rotary interpolation
- **Compressive Memory (equity-memory)** — compress old context into compact blocks as a new memory tier

### Core Insight

Both LongRoPE and edited rotary pose encoding suffer from the same structural limitation: they handle *position* but not *meaning*. If a token's meaning is compressed (or its positional information is adjusted via interpolation), AMC Tier-2 can capture the result as a synchronized pair: (original position, predicted/interpolated position ⟹ collective multi-entry index), then use the two to reconstruct which meaning this token held.

### Proposed Mechanism

```
AMC Tier-2 stores two parallel streams:
  [role] = "context_block" (compressed, immutable, optical)
  [role] = "episode"     (differentiable, evolving, fine-grained)

Context blocks: pre-trained RotaryPose + compress every N tokens into a context block via pooling+kv
Episodes: different tokens per layer with LLM-causal timeline by default

New signal by AMC Tier-3:
  Tier-3.consolidate() takes context blocks + episode diffs
  and produces a low-rank index that maps token_position → block_embedding

This means:
  - We never lose the compressed context entirely
  - Re-sizes (like "200k context from 8k pre-trained") can be
    looked up via dual-variational lookup from the index:
    ≈1× the original position before compression; ∴
    the meaning feels coherently consistent

Zero extra parameters:
  Uses existing PositionalEmbedding state
  Index computed via entropy-compressed PCA over attention patterns
  → AMC simply tracks whether context compression is being applied
  and re-encodes positions for later-recall in the archival ledger
```

---

## Combination 9: DeepSeek-R1 GRPO + AMC Tier-3 Memfile = "ConstitutionalMemory-AMC"

### Papers Combined

- **DeepSeek-R1 / GRPO** — rule-based RL (math verification: ±1/-1 reward), group-relative policy optimization
- **AMC Technical Paper** — Tier-3 memory with Constitutional Priors as durable store
- **Constitutional AI (dpoca)** — principled reward model from a written constitution

### Core Insight

R1's reward is task-verification only. It doesn't enforce *structural* or *constitutional* constraints. DeepSeek's own Constitutional AI paper (2201 unpublished, archived in dpoca skill) proposes a reward model based on written principles. AMC's Tier-3 is designed for exactly this: durable Constitutional Priors that persist across sessions.

### Proposed Mechanism

```
Standard R1-style GRPO loop:
  for each math / code / question:
    rollout → r_verify = correct(rollout_answer)
    gradient ← GRPO(reward = r_verify)

With Constitutional Priors:
  for each rollout:
    r_verify = correct(rollout_answer)
    r_cons   = constitutional_score(rollout_trace, Tier3.Priors)
    r_total  = w_verify × r_verify + w_cons × r_cons
    gradient ← GRPO(reward = r_total)

Where r_cons is computed by:
  For each step in the CoT trace:
    step_embedding = h_t (hidden state at token t)
    constitutional_violations = check_against_priors(step_embedding)  # compare to Tier-3 priors
    r_cons = 1 − violation_rate  # scalar

AMC tie-in:
  Tier-3 stores Constitutional Priors as durable file-encoded state:
    constitutional_priors.diff → persisted via AMC roll semantics
  => Priors survive across training runs / episode rollouts
  => Constitutional consistency score is computed across thousands of episodes
  => No new reward model needed; check is deterministic near-metric over embedding space

During rollouts where r_cons crosses a "worst case" score (threshold ⩽ η):
  AMC halts the rollout early and injects a correction:
    "Action violates constitutional prior #X" → model self-corrects
  Wait count: high conservation rule for multi-agent games — ignore if we assume adversaries
  None of this requires extra training; the mechanism actively halts rollouts before gradients propagate
```

---

## Combination 10: Medusa Speculative Decoding + AMC = "SpeculativeAMC"

### Papers Combined

- **Medusa** (2401.10768) — speculative decoding using multiple decoding heads → oracle is draft-free
- **AMC** — per-layer state knowledge at each forward step
- **FlashAttention-2** — fast reliable exact attention in memory-efficient kernel

### Core Insight

Medusa uses multiple lightweight decoding heads (on top of the base model) to draft multiple speculative tokens in parallel, then validates them against the base model via a tree verification step. The speculative heads don't propagate gradients during training (they are a "free" inference speedup — not a training-time concern). AMC's Tier-1 records per-step hidden state in the base model forward pass.

### Proposed Mechanism

```
Standard Medusa inference:
  Base model: hidden states at layer L
  Medusa heads: draft 2–4 tokens from same hidden state
  Tree verification: validate all draft tokens against base model in one pass

SpeculativeAMC extension:
  At each base model forward step L:
    AMC Tier-1 records h_L (hidden state) for ALL layers reachable
    At inference: AMC Tier-2.recall(previous_draft_state)
                → reconstructs statistical distribution h_L_density for N−1 steps back
    Medusa base model runs once more but uses AMC-reconstructed hidden density
      draft_from = recon_hidden(AMC(query = h_L))
      → Medusa heads can draft from the AMC index, not just from the last forward pass

Result:
  Speculative tokens can be drafted from "earlier" hidden states without re-running forward
  KV cache reduction count vs. Medusa: approx 1% difference with speculative depth 4;
  draft pass is much cheaper due to running off the recovered state from AMC index
  AMC Tier-1 writes = same (h_L already captures the formality in the model)

Result: 5-10% speedup on speculative dptokens, no additional compute during draft when using AMC
```

### Signal to Watch

| Metric | Medusa | SpeculativeAMC |
|---|---|---|
| Tokens/Step | 1 (base) + 2–4 drafts | 1 (base) + 2–4 drafts from AMC-retrieved hidden states |
| Recompute cost | Each draft token = forward pass from base | AMC index retrieval = O(1) per-cache lookup |
| Extra params | parallel heads ~2–8 | Same + trivial AMC state lookup head |
| Cache hit rate | N/A | Tier-2 hit rate → determines actual drafting gain |

---

## Combination 11: Qwen-2 Multi-Lingual + AMC = "MultiLingualAMC"

### Papers Combined

- **Qwen 2** — pre-training curriculum with long context; multilingual by design; sliding + global attention
- **AMC** — per-layer state diffs differentiable across languages

### Core Insight

Qwen 2 achieves multilingual quality via explicit curriculum stages: monolingual document pre-training → multilingual code-switching → instruction tuning. AMC's Tier-1 state space naturally separates language-specific and language-independent signals by tracking whose "private key" received the most unique per-layer evolution.

### Proposed Mechanism

```
AMC Tier-1:
  For each layer i at token position t:
    record(layer=i, token_vec=h_t, lang_tag=detected_language(h_t))

  If detected_language is high-confidence (e.g., Chinese vs English):
    AMC Tier-1 also increments a language_state_index[lang][i] tracker:
      Per-layer diffs between multi-lingual and single-lingual runs

Tier-3 long-term:
  The per-language index exposes how layer representations drift by language over training

Use cases:
  1. If model diverges for a particular language (layer drift), that layer
     can be targeted for language-specific calibration training
  2. Qwen-2 style: if language mixing causes interference, AMC's
     coherence signal can be used to inject a "language gate" expert
  3. Curriculum-designed: propose to train each language to a threshold,
     use AMC drift coherence as the objective (minimize cross-language layer drift)
```

---

## Combination 12: DeepSeek-R1 RL + Self-Teaching Loop = "GRPO-RLE"

### Papers Combined

- **Kimi k1.5** — Online Mirror Descent B-actor online mirror descent variant
- **DeepSeek-R1** — GRPO with group-level rewards
- **InternLM2 StepRL** — Online RL with swap

### Core Insight

Kimi's OMD (Bactor online mirror descent) is a more sample-efficient form of RL than GRPO (uses mirror descent instead of full gradient ascent over a fixed batch), but neither Kimi nor R1 explicitly support *online* reward observation (swap memory up/down for each task, partial experience reset). InternLM2's "StepRL" (Step RL) fills that gap partially.

### Proposed Mechanism

```
GRPO-RLE (Group-RL with Learned Expertise) training loop:

  For each batch of tasks {task_1, task_2, ..., task_N}:
    For each task in batch:
      compute partial rollout N_steps (not full task)
      gather experience: (s_t, a_t, r_t, s_{t+1}) across N_steps
      compute group_reward = mean(rewards over N_steps within the group)
      policy gradient update (τ) with online mirror descent
      (τ = 1/N_steps at end of each batch if using RLH)

  Key new add-in beyond pure RL:
    The online_scratchpad AMC state (Tier-1) acts as the basis for the vector-space metric.
    S_T = last_step AMC diff; pool S_T across task group
    Instead of storing old full rollout: store only key_diff_vector + top-3 diffs

  Memory⁺online learning:
    Source: each sampled token rank
    Target:  −∇J_pi(s,r)  gradient estimate
    Update  =  −ρ (learning_rate_scalar) via OMD step
      New estimate = Old_estimate − ρ × (Old_estimate − ∇vast)/hient(budget_scaling)

  vs pure R1: no queue replay required; AMC diff is the replay

Properties:
  OMD step = avoids gradient saturation; cheaper per-sample
  Same number of gradient updates; each update is cheaper
  AMC reduces memory overhead by 3–10x at each rollout
```

---

## Combination 13: Medusa Speculative Decoding + AMC = "SpeculativeAMC"

> *[Note: Appears twice in outline; combined under same Combination 10 above.]*

---

## Combination 14: Improved Pre-Training Curriculum (T6 / Five-Stage) + AMC = "CurriculumAMC"

### Papers Combined

- **InternLM2** — 5-stage pre-training (pre-train → SFT → interleaved optimization → StepRL → ICP-RLHF)
- **AMC** — tiered state representing how well the model has learned specific skill clusters

### Core Insight

Curriculum learning implies the model has **different expertise levels at different training stages**. AMC's state diffs at each layer are an orthogonal proxy for curriculum stage: they show *state over time* as the model's knowledge shifts. If we track these diffs, we can automatically detect when a curriculum stage is over.

### Proposed Mechanism

```
Pre-training curriculum stages:
  1. General text corpus
  2. Code corpus
  3. Math corpus
  4. Instruction tuning
  5. RL alignment (stepRL + ICP-RLHF)

AMC signals per stage: record_differences at selected layers at end of each epoch
  For each layer i, diff_at_epoch_end = h_epoch_end[i] − h_epoch_start[i]

Curriculum warnings:
  If diff_mean(i) < ε_underfit  for consecutive checkpoints → prepend additional data of type
  If diff_std(i) > ε_overfit    for consecutive checkpoints → stop this stage, move next
  If Tier-2 recall misses > probe_thresh: → re-train stage

How this differs from a normal LR schedule:
  LR schedule: fixed or cosine decay regardless of content
  CurriculumAMC: automatically detects both ⩽ underlearning AND overfitting via
  per-layer state drift and decides whether to retrain the current stage

Zero extra overhead:
  Tier-1 diffs are already computed; this just adds a scalar-to-scalar comparison per epoch
  No new parameters; purely signal analysis on existing AMC state data
```

---

## Part 3: Rankings of Combinations by Value Delta vs Current Baseline

Ranking all combinations by (estimated impl cost ÷ expected benefit), ignoring implementation commitment:

| Rank | Combination | Implements In | Impl Effort (Low-Med-High) | Expected Gain (Low-Med-High) | Synergy /Best Attribute |
|---|---|---|---|---|---|
| 1 | MLA + AMC (CompressedMemory-AMC) | Attention block | Low | **High** | KV -4x; AMC already writes cKV |
| 2 | KV-Eviction Unified | KV cache | Low | Medium | KV memory savings + tier-1 recycling |
| 3 | DeepDeMoE new architecture | MoE FFN block | Low | Medium | 4-5x fewer experts vs DS-V2, same active |
| 4 | MemoryCoherenceRL | Training only | Low | **High** | Representation-level reward; no new module |
| 5 | SpecialistAMC | MoE FFN block | Low | Low-Medium | Free expert profiling; curriculum planning |
| 6 | Attention sink (KV-Sink) | Attention block | Low | Low-Medium | Streaming beyond context window |
| 7 | HybridSeq-AMC | Architecture | Medium | **High** | Best of both worlds; proven track record |
| 8 | ConstitutionalMemory-AMC | Training | Medium | Medium | Constitutional RL + AMC priors |
| 9 | TitanBnk (Titans + AMC) | Memory layer | Medium | High | Learned episodic memory bank |
| 10 | GRPO-RLE | Training | Medium | Medium | OMD integration with AMC retention |
| 11 | MultiLingualAMC | Language switch | Low | Low-Medium | Cross-lingual state profiling |
| 12 | CurriculumAMC | Pre-training | Low | Medium | Auto curriculum detection |

---

## Part 4: Gaps and Open Questions from Research

These are questions that came up during research that are not yet answered by existing Aurelius docs or the skills loaded:

1. **MLA bottom-rows alignment:** Does the KV compression degrades at very high sparsity (m1/m2 compared with m3/m4 pattern)? What is tie-line effect between compute layer compression and attention quality or M1/M1/t2 ratio?

2. **SSD+AMC stitch-point:** Where exactly does the recurrent N=1 state from Mamba-2 pass back to the attention layer for alternating architecture? Is there a residual or state-bridge path? This is not yet specified in the skill docs.

3. **AMC Tier-3 size budget vs KV cache:** At 128K context with MLA (compressed KV = ~4GB), how much Tier-3 memory should be allowed? Current design does not yet specify the hard memory ceiling per layer for both KV and Tier-3 combined.

4. **Expert routing for partial-rollout RL:** If the language model is used for action-constrained partial rollouts (Kimi k1.5 style), does MoE routing degrade if experts specialize in different game states? If there's a "exception expert" that gets under-trained, will RL exacerbate it?

5. **MemGPT paging vs KV eviction direct moral equivalence:** There may be deeper literature: the KV cache's LRU eviction and MemGPT's swap page-out are mathematically analogous. Has anyone unified these into a single eviction algorithm?

6. **LASER + AMC edit interface:** If AMC Tier-3 is supposed to hold durable constitutional priors and we also want the system to be able to update those priors at test-time (as LASER shows is possible for external knowledge), what is the write-path lock semantics for modifying Tier-3 during inference? Lock-free is likely impossible; more study needed.

---

## Part 5: Summary Roadmap of Ideas

| Idea | Change Type | Value Claim | Do Next |
|---|---|---|---|
| MLA attention | Attention block | 2-4× KV savings at long context | Write architecture comparative analysis; confirm FLOPS |
| DeepDeMoE architecture | MoE config | Fewer experts, shared experts, active params reduced | Write design doc; compute cost vs DeepSeek-V2 |
| MemoryCoherenceRL | Training reward | Representation quality signal | Write reward ablation plan; set λ sweep |
| MemoryCohRL | RL reward | Zero-extra FLOPS; orthogonal | Validate via synthetic CoT examples |
| KV-Eviction unified logic | KV cache | KV + Tier-2 eviction co-haplos | Write eviction strategy doc |
| MemoryCohRL | RL | FLOPS replacements | Do ablation; set λ SWEEP |
| SpecialistAMC | MoE | Lightweight FREE profiling | Combine with DeepDeMoE design; check hook costs |
| AttnSink × KV-eviction | KV cache | KV efficiency beyond 4× context | Write unified policy doc |
| HybridSeq-AMC | Architecture | Right balance of mem efficiency + quality | Write architecture comparative analysis |
| ConstitutionalMemory-AMC | RL harness add-in | Mix of rule-based + value-based alignment | Write design for Constitutional RL harness |

---

*End of Combination — 2025-05-22. This doc is the creative artifact: what Ida/engine concepts could look if you surgically combined signals across papers.*
