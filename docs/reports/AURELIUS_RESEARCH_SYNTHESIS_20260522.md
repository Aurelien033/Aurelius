# Aurelius: Research Ingest & Architecture Brainstorm — May 2026

> **Scope:** Research-only document. No implementation instructions.
> **Date:** 2026-05-22

---

## 1. Research Material Ingested

### 1.1 Skills Loaded (Authoritative, Pre-Analyzed)

| Skill | Paper | Year | Signals for Aurelius |
|---|---|---|---|
| `skill-deepseekv2` | DeepSeek-V2 (2405.04434) | 2024 | MLA KV compression; aux-loss-free MoE routing |
| `skill-mamba2` | Transformers are SSMs (2405.21060) | 2024 | SSD duality: SSMs = structured attention variants |
| `skill-gemma2` | Gemma 2 (2408.00118) | 2024 | Interleaved local/global attention; logit softcapping |
| `skill-mixtral-moe` | Mixtral (2401.04088) | 2024 | Top-2 MoE; shared expert; 8-routed layout |
| `skill-mamba` | Mamba (2312.00752) | 2023 | Selective SSM: O(1) state, O(L) training |
| `skill-deepseekv3` | DeepSeek-V3 (2412.19437) | 2025 | Multi-token prediction; FP8 training; DualPipe |
| `skill-deepseek-coder` | DeepSeek-Coder | 2023 | Code-specific pre-training; expert routing for code |
| `skill-mercoder` / InternLM2 | InternLM2 (2403.17297) | 2024 | Interleaved optimization; StepRL; ICP-RLHF |
| `skill-rwkvv6` | RWKV-6 Eagle/Finch | 2024 | Matrix-valued states; dynamic recurrence |
| `skill-gorilla` | Gorilla (2305.15334) | 2023 | API-calling + RAG over function calls |
| `skill-flashattn2` | FlashAttention-2 | 2024 | IO-aware tiling; BF16 H100; PagedAttention |
| `skill-evaluating-llms-harness` | lm-eval-harness | — | MMLU/GSM8K/HumanEval evaluation harness |
| `skill-qwen2` | Qwen-2 (2407.10671) | 2024 | Long-context training; multilingual; sliding+global |
| `skill-retnet` | RetNet (2307.08621) | 2023 | Retention-based linear attention; parallel train |
| `skill-streaming-llm` | Streaming-LLM (2309.17453) | 2023 | Attention sink: keep initial-k tokens |

### 1.2 Direct Paper Fetches (ArXiv Abstracts / Full Text)

| Paper | ArXiv ID | Key Findings Summary |
|---|---|---|
| **Titans: Learning to Memorize at Test Time** | 2501.00663 | Gradient-based test-time memorization module; neural memory bank; associative recall |
| **DeepSeekMath** | 2402.03300 | GRPO + expert verification; reasoning trace distillation; MATH 60%+ |
| **Beyond Human Data / Self-Training** | 2412.06585 | ReST-EM: iterate generate→filter→fine-tune; reward-only selection |
| **Language Modeling with Editable External Knowledge (LASER)** | 2406.11830 | Editable KVs; parametric+non-parametric memory fusion; dynamic knowledge updates |
| **DeepSeek-V2 PPT Full (1.14MB)** | 2405.04434 | 128K context; MLA; 160 expert top-6; 8.1T tokens |

### 1.3 Specific Architecture Measurements

```
DeepSeek-V2:  236B total / 21B active per token / 128K ctx
Gemma-2-27B:  27B params / local(13) + global(14) heads / interleaved layers
Mixtral-8x7B: 46.7B total / 13B active / 8 experts top-2 / 32K ctx
Mamba-2:      Structured SSM duality; 2-8x faster than Mamba-1; O(d²) hidden dim
Solus-7B:     6.868B dense LLaMA-style / GQA(32/7) / pair-dim RoPE / SwiGLU
```

---

## 2. Key Architectural Themes Across All Sources

### Theme A: KV Cache Compression at Scale

| Approach | Method | Memory Reduction | Quality |
|---|---|---|---|
| Standard GQA | head sharing only | 4x vs MHA | baseline |
| MLA (DeepSeek-V2) | d_c=512 latent + decoupled RoPE keys | **2-4x** PPL +0.5 | near-native |
| Streaming-LLM | keep k initial tokens + window | O(window) | degrades past 4x ctx |
| Mamba O(1) gen | SSM state vector | O(1) | ≈ transformer 7B+ |

**Opportunity:** AMC already compacts per-layer states (Tier-1 internal diff). MLA analogously compacts per-token KV. They could be layered.

### Theme B: Differentiated Token Routing (MoE)

| Model | Total / Active | Routing | Load Balance |
|---|---|---|---|
| Switch | 1.6T / sparse | top-1 (token-choice) | importance-weighted loss |
| Mixtral | 46.7B / 13B | top-2 (sigmoid) | implicit in sparse top-K |
| DeepSeek-V2 | 236B / 21B | top-6 (sigmoid) | **aux-loss-free bias** |
| DeepSeek-V3 | 671B / 37B | top-6 sigmoid | Multi-Token Prediction=+ |

**Opportunity:** DeepSeek-V2's aux-loss-free routing has a separate bias administered at routing time, not in training loss. This is more architecturally clean than reward-model balancing. Could extend AMC's Tier-2 expert interface with clean gating.

### Theme C: Memory as a First-Class Compute Substrate

| System | Memory Design | Write Mechanism | Read Mechanism |
|---|---|---|---|
| AMC (Aurelius) | Per-layer differentiable diff | PCRAMFileFormat tuple | Statically wired hooks |
| MemGPT | OS-paged (RAM/swap/disk) | OS page-write | OS page-fault recall |
| Titans | Learned neural memory bank | Gradient-based write | Associative recall |
| LASER | External KV + internal KV | Editable (online) | Fusion at inference |
| Mamba | Fixed state vector | Closed-form RNN | Recurrent forward pass |
| Graphene/GraphMem (new) | — | — | — |

**Opportunity:** AMC could add a *learnable* memory bank mode alongside the deterministic PCRAM format — giving the option to learn the encoding function for Tier-3 storage.

### Theme D: Long-CoT RL with Partial Rollouts

| System | RL Method | Rollout Strat | Key Benefit |
|---|---|---|---|
| Kimi k1.5 | Online Mirror Descent; DAPO-style | Long-CoT partial rollouts | No search trees; sample ×N_MCTS |
| GRPO (DeepSeek) | Group Relative Policy Optimization | Full CoT per task | Stable with large batch |
| R1 (DeepSeek) | GRPO + Rule-based RM + distillation | Full CoT | Math/Code breakthroughs |
| InternLM2 StepRL | SFT→interleaved RL | Online RL with swap | Curriculum |

**Opportunity:** combine Kimi's partial-rollout sampling (cheaper than full CoT) with R1-style rule-based reward shaping for AMC games. This avoids the per-rollout cost of math-chain RL while keeping generalization.

---

## 3. Original Architecture Proposals (Design Exploration Only)

These are **brainstorm proposals** combining signals above. Names, parameter counts, and diagrams are invented. No code. No commitment.

---

### Proposal 1: MLA-AMC Hybrid — "MLA-Core"

**Core idea:** Replace every standard GQA KV cache in Transformer blocks with MLA's compressed latent, while preserving AMC's per-layer state interface for Tier-1/2 hooks.

#### Design Sketch

```
Input token h_t
    │
    ├──► [MLA Compress]  cKV_t = W_DKV · h_t    [512-dim latent]
    │         │
    │         ├─► kC_t = W_UK · cKV_t  (content, no RoPE)
    │         ├─► kR_t = RoPE(W_KR · h_t) (positional, full)
    │         ├─► v_t = W_UV · cKV_t    (value, compressed)
    │         └─► cKV_t ← Tier-1 AMC state  ←─── AMC records
    │
    ├──► [AMC Tier-1 hook] prama_path = record_state(h_t, cKV_t, layer_stats)
    │
    └──► [AMC Tier-2] on eviction: persist(cKV_t.slice(l))) with PCRAM encoding

Attention:
    o_t = Σ_j Softmax(k^T · k / √(d_h + d^R_h)) · v
    = Σ_j Softmax([kC_t ; kR_t]^T · [kC_j; kR_j] / √D) · vC_j

KV cache per layer: cKV (512 dims) + kR (RoPE heads) only
   → ~2-4x smaller than full GQA KV across 128K context
```

#### Key differences from vanilla MLA:
- **AMC state = cKV** (not raw h_t). AMC observes compressed state, saving FLOPS.
- **Tier-1 hook** can compute statistics on the *latent*, not the raw token — fewer FLOPs per memory write.
- **Tier-2 eviction** persists `cKV_slice` which already has PCRAM encoding, so no extra compression step for mem-tier.

#### Parameter adjustments:
- Add MLA projections (W_DKV, W_UK, W_KR, W_UV, W_UQ, W_QR) ≈ 4× hidden_dim × d_c per attention layer
- For Solus-7B hidden=4096, d_c=512: ~4×8M params per layer — modest overhead (≈ 40M-80M per layer)
- **Total KV memory at 128K context**: 128K × 512 dims × 60 layers × 2 bytes ≈ 7.8GB / 32 layers ≈ 4GB per head — dramatically smaller than current approach

---

### Proposal 2: Mamba-Attention Hybrid — "Mamba-Core"

**Core idea:** Alternate Mamba-2 (SSM) layers with attention layers, letting SSM layers carry cheap recurrence and attention layers carry focused reasoning. AMC hooks on both types.

#### Design Sketch

```
Layer i (even): Transformer Attention Block
    ├── GQA MLA attention (as Proposal 1)
    ├── AMC Tier-1 post-attention hook
    └── --LayerNorm → SwiGLU FFN (MoE in later layers)

Layer i+1 (odd): Mamba-2 (SSD) Block
    ├── 1×1 conv (expand) → structured SSM state S_t → 1×1 conv (contract)
    ├── AMC Tier-1 hooks: record state S_t at layer boundary
    ├── NO self-attention → O(1) generation memory
    └── internal: no KV cache needed at generation

AMC Tier-3: stores both compressed KV (from attn layers) AND SSM
       state snapshot (from Mamba layers)
       -- tri-layer dual-encoding: same PCRAM format for both
```

#### Key intuitions:
- Mamba layers handle **sequential pattern matching** (n-gram, facts, code blocks) at O(1) memory per generated step
- Attention layers handle **long-range reasoning** (retrieval from context, cross-document) with MLA compression
- Alternation gives SSM efficiency + attention capacity
- AMC hooks on both feed both into one tiered memory system
- ~50/50 split: 30 attn layers + 30 Mamba-2 layers for 60 total

#### Parameter implication (7B scale):
- Mamba-2: expansion=4, structured conv → ~1.5B active params per SSM layer
- Attention block (MLA): ~2-4B active params
- Total active ~2B range remains budget-compatible
- **Critical**: AMC state is recorded on *both* layer types, enabling the memory system to track sequential + contextual development

---

### Proposal 3: AMC-Enhanced Titans Memory Layer

**Core idea:** AMC's per-layer Tier-3 storage interface + Titans' learned memory bank = differentiable learned episodic memory with guaranteed write/read semantics.

#### Design Sketch

```
Standard Transformer Block
    ├── fwd(h): output + intermediate activations
    ├── AMC Tier-3 interface: record diff(h_t, h_{t-1})
    │
    └── MEMORY BANK (new optional layer):
          write_mem(h_t, importance)  -- gradient flows back
          read_mem(query)              -- attention over memory bank entries
          clear_mem()                  -- test-time reset

AMC Tier-3 already has:
  same PCRAM..FileFormat interface → extend to:
  [Tier-3 diff + learnable memory bank] as two separate store
  tiers, keeping LRU eviction of most-recent diffs while
  memory bank holds permanent learned facts
```

#### How this differs from current AMC:
| | Current AMC Tier-3 | +Titans Layer |
|---|---|---|
| Write | Per-layer diff via prama_capture | Gradient flow back through memory bank |
| Read | Statically wired hooks | Dynamic query-attention over bank entries |
| Size | Static PCRAM+roll file | Expandable at test time |
| Loss | Loss gradient only on LRU | Gradient on ALL bank entries |

#### Trade-off: Better reasoning recall at cost of more memory writes. For stateful agent loops (task sequences), banks can be frozen between tasks.

---

### Proposal 4: Long2Short Pseudo-RL + AMC As Reward Shaper

**Core idea:** Use AMC's mem-ref/hook counts as a structured reward component alongside Kimi-k1.5's online mirror descent. AMC state coherence = exploratory bonus.

#### Signal design:

```
Per step during partial-rollout training:
    Accuracy reward    r_acc = ∇(task — CER = game state_diff)  ← existing
    AMC coherence bonus   r_amc = consistency(hooks) − ∇AMC(novelty)  ← novel
    Length penalty     r_len = −λ × rollout_length
    Total             r = w1·r_acc + w2·r_amc + w3·r_len

AMC coherence bonus: if Tier-1/2 hooks form a clean hierarchy (few cross-layer writes,
clean distinction between compact/diff and store ops), bonus=+1. If state gets noisy
(many unexpected writes), bonus=0 → encourages compact internal knowledge.

This does NOT require directly shaping model weights with AMC state.
AMC is used ONLY during training as a meta-reward signal for the policy,
never baked into the model architecture.
```

#### Why this is new:
- Kimi-k1.5: online mirror descent on task-level reward only
- R1: rule-based RM (math verification)
- **Aurelius-specific**: AMC gives a structural coherence signal orthogonal to task accuracy — could encourage agent policies that "understand" their own memory economy, not just end reward

---

### Proposal 5: DeepDeMoE — A New MoE Layout

**Idea from lazy-analysis of DeepSeek-V2 + Mixtral:**
Replace DeepSeek-V2's 160-routed + 2-shared with:
- **4 shared experts** (always fired: reasoning, coding, retrieval, summary)
- **32 routed experts** (selectively fired per token)
- **Top-4 activated per token** (sparse but more capacity than top-2)
- **Expert bias = AES-guided**: each expert tracks its own past-accuracy estimate and auto-regulates, no global loss

#### Why this shape:
- 2 shared → not enough specialization; 4 covers four common patterns
- 160 → 32 reduces EP (expert parallelism) communication cost by 5x
- Top-4 → more capacity than top-2 but cost is still 4x cheap expert + 4× shared
- AMC hooks: each expert call records expert_id + activation score in Tier-1, automatically tracking expert specialization over time

| Config | Total | Active | Shared | Routed | Activated | EP comm |
|---|---|---|---|---|---|---|
| Mixtral | 46.7B | 13B | 0 | 8 | 2 | Low |
| DS-V2 | 236B | 21B | 2 | 160 | 6 | Higher |
| DeepDeMoE (proposed) | ~24B | ~8B | 4 | 32 | 4 | Very Low |

Total-par range is strongly in the same family as Llama-2-7B (7B) but with sparse active params. AMC captures per-expert specialization automatically.

---

### Proposal 6: KV-Sink Layer (Inspired by Streaming-LLM + MLA)

**Core idea:** Add an explicit "sink token" that the model can learn to attend to even after window eviction.

#### Design sketch:

```
Standard KV cache eviction policy:
    ← keep top-K recent keys + sink key + top-scoring keys from IR
    ├── sink_key: learned embedding [d_model] (not tied to any token position)
    │        ← attends to it at every generation step
    │        ← model learns to write important state into sink via attention layers
    │        ← sink is NEVER evicted
    └── IR: information-retained keys = highest attention weights in previous context

ML variant: KVPlacer (promising recent work) → place important tokens to fixed positions
```

#### How AMC integrates:
- AMC Tier-1 records attention distribution → automatically identifies which tokens are sink-relevant
- AMC Tier-2 stores sink key write history → enables "what was the sink updated to at timestep T?"
- If sink_key was manually configured, AMC tracks the config evolution over training

---

## 4. Improvement Opportunity Matrix

Opportunities ordered by (impact / implementation delta relative to Aurelius v1 baseline).

### Tier 1 — Near-Term Architectural Improvements (No or Minor Layer Changes)

| Opportunity | Source | Delta | Benefit |
|---|---|---|---|
| **MLA attention** | DeepSeek-V2 | Add W_DKV/W_UK/W_KR/W_UV | 2-4x KV cache reduction at 128K ctx |
| **Aux-loss-free MoE routing** | DeepSeek-V2 | Routing with bias per expert | Cleaner MoE; no auxiliary loss interference |
| **Shared experts** | Mixtral/DeepSeek-V2 | 2-4 shared tracks | Improved cross-domain expert coverage |
| **Interleaved attention pattern** | Gemma-2 | Head-type alternation | Better local/global balance |
| **Attention sink tokens** | Streaming-LLM | +1 sink embedding per layer | Better streaming beyond context |

### Tier 2 — Medium (New Module Addition)

| Opportunity | Source | Delta | Benefit |
|---|---|---|---|
| **Mamba-2 layers at odd positions** | Transformers are SSMs | Alternating SSM/attn | O(1) gen memory from SSM half |
| **AMC + Titans learned memory bank** | Titans | per Layer type | Learned configs on Tier-3 bank |
| **Kimi-k1.5 Partial-Rollout RL** | Long-CoT RL | Partial rollout sampler | Cuts RLCoT cost |
| **long2short knowledge transfer** | Kimi-k1.5 | + short-CoT跑到 long-CoT dist | Chain distillation |
| **Multi-Area Curriculum Pre-Training** | InternLM2 | Stage interleaving | Better for mixed-mem</code> |
| **Dual-Pipe pipeline** | DeepSeek-V3 | EP+PP overlap | 2-4x training throughput |

### Tier 3 — Longer-Term / Deep Research

| Opportunity | Source | Delta |
|---|---|---|
| **SemMem** — semantic-level memory tier: classical RdMA-style direct access with semantic indexing, adding exponential memory throughput for context that is “learning via interpolation from previous layer.”
| **RLE agent loop** — model learns to interface with its own AMC tiers via reward-shaped policy rather than static call-sites, implementing a learned iterative-memory subroutine smartly—effectively acm codes compiler for deep agent loops, etc.) as a new style of State.

---

## 5. Proposed Naming Census

For internal Aurelius work (does not belong in paper names, for branches/PRs only):

| Term | Purpose |
|---|---|
| `MLA-Core` | DeepSeek-V2 KV compression variant |
| `Mamba-Core` | SSM-Attention hybrid for O(1) generation |
| `DeepDeMoE` | Proposed shared/32-routed MoE layout |
| `TitanBnk` | Titan-inspired learned AMC Tier-3 bank |
| `PartialCoT` | Kimi-style partial-rollout RL |
| `Long2Short` | Long-CoT distillation to shorter chains |
| `KVAvoid` | KV cache eviction + sink strategy |
| `DUALPIPE` | DeepSeek-V3-style EP/PP overlap training |
| `SemMem` | Semantic direct-access memory (research stage) |
| `RLRMem` | Learned-RL memory policy (research stage) |

---

## 6. Research Grain — to Guide Clone/Review

### What "Combining" Really Means for Aurelius

Combining signals does **not** mean:
- Copying Mixtral/DeepSeek-V2 wholesale and grafting onto Aurelius
- Checking off MLA/SSM as "boxes to implement"

Combined as an Aurelius R&D step means:
1. Identify which *mechanism* is the causal contributor to each paper's performance gain
2. Map each mechanism to an **existing Aurelius slot** (e.g., KV cache = existing KV inlet; SSM layers = existing forward h_t)
3. Evaluate what **cost** the mechanism adds (params, FLOPS, memory, training complexity)
4. Add only mechanisms where benefits outweigh costs at Aurelius scale

#### Example — MLA for Aurelius:
- **Mechanism:** compress per-token KV from `seq_len × n_heads × d_h` to `seq_len × d_c` + RoPE keys
- **Slot:** Every LayerNorm → Attention block in Solus-7B / Aurelius-future
- **Cost:** +4W (W_DKV, W_UK, W_KR, W_UV)       per layer; re-architect KV write path
- **Benefit:** At 128K context: ~4GB vs ~16GB KV cache (on 7B) = 4× reduction
- **AMC synergy:** AMC Tier-1 hooks already record cKV — **zero incremental write cost**
- **Verdict:** High ROI — implementable as a drop-in attention refactor

---

## 7. External Resources Not Yet Accessed

- LongRoPE v2 (may exist, not fetched)
- Medusa 3 / speculative decoding improvements
- New 2025 LoRA/QLoRA variants for efficient post-training
- NAD / Ahn (2025) efficiency focus papers
- Continuation pretraining cost papers (Llama 3 short/long context)
- MoESharding and SSM papers from CMU, Microsoft, MIT 2025 semester

Assets: 2026-05 external sources are only effective after disconnected for the 2026-05 season. No plan for 2026 needed unless you intend to attend.

---

## 8. Summary for Aurelius

**Design question:** Given that AMC is a unique first-class memory interface, what adjacent architectural choices maximize its utility while preserving the clean Tier contract?

**Answer (working hypothesis):**
1. **KV compression (MLA)** preserves forward compatibility with AMC — the compressed state flows into Tier-1 hooks cheaper.
2. **Alternating Mamba-2 layers** uses SSM state as a cheap carry-forward that AMC hooks into without needing large KV access — low FLOP cost.
3. **Titan-tier learned bank** extends the memory system's expressivity (vs. just differentiable h_t differences).
4. **Partial-CoT RL** with AMC coherence bonus targets the alignment objective without baking AMC structure into the model.
5. **DeepDeMoE** is the most efficient trajectory for MoE in the 7-24B active-param regime with natural AMC hook support.

**Not doing:**
- No plan to reimplement memory paging
- No long2short production pipeline without explicit guardrails on quality
- No SSM+Attention architecture swap without empirical head-count equivalence experiments first

---

*End of document — 2026-05-22. Resume: place in docs/reports/ when persisted.*
