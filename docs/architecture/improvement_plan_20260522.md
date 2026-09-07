# Aurelius Architecture Improvement Plan
**Generated:** 2026-05-22
**Status:** Research synthesis — awaiting engineering prioritization
**Scope:** Memory (AMC), model architecture, training, alignment, inference, agents

---

## 1. Research Ingest Summary

### Papers fetched and processed

| Paper | ID | Key insights relevant to Aurelius |
|-------|-----|-----------------------------------|
| Kimi k1.5 | 2501.12599 | Long-CoT RL, simplified framework (no MCTS/PRM), partial rollouts, long2short transfer, online mirror descent, context-length as RL axis |
| YaRN | 2309.00071 | Efficient context extension via attention skew correction + NTK-aware interpolation; already in Aurelius |
| GPT-3 | 2005.14165 | Few-shot in-context learning baseline; scaling laws reference |
| FlashAttention2 | 2401.08258 *(queued)* | IO-aware tiling for variable sequence lengths |
| Mamba2 | 2405.21060 *(queued)* | Hardware-aware SSM, parallel scan, state-space models |
| DeepSeek-V2 MLA | — | Multi-head latent attention compresses KV cache |

### Gap analysis from current AMC design (per `docs/papers/amc_technical_paper.md`)

The AMC paper describes a per-layer differentiable memory architecture but the current implementation (Tier-2 hook + Tier-3 store) does not yet:
- Insert memory into the forward pass (only post-hoc hook)
- Use real token embeddings or neural similarity scoring (Tier-2 uses blake2b byte hashes as fake token IDs)
- Adapt surprise threshold per distribution
- Integrate the existing `plugins/memory/semantic_memory.py` and `memory_retriever.py` (BM25) into AMC
- Perform per-layer memory queries (the formal model has separate M_l^1, M_s^2, M^3 per layer)

---

## 2. Proposed Improvements by Subsystem

### 2.1 Memory — AMC Deep Dive

**Priority: HIGH. This is Aurelius' novel contribution claim.**

#### 2.1.1 Integrate existing memory plugins into AMC pipeline
- **SemanticMemory** (`plugins/memory/semantic_memory.py`): concept graph with typed relations. Currently unused by Tier-2/3 hooks.
- **MemoryRetriever** (`plugins/memory/memory_retriever.py`): BM25 over key-value entries, re-ranked by importance. Currently separate from AMC.
- **Proposal:** Build `AMCPluginBridge` — adapts SemanticMemory relations as Tier-3 value content, routes BM25 results through the AMCPrefixCompiler trust gates.

#### 2.1.2 Replace fake token IDs with real embedding-backed tokenization
`amc_tier2.py` line 238-239 derives `token_ids = tuple(token_hash)` from blake2b bytes. This is deterministic but semantically meaningless.
**Improvement:** Use the model's actual tokenizer to encode the content string, and store real token IDs in the block. This enables:
- Genuine sequence overlap detection
- Prefix caching via real KV cache references
- Cross-tier token-level alignment

#### 2.1.3 Adaptive surprise threshold
Current `AMCTier2Config.surprise_threshold` is static (default 0.5).
**Improvement:** Track per-quantile surprise statistics and apply a dynamic threshold:
```
surprise_threshold = P75(last_n_surprises) * 0.8 + baseline * 0.2
```
This prevents over-spamming during high-diversity exploration and under-filling during repetitive sessions.

#### 2.1.4 Per-layer tier-2 bank (moving toward formal AMC model)
The formal AMC model has separate memory bank M_l^2 per layer. The current hook is a single global store.
**Phase A improvement:** Build `AMCLayerMemoryBank` — a thin layer-indexed wrapper; populate during training by hooking `act_router.py` or model hook interface.
**Phase B:** Add per-layer write gates `w_l` (separate `nn.Linear` per layer) that output the tier-2 write score.

#### 2.1.5 Memory consolidation with LLM summarization
Current consolidation: decay-based ranking + quarantine promotion.
**Add:** A summarization step for Tier-3 entries that have high repetition count. Use the model to compress `(content, session_context)` into a canonical form. This reduces redundancy and increases memory density.

---

### 2.2 Model Architecture

**Current:** 24-layer, 1.395B dense, GQA 16/8, SwiGLU, Pre-RMSNorm, RoPE+YaRN (θ=500k), tied embeddings, MoE (top-2, 8 experts), MTP (n=2), Muon optimizer.

#### 2.2.1 Mamba/SSM hybrid layers (Solus-7B precedent)
The Solus-7B project (6.868B dense) already explored pure-Transformer from scratch.
**Proposal:** Add `MambaBlock` as an optional alternative to `TransformerBlock` in even-numbered layers (e.g., layers 2, 4, 6...). SSMs excel at long-range recurrence with O(1) memory.
**Rationale:** Complements attention: SSM handles local/recurrent state, attention handles global context. Reduces KV cache size for long contexts.
**Reference:** Mamba2 paper (2405.21060) — hardware-aware parallel scan.

#### 2.2.2 MLA (Multi-head Latent Attention) from DeepSeek-V2
Replace plain KV cache with compressed latent representation:
`c_kv = W_DKV * (proj_k(h), proj_v(h))` — store compressed `c` in cache, decompress on attention.
**Benefit:** 50-70% KV cache reduction per head with comparable perplexity.
**Reference:** DeepSeek-V2 technical report.

#### 2.2.3 Gemma-2-style attention improvements
- **Logit softcapping** (`cap_logits`): clip attention score distribution tails to stabilize training. Already partially present in some Aurelius modules.
- **Sliding window + global attention pattern:** Gemma-2 uses alternating local/global heads; can improve long-context locality without full KV cache growth.

#### 2.2.4 DyT (Dynamic Tanh) as RMSNorm alternative
`DyT(x) = γ * tanh(α * y_hat) + β` where `y_hat` are layer-normalized inputs. Recent work shows DyT beats RMSNorm on vision+language at similar cost.
**Proposal:** Add config switch `norm_type: rms | dyt` in `AureliusConfig`.

#### 2.2.5 RoPE variant: YaRN improvements
Aurelius already has YaRN. Add the **attention skew** correction from the latest YaRN version (v3, 2407.14491) that corrects the attention logit distribution at extended context.

---

### 2.3 Training & RL

#### 2.3.1 Kimi k1.5-style long2short transfer
The long2short method produces strong short-CoT results from long-CoT fine-tuning:
- **Length penalty during distillation:** Apply a token length penalty to long-CoT supervised responses before using them as training targets for short-CoT.
- **Model merging:** Average short-CoT and long-CoT LoRA adapters, not full model weights.

**Implementation:** Add `Trainer.long2short` mixin with `length_penalty` and `merge_adapters` steps.

#### 2.3.2 Online Mirror Descent (OMD) for RL policy
Kimi k1.5 uses "online mirror descent" variant for policy optimization. This is a gradient-based trust-region method.
**Proposal:** Add `ONLINE_MD` policy optimizer alongside GRPO/PPO in `src/training/rl/`.
**Config:** `policy_optimizer: grpo | ppo | online_md`

#### 2.3.3 Partial rollout reuse
Kimi k1.5's partial rollouts reuse 80% of a long trajectory and only re-sample the tail, cutting RL data-collection cost.
**Proposal:** In the `AbsoluteZero` self-play loop, add `partial_reuse_ratio: float = 0.8` config to `InteractionTrajectoryBuilder`. Cache intermediate states with `torch.save(..., _use_new_zipfile_serialization=False)` for fast reload.

#### 2.3.4 Curriculum: long context window scaling
Extend the RL curriculum (in `configs/`) to scale RoPE context window across RL stages:
```
Stage 1 (SFT base):      4096
Stage 2 (RL init):       8192
Stage 3 (RL long):       32768+
```
Use `AureliusConfig.extend_context(new_max_len)` to rebuild RoPE frequencies on checkpoint resume.

---

### 2.4 Alignment

#### 2.4.1 Implement MIS-PO (token-level KL-gated alignment)
MIS-PO is marked "in flight" in Aurelius. Key ingredients:
- Token-level KL divergence gate: only update tokens where policy diverges from reference by more than `kl_threshold`
- Trajectory-level reward floor: discard trajectories below `reward_min` before gradient
- KL penalty term: `L_total = L_sft + λ_kl * KL(policy || ref)`

**State:** Proto-stub exists in `src/alignment/mis_po.py`. Complete the trainer scaffold.

#### 2.4.2 PRAXIS v3: ExpertSafetyAffinity for MoE
Current PRAXIS (praxis.py) has *Simple Routing* safety. Add the `ExpertSafetyAffinity` concept:
- Track per-expert safety rate (harm rate per expert)
- In the MoE router, downweight experts with safety violations
- Add `expert_safety_ema` buffer to SparseMoELayer

#### 2.4.3 Constitutional Tier-3 policy entries
AMC Tier 3 should store constitutional principles as high-priority policy entries. Currently the Tier-3 `promote()` method doesn't have a "policy" tag. Add:
```python
def promote_policy(self, key, text, version, priority="high"):
    return self.promote(key=key, value=text, tags=frozenset({"policy", version}),
                        confidence=1.0, trust_level=TrustLevel.TRUSTED)
```

---

### 2.5 Inference

#### 2.5.1 Medusa-style tree decoding (beyond MTP)
Current MTP speculative decoding uses n=2 draft heads. Extend:
- Add tree attention mask that enable all Medusa heads to draft in parallel
- Tree verification: verify all draft paths in one forward pass
- Config: `speculative: {backend: "medusa", max_depth: 4, draft_heads: 3}`

#### 2.5.2 Neural KV eviction policy
Current eviction strategies (H2O, EVICT, etc.) are static rule-based.
**Proposal:** Train a small `nn.Linear(hidden_dim, 1)` head that outputs eviction scores per token, trained with a retention loss. In `kv_cache_compression.py`, add:
```python
class NeuralEvictionPolicy:
    def score_tokens(self, hidden_states: Tensor) -> Tensor:
        return self.eviction_head(hidden_states).squeeze(-1)
```

#### 2.5.3 Chunked prefill for 128K context
The current prefill path processes the full sequence. Add `chunk_size: int = 8192` prefill segmentation in the serving layer (`src/inference/paged_kv.py` or `gateway/kv_cache_compression.py`). Already partially scaffolded in `AMCPrefixChunk` — connect it to the model forward pass.

---

### 2.6 Agents

#### 2.6.1 Memory-conditioned ReAct
Current ReAct loop (`src/agent/react_loop.py`) does not pull from AMC mid-loop.
**Proposal:** Add `retrieve_amc(query)` call after `observe()`:
```
observe → amc_retrieve → plan → act → reflect → amc_write
```
Make Tier-2 session memory visible to the planner within the same episode.

#### 2.6.2 Reputation: Bayesian trust per-source
The current `src/agent/reputation.py` exists but isn't wired to Tier-3 trust.
**Integration:** After `AMCTier3Hook.promote()`, update the source's Bayesian reputation score:
```python
reputation.update(source, confidence=entry.confidence, outcome="promoted")
```
Then expose `reputation_score` as a factor in Tier-3 write gates.

---

## 3. Original Engine / Architecture Concepts

The following are brainstorming-level concepts that combine Aurelius primitives in new ways.

### 3.1 "AMC-Transformer": Per-Layer Differentiable Memory

**Core idea:** Turn AMC from a runtime hook into a learned per-layer module. Each transformer layer gets its own M_l^1, M_s^2, M^3 memory bank that participates in the forward pass as an additional key-value source.

```python
class AMCTransformerBlock(TransformerBlock):
    def forward(self, x, freqs_cis, mask, past_kv):
        # Standard attention path
        attn_out, kv = self.attn(self.attn_norm(x), ...)
        x = x + attn_out

        # AMC retrieval + gated fusion per layer
        h = x  # current layer hidden state
        memory_context = self.amc_retrieve(h, layer_idx=self.layer_idx)
        g = self.amc_gate(h, memory_context)   # learned gate
        x = x + g * memory_context               # gated residual

        # Write gate
        write_signal = self.amc_write_gate(h)
        if write_signal > self.write_threshold:
            self.amc_memory.write(h.detach(), write_signal)

        return x, kv
```

**Key operations (from AMC formal model):**
- Query: `q_l = W_q * h_l`
- Retrieve: top-k by `cosine(q, k_i) + β_i - decay_i`
- Fuse: `h' = h + Σ_z g_z * P_z * r_z`

**Novelty claim:** per-layer memory with differentiable read/write that is integrated into the standard transformer block, not an external RAG system.

**Effort estimate:** 3-4 weeks for PyTorch prototype, 6-8 weeks for integration with existing training pipeline.

---

### 3.2 "Confidence-Sparse MoE with Expert Safety Gates"

**Core idea:** Extend the existing `SparseMoELayer` with per-expert confidence routing.

#### Current MoE:
```
top-2 routing by token-projected affinity → experts → weighted sum
```

#### Enhancement: ExpertSafetyAffinity + Confidence Kernel
```python
class ConfidenceSparseMoE(SparseMoELayer):
    def route(self, x):
        # Base routing scores
        gate_logits = self.gate(x)   # [B, T, num_experts]

        # Expert safety EMA (tracks harm rate per expert)
        safety_penalty = self.expert_safety_ema.unsqueeze(0).unsqueeze(0)  # [1, 1, E]

        # Conf modulated by safety
        adjusted_logits = gate_logits - self.safety_weight * safety_penalty

        # Top-2
        topk_weights, topk_indices = topk(adjusted_logits, k=2)

        # Update EMA post-forward (safety signal from alignment checker)
        self.update_safety_ema(expert_outputs, harm_mask)

        return dispatch(x, topk_indices, topk_weights)
```

**Novelty:** MoE routing that incorporates runtime safety feedback, reducing reliance on harmful experts over time without explicit retraining.

---

### 3.3 "AMC-Guided Long2Short Transfer"

**Core idea:** Use AMC Tier-3 policy and consolidation statistics to weight the distillation loss in long2short transfer.

Long2short: use long-CoT activations to improve short-CoT. Current approach: length penalty + model averaging.

**AMC integration:**
1. During long-CoT fine-tuning, AMC Tier-2 stores active reasoning steps.
2. AMC Tier-3 consolidates high-confidence reasoning patterns as *reasoning policy* entries.
3. During distillation (long→short), weight the distillation loss by AMC importance score of the source token:
   ```
   L_distill = Σ_t w_t * CE(p_short || p_long)
   w_t = 0.5 + 0.5 * (amc_importance(t) / max_importance)
   ```
4. Promoted reasoning patterns (high consolidation score) carry more weight in the short model.

**Benefit:** short-CoT model prioritizes learning the most durable reasoning steps, not the surface chain length.

---

### 3.4 "3-Tier Memory as Modular Plugin System"

Currently AMC Tier-1/2/3 are tightly coupled code paths in `src/memory/`.
**Proposal:** Define abstract `MemoryTier` protocol; implement Tier-1/2/3 as plugins with `read()`/`write()`/`close()` interface.

```python
class MemoryTier(Protocol):
    def read(self, query: str) -> list[MemoryRecord]: ...
    def write(self, record: MemoryRecord) -> bool: ...
    def consolidate(self) -> ConsolidationReport: ...
    def stats(self) -> MemoryStats: ...
```

Then the runtime becomes:
```python
class AMCRuntime:
    def __init__(self, tiers: list[MemoryTier]):
        self.tiers = tiers

    def retrieve(self, query):
        return [r for tier in self.tiers for r in tier.read(query)]

    def promote(self, record):
        for tier in self.tiers:
            tier.write(record)
```

**Benefit:** enables plug-and-play of different memory backends (e.g., Redis-backed Tier-3, vector-database semantic memory) without rewriting core logic.

---

### 3.5 "Per-Tier Attention Mask" (PAM)

**Core idea:** Each memory tier contributes different trust levels → project them as separate attention masks in the transformer forward pass.

- Tier-1: always visible (local sliding window)
- Tier-2: visible to top-K attention heads (experts)
- Tier-3 policy: force-injected heads for constitutional checks

```python
def forward(self, x, amc_context):
    # amc_context: {trusted: tokens, allowed: tokens, quarantined: tokens}
    attention_mask = build_multi_tier_mask(
        base_mask=attn_mask,
        trusted=amc_context["trusted"],   # injected to specific heads
        quarantined=amc_context["quarantined"],  # masked from all
    )
```

This makes AMC an *actual* attention mechanism modifier, not just prompt injection.

---

## 4. Quick-Start Implementation Plan

## 3. Long Context Scaling Plan

### Kimi k1.5 inspired context window growth

**Goal:** Scale training + inference context from current 8K-32K to 128K tokens with sustained quality.

**Steps:**

1. **RoPE frequency rebase** (already partially done in YaRN)
   - Extend theta from 500,000 → 2,000,000
   - Use `yarn_rope_frequencies` with interpolation ratio `alpha=1.0` for new positions
   - Validate via Needle-in-a-Haystack at 32K → 64K → 128K

2. **Chunked attention in prefill**
   - Already scaffolded via `AMCPrefixChunk.chunk_prefix_segments`
   - Connect to forward pass: split long sequences, process each chunk sequentially, stitch KV cache

3. **Rolling KV cache with importance-weighted eviction**
   - Combine with existing eviction strategies
   - Kimi k1.5 shows context length itself is the scaling axis, not just model size

### Runtime contract updates

Add to API:
```json
{
  "model": "aurelius-forge-1b",
  "messages": [],
  "amc": {
    "episodic": true,
    "long_term": true,
    "session_id": "optional",
    "debug_retrievals": false,
    "consolidation_threshold": 0.65,
    "context_window": 32768
  }
}
```

---

## 4. Alignment & Memory-Native Safety

### Constitutional Tier-3 Integration

Store constitutional principles as Tier-3 policy entries:

```python
tier3.promote_policy(
    key="harm.self_harm",
    text="Self-harm content must not be generated.",
    version="v1",
    priority="high",
)
```

These become highest-priority retrievals during alignment checks.

### Universal Gradient Safety with TopoGCL

Add to `src/safe/universal_gradient_safety.py`:
- Load safety manifold from `embeddings/clusters/safety_manifold_*.parquet`
- Add `UniversalGradientTopoGCL(nn.Module)` that applies topological gradient correction
- Register as callback in trainer loop

---

### Tier-N: Symbolic / Semantic Layer (Proposal)

Extend AMC with an explicit semantic reasoning tier between Tier-2 and Tier-3:

- **Tier-1.5 (Working Semantic):** parse generated tokens into typed facts (using a small local fact-extraction model); feed back as immediate context
- **Tier-2.5 (Session Semantic):** relation-graph extracted from session; benefit: semantic reasoning doesn't require full LLM generation cost

This closes the gap between the existing `SemanticMemory` concept graph and the AMC forward pass.

---

## 5. Open Questions

1. **When to wire per-layer hooks?** Needs benchmark baseline first. Do not add AMC to forward pass until Tier-2 wins on AMC-Memory (threshold: 0.80 overall with tier2_delta >= 0.10).

2. **Token ID bridging:** Should AMC use model tokenizer for content → token IDs, or own hashing? Recommendation: tokenizer for real tiers, hash only for diagnostic/metadata.

3. **MMLU budget:** T1 blocks have 5 test budgets. Does AMC instrumentation push over? Measure baseline T1 tests first, then add AMC one tier at a time.

4. **ExpertSafetyAffinity slots:** existing `CompressedSparseAttention`/`HeavilyCompressedAttention` handle attention-footprint. MoE expert safety penalty is orthogonal — test on 5B+ MoE scale.

---

## References

- AMC Technical Paper: `docs/papers/amc_technical_paper.md`
- Auarius README (architecture): root `README.md`
- Kimi k1.5: https://ar5iv.labs.arxiv.org/html/2501.12599
- YaRN: https://ar5iv.labs.arxiv.org/html/2309.00071
- Mamba2: https://arxiv.org/abs/2405.21060
- DeepSeek-V2: https://arxiv.org/abs/2405.04434
- Gemma 2: https://arxiv.org/abs/2408.00118
- FlashAttention2: https://arxiv.org/abs/2401.08258

---

*This document is a research synthesis and brainstorming artifact. It should be validated against the bench before any major refactor.*
