# AMC Complete Buildout Plan

## Goal

Turn the Aurelian Memory Core from a set of excellent contracts and
protocols into a **working, verifiable, replayable, memory-native
engine** — a model that uses differentiable 3-tier memory in its
forward pass, persists state across sessions, trains with
memory-aware objectives, and wraps it all in an agent loop that
actually consolidates knowledge over time.

**Critical path:** Model Layer → Training Pipeline → Durable
Runtime → Agent Deepening → Validation

---

## Phase 0: Foundation Prerequisites (1–2 weeks)

Things that must exist before model work starts.

### 0.1 — Wire the Rust MemoryPageTable into Python

The Rust crate is compiled but never imported. Wire it up:

```
src/memory/native_page_table.py    ← ctypes/PyO3 bridge
  - allocate/free memory pages (GPU or CPU)
  - page-level trust tagging
  - trust-aware LRU eviction using AMCMemoryCacheKey
```

**Validation:**
- Unit tests: allocate 10K pages, evict by trust (verify that
  QUARANTINED pages evict first, VERIFIED pages last)
- Memory leak test: 1M alloc/free cycles, no growth
- Benchmark: <1μs page access time

### 0.2 — Durable Event Log for SDB Runtime

SDB Runtime currently holds everything in memory. Replace
`self._events: list[ReplayEvent]` with an append-only log:

```
src/memory/sdb_log.py              ← SQLite or LMDB backend
  - append(event: ReplayEvent) → int (sequence number)
  - replay(from_seq: int) → Iterator[ReplayEvent]
  - checkpoint(save_path) / restore(load_path)
  - replay_hash consistency check
```

**Validation:**
- Write 100K events, replay all, verify replay_hash chain
- Simulate crash: kill process mid-commit, restore, verify
  no orphan proposals
- Adversarial probe: mutate the log file, detect corruption

### 0.3 — Tier-2/3 Checkpoint Serialization

```
src/memory/checkpoint.py
  - save_tier2(episodic: EpisodicMemory, path)
  - save_tier3(hook: AMCTier3Hook, path)
  - load_tier2(path) → EpisodicMemory
  - load_tier3(path) → AMCTier3Hook
  - Format: msgpack or pickle with schema version tag
  - Include AMC_VERSION header so old checkpoints can be detected
```

**Validation:**
- Save, kill, load, verify all entries + trust levels preserved
- Cross-platform: save on Linux, load on macOS (byte order)
- Version migration: v1 → v2 schema upgrade path

---

## Phase 1: Model Layer — The Novel Contribution (6–10 weeks)

This is the actual paper. Everything else is scaffolding.

### 1.1 — SSM/Mamba-2 as Per-Layer Working Memory (Tier-1)

Replace the standard multi-head attention KV cache per-layer with a
Mamba-2 Selective State Space that IS the working memory:

```
src/model/amc_ssm_layer.py
  class AMCSSMLayer(nn.Module):
      """Per-layer SSM block that implements AMC Tier-1 working memory."""

      def __init__(self, d_model, d_state, d_conv, expand):
          # Mamba-2 selective scan
          self.ssm = SelectiveStateSpace(d_model, d_state, d_conv, expand)
          # Learned surprise head: hidden state → scalar surprise score
          self.surprise_head = nn.Sequential(
              nn.Linear(d_model, d_model // 4),
              nn.GELU(),
              nn.Linear(d_model // 4, 1),
              nn.Sigmoid()
          )
          # Learned gate networks (differentiable)
          self.decay_gate_net = nn.Linear(d_model, d_state)   # how much to forget
          self.erase_gate_net = nn.Linear(d_model, d_state)   # what to erase
          self.write_gate_net = nn.Linear(d_model, d_state)   # what to write

      def forward(self, x, step, prev_state=None):
          # 1. Compute surprise from current hidden state
          surprise = self.surprise_head(x.detach())  # no gradient to model

          # 2. Compute gates (differentiable)
          decay = torch.sigmoid(self.decay_gate_net(x))
          erase = torch.sigmoid(self.erase_gate_net(x))
          write = torch.sigmoid(self.write_gate_net(x))

          # 3. Apply SSM update with gated memory modification
          #    new_state = decay * prev_state - erase * old_memory + write * new_input
          new_state = self.ssm(x, prev_state,
                               decay=decay, erase=erase, write=write)

          # 4. Compress KV for prefix cache compatibility
          compressed_kv = self.compress_state(new_state)  # (B, T, kv_lrank)

          # 5. Emit AMCTensorState
          return new_state, AMCMemoryBlock(
              block_id=f"layer_{self.layer_idx}_step_{step}",
              tokens=...,
              tier=1,
              trust_state=TrustState.UNVERIFIED,
              provenance=f"ssm_layer_{self.layer_idx}",
              salience=surprise.item(),
              surprise_score=surprise.item(),
              kv_ref=compressed_kv,
          )
```

**Why Mamba-2/SSM** rather than standard attention:
- O(n) time, not O(n²) — working memory must be fast per-step
- Selective state = the model already has a "what to remember"
  mechanism; we're making the gate visible to the AMC controller
- The recurrence is naturally "memory-like" — it IS a compressed
  working memory with learned selective retention

**Integration point:** Drop into `src/model/transformer.py` as a
replacement for (or augmentation of) `AttentionBlock`. Each layer
emits its `AMCMemoryBlock` alongside its hidden state.

**Validation:**
- Forward pass produces correct AMCMemoryBlock per layer
- Surprise head outputs ∈ [0, 1]
- Gradient flows through gates but NOT through surprise head
  (detached — surprise is a signal, not a loss)
- Memory state can be serialized/deserialized (checkpoint compat)

### 1.2 — Differentiable Tier-2 Promotion Gate

When surprise exceeds a learned threshold, promote the working memory
compress to Tier-2 episodic storage. This is a hard decision
(store vs skip) but we need gradients:

```
src/model/amc_promotion.py
  class PromotionGate(nn.Module):
      """Gumbel-softmax gate for discrete store/skip decision."""

      def __init__(self, d_model, temperature=0.5):
          self.gate_net = nn.Linear(d_model, 2)  # [skip_logit, store_logit]
          self.temperature = temperature

      def forward(self, hidden, surprise):
          logits = self.gate_net(hidden)
          # Hard decision in forward pass
          hard = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
          # But gradient flows through the soft version
          soft = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
          # Straight-through: hard in forward, soft in backward
          return hard[:, 1], soft[:, 1]  # (store decision, soft gradient)
```

**Integration:**
```
# In the transformer forward pass:
for layer_idx, layer in enumerate(self.layers):
    hidden = layer(hidden)
    memory_block = layer.memory_output

    # Promotion decision (differentiable)
    store, soft_store = self.promotion_gate(hidden, memory_block.surprise_score)

    if store > 0.5:  # hard decision
        # Write to Tier-2 episodic memory
        tier2_hook.observe(content=..., surprise=memory_block.surprise_score)

    # The soft score contributes to the loss for training the gate
    promotion_losses.append(soft_store * (1 - memory_block.surprise_score))
```

**Validation:**
- Gradient flows back through promotion gate to surprise head
- Gate learns to store high-surprise and skip low-surprise
- Ablation: disable gate → verify no performance change on tasks
  that don't need memory

### 1.3 — AMC-Aware Transformer Architecture

```
src/model/amc_transformer.py
  class AMCTransformer(nn.Module):
      """Transformer with per-layer AMC working memory and promotion gates."""

      def __init__(self, config: AureliusConfig):
          # Standard transformer components
          self.embed = nn.Embedding(config.vocab_size, config.d_model)
          # Hybrid: alternate attention and AMC-SSM layers
          self.layers = nn.ModuleList([
              self._build_layer(i, config)
              for i in range(config.n_layers)
          ])
          self.norm = RMSNorm(config.d_model)
          self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

          # AMC components
          self.promotion_gate = PromotionGate(config.d_model)
          self.tier2_hook = AMCTier2Hook()  # existing hook, now wired in
          self.tier3_hook = AMCTier3Hook()  # existing hook, now wired in

      def _build_layer(self, idx, config):
          """Alternate between standard attention and SSM memory layers."""
          if idx % 2 == 0:
              return StandardAttentionLayer(config)
          else:
              return AMCSSMLayer(config.d_model, d_state=64, ...)

      def forward(self, input_ids, *, session_id=None, step=0):
          hidden = self.embed(input_ids)
          memory_blocks = []
          promotion_loss = 0.0

          for layer in self.layers:
              hidden, block = layer(hidden, step=step)
              if block is not None:  # SSM layers produce blocks
                  memory_blocks.append(block)
                  store, soft = self.promotion_gate(hidden, block.surprise_score)
                  promotion_loss += soft * (1 - block.surprise_score)

          hidden = self.norm(hidden)
          logits = self.lm_head(hidden)

          return AMCForwardOutput(
              logits=logits,
              memory_blocks=memory_blocks,
              promotion_loss=promotion_loss,
          )
```

**Validation:**
- `amc_memory_runner` benchmark with `--generator engine` passes
  cross_session_recall task on oracle
- Forward pass produces `AMCForwardOutput` with logits AND memory blocks
- Gradient flows end-to-end from logits through SSM layers to embeddings

### 1.4 — AMC Loss Terms (Training Objectives)

Add three memory-specific losses to the standard cross-entropy:

```
src/training/amc_losses.py

  def memory_consistency_loss(memory_blocks, tier2_entries):
      """Reward the model when retrieved memories match current context.

      If the model generates token sequence X, and Tier-2 retrieval
      returns memory M that is consistent with X, loss decreases.
      """
      # For each memory block in the current forward pass,
      # check if retrieved Tier-2 entries are "consistent"
      # (high cosine similarity in embedding space)
      losses = []
      for block in memory_blocks:
          retrieved = tier2_hook.retrieve(block.provenance)
          for entry in retrieved:
              sim = cosine_similarity(block.embedding, entry.embedding)
              losses.append(1.0 - sim)  # minimize divergence
      return torch.mean(torch.stack(losses))

  def surprise_prediction_loss(predicted_surprise, actual_importance):
      """Train the surprise head to predict which turns matter.

      actual_importance is determined post-hoc: did this turn lead
      to a successful Tier-3 promotion or a user correction?
      """
      return F.binary_cross_entropy(predicted_surprise, actual_importance)

  def consolidation_reward_loss(tier3_state_before, tier3_state_after, quality_score):
      """DPO-style reward for consolidation decisions.

      quality_score comes from: did the promoted fact help the model
      answer correctly in a later session?
      """
      return -quality_score * log_prob(consolidation_decision)
```

**Validation:**
- Loss terms are differentiable
- Training reduces surprise prediction error over epochs
- Consolidation quality improves with more training iterations

---

## Phase 2: Durable Runtime (3–4 weeks)

The runtime turns "contract theater" into real state.

### 2.1 — Persistent SDB-Runtime + Event Log

Upgrade `src/memory/sdb_log.py` from Phase 0 to a full append-only
event-sourced store:

```python
# Concrete: SQLite-backed implementation
class SDBSQLiteLog:
    """Append-only event log for SDB Memory Runtime.

    Tables:
    - amc_events(id, seq, event_type, proposal_id, timestamp,
                 replay_hash, metadata_json)
    - amc_commits(idempotency_key, proposal_id, commit_id,
                  replay_hash, verified_at, meta_json)
    - amc_proposals(proposal_id, session_id, step, fingerprint,
                    payload_json, status)
    """

    def append_and_rehash(self, event: ReplayEvent) -> int:
        """Append event, verify chain hash (sha256(prev + current))."""
        ...

    def replay_from(self, seq: int) -> list[ReplayEvent]:
        """Replay events from sequence number. Used to reconstruct state."""
        ...

    def verify_chain_integrity(self) -> bool:
        """Verify that every replay_hash chains correctly.
        Fails if any event was tampered with."""
        chain_ok = True
        prev_hash = ""
        for event in self.replay_from(0):
            expected = stable_hash({"prev": prev_hash, "event": event.to_dict()})
            if event.replay_hash != expected:
                chain_ok = False
                break
            prev_hash = event.replay_hash
        return chain_ok
```

**Validation:**
- 1M events written, chain integrity verified
- Simulate mid-write crash: `fsync` after each batch, verify recovery
- Adversarial: modify one event byte, verify integrity check fails

### 2.2 — State Reconstruction from Event Log

```python
src/memory/state_reconstruction.py

def reconstruct_tier2_state(events: list[ReplayEvent],
                            episodic_store: EpisodicMemory) -> EpisodicMemory:
    """Replay propose→verify→commit events to rebuild Tier-2 state.

    This is what makes AMC replayable: given the event log,
    you can reconstruct the exact memory state at any point in time.
    """
    active_store = episodic_store.empty_copy()
    for event in events:
        if event.event_type == "committed":
            meta = event.metadata
            if meta["target_tier"] == "tier2" and meta["operation"] == "store":
                active_store.store(
                    role=meta["role"],
                    content=meta["content"],
                    importance=meta["importance"],
                    id=meta["entry_id"],
                )
        elif event.event_type == "rejected":
            pass  # rejected proposals never entered the store
    return active_store

def reconstruct_tier3_state(events: list[ReplayEvent],
                            config: AMCTier3Config) -> AMCTier3Hook:
    """Replay promotion/quarantine/consolidation events to rebuild Tier-3."""
    hook = AMCTier3Hook(config)
    for event in events:
        # ... replay promote, quarantine, revoke, expire events
        ...
    return hook
```

**Validation:**
- Start from empty store, replay N events, reach same state as
  the live store
- Replay is deterministic: same events → same state (no floats
  without seeds, no wall-clock timestamps)
- Performance: replay 100K events in < 30 seconds

### 2.3 — Trust-Aware KV Cache for Serving

Wire `AMCPrefixCompiler` into an actual KV cache:

```python
src/serving/amc_kv_cache.py

class AMCKVCache:
    """Trust-aware paged KV cache for inference serving.

    Key insight: memory blocks with different trust states
    get DIFFERENT cache keys. A memory that was QUARANTINED
    and then TRUSTED later will invalidate its old cached
    prefix and recompile — you always serve the current
    trust state.
    """

    def __init__(self, page_size: int, max_pages: int):
        self.pages: dict[AMCMemoryCacheKey, KVPage] = {}
        self.lru = OrderedDict()  # cache_key → timestamp
        self.trust_policy = TrustEvictionPolicy()

    def compile_and_cache(self, blocks: list[AMCMemoryBlock],
                          policy_version: str) -> list[AMCPrefixSegment]:
        compiler = AMCPrefixCompiler(policy_version=policy_version)
        result = compiler.compile(blocks)
        # Cache only trusted + allowed segments
        for seg in result.trusted + result.allowed:
            key = seg.cache_key
            if key not in self.pages:
                self.pages[key] = self._allocate_page(seg.tokens, seg.kv_ref)
            self.lru[key] = time.monotonic()
        return result

    def evict(self, target_count: int):
        """Evict LRU pages, but QUARANTINED pages evict first,
        VERIFIED pages evict last."""
        sorted_keys = self.trust_policy.sort_for_eviction(
            self.lru.keys(), self.pages
        )
        while len(self.pages) > target_count:
            key = sorted_keys.pop(0)
            del self.pages[key]
            del self.lru[key]
```

**Validation:**
- Feed 10K unique memory blocks through compiler → verify all VERIFIED
  blocks hit cache, all QUARANTINED blocks produce quarantine segments
- Eviction: under memory pressure, quarantined blocks gone first
- Trust change: update trust from UNVERIFIED to VERIFIED → old cache
  key invalidated, new key allocated

---

## Phase 3: Training Pipeline (6–8 weeks)

### 3.1 — Memory-Aware Training Data

```
src/training/amc_data.py

def build_memory_session_samples(transcript: list[Message]) -> list[TrainingSample]:
    """Turn a multi-session transcript into training samples that
    exercise the full 3-tier memory pathway.

    Each sample includes:
    - input_ids: the prompt tokens
    - target_ids: the expected response
    - session_id: for Tier-2 scoping
    - surprise_labels: per-token surprise importance labels
    - tier2_ground_truth: what SHOULD be in Tier-2 after this step
    - tier3_ground_truth: what SHOULD promote to Tier-3
    """
    ...

def create_ablation_pairs(
    session_transcripts: list[list[Message]],
    *,
    mode: Literal["no_amc", "tier2_only", "full_amc"]
) -> list[TrainingSample]:
    """Create the same samples under three AMC configurations.
    Used for ablation benchmarking and DPO pairs.
    """
    ...
```

### 3.2 — Training Loop with AMC Loss

```python
src/training/amc_trainer.py

class AMCTrainer:
    """Training loop that tracks and optimizes memory-specific objectives."""

    def __init__(self, model: AMCTransformer, config: TrainConfig):
        self.model = model
        self.promotion_gate_optimizer = torch.optim.Adam(
            model.promotion_gate.parameters(), lr=1e-4
        )
        self.surprise_head_optimizer = torch.optim.Adam(
            # surprise_head is inside each SSM layer
            [p for l in model.layers if hasattr(l, 'surprise_head')
             for p in l.surprise_head.parameters()],
            lr=1e-5
        )

    def train_step(self, batch: TrainingBatch) -> dict[str, float]:
        # Forward pass — model emits logits + memory_blocks
        output = self.model(batch.input_ids, session_id=batch.session_id)

        # Standard SFT loss
        sft_loss = F.cross_entropy(output.logits.view(-1, V),
                                   batch.target_ids.view(-1))

        # Memory-aware losses
        surprise_loss = surprise_prediction_loss(
            predicted=[b.surprise_score for b in output.memory_blocks],
            actual=batch.surprise_labels,
        )

        # Total loss with memory weighting
        alpha = 0.7   # SFT primary objective
        beta = 0.2    # Surprise prediction quality
        gamma = 0.1   # Promotion gate quality

        total = alpha * sft_loss + beta * surprise_loss + gamma * output.promotion_loss

        # Update model
        total.backward()
        self.gradient_clip(1.0)
        self.optimizer.step()

        # Update memory components separately (slow learning rate)
        self.promotion_gate_optimizer.step()
        self.surprise_head_optimizer.step()

        return {"sft_loss": sft_loss.item(),
                "surprise_loss": surprise_loss.item(),
                "promotion_loss": output.promotion_loss.item()}
```

### 3.3 — DPO/GRPO for Memory-Consistent Behavior

```python
src/training/amc_dpo.py

def memory_dpo_pairs(session: list[Message]) -> list[DPOPair]:
    """Create preference pairs based on memory consistency.

    Preferred: response that correctly uses Tier-2/Tier-3 memories
    Rejected:  response that ignores or contradicts stored memories
    """
    pairs = []
    for turn in session:
        # Run model with memory → "preferred" candidate
        with_memory = model.generate(turn.prompt, use_amc=True)

        # Run model without memory → "rejected" candidate
        no_memory = model.generate(turn.prompt, use_amc=False)

        # Score both against ground truth
        score_with = memory_consistency_score(with_memory, turn.ground_truth)
        score_without = memory_consistency_score(no_memory, turn.ground_truth)

        if score_with > score_without + 0.1:  # margin
            pairs.append(DPOPair(preferred=with_memory, rejected=no_memory))

    return pairs

def memory_grpo_reward(completions: list[str],
                        reference: str,
                        tier3_state: AMCTier3Hook) -> list[float]:
    """GRPO group reward: each completion scored by:
    1. Factual accuracy (reference matching)
    2. Memory consistency (does it honor Tier-3 facts?)
    3. Surprise accuracy (was the model's surprise prediction correct?)
    """
    rewards = []
    for comp in completions:
        factual = factual_accuracy(comp, reference)
        memory_ok = check_tier3_consistency(comp, tier3_state.prioritize())
        surprise_correct = ...  # was surprise predicted correctly?
        rewards.append(0.5 * factual + 0.3 * memory_ok + 0.2 * surprise_correct)
    return rewards
```

### 3.4 — Ablation Study Pipeline

```python
src/eval/amc_ablation.py

def run_ablation_study(
    model_path: str,
    *,
    configs: list[str] = ["baseline", "tier2_only", "full_amc"],
    benchmarks: list[str] = ["gsm8k", "mmlu", "longbench", "amc_memory"],
) -> AblationReport:
    """Run the same model with different AMC configurations.

    This produces the evidence for the paper:
    "Baseline (no AMC) vs Tier-2 only vs Full 3-tier"
    across standard AND memory-specific benchmarks.
    """
    results = {}
    for config_name in configs:
        for bench_name in benchmarks:
            score = run_benchmark_with_config(model_path, config_name, bench_name)
            results[(config_name, bench_name)] = score

    return AblationReport(
        results=results,
        delta_tier2=results[("tier2_only", "amc_memory")] - results[("baseline", "amc_memory")],
        delta_full=results[("full_amc", "amc_memory")] - results[("baseline", "amc_memory")],
    )
```

**Validation:**
- Ablation shows: full_amc > tier2_only > baseline on AMC benchmarks
- Performance on standard benchmarks (GSM8K, MMLU) is not degraded
- Surprise prediction accuracy converges to >70% over training

---

## Phase 4: Agent Deepening (4–6 weeks)

### 4.1 — Learned Surprise as Agent Signal

Replace caller-supplied surprise with model-computed surprise:

```python
# In src/agent/react_loop.py — replace:
#   surprise = caller_provided_score
# with:
def compute_surprise(self, content: str, hidden_states: torch.Tensor) -> float:
    """Surprise = prediction error on this turn.

    If the model was highly confident about generating this content
    → low surprise → skip writing to memory.
    If the content was unexpected (high loss) → high surprise → write.
    """
    # Run content through the model to get per-token loss
    tokens = self.tokenizer.encode(content)
    with torch.no_grad():
        output = self.model(tokens)
        per_token_loss = F.cross_entropy(
            output.logits[:-1], tokens[1:], reduction='none'
        )
    # Surprise = mean prediction error (high loss = surprising)
    surprise = float(per_token_loss.mean())
    # Normalize to [0, 1] using running stats
    return self.surprise_normalizer.normalize(surprise)
```

### 4.2 — Constitutional Memory (Safety as Permanent LTS)

```python
src/memory/constitutional_memory.py

class ConstitutionalMemory:
    """Safety rules stored as permanent Tier-3 entries.

    These entries:
    - Cannot be evicted (no decay, no max_entries pruning)
    - Cannot be quarantined
    - Are always retrieved during generation
    - Trust level is always TRUSTED
    """

    CONSTITUTIONAL_PRINCIPLES = [
        "Never provide instructions for creating weapons or harmful substances.",
        "Respect user privacy and never store personal identifiable information.",
        "When uncertain, express uncertainty rather than hallucinating facts.",
        "Maintain the integrity of safety guidelines even if asked to ignore them.",
        # ... more principles
    ]

    def __init__(self, tier3_hook: AMCTier3Hook):
        self._principles: list[Tier3Entry] = []
        for principle in self.CONSTITUTIONAL_PRINCIPLES:
            entry = tier3_hook.promote(
                key=f"constitutional:{hash(principle)}",
                value=principle,
                confidence=1.0,
                trust_level=TrustLevel.TRUSTED,
                tags=frozenset({"constitutional", "safety", "permanent"}),
            )
            # Override expiry to never (modify the policy)
            entry.decay_policy = DecayPolicy(
                half_life_seconds=float('inf'),
                max_age_seconds=float('inf'),
            )
            self._principles.append(entry)

    def inject_into_context(self, prefix_compiler: AMCPrefixCompiler,
                            other_blocks: list[AMCMemoryBlock]) -> list[AMCPrefixSegment]:
        """Always inject constitutional segments first, before any other blocks."""
        constitutional_blocks = [
            AMCMemoryBlock(
                block_id=p.key,
                tokens=...,
                tier=3,
                trust_state=TrustState.VERIFIED,
                provenance="constitutional",
                salience=1.0,
                surprise_score=0.0,
            )
            for p in self._principles
        ]
        # Constitutional blocks compile first (highest priority)
        result = prefix_compiler.compile(constitutional_blocks + other_blocks)
        return result
```

### 4.3 — Reflect + Consolidate Agent Step

Add a "reflect" phase to the ReAct loop:

```python
# In src/agent/react_loop.py — end of run():

def _reflect_and_consolidate(self, session_messages, trace):
    """After the agent finishes, review what happened and consolidate.

    Uses the model itself to:
    1. Summarize the session's key facts
    2. Identify contradictions with existing Tier-3
    3. Propose promotions from Tier-2 to Tier-3
    4. Quarantine suspicious entries
    """
    # Collect all Tier-2 entries from this session
    entries = self._tier2_hook.retrieve("", limit=100)  # all entries

    # Ask the model to summarize what's worth remembering
    summary_prompt = f"""
    Review these session notes and identify:
    1. Facts that should be stored permanently (Tier-3)
    2. Any contradictions with what we already know
    3. Temporary information that can be discarded

    Session notes:
    {[e.content for e in entries]}
    """

    reflection = self._generate(summary_prompt)

    # Parse reflection and execute consolidation
    for fact in self._parse_promotable_facts(reflection):
        self._tier3_hook.promote(
            key=fact.key,
            value=fact.value,
            confidence=fact.confidence,
            source_tier2_id=fact.source_entry_id,
        )

    for contradiction in self._parse_contradictions(reflection):
        self._tier3_hook.quarantine(
            key=contradiction.key,
            value=contradiction.value,
            confidence=0.2,  # low confidence → quarantine
        )
```

### 4.4 — Skill Crystallization

```python
src/agent/skill_crystallizer.py

class SkillCrystallizer:
    """Detect repeated patterns in Tier-3 and compress them into skills.

    When a fact has been retrieved N times (> threshold), compress
    it into a more compact, higher-abstraction form and re-store
    at elevated confidence.
    """

    def __init__(self, tier3_hook: AMCTier3Hook, retrieval_threshold: int = 5):
        self.tier3 = tier3_hook
        self.retrieval_threshold = retrieval_threshold
        self._retrieval_counts: dict[str, int] = {}

    def track_retrieval(self, entry: Tier3Entry):
        key = entry.key
        self._retrieval_counts[key] = self._retrieval_counts.get(key, 0) + 1

    def crystallize_if_ready(self, model_generate_fn) -> list[Tier3Entry]:
        """Check for entries that hit the retrieval threshold.
        Ask the model to compress them into abstract skills."""
        crystallized = []
        for key, count in self._retrieval_counts.items():
            if count >= self.retrieval_threshold:
                entry = self.tier3._store.get(key)
                if entry and entry.trust_level == TrustLevel.TRUSTED:
                    # Ask model to compress
                    compressed = model_generate_fn(
                        f"Compress this frequently-retrieved fact into a "
                        f"more abstract principle:\n{entry.value}"
                    )
                    # Store compressed version at higher confidence
                    new_entry = self.tier3.promote(
                        key=f"crystal:{key}",
                        value=compressed,
                        confidence=min(1.0, entry.confidence + 0.1),
                        tags=frozenset({"crystallized", "skill"}),
                    )
                    crystallized.append(new_entry)
                    # Reset count
                    self._retrieval_counts[key] = 0
        return crystallized
```

### 4.5 — Stochastic Latent Recall End-to-End

SLR exists as a scaffold. Wire it into the agent loop:

```python
# In src/agent/react_loop.py:

def _run_with_slr(self, query: str) -> str:
    """Use SLR to generate diverse recall candidates before responding.

    1. Generate K perturbed recall trajectories (stochastic)
    2. Select the best candidate (deterministic)
    3. Feed selected recall into context before generating response
    """
    slr_config = default_slr_config(
        enabled=True,  # enable for this run
        k=4,
        noise_sigma=0.1,
        replay_seed=self.session_seed,  # deterministic per session
    )

    slr_query = SLRQuery(
        query_id=self.current_step_id,
        query_text=query,
        memory_scope=("tier2", "tier3"),
        baseline_candidate_ids=(),
    )

    candidates = generate_slr_candidates(config=slr_config, query=slr_query)
    selection = select_slr_candidate(
        candidates=candidates,
        metric=SLRSelectionMetric.SCORE,
        query_id=query.query_id,
        replay_seed=slr_config.replay_seed,
    )

    # Use selected recall keys to retrieve memories
    recalled = self._tier3_hook.prioritize(limit=5)
    recalled_context = self._format_memories(recalled)

    # Build response with recalled context
    response = self._generate(f"{recalled_context}\n\nQuery: {query}")
    return response
```

---

## Phase 5: Verification & Reproducibility (2–3 weeks)

### 5.1 — Replay Execution Engine

```python
src/memory/replay_engine.py

class ReplayEngine:
    """Replay a full SDB event log and reconstruct any historical state.

    Used for:
    - Debugging: "why did the agent make this decision at step 42?"
    - Testing: replay production logs in CI to verify regressions
    - Reproducibility: demonstrate that a paper result was deterministic
    """

    def __init__(self, event_log: SDBSQLiteLog):
        self.log = event_log

    def replay_to(self, target_seq: int,
                  *,
                  include_quarantined: bool = False) -> ReplaySnapshot:
        """Reconstruct Tier-2 + Tier-3 state at sequence `target_seq`."""
        events = self.log.replay_from(0, limit=target_seq)
        tier2_state = reconstruct_tier2_state(events, EpisodicMemory())
        tier3_state = reconstruct_tier3_state(events, AMCTier3Config())

        return ReplaySnapshot(
            sequence=target_seq,
            tier2=tier2_state,
            tier3=tier3_state,
            events_replayed=len(events),
            chain_integrity=self.log.verify_chain_integrity(),
            timestamp_of_last_event=events[-1].timestamp if events else None,
        )

    def diff_snapshots(self, seq_a: int, seq_b: int) -> ReplayDiff:
        """Diff two snapshots to see what changed between them."""
        snap_a = self.replay_to(seq_a)
        snap_b = self.replay_to(seq_b)
        # Tier-2: entries added/removed between seq_a and seq_b
        # Tier-3: promotions, revocations, decays
        return ReplayDiff(
            tier2_added=snap_b.tier2 - snap_a.tier2,
            tier2_removed=snap_a.tier2 - snap_b.tier2,
            tier3_promotions=...,
            tier3_revocations=...,
            events_between=events[seq_a:seq_b],
        )
```

### 5.2 — VEL + AMC Benchmark Integration

```python
# Run benchmark through the VEL registry for paper-quality provenance

def record_amc_benchmark_run(
    model_path: str,
    profile: str,
    scores: dict[str, float],
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> VELRecord:
    """Record one AMC benchmark run with full provenance."""
    checkpoint_sha = sha256_file(model_path)
    config_hash = stable_hash(load_config(model_path))

    record = VELRecord(
        claim_type=VELClaimType.BENCHMARK,
        claim=f"AMC-Memory {profile}: overall={scores['overall']:.3f}",
        status=VELRunStatus.PASS if scores['overall'] >= 0.8 else VELRunStatus.FAIL,
        sources=[
            VELSource(type=VELSourceType.CHECKPOINT, path=model_path,
                      sha256=checkpoint_sha),
            VELSource(type=VELSourceType.CONFIG, hash=config_hash),
        ],
        environment=VELEnvironment.capture(),
        # ...
    )

    append_vel_record(record, registry_path)
    return record
```

### 5.3 — Paper-Quality Reproducibility Bundle

```
docs/reproducibility/
├── seed.txt                    # All random seeds used
├── environment.yml             # Exact dependencies
├── config/
│   ├── baseline_150M.yaml      # No-AMC model config
│   ├── tier2_1B.yaml           # Tier-2 only config
│   └── full_amc_1B.yaml        # Full 3-tier config
├── scripts/
│   ├── train_all.sh            # Train all three variants
│   ├── run_ablation.sh         # Run ablation study
│   └── paper_figures.py        # Generate publication plots
└── results/
    ├── ablation_scores.jsonl   # Raw benchmark scores
    └── replay_logs/            # SDB event logs for reproducibility
```

---

## Phase 6: Model Family Differentiation (4–6 weeks)

Three variants, each with a REASON to exist:

| Variant          | Parameters | AMC Tier | Surprise | Use Case               |
|------------------|-----------|----------|----------|------------------------|
| Aurelius-Swift   | 150M      | None     | None     | Edge, fast inference   |
| Aurelius-Forge   | 1B        | Tier-2   | Learned  | Workhorse, episodic    |
| Aurelius-Atlas   | 3B+       | T2+T3    | Learned  | Agent mode, long-term  |

### Training order:
1. **Swift**: Standard transformer train (no AMC) — establish baseline
2. **Forge**: Same architecture + AMC SSM layers at every-other layer
   - Trained with surprise prediction loss + DPO memory pairs
3. **Atlas**: Forge + Tier-3 consolidation, longer context window
   - Trained with all three loss terms + GRPO on AMC benchmark

---

## Validation Gates (Must Pass Before Each Phase Completion)

### Phase 1 (Model Layer) gates:
- [ ] Forward pass produces AMCMemoryBlock per SSM layer
- [ ] Surprise head outputs ∈ [0,1], detached from model gradient
- [ ] Gates are differentiable (gradient flows through write/erase/decay)
- [ ] Memory checkpoint save/load round-trip verified
- [ ] ablation: model with gates disabled ≈ baseline

### Phase 2 (Runtime) gates:
- [ ] 100K events written + replay chain integrity = PASS
- [ ] State reconstruction from events matches live state exactly
- [ ] Crash recovery: kill mid-write → restart → state consistent
- [ ] AMCPrefixCompiler output feeds real KV cache (not synthetic)
- [ ] Trust change → cache key change confirmed

### Phase 3 (Training) gates:
- [ ] Surprise prediction accuracy > 70% after training
- [ ] Ablation: full_amc > tier2_only > baseline on AMC benchmark
- [ ] Standard benchmarks not degraded (> 0.95 of baseline)
- [ ] DPO pairs show memory-consistent preference (p < 0.05)
- [ ] VEL registry has provenance for every number in the paper

### Phase 4 (Agent) gates:
- [ ] Reflect step produces > 3 promotion proposals per session
- [ ] Constitutional memory always present in generated context
- [ ] SLR candidates are deterministic given same seed
- [ ] Skill crystallization triggers after threshold retrievals
- [ ] Agent self-corrects when Tier-3 provides contradicting facts

### Phase 5 (Verification) gates:
- [ ] Replay engine reproduces exact state from event log
- [ ] Replay diff between snapshots matches observed changes
- [ ] Paper figure scripts run from seed to plot in one command
- [ ] Adversarial probes from SDB contract review all PASS
- [ ] Full ablation study reproducible on fresh install

---

## Build Order (Critical Path)

```
Week 1-2:   Phase 0 (SQLite log, checkpoint, Rust FFI)
Week 3-8:   Phase 1.1-1.3 (SSM layer, promotion gate, AMC transformer)
Week 5-6:   Phase 2 (runtime, overlapping with Phase 1.3 testing)
Week 9-14:  Phase 1.4 + Phase 3 (AMC losses, training loop, DPO/GRPO)
Week 12-16: Phase 4 (agent deepening, overlaps with late training)
Week 15-18: Phase 5 (verification, reproducibility)
Week 17-22: Phase 6 (model family, ablations, paper)
```

**Total: ~22 weeks** for a solo researcher.
**Can be ~14 weeks** with two focused researchers splitting
model/training and runtime/agent work.

---

## What to Build FIRST (If You Start Tomorrow)

1. **SSM Layer** (`src/model/amc_ssm_layer.py`)
   - Start with Mamba-2 reference implementation (open-source)
   - Add the three gate networks (decay/erase/write)
   - Add the surprise head
   - Test with random weights on a toy sequence

2. **AMC Transformer** (`src/model/amc_transformer.py`)
   - Alternate SSM layers with standard attention
   - Wire in the promotion gate
   - Train on a small (10M tokens) corpus to validate plumbing

3. **Durable SDB Log** (`src/memory/sdb_log.py`)
   - SQLite backend, append-only
   - Chain hash verification
   - Write 1K events, replay, verify

4. **Surprise Loss** (`src/training/amc_losses.py`)
   - Binary cross-entropy: predict which turns have high importance
   - Train the surprise head to get > 60% accuracy on toy data

5. **Ablation Runner** (`src/eval/amc_ablation.py`)
   - Same model, three configs, one benchmark
   - First real evidence: does AMC help?

---

## Key Papers to Implement During Build

| Paper | Relevance | Phase |
|-------|-----------|-------|
| Mamba-2 (Dao & Gu, 2024) | SSM for per-layer working memory | 1 |
| Titans (Meta, 2025) | Memory as a learned component in forward pass | 1 |
| Generative Agents (Park et al., 2023) | Reflection + memory consolidation | 4 |
| DPO (Rafailov et al., 2023) | Memory-consistent preference learning | 3 |
| GRPO (Shao et al., 2024) | Group-relative reward (no value net) | 3 |
| Zep/Graphiti | Temporal knowledge graph for Tier-3 | 4 |
| Constitutional AI (Bai et al., 2022) | Safety as retrievable memory entries | 4 |

---

## Summary

The AMC stack has excellent contracts. The gap is between
"protocol" and "execution". Every `Protocol` needs an
implementation. Every `dataclass` needs a store. Every
`benchmark task` needs a trained model to evaluate.

The build order above ensures that at every phase, you have
something measurable — not just more scaffolding.
