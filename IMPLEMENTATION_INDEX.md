# Aurelius Implementation Index — Pre-Deployment Reference
# Date: 2026-06-21
#
# This file is the single entry point for Claude reviewing the Aurelius
# dataset, context-window, and memory-management stack before any training or
# inference run. Read this first, then the referenced files, then run the
# self-tests before starting compute.
#
# Delivery format: plain Markdown. Each section states purpose, pointer to the
# source file, the invariants that must hold, and the failure modes.

---

# 1. Context-Window Extension: YaRN / LongRoPE

Purpose:
  Permit the 14B / 10B Aurelius model to attend to 256.2k tokens now and 1M tokens
  in v4 without replacing positional embeddings or retraining from scratch. YaRN
  rescales the RoPE frequency basis mathematically; weights stay compatible.

Source:
  aurelius/training/yarn_rescaler.py

Presets:
  aurelius/data/yarn_params.yaml
    - v3_256k: 256,192 target context, standard settings
    - v4_1M:  1,048,576 target context, attention_factor=1.2 for extreme stretch

What Claude must verify before running:
  - Base model is Qwen2.5-family with rope_theta present (script auto-detects from
    model name or config.json; fall back to --base-context).
  - target-context > base-context (script asserts).
  - If hardware is local Mac (28-48 GB RAM), do NOT run with 1M unless KV
    cache offload or ring attention is available. Default guidance: 256k first.
  - Read the dataset_registry.yaml and ensure the longcontext_adaptation mix
    is queued immediately after YaRN rescaling, before any main-phase training.

Invariants:
  - rope_theta after rescaling = rope_theta_old * yarn_correction_range_high
  - max_position_embeddings in saved config == the requested target
  - attention_factor, beta, yarn_alpha, yarn_correction_dim, yarn_correction_range_low,
    yarn_correction_range_high, yarn_mscale all populated in the saved config
  - Sanity check at end of script asserts max_position_embeddings == target_context

Failure modes caught explicitly:
  - Base model without RoPE theta: abort with clear error (not silent corruption)
  - Target <= base: ValueError
  - Missing datasets library: ImportError with install hint
  - No rotary tensors in checkpoint when --rescale-checkpoint is used: KeyError with
    diagnostic listing of actual checkpoint keys
  - Scientific notation handling: not applicable (Python floats) but a note against
    YAML-author confusion if that surface is touched

How to run (v3 phase):
  python training/yarn_rescaler.py Qwen3-Coder-14B-Instruct \
    --target-context 262144 --output-dir ./extended-256k-config

  # Then continue training from that config - do not use the original config.

How to run (v4 phase):
  python training/yarn_rescaler.py Qwen3-Coder-14B-Instruct \
    --target-context 1048576 --preset v4 --output-dir ./extended-1M-config

---

# 2. Dataset Registry

Purpose:
  Canonical source-of-truth for all training data entering Aurelius. The registry
  enforces: link-only on-prem storage (no hardcoded paths except small eval corpora),
  license gating, provenance requirements, and named mixes with validated weights.

Source:
  aurelius/data/dataset_registry.yaml

Companion loader:
  aurelius/data/streaming_loaders.py

Schema highlights that Claude must enforce:
  - Every entry has: id, type, url, license, size_tokens, task_families
  - Streaming is default; local_path is opt-in for small eval corpora
  - Provenance fields are mandatory when provenance_required=true
  - Mix weights normalize to 1.0 within 1e-6; script asserts this invariant
  - usage_type per mix: pretrain | sft | rlvr (controls eval-only dataset gating)

Mixes and their usage types:
  - continual_pretrain_phaseA -> pretrain
  - sft_phaseB_verified -> sft
  - rlvr_phaseC_code -> rlvr   (eval-only IDs allowed here: mbpp-train, humaneval-train)
  - rlvr_phaseC_math_reasoning -> rlvr
  - longcontext_adaptation -> sft

Token budget summary (cross-check before running):
  - Phase A (continual pretrain): 500B tokens
  - Phase B (SFT verified): 100B tokens
  - Phase C (RLVR code): 500B tokens
  - Phase C (RLVR math): 200B tokens
  - YaRN adaptation: 5B tokens
  - TOTAL: 1.305T tokens

What Claude must verify before starting a run:
  1. load the YAML, verify all referenced ids exist via load_registry()
  2. verify licenses match the deployment jurisdiction
  3. assert no eval-only dataset appears in a pretrain or sft mix
  4. assert sum(mix_weights) == 1.0 within 1e-6 (self-test does this)
  5. spot-check provenance fields if provenance_required=true for any dataset in the mix
  6. confirm cloud-only corpora (the-stack-v2-train-smol, codeparrot-github-code-seed)
     are streamed from HF and not downloaded locally

Invariants:
  - No local bulk storage of prohibited corpora (the-stack, codeparrot at full scale)
  - Every dataset referenced in a mix is reachable via its url or local_path
  - Provenance fields present in every row when provenance_required=true
  - Token budgets do not exceed 1.305T across all mixes combined

Running the self-test:
  cd ~/aurelius && python data/streaming_loaders.py

Output to verify:
  - "Loaded 15 dataset entries"
  - All five mixes have weights summing to 1.000000
  - "Mixes with errors: none"
  - Total token budget printed (~1.305T)

---

# 3. Streaming Loader

Purpose:
  Production-quality Python module to load HuggingFace and local JSONL datasets
  with streaming, provenance attachment, sample-fraction controls, license gating,
  and budget-aware batch iteration.

Source:
  aurelius/data/streaming_loaders.py

Key functions:
  - load_registry(path=None) -> DatasetRegistry  (reads dataset_registry.yaml)
  - DatasetRegistry.get(id) / resolve_mix(name) / assert_no_eval_only_in_mix(name)
  - stream_dataset(entry, sample_fraction, max_rows, provenance) -> Iterator[row]
  - iter_batches(mix_name, registry, batch_size, max_total_tokens, seed, provenance) -> Iterator[Batch]

What Claude must understand before using this in a training loop:
  - iter_batches() does token-budget-aware interleaving across datasets in the mix,
    weighted by mix_weights. It approximates token count via len(text) // 4.
  - provenance=True triggers attach_provenance() on every row; this adds schema
    fields and computes a deterministic source_hash.
  - Eval-only datasets are allowed in rlvr usage_type mixes only; pretrain/sft mixes
    abort assert_no_eval_only_in_mix() if they contain them.
  - _load_hf_streaming() and _load_hf_full() guard against missing HF library
    with ImportError + type: ignore comments for static analyzers
  - License gating blocks gpl-3.0 from the-stack unless gpl_block=False is explicitly
    set (not recommended without legal clearance).

Dependencies:
  - Python 3.8+
  - PyYAML (yaml.safe_load)
  - datasets (HuggingFace datasets library) for HuggingFace type entries
  - Optional: torch if checkpoint rescaling is used

---

# 4. Relationships Between Components

 Data source (HF Hub / local)                          registry.yaml
      |                                                        |
      v                                                        v
 DatasetRegistry + MixDefinition                  streaming_loaders.py
      |                                                        |
      +---------- iter_batches() ------------------------------+
                        |
                        v
                   training loops
               (continual_pretrain,
                sft_verified,
                rlvr_code,
                rlvr_math)
                        |
                        | immediately after continualPretrain
                        v
            yarn_rescaler.py  (256.2k for v3; 1M for v4)
                        |
                        v
            longcontext_adaptation SFT mix (5B tokens)
                        |
                        v
                   main phases resume

---

# 5. Claude Pre-Deployment Checklist (run in this order)

  [ ] 1. Read this INDEX file (you are reading it now).
  [ ] 2. Read dataset_registry.yaml — ensure all mixes reference known ids.
  [ ] 3. Read streaming_loaders.py — understand iter_batches() token accounting
           and provenance attachment.
  [ ] 4. Read yarn_params.yaml — pick the correct preset (v3_256k or v4_1M).
  [ ] 5. Read yarn_rescaler.py — understand the YaRN math and the checkpoint
           rescaling path. Verify the base model name is in QWEN_FAMILY_BASE_CONTEXT
           or plan to pass --base-context.
  [ ] 6. Read aurelius-amc-feasibility-2026-06-21.md — confirm AMC tier design
           and the relationship to context-window extension.
  [ ] 7. Run the streaming_loaders.py self-test:
           cd ~/aurelius && python data/streaming_loaders.py
       Expected: parse OK, all five mixes validated, no eval-only violations,
       total token budget ~1.305T. If this fails, fix registry before touching
       training compute.
  [ ] 8. Spot-check the first entry of each dataset task_family in registry.yaml
       against the actual dataset (open HF page or local file) to verify
       the task_family annotations are sensible.
  [ ] 9. For the chosen preset in yarn_params.yaml, verify the
       "required_long_context_dataset" exists in the registry and is in the
       longcontext_adaptation mix.
  [ ] 10. Confirm license gates match the deployment jurisdiction:
            - openwebmath: mixed; internal/research only unless cleared
            - the-stack-v2-train-smol: do not disable gpl-3.0 filter
            - codeparrot-github-code-seed: commercial clearance required
  [ ] 11. Confirm KV cache math for target context + hardware:
            - 256.2k / 4-bit / GQA-8 -> ~5.1 GB KV cache
            - 1M / 4-bit / GQA-8 -> ~20 GB KV cache (ring-attention required)
            - AMC target active working set: 8k-32k tokens -> 160-640 MB
  [ ] 12. Only after all checks pass, proceed to training. State which phase
           is starting and which mix is being consumed.

---

# 6. File Locations (absolute)

  ~/aurelius/data/dataset_registry.yaml
  ~/aurelius/data/streaming_loaders.py
  ~/aurelius/data/yarn_params.yaml
  ~/aurelius/training/yarn_rescaler.py
  ~/Desktop/AI Plans/aurelius-amc-feasibility-2026-06-21.md
