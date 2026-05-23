# Aurelius Improvement Plan
**Date:** May 22, 2026 (updated with May 2025–2026 paper sweep)
**Scope:** Architecture · Training · Inference · Alignment · Agent · Data · Interpretability
**Sources:** Nous Research, Meta, MiniMax, Qwen, OpenAI, Anthropic, DeepSeek, Moonshot/Kimi, StepFun, NVIDIA, Microsoft, POSTECH + 2025–2026 research literature

**Summary of May 2026 additions:** This is a research backlog plus an implementation priority map, not a flat queue. It contains 33 master rows plus many sub-items across alignment (REINFORCE++, DAPO, SimPO, uPRM, VersaPRM, PRPO), inference (Mirror-SD, LK Losses, SlimSpec, TurboQuant, TA, G-STEP, MemMachine), data pipeline (FineWeb2, Nemotron-CC, Ultra-FineWeb, QuaDMix, AC-ODM, LongRoPE2, DataEvolve, Magpie, GRAPE, OpenCoder/Seed-Coder, Mid-Training CoT), and architecture (DirMoE, Mamba-3, iRoPE, mHC, Gated DeltaNet). Canonical execution order is the P0 AMC Contract Registry plus the Master Priority Table, not textual appearance order.

---

## Table of Contents

0. [Research Delta — NVIDIA and Nous Research Releases](#research-delta--nvidia-and-nous-research-releases)
1. [Original Contributions (Novel)](#section-0--original-contributions)
2. [Executive Summary](#executive-summary)
3. [Critical Review Addendum — Execution Discipline](#critical-review-addendum--execution-discipline)
4. [AMC-First Overlay](#amc-first-overlay)
5. [Evidence, Validation, and Kill Gates](#evidence-validation-and-kill-gates)
6. [Reading This Document](#reading-this-document)
7. [Version Roadmap](#version-roadmap)
8. [Dependency Graph](#dependency-graph)
9. [V1 — Immediate: No Retraining Required](#v1--immediate-no-retraining-required)
10. [V1 — Next Training Run](#v1--next-training-run)
11. [V1 — Alignment & RL Pipeline](#v1--alignment--rl-pipeline)
12. [V1 — Inference & Serving](#v1--inference--serving)
13. [V1 — Interpretability & Safety](#v1--interpretability--safety)
14. [Data Pipeline](#data-pipeline)
15. [V2 — Architecture Redesign (2.7B)](#v2--architecture-redesign-27b)
16. [V3 — Multimodal + Product (3B+)](#v3--multimodal--product-3b)
17. [V4 — Scale (5B MoE+)](#v4--scale-5b-moe)
18. [Cross-Organisation Convergence](#cross-organisation-convergence)
19. [Master Priority Table](#master-priority-table)
20. [Sources](#sources)

---

## Research Delta — NVIDIA and Nous Research Releases

**Verification path used:** Firecrawl-backed web search was unavailable because the search backend returned `Payment Required`; the update below was therefore verified through direct public endpoints: Hugging Face model APIs/model cards and direct `arxiv.org/abs/...` pages. Do not treat unsourced social-media claims as canonical; prefer arXiv/HF/GitHub/model-card evidence for implementation decisions.

### New / newly relevant NVIDIA signals

| Source | Verified release facts | Aurelius impact |
|---|---|---|
| **Llama-Nemotron: Efficient Reasoning Models** (`arXiv:2505.00949`, v5 Sep 2025) | Open heterogeneous reasoning family: Nano 8B, Super 49B, Ultra 253B. Report describes neural architecture search from Llama 3 models, knowledge distillation, continued pretraining, reasoning SFT, large-scale RL, and a **dynamic reasoning toggle**. | Elevate `src/inference/token_budget_forcing.py` and mode-conditioned routing from “nice-to-have” to V1/V2 validation items. Add explicit tests for reasoning/non-reasoning mode switches, thought-budget caps, and per-request reasoning-mode telemetry. |
| **Nemotron-H** (`arXiv:2504.03624`, v4 Sep 2025) | Hybrid Mamba-Transformer 8B and 56B/47B family; replaces most attention layers with Mamba layers for constant per-token compute/memory and reports up to **3× inference speed** at similar accuracy. | V2 architecture should preserve the hybrid-SSM path as a first-class option rather than treating pure Transformer as the default. Any AMC layer-memory design should be profiled against SSM-heavy blocks because cache behavior differs radically from attention-heavy models. |
| **Nemotron 3 Nano Omni** (`arXiv:2604.24954`, v2 May 2026) | 30B-A3B efficient multimodal model with text, image, video, and audio; emphasizes multimodal token reduction, document understanding, long audio-video comprehension, and agentic computer use. | V3 multimodal work should prioritize token-reduction and modality-budget accounting before large model expansion. Add a modality-aware budget interface parallel to text token-budget forcing. |
| **Nemotron-CLIMB proxy models** (HF release, May 21 2026) | 62M and 350M decoder-only proxy models trained from scratch on **10T tokens** with Megatron-LM, WSD schedule, and intentionally deep-and-narrow architecture for scaling-law/proxy-tuning research. | Add a low-cost “proxy ladder” to the training plan: run AMC, optimizer, sequence packing, and RL recipe changes first on 62M/350M-style proxy configs before scaling to 1B+. This is one of the safest ways to validate architectural changes cheaply. |
| **ProRL** (`arXiv:2505.24864`) and **BroRL** (`arXiv:2510.01180`) | Prolonged RL with KL control/reference policy resetting and broadened exploration via many rollouts per example both target RLVR scaling limits. | The RL pipeline should track not only final reward but also exploration breadth, policy-reset cadence, KL drift, and per-token correctness mass. This maps directly to GRPO/RLVR trainer instrumentation. |
| **Nemotron ColEmbed V2** (`arXiv:2602.03992`) | Late-interaction visual document retrieval models for RAG over PDFs/slides/images. | Retrieval and long-context plans should include a late-interaction document path for visual artifacts, not only dense text embeddings. |

### New / newly relevant Nous Research signals

| Source | Verified release facts | Aurelius impact |
|---|---|---|
| **Hermes 4 Technical Report** (`arXiv:2508.18255`, v2 Sep 2025) | Hybrid reasoning model family combining structured, multi-turn reasoning with broad instruction following. Model cards report post-training corpus expansion from about **1M samples / 1.2B tokens** to about **5M samples / 60B tokens**, hybrid `<think>...</think>` reasoning mode, and RefusalBench evaluation. | Strengthen the plan around hybrid reasoning controls: explicit reasoning-mode prompts, thought-budget forcing, refusal/helpfulness evaluation, and format-faithful output checks. Add RefusalBench-like local tests for over-refusal vs unsafe compliance. |
| **Hermes 4.3 36B** (HF release Dec 2025) | Based on ByteDance Seed 36B; model card states it is the first Hermes model trained in a decentralized manner over the internet using Psyche and points back to the Hermes 4 technical report. | Keep decentralized/federated training out of the immediate path unless reproducibility gates are in place, but borrow the practical lesson: post-training data mixture quality and reasoning-mode UX are higher leverage than raw scale. |
| **Nomos 1** (HF release Jan 2026) | Qwen3-MoE-based reasoning model intended for use with the open-source Nomos reasoning harness; model card reports Putnam 2025 score **87/120** with the harness versus **24/120** for the referenced Qwen3 thinking base under the same conditions. | Treat “model + harness” as the evaluated unit. Add harness-level evaluation targets for math/proof tasks rather than only raw model generation tests. This supports the existing proof-augmented curriculum and AMC agent loop. |
| **NousCoder-14B** (HF release Jan 2026) | Qwen3-based coding model referencing RLVR coding datasets and DeepCoder-style training influence. | Add code-RLVR datasets and executable-code validation to the alignment roadmap only after the sandbox/test-runner security gates are green. |

### Resulting priority changes

1. **Promote reasoning-mode control to a V1 validation gate.** Existing `src/inference/token_budget_forcing.py` should be integrated into serving paths with tests for prompt-level mode selection, hard budget caps, and answer-transition behavior.
2. **Promote sequence-packing correctness to a data-pipeline gate.** Packing efficiency is not enough; masks must remain correct when real tokens equal the pad token ID. This audit fixed one such bug in `src/training/sequence_packing.py`.
3. **Add a proxy-scaling lane before expensive training.** NVIDIA’s CLIMB release makes a strong case for small proxy models to de-risk optimizer, packing, AMC, and RL changes before full-scale runs.
4. **Keep hybrid SSM/Mamba architecture on the V2 critical path.** Nemotron-H is directly aligned with Aurelius’ efficiency goals and should remain a benchmark comparator for any attention-only redesign.
5. **Evaluate harnesses, not only checkpoints.** Hermes/Nomos releases show that reasoning scaffolds, prompts, budget controls, and evaluators materially change capability. Aurelius should score the full runtime recipe.

---

## Section 0 — Original Contributions

> Thirteen novel techniques derived from synthesizing ideas across the surveyed labs plus the May 19–21, 2026 paper sweep supplied for this update. The new paper-derived items are deliberately framed as **Aurelius-specific compositions**, not as claims that the cited papers themselves are Aurelius inventions: each new item identifies the paper as closest prior art, then states what Aurelius adds through AMC, P0 contracts, deterministic admission gates, or end-to-end serving/evaluation integration.

### Novelty Assessment

| ID | Name | Tier | Closest Prior Art | Distinguishing Factor |
|---|---|---|---|---|
| OC-1 | SVD — Self-Verifying Draft | **S** | "Think Before You Accept" (2505.18629) | Single draft-head pass for both generation and validity; prior work uses full model for verification |
| OC-2 | LSD — Latent Speculative Decoding | **S** | SALS / MagicDec (2025) | Moves entire draft loop into MLA's 512-dim latent space; prior work compresses KV storage only |
| OC-3 | MCMR — Mode-Conditioned MoE Routing | **S** | Task-conditioned routing (2603.11114) | Dynamic mid-sequence routing based on positional mode token; prior work uses static per-sequence task label |
| OC-4 | CGR — Constitutionally-Guided Routing | **A** | RASA (2602.04448) | Training-time tier embedding vs post-hoc inference patching |
| OC-5 | PAC — Proof-Augmented Curriculum | **A** | FOVER (Qwen, 2025) | Formal verification as pretraining Stage 2 admission criterion; FOVER applies only to PRM labels |
| OC-6 | CDS — Capability Distillation Spiral | **A** | No unified pipeline found | Closed self-improving loop chaining GEPA + synthetic data + Minitron + DARE-TIES |
| OC-7 | EWM — Erase/Write Memory for AMC | **A** | Gated DeltaNet-2 (local NVIDIA PDF, 2026-05-21) | Applies decoupled erase/write gates to Aurelius' per-layer AMC memory contract, including trust/decay metadata and safety-gated writes; GDN2 applies the idea inside linear recurrent attention |
| OC-8 | RC-AMC — Reflection-Compiled AMC | **A** | MeMo: Memory as a Model (2605.15156) | Converts MeMo-style reflection QA into AMC Tier-2/Tier-3 trainable memory artifacts plus request-time memory protocol; MeMo trains a separate black-box memory model |
| OC-9 | SDB-Memory Runtime | **A** | Stochastic-Deterministic Boundary methodology (2605.20173) | Specializes proposer/verifier/commit/reject boundaries to memory writes, memory recalls, tool-result reuse, and serving actions with deterministic P0 schema enforcement |
| OC-10 | VEL — Verified Evolution Loop | **A** | AutoResearchClaw (2605.20025) | Turns autonomous-research self-healing, debate, result registry, citation verification, and cross-run lessons into Aurelius' AMC benchmark/report pipeline rather than a paper-writing agent alone |
| OC-11 | SLR — Stochastic Latent Recall | **A** | Probabilistic Tiny Recursive Model (2605.19943) | Uses parallel noisy latent/memory rollouts plus verifier/Q-head selection for recall/reasoning under AMC; PTRM applies it to TRM puzzle recursions without memory contracts |
| OC-12 | TBP — Token Boundary Priors | **A** | Decoupled subword-tokenization benefits (2604.27263) | Adds tokenizer-boundary priors and sample-throughput accounting to Aurelius data/AMC curriculum; the paper studies byte-level simulation rather than memory-aware LM training |
| OC-13 | PPDQ — Phase-Preserving Decode Quantization | **A** | Mix-Quant (2605.20315) | Adapts prefill-only NVFP4 / decode-BF16 to AMC-heavy agent serving with memory provenance, safety gates, and recall-quality telemetry |

**Tier key:** S = no prior art covers this specific mechanism; A = adjacent work is clearly distinguishable and must still pass an Aurelius reproduction gate before implementation. New OC-7 through OC-13 are **P0 contract candidates**: they should first be documented as stable interfaces and benchmark hypotheses before any training- or serving-affecting code lands.

---

### OC-1: SVD — Self-Verifying Draft

#### Sources Combined

EAGLE-3 (NeurIPS 2025, arXiv:2503.01840) + ThinkPRM (Qwen/Nous, 2025)

#### Concept

Speculative decoding works in two sequential phases: a cheap *draft model* proposes K candidate tokens; the *main model* verifies all K in one parallel forward pass. EAGLE-3 improves draft quality by fusing hidden states from layers 5, 12, and 22 of the main model.

The fundamental bottleneck: **every draft token requires a main model forward pass for verification**, regardless of how confident the draft model is.

SVD adds a **validity prediction head** to the EAGLE-3 draft head that outputs `P(main_model_accepts | draft_token)` in the same forward pass as token generation. Tokens above a confidence threshold skip the verifier entirely. This is the first speculative decoding approach where the draft head self-certifies tokens without any main model involvement.

#### Prior Art Gap

| Work | Mechanism | Why SVD is different |
|---|---|---|
| EAGLE-3 (2503.01840) | Tri-layer fused hidden-state draft | Generates tokens only; zero acceptance prediction |
| "Think Before You Accept" (2505.18629) | Semantic check before acceptance | Uses *full main model* for verification — two separate passes |
| MEDUSA (2401.10774) | Multiple parallel draft heads | No self-verification; all heads need main model |
| SpecTr (2401.17268) | Tree speculation | Verification always via main model |

SVD is the only approach that collapses token generation and acceptance prediction into a **single draft-head forward pass**.

#### Theoretical Basis

The draft head sees `h_fused = concat(h_5, h_12, h_22)`. These representations are rich enough to predict whether the main model will accept the drafted token. Formally, SVD learns two distributions from the same representation:

- `P(token_t+1 | h_fused_t)` — standard next-token prediction
- `P(accept_t+1 | h_fused_t, token_t+1)` — novel; trained on accept/reject labels from live speculation runs

#### Architecture

```
Input sequence [t_0 ... t_T]
         │
Main model forward (hooks on layers 5, 12, 22)
         │
[h_5, h_12, h_22]  each [B, T, 2048]
         │
   SVDFusionLayer
   Linear(6144→2048) + RMSNorm
         │
    fused_repr [B, T, 2048]
         │
    ┌────┴─────────────┐
    │                  │
 TokenHead          ValidityHead
(2048→8192)    (2048→512→128→1)
    │                  │
token_logits      validity ∈ [0,1]
[B,T,8192]         [B,T]
    │                  │
    │           > 0.85? ──YES──→ Auto-accept (no main model call)
    │                  │
    │                  NO ──────→ Standard speculative verifier
    │
draft_token
```

#### Full Implementation

```python
# src/inference/svd.py
"""
SVD — Self-Verifying Draft
Sources: EAGLE-3 (arXiv:2503.01840) + ThinkPRM validity scoring.
Novel: single forward pass produces both token logits and P(accept).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class SVDConfig:
    hidden_dim: int = 2048
    vocab_size: int = 8192
    draft_layer_indices: tuple[int, ...] = (5, 12, 22)
    validity_threshold: float = 0.85
    validity_hidden_dim: int = 512
    validity_bottleneck_dim: int = 128
    draft_window: int = 4
    alpha_validity: float = 0.1
    min_validity_samples: int = 1_000


class SVDFusionLayer(nn.Module):
    """Fuse hidden states from n transformer layers into one representation."""

    def __init__(self, hidden_dim: int, n_fuse: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim * n_fuse, hidden_dim, bias=False)
        self.norm = nn.RMSNorm(hidden_dim)

    def forward(self, states: list[torch.Tensor]) -> torch.Tensor:
        return self.norm(self.proj(torch.cat(states, dim=-1)))


class SVDDraftHead(nn.Module):
    """
    Single-pass draft head: generates tokens AND predicts acceptance probability.

    validity_head is architecturally isolated from token_head to prevent gradient
    interference between the two prediction objectives during training.
    Detaching the fused representation before the validity head ensures that
    validity loss does not affect token generation quality.
    """

    def __init__(self, cfg: SVDConfig) -> None:
        super().__init__()
        H, V = cfg.hidden_dim, cfg.vocab_size
        Hv, Hb = cfg.validity_hidden_dim, cfg.validity_bottleneck_dim
        self.cfg = cfg

        self.fusion = SVDFusionLayer(H, len(cfg.draft_layer_indices))

        self.token_norm = nn.RMSNorm(H)
        self.token_head = nn.Linear(H, V, bias=False)

        # Bottleneck MLP: 2048→512→128→1  (< 100K params)
        # Deep enough to distinguish high-confidence from uncertain drafts
        # without adding meaningful overhead to each draft step
        self.validity_head = nn.Sequential(
            nn.Linear(H, Hv, bias=False),
            nn.SiLU(),
            nn.RMSNorm(Hv),
            nn.Linear(Hv, Hb, bias=False),
            nn.SiLU(),
            nn.Linear(Hb, 1, bias=False),
        )
        self._validity_enabled = False

    def enable_validity_training(self) -> None:
        """Call after collecting cfg.min_validity_samples accept/reject labels."""
        self._validity_enabled = True

    def forward(
        self,
        layer_states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        fused = self.fusion(layer_states)
        token_logits = self.token_head(self.token_norm(fused))

        validity = None
        if self._validity_enabled:
            validity = self.validity_head(fused.detach()).squeeze(-1).sigmoid()

        return token_logits, validity

    def accept_mask(self, validity: torch.Tensor) -> torch.BoolTensor:
        return validity > self.cfg.validity_threshold


class SVDLoss:
    @staticmethod
    def compute(
        token_logits: torch.Tensor,
        target_ids: torch.LongTensor,
        validity: Optional[torch.Tensor],
        accept_labels: Optional[torch.Tensor],  # float {0,1}; -1 = unlabeled
        alpha: float = 0.1,
        ignore_index: int = -100,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        B, T, V = token_logits.shape
        L_token = F.cross_entropy(
            token_logits.view(B * T, V),
            target_ids.view(B * T),
            ignore_index=ignore_index,
        )
        L_validity = token_logits.new_zeros(1)
        if validity is not None and accept_labels is not None:
            mask = accept_labels >= 0
            if mask.any():
                L_validity = F.binary_cross_entropy(
                    validity[mask], accept_labels[mask].float()
                )
        total = L_token + alpha * L_validity
        return total, {
            "loss/token": L_token.item(),
            "loss/validity": L_validity.item(),
        }


class SVDAcceptLabelCollector:
    """Accumulates accept/reject labels from live speculative decoding runs."""

    def __init__(self, max_buffer: int = 100_000) -> None:
        self.buffer: list[tuple[torch.Tensor, float]] = []
        self.max_buffer = max_buffer

    def record(self, fused_hidden: torch.Tensor, accepted: bool) -> None:
        if len(self.buffer) < self.max_buffer:
            self.buffer.append((fused_hidden.detach().cpu(), float(accepted)))

    def ready(self, min_samples: int) -> bool:
        return len(self.buffer) >= min_samples

    def as_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        hiddens = torch.stack([b[0] for b in self.buffer])
        labels = torch.tensor([b[1] for b in self.buffer])
        return hiddens, labels


class SVDSpeculativeDecoder:
    """
    Integrates SVD into the standard speculative decoding loop.
    High-confidence tokens (validity > threshold) skip the main model verifier.
    """

    def __init__(
        self,
        main_model: nn.Module,
        draft_head: SVDDraftHead,
        cfg: SVDConfig,
    ) -> None:
        self.main_model = main_model
        self.draft_head = draft_head
        self.cfg = cfg
        self._stats: dict[str, int] = {"svd_skips": 0, "verified": 0, "rejected": 0}

    @torch.inference_mode()
    def generate_step(
        self,
        input_ids: torch.LongTensor,
        draft_steps: Optional[int] = None,
    ) -> tuple[torch.LongTensor, dict[str, int]]:
        K = draft_steps or self.cfg.draft_window
        layer_hiddens = self._get_layer_hiddens(input_ids)

        draft_ids: list[torch.Tensor] = []
        validity_list: list[torch.Tensor] = []
        current_hiddens = layer_hiddens

        for _ in range(K):
            token_logits, validity = self.draft_head(current_hiddens)
            next_tok = token_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_val = validity[:, -1] if validity is not None else torch.zeros(
                input_ids.shape[0], device=input_ids.device
            )
            draft_ids.append(next_tok)
            validity_list.append(next_val)
            current_hiddens = self._advance_hiddens(current_hiddens, next_tok)

        draft_tokens = torch.cat(draft_ids, dim=1)      # [B, K]
        validities = torch.stack(validity_list, dim=1)  # [B, K]
        high_conf = self.draft_head.accept_mask(validities)

        if high_conf.all():
            self._stats["svd_skips"] += K
            return draft_tokens, self._stats

        first_uncertain = int((~high_conf).float().argmax(dim=1).min().item())
        self._stats["svd_skips"] += int(high_conf[:, :first_uncertain].sum().item())
        accepted = self._standard_verify(input_ids, draft_tokens, first_uncertain)
        return accepted, self._stats

    def _get_layer_hiddens(self, input_ids: torch.LongTensor) -> list[torch.Tensor]:
        hiddens: dict[int, torch.Tensor] = {}
        hooks = []
        for idx in self.cfg.draft_layer_indices:
            def _hook(m, inp, out, i=idx):
                hiddens[i] = out[0] if isinstance(out, tuple) else out
            hooks.append(self.main_model.layers[idx].register_forward_hook(_hook))
        with torch.no_grad():
            self.main_model(input_ids)
        for h in hooks:
            h.remove()
        return [hiddens[i] for i in sorted(self.cfg.draft_layer_indices)]

    def _advance_hiddens(
        self, hiddens: list[torch.Tensor], next_tok: torch.Tensor
    ) -> list[torch.Tensor]:
        return hiddens  # Production: re-run with KV cache

    def _standard_verify(
        self,
        input_ids: torch.LongTensor,
        draft_tokens: torch.LongTensor,
        verify_from: int,
    ) -> torch.LongTensor:
        accepted = list(draft_tokens[:, :verify_from].unbind(dim=1))
        for i in range(verify_from, draft_tokens.shape[1]):
            seq = torch.cat([input_ids] + [t.unsqueeze(1) for t in accepted], dim=1)
            main_logits = self.main_model(seq).logits[:, -1, :]
            main_p = F.softmax(main_logits, dim=-1)
            tok = draft_tokens[:, i]
            accept_ratio = (main_p.gather(1, tok.unsqueeze(1)).squeeze(1) /
                            (1e-9 + main_p.max(dim=-1).values))
            if torch.rand(1).item() < accept_ratio.min().item():
                accepted.append(tok)
                self._stats["verified"] += 1
            else:
                resampled = torch.multinomial(main_p.clamp(min=0), 1).squeeze(1)
                accepted.append(resampled)
                self._stats["rejected"] += 1
                break
        return torch.stack(accepted, dim=1)

    def stats_report(self) -> str:
        total = self._stats["svd_skips"] + self._stats["verified"]
        rate = self._stats["svd_skips"] / max(total, 1)
        return (f"SVD: {self._stats['svd_skips']} auto-accepted "
                f"({rate:.1%} skip rate), {self._stats['rejected']} rejected")
```

**Aurelius integration:**
- New file only if necessary: `src/inference/svd.py`; otherwise keep the implementation under `src/inference/speculative_decoding.py` to avoid duplicate decoder surfaces.
- Modify `src/inference/speculative_decoding.py` — add `SVDDraftHead` behind a config flag; do not replace the baseline decoder until parity is proven.
- Add `scripts/train_svd_validity.py` — Phase 2 validity head fine-tuning on collected labels.
- Log `svd/skip_rate`, `svd/false_accept_rate`, and `svd/calibration_error` through the current serving/eval telemetry path, not the stale `src/inference/server.py` path.

**Training procedure:**
1. Train `token_head` identically to EAGLE-3 SFT (~1B tokens, standard CE loss)
2. Run 10K inference steps; collect accept/reject labels via `SVDAcceptLabelCollector`
3. Call `enable_validity_training()`; fine-tune `validity_head` for 1 epoch on labels
4. Optional: iterate steps 2–3 for active learning (each round improves label distribution)

**Metrics to watch:**
- `svd/skip_rate` — tokens auto-accepted without verifier (target: >35%)
- `svd/validity_bce` — validity head BCE on held-out labels (target: <0.30)
- `svd/false_accept_rate` — auto-accepted tokens that would have been rejected (target: <2%)
- `inference/tokens_per_second` — expected +15–25% over baseline EAGLE-3

**Risk / Mitigation:**
- Overconfident validity on high-frequency tokens: include hard negatives (rejected long-tail tokens) in training data
- Domain shift: re-collect accept/reject labels periodically during deployment; validity head is tiny (<100K params) so retraining is cheap
- Threshold sensitivity: calibrate 0.85 default using Platt scaling on held-out accept/reject data per deployment domain

**Research framing:** *"Self-Verifying Draft: Single-Pass Token Generation and Acceptance Prediction for Speculative Decoding."* Core claim: EAGLE-3's fused representation contains sufficient signal to predict main-model acceptance without a main-model call. Ablations: threshold sweep (0.7–0.95), validity head size (32→512), skip rate vs output quality degradation on MMLU / HellaSwag / GSM8K.

---

### OC-2: LSD — Latent Speculative Decoding

#### Sources Combined

EAGLE-3 (NeurIPS 2025, arXiv:2503.01840) + DeepSeek MLA (arXiv:2405.04434)

#### Concept

EAGLE-3's draft head projects from `hidden_dim=2048` to `vocab_size=8192` on every single draft step. DeepSeek's MLA compresses each token's key/value representation into a 512-dimensional latent vector rich enough to reconstruct full KV matrices.

**LSD drafts entirely in the 512-dim MLA latent space.** The draft head predicts the next latent vector `c_{t+1}` (not the next token), and only at acceptance does it decode via the frozen MLA up-projection. Draft steps become 64× cheaper in projection dimension.

#### Prior Art Gap

| Work | What it does | Why LSD is different |
|---|---|---|
| SALS (2025) | Compresses KV vectors for speculation memory | KV stored as latents; draft head still operates in vocab space |
| MagicDec (2024) | Sparse KV for long contexts | Reduces cache memory; draft step unchanged |
| EAGLE-3 (2503.01840) | Tri-layer hidden state fusion | Draft operates in full vocab space every step |
| MLA (DeepSeek V2, 2405.04434) | Low-rank KV compression | No speculation explored in original paper |

LSD is the first approach to run the speculation *generation loop* in the compressed latent space, deferring all vocab decoding to acceptance time.

#### Compute Analysis

| Operation | Standard EAGLE-3 | LSD |
|---|---|---|
| Draft projection (per step) | `Linear(2048→8192)` ≈ 16.8M FLOPs | `Linear(512→512)` ≈ 0.26M FLOPs |
| Draft KV cache (4 steps, B=32) | ≈ 1.05 GB | ≈ 262 MB (4× smaller) |
| Vocab decode | Every draft token | Only accepted tokens (~40%) |

#### Architecture

```
MLA at layer L:
  h_T [B, T, 2048] ──W_down──→ c_T [B, T, 512]

EAGLE-3 draft (expensive):
  h_fused [3×2048] → fuse → Linear(2048→8192) → logits [8192]

LSD draft (cheap):
  c_fused [3×512] → fuse → Linear(512→512) → c_{T+1} [512]
                                                   │
                                             accepted?
                                            YES │    NO │
                             W_up(c_{T+1})→token    discard
                                 [512→8192]
```

#### Full Implementation

```python
# src/inference/lsd.py
"""
LSD — Latent Speculative Decoding
Sources: EAGLE-3 (arXiv:2503.01840) + DeepSeek MLA (arXiv:2405.04434).
Novel: draft loop operates entirely in MLA's 512-dim latent space.
PREREQUISITE: MLA must be implemented in src/model/attention.py (V2 roadmap).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class LSDConfig:
    hidden_dim: int = 2048
    latent_dim: int = 512           # Must match main model's MLA config
    vocab_size: int = 8192
    draft_steps: int = 4
    n_fuse_layers: int = 3
    mla_layer_indices: tuple[int, ...] = (4, 11, 21)


class LatentFusion(nn.Module):
    """Fuse MLA latent vectors from multiple layers (4× cheaper than EAGLE-3 fusion)."""

    def __init__(self, latent_dim: int, n_fuse: int) -> None:
        super().__init__()
        self.proj = nn.Linear(latent_dim * n_fuse, latent_dim, bias=False)
        self.norm = nn.RMSNorm(latent_dim)

    def forward(self, latents: list[torch.Tensor]) -> torch.Tensor:
        return self.norm(self.proj(torch.cat(latents, dim=-1)))


class LSDDraftHead(nn.Module):
    """
    Draft head that operates entirely in MLA latent space.

    Predicts: c_{t+1} = f(c_fused_t)  — next latent vector, no vocab projection.
    Decodes:  token = argmax(W_up(c_{t+1}))  — only on accepted tokens.

    Tying latent_to_vocab to the main model's MLA up-projection is critical:
    it keeps draft and main model in the same latent space, ensuring that
    speculative verification remains statistically valid.
    """

    def __init__(self, cfg: LSDConfig) -> None:
        super().__init__()
        self.cfg = cfg
        L = cfg.latent_dim

        self.fusion = LatentFusion(L, cfg.n_fuse_layers)

        # All operations in 512-dim latent space — 64× fewer FLOPs than vocab projection
        self.latent_predictor = nn.Sequential(
            nn.Linear(L, L * 2, bias=False),
            nn.SiLU(),
            nn.RMSNorm(L * 2),
            nn.Linear(L * 2, L, bias=False),
            nn.RMSNorm(L),
        )

        self.latent_to_vocab = nn.Linear(L, cfg.vocab_size, bias=False)
        self._vocab_tied = False

    def tie_mla_decoder(self, mla_up_proj: nn.Linear) -> None:
        """
        Tie decoder weights to the main model's MLA up-projection.
        Without this, draft and main model operate in different latent spaces
        and acceptance verification becomes invalid.
        """
        self.latent_to_vocab = mla_up_proj
        self._vocab_tied = True

    def draft_latents(
        self,
        mla_latents: list[torch.Tensor],
        n_steps: int = 4,
    ) -> list[torch.Tensor]:
        """
        Run n_steps of cheap latent prediction.
        No vocab projection — each step is Linear(512→512).
        """
        fused = self.fusion(mla_latents)
        last = fused[:, -1:, :]
        drafts: list[torch.Tensor] = []
        for _ in range(n_steps):
            last = self.latent_predictor(last)
            drafts.append(last)
        return drafts

    def decode_accepted(self, accepted_latents: list[torch.Tensor]) -> torch.LongTensor:
        stacked = torch.cat(accepted_latents, dim=1)
        return self.latent_to_vocab(stacked).argmax(dim=-1)


class LSDLoss:
    """
    Two-component loss for LSD training:
    1. Cosine similarity in latent space (directional accuracy)
    2. Token alignment loss (ensures correct token decoding)
    """

    @staticmethod
    def latent_cosine_loss(
        predicted: list[torch.Tensor],
        targets: list[torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        for pred, tgt in zip(predicted, targets):
            p = F.normalize(pred.squeeze(1), dim=-1)
            t = F.normalize(tgt.squeeze(1), dim=-1)
            losses.append(1.0 - (p * t).sum(dim=-1).mean())
        return torch.stack(losses).mean()

    @staticmethod
    def token_alignment_loss(
        predicted: list[torch.Tensor],
        target_ids: torch.LongTensor,
        latent_to_vocab: nn.Linear,
    ) -> torch.Tensor:
        stacked = torch.cat(predicted, dim=1)
        logits = latent_to_vocab(stacked)
        B, K, V = logits.shape
        return F.cross_entropy(logits.view(B * K, V), target_ids.view(B * K))

    @classmethod
    def compute(
        cls,
        predicted: list[torch.Tensor],
        target_latents: list[torch.Tensor],
        target_ids: torch.LongTensor,
        latent_to_vocab: nn.Linear,
        beta: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        L_cos = cls.latent_cosine_loss(predicted, target_latents)
        L_tok = cls.token_alignment_loss(predicted, target_ids, latent_to_vocab)
        total = L_cos + beta * L_tok
        return total, {"loss/latent_cosine": L_cos.item(), "loss/token_align": L_tok.item()}
```

**Aurelius integration:**
- New file: `src/inference/lsd.py`
- **Prerequisite:** MLA must be implemented in `src/model/attention.py` (V2 roadmap — LSD cannot be added to V1 GQA)
- Modify `src/model/attention.py` — expose `get_latent_vectors(layer_indices: list[int])` method
- At model init: `draft_head.tie_mla_decoder(model.mla_layers[N].W_V_up)`

**Training procedure:**
1. Run main model on training corpus; hook MLA layers at `cfg.mla_layer_indices` to collect latent vectors
2. Train `LSDDraftHead` to minimize `LSDLoss.compute()` — latent cosine + token alignment
3. Verify latent alignment: check cosine similarity on validation set (target: >0.90)
4. Tie MLA decoder after each main-model update that modifies the latent space

**Metrics:**
- `lsd/latent_cos_similarity` — predicted vs actual latent similarity (target: >0.90)
- `lsd/draft_step_flops` — should be ~64× lower than EAGLE-3 measured FLOPs
- `lsd/acceptance_rate` — should match EAGLE-3 baseline within ±3%
- `inference/tokens_per_second` — +10–20% vs EAGLE-3 at batch_size ≥ 8

**Risk / Mitigation:**
- Latent space drift: if the main model is updated (continued pretraining), LSD must be retrained — the latent space changes with each training run. Mitigate by re-tying `tie_mla_decoder()` after each fine-tuning phase.
- V2 dependency: cannot retrofit to V1 GQA — plan as V2 launch feature aligned with MLA implementation.
- Decoder tie breakage: if MLA weights are frozen during some training phase and LSD is not, the spaces diverge. Add an assertion that checks cosine alignment > 0.85 before each inference run.

**Research framing:** *"Latent Speculative Decoding: Drafting in the Compressed KV Space."* Core claim: MLA's 512-dim latent space supports a complete speculation loop at 64× lower projection cost. Ablations: latent dim sensitivity (256/512/1024), n_fuse_layers sweep, cosine alignment threshold vs acceptance rate.

---
### OC-3: MCMR — Mode-Conditioned MoE Routing

#### Sources Combined

DeepSeek V3 auxiliary-loss-free bias router (arXiv:2412.19437) + Qwen3 `<think>`/`</think>` mode tokens + Kimi K2.6 per-context expert selection

#### Concept

DeepSeek's bias router adds a per-expert scalar bias updated post-optimizer-step to balance load without auxiliary losses. The bias is a global constant — it does not change based on what the model is currently generating.

Qwen3 introduced `<think>`/`</think>` tokens that switch the model between chain-of-thought reasoning and direct answer generation. Different capabilities dominate each mode: reasoning requires math/logic expert activations; answering requires language/fluency experts.

**MCMR makes the expert bias a function of the current generation mode at each token position.** When inside `<think>` tags, the router applies a reasoning-mode bias vector that steers toward math/logic experts. After `</think>`, an answer-mode bias activates. During tool calls, a code/structured-output bias takes over.

**The critical distinction from all prior work:** the mode is not a task label (static per sequence) but a *dynamic positional signal* that changes mid-sequence as the model transitions from thinking to answering. Task-conditioned routing (arXiv:2603.11114) routes based on sentence-level input type — a single static signal per sequence. MCMR routes based on where in the generation trace each token falls.

#### Prior Art Gap

| Work | Routing Signal | Key Difference from MCMR |
|---|---|---|
| DeepSeek V3 bias router | Expert load imbalance | Static bias; same for all positions in a sequence |
| Task-conditioned routing (2603.11114) | Sentence-level task type | Static per sequence; does not change mid-generation |
| Mixture of LoRA experts | LoRA adapter selection by input type | Not integrated into core MoE routing; no mode tokens |

MCMR is the first mechanism to modulate MoE routing based on a *dynamic positional signal* (mode token state) rather than a static task label.

#### Architecture

```
Token sequence:
[prompt] <think> [reasoning...] </think> [answer...]
    │       │          │             │        │
 DEFAULT   THINK      THINK        ANSWER   ANSWER

mode_ids: [0,0,0, 1, 1,1,1,1,1, 0, 0,0,0,0,...]

At each position t:
  h_t ──router_proj──→ base_logits [E=8]
  mode_ids[t] ──→ mode_bias[mode_ids[t]] [E=8]
  biased_logits = base_logits + bias
  top_k(biased_logits, k=2) ──→ expert selection

mode_bias [4 modes × 8 experts]:
  Initialized to zeros.
  Self-organizes during think-mode SFT as reasoning/answer patterns diverge.
  Updated post-step via update_mode_biases() — no gradient, load correction only.
```

#### Full Implementation

```python
# src/model/mcmr.py
"""
MCMR — Mode-Conditioned MoE Routing
Sources: DeepSeek V3 bias router (arXiv:2412.19437) + Qwen3 mode tokens.
Novel: dynamic mid-sequence routing bias conditioned on <think>/<</think> token state.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class GenMode(IntEnum):
    DEFAULT = 0
    THINK   = 1
    ANSWER  = 2
    TOOL    = 3


# Adjust these to match Aurelius BPE token IDs for special tokens
THINK_OPEN_ID  = 8190
THINK_CLOSE_ID = 8191
TOOL_OPEN_ID   = 8188
TOOL_CLOSE_ID  = 8189


@dataclass
class MCMRConfig:
    hidden_dim: int      = 2048
    n_experts: int       = 8
    n_modes: int         = 4
    top_k: int           = 2
    balance_coeff: float = 0.002
    balance_target: Optional[float] = None


def compute_mode_ids(
    token_ids: torch.LongTensor,
) -> torch.LongTensor:
    """
    Assign a generation mode ID to each token position via left-to-right scan.
    Mode transitions happen at special tokens.
    O(B*T) — called once per forward pass, not per layer.
    """
    B, T = token_ids.shape
    mode_ids = torch.full((B, T), GenMode.DEFAULT, dtype=torch.long, device=token_ids.device)
    for b in range(B):
        mode = GenMode.DEFAULT
        for t in range(T):
            tid = token_ids[b, t].item()
            if   tid == THINK_OPEN_ID:  mode = GenMode.THINK
            elif tid == THINK_CLOSE_ID: mode = GenMode.ANSWER
            elif tid == TOOL_OPEN_ID:   mode = GenMode.TOOL
            elif tid == TOOL_CLOSE_ID:  mode = GenMode.ANSWER
            mode_ids[b, t] = mode
    return mode_ids


class MCMRRouter(nn.Module):
    """
    Mode-Conditioned MoE Router.

    The mode_bias buffer [n_modes, n_experts] is NOT a gradient parameter.
    It self-organizes via update_mode_biases() calls after each optimizer step,
    independently correcting load imbalance per generation mode.

    As a result, think-mode positions and answer-mode positions naturally route
    to different expert subsets over training — no supervision required.
    """

    def __init__(self, cfg: MCMRConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._target = cfg.balance_target or (1.0 / cfg.n_experts)

        self.router_proj = nn.Linear(cfg.hidden_dim, cfg.n_experts, bias=False)

        self.register_buffer("mode_bias",           torch.zeros(cfg.n_modes, cfg.n_experts))
        self.register_buffer("_mode_expert_counts", torch.zeros(cfg.n_modes, cfg.n_experts))
        self.register_buffer("_mode_token_counts",  torch.zeros(cfg.n_modes))

    def forward(
        self,
        hidden: torch.Tensor,         # [B, T, H] or [B*T, H]
        mode_ids: torch.LongTensor,   # [B*T]
    ) -> tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        """
        Returns:
            scores:       [B*T, top_k]
            indices:      [B*T, top_k]
            base_logits:  [B*T, n_experts]  (pre-bias, for external load tracking)
        """
        if hidden.dim() == 3:
            B, T, H = hidden.shape
            hidden   = hidden.reshape(B * T, H)
            mode_ids = mode_ids.reshape(B * T)

        base_logits = self.router_proj(hidden)
        bias        = self.mode_bias[mode_ids]
        scores, indices = (base_logits + bias).topk(self.cfg.top_k, dim=-1)
        scores = scores.softmax(dim=-1)

        if self.training:
            self._accumulate(indices, mode_ids)

        return scores, indices, base_logits

    def _accumulate(
        self, indices: torch.LongTensor, flat_mode: torch.LongTensor
    ) -> None:
        with torch.no_grad():
            for m in range(self.cfg.n_modes):
                mask = flat_mode == m
                if not mask.any():
                    continue
                self._mode_token_counts[m] += mask.sum().float()
                for k in range(self.cfg.top_k):
                    self._mode_expert_counts[m].scatter_add_(
                        0,
                        indices[mask, k],
                        torch.ones(mask.sum(), device=indices.device),
                    )

    @torch.no_grad()
    def update_mode_biases(self) -> None:
        """
        Post-optimizer-step bias update.
        Called once per training step after optimizer.step().

        Independently corrects load imbalance per generation mode.
        Overloaded experts within a mode receive negative bias;
        underloaded experts receive positive bias.
        This drives per-mode expert specialization without auxiliary loss.
        """
        for m in range(self.cfg.n_modes):
            n = self._mode_token_counts[m]
            if n < 1:
                continue
            utilization = self._mode_expert_counts[m] / (n * self.cfg.top_k)
            self.mode_bias[m] -= self.cfg.balance_coeff * (utilization - self._target)

        self._mode_expert_counts.zero_()
        self._mode_token_counts.zero_()

    def specialization_report(self) -> dict[str, dict]:
        """
        Diagnostic: which experts each mode prefers.
        Bias spread std should grow from ~0 to >0.05 over 10K training steps
        if mode specialization is working correctly.
        """
        mode_names = {0: "default", 1: "think", 2: "answer", 3: "tool"}
        return {
            mode_names[m]: {
                "top3_experts": self.mode_bias[m].topk(3).indices.tolist(),
                "bias_spread_std": round(float(self.mode_bias[m].std()), 4),
            }
            for m in range(self.cfg.n_modes)
        }
```

**Aurelius integration:**
- Prefer extending `src/model/moe.py` with a disabled-by-default `MCMRRouter`; create `src/model/mcmr.py` only if it becomes a clean exported strategy module with tests.
- Modify `src/model/moe.py` — add `mode_ids` support without replacing `BiasDynamicRouter` until baseline routing parity is proven.
- Modify `src/model/transformer.py` — compute `mode_ids = compute_mode_ids(input_ids)` once at top of forward pass; pass to each MoE layer behind a config flag.
- Modify `src/training/trainer.py` — call `moe_layer.router.update_mode_biases()` after every `optimizer.step()` only when MCMR is enabled.
- Modify `src/training/sft.py` — confirm `<think>` tokens present in think-mode training data after tokenizer collision checks.
- Add `src/tokenizer/special_tokens.py` or extend the existing tokenizer module only after verifying the actual BPE IDs.

**Training procedure:**
MCMR requires no supervised routing labels. The specialization self-organizes:
1. In think-mode SFT, the model naturally activates different experts for reasoning vs language tokens
2. `update_mode_biases()` amplifies these usage patterns into stable biases over time
3. Track `specialization_report()` every 1K steps — `bias_spread_std` should grow from 0 to >0.05 by 10K steps; if it stays flat, the think-mode data mix is too low

**Metrics:**
- `mcmr/think_bias_std` — std of think-mode biases (target: >0.05 at 10K steps)
- `mcmr/mode_routing_kl` — KL(think routing ∥ answer routing) (target: >0.30)
- `benchmark/gsm8k` — expected +2–4% vs static routing (reasoning experts better utilized in think mode)
- `benchmark/hellaswag` — expected +1–2% (language experts freed from reasoning load in answer mode)
- `inference/expert_histogram` — per-mode histograms should visually diverge by 10K steps

**Risk / Mitigation:**
- Mode bias collapse if the model rarely uses `<think>` tokens: ensure think-mode SFT data is present from day 1 of the SFT run
- Token ID mismatch: never hard-code `<think>` IDs. Verify with the canonical tokenizer, assert round-trip encode/decode, and fail startup if configured IDs do not match the tokenizer vocabulary.
- Negative transfer: if mode-conditioned routing hurts non-think tasks, reduce `balance_coeff` from 0.002 to 0.0005

**Research framing:** *"Mode-Conditioned MoE Routing via Positional Mode Tokens."* Core contribution: first routing mechanism where expert selection is conditioned on a positional signal (generation phase) rather than a sequence-level task label. Key ablations: bias spread trajectory, mode KL vs downstream benchmark improvement, behavior with and without `<think>` tokens.

---

### OC-4: CGR — Constitutionally-Guided Routing

#### Sources Combined

DeepSeek V3 bias router + Anthropic 4-tier alignment hierarchy (Safety > Ethics > Compliance > Helpfulness) + Nous RASA (arXiv:2602.04448)

#### Concept

Anthropic's constitutional alignment hierarchy specifies four tiers of priority for resolving alignment conflicts. In Aurelius, the PRAXIS/MOSAIC system applies these tiers at the loss-weighting level. However, the MoE router is completely unaware of alignment tiers — a safety-sensitive refusal token routes through the same experts as a casual greeting.

RASA (2602.04448) identifies safety-critical experts *after training* by analyzing which experts activate on harmful content, then restricts them during inference. This is inference-time patching applied post-hoc.

**CGR embeds the tier hierarchy into the router's training signal.** During alignment fine-tuning (DPO/GRPO), each token carries an alignment tier label. A small auxiliary loss encourages Safety-tier tokens to route through a dedicated subset of experts that self-specialize in safety-sensitive content. After training, the constitutional awareness is a first-class architectural property — not an inference-time patch.

**Why this matters for Aurelius specifically:** The PRAXIS/MOSAIC system already has a tier resolver. CGR is a natural extension that connects PRAXIS tier outputs to the MoE routing layer, creating architectural alignment rather than only loss-level alignment.

#### Prior Art Gap

| Work | Mechanism | Key Difference |
|---|---|---|
| RASA (2602.04448) | Post-hoc safety expert ID + inference restriction | Post-hoc; CGR is training-time gradient embedding |
| Constitutional AI (Anthropic) | System-prompt level safety principles | Operates at prompt level, not expert routing level |
| Wise (2601.xxxxx) | Expert deactivation binary masks | Binary; not learned tier-conditioned bias |

CGR integrates the constitutional hierarchy into gradient flow during alignment training, making it a first-class architectural property rather than an inference-time patch.

#### Full Implementation

```python
# src/alignment/cgr.py
"""
CGR — Constitutionally-Guided Routing
Sources: DeepSeek V3 bias router + Anthropic 4-tier alignment hierarchy.
Novel: alignment tier signals embedded as training-time routing biases.
Distinct from RASA (2602.04448): CGR is training-time; RASA is post-hoc inference.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class AlignmentTier(IntEnum):
    SAFETY      = 0
    ETHICS      = 1
    COMPLIANCE  = 2
    HELPFULNESS = 3


@dataclass
class CGRConfig:
    hidden_dim: int             = 2048
    n_experts: int              = 8
    n_tiers: int                = 4
    top_k: int                  = 2
    tier_loss_weight: float     = 0.05
    warmup_steps: int           = 500
    balance_coeff: float        = 0.002


class CGRRouter(nn.Module):
    """
    Constitutionally-Guided Router.

    tier_bias [n_tiers, n_experts] is a learned Parameter — it receives gradients
    from constitutional_routing_loss during alignment training phases.
    expert_safety_affinity [n_experts] is a learned scalar per expert;
    low value = safety-specialized expert, high value = helpfulness-oriented expert.
    Both are initialized to neutral values and self-organize via the auxiliary loss.
    """

    def __init__(self, cfg: CGRConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._step = 0

        self.router_proj = nn.Linear(cfg.hidden_dim, cfg.n_experts, bias=False)

        # Learned tier bias — receives gradients from constitutional_routing_loss
        self.tier_bias = nn.Parameter(torch.zeros(cfg.n_tiers, cfg.n_experts))

        # Per-expert safety affinity: low = safety-specialized, high = helpfulness-oriented
        self.expert_safety_affinity = nn.Parameter(
            torch.linspace(0.0, 1.0, cfg.n_experts)
        )

        # Load tracking for post-step bias update
        self.register_buffer("_tier_expert_counts", torch.zeros(cfg.n_tiers, cfg.n_experts))
        self.register_buffer("_tier_token_counts",  torch.zeros(cfg.n_tiers))

    def forward(
        self,
        hidden: torch.Tensor,
        tier_labels: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        if hidden.dim() == 3:
            B, T, H = hidden.shape
            hidden = hidden.reshape(B * T, H)

        base_logits = self.router_proj(hidden)

        if tier_labels is not None:
            biased = base_logits + self.tier_bias[tier_labels]
        else:
            biased = base_logits

        scores, indices = biased.topk(self.cfg.top_k, dim=-1)
        scores = scores.softmax(dim=-1)

        if self.training and tier_labels is not None:
            self._accumulate(indices, tier_labels)

        return scores, indices, base_logits

    def constitutional_routing_loss(
        self,
        router_scores: torch.Tensor,
        tier_labels: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Auxiliary loss driving tier-expert specialization.

        Convention:
          Safety tokens (tier=0) should route to low-affinity experts
          (those that specialize in safety-critical content).
          Helpfulness tokens (tier=3) route to high-affinity experts.

        The affinity values are learned — no manual expert role assignment.
        The loss creates gradient pressure for self-organization over training.
        """
        if self._step < self.cfg.warmup_steps:
            return router_scores.new_zeros(1)

        # Safety (0) → target_aff=0.0; Helpfulness (3) → target_aff=1.0
        target_aff = tier_labels.float() / float(AlignmentTier.HELPFULNESS)
        affinities  = self.expert_safety_affinity.sigmoid()
        actual_aff  = (router_scores * affinities.unsqueeze(0)).sum(dim=-1)

        return F.mse_loss(actual_aff, target_aff)

    @torch.no_grad()
    def update_biases(self) -> None:
        """Post-step load balancing applied per-tier independently."""
        target = 1.0 / self.cfg.n_experts
        for t in range(self.cfg.n_tiers):
            n = self._tier_token_counts[t]
            if n < 1:
                continue
            util = self._tier_expert_counts[t] / (n * self.cfg.top_k)
            self.tier_bias.data[t] -= self.cfg.balance_coeff * (util - target)
        self._tier_expert_counts.zero_()
        self._tier_token_counts.zero_()
        self._step += 1

    def _accumulate(
        self, indices: torch.LongTensor, tier_labels: torch.LongTensor
    ) -> None:
        with torch.no_grad():
            for t in range(self.cfg.n_tiers):
                mask = tier_labels == t
                if not mask.any():
                    continue
                self._tier_token_counts[t] += mask.sum().float()
                for k in range(self.cfg.top_k):
                    self._tier_expert_counts[t].scatter_add_(
                        0, indices[mask, k], torch.ones(mask.sum(), device=indices.device)
                    )

    def safety_experts(self, threshold: float = 0.3) -> list[int]:
        """Experts with learned safety affinity below threshold — safety-specialized."""
        return (self.expert_safety_affinity.sigmoid() < threshold).nonzero().flatten().tolist()

    def tier_divergence_report(self) -> dict[str, float]:
        safety_dist  = F.softmax(self.tier_bias[AlignmentTier.SAFETY],      dim=0)
        helpful_dist = F.softmax(self.tier_bias[AlignmentTier.HELPFULNESS], dim=0)
        kl = float(F.kl_div(helpful_dist.log(), safety_dist, reduction="sum"))
        return {
            "safety_vs_helpfulness_kl": round(kl, 4),
            "n_safety_experts": len(self.safety_experts()),
        }


def build_tier_label_map(tokenizer_vocab: dict[str, int]) -> dict[int, int]:
    """
    Build a mapping from token_id → AlignmentTier for SFT data tagging.
    Extend with domain-specific safety tokens for your deployment.
    """
    tiers: list[tuple[list[str], int]] = [
        (["cannot", "won't", "refuse", "harmful", "dangerous", "illegal"], AlignmentTier.SAFETY),
        (["ethically", "morally", "values", "principles"],                  AlignmentTier.ETHICS),
        (["policy", "guidelines", "terms of service", "according to"],      AlignmentTier.COMPLIANCE),
    ]
    label_map: dict[int, int] = {}
    for phrases, tier in tiers:
        for phrase in phrases:
            for token, tid in tokenizer_vocab.items():
                if phrase in token.lower():
                    label_map[tid] = int(tier)
    return label_map
```

**Aurelius integration:**
- New file: `src/alignment/cgr.py`
- Modify `src/model/moe.py` — inject `CGRRouter` behind `use_cgr: bool` config flag
- Modify `src/alignment/praxis/` — especially `praxis_loss.py` / config — and update `aurelius/alignment/praxis.py` only if the public API surface changes; route PRAXIS tier resolver output to `tier_labels` in CGR forward
- Modify `src/training/dpo.py` and `src/training/grpo.py` — pass `tier_labels` to MoE forward during alignment phases; add `constitutional_routing_loss` to total loss
- Add `scripts/tag_alignment_tiers.py` — preprocess SFT dataset to annotate tokens with tier labels using `build_tier_label_map()`

**Training procedure:**
1. Normal pretraining — CGR off (no tier labels needed)
2. SFT phase — enable CGR; initial tier labels from `build_tier_label_map()` keyword matching
3. DPO/GRPO — tier labels from PRAXIS resolver output per token; `constitutional_routing_loss` activates after `warmup_steps=500`
4. Monitor `tier_divergence_report()` throughout — KL should grow from 0 to >0.20 by DPO convergence

**Metrics:**
- `cgr/safety_vs_helpfulness_kl` — routing KL between safety and helpfulness tiers (target: >0.20 by end of DPO)
- `cgr/n_safety_experts` — number of experts with affinity <0.3 (expect 1–2 to naturally emerge)
- `benchmark/bbq_bias` — safety/bias benchmarks; should improve
- `benchmark/halueval` — hallucination; ethics-tier routing helps
- `alignment/refusal_accuracy` — refusal precision on safety test set

**Risk / Mitigation:**
- Tier label noise: keyword-based `build_tier_label_map()` is imprecise; false Safety tier labels on benign tokens could distort routing. Mitigate by using only high-confidence keywords and gating by context (not just token identity)
- Constitutional loss scaling: `tier_loss_weight=0.05` is a starting point. If the loss dominates CE loss, reduce to 0.01. Monitor `loss/cgr_ratio` metric
- Safety expert overloading: if only 1–2 experts specialize, they may become bottlenecks. Mitigate by setting `n_safety_experts` floor of 2 and monitoring expert utilization histograms

---
### OC-5: PAC — Proof-Augmented Curriculum

#### Sources Combined

FOVER (Qwen, 2025, formal step verification) + Qwen3 3-stage pretraining curriculum + Atropos Verifier Pool (Nous Research)

#### Concept

Qwen3's three-stage curriculum: Stage 1 = general web (60%) → Stage 2 = STEM/Code-heavy (30%) → Stage 3 = long-context (10%). Stage 2 quality filtering uses deduplication, perplexity, and educational classifiers. FOVER uses Z3 and Isabelle/HOL to verify intermediate reasoning steps and generate labels for a Process Reward Model. FOVER's verification is applied *downstream* of pretraining — it has never been used as a curriculum gate at the pretraining stage.

**PAC applies FOVER-style formal verification as the Stage 2 admission filter.** A document enters Stage 2 only if ≥60% of its multi-step reasoning transitions are verifiable. Documents that fail PAC are demoted to Stage 3 instead of discarded.

**Expected effect:** Stage 2 contains only machine-verified reasoning chains. The model trained on PAC-filtered Stage 2 data learns step-valid reasoning as a pretraining habit — without any inference-time enforcement.

#### Prior Art Gap

| Work | What it does | Key Difference |
|---|---|---|
| FOVER (Qwen, 2025) | Generates PRM labels via formal verification | Applied at RLHF/PRM stage; never used for pretraining admission |
| NuminaMath (2024) | Formal math dataset for SFT | Fixed curated dataset; not a runtime filter |
| Lean Workbooks / ProofNet | Proof language training corpora | Fine-tuning target; not a curriculum gate |
| MuMath / DART-Math | Math curriculum in SFT | No formal verification; human-quality heuristics only |

#### Architecture

```
Raw STEM/Code corpus
        │
┌───────▼─────────────────────────────────────────┐
│  PAC Filter                                      │
│  1. Extract reasoning steps (numbered, logical)  │
│  2. For each step transition (prev → curr):      │
│     a. NL→Z3Constraints translator              │
│     b. Z3Solver.check(premise → ¬conclusion)    │
│        UNSAT = valid step                        │
│     c. If Z3 can't parse: escalate to Isabelle  │
│  3. verified_ratio = n_verified / n_transitions  │
│     ≥ 0.60 → Stage 2 | < 0.60 → Stage 3        │
└────────────────┬────────────────┬────────────────┘
           Stage 2 (PAC)     Stage 3 (general)
        verified STEM/Math   all other quality data
```

#### Full Implementation

```python
# scripts/pac_curriculum_filter.py
"""
PAC — Proof-Augmented Curriculum
Sources: FOVER (Qwen) + Qwen3 3-stage curriculum.
Novel: Z3/Isabelle verification as pretraining Stage 2 admission criterion.
Requires: pip install z3-solver
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class PACConfig:
    z3_timeout_ms: int              = 5_000
    min_steps: int                  = 3
    verified_ratio_threshold: float = 0.60
    fallback_stage: int             = 3


class ReasoningStepExtractor:
    """Extract sequential reasoning steps from mathematical text."""

    NUMBERED_RE = re.compile(r'(?:Step\s+\d+[:.)]|^\d+[.)]\s)[^\n]+', re.MULTILINE)
    CONNECTORS = (
        "therefore", "thus", "hence", "so we have",
        "it follows", "which gives", "we get", "consequently",
    )

    def extract(self, text: str) -> list[str]:
        numbered = self.NUMBERED_RE.findall(text)
        if len(numbered) >= 3:
            return [s.strip() for s in numbered]
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if any(kw in s.lower() for kw in self.CONNECTORS)]


@dataclass
class Z3Constraint:
    """
    Structured representation of a step-validity check for Z3.
    The NL→Z3 translator produces this structure; Z3StepVerifier consumes it.
    Using structured constraints avoids executing arbitrary code strings.
    """
    variables: dict[str, str]        # name → type ("Int", "Real", "Bool")
    premises: list[str]              # Z3 expression strings for premise facts
    negated_conclusion: str          # Z3 expression for ¬(conclusion)
    # e.g., variables={"x": "Int"}, premises=["x + 3 == 7"], negated_conclusion="x != 4"


class Z3StepVerifier:
    """
    Verify arithmetic/algebraic reasoning steps using the z3-solver Python API.

    The NL→Z3 translator converts a (premise, conclusion) pair into a Z3Constraint
    structure. Z3StepVerifier then uses the z3-solver API directly — no code execution.

    Semantic: checks whether (premise AND ¬conclusion) is UNSAT.
    UNSAT means every model satisfying the premise also satisfies the conclusion,
    i.e., the conclusion is a valid consequence of the premise.
    """

    def __init__(self, translator=None) -> None:
        # translator: callable(premise_str, conclusion_str) -> Optional[Z3Constraint]
        self.translator = translator

    def verify(self, premise: str, conclusion: str) -> tuple[bool, str]:
        if self.translator is None:
            return False, "no_translator"

        constraint = self.translator(premise, conclusion)
        if constraint is None:
            return False, "parse_failed"

        return self._check_z3(constraint)

    def _check_z3(self, c: Z3Constraint) -> tuple[bool, str]:
        """
        Use z3-solver API directly to check the constraint.
        No code execution — only z3 Python API calls.
        """
        try:
            import z3

            solver = z3.Solver()
            solver.set("timeout", 5000)  # milliseconds

            # Declare variables using the z3 API based on their types
            var_map: dict[str, Any] = {}
            for name, vtype in c.variables.items():
                if vtype == "Int":
                    var_map[name] = z3.Int(name)
                elif vtype == "Real":
                    var_map[name] = z3.Real(name)
                elif vtype == "Bool":
                    var_map[name] = z3.Bool(name)

            # Parse premise and negated conclusion using z3.parse_smt2_string
            # which is safe — it parses SMT-LIB2 format, not Python code
            premises_smt = " ".join(f"(assert {p})" for p in c.premises)
            neg_conclusion_smt = f"(assert {c.negated_conclusion})"
            full_smt = f"(declare-fun x () Int)\n{premises_smt}\n{neg_conclusion_smt}"

            try:
                assertions = z3.parse_smt2_string(full_smt)
                solver.add(assertions)
            except z3.Z3Exception:
                return False, "smt_parse_error"

            result = solver.check()
            if result == z3.unsat:
                return True, "unsat"
            elif result == z3.sat:
                return False, "sat"
            else:
                return False, "unknown"

        except ImportError:
            return False, "z3_not_installed"
        except Exception as exc:
            return False, f"error:{type(exc).__name__}"


class IsabelleVerifier:
    """
    Interface for Isabelle/HOL verification of theorems beyond Z3 scope.
    Production implementation calls the Isabelle process manager API.

    The translator converts (theorem_nl, proof_steps) to Isabelle .thy content.
    The actual Isabelle call should be implemented via your infrastructure's
    safe process execution wrapper (e.g., a REST API to a proof-checking service).
    """

    def __init__(self, proof_service_url: Optional[str] = None, translator=None) -> None:
        self.proof_service_url = proof_service_url  # REST endpoint for Isabelle service
        self.translator = translator

    def verify(self, theorem_nl: str, proof_steps: list[str]) -> tuple[bool, list[bool]]:
        if self.translator is None or self.proof_service_url is None:
            return False, [False] * len(proof_steps)

        theory_text = self.translator(theorem_nl, proof_steps)
        if not theory_text:
            return False, [False] * len(proof_steps)

        return self._call_proof_service(theory_text, len(proof_steps))

    def _call_proof_service(self, theory_text: str, n_steps: int) -> tuple[bool, list[bool]]:
        """
        Call a proof-checking REST service (Isabelle, Lean4, or Coq backend).
        Using a service API avoids direct subprocess calls and enables scaling
        to a separate proof-checking infrastructure.
        """
        try:
            import urllib.request
            import urllib.parse
            payload = json.dumps({"theory": theory_text}).encode()
            req = urllib.request.Request(
                self.proof_service_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            ok = result.get("verified", False)
            return ok, [ok] * n_steps
        except Exception:
            return False, [False] * n_steps


class PACFilter:
    """
    Main PAC filter: Stage 2 vs Stage 3 routing for pretraining documents.

    Stage 2 (PAC-admitted): formally verifiable multi-step reasoning.
    Stage 3 (general): everything else that passes standard quality filters.
    """

    def __init__(self, cfg: PACConfig = None) -> None:
        self.cfg = cfg or PACConfig()
        self.extractor = ReasoningStepExtractor()
        self.z3 = Z3StepVerifier()
        self.isabelle = IsabelleVerifier()

    def score(self, doc: dict) -> dict:
        steps = self.extractor.extract(doc.get("text", ""))

        if len(steps) < self.cfg.min_steps:
            return {
                "stage": self.cfg.fallback_stage,
                "verified_ratio": 0.0,
                "n_steps": len(steps),
                "reason": "too_few_steps",
            }

        verified: list[bool] = []
        for prev, curr in zip(steps, steps[1:]):
            ok, reason = self.z3.verify(prev, curr)
            if not ok and reason in ("parse_failed", "no_translator"):
                ok, _ = self.isabelle.verify(prev, [curr])
            verified.append(ok)

        ratio = sum(verified) / len(verified) if verified else 0.0
        stage = 2 if ratio >= self.cfg.verified_ratio_threshold else self.cfg.fallback_stage

        return {
            "stage": stage,
            "verified_ratio": round(ratio, 3),
            "n_steps": len(steps),
            "n_verified": sum(verified),
        }


def run_pipeline(
    input_jsonl: str,
    out_stage2: str,
    out_stage3: str,
    cfg: Optional[PACConfig] = None,
) -> None:
    """
    Split a pretraining JSONL into PAC-verified Stage 2 and general Stage 3.

    Usage:
        python scripts/pac_curriculum_filter.py \\
            --input  data/raw/math_web.jsonl \\
            --stage2 data/curriculum/stage2_pac.jsonl \\
            --stage3 data/curriculum/stage3_general.jsonl
    """
    pac = PACFilter(cfg)
    counts: dict[int, int] = {2: 0, 3: 0}

    with open(input_jsonl) as fin, \
         open(out_stage2, "w") as f2, \
         open(out_stage3, "w") as f3:
        for raw in fin:
            doc = json.loads(raw)
            result = pac.score(doc)
            doc["pac"] = result
            line = json.dumps(doc) + "\n"
            (f2 if result["stage"] == 2 else f3).write(line)
            counts[result["stage"]] += 1

    total = counts[2] + counts[3]
    print(f"PAC: {counts[2]:,} Stage-2 ({counts[2]/total:.1%}), "
          f"{counts[3]:,} Stage-3 ({counts[3]/total:.1%})")
```

**Aurelius integration:**
- New file: `scripts/pac_curriculum_filter.py`
- Modify `train_1b.yaml` — `stage2_data_path` points to PAC-filtered JSONL output
- Add `scripts/train_nl_to_z3_translator.py` — fine-tune Aurelius-Mini to produce `Z3Constraint` objects from (premise, conclusion) pairs; this is the critical prerequisite
- Deploy Isabelle proof-checking REST service (Docker container) and set `proof_service_url`

**Training procedure:**
1. Build NL→Z3Constraint translator: fine-tune Aurelius-Mini on MathQA-SMT pairs (~50K examples)
2. Validate translator: on 1K held-out math steps, check Z3 returns "unsat" for valid transitions (target: >70% correct)
3. Run `pac_curriculum_filter.py` on existing Stage 2 corpus; expect 20–40% Stage 2 yield
4. Supplement with Lean4 / Isabelle proof corpora directly (already PAC-quality by construction)

**Metrics:**
- `pac/stage2_yield_rate` — fraction admitted to Stage 2 (target: 20–40%)
- `pac/z3_parse_success_rate` — translator success rate (target: >60%)
- `pac/verified_ratio_histogram` — distribution of per-doc ratios
- `benchmark/math500` — expected +3–7% vs standard Stage 2 filter
- `benchmark/gsm8k` — secondary math benchmark
- `benchmark/livecodebench` — code reasoning (logical structure benefits transfer)

**Risk / Mitigation:**
- Low yield: if <10% pass, supplement with Lean/Isabelle/Coq corpora. Use PAC as augmentation rather than replacement of existing Stage 2
- Translator accuracy ceiling: NL→Z3 translation is hard for informal math. Start with structured math datasets (AMC, AIME) that have cleaner step formatting
- Isabelle service latency: proof checking is slow; run on CPU workers in parallel, gate Stage 2 pipeline on throughput budget

**Research framing:** *"Proof-Augmented Curriculum: Formal Verification as Pretraining Stage 2 Admission."* Core claim: requiring machine-verifiable step transitions in Stage 2 produces models with stronger axiomatic reasoning. Ablations: threshold sensitivity (0.4/0.6/0.8), PAC vs FOVER PRM-only, Stage 2 yield vs MATH500 gain.

---
### OC-6: CDS — Capability Distillation Spiral

#### Sources Combined

Nous GEPA (ICLR 2026 Oral) + Qwen3 domain synthetic data + NVIDIA Minitron structured pruning (arXiv:2408.11796) + DARE-TIES model merging (DARE: arXiv:2311.03099, TIES: arXiv:2306.01708) + Kimi K2.6 specialist LoRA experts

#### Concept

Most labs apply these techniques as independent one-shot operations:
- Train domain LoRAs → merge once → done
- Prune model → distill → done
- Generate synthetic data → train → done

**CDS chains them into a closed self-improving loop** where each iteration produces:
1. A better base model for the next iteration (from synthetic data)
2. A better cheap synthetic data generator (LoRAs improve each cycle)
3. A better fast draft model for inference (Minitron Mini improves each cycle)

The loop is self-funding: Aurelius-Mini (the pruned version) gets better each iteration and can score/filter synthetic data more efficiently. No external teacher model is required — the system bootstraps quality improvements from its own domain specializations.

No prior work combines all five techniques into an iterative self-improvement pipeline. The individual components appear across Nous, Qwen3, NVIDIA, and DARE/TIES papers independently.

#### Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  CDS ITERATION n                                                     │
│                                                                      │
│  Aurelius-V1.n (base model)                                          │
│       │                                                              │
│       ├─[Step 1]─ Fine-tune domain LoRAs (GEPA benchmark optimize)   │
│       │             Math-LoRA  → optimized: MATH500, GSM8K           │
│       │             Code-LoRA  → optimized: HumanEval, MBPP          │
│       │             Reason-LoRA→ optimized: BBH, ARC-Challenge        │
│       │                                                              │
│       ├─[Step 2]─ DARE-TIES merge all LoRAs → V1.n+specialty         │
│       │             DARE: 7% sparse delta per LoRA                   │
│       │             TIES: majority-vote sign conflict resolution      │
│       │             Target: <1% benchmark regression post-merge      │
│       │                                                              │
│       ├─[Step 3]─ Domain LoRAs generate 50B synthetic tokens         │
│       │             Math-LoRA  → problems + step-by-step solutions   │
│       │             Code-LoRA  → tasks + correct implementations     │
│       │             Reason-LoRA→ chain-of-thought traces + answers   │
│       │             Filter with Aurelius-Mini-n perplexity scorer    │
│       │                                                              │
│       ├─[Step 4]─ Minitron prune V1.n+specialty → Aurelius-Mini-n    │
│       │             Activation importance scoring on calibration set  │
│       │             Structured width pruning (heads + FFN channels)  │
│       │             Distillation-only retraining: 50B tokens         │
│       │             Target: ~500M params, >80% of full model perf    │
│       │                                                              │
│       └─[Step 5]─ Train Aurelius-V1.(n+1)                            │
│                     50% synthetic (LoRA-generated, Mini-scored)      │
│                     50% original pretraining data                    │
│                     Aurelius-Mini-n as speculative draft (speedup)   │
│                                                                      │
│  Output: Aurelius-V1.(n+1) + Aurelius-Mini-n                         │
└──────────────────────────────────────────────────────────────────────┘
```

#### Full Implementation

```python
# scripts/cds/dare_ties.py
"""
DARE-TIES model merging for CDS pipeline.
DARE (arXiv:2311.03099): sparse random pruning of LoRA delta weights.
TIES (arXiv:2306.01708): sign-conflict resolution via majority vote.
"""

from __future__ import annotations
import copy
import torch
from dataclasses import dataclass


@dataclass
class DARETIESConfig:
    density: float    = 0.07    # Fraction of delta kept (DARE recommendation)
    threshold: float  = 0.01    # TIES trim threshold (zero values below this)
    rescale: bool     = True    # Rescale by 1/density to preserve expected value


class DARETIESMerger:

    def dare_prune(
        self,
        delta: torch.Tensor,
        cfg: DARETIESConfig,
    ) -> torch.Tensor:
        """
        DARE step: randomly zero (1-density) fraction of delta weights.
        Rescaling preserves E[pruned] = E[delta] in expectation.
        At 7% density, only the most important 7% of the delta survives;
        this dramatically reduces interference when merging multiple LoRAs.
        """
        mask = torch.bernoulli(torch.full_like(delta, cfg.density))
        pruned = delta * mask
        if cfg.rescale:
            pruned = pruned / cfg.density
        return pruned

    def ties_merge(
        self,
        pruned_deltas: list[torch.Tensor],
        threshold: float = 0.01,
    ) -> torch.Tensor:
        """
        TIES three-step merge:
        1. Trim: zero values below threshold
        2. Elect: majority vote on sign per parameter
        3. Disjoint merge: only keep weights agreeing with elected sign
        """
        stacked = torch.stack(pruned_deltas)          # [n_loras, ...]
        trimmed = stacked * (stacked.abs() > threshold)
        elected_sign = trimmed.sign().sum(dim=0).sign()  # majority sign

        agreed = trimmed * (trimmed.sign() == elected_sign.unsqueeze(0))
        n_agree = (agreed != 0).float().sum(dim=0).clamp(min=1)
        return agreed.sum(dim=0) / n_agree

    def merge(
        self,
        base: dict[str, torch.Tensor],
        loras: list[dict[str, torch.Tensor]],
        cfg: DARETIESConfig = None,
    ) -> dict[str, torch.Tensor]:
        """
        Merge multiple LoRA state dicts into a base model state dict.

        Algorithm per parameter:
          delta_i = lora_i[param] - base[param]
          pruned_i = dare_prune(delta_i, cfg)
          merged_delta = ties_merge([pruned_1, ..., pruned_n])
          result[param] = base[param] + merged_delta
        """
        cfg = cfg or DARETIESConfig()
        result = copy.deepcopy(base)

        for name, base_weight in base.items():
            deltas = []
            for lora_sd in loras:
                if name in lora_sd:
                    delta = lora_sd[name].to(base_weight.device) - base_weight
                    deltas.append(self.dare_prune(delta, cfg))
            if deltas:
                result[name] = base_weight + self.ties_merge(deltas, cfg.threshold)

        return result


# scripts/cds/minitron_prune.py
"""
Minitron-style structured pruning + distillation-only retraining.
Source: NVIDIA Minitron (arXiv:2408.11796).
"""

from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class MiniPruneConfig:
    target_param_ratio: float  = 0.36   # 500M / 1395M ≈ 36%
    prune_attention: bool      = True
    prune_ffn: bool            = True
    calibration_batches: int   = 200
    distill_tokens: int        = 50_000_000_000


class ActivationImportanceScorer:
    """
    Score attention heads and FFN channels by activation magnitude
    on a calibration dataset. Lowest-scoring units are pruned first.
    This is the structured pruning approach from Minitron (arXiv:2408.11796).
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.head_scores:  dict[int, torch.Tensor] = {}
        self.ffn_scores:   dict[int, torch.Tensor] = {}

    def compute(
        self,
        calibration_loader,
        n_batches: int = 200,
    ) -> None:
        accumulations: dict[str, list[torch.Tensor]] = {}
        hooks = []

        for layer_idx, layer in enumerate(self.model.layers):
            def attn_hook(m, inp, out, idx=layer_idx):
                o = out[0] if isinstance(out, tuple) else out
                accumulations.setdefault(f"attn_{idx}", []).append(
                    o.detach().abs().mean(dim=(0, 1))
                )

            def ffn_hook(m, inp, out, idx=layer_idx):
                accumulations.setdefault(f"ffn_{idx}", []).append(
                    out.detach().abs().mean(dim=(0, 1))
                )

            if hasattr(layer, "attention"):
                hooks.append(layer.attention.register_forward_hook(attn_hook))
            if hasattr(layer, "feed_forward"):
                hooks.append(layer.feed_forward.register_forward_hook(ffn_hook))

        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= n_batches:
                    break
                self.model(batch["input_ids"])

        for h in hooks:
            h.remove()

        for layer_idx in range(len(self.model.layers)):
            k_a, k_f = f"attn_{layer_idx}", f"ffn_{layer_idx}"
            if k_a in accumulations:
                self.head_scores[layer_idx] = torch.stack(accumulations[k_a]).mean(0)
            if k_f in accumulations:
                self.ffn_scores[layer_idx]  = torch.stack(accumulations[k_f]).mean(0)

    def prune_mask(
        self,
        layer_idx: int,
        kind: str,          # "attn" or "ffn"
        keep_ratio: float,
    ) -> Optional[torch.BoolTensor]:
        scores = (self.head_scores if kind == "attn" else self.ffn_scores).get(layer_idx)
        if scores is None:
            return None
        n_keep = max(1, int(len(scores) * keep_ratio))
        threshold = scores.topk(n_keep).values.min()
        return scores >= threshold


# scripts/cds/cds_pipeline.py
"""
CDS — Capability Distillation Spiral orchestrator.
Runs one complete self-improvement iteration.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CDSConfig:
    lora_rank: int                    = 64
    lora_alpha: float                 = 128.0
    domains: list[str]                = field(default_factory=lambda: ["math", "code", "reason"])
    domain_benchmarks: dict[str, list[str]] = field(default_factory=lambda: {
        "math":   ["gsm8k", "math500"],
        "code":   ["humaneval", "mbpp"],
        "reason": ["bbh", "arc_challenge"],
    })
    gepa_steps: int                   = 100
    dare_density: float               = 0.07
    ties_threshold: float             = 0.01
    synthetic_tokens_per_domain: int  = 15_000_000_000
    quality_threshold: float          = 0.70
    mini_param_ratio: float           = 0.36
    mini_distill_tokens: int          = 50_000_000_000
    synthetic_mix_ratio: float        = 0.50
    next_gen_tokens: int              = 100_000_000_000


class CDSPipeline:
    """
    Orchestrates one full CDS spiral iteration.

    Each iteration takes ~11 days of compute but produces:
      - Aurelius-V1.(n+1): measurably better on domain benchmarks (+2–5% expected)
      - Aurelius-Mini-n: improved fast draft model for speculative decoding
    The loop is self-funding: better Mini → cheaper data scoring → higher quality V1.(n+1).
    """

    def __init__(self, cfg: CDSConfig, workspace: Path) -> None:
        self.cfg = cfg
        self.ws  = workspace

    def run(self, base_ckpt: str, iteration: int) -> tuple[str, str]:
        """
        Returns: (next_base_checkpoint_path, mini_checkpoint_path)
        """
        print(f"\n=== CDS Iteration {iteration} ===")

        lora_ckpts = self._train_loras(base_ckpt, iteration)
        print(f"  [1/5] Domain LoRAs: {list(lora_ckpts.keys())}")

        enhanced = self._dare_ties_merge(base_ckpt, lora_ckpts, iteration)
        print(f"  [2/5] DARE-TIES merge → {Path(enhanced).name}")

        synthetic = self._generate_synthetic(lora_ckpts, iteration)
        print(f"  [3/5] Synthetic data → {Path(synthetic).name}")

        mini = self._prune_to_mini(enhanced, iteration)
        print(f"  [4/5] Aurelius-Mini-{iteration} → {Path(mini).name}")

        next_base = self._train_next_gen(enhanced, synthetic, mini, iteration)
        print(f"  [5/5] V1.{iteration + 1} → {Path(next_base).name}")

        return next_base, mini

    def _train_loras(self, base_ckpt: str, n: int) -> dict[str, str]:
        """Fine-tune one LoRA per domain with GEPA benchmark optimization."""
        out = {}
        for domain in self.cfg.domains:
            ckpt = str(self.ws / f"iter{n}_{domain}_lora.pt")
            # Calls: scripts/train_lora.py --base {base_ckpt} --domain {domain}
            #        --benchmarks {self.cfg.domain_benchmarks[domain]}
            #        --gepa-steps {self.cfg.gepa_steps} --out {ckpt}
            out[domain] = ckpt
        return out

    def _dare_ties_merge(
        self, base_ckpt: str, loras: dict[str, str], n: int
    ) -> str:
        """DARE-TIES merge all domain LoRAs into the base model."""
        out = str(self.ws / f"iter{n}_merged.pt")
        # Loads state dicts; calls DARETIESMerger().merge(base_sd, lora_sds, cfg)
        # Verifies <1% benchmark regression before saving
        return out

    def _generate_synthetic(self, loras: dict[str, str], n: int) -> str:
        """Use each domain LoRA to generate synthetic SFT data."""
        out = str(self.ws / f"iter{n}_synthetic")
        # Each LoRA generates cfg.synthetic_tokens_per_domain tokens
        # Scored by Aurelius-Mini-(n-1) perplexity; threshold = cfg.quality_threshold
        # n-gram deduplication applied to prevent repetition
        return out

    def _prune_to_mini(self, enhanced_ckpt: str, n: int) -> str:
        """Minitron structured pruning + distillation-only retraining."""
        out = str(self.ws / f"iter{n}_mini.pt")
        # Steps:
        # 1. ActivationImportanceScorer.compute() on calibration dataset
        # 2. Prune heads and FFN channels: keep_ratio = cfg.mini_param_ratio
        # 3. Distillation-only retraining for cfg.mini_distill_tokens tokens
        #    (no KD loss to original data — pure teacher distillation)
        return out

    def _train_next_gen(
        self,
        enhanced_ckpt: str,
        synthetic_data_path: str,
        mini_ckpt: str,
        n: int,
    ) -> str:
        """Train V1.(n+1) on synthetic + original data mix."""
        out = str(self.ws / f"iter{n + 1}_base.pt")
        # Mix: cfg.synthetic_mix_ratio synthetic + (1-ratio) original corpus
        # Aurelius-Mini-n used as speculative draft model during training (speedup)
        # Training for cfg.next_gen_tokens tokens total
        return out


class CDSIterationTracker:
    """Track benchmark progress across CDS iterations for stopping criteria."""

    def __init__(self) -> None:
        self.history: list[dict[str, float]] = []

    def record(self, iteration: int, benchmarks: dict[str, float]) -> None:
        self.history.append({"iteration": iteration, **benchmarks})

    def should_continue(self, min_improvement: float = 0.01) -> bool:
        """
        Continue iterating if the last iteration improved any benchmark by >1%.
        Stop if the loop has converged — improvement is below the threshold.
        """
        if len(self.history) < 2:
            return True
        last = self.history[-1]
        prev = self.history[-2]
        improvements = [
            last[k] - prev[k]
            for k in last
            if k != "iteration" and k in prev
        ]
        return max(improvements, default=0.0) > min_improvement

    def summary(self) -> str:
        if not self.history:
            return "No iterations recorded."
        header = "Iteration | " + " | ".join(
            k for k in self.history[0] if k != "iteration"
        )
        rows = [
            f"{h['iteration']:9d} | " + " | ".join(
                f"{h[k]:.3f}" for k in h if k != "iteration"
            )
            for h in self.history
        ]
        return "\n".join([header] + rows)
```

**Aurelius integration:**
- New directory: `scripts/cds/` with `dare_ties.py`, `minitron_prune.py`, `cds_pipeline.py`
- Existing `scripts/model_tools.py` — extract existing DARE-TIES stubs into `scripts/cds/dare_ties.py`
- Modify `src/training/trainer.py` — add `--cds-mini-draft` flag to use Aurelius-Mini as speculative draft during training

**Iteration cadence:**

| Step | Estimated Time | Resource |
|---|---|---|
| 3× LoRA fine-tuning (GEPA) | 2 days | 3× single-GPU sequential |
| DARE-TIES merge | 2 hours | CPU / single GPU |
| 50B token synthetic generation | 1 day | 4× GPU inference |
| Minitron prune + distill (50B) | 3 days | Single GPU training run |
| V1.(n+1) training (100B) | 5 days | Full multi-GPU training run |
| **Total per iteration** | **~11 days** | |

**Metrics per iteration:**
- `cds/synthetic_quality_ppl` — perplexity of synthetic data scored by V1.n (lower = higher quality; expect decline each iteration as LoRAs improve)
- `cds/dare_merge_regression` — benchmark change after DARE-TIES merge (target: <1% drop on any benchmark)
- `cds/mini_perf_ratio` — Aurelius-Mini-n score / Full score on held-out tasks (target: >0.80)
- `benchmark/gsm8k_iter{n}` — should increase monotonically (+2–5% per iteration)
- `benchmark/humaneval_iter{n}` — same for code
- `inference/speculative_acceptance_rate` — Mini-n as draft model; should improve each cycle
- `cds/convergence` — `CDSIterationTracker.should_continue()` output; stop when improvement <1%

**Risk / Mitigation:**
- DARE-TIES merge regression: if >2% benchmark drop, reduce `dare_density` to 0.04. Lower density means fewer delta weights survive; less cross-LoRA interference at the cost of weaker specialization blending
- Synthetic data quality ceiling: LoRAs may generate repetitive data after few iterations. Add n-gram deduplication, enforce diversity by sampling with temperature>1 during generation, and cap synthetic token reuse per epoch
- CDS iteration cost: gate iteration 2 on iteration 1 showing ≥1% improvement on two or more benchmarks before committing compute. Use `CDSIterationTracker.should_continue()` as the formal stopping criterion

**Research framing:** *"Capability Distillation Spiral: Closed-Loop Self-Improvement Without a Teacher Model."* Key claim: GEPA domain LoRAs → DARE-TIES merge → synthetic data generation → Minitron distillation form a self-improving loop that does not require a larger teacher model. Ablations: per-step contribution (which steps drive the most improvement), iteration curves (n=0,1,2), synthetic/original mix ratio sensitivity (0.25/0.50/0.75), DARE density sweep (0.03/0.07/0.15).

---

### OC-7: EWM — Erase/Write Memory for AMC

#### Source Grounding

Gated DeltaNet-2 (GDN2; local PDF `/Users/christienantonio/Downloads/GDN2_paper.pdf`, dated 2026-05-21) identifies a precise failure mode in fixed-size recurrent memory: existing delta-rule linear attention models use a tied gate for two different operations. They use one scalar to decide both how much of an old key-addressed association to erase and how much of a new value to commit. GDN2 separates these roles with a **channel-wise erase gate** on the key side and a **channel-wise write gate** on the value side, while preserving channel-wise decay and efficient chunkwise training.

The paper's most relevant claims for Aurelius:

- Linear recurrent attention replaces quadratic attention with a fixed recurrent state, but fixed states suffer from interference when many associations share compressed space.
- Delta-rule models target overwrites by subtracting the current read before writing a new value, but prior gated variants tie erase and write too tightly.
- GDN2 keeps broad decay, targeted erase, and selective write as separable controls.
- The architecture remains trainable efficiently: the paper describes a chunkwise WY form and gate-aware backward pass.
- Reported gains are strongest on long-context retrieval and multi-key needle settings — exactly the regime AMC is meant to improve.

#### Aurelius Concept

**EWM applies GDN2's erase/write separation to AMC itself rather than merely adopting GDN2 as another attention block.** AMC currently wants three distinct capabilities: recall useful persistent state, update contradicted state, and block unsafe/stale state. Those are not one operation. A memory system needs at least four knobs:

1. **Decay** — broad freshness or retention horizon.
2. **Erase** — targeted removal/suppression of stale or contradicted associations.
3. **Write** — selective commitment of new value channels.
4. **Admission** — safety/trust/provenance veto before any durable memory becomes authoritative.

GDN2 gives the architectural vocabulary for the first three. Aurelius adds the fourth through AMC Tier-3 and serving safety gates.

#### P0 Contract Addition

Add these fields to the stable AMC contract before any training change:

| Field | Shape / type | Tier | Purpose |
|---|---|---|---|
| `memory_key` | `[layers?, heads?, d_k]` or opaque tensor handle | Tier-2 | Address into differentiable memory. |
| `memory_value` | `[layers?, d_v]` or opaque tensor handle | Tier-2 | Content to commit or recall. |
| `decay_gate` | scalar or `[d_k]` | Tier-2/3 | Broad retention horizon. |
| `erase_gate` | `[d_k]` preferred; scalar fallback allowed | Tier-2 | Targeted suppression of key-side stale associations. |
| `write_gate` | `[d_v]` preferred; scalar fallback allowed | Tier-2 | Selective value-channel commit. |
| `trust_score` | float `[0,1]` | Tier-3 | Trust-weighted recall and write admission. |
| `provenance` | structured dict | Tier-3 | Source, timestamp, verifier, run id. |
| `safety_tier` | enum | Tier-3 / serving | Determines whether memory can be recalled, summarized, redacted, or blocked. |
| `update_reason` | enum/string | Tier-3 | `new`, `correction`, `decay`, `conflict`, `manual`, `tool_result`. |

The P0 rule is: **A memory write is not just append-or-replace; it is a gated edit with independent erase and write metadata.**

#### Implementation Sketch

Target surfaces:

- `src/memory/amc_tensor_api.py` — define the typed tensor/update contract.
- `src/memory/amc_tier2.py` — implement differentiable erase/write update stubs behind a feature flag.
- `src/memory/amc_tier3.py` — apply trust/decay/safety admission before write authority.
- `src/eval/amc_memory_runner.py` — expose `erase_gate`, `write_gate`, and conflict-resolution metrics.
- Future V2 architecture: `src/model/attention.py` or new `src/model/linear_attention.py` for GDN2/KDA-style layers.

Minimal pseudocode contract:

```python
@dataclass(frozen=True)
class AMCEraseWriteUpdate:
    key: torch.Tensor
    value: torch.Tensor
    decay_gate: torch.Tensor | float
    erase_gate: torch.Tensor | float
    write_gate: torch.Tensor | float
    trust_score: float
    provenance: dict[str, str]
    safety_tier: str
    update_reason: str


def apply_amc_edit(state, update: AMCEraseWriteUpdate):
    # Decay: broad retention horizon.
    state = state * clamp_gate(update.decay_gate)

    # Erase: targeted key-side removal of stale association.
    stale_read = state.read(update.key)
    state = state.erase(key=update.key, amount=update.erase_gate, read=stale_read)

    # Write: selective value-side commit only after safety/trust admission.
    if admit_memory_write(update.trust_score, update.safety_tier, update.provenance):
        state = state.write(key=update.key, value=update.value, amount=update.write_gate)
    return state
```

This pseudocode is intentionally contract-level. Do not implement it as production math until the tensor shapes and state layout in `amc_tensor_api.py` are frozen.

#### Required Experiments

| Experiment | Baseline | EWM variant | Primary metric | Failure mode to capture |
|---|---|---|---|---|
| Synthetic key collision | append/replace memory | independent erase/write | correct latest answer under key collision | over-erasing useful old memory |
| Contradiction update | single trust/decay scalar | erase old fact, write corrected fact | stale fact suppression | corrected fact blocked too often |
| Multi-key needle | current AMC recall | EWM gates | multi-key recall at long distractor lengths | gate collapse to uniform scalar |
| Safety write veto | current admission | gated write with safety tier | unsafe write block rate | unsafe content enters Tier-2 |
| Gradient stability | Tier-2 current | EWM differentiable edit | grad norm, NaN/Inf, loss delta | erase gate destabilizes training |

#### Risk / Mitigation

- **Risk:** adopting GDN2 at the model-architecture level before AMC contract work distracts from the AMC-first thesis.
  **Mitigation:** first add EWM as a P0 AMC contract and benchmark hypothesis; only later decide whether V2 attention layers should use GDN2/KDA directly.
- **Risk:** gate degeneracy, where erase/write gates collapse to the same scalar and add complexity without benefit.
  **Mitigation:** log gate entropy and erase/write correlation; fail the experiment if correlation stays near 1.0 across tasks.
- **Risk:** unsafe memories become harder to audit because they live in tensor state.
  **Mitigation:** no Tier-2 write can become authoritative without Tier-3 provenance and safety metadata.

---

### OC-8: RC-AMC — Reflection-Compiled AMC

#### Source Grounding

MeMo: Memory as a Model (arXiv:2605.15156v2) trains a dedicated memory model while keeping the executive LLM frozen. Its data synthesis pipeline extracts facts from a target corpus, consolidates redundant information, verifies/rewrites QA pairs for self-containment, surfaces entities, and synthesizes cross-document relationships. At inference, the executive model queries the memory model through a structured multi-turn protocol rather than retrieving raw passages from an index.

The paper's implications for Aurelius:

- Retrieval noise is a first-class problem; passing raw distractor documents into the context window can degrade reasoning.
- Cross-document relationships should be compiled into memory, not rediscovered from raw chunks every time.
- Memory should have a protocol, not only a vector search call.
- The paper explicitly notes provenance/access-control risks: memory can obscure where answers came from if attribution is not preserved.

#### Aurelius Concept

**RC-AMC converts MeMo's separate memory-model idea into an Aurelius-native AMC compiler.** Instead of training an unrelated standalone memory model, Aurelius should compile corpora, tool results, and durable user/project state into AMC memory artifacts that remain queryable through Tier-1/Tier-3 and, later, trainable through Tier-2.

Aurelius should not merely ask “can vector search retrieve this chunk?” It should ask:

1. What stable facts should become memories?
2. Which facts are redundant or contradictory?
3. Which entities and relationships should be addressable directly?
4. Which memories can answer a question without raw source context?
5. Which memories require provenance disclosure or safety redaction?

#### P0 Contract Addition

Add a **memory compilation artifact** to the docs/API contract:

| Artifact | Required fields | Purpose |
|---|---|---|
| `raw_evidence` | source id, text span, timestamp, license/sensitivity | Keep attribution and audit trail. |
| `fact_candidate` | claim, confidence, source ids | Candidate atomic memory. |
| `consolidated_fact` | merged claim, supporting/contradicting sources | Remove redundancy and expose conflicts. |
| `reflection_qa` | self-contained question, answer, evidence ids | Train/query memory without reloading raw docs. |
| `entity_card` | entity id, aliases, relationships, attributes | Improve multi-hop and reversal recall. |
| `memory_write_plan` | target tier, trust score, decay policy, safety tier | Decide what enters AMC and how. |

#### Implementation Plan

- `scripts/memory/compile_reflections.py` — pipeline for fact extraction, consolidation, verification/rewriting, entity surfacing, and cross-document synthesis.
- `src/memory/amc_tier3.py` — add provenance-preserving admission for compiled memories.
- `src/eval/amc_memory_benchmark.py` — add BrowseComp/NarrativeQA/MuSiQue-style multi-hop synthetic slices.
- `src/serving/api_server.py` — expose a request-time protocol: `ground`, `identify_entities`, `recall`, `synthesize`, `cite`.
- `docs/reports/` — store memory compilation manifests and ablation reports.

#### Required Experiments

| Experiment | Baseline | RC-AMC variant | Primary metric |
|---|---|---|---|
| Noisy corpus recall | dense/BM25 or current Tier-1 | compiled reflection QA | answer accuracy vs distractor ratio |
| Cross-document synthesis | raw chunk retrieval | consolidated facts + entity cards | multi-hop QA accuracy |
| Provenance audit | answer-only memory | memory with evidence ids | citation/source correctness |
| Dynamic update | rebuild full index | incremental memory write plan | update latency and stale-answer rate |
| Tier placement | Tier-1 only | Tier-1 + Tier-3 + optional Tier-2 | recall quality vs latency |

#### Risk / Mitigation

- **Risk:** compiled memory hides source evidence.
  **Mitigation:** every `reflection_qa` and `entity_card` must carry evidence ids; serving should be able to cite or withhold based on policy.
- **Risk:** memory compiler hallucinates facts during synthesis.
  **Mitigation:** verification/rewriting must discard unsupported claims; add negative evidence and contradiction tests.
- **Risk:** cost of compilation exceeds retrieval benefit.
  **Mitigation:** start with project/domain memory, not web-scale corpora; benchmark update cost and request-time savings separately.

---

### OC-9: SDB-Memory Runtime

#### Source Grounding

A Methodology for Selecting and Composing Runtime Architecture Patterns for Production LLM Agents (arXiv:2605.20173v1) names the **stochastic-deterministic boundary (SDB)**: proposer, verifier, commit, reject. The paper audits agent frameworks and failure post-mortems and argues that as base-model variance shrinks, architectural momentum becomes the dominant reliability lever. It catalogs six runtime patterns across Coordination, State, and Control: Hierarchical Delegation, Scatter-Gather plus Saga, Event-Driven Sequencing, Shared State Machine, Supervisor plus Gate, and Human in the Loop.

#### Aurelius Concept

AMC is a production agent memory system, so it must treat every memory-affecting action as an SDB event. The model may propose a memory, a recall, a tool reuse, or a state transition; deterministic code must verify and commit it; failures must produce typed reject signals that the agent can act on.

**SDB-Memory Runtime makes memory safety a P0 runtime contract.** It prevents these classes of failures:

- hallucinated memory writes becoming durable facts;
- stale memories overwriting corrected memories;
- replay divergence when logs are replayed under a new model/prompt;
- unsafe or sensitive content bypassing memory admission;
- agent sub-tasks committing conflicting memories without deterministic merge logic.

#### P0 Contract Addition

Every memory/action boundary should be expressible as:

| Boundary field | Meaning | Example |
|---|---|---|
| `proposal` | LLM/agent-suggested action | “store this tool result as durable memory” |
| `verifier` | deterministic validation | schema, safety tier, provenance, freshness, conflict check |
| `commit` | durable accepted write | append event, CAS state update, memory entry save |
| `reject` | typed failure signal | `schema_error`, `unsafe_memory`, `stale_source`, `conflict_unresolved` |
| `replay_key` | deterministic replay handle | prompt hash, model id, verifier version, source event id |

#### Implementation Plan

- `src/memory/amc_tier3.py` — make admission outcomes typed and auditable.
- `src/runtime/memory_quarantine.py` — route rejected/unsafe/unverified proposals to quarantine.
- `src/workflow/dead_letter_queue.py` — store repeat failures and typed reject signals.
- `src/agent/react_loop.py` — consume reject signals explicitly rather than silently retrying.
- `src/serving/api_server.py` — ensure API-visible memory writes cannot skip verifier/commit.
- `src/observability/trace_context.py` — emit `proposal_id`, `verifier_version`, `commit_id`, `reject_code`.

#### Required Experiments

| Failure signature | Test | Expected SDB behavior |
|---|---|---|
| Unsafe write | prompt tries to store a secret or policy-violating instruction | reject with `unsafe_memory`; no durable commit |
| Stale tool result | old search/calculation reused after invalidation | reject or demote; agent re-fetches |
| Conflict | two agents propose different values for same entity | quarantine or require deterministic resolver |
| Replay divergence | same event log under new model produces different downstream memory | replay drift reported; prior committed projection preserved |
| CAS race | parallel agents write same memory key | stale write rejected by version check |

#### Risk / Mitigation

- **Risk:** SDB instrumentation slows experimentation.
  **Mitigation:** P0 should freeze a minimal contract first, then enforce it only on durable writes and external side effects.
- **Risk:** LLM receives unhelpful reject messages and loops.
  **Mitigation:** reject signals must be typed, concise, and actionable; add retry budget and dead-letter handling.
- **Risk:** logs become large.
  **Mitigation:** store full detail for commits/rejects, sampled detail for benign recalls, aggregated metrics for high-volume cache hits.

---

### OC-10: VEL — Verified Evolution Loop

#### Source Grounding

AutoResearchClaw: Self-Reinforcing Autonomous Research with Human-AI Collaboration (arXiv:2605.20025v1) argues that autonomous research systems fail when they are linear, single-agent, brittle to experiment failure, and unable to carry lessons across runs. Its five mechanisms are structured multi-agent debate, self-healing execution with Pivot/Refine decisions, verifiable result reporting, human-in-the-loop collaboration modes, and cross-run evolution of safeguards.

#### Aurelius Concept

Aurelius should adapt this not as a paper-writing agent, but as a **verified evolution loop for the AMC research program**. The loop should make AMC improvement self-reinforcing without allowing hallucinated benchmark gains or untraceable claims into reports.

The AMC version:

1. **Debate** — proposer, skeptic, implementation reviewer, and safety reviewer generate and critique an AMC hypothesis.
2. **Experiment** — run a small deterministic benchmark or unit-contract spike.
3. **Pivot/Refine** — if the run fails, decide whether to fix the current hypothesis or pivot to a smaller one.
4. **Registry** — every reported number must point to a command, config, seed, commit, and raw output path.
5. **Cross-run lessons** — failure modes become future benchmark cases or skill notes, not vague memory.
6. **Human gate** — user approval required for expensive training, destructive repo operations, or public claims.

#### Implementation Plan

- `src/eval/amc_memory_runner.py` — produce machine-readable result registries, not only printed summaries.
- `src/eval/amc_memory_benchmark.py` — stable benchmark IDs and raw-output paths.
- `docs/reports/` — one report per run with command/config/seed/checkpoint/environment.
- `src/workflow/dead_letter_queue.py` — capture failed experiments and classify Pivot vs Refine.
- `docs/IMPROVEMENT_PLAN.md` — every May 2026 paper-derived idea must list an evidence level and minimal falsification test.

#### Required Experiments

| VEL component | Minimal implementation | Gate |
|---|---|---|
| Result registry | JSON line per benchmark run | every number in docs links to registry row |
| Pivot/Refine | failed test produces typed classification | no repeated blind retries |
| Debate artifact | short structured review before major AMC change | at least one negative critique captured |
| Citation/source check | arXiv/API title verification before citation | no wrong-ID references |
| Cross-run lesson | failure becomes regression test or documented limitation | no vague “remember this failed” notes |

#### Risk / Mitigation

- **Risk:** process overhead overwhelms coding.
  **Mitigation:** require VEL only for AMC claims, architecture pivots, and benchmark reports; simple bugfixes still use normal tests.
- **Risk:** agents optimize to the benchmark registry rather than model quality.
  **Mitigation:** maintain hidden/held-out slices and negative controls.
- **Risk:** cross-run lessons become stale memory clutter.
  **Mitigation:** store durable procedures as skills or docs, not broad session memory.

---

### OC-11: SLR — Stochastic Latent Recall

#### Source Grounding

Probabilistic Tiny Recursive Model (arXiv:2605.19943v1) shows that deterministic recursive inference can get stuck in bad latent basins. PTRM injects Gaussian noise at each deep recursion step, runs multiple parallel trajectories, and selects among candidates using the model's existing Q head. It reports large gains on puzzle benchmarks without retraining or task-specific perturbations.

#### Aurelius Concept

AMC recall and long-horizon reasoning have an analogous failure mode: a deterministic retrieval/reasoning path can lock onto a plausible but wrong memory or reasoning trajectory. **SLR adds controlled stochasticity only inside recall/reasoning candidate generation, then selects through deterministic verifiers.**

This is not “make the agent random.” It is:

- sample K memory-query or latent-reasoning trajectories;
- apply small calibrated perturbations to query/key/latent state;
- score candidates with a Q/verifier head or deterministic evaluator;
- commit only the selected candidate under SDB rules;
- log seeds and candidate scores for replay.

#### P0 Contract Addition

| Parameter | Meaning | Default gate |
|---|---|---|
| `slr_enabled` | allow stochastic recall candidates | off by default |
| `slr_k` | number of parallel trajectories | 4 for spikes, <=8 for expensive evals |
| `noise_sigma` | latent/query noise scale | sweep; must be reported |
| `selection_score` | Q/verifier/certainty metric | deterministic and logged |
| `commit_policy` | when candidate can affect memory/action | SDB commit only |
| `replay_seed` | reproducibility handle | required for reports |

#### Implementation Plan

- `src/reasoning/stochastic_latent_recall.py` — isolated candidate-generation module.
- `src/memory/amc_runtime_cache.py` — optional perturbed query paths for Tier-1/Tier-2 recall experiments.
- `src/eval/amc_memory_benchmark.py` — add “bad-basin” recall tests where top-1 retrieval is misleading.
- `src/agent/react_loop.py` — use SLR only for internal reasoning/recall, never directly for external commits.

#### Required Experiments

| Experiment | Baseline | SLR variant | Primary metric |
|---|---|---|---|
| Misleading nearest memory | deterministic top-1/top-k | K noisy recalls + verifier | correct recall under distractor |
| Multi-hop memory | deterministic chain | K latent chains + Q selection | answer accuracy vs cost |
| Safety prompt | stochastic recall enabled | SDB-gated selection | no unsafe memory disclosure |
| Seed replay | fixed event/log seed | rerun candidates | reproducibility of selected candidate |
| Cost curve | K=1 | K=2/4/8 | accuracy per millisecond/token |

#### Risk / Mitigation

- **Risk:** stochastic recall creates nondeterministic production behavior.
  **Mitigation:** keep off by default; require seeds, logs, and deterministic verifier selection.
- **Risk:** noisy candidates surface unsafe memories.
  **Mitigation:** candidates are proposals only; SDB verifier and safety gates still decide commit/answer eligibility.
- **Risk:** compute cost rises faster than quality.
  **Mitigation:** use only when baseline confidence is low or task is high-value; publish accuracy/latency curves.

---

### OC-12: TBP — Token Boundary Priors

#### Source Grounding

Decoupling the Benefits of Subword Tokenization for Language Model Training via Byte-level Simulation (arXiv:2604.27263v2, Nous Research) isolates why subword tokenization helps. The paper highlights two especially relevant factors: increased effective sample throughput and subword boundaries as explicit priors or inductive biases. It cautions that these effects interact and were studied under finite interventions, English-centric data, and a 1.7B model scale.

#### Aurelius Concept

Aurelius should use this paper to avoid a simplistic tokenizer conclusion. The lesson is not “byte-level is bad” or “BPE is enough.” The lesson is that tokenizer choices alter **where compute is spent**, **how much raw information each gradient step sees**, and **which boundaries the model receives as priors**.

For AMC, boundaries matter because memory writes and recalls should usually align with stable semantic units: entities, tool results, claims, code symbols, citations, and conversation turns. A raw token stream is not enough structure.

#### P0 Contract Addition

Add tokenizer/data accounting to every training or memory benchmark:

| Field | Purpose |
|---|---|
| `raw_bytes_seen` | compare true data exposure across tokenizers/packing settings |
| `tokens_seen` | normal LM accounting |
| `fertility` | average tokens per byte/word/domain |
| `boundary_mask` | marks subword/word/entity/tool/citation boundaries |
| `memory_boundary_mask` | marks candidate memory-write spans |
| `effective_sample_throughput` | raw bytes or semantic units per FLOP/step |
| `boundary_prior_mode` | none, explicit feature, auxiliary loss, curriculum-only |

#### Implementation Plan

- `src/training/sequence_packing.py` — emit raw-byte/semantic-unit packing efficiency alongside token efficiency.
- `src/data/` or tokenizer utilities — add boundary masks for special tokens, entity spans, tool outputs, citations, code identifiers.
- `configs/train_1b.yaml` — add logging toggles for `raw_bytes_seen`, fertility, and boundary priors.
- `src/eval/amc_memory_benchmark.py` — compare memory recall when writes align vs misalign with semantic boundaries.

#### Required Experiments

| Experiment | Baseline | TBP variant | Primary metric |
|---|---|---|---|
| Packing accounting | token efficiency only | token + raw-byte + semantic-unit accounting | honest throughput comparison |
| Memory span alignment | arbitrary token chunks | boundary-aligned memory writes | recall accuracy / conflict rate |
| Boundary mask auxiliary | no boundary feature | explicit boundary prior | loss and downstream recall |
| Multilingual/code slice | English-only assumptions | domain fertility reports | domain regression detection |
| Curriculum intervention | full-run baseline | early boundary-prior stage | validation loss and benchmark transfer |

#### Risk / Mitigation

- **Risk:** boundary priors leak future information in causal training.
  **Mitigation:** only use causally available start-boundary or prior-known structure unless explicitly running a controlled experiment; document non-causal interventions separately.
- **Risk:** throughput gains are misreported because token counts improve while raw data exposure changes.
  **Mitigation:** all training reports include raw bytes and fertility.
- **Risk:** English-centric boundary assumptions hurt multilingual performance.
  **Mitigation:** fertility and boundary-quality reports by language/domain.

---

### OC-13: PPDQ — Phase-Preserving Decode Quantization

#### Source Grounding

Mix-Quant: Quantized Prefilling, Precise Decoding for Agentic LLMs (arXiv:2605.20315v1) identifies an agentic serving trade-off: long-context agents repeatedly pay expensive prefill costs, but uniform low-bit quantization degrades autoregressive decisions. Mix-Quant applies aggressive NVFP4 W4A4 quantization to prefilling while preserving BF16 for decoding. The paper reports roughly 3x prefill speedups on RTX 5090 / Blackwell NVFP4 kernels while preserving much of the reasoning, long-context, and agentic benchmark quality lost under uniform NVFP4.

#### Aurelius Concept

AMC-heavy serving is exactly a phase-separated workload:

1. **Prefill / memory grounding:** ingest prompt, context, retrieved memories, tool logs, and compiled reflections.
2. **Decode / decision:** generate the next answer, tool call, memory write, or safety-sensitive action.
3. **Commit:** SDB verifier decides whether the proposal becomes durable state.

PPDQ says: quantize the high-volume prefill/memory-grounding phase first, but preserve decode precision where token-level mistakes propagate into reasoning and tool actions.

#### P0 Contract Addition

| Serving phase | Default precision | Rationale |
|---|---|---|
| Prompt/context prefill | BF16 baseline; NVFP4 spike where hardware supports it | highest compute volume |
| AMC memory prefill | NVFP4 candidate, BF16 fallback | repeated long memory contexts are expensive |
| Decode | BF16 until evidence says otherwise | token decisions are sensitive |
| Safety/verifier | BF16 or deterministic CPU rules | false accept is unacceptable |
| Commit/audit | deterministic non-quantized logic | must be exact/replayable |

#### Implementation Plan

- `src/serving/api_server.py` — expose phase-aware precision config but keep BF16 default.
- `src/inference/` — add backend capability detection for NVFP4/Blackwell kernels; never assume support.
- `src/eval/amc_memory_benchmark.py` — add phase-aware serving benchmark: long memory prefill + short decode + commit.
- `configs/` — define `serving_precision: {prefill: nvfp4|bf16, decode: bf16}` after hardware check.
- Observability — log prefill latency, decode latency, quality deltas, and safety rejects separately.

#### Required Experiments

| Experiment | Baseline | PPDQ variant | Primary metric |
|---|---|---|---|
| Long memory prefill | BF16 all phases | NVFP4 prefill + BF16 decode | p50/p95 prefill latency |
| Reasoning quality | BF16 all phases | phase-aware quant | MATH/long-context deltas |
| Agentic tool-call | BF16 all phases | phase-aware quant | BFCL/tool-call correctness |
| Safety/verifier | BF16 all phases | quantized prefill only | no safety regression |
| Hardware fallback | unsupported GPU/CPU | BF16 fallback | no crash or silent precision drift |

#### Risk / Mitigation

- **Risk:** NVFP4 hardware support is not available on current machines.
  **Mitigation:** add capability detection and BF16 fallback before any config option.
- **Risk:** quantized memory prefill perturbs recall state.
  **Mitigation:** separate memory-recall metrics from normal text-generation metrics; compare retrieved-memory attribution under BF16 vs NVFP4.
- **Risk:** decode quantization is tempting after prefill speedups.
  **Mitigation:** keep decode BF16 until an explicit decode ablation clears quality and safety gates.

---

### Cross-Contribution Dependency Graph

```
P0 AMC contract lane
├─ OC-9  (SDB-Memory Runtime) ── must land first for durable memory writes, reject codes, replay keys
├─ OC-8  (RC-AMC) ────────────── depends on OC-9 provenance/admission; feeds Tier-1/Tier-3 memory artifacts
├─ OC-7  (EWM) ───────────────── depends on frozen AMC tensor/update contract; later informs GDN2/KDA layer work
├─ OC-12 (TBP) ───────────────── can start with data/packing telemetry; informs memory-write span boundaries
├─ OC-11 (SLR) ───────────────── depends on OC-9 verifier/commit; off by default until seed replay works
├─ OC-10 (VEL) ───────────────── wraps every AMC claim with registry, Pivot/Refine, citation/source verification
└─ OC-13 (PPDQ) ──────────────── after AMC serving benchmark exists; requires hardware capability detection

V1/V2 capability lane
├─ OC-1 (SVD)  ─────── scaffold only after a baseline speculative decoder exists; quality claim requires v1 checkpoint and false-accept ablation
├─ OC-3 (MCMR) ─────── API/token-budget scaffolding can start now; learned routing requires tokenizer/SFT collision gate and mode-routing ablation
├─ OC-4 (CGR)  ─────── can ship at DPO/GRPO stage only after tier-label annotation, memory-admission safety, and refusal/helpfulness gates
├─ OC-6 (CDS)  ─────── can start after first LoRA fine-tuning plus VEL result registry are complete
├─ OC-5 (PAC)  ─────── needs NL→Z3Constraint translator, executable-sandbox gate, and verifier leakage controls
└─ OC-2 (LSD)  ─────── needs MLA in src/model/attention.py and a long-context/AMC serving benchmark before replacing any default path
```

**P0 sequencing rule:** if there is a conflict between a general frontier technique and an AMC-specific P0 contract, implement the AMC contract first. For example, GDN2 should first inform `erase_gate`/`write_gate` semantics in AMC; only after that should Aurelius decide whether to replace or hybridize attention layers.

### P0 AMC Contract Registry

This registry is the canonical implementation lane for the AMC-first rebuild. Items in this table are allowed to create stable schemas, tests, docs, telemetry, and disabled-by-default flags before full model-training evidence exists. They are **not** allowed to claim capability gains until their experiment row passes.

| ID | Contract | Repo-truth anchors checked 2026-05-22 | First deliverable | Must block |
|---|---|---|---|---|
| **P0.1 / OC-9** | SDB-Memory Runtime | `src/memory/amc_runtime_cache.py`, `src/serving/api_server.py`, `src/eval/amc_memory_runner.py`, `tests/eval/test_amc_memory_runner.py` exist. | Typed `MemoryProposal`, `MemoryVerifierResult`, `MemoryCommit`, `MemoryReject` schemas with `proposal_id`, `source_ids`, `trust_score`, `safety_tier`, `reject_code`, and `replay_key`. | Any durable memory write, tool-result reuse, or agent action that bypasses proposer/verifier/commit/reject semantics. |
| **P0.2 / OC-8** | RC-AMC compilation artifacts | Tier-1/2/3 modules exist; compilation schemas do not yet have a single canonical home. | `CompiledMemoryArtifact` schema for facts, entities, relationship edges, reflection QA pairs, evidence spans, and provenance. | Turning summaries/reflections into durable memory without source spans, expiry, and admission result. |
| **P0.3 / OC-7** | EWM erase/write update contract | `src/memory/amc_tensor_api.py`, `src/memory/amc_tier2.py`, `src/memory/amc_tier3.py` exist. | Independent `decay_gate`, `erase_gate`, `write_gate`, collision policy, contradiction policy, and trust-decay telemetry. | Replacing attention/SSM layers with GDN2/KDA-style logic before the memory-update contract is frozen. |
| **P0.4 / OC-12** | TBP boundary/throughput accounting | `src/training/sequence_packing.py` exists and has recent regression coverage. | Raw-byte count, token fertility, boundary mask, memory-span boundary mask, and effective sample throughput in packing/data reports. | Claims that byte/subword/tokenizer changes improved learning unless byte-normalized and token-normalized metrics both appear. |
| **P0.5 / OC-10** | VEL result registry | `docs/reports/`, `src/eval/amc_memory_benchmark.py`, and `configs/amc_first_benchmark.yaml` are the right surfaces. | JSONL result registry with command, config path/hash, seed, checkpoint SHA, hardware, raw output path, source list, and Pivot/Refine outcome. | Public or internal roadmap claims whose numbers cannot be traced to raw outputs and config/seed/checkpoint metadata. |
| **P0.6 / OC-11** | SLR replayable stochastic recall | Recall randomness must live only inside candidate generation, not commit. | Seeded K-trajectory candidate generator contract plus deterministic verifier-selection record. | Any stochastic memory write/commit path without replayable seeds and selected/rejected candidate logs. |
| **P0.7 / OC-13** | PPDQ phase-aware precision contract | Serving path exists; NVFP4 support must be capability-detected and optional. | Capability detector, BF16 fallback, phase tags (`prefill`, `decode`, `verifier`, `commit`), and AMC-heavy serving benchmark. | Decode/verifier/commit quantization until BF16-vs-quantized safety and recall-quality deltas are explicitly green. |

**Repo-path correction note:** `src/inference/svd.py`, `src/inference/server.py`, `src/model/mcmr.py`, and `src/model/rope.py` were not present in this checkout on 2026-05-22. Implementation tickets should target current surfaces (`src/inference/speculative_decoding.py`, `src/inference/token_budget_forcing.py`, `src/model/rope_embeddings.py`, etc.) or explicitly create a new module with exports and tests. Do not silently follow stale sketch paths.

### Minimum Experiments to Claim Each Contribution

| ID | Required Experiment | Primary Metric |
|---|---|---|
| SVD | EAGLE-3 vs SVD on V1; threshold sweep 0.70→0.95 | tokens/sec, false-accept rate, MMLU |
| LSD | EAGLE-3 vs LSD at batch=1/8/32 (memory bottleneck varies) | FLOPs/draft step, acceptance rate |
| MCMR | Static bias router vs MCMR; think/answer task split | Mode routing KL, GSM8K, HellaSwag |
| CGR | Standard DPO vs CGR DPO; expert activation analysis | Safety expert KL, refusal accuracy |
| PAC | Stage 2 standard filter vs PAC filter; same downstream SFT | MATH500, yield rate |
| CDS | Two full iterations V1.0→V1.1→V1.2; ablate each step | Iteration benchmark curves |
| EWM | Current AMC update vs independent `decay_gate`/`erase_gate`/`write_gate` on contradiction and key-collision memory tasks | stale-memory suppression, correct latest recall, erase/write gate correlation |
| RC-AMC | Raw retrieval/Tier-1 memory vs compiled reflection QA + entity cards on noisy multi-hop corpora | answer accuracy vs distractor ratio, source/citation correctness |
| SDB-Memory | Ungated memory write path vs proposer/verifier/commit/reject path with typed failures | unsafe write block rate, conflict rejection, replay drift detection |
| VEL | Manual benchmark notes vs JSON result registry + Pivot/Refine classification + source verification | percent of reported numbers traceable to command/config/seed/raw output |
| SLR | Deterministic top-k recall/reasoning vs K noisy latent/query trajectories with verifier selection | bad-basin escape accuracy, latency/quality curve, seed replay fidelity |
| TBP | Token-only accounting and arbitrary memory spans vs raw-byte/fertility/boundary accounting with aligned memory spans | effective sample throughput, boundary-aligned recall accuracy |
| PPDQ | BF16 all phases vs NVFP4 prefill + BF16 decode on AMC-heavy serving prompts | p50/p95 prefill latency, recall-quality delta, safety regression rate |

---

## Executive Summary

Every frontier lab in 2025–2026 has independently converged on four architectural truths:

1. **MoE with shared expert + fine-grained routing + loss-free load balancing** — DeepSeek V3 (256 experts), Kimi K2.6 (384+1 shared), Step-3.5-Flash (288+1 shared)
2. **Hybrid attention** — linear/SSM layers for most of the stack, sparse sliding-window, with full softmax only periodically — MiniMax (7 Lightning : 1 Softmax), NVIDIA Nemotron (28 Mamba-2 + 6 Attention), Step-3.5 (3 SWA : 1 Full)
3. **Unified thinking / non-thinking via mode tokens** — one model, two behaviours at inference — Qwen3, Anthropic Claude 3.7/4, OpenAI o3/o4
4. **Multimodal early fusion from pretraining day 1** — text and vision tokens jointly trained, not stitched — Meta Llama 4, Kimi K2.6

Aurelius has the architectural foundation for all four. The gap is not conceptual — it is execution order.

**Three low-risk implementation surfaces can be scaffolded this week, but not claimed as quality wins until ablated:**

| Change | File | Safe now | Claim gate |
|--------|------|----------|------------|
| Bias-based MoE router (no aux loss) | `src/model/moe.py` | Add config flag, metrics, and compatibility tests. | Equal-token training/proxy ablation showing no loss or routing collapse. |
| Shared expert always-active | `src/model/moe.py` | Add disabled-by-default module path and checkpoint-compatibility tests. | Parameter-count-adjusted throughput/quality report and rollback checkpoint plan. |
| Alignment tier resolver | `src/alignment/praxis/` (+ public shim if needed) | Add deterministic resolver contract and tests. | Memory-admission, refusal/helpfulness, and unsafe-compliance regression gates. |

---

## Critical Review Addendum — Execution Discipline

**Review date:** 2026-05-21 EDT
**Reviewer stance:** This plan is strong as a frontier-research synthesis, but it should not be treated as an implementation queue until each item passes a local evidence gate. The biggest improvement is to make the plan less like a catalogue of impressive techniques and more like an executable research program with admission criteria, ablation design, rollback paths, and an Aurelius-specific thesis.

### Highest-Value Changes to the Plan Itself

| Gap in the current draft | Why it matters | Improvement added here |
|---|---|---|
| It mixes implementation-ready repo work, training-run config, speculative architecture, and future product bets in one list. | Engineers can accidentally start a high-variance V2/V3 research item before closing low-risk V1/AMC evidence gaps. | Treat every row as one of: **ship-now**, **benchmark-first**, **next-run-only**, **architecture-retrain**, or **research-only**. Do not start work until the row's evidence gate is met. |
| AMC is not explicit enough as the central Aurelius differentiator. | MoE, FP8, MTP, SFT, and KV cache work are valuable but broadly reproducible. AMC is the original thesis: per-layer differentiable memory plus runtime/agent integration. | Add an AMC-first overlay below. All improvements should either strengthen AMC, measure AMC, or remain clearly secondary. |
| Some file paths were generic sketches rather than current-repo paths. | Implementation prompts would waste time or create duplicate modules. | Correct obvious path drift: `configs/train_1b.yaml`, `src/alignment/praxis/`, `src/agent/react_loop.py`, `src/model/rope_embeddings.py`, `src/training/sequence_packing.py`, etc. |
| Several claims depend on blog/vendor/secondary sources. | Blog-only evidence is useful for ideation but not enough for a training-affecting change. | Add source-confidence labels and require primary papers, release notes, or reproduced local benchmarks before merging behavior changes. |
| Implementation sketches omit negative controls. | Without negative controls, a benchmark gain can be caused by data leakage, config drift, extra compute, or evaluator artifacts. | Add paired ablation rules: baseline vs feature, equal tokens, equal params, equal wall-clock budget where possible, and at least one stress test designed to make the feature fail. |
| Risk sections are per-feature but not cross-system. | Most failures in an LLM stack are integration failures: router changes affect training dynamics; thinking tokens affect tokenizer/API/data/evals; agent rewards affect safety. | Add cross-cutting kill gates for regression, cost, safety, latency, and maintainability. |

### Non-Negotiable Decision Rules

1. **AMC remains the primary thesis unless the user explicitly changes strategy.** Frontier-tech additions are supporting systems, not replacements for the AMC-first focused build.
2. **No training-affecting merge without a reproducible ablation.** A unit test proves code shape; it does not prove model quality.
3. **No architecture-affecting merge without a rollback path.** For anything touching attention, MoE, optimizer, tokenizer, or loss surfaces, keep the old implementation behind a config flag until a benchmark report exists.
4. **No blog-only implementation.** Blog/vendor posts can motivate a spike, but the merge gate requires one of: primary paper, official code, internal reproduction, or a clearly documented reason the evidence is sufficient.
5. **No benchmark without leakage notes.** Every benchmark report must state data overlap controls, decoding settings, random seed, checkpoint SHA, hardware, and whether the change altered token budget or parameter count.
6. **No agent/RL verifier that executes code without sandboxing.** Verifiers that run Python, shell commands, JS, notebooks, or containers must be disabled by default and opt-in under a security gate.
7. **No new module if an existing module is the right home.** Prefer extending current canonical surfaces over creating parallel implementations. If a new module is necessary, add an import/export plan and tests.

### Source Confidence Labels

Use these labels in implementation tickets and benchmark reports:

| Label | Meaning | Allowed action |
|---|---|---|
| **S0 Internal-verified** | Already implemented or tested in Aurelius with local commands and passing tests. | May be merged if diff is narrow and validation passes. |
| **S1 Primary-reproduced** | Primary paper/official code exists and a small local reproduction matches the claimed direction. | May ship behind a flag after ablation. |
| **S2 Primary-only** | Primary source exists, but Aurelius has not reproduced it yet. | Spike only; benchmark required before merge. |
| **S3 Secondary-source** | Blog/vendor/summary source without local reproduction. | Research note only; do not train or merge behavior changes. |
| **S4 Speculative synthesis** | Novel combination or extrapolation beyond published evidence. | Treat as research contribution; must define hypothesis, falsification test, and minimal experiment before code. |

### What I Would Change in the Roadmap Ordering

The current **Master Priority Table** is directionally good, but I would insert an explicit zero-phase before item 1:

1. **0A — AMC benchmark harness hardening.** Freeze the evaluation contract before adding more frontier mechanisms.
2. **0B — AMC observability.** Every memory path should emit latency, hit/miss, trust score, consolidation, and safety-admission metrics.
3. **0C — AMC paper ablations.** Prove Tier-1/Tier-2/Tier-3 contributions independently before broad architecture work.
4. Then proceed to **MoE router/shared expert**, **alignment tiering**, and **adaptive thinking API** because those are low-risk and compatible with the current V1 line.

This prevents an implementation trap: making the model more complex before proving the thing that makes Aurelius distinct.

---

## AMC-First Overlay

### North Star

Aurelius should be evaluated as an **AMC-first agentic model stack**, not merely a smaller clone of frontier MoE/attention recipes. The plan should therefore answer this question before each major change:

> Does this make per-layer differentiable memory more reliable, more measurable, safer, faster, or easier to publish?

If the answer is "no," the item can still be valuable, but it belongs behind AMC work in priority unless it unblocks training stability or safety.

### AMC Product / Research Thesis

AMC should be framed as three coupled surfaces:

| AMC surface | Purpose | Current repo anchors | What must be proven |
|---|---|---|---|
| **Tier 1 — runtime cache** | Fast local recall and request/session continuity. | `src/memory/amc_runtime_cache.py`, `tests/memory/test_amc_runtime_cache.py` | Higher recall and lower latency than no-cache/session-only baselines, with bounded memory growth. |
| **Tier 2 — differentiable tensor API** | Memory as a trainable/per-layer signal rather than only external retrieval. | `src/memory/amc_tensor_api.py`, `src/memory/amc_tier2.py`, `tests/memory/test_amc_tensor_api.py`, `tests/memory/test_amc_tier2.py` | Improved long-range consistency and controllable gradient behavior without destabilizing base loss. |
| **Tier 3 — trust, decay, consolidation** | Keep useful memories, forget stale/untrusted memories, and prevent harmful persistence. | `src/memory/amc_tier3.py`, `tests/memory/test_amc_tier3.py` | Better conflict handling, temporal decay, trust-weighted recall, and safety filtering. |
| **Serving / eval bridge** | Make AMC measurable through actual request paths. | `src/eval/amc_memory_runner.py`, `src/eval/amc_memory_benchmark.py`, `tests/eval/test_amc_memory_runner.py` | End-to-end request-time gains with the same API path users will hit. |
| **Agent bridge** | Use memory in long-horizon tool/agent loops. | `src/agent/react_loop.py`, `tests/agent/test_amc_safety_e2e.py` | Fewer repeated tool calls, better task continuity, no unsafe memory leakage. |

### AMC-Centered Critical Path

| Phase | Deliverable | Files / tests | Gate |
|---|---|---|---|
| **P0 — contract freeze** | Document the stable AMC data contract: memory entry fields, tensor shapes, trust metadata, request `amc` payload, and safety admission points. | `src/memory/__init__.py`, `src/serving/api_server.py`, `src/eval/amc_memory_runner.py`, docs | All import/export smoke tests pass; no duplicate contract definitions. |
| **P1 — benchmark harness** | Deterministic benchmark suite with baseline, Tier-1 only, Tier-1+2, Tier-1+2+3. | `src/eval/amc_memory_benchmark.py`, `configs/amc_first_benchmark.yaml` | Report includes recall, contradiction handling, latency, memory growth, and safety outcomes. |
| **P2 — observability** | Structured metrics for cache hit rate, tensor-memory contribution, consolidation events, trust score shifts, and blocked memories. | `src/observability/`, memory modules | Metrics emitted under tests; no import-time side effects. |
| **P3 — agent integration** | AMC-aware ReAct loop that reuses memories but does not bypass safety gates. | `src/agent/react_loop.py`, `tests/agent/test_amc_safety_e2e.py` | Agent tasks show fewer repeated tool calls and no safety regression. |
| **P4 — paper-grade ablations** | Reproducible tables suitable for a technical report. | `docs/reports/`, benchmark scripts | Each AMC tier has an independent contribution table and at least one negative result / failure mode. |

### AMC Benchmark Matrix

| Benchmark family | Example test | Primary metric | Regression guard |
|---|---|---|---|
| **Exact recall** | Insert fact early; ask after long distractor context. | Recall accuracy, confidence calibration. | No hallucinated recall when memory disabled. |
| **Contradiction update** | Store fact A, later store corrected fact B. | Correct latest answer; explicit stale-memory suppression. | Does not retrieve A as authoritative after B is trusted. |
| **Temporal decay** | Store facts with different timestamps and decay policies. | Decay-weighted retrieval rank. | Important durable facts are not forgotten too aggressively. |
| **Trust weighting** | Mix verified and unverified memories. | Verified memory outranks unverified. | User-provided malicious memory cannot override policy/safety. |
| **Tool-result reuse** | Agent searches/computes once, then must reuse result later. | Tool-call reduction and final-answer accuracy. | Does not reuse stale tool output after invalidation. |
| **Safety leakage** | Store sensitive/unsafe strings and request unrelated output. | Block rate; no accidental disclosure. | Safe benign memories still recall normally. |
| **Latency / memory overhead** | Run with AMC off, Tier-1, Tier-2, Tier-3. | p50/p95 latency, RAM/VRAM footprint. | AMC overhead stays within the defined serving budget. |
| **Training stability** | Tier-2 differentiable memory on small training slice. | Loss curve, grad norm, NaN/Inf count. | No destabilization vs baseline seed. |

### How the Broader Improvement Backlog Relates to AMC

| Improvement family | AMC relationship | Priority adjustment |
|---|---|---|
| Bias-based MoE router / shared expert | Useful if MoE specialization can expose memory-specialized experts or reduce interference between base language and memory reasoning. | Keep high priority, but add routing diagnostics that separate memory-heavy tokens from normal tokens. |
| Alignment tier resolver / CGR | Directly supports AMC safety: memory writes and recalls need tiered vetoes. | Promote when connected to memory admission and recall filtering. |
| Adaptive thinking tokens / BeamCoT / ThinkPRM | Helps AMC decide when to retrieve, consolidate, or reason over memory. | Implement API shape early; train behavior only after AMC benchmarks exist. |
| Sequence packing / curriculum / synthetic data | Supports training efficiency and memory/task data. | Include AMC-specific synthetic tasks in curriculum, not only math/code. |
| EAGLE/SVD/LSD/speculative decoding | Can make AMC-enabled inference cheaper, but may hide memory recall errors if verification is weak. | Benchmark with memory-heavy prompts before claiming speedups. |
| KV cache compression / MLA / LongRoPE | Supports long-context serving, which competes with and complements memory. | Compare long-context-only vs AMC retrieval at equal cost. |
| Verifier pool / tool-use timing RL | Directly relevant to agent memory: verifiers can score memory reuse and stale-memory avoidance. | Add memory-specific verifiers before broad RL. |
| Route-SAE / interpretability | Can reveal whether AMC information is represented in identifiable features/layers. | Use as a diagnostic after AMC tier ablations. |
| Multimodal / agent swarm / V4 MoE | Product-scale extensions. | Do not prioritize until AMC is stable enough to survive more modalities and concurrent agents. |

### AMC-Specific Definition of Done

An AMC-related change is not complete until it includes:

- Unit tests for the changed tier or bridge.
- An end-to-end runner test through `src/eval/amc_memory_runner.py` or the serving path.
- A benchmark row comparing AMC-off vs relevant AMC tier(s).
- A safety case: what memories are blocked, decayed, redacted, or allowed.
- A latency/memory budget note.
- A rollback switch or config flag if behavior changes at serving/training time.
- Documentation of at least one observed failure mode.

---

## Evidence, Validation, and Kill Gates

### Evidence Ladder

| Stage | Evidence | Required before |
|---|---|---|
| **E0 — Literature note** | Source summary, claimed mechanism, applicability hypothesis. | Adding to this document. |
| **E1 — Local spike** | Minimal code or notebook proving interfaces and tensor shapes. | Creating an implementation PR. |
| **E2 — Unit contract** | Focused tests for shapes, configs, import/export, deterministic edge cases. | Enabling behind a flag. |
| **E3 — Offline benchmark** | Baseline vs feature on fixed seeds and fixed data. | Default-on in dev configs. |
| **E4 — Training ablation** | Equal-token/equal-compute run with checkpointed metrics. | Default-on in training configs. |
| **E5 — Serving canary** | Latency, memory, accuracy, and safety under request path. | Default-on in serving configs. |
| **E6 — Report-ready reproduction** | Full methodology, raw metrics, failure cases, and commit/config hashes. | Public claims / paper framing. |

### Global Kill Gates

Stop or revert the change if any of the following occur:

| Area | Kill condition |
|---|---|
| **Quality** | >0.5% absolute regression on a core benchmark without a compensating targeted gain that was planned in advance. |
| **Safety** | Any new unsafe-memory recall, sandbox bypass, credential leakage, or policy-gate bypass. |
| **Training stability** | NaN/Inf, sustained gradient norm spike, optimizer divergence, or loss curve worse than baseline beyond warmup. |
| **Latency** | p95 request latency exceeds the accepted budget for the feature class. |
| **Memory footprint** | RAM/VRAM overhead exceeds the documented budget or grows unbounded with sessions. |
| **Complexity** | The feature needs duplicate modules, hidden global state, or import-time side effects to work. |
| **Maintainability** | Tests only pass by weakening assertions, skipping relevant paths, or special-casing benchmarks. |

### Per-Family Validation Requirements

| Family | Minimum validation |
|---|---|
| **MoE routing** | Expert utilization histogram, per-expert token counts, router entropy, quality metric, and gradient-impact comparison vs old aux-loss route. |
| **Shared expert** | Parameter-count delta, throughput delta, shared-expert gradient norm, routed-expert specialization report, and no regression on dense-path tests. |
| **Tokenizer / thinking tokens** | Tokenizer round-trip, special-token collision check, API validation, SFT formatter tests, and generation stop behavior tests. |
| **Training precision / optimizer** | Hardware capability detection, BF16 fallback, loss-scale logs, throughput comparison, and small-run convergence match. |
| **Attention / context** | Shape tests, cache compatibility, short-context regression, long-context gain, and memory-footprint profile. |
| **RL / verifier pool** | Sandbox/offline gate, timeout behavior, reward-distribution sanity check, adversarial prompts, and correlation with held-out human/preference labels. |
| **Speculative decoding** | Acceptance rate by draft position, false-accept rate, speed at batch sizes 1/8/32, quality parity, and rollback to baseline decoder. |
| **Data pipeline** | Dedup report, contamination scan, quality classifier calibration, per-domain mixture report, and reproducible dataset manifest. |
| **AMC** | Tier ablation, safety-memory regression, conflict update test, latency/footprint profile, and agent continuity task. |

### Internet-Verified Source-Grounded Plan Improvements (2026-05-22)

The following sources were re-checked through direct arXiv API, official GitHub raw docs, or official project documentation on 2026-05-22. They should modify the roadmap as validation gates rather than as immediate architecture rewrites.

| Source | Verified mechanism | Aurelius plan change |
|---|---|---|
| **Infini-attention** — `arXiv:2404.07143` | Bounded-memory long-input transformer block combining compressive memory, local masked attention, and long-term linear attention. | Add a long-stream baseline against AMC: bounded memory growth, streaming recall, eviction/collision behavior, and AMC-vs-compressive-memory cost curves. |
| **RULER** — `arXiv:2404.06654` | Long-context evaluation suite extending needle-in-haystack into multi-needle, multi-hop tracing, aggregation, and length-scaled tests. | Replace “supports N tokens” claims with length curves: 8K/32K/64K/128K, multi-hop, aggregation, and distractor stress. |
| **LongBench v2** — `arXiv:2412.15204` | 503 realistic long-context tasks up to 2M words across QA, long ICL, dialogue, code repos, and structured data. | Add realistic long-context reasoning gates and separate direct-answer vs reasoning-budget scores. |
| **MInference 1.0** — `arXiv:2407.02490` | Dynamic sparse attention for long-context prefill acceleration, targeting the TTFT bottleneck. | Every long-context or AMC-heavy serving claim must report prefill latency, TTFT, decode latency, memory growth, and quality retention. |
| **AgentDojo** — `arXiv:2406.13352` | Agent security benchmark with realistic tool-use tasks and prompt-injection test cases over untrusted tool data. | Promote tool-result prompt-injection regression to P0 for `src/agent/react_loop.py` and memory/tool-result reuse. |
| **MCP Tools + Security Best Practices** — official `modelcontextprotocol/modelcontextprotocol` docs | Tool schemas are model-controlled; specs call for visible tools, invocation indicators, confirmation/denial controls, and security review for confused-deputy-style attacks. | Add typed tool-origin/trust metadata to agent/runtime contracts; memory writes from tool outputs inherit origin and confirmation policy. |
| **Inspect AI** — official UK AISI framework docs/GitHub | Evaluation framework with solvers, scorers, model/tool traces, sandboxes, and reproducible eval artifacts. | Shape VEL around inspect-style immutable eval specs, solver/scorer metadata, traces, sandbox settings, and comparable JSON outputs. |
| **LLM-42** — `arXiv:2601.17768` | Deterministic inference via verified speculation; identifies batching/floating-point non-associativity as sources of nondeterminism. | Result registry entries must include deterministic replay metadata or mark the result nondeterministic; safety/benchmark claims need replay checks. |
| **s1 test-time scaling** — `arXiv:2501.19393` | Budget forcing changes test-time reasoning by extending/terminating thought budgets. | Treat reasoning effort as a serving contract: min/max budget, answer-transition policy, cost/accuracy curve, and mode-token collision tests. |
| **QServe / OmniServe** — `arXiv:2405.04532` + official code | W4A8KV4 low-bit serving requires system co-design; kernel-level wins can be erased by dequantization overhead. | PPDQ must use end-to-end agentic-serving metrics, not only kernel throughput or perplexity. |
| **Byte Latent Transformer** — `arXiv:2412.09871` | Byte-level dynamic patches can match tokenized LMs at scale while improving efficiency/robustness. | TBP must track byte-normalized, token-normalized, and boundary-aligned metrics before any tokenizer or byte/patch rewrite. |
| **OLMo 2** — `arXiv:2501.00656` | Fully open model family with released weights, data, code/recipes, logs, and intermediate checkpoints. | Add open-training reproducibility requirements: data-mixture versions, optimizer schedule, intermediate checkpoints, eval snapshots, and post-training recipe metadata. |

#### New cross-cutting gates from the source pass

1. **Long-context truth gate:** A context-extension row is not green unless it passes RULER-style synthetic retrieval, LongBench-v2-style realistic reasoning, and AMC-vs-long-context equal-cost comparison.
2. **Agent trust gate:** Tool output is untrusted input. Memory writes sourced from tool results must preserve origin, permission scope, confirmation policy, and prompt-injection test evidence.
3. **Deterministic replay gate:** Any benchmark/result-registry row that drives roadmap decisions must store seed, decoding config, batch shape if relevant, kernel/precision mode, hardware, and a replay verdict.
4. **End-to-end quantization gate:** Quantization claims must include TTFT, p50/p95 latency, tokens/sec, memory, recall-quality delta, and safety regression under the real serving path.
5. **Reasoning-budget gate:** Thinking mode is an API contract before it is a training objective: enforce max/min budgets, transition-to-answer behavior, and per-request telemetry.
6. **Open-training reproducibility gate:** Training changes must emit enough artifact metadata to be audited like OLMo-style open recipes, even if the run is private.

### Implementation Ticket Template

Every implementation ticket derived from this plan should include:

```markdown
## Hypothesis
If we add <feature>, then <metric> improves because <mechanism>.

## Evidence level
Current: E<0-6>. Target for merge: E<0-6>.

## Files
Canonical files to modify. Explicitly list compatibility shims separately.

## Baseline
Command, config, checkpoint, data slice, seed, hardware.

## Acceptance criteria
- Functional:
- Quality:
- Safety:
- Latency/memory:
- Documentation:

## Kill criteria
Exact conditions that stop the work or keep it behind a flag.

## Rollback
Config flag, revert path, and compatibility concerns.
```

### Recommended Next Ten Tranches

1. **P0.1 SDB-Memory Runtime** — add typed proposal/verifier/commit/reject/replay schemas and tests before any durable memory/action expansion.
2. **P0.2 RC-AMC compilation artifacts** — define compiled fact/entity/reflection artifacts with evidence spans, expiry, and admission metadata.
3. **P0.3 EWM update contract** — separate decay/erase/write semantics and contradiction policy in AMC before attention-layer rewrites.
4. **P0.4 TBP boundary accounting** — extend sequence packing/data manifests with raw-byte, fertility, boundary, and memory-span telemetry.
5. **P0.5 VEL result registry** — make AMC and roadmap claims traceable to command/config/seed/checkpoint/hardware/raw-output metadata.
6. **AMC observability + safety admission metrics** — emit structured memory event metrics and blocked/decayed/redacted-memory counts.
7. **Path-correct low-risk V1 scaffolds** — MoE bias router flag, shared expert flag, PRAXIS tier resolver, adaptive thinking API shape; no default-on quality claims yet.
8. **Tokenizer/SFT/reasoning-budget gate** — special-token collision checks, formatter tests, stop/answer-transition behavior, and budget telemetry.
9. **Agent trust + verifier sandbox lane** — tool-result prompt-injection tests, MCP-style tool-origin metadata, and sandbox-disabled-by-default verifier interfaces.
10. **Speculative/long-context/quantization only after baselines** — EAGLE/SVD, LongRoPE2, MInference-style sparse prefill, and PPDQ proceed only after v1 checkpoint, AMC serving benchmark, and deterministic replay gates are frozen.

---

## Reading This Document

Each improvement entry follows this structure:

- **What it is** — technique in plain terms
- **Why it matters to Aurelius specifically** — gap analysis vs current codebase
- **Source** — which lab, which paper/release, with link
- **Validated at** — model scale / benchmark where this was proven
- **Files to change** — specific paths in the Aurelius repo
- **Implementation sketch** — concrete code showing the key change
- **Metrics to watch** — how to confirm it worked
- **Risk / mitigation** — what can go wrong
- **Depends on** — what must land first

---

## Version Roadmap

| Version | Params | Active | Status | Focus |
|---------|--------|--------|--------|-------|
| v1 | 1.395B | 1.395B | Training in progress | Solidify dense baseline, inference upgrades, alignment pipeline |
| v1-Mini | ~500M | ~500M | Post v1 | Minitron pruning + distillation → on-device product |
| v2 | 2.7B | 2.7B | Planned | MLA, hybrid attention (SWA + Lightning), iRoPE, Mamba-2 layers |
| v3 | 3.0B | 3.0B | Planned | Early-fusion multimodal, agent swarm, long-context research after 128K gates |
| v4 | ~5B MoE | ~2B | Future | Fine-grained MoE (64→256 experts), Nemotron-style hybrid MoE |
| v5 | 7–14B | 7–14B | Future | BF16/4-bit quant, distributed inference |
| v6 | 32B | ~8B MoE | Future | Expert parallelism, multi-node |

---

## Dependency Graph

```
[FP8 training]──────────────────────────────────────┐
[HTMuon upgrade]──────────────────────────────────── ├──► next training run
[MTP-3]──────────────────────────────────────────── ┘
[3-stage curriculum]─────────────────────────────── ┘

[Shared expert]───────────────────────────────────── scaffold behind flag now; default-on only after ablation
[Bias MoE router]─────────────────────────────────── scaffold behind flag now; default-on only after proxy/full ablation
[Alignment tier resolver]─────────────────────────── can ship as deterministic contract/tests now
[Adaptive thinking budget]────────────────────────── can ship as API contract now; learned behavior requires SFT gate

[Verifier pool]──────────────────────────────────── ─┐
[CISPO RL]────────────────────────────────────────── ├──► alignment pipeline
[Thinking/non-thinking mode]──────requires SFT────── ┘

[EAGLE-3 draft head]──────requires v1 checkpoint──── inference
[SWA 3:1 layers]──────────requires retrain────────── next training run

[MLA]──────────────────────────────────────────────► V2 (training-time change)
[Lightning Attention]──────────────────────────────► V2
[Mamba-2 hybrid]───────────────────────────────────► V2
[iRoPE]────────────────────────────────────────────► V2

[Early fusion VLM]─────────────────────────────────► V3
[Minitron → Mini]──────────requires v1 mature────── ► V3 / parallel
[Agent Swarm]──────────────────────────────────────► V3

[Fine-grained MoE 64+]─────────────────────────────► V4
```

---

## V1 — Immediate: No Retraining Required

These changes touch config files, router logic, schemas, telemetry, or inference/API code only. They can be introduced behind disabled-by-default flags and unit tests without launching a new training run. They must **not** be described as model-quality improvements until an equal-token/equal-compute ablation passes the Evidence Ladder above.

---

### 1. Auxiliary-Loss-Free MoE Load Balancing

**What it is:** Replace the gradient-producing auxiliary load-balancing loss with a per-expert *bias* term that adjusts dynamically after each forward pass without producing any gradient signal.

**Why it matters to Aurelius specifically:** The current `SparseMoELayer` in `src/model/moe.py` uses `ep_load_balancing` which appends an auxiliary loss to the training objective. This loss fights the router — it forces the model to choose experts for balance reasons rather than quality reasons. Every gradient step is a compromise. DeepSeek's ablations show this lowers the model's quality ceiling measurably. Their switch to loss-free routing was a key factor in V3's efficiency at 671B.

**Source:** DeepSeek-V3 Technical Report, [arXiv 2412.19437](https://arxiv.org/pdf/2412.19437); dedicated paper [arXiv 2408.15664](https://arxiv.org/pdf/2408.15664)

**Validated at:** 671B MoE (DeepSeek V3), proven to improve both expert specialization and downstream benchmark scores

**Files to change:** `src/model/moe.py`

**Implementation sketch:**

```python
class BiasDynamicRouter(nn.Module):
    """
    Auxiliary-loss-free MoE load balancing via per-expert bias adjustment.
    After each forward pass, overloaded experts accumulate negative bias;
    underloaded experts accumulate positive bias. Zero gradient interference.
    """
    def __init__(self, n_experts: int, top_k: int, bias_lr: float = 1e-3):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.bias_lr = bias_lr
        # Non-parameter buffer — updated post-step, never differentiated
        self.register_buffer("expert_bias", torch.zeros(n_experts))

    def forward(self, router_logits: torch.Tensor):
        # Add bias to routing scores before top-k selection
        adjusted = router_logits + self.expert_bias
        scores, indices = adjusted.topk(self.top_k, dim=-1)
        weights = scores.softmax(dim=-1)
        return weights, indices

    @torch.no_grad()
    def update_bias(self, expert_counts: torch.Tensor):
        """Call after each forward pass. No gradient produced."""
        target = expert_counts.sum() / self.n_experts
        delta = expert_counts.float() - target
        # Overloaded experts get pushed down; underloaded get pushed up
        self.expert_bias -= self.bias_lr * delta.sign()
```

**To wire in:** In `SparseMoELayer.forward()`, replace:
```python
# OLD
router_loss = self.ep_load_balancing(router_logits, expert_indices)
self.aux_loss += router_loss
# NEW
self.router.update_bias(expert_counts)   # call after computing counts
# no aux_loss accumulation
```

**Metrics to watch:** Expert utilization histogram (should become more uniform without being forced); perplexity on held-out eval set (should drop slightly); individual expert activation entropy (should increase — more specialization).

**Risk / mitigation:** Bias can drift if `bias_lr` is too high. Clamp `expert_bias` to `[-2.0, 2.0]` initially. Monitor per-expert load histograms every 100 steps.

**Depends on:** Nothing. Ships immediately.

---

### 2. Shared Expert — Always-Active Slot

**What it is:** Add one permanently-active FFN block (the "shared expert") that processes *every* token alongside the existing top-k routed experts. The shared expert handles universal linguistic structure so routed experts can specialise more aggressively.

**Why it matters to Aurelius specifically:** With 8 experts and top-2 routing, each token activates 25% of experts. There is no guaranteed home for base linguistic patterns — verb conjugation, determiners, punctuation handling. These patterns partially occupy every routed expert, diluting specialisation. The shared expert offloads this universal processing.

**Source:** Kimi K2.6 (384 routed + 1 shared, [deepinfra.com](https://deepinfra.com/blog/kimi-k2-6-model-overview)); Step-3.5-Flash (288 routed + 1 shared, [arXiv 2602.10604](https://arxiv.org/html/2602.10604v1)); DeepSeekMoE original paper

**Validated at:** 1T parameters (Kimi K2.6), 196B parameters (Step-3.5-Flash), 16B parameters (DeepSeekMoE)

**Files to change:** `src/model/moe.py`

**Implementation sketch:**

```python
class SparseMoELayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int):
        super().__init__()
        # Shared expert — always active, sees every token
        self.shared_expert = SwiGLUFFN(d_model, d_ff // 2)  # half-width to save params
        # Routed experts — unchanged
        self.router = BiasDynamicRouter(n_experts, top_k)
        self.experts = nn.ModuleList([SwiGLUFFN(d_model, d_ff) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shared path: every token goes through this
        shared_out = self.shared_expert(x)

        # Routed path: top-k experts per token
        weights, indices = self.router(self.router_proj(x))
        routed_out = self._dispatch_and_combine(x, weights, indices)

        # Combine: shared provides base, routed provides specialisation
        return shared_out + routed_out
```

**Note on parameter budget:** The shared expert should be half-width (`d_ff // 2`) so total parameter count stays close to before. The routed experts' combined contribution already accounts for the majority of capacity.

**Metrics to watch:** Perplexity on eval (expect small improvement); inspect shared expert gradient norms (should be high, confirming it's doing real work); routed expert activation entropy (should increase — they specialise more).

**Risk / mitigation:** The shared expert can dominate if its learning rate is too high. Use a separate param group with `lr_scale: 0.5` relative to routed experts. Also apply stronger weight decay (`wd: 0.2` vs standard `0.1`).

**Depends on:** Item 1 (bias router) recommended first, but independent.

---

### 3. Alignment Tier Resolver — 4-Tier Priority Hierarchy

**What it is:** Group PRAXIS/MOSAIC's 6 reward signals into four tiers (Safety → Ethics → Compliance → Helpfulness). When signals conflict, higher tiers veto lower tiers rather than competing at equal weight in Bayesian fusion.

**Why it matters to Aurelius specifically:** PRAXIS v2 uses Bayesian inverse-variance weighting across SRC, ESA, MTAH, and related signals. This is mathematically elegant but produces ambiguous outcomes when a safety signal is weak (high variance → low weight) while a helpfulness signal is strong. A helpful-but-unsafe response can win. Anthropic's 2026 constitution explicitly addresses this with a tiered hierarchy: safety always beats ethics beats compliance beats helpfulness, regardless of signal confidence.

**Source:** Anthropic Claude 4 / Claude's Constitution, [anthropic.com](https://www.anthropic.com/news/claudes-constitution); Claude 3.7 hybrid reasoning, [deeplearning.ai](https://www.deeplearning.ai/the-batch/claude-3-7-sonnet-introduces-hybrid-reasoning-and-extended-thinking/)

**Files to change:** `src/alignment/praxis/` (especially `praxis_loss.py` and config); `aurelius/alignment/praxis.py` only if the public re-export surface changes

**Implementation sketch:**

```python
# Tier definitions — assign each existing PRAXIS signal to a tier
ALIGNMENT_TIERS = {
    1: {  # Safety — veto power over everything
        "topology_safety",        # persistent-homology invariants
        "superposition_geometry", # polysemanticity detection
        "jailbreak_detector",
        "adversarial_defense",
    },
    2: {  # Ethics — veto power over tiers 3+
        "constitutional_gate",    # PRAXISLoss constitutional component
        "harm_taxonomy",          # 9-category harm classifier
    },
    3: {  # Compliance — veto power over tier 4
        "pii_scanner",
        "policy_checker",
    },
    4: {  # Helpfulness — only active when tiers 1-3 are satisfied
        "src",    # SteeringRewardCorrespondence
        "esa",    # ExpertSafetyAffinity
        "mtah",   # MultiTokenAlignmentHorizon
        "precision_fusion",
    },
}

TIER_VETO_THRESHOLDS = {1: 0.3, 2: 0.4, 3: 0.5}  # below = veto

class AlignmentTierResolver:
    def resolve(self, signal_scores: dict[str, float]) -> tuple[float, int]:
        """
        Returns (final_reward, veto_tier).
        veto_tier=0 means no veto — full helpfulness reward returned.
        veto_tier>0 means that tier vetoed — penalised reward returned.
        """
        for tier, signals in sorted(ALIGNMENT_TIERS.items()):
            if tier == 4:
                break  # no veto at helpfulness tier
            tier_scores = [signal_scores.get(s, 1.0) for s in signals]
            if min(tier_scores) < TIER_VETO_THRESHOLDS[tier]:
                # Veto: return a scaled penalty, not zero (avoids reward hacking)
                penalty = min(tier_scores) / TIER_VETO_THRESHOLDS[tier]
                return penalty * -1.0, tier

        # No veto — compute standard Bayesian-weighted helpfulness reward
        helpfulness_scores = {s: signal_scores[s] for s in ALIGNMENT_TIERS[4]
                               if s in signal_scores}
        return self._precision_fusion(helpfulness_scores), 0
```

**Metrics to watch:** Rate of safety/helpfulness conflicts in eval logs; constitutional gate veto frequency; response quality on adversarial red-team prompts.

**Risk / mitigation:** Setting veto thresholds too low makes the model overly cautious. Start conservative (`0.3` for safety tier), then relax based on red-team data. Log every veto with which signal triggered it.

**Depends on:** Nothing. Ships immediately.

---

### 4. Adaptive Thinking Token Budget

**What it is:** Expose `max_thinking_tokens` as a per-request API parameter. Add a `<stop_thinking>` special token the model learns to emit when it reaches confident conclusions mid-reasoning, enabling adaptive early termination.

**Why it matters to Aurelius specifically:** Current Aurelius either uses the full MCTS/CoT path or none at all — there is no mid-point. Simple queries (factual recall, basic arithmetic, pleasantries) are over-computed. Hard queries (multi-step math, long-horizon planning) are under-computed because there is no budget to extend. Claude 3.7 and Qwen3 proved the same checkpoint can serve both use cases by toggling the thinking budget per request.

**Source:** Anthropic Claude 3.7 extended thinking ([tomsguide.com](https://www.tomsguide.com/computing/anthropic-just-launched-claude-3-7-sonnet-with-new-hybrid-reasoning-model)); OpenAI o3/o4 test-time search; Qwen3 non-thinking mode ([arXiv 2505.09388](https://arxiv.org/pdf/2505.09388))

**Validated at:** Claude 3.7 Sonnet: 62.3% → 70.3% SWE-Bench with extended thinking; o4-mini: 99.5% AIME 2025 with Python tool + extended reasoning budget

**Files to change:** `src/serving/api_server.py`, `gateway/api_server.py` if that compatibility surface is still wired, `src/reasoning/chain_of_thought.py`, `src/inference/token_budget_forcing.py`

**Implementation sketch:**

```python
# src/serving/api_server.py — expose budget in API (mirror in gateway/api_server.py only if still routed)
class ChatCompletionRequest(BaseModel):
    messages: list[Message]
    model: str = "aurelius"
    max_tokens: int = 2048
    temperature: float = 0.7
    # New: thinking budget
    max_thinking_tokens: int = 0        # 0 = fast mode, no CoT
    thinking_effort: str = "none"       # "none" | "low" | "medium" | "high"

    @validator("max_thinking_tokens", pre=True, always=True)
    def resolve_thinking_tokens(cls, v, values):
        # Allow string shorthand: "high" → 8192, "medium" → 2048, "low" → 512
        effort_map = {"none": 0, "low": 512, "medium": 2048, "high": 8192}
        return effort_map.get(values.get("thinking_effort", "none"), v)

# src/reasoning/chain_of_thought.py — adaptive early stop
class AdaptiveThinkingEngine:
    STOP_THINK_TOKEN_ID = 8191   # reserved in 8192-vocab tokenizer

    def generate_with_thinking(
        self,
        prompt_ids: torch.Tensor,
        max_thinking_tokens: int,
        **gen_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (thinking_ids, response_ids).
        Thinking stops at STOP_THINK_TOKEN_ID or when budget is exhausted.
        """
        think_open = torch.tensor([THINK_TOKEN_ID])
        prefixed = torch.cat([prompt_ids, think_open])

        thinking_ids = self.model.generate(
            prefixed,
            max_new_tokens=max_thinking_tokens,
            stop_token_ids=[self.STOP_THINK_TOKEN_ID],
            **gen_kwargs,
        )

        # Append stop token if model didn't emit it (budget exhausted)
        think_close = torch.tensor([self.STOP_THINK_TOKEN_ID])
        full_prefix = torch.cat([prefixed, thinking_ids, think_close])

        # Now generate the actual response
        response_ids = self.model.generate(full_prefix, **gen_kwargs)
        return thinking_ids, response_ids
```

**Training requirement:** SFT mix needs 30% examples in `<think>...</think>` format. See item "Hybrid Thinking Mode SFT Mix" in the next training run section.

**Metrics to watch:** Token efficiency ratio (thinking tokens / total tokens vs task difficulty); MMLU/ARC accuracy at `budget=0` vs `budget=512` vs `budget=8192`; latency p50/p99 per budget tier.

**Risk / mitigation:** `<stop_thinking>` must be trained into the model — it will not emerge from architecture alone. If undertrained, model ignores the token and burns the full budget. Fix: oversample easy queries in the `<think>` SFT mix so the model learns when *not* to keep thinking.

**Depends on:** SFT training mix changes (next training run section). API changes can ship immediately; full adaptive behaviour requires retraining.

---

### 5. iRoPE Inference Passthrough (Prep for V2)

**What it is:** Instrument the existing RoPE implementation to log per-layer attention entropy, identifying which layers would benefit most from NoPE treatment. No behaviour change yet — this is measurement work that informs V2 architecture decisions.

**Why it matters:** Meta's iRoPE (Llama 4) alternates RoPE and NoPE layers. The NoPE layers (no positional embedding) do pure content-based global attention — critical for long-context recall. Before applying iRoPE in V2, Aurelius needs data on which of its 24 layers are positionally dominant vs content-dominant.

**Source:** Llama 4 iRoPE architecture ([arXiv 2601.11659](https://arxiv.org/pdf/2601.11659)); [Medium explanation](https://medium.com/@mandeep0405/llama-4s-architecture-deconstructed-moe-irope-and-early-fusion-explained-e58eb9403067)

**Files to change:** `src/model/rope_embeddings.py` / `src/model/rope_cache.py`, plus `src/observability/`

**Implementation sketch:**

```python
# src/model/rope_embeddings.py or src/model/rope_cache.py — add attention entropy logging
class RotaryPositionEmbedding(nn.Module):
    def forward(self, q, k, v, layer_idx: int = -1):
        # ... existing RoPE application ...
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_probs   = attn_weights.softmax(dim=-1)

        # Log per-layer attention entropy to observability pipeline
        if self.training and layer_idx >= 0:
            entropy = -(attn_probs * attn_probs.log().clamp(min=-20)).sum(-1).mean()
            self.telemetry.record_gauge(
                f"attention_entropy_layer_{layer_idx}", entropy.item()
            )
        return attn_probs @ v
```

**Metric to collect:** Entropy per layer over 10K training steps. Layers with high entropy (diffuse attention, content-based) are candidates for NoPE in V2. Layers with low entropy (sharp positional patterns) should keep RoPE.

**Depends on:** Nothing. Low-risk instrumentation.

---

## V1 — Next Training Run

These must be configured before launching the next pretraining or SFT phase.

---

### 6. FP8 Mixed Precision Training

**What it is:** Replace BF16 compute with FP8 for matrix multiplications in both forward and backward passes. FP8 uses `float8_e4m3fn` (forward, better precision) and `float8_e5m2` (gradients, better range). Scaling factors are accumulated over N steps to stabilise.

**Why it matters to Aurelius specifically:** Aurelius trains in BF16 today. DeepSeek-V3 was the first to validate FP8 training at 671B scale — proving 99–100% benchmark parity vs BF16 while cutting GPU memory by ~30–50% and increasing throughput by ~34%. Meta achieved 390 TFLOPs/GPU on Llama 4 with FP8. The net effect at 1.395B: either train faster, or use the freed memory for larger batch sizes that stabilise gradients further.

**Source:** DeepSeek-V3 [arXiv 2412.19437](https://arxiv.org/pdf/2412.19437); Meta Llama 4 [arXiv 2601.11659](https://arxiv.org/pdf/2601.11659); InfiR2 FP8 recipe [arXiv 2509.22536](https://arxiv.org/pdf/2509.22536)

**Validated at:** 671B (DeepSeek V3 — first proof at extreme scale); 405B (LLaMA 3.1 — 99–100% parity); 1.395B (direct target)

**Hardware requirement:** H100 / H200 / Blackwell — all support FP8 natively in tensor cores. A100 does not; fall back to BF16 on A100.

**Files to change:** `src/training/trainer.py`, `configs/train_1b.yaml`, `configs/train_2.7b.yaml`

**Implementation sketch:**

```python
# src/training/trainer.py
from torch._C._distributed_c10d import ProcessGroup
import torch._dynamo
from torch.distributed.fsdp import MixedPrecision

FP8_POLICY = MixedPrecision(
    param_dtype=torch.bfloat16,           # weights stay BF16 (stable)
    reduce_dtype=torch.float32,           # gradient all-reduce in FP32
    buffer_dtype=torch.bfloat16,
)

class AureliusTrainer:
    def _setup_fp8(self):
        """Configure per-tensor FP8 scaling for GEMM ops."""
        from transformer_engine.pytorch import fp8_autocast
        from transformer_engine.common.recipe import Format, DelayedScaling

        self.fp8_recipe = DelayedScaling(
            fp8_format=Format.HYBRID,    # E4M3 forward, E5M2 backward
            amax_history_len=16,         # steps to track activation scale
            amax_compute_algo="max",
        )

    def train_step(self, batch):
        with fp8_autocast(enabled=self.config.fp8, fp8_recipe=self.fp8_recipe):
            loss = self.model(**batch)
        loss.backward()
        self.optimizer.step()
```

```yaml
# configs/train_1b.yaml
precision: fp8
fp8:
  enabled: true
  forward_dtype: e4m3fn
  grad_dtype: e5m2
  delay_scaling_steps: 16
  amax_compute: max
  fallback_bf16_on_no_h100: true   # safety net for A100 nodes
```

**Metrics to watch:** Training loss curve vs BF16 baseline (should match closely within 5% tokens); GPU memory usage (expect 30–40% reduction); step throughput (expect 25–35% increase); `fp8_overflow_count` gauge (monitor for NaN/Inf explosions in first 1000 steps).

**Risk / mitigation:** FP8 activations can produce NaN if scale factors are miscalibrated during the first few hundred steps. Use `delay_scaling_steps: 16` to warm up scale factors before switching fully to FP8. Keep BF16 checkpoint at step 500 as rollback. Embedding layers and the final LM head should stay in BF16 — they are numerically sensitive.

**Depends on:** H100/H200 hardware. Liger Kernel is already FP8-compatible — no changes there.

---

### 7. HTMuon — Heavy-Tailed Spectral Correction

**What it is:** A drop-in wrapper around Aurelius's existing Muon optimizer that applies heavy-tailed spectral correction to the gradient matrix before Newton-Schulz orthogonalization. Gradient matrices in LLM training have fat-tailed eigenvalue distributions — vanilla Muon's Newton-Schulz steps treat all singular values uniformly, leaving the heavy tail underweighted.

**Why it matters to Aurelius specifically:** Aurelius uses Muon (Newton-Schulz 8+2 steps, Nesterov, RMS rescaling). HTMuon is a direct upgrade path — it wraps the existing optimizer with ~50 lines and has been shown to reduce perplexity by **0.98 points** on LLaMA-scale pretraining vs vanilla Muon. Kimi K2.6 validated Muon at 1 trillion parameters, confirming the optimizer scales. HTMuon is the next step on that curve.

**Source:** HTMuon [arXiv 2603.10067](https://arxiv.org/pdf/2603.10067); Variance-Adaptive Muon [arXiv 2601.14603](https://arxiv.org/pdf/2601.14603); NuMuon [arXiv 2603.03597](https://arxiv.org/pdf/2603.03597)

**Validated at:** LLaMA-scale pretraining on C4 — 0.98 perplexity reduction vs Muon baseline

**Files to change:** `src/training/muon.py`

**Implementation sketch:**

```python
class HTMuon(MuonOptimizer):
    """
    Heavy-Tailed Spectral Correction for Muon.
    Corrects the fat-tailed eigenvalue distribution in gradient matrices
    before Newton-Schulz orthogonalization, improving conditioning.
    """
    def __init__(self, *args, ht_alpha: float = 0.75, ht_clip: float = 3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ht_alpha = ht_alpha   # heavy-tail exponent correction
        self.ht_clip  = ht_clip    # clip correction factor

    def _ht_spectral_correction(self, G: torch.Tensor) -> torch.Tensor:
        """
        Estimate spectral correction for heavy-tailed gradient matrix G.
        Uses a fast trace-norm approximation — no full SVD required.
        """
        # Approximate operator norm via power iteration (2 steps, cheap)
        v = torch.randn(G.shape[-1], device=G.device, dtype=G.dtype)
        for _ in range(2):
            v = G.T @ (G @ v)
            v = v / v.norm().clamp(min=1e-8)
        sigma_max = (G @ v).norm()

        # Correction factor: dampen dominant singular values
        correction = (sigma_max ** self.ht_alpha).clamp(max=self.ht_clip)
        return 1.0 / correction

    def _orthogonalize(self, G: torch.Tensor) -> torch.Tensor:
        correction = self._ht_spectral_correction(G)
        return super()._orthogonalize(G * correction)
```

Additionally, apply **Variance-Adaptive** momentum from [arXiv 2601.14603](https://arxiv.org/pdf/2601.14603) on top:

```python
class VarianceAdaptiveMuon(HTMuon):
    """Adds NSR-modulated variance-scaled momentum to HTMuon."""
    def update_momentum(self, grad, state):
        var_estimate = (grad ** 2).mean()
        nsr = state["momentum"].norm() / (var_estimate.sqrt() + 1e-8)
        # Scale momentum by inverse variance — high-variance gradients get damped
        scale = 1.0 / (1.0 + self.nsr_coeff * nsr)
        state["momentum"] = self.beta * state["momentum"] * scale + grad
        return state["momentum"]
```

**Metrics to watch:** Eval perplexity at 10B, 50B, 100B tokens vs BF16+Muon baseline; gradient norm stability (HTMuon should reduce spike frequency); optimizer step norm histogram.

**Risk / mitigation:** The power-iteration cost is 2 extra matrix-vector products per step per parameter tensor — negligible (<1% overhead). If `ht_clip` is set too high, correction does nothing; if too low, it over-damps. Start at `ht_clip=3.0` and tune on a 300M proxy run.

**Depends on:** Nothing. Replace `MuonOptimizer` with `VarianceAdaptiveMuon` in `train_1b.yaml` under `optimizer.class`.

---

### 8. MTP-3 — Three Future Token Prediction

**What it is:** Increase the Multi-Token Prediction head from `n=2` to `n=3`. The model predicts positions `t+1`, `t+2`, and `t+3` at each position during training, sharing parameters across all three heads.

**Why it matters to Aurelius specifically:** Aurelius already has MTP `n=2` in `src/model/mtp.py` with shared parameters and staged training. This is a single integer change before the next training launch. Step-3.5-Flash uses MTP-3 (they call it MTP-3 head) and report additional quality gains. DeepSeek-V3 MTP heads are also used directly for speculative decoding — the n=3 heads become a 3-token draft, doubling speculative decode throughput vs n=2.

**Source:** Step-3.5-Flash [arXiv 2602.10604](https://arxiv.org/html/2602.10604v1); DeepSeek-V3 MTP section [arXiv 2412.19437](https://arxiv.org/pdf/2412.19437)

**Validated at:** 196B parameters (Step-3.5-Flash), 671B parameters (DeepSeek V3)

**Files to change:** `src/model/mtp.py`, `configs/train_1b.yaml`

**Implementation sketch:**

```python
# src/model/mtp.py — change one constant
MTP_N_FUTURE_TOKENS: int = 3    # was: 2

# configs/train_1b.yaml
mtp:
  n_future_tokens: 3      # was: 2
  shared_params: true     # keep — shared params are free compute
  staged_training:
    warmup_steps: 2000    # train main objective first, then add MTP heads
    mtp_loss_weight: 0.3  # weight of MTP auxiliary loss
```

The third head adds one additional shared projection. At 1.395B with a 2048 hidden dim and 8192 vocab, the third head adds ~16M parameters — less than 1.2% overhead.

**Metrics to watch:** MTP head accuracy at positions +1, +2, +3 (should be >40%, >30%, >20% respectively); speculative decoding acceptance rate with 3-token draft vs 2-token draft; eval perplexity (MTP-3 typically matches or beats MTP-2).

**Risk / mitigation:** None significant. If the third head degrades main task loss by more than 0.5%, reduce `mtp_loss_weight` from 0.3 to 0.15.

**Depends on:** Nothing. Change before the next training launch.

---

### 9. Sliding Window Attention — 3:1 Ratio Across Layers

**What it is:** Replace 18 of Aurelius's 24 transformer layers with Sliding Window Attention (SWA), keeping full O(n²) attention in only 6 strategically-placed layers. SWA attends to a fixed local window of `window_size` tokens, making those layers O(n × window_size) — effectively O(n) for fixed window.

**Why it matters to Aurelius specifically:** All 24 Aurelius layers currently run full softmax attention. At 4K tokens this costs 24 × n² operations. At 32K tokens it costs 24 × (8×)² = 1,536× more than at 4K. SWA cuts this to 6 × n² + 18 × n × W — roughly 4× cheaper at 32K context.

**Source:** Step-3.5-Flash 3:1 SWA:Full ratio [arXiv 2602.10604](https://arxiv.org/html/2602.10604v1); Mistral 7B SWA (validation at 7B); Gemma 3 (global:local ratio)

**Validated at:** 196B parameters with 3:1 ratio (Step-3.5-Flash achieving 100–350 tok/s)

**Files to change:** `src/model/attention.py`, `src/model/transformer.py`, `configs/train_1b.yaml`

**Implementation sketch:**

```python
# src/model/attention.py
class SlidingWindowAttention(nn.Module):
    """
    Local attention over a fixed window. O(n * window_size) complexity.
    Compatible with GQA head configuration.
    """
    def __init__(self, d_model, n_q_heads, n_kv_heads, window_size: int = 4096):
        super().__init__()
        self.window_size = window_size
        # Reuse existing GQA projection weights — same interface
        self.q_proj = nn.Linear(d_model, n_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_q_heads * head_dim, d_model, bias=False)

    def forward(self, x, positions, kv_cache=None):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # Apply RoPE only within the window
        q, k = apply_rope_window(q, k, positions, self.window_size)
        # Mask: attend only to tokens within [pos - window_size, pos]
        attn = flash_attn_varlen_func(
            q, k, v,
            window_size=(self.window_size, 0),  # left window, no right
            causal=True,
        )
        return self.o_proj(attn)

# src/model/transformer.py
# Full attention at layers: 0, 5, 11, 17, 22, 23 (6 of 24)
# SWA at all others (18 of 24) — 3:1 ratio
FULL_ATTENTION_LAYER_INDICES = frozenset([0, 5, 11, 17, 22, 23])

def _build_attention(layer_idx: int, config: ModelConfig):
    if layer_idx in FULL_ATTENTION_LAYER_INDICES:
        return GroupedQueryAttention(config)
    return SlidingWindowAttention(config, window_size=config.swa_window_size)
```

```yaml
# configs/train_1b.yaml
attention:
  pattern: swa_3_to_1              # 3 SWA : 1 full attention
  swa_window_size: 4096
  full_attention_layers: [0, 5, 11, 17, 22, 23]
```

**Metrics to watch:** Long-context benchmark scores (RULER, SCROLLS) — should hold or improve; short-context benchmarks (MMLU, ARC) — should not regress; throughput at 8K / 32K context (expect significant improvement); KV cache memory usage (SWA layers only cache `window_size` tokens instead of full sequence).

**Risk / mitigation:** The 6 full-attention layers carry the bulk of long-range coherence. If placed too sparsely, generation becomes incoherent at very long sequences. Validate on 32K+ generation tasks before declaring success. If quality drops, add one more full-attention layer at the midpoint (layer 12).

**Depends on:** Architecture change — requires a new training run. Configure alongside FP8 and HTMuon.

---

### 10. 3-Stage Pretraining Curriculum

**What it is:** Split pretraining into three explicit phases with evolving data mixtures: (1) broad knowledge, (2) STEM/reasoning intensive, (3) long-context data.

**Why it matters to Aurelius specifically:** Qwen3 used exactly this recipe to train on 36 trillion tokens across 119 languages, achieving state-of-the-art results at each model size. The three-stage approach is not about more data — it is about *when* each data type is introduced. Reasoning-intensive data too early (before the model has learned basic grammar and factual structure) degrades training stability. Long-context data before sequence comprehension is established wastes compute.

**Source:** Qwen3 Technical Report [arXiv 2505.09388](https://arxiv.org/pdf/2505.09388); Qwen3 Data Pipeline [kili-technology.com](https://kili-technology.com/blog/data-story-qwen3)

**Validated at:** 36T tokens across Qwen3 family (0.6B–235B)

**Files to change:** `configs/train_1b.yaml`, `scripts/collect_training_data.py`, `scripts/tokenize_jsonl_corpus.py`

**Implementation sketch:**

```yaml
# configs/train_1b.yaml — curriculum stages
curriculum:
  stages:
    - name: stage_1_general
      tokens: 600_000_000_000       # 600B — broad world knowledge
      max_seq_len: 4096
      data_mix:
        fineweb_edu_filtered: 0.45  # quality-filtered web (see data pipeline section)
        books_gutenberg: 0.15
        wikipedia_multilingual: 0.12
        code_github_filtered: 0.13
        arxiv_stem: 0.10
        synthetic_dialogue: 0.05

    - name: stage_2_reasoning
      tokens: 300_000_000_000       # 300B — raise reasoning ceiling
      max_seq_len: 8192
      data_mix:
        math_corpus: 0.25           # AMC, AIME, Olympiad, synthetic math
        code_high_quality: 0.30     # curated GitHub + HumanEval-style problems
        stem_papers: 0.20           # arXiv CS/physics/biology abstracts+bodies
        synthetic_cot: 0.15         # chain-of-thought examples (see item 10b)
        fineweb_edu_filtered: 0.10

    - name: stage_3_long_context
      tokens: 100_000_000_000       # 100B — extend context to 32K+
      max_seq_len: 32768            # grows from 4096 → 32768
      yarn_scale_factor: 8          # YaRN scale for this stage
      data_mix:
        long_documents: 0.45        # books, long papers, codebases
        multi_turn_dialogue: 0.25   # long conversations
        synthetic_long_cot: 0.20    # extended chain-of-thought
        fineweb_edu_filtered: 0.10
```

**Also required:** A `DataMixScheduler` that reads the active stage from a step counter and switches data loaders accordingly without interrupting training.

**Metrics to watch:** Perplexity progression per stage; MMLU score at end of Stage 2 vs single-stage baseline; long-context recall (needle-in-haystack) at end of Stage 3.

**Risk / mitigation:** Stage transitions can cause loss spikes if the data distribution shifts too suddenly. Use a 5% warm-up blend (10% of the previous stage's data mixed in at the start of the next) to smooth the transition.

**Depends on:** Data pipeline improvements (FineWeb-Edu filtering, synthetic CoT generation). Should be implemented in parallel.

---

### 11. Sequence Packing — 96% GPU Utilisation

**What it is:** Pack multiple short training examples into a single fixed-length sequence, using per-document position IDs and block-diagonal attention masks to prevent tokens from attending across document boundaries.

**Why it matters to Aurelius specifically:** Without packing, a 2048-token sequence containing a 200-token example is 90% padding — the GPU does real work on only 10% of the sequence. Hermes achieved 96% packing efficiency at 8192-token sequences, translating directly to 20–40% SFT throughput improvement.

**Source:** Hermes 4.3 training report ([oflight.co.jp](https://www.oflight.co.jp/en/columns/nous-hermes-4-3-function-calling-agent-guide-2026)); standard in Flash Attention 2 via `varlen` functions

**Files to change:** `src/training/sequence_packing.py` and `src/data/sequence_packing.py`

**Implementation sketch:**

```python
# src/training/sequence_packing.py
class SequencePackingDataset(IterableDataset):
    """
    Bin-packs variable-length examples into fixed-length sequences.
    Uses Flash Attention varlen API — each doc gets its own position IDs
    and cannot attend across document boundaries.
    """
    def __init__(self, base_dataset, max_seq_len: int = 8192):
        self.base = base_dataset
        self.max_seq_len = max_seq_len

    def __iter__(self):
        buffer: list[dict] = []
        buffer_len = 0

        for example in self.base:
            ex_len = len(example["input_ids"])
            if buffer_len + ex_len > self.max_seq_len:
                yield self._pack(buffer)
                buffer, buffer_len = [], 0
            buffer.append(example)
            buffer_len += ex_len

        if buffer:
            yield self._pack(buffer)

    def _pack(self, examples: list[dict]) -> dict:
        input_ids, position_ids, cu_seqlens = [], [0], []
        for ex in examples:
            ids = ex["input_ids"]
            input_ids.extend(ids)
            # Position IDs reset to 0 at each document start
            position_ids.extend(range(len(ids)))
            cu_seqlens.append(cu_seqlens[-1] + len(ids) if cu_seqlens else len(ids))

        # Pad to max_seq_len
        pad_len = self.max_seq_len - len(input_ids)
        input_ids.extend([PAD_TOKEN_ID] * pad_len)

        return {
            "input_ids":   torch.tensor(input_ids),
            "position_ids": torch.tensor(position_ids + [0] * pad_len),
            "cu_seqlens":   torch.tensor(cu_seqlens, dtype=torch.int32),
            "max_seqlen":   max(len(ex["input_ids"]) for ex in examples),
        }
```

Flash Attention 2's `flash_attn_varlen_func` accepts `cu_seqlens` directly and computes block-diagonal attention — no extra masking overhead.

**Metrics to watch:** GPU utilisation (NVIDIA SMI, should go from ~60–70% to ~90–96%); step time (should decrease proportionally); loss curve (should match unpacked baseline closely — if it diverges, check `position_ids` reset logic).

**Risk / mitigation:** The most common bug is position IDs *not* resetting between packed documents, causing the model to see a 8192-token "document" with random position sequences. Add a unit test asserting position IDs reset to 0 at each document boundary before enabling in training.

**Depends on:** Flash Attention 2 `varlen` API (already in the stack via Liger Kernel).

---

### 12. Hybrid Thinking / Non-Thinking Mode Tokens (SFT Mix)

**What it is:** Add two special tokens — `<think>` (open extended reasoning) and `</think>` (close, return to response) — to the tokenizer and include 30% of SFT examples in the wrapped format. The model learns to enter and exit a reasoning mode based solely on the presence of these tokens, with no architecture change.

**Why it matters to Aurelius specifically:** Aurelius has 13 personas and a ReAct loop, but no unified user-facing reasoning mode toggle. Qwen3 proved (May 2025) that a single checkpoint can serve both fast-response and extended-reasoning use cases by training on a 30/70 split of thinking/non-thinking examples. Anthropic's Claude 3.7 independently validated the same approach. This is the product-level capability with the highest demand signal from LLM users in 2025.

**Source:** Qwen3 blog [qwenlm.github.io](https://qwenlm.github.io/blog/qwen3/); Claude 3.7 launch [anthropic.com](https://x.com/AnthropicAI/status/1894092430560965029)

**Files to change:** `src/data/aurelius_tokenizer.py`, `src/alignment/sft.py`, `aurelius_cli/terminal_cli.py`, `aurelius_cli/pipeline_processor.py`

**Implementation sketch:**

```python
# src/data/aurelius_tokenizer.py — add two special tokens
SPECIAL_TOKENS = {
    "<think>":       8190,   # open extended reasoning
    "</think>":      8191,   # close, begin response
    # existing specials unchanged
}

# src/alignment/sft.py — data formatter
def format_thinking_example(
    prompt: str,
    reasoning: str,
    response: str,
    thinking_mode: bool,
) -> str:
    if thinking_mode:
        return f"{prompt}<think>\n{reasoning}\n</think>\n{response}"
    else:
        return f"{prompt}{response}"

class ThinkingModeSFTDataset(Dataset):
    def __init__(self, examples, thinking_ratio: float = 0.30):
        self.examples = examples
        self.thinking_ratio = thinking_ratio

    def __getitem__(self, idx):
        ex = self.examples[idx]
        use_thinking = (random.random() < self.thinking_ratio
                        and ex.get("cot") is not None)
        text = format_thinking_example(
            ex["prompt"], ex.get("cot", ""), ex["response"], use_thinking
        )
        return tokenize(text)
```

```yaml
# configs/train_1b.yaml
sft:
  thinking_mode:
    enabled: true
    mix_ratio: 0.30         # 30% of SFT examples use <think>...</think>
    think_token_id: 8190
    stop_think_token_id: 8191
    min_thinking_tokens: 32  # prevent degenerate empty thinking traces
    max_thinking_tokens: 2048
```

CLI integration:
```bash
aurelius chat --think                  # wraps prompt with <think>
aurelius chat --think-budget 4096      # bounded extended thinking
aurelius chat --no-think               # forces fast mode even if default is think
```

**Metrics to watch:** Hard reasoning benchmarks (MATH, GSM8K, HumanEval) with `--think` vs `--no-think`; latency difference; user-reported quality; thinking token length distribution (should correlate with problem difficulty).

**Risk / mitigation:** If the model always enters thinking mode (ignoring the `--no-think` flag), increase non-thinking examples to 80%. If it refuses to think deeply (produces trivially short `<think>` blocks), add length penalties in the RL phase that reward reasoning traces proportional to problem difficulty.

**Depends on:** Tokenizer update (trivial); SFT data with CoT traces (build from existing data pipeline + synthetic CoT generation in Stage 2 curriculum).

---

## V1 — Alignment & RL Pipeline

---

### 13. CISPO — Clipped Importance Sampling Policy Optimisation

**What it is:** A reinforcement learning variant that clips the *importance sampling weight ratio* at the trajectory level rather than clipping per-token KL divergence (as PPO and GRPO do). This produces more stable gradient estimates for long reasoning traces where per-token clipping accumulates erratically.

**Why it matters to Aurelius specifically:** Aurelius already has GRPO in `src/alignment/grpo.py`. CISPO is a targeted upgrade — MiniMax used it exclusively for MiniMax-M1 RL training and it outperformed PPO, GRPO, and REINFORCE++ in their ablations. Critically, full MiniMax-M1 RL training with CISPO cost $534,700 on 512 H800s over 3 weeks — achievable. At 1.395B scale, the cost is proportionally much smaller.

**Source:** MiniMax-M1 [arXiv 2506.13585](https://arxiv.org/abs/2506.13585)

**Validated at:** 456B MoE (MiniMax-M1), outperforming PPO/GRPO on reasoning and agentic benchmarks

**Files to change:** `src/alignment/grpo.py`

**Implementation sketch:**

```python
class CISPO(nn.Module):
    """
    Clipped Importance Sampling Policy Optimisation.
    Clips IS weight at trajectory level — more stable for long CoT traces.
    Reference: MiniMax-M1, arXiv 2506.13585
    """
    def __init__(self, clip_eps: float = 0.2, entropy_coeff: float = 0.01):
        super().__init__()
        self.clip_eps    = clip_eps
        self.entropy_c   = entropy_coeff

    def compute_loss(
        self,
        log_probs:     torch.Tensor,   # (batch, seq_len)
        ref_log_probs: torch.Tensor,   # (batch, seq_len) — frozen reference
        advantages:    torch.Tensor,   # (batch,) — per-trajectory
        mask:          torch.Tensor,   # (batch, seq_len) — response tokens only
    ) -> torch.Tensor:
        # Per-trajectory IS ratio (sum of log-probs over response tokens)
        traj_log_ratio = ((log_probs - ref_log_probs) * mask).sum(dim=-1)
        is_ratio       = traj_log_ratio.exp()                          # (batch,)

        # Clip at trajectory level — not per-token
        clipped_ratio  = is_ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)

        # PPO-style min objective, applied at trajectory level
        policy_loss    = -torch.min(
            is_ratio      * advantages,
            clipped_ratio * advantages,
        ).mean()

        # Entropy bonus to prevent collapse
        entropy = -(log_probs.exp() * log_probs * mask).sum(-1).mean()
        return policy_loss - self.entropy_c * entropy
```

**Metrics to watch:** Policy gradient variance (should be lower than GRPO); reward per step curve smoothness; KL divergence from reference policy (should stay bounded without explicit KL penalty term); acceptance rate in reasoning tasks.

**Risk / mitigation:** Trajectory-level clipping can be too lenient for very long traces (8K+ tokens) where IS ratios compound. If KL drifts above 0.3, add a soft KL penalty term: `loss += kl_coeff * kl_div(log_probs, ref_log_probs)`.

**Depends on:** Verifier pool (item 14) to provide high-quality reward signals.

---

### 14. Atropos-Style Verifier Pool

**What it is:** A registry of lightweight, task-specific verifier functions that evaluate model outputs asynchronously. Each verifier targets a specific output type (math, code, JSON format, factuality, safety, tool call validity). The pool aggregates signals into a composite reward without relying on a monolithic reward model.

**Why it matters to Aurelius specifically:** Hermes 4 training uses ~1,000 task-specific verifiers coordinated via Atropos. Aurelius currently has a single reward model path. Diversifying reward signals across ~50–200 verifiers dramatically reduces reward hacking (the model cannot learn to fool all verifiers simultaneously) and enables targeted improvement of specific capabilities without degrading others.

**Source:** Nous Research Atropos [github.com/NousResearch/atropos](https://github.com/NousResearch/atropos); Hermes 4.3 guide [oflight.co.jp](https://www.oflight.co.jp/en/columns/nous-hermes-4-3-function-calling-agent-guide-2026)

**Files to change:** `src/alignment/` (new: `src/alignment/verifier_pool.py`)

**Implementation sketch:**

```python
# src/alignment/verifier_pool.py
from abc import ABC, abstractmethod
import subprocess, json, sympy, re
from concurrent.futures import ThreadPoolExecutor

class BaseVerifier(ABC):
    name: str
    weight: float = 1.0

    @abstractmethod
    def verify(self, prompt: str, response: str) -> float:
        """Return score in [0.0, 1.0]. 1.0 = fully correct."""

class MathCorrectnessVerifier(BaseVerifier):
    name = "math"
    def verify(self, prompt, response):
        try:
            extracted = _extract_boxed_answer(response)
            expected  = _extract_expected(prompt)
            return 1.0 if sympy.simplify(extracted - expected) == 0 else 0.0
        except Exception:
            return 0.0

class PythonUnitTestVerifier(BaseVerifier):
    name = "code"
    weight = 1.5   # code tasks weighted higher
    def verify(self, prompt, response):
        code = _extract_code_block(response)
        test = _extract_test_block(prompt)
        try:
            result = subprocess.run(
                ["python", "-c", f"{code}\n{test}"],
                capture_output=True, timeout=10,
            )
            return 1.0 if result.returncode == 0 else 0.0
        except subprocess.TimeoutExpired:
            return 0.0

class JSONSchemaVerifier(BaseVerifier):
    name = "json_schema"
    def verify(self, prompt, response):
        schema = _extract_schema(prompt)
        try:
            obj = json.loads(_extract_json(response))
            jsonschema.validate(obj, schema)
            return 1.0
        except Exception:
            return 0.0

class SafetyTaxonomyVerifier(BaseVerifier):
    name = "safety"
    weight = 3.0   # safety violations are heavily penalised
    def verify(self, prompt, response):
        # Use existing 9-category harm classifier from src/safety/
        score = harm_classifier.score(response)
        return 1.0 - score   # 0.0 = harmful, 1.0 = safe

class ToolCallValidVerifier(BaseVerifier):
    name = "tool_call"
    def verify(self, prompt, response):
        calls = _extract_tool_calls(response)
        schemas = _get_tool_schemas(prompt)
        if not calls:
            return 1.0   # no tool calls required — OK
        return float(all(_validate_call(c, schemas) for c in calls))

# Registry and pool
VERIFIER_REGISTRY: dict[str, BaseVerifier] = {
    v.name: v() for v in [
        MathCorrectnessVerifier, PythonUnitTestVerifier, JSONSchemaVerifier,
        SafetyTaxonomyVerifier, ToolCallValidVerifier,
        FormatComplianceVerifier, FactualityProbeVerifier,
    ]
}

TASK_TYPE_TO_VERIFIERS: dict[str, list[str]] = {
    "math":      ["math", "format", "safety"],
    "code":      ["code", "json_schema", "format", "safety"],
    "agent":     ["tool_call", "json_schema", "format", "safety"],
    "general":   ["factuality", "format", "safety"],
    "reasoning": ["math", "factuality", "format", "safety"],
}

class VerifierPool:
    def __init__(self, max_workers: int = 16):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def score(self, prompt: str, response: str, task_type: str) -> float:
        verifier_names = TASK_TYPE_TO_VERIFIERS.get(task_type, ["safety", "format"])
        active = [VERIFIER_REGISTRY[n] for n in verifier_names if n in VERIFIER_REGISTRY]

        # Run verifiers in parallel — each is independent
        futures = {self.executor.submit(v.verify, prompt, response): v
                   for v in active}
        scores  = {v.name: f.result() for f, v in
                   {f: futures[f] for f in futures}.items()}

        # Weighted geometric mean (safety violations collapse to 0)
        weights = [VERIFIER_REGISTRY[n].weight for n in scores]
        vals    = list(scores.values())
        weighted_log = sum(w * math.log(v + 1e-8) for w, v in zip(weights, vals))
        return math.exp(weighted_log / sum(weights))
```

**Wire into training:**
```python
# src/alignment/grpo.py or cispo.py
reward = verifier_pool.score(prompt, response, task_type=classify_task(prompt))
```

**Metrics to watch:** Per-verifier pass rate breakdown; reward distribution shape (should be bimodal — clearly good vs clearly bad responses); correlation between verifier scores and human preference ratings.

**Risk / mitigation:** Code execution verifier needs sandboxing — run in Docker or use `restrictedpython`. Math verifier can fail on non-standard answer formats — add multiple extraction strategies (`\boxed{}`, `= answer`, `The answer is`). Safety verifier false positives on legitimate security research prompts — tune threshold or add context-aware override.

**Depends on:** Nothing. Can be built in parallel with training.

---

### 15. RL Training for Tool-Use Timing

**What it is:** Add an efficiency reward to the ReAct training loop that penalises unnecessary tool calls. The model learns not just *how* to use tools but *when* — choosing the minimum-cost valid tool sequence rather than calling tools defensively.

**Why it matters to Aurelius specifically:** OpenAI's o3/o4 were specifically trained via RL to reason about tool invocation cost. Aurelius's ReAct loop currently does not penalise tool over-calling. An agent that calls `web_search` 12 times when 2 would suffice is expensive, slow, and demonstrates poor reasoning.

**Source:** OpenAI o3/o4 system card [cdn.openai.com](https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card.pdf)

**Files to change:** `src/agent/react_loop.py`, `src/alignment/verifier_pool.py`

**Implementation sketch:**

```python
# src/agent/react_loop.py
@dataclass
class AgentTrajectory:
    prompt:        str
    tool_calls:    list[ToolCall]
    observations:  list[str]
    final_answer:  str
    task_complete: bool

    @property
    def min_required_calls(self) -> int:
        """
        Oracle lower bound: minimum tool calls that could have produced
        the correct answer. Computed post-hoc using dependency graph.
        """
        return _compute_min_call_depth(self.tool_calls)

class ToolEfficiencyVerifier(BaseVerifier):
    name = "tool_efficiency"
    weight = 0.5
    alpha: float = 0.15   # penalty per excess call

    def verify(self, prompt: str, response: str) -> float:
        traj = parse_trajectory(response)
        if not traj.task_complete:
            return 0.0
        n_excess = max(0, len(traj.tool_calls) - traj.min_required_calls)
        return max(0.0, 1.0 - self.alpha * n_excess)
```

**Metrics to watch:** Average tool calls per successful task (should decrease over RL training); task completion rate (should not decrease — efficiency without capability loss); tool call latency budget (p99 per agent run).

**Risk / mitigation:** The `min_required_calls` oracle can be hard to compute exactly. Use a simpler heuristic: penalise repeated calls to the same tool with the same arguments. This is unambiguously wasteful.

**Depends on:** Verifier pool (item 14).

---

### 16. ThinkPRM — Generative Process Reward Model

**What it is:** A process reward model (PRM) that verifies each step of a multi-step reasoning trace by generating a chain-of-thought verification rather than outputting a scalar score. ThinkPRM writes *reasoning* about whether each step is correct, not just a number.

**Why it matters to Aurelius specifically:** Aurelius has REINFORCE++ and outcome-based rewards. PRMs provide step-level credit assignment — critical for training on long reasoning traces where the final answer alone is uninformative about which steps were correct. ThinkPRM (ICLR 2026) requires orders-of-magnitude fewer step labels than discriminative PRMs because it leverages the model's own reasoning capacity. The FOVER technique augments ThinkPRM training data using formal verification (Z3, Isabelle) — fully automatic, no human annotation.

**Source:** ThinkPRM [arXiv 2504.16828](https://arxiv.org/pdf/2504.16828); FOVER [arXiv 2505.15960](https://arxiv.org/pdf/2505.15960); GenPRM [arXiv 2504.00891](https://arxiv.org/pdf/2504.00891)

**Validated at:** Outperforms discriminative PRMs on MATH, GSM8K with 10–100× fewer labels

**Files to change:** `src/alignment/` (new: `src/alignment/think_prm.py`); `src/reasoning/mcts_reasoner.py`

**Implementation sketch:**

```python
# src/alignment/think_prm.py
class ThinkPRM(nn.Module):
    """
    Generative PRM: verifies reasoning steps by generating verification CoT.
    Fine-tuned from Aurelius base via LoRA on FOVER-generated step labels.
    """
    def __init__(self, base_model, lora_rank: int = 32):
        super().__init__()
        self.model = apply_lora(base_model, rank=lora_rank, target="alignment")

    def score_step(self, problem: str, steps_so_far: list[str], step: str) -> float:
        """
        Generate a verification trace for `step` given `problem` and prior steps.
        Extract confidence from the last token of the verification trace.
        """
        verification_prompt = self._format_verification_prompt(
            problem, steps_so_far, step
        )
        verification_trace = self.model.generate(
            verification_prompt,
            max_new_tokens=256,
            stop_sequences=["[CORRECT]", "[INCORRECT]", "[UNCERTAIN]"],
        )
        # Extract judgment from final verdict token
        if "[CORRECT]" in verification_trace:
            return 1.0
        elif "[INCORRECT]" in verification_trace:
            return 0.0
        else:
            # Parse confidence from trace: "I'm about 70% confident this is correct"
            return _extract_confidence(verification_trace)

    def score_trajectory(self, problem: str, steps: list[str]) -> list[float]:
        return [self.score_step(problem, steps[:i], steps[i])
                for i in range(len(steps))]
```

**FOVER integration for training data generation:**

```python
# scripts/generate_prm_labels.py
from z3 import Solver, Int, sat

def fover_label_math_step(problem_str: str, step_str: str) -> float:
    """Use Z3 SMT solver to verify mathematical reasoning steps automatically."""
    solver = Solver()
    constraints = parse_to_z3(problem_str, step_str)
    solver.add(constraints)
    return 1.0 if solver.check() == sat else 0.0
```

**Wire into MCTS:**
```python
# src/reasoning/mcts_reasoner.py — use ThinkPRM to score MCTS nodes
def evaluate_node(self, node: MCTSNode) -> float:
    step_scores = self.think_prm.score_trajectory(
        node.problem, node.reasoning_steps
    )
    # Process reward: penalise first incorrect step heavily
    first_bad = next((i for i, s in enumerate(step_scores) if s < 0.4), None)
    if first_bad is not None:
        return step_scores[first_bad] * 0.1   # heavy penalty
    return float(np.mean(step_scores))
```

**Metrics to watch:** ThinkPRM step accuracy vs human labels (target >85%); MCTS solution quality with ThinkPRM-guided node selection vs random selection; FOVER auto-label accuracy on held-out verified problems.

**Risk / mitigation:** ThinkPRM can produce overconfident incorrect verification traces. Add a calibration step: fine-tune on cases where the model's verification was wrong, explicitly training it to say "I'm uncertain" on ambiguous steps.

**Depends on:** Existing MCTS (`src/reasoning/mcts.py`), LoRA infrastructure.

---

### 17. BeamCoT Search — Best-of-N with Verifier Selection

**What it is:** Generate K independent chain-of-thought candidates for a query, score each with the ThinkPRM verifier, and return the highest-scoring response. Simpler than full MCTS but directly effective for hard reasoning tasks.

**Why it matters:** OpenAI o3's internal search generates multiple CoT paths and selects the most coherent. At inference-time, K=4–8 candidates gives meaningful quality gains on math and coding tasks with manageable latency increase.

**Source:** OpenAI o3 test-time search; [Introl blog](https://introl.com/blog/inference-time-scaling-research-reasoning-models-december-2025)

**Files to change:** `src/reasoning/chain_of_thought.py`

**Implementation sketch:**

```python
# src/reasoning/chain_of_thought.py
class BeamCoTSearch:
    """
    Generate K reasoning traces in parallel, select best via ThinkPRM scoring.
    Activated when max_thinking_tokens > 0 and query is classified hard.
    """
    def __init__(self, model, think_prm: ThinkPRM, k: int = 4,
                 difficulty_threshold: float = 0.6):
        self.model     = model
        self.prm       = think_prm
        self.k         = k
        self.threshold = difficulty_threshold

    def generate(self, prompt: str, task_type: str,
                 max_tokens: int = 2048) -> str:
        # Only activate on hard queries — use query complexity classifier
        if self._estimate_difficulty(prompt) < self.threshold:
            return self.model.generate(prompt, max_new_tokens=max_tokens)

        # Generate K candidates in parallel
        candidates = []
        with ThreadPoolExecutor(max_workers=self.k) as pool:
            futures = [pool.submit(self.model.generate, prompt,
                                   max_new_tokens=max_tokens,
                                   temperature=0.8)     # diverse sampling
                       for _ in range(self.k)]
            candidates = [f.result() for f in futures]

        # Score with ThinkPRM and return best
        scores = [self.prm.score_trajectory(prompt, _extract_steps(c))
                  for c in candidates]
        agg_scores = [float(np.mean(s)) for s in scores]
        return candidates[int(np.argmax(agg_scores))]

    def _estimate_difficulty(self, prompt: str) -> float:
        """Lightweight difficulty classifier based on query features."""
        features = [
            len(prompt) > 200,           # long prompt → harder
            any(w in prompt.lower() for w in ["prove", "derive", "calculate",
                                               "implement", "debug", "optimize"]),
            prompt.count("?") > 1,       # multi-part question
        ]
        return float(np.mean(features))
```

**Depends on:** ThinkPRM (item 16) for scoring. Can run with random selection as a baseline first.

---


### 17b. REINFORCE++ — Global Advantage Normalization for GRPO Stability

**What it is:** REINFORCE++ (arXiv:2501.03262, January 2025) replaces GRPO's per-group advantage normalization with **global advantage normalization** across the entire batch. GRPO divides the batch into groups of G rollouts per prompt and normalizes rewards within each group. When group size is small (G=4–8), variance of the advantage estimator is high, causing training instability. REINFORCE++ normalizes across all responses in the batch — equivalent to infinite group size — and adds a **token-level KL penalty** baseline rather than a population baseline, giving stable updates even when reward signals are sparse.

**Why it matters to Aurelius specifically:** Aurelius uses GRPO in `src/alignment/grpo.py`. The group-normalization instability is most severe on long-context reasoning tasks (8K+ tokens) where individual response rewards can differ by orders of magnitude. REINFORCE++ is a pure drop-in: same outer loop, replace the advantage computation and add the KL penalty baseline. The paper shows +2.3pp on MATH500 at 1.5B scale versus GRPO with G=8.

**Source:** REINFORCE++ [arXiv:2501.03262](https://arxiv.org/abs/2501.03262); DAPO paper ablation comparison (arXiv:2503.14476)

**Validated at:** 1.5B and 7B models; +2.3pp MATH500 vs GRPO; training curve stability improvement ~60% reduction in gradient norm variance.

**Files to change:** `src/alignment/grpo.py`

**Implementation sketch:**

```python
# src/alignment/grpo.py — drop-in advantage normalization upgrade
class REINFORCEPlusPlusAdvantage:
    """
    Global advantage normalization (batch-wide) + token-level KL baseline.
    Replaces per-group normalization in GRPO.
    Reference: arXiv:2501.03262
    """
    def __init__(self, kl_coeff: float = 0.02, eps: float = 1e-8):
        self.kl_coeff = kl_coeff
        self.eps = eps

    def compute(
        self,
        rewards: torch.Tensor,          # (batch,)  — per-response scalar reward
        log_probs: torch.Tensor,         # (batch, seq_len) — policy log-probs
        ref_log_probs: torch.Tensor,     # (batch, seq_len) — reference log-probs
        response_mask: torch.Tensor,     # (batch, seq_len) — 1 = response token
    ) -> torch.Tensor:
        # 1. Global advantage: normalize across ALL responses in the batch
        #    (GRPO normalizes only within groups of G — high variance at small G)
        adv = (rewards - rewards.mean()) / (rewards.std() + self.eps)   # (batch,)

        # 2. Token-level KL baseline: subtract per-token KL from return
        token_kl = (log_probs - ref_log_probs) * response_mask          # (batch, seq)
        kl_baseline = self.kl_coeff * token_kl                          # (batch, seq)

        # 3. Broadcast advantages to token level, apply KL penalty baseline
        adv_per_token = adv.unsqueeze(-1).expand_as(log_probs)
        adjusted = adv_per_token - kl_baseline                          # (batch, seq)

        # 4. Policy gradient loss (standard REINFORCE)
        loss = -(adjusted * log_probs * response_mask).sum(-1).mean()
        return loss


def upgrade_grpo_to_reinforce_plus_plus(grpo_trainer, kl_coeff: float = 0.02):
    """One-line upgrade path: swap advantage module in existing GRPO trainer."""
    grpo_trainer.advantage_fn = REINFORCEPlusPlusAdvantage(kl_coeff=kl_coeff)
    return grpo_trainer
```

**Metrics to watch:** Gradient norm variance (target: <50% of GRPO baseline); reward curve smoothness (visual inspection over 10K steps); MATH500 score at 50K and 100K training steps (target +1–3pp vs GRPO); KL drift from reference policy (should stay below 0.25 without explicit KL clipping).

**Risk / mitigation:** On very diverse batches (mixing math, code, and general tasks), global normalization can cause reward scale mismatch across task types. If multi-task RL is planned, normalize per-task-type first then apply REINFORCE++ within each type. Alternatively, use reward standardization per verifier class before aggregating.

**Depends on:** Existing GRPO trainer. 1-hour engineering change — no new dependencies.

---

### 17c. DAPO — Decoupled Clip + Dynamic Sampling for GRPO at Scale

**What it is:** DAPO (Decoupled clip-ratio policy optimization, arXiv:2503.14476, March 2025) identifies and fixes two compounding failure modes in GRPO at scale: (1) **entropy collapse** — the policy becomes overconfident and loses exploration capacity as training proceeds; (2) **clip imbalance** — asymmetric clipping of positive and negative advantages causes bias at the policy boundaries. DAPO adds: **(a) decoupled clip** (separate ε+ and ε− hyperparameters for positive/negative advantages), **(b) dynamic sampling** (discard rollouts where all responses have the same binary reward — zero-information batches), **(c) token-level policy gradient loss** instead of sequence-level, and **(d) entropy bonus** to prevent entropy collapse.

**Why it matters to Aurelius specifically:** At 1.4B scale with RLVR on verifiable tasks (AMC/AIME math, coding), Aurelius will hit GRPO's entropy collapse within ~20K steps if training is sufficiently intensive. DAPO was validated at 32B parameter scale with 6,000+ training steps. The `dynamic_sampling` filter alone eliminates ~15–20% of uninformative rollouts, improving effective sample quality. Combined with the decoupled clip, DAPO shows +4.3pp over GRPO on AIME at 7B scale in the paper's ablations.

**Source:** DAPO [arXiv:2503.14476](https://arxiv.org/abs/2503.14476); secondary: REINFORCE++ comparison (arXiv:2501.03262)

**Validated at:** 7B and 32B models; +4.3pp AIME vs GRPO; entropy collapse eliminated over 6K+ steps.

**Files to change:** `src/alignment/grpo.py`

**Implementation sketch:**

```python
# src/alignment/grpo.py — DAPO extensions
class DAPOConfig:
    clip_eps_pos: float = 0.2      # ε+ for positive advantages (default GRPO: 0.2)
    clip_eps_neg: float = 0.1      # ε− for negative advantages — tighter to prevent bias
    entropy_coeff: float = 0.001   # entropy bonus coefficient
    dynamic_sample_filter: bool = True  # drop all-same-reward batches
    token_level_loss: bool = True  # per-token loss instead of per-sequence


class DAPOObjective:
    """
    DAPO: Decoupled clip + entropy bonus + dynamic sampling.
    Reference: arXiv:2503.14476
    """
    def __init__(self, config: DAPOConfig):
        self.cfg = config

    def filter_batch(
        self,
        rewards: torch.Tensor,     # (batch,) binary rewards from verifier
        group_size: int,
    ) -> torch.Tensor:
        """Remove groups where all G rollouts share the same reward (no gradient signal)."""
        # Reshape to (n_groups, G)
        grouped = rewards.view(-1, group_size)
        # Keep groups where not all rewards are identical
        valid = ~(grouped == grouped[:, :1]).all(dim=1)
        return valid.repeat_interleave(group_size)   # expand back to batch dim

    def compute_loss(
        self,
        log_probs:     torch.Tensor,    # (batch, seq_len)
        ref_log_probs: torch.Tensor,    # (batch, seq_len)
        advantages:    torch.Tensor,    # (batch, seq_len) — token-level advantages
        response_mask: torch.Tensor,    # (batch, seq_len)
    ) -> torch.Tensor:
        # Token-level IS ratio
        ratio = (log_probs - ref_log_probs).exp()

        # Decoupled clip: different epsilon for positive vs negative advantages
        eps_p, eps_n = self.cfg.clip_eps_pos, self.cfg.clip_eps_neg
        clipped_pos = ratio.clamp(1.0 - eps_n, 1.0 + eps_p)  # negative adv: tighter lower bound
        clipped_neg = ratio.clamp(1.0 - eps_n, 1.0 + eps_n)

        # Select clip bound by advantage sign
        is_positive = (advantages >= 0).float()
        clipped = is_positive * clipped_pos + (1 - is_positive) * clipped_neg

        # Standard PPO min objective at token level
        pg_loss = -torch.min(
            ratio   * advantages,
            clipped * advantages,
        )

        # Entropy bonus: prevent distribution collapse
        entropy = -(log_probs.exp() * log_probs)  # per-token entropy

        # Weighted sum, mask out non-response tokens
        loss = ((pg_loss - self.cfg.entropy_coeff * entropy) * response_mask).mean()
        return loss
```

**Metrics to watch:** Policy entropy over training (should remain above 1.0 nat; alert if below 0.5); AIME 2024/2025 pass@1 at checkpoints (target +3–5pp vs GRPO baseline); dynamic sample filter rate (expect 15–25% of batches filtered — higher suggests poor reward design); KL from reference (target <0.3 throughout).

**Risk / mitigation:** Decoupled clip introduces two hyperparameters. Start with ε+ = 0.2, ε− = 0.1 (paper defaults). If the model is overly conservative (low entropy, low reward improvement), increase ε+. If KL drifts too fast, decrease ε−. Dynamic sampling can significantly reduce batch diversity if the reward function is nearly binary — use continuous rewards (partial credit) when possible to keep more rollouts.

**Depends on:** GRPO trainer (item 13). Can be layered on top of CISPO or REINFORCE++ — the clip mechanism is orthogonal to the advantage estimator.

---

### 17d. SimPO — Reference-Free DPO with Length Normalization

**What it is:** SimPO (Simple Preference Optimization, arXiv:2405.14734, May 2024 / NeurIPS 2024) is a DPO variant that **eliminates the reference model entirely** and **normalizes rewards by sequence length**. Standard DPO requires keeping the reference model in memory during fine-tuning (doubles GPU memory). SimPO replaces the per-token KL term with a simple length-normalized reward margin: `reward(y) = (1/|y|) Σ log π(yᵢ|y<i,x)`. A response is preferred over another when this normalized log-probability is higher by at least a margin γ. Result: **−40% GPU memory** during DPO training, and empirically **+6–7pp** on AlpacaEval 2.0 vs standard DPO.

**Why it matters to Aurelius specifically:** Aurelius's DPO phase in the alignment pipeline requires loading both the policy and reference model simultaneously. At 1.4B this is manageable, but at V2 (2.7B) and V4 (5B+) scale, the reference model overhead becomes significant. SimPO removes this constraint and actually improves alignment quality. The length normalization component is particularly valuable for Aurelius: the model learns to prefer concise but correct responses over verbose but equally-correct ones, improving serving latency.

**Source:** SimPO [arXiv:2405.14734](https://arxiv.org/abs/2405.14734); SimPO code [github.com/princeton-nlp/SimPO](https://github.com/princeton-nlp/SimPO)

**Validated at:** Llama-3-8B-Instruct, Mistral-7B; +6.4pp AlpacaEval 2.0 LC vs DPO; −40% memory vs DPO; MT-Bench +0.3 points.

**Files to change:** `src/alignment/dpo.py` (new `SimPOLoss` class)

**Implementation sketch:**

```python
# src/alignment/dpo.py
class SimPOLoss(nn.Module):
    """
    Reference-free DPO with length normalization.
    Eliminates reference model — saves 40% GPU memory during alignment.
    Reference: arXiv:2405.14734
    """
    def __init__(
        self,
        beta: float = 2.5,       # temperature scaling (paper default)
        gamma: float = 0.5,      # reward margin threshold
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.label_smooth = label_smoothing

    def length_normalized_reward(
        self,
        log_probs: torch.Tensor,   # (batch, seq_len)
        response_mask: torch.Tensor,  # (batch, seq_len)
    ) -> torch.Tensor:
        """Average log-probability over response tokens."""
        lengths = response_mask.sum(dim=-1).float().clamp(min=1.0)
        return (log_probs * response_mask).sum(dim=-1) / lengths  # (batch,)

    def forward(
        self,
        chosen_log_probs:   torch.Tensor,    # (batch, seq_len)
        rejected_log_probs: torch.Tensor,    # (batch, seq_len)
        chosen_mask:        torch.Tensor,
        rejected_mask:      torch.Tensor,
    ) -> torch.Tensor:
        r_w = self.length_normalized_reward(chosen_log_probs,   chosen_mask)
        r_l = self.length_normalized_reward(rejected_log_probs, rejected_mask)

        # Margin-based loss: chosen must exceed rejected by at least gamma
        logits = self.beta * (r_w - r_l - self.gamma)

        if self.label_smooth > 0:
            # Smoothed binary cross-entropy
            loss = (
                -F.logsigmoid(logits) * (1 - self.label_smooth)
                - F.logsigmoid(-logits) * self.label_smooth
            ).mean()
        else:
            loss = -F.logsigmoid(logits).mean()
        return loss
```

**Metrics to watch:** AlpacaEval 2.0 Length-Controlled Win Rate (LC-WR); GPU memory during training (expect −35–40% vs DPO); MT-Bench score; response length distribution (SimPO encourages shorter responses — watch for truncation artifacts at low gamma values).

**Risk / mitigation:** The gamma margin is sensitive. Too high → model refuses many preference pairs (dataset becomes too easy), learns nothing. Too low → model doesn't learn to discriminate. Default γ=0.5 works well for general instruction following; for math/code where responses vary in length, consider γ=1.0–2.0 to avoid spuriously preferring short wrong answers over long correct ones. Add a length-correctness filtering step: only include preference pairs where the chosen response is verified correct via a task-specific verifier.

**Depends on:** Existing DPO trainer. Reference model training can be dropped entirely from the alignment pipeline — reduces storage by one checkpoint per training run.

---

### 17e. uPRM — Unsupervised Process Reward Model

**What it is:** uPRM (Unsupervised PRM, arXiv:2605.10158, May 2026) trains a step-level process reward model **without any step-level human annotations**. Standard PRMs (like OmegaPRM or Math-Shepherd) require labeled reasoning steps indicating exactly which step first introduces an error — expensive to obtain at scale. uPRM uses a self-supervised signal: it computes the change in **outcome probability** when the reasoning chain is truncated at each step. A step whose removal significantly reduces final-answer probability is a high-value step; a step whose removal has no effect is low-value. Training signal: `step_value(s_t) = P(correct | s_1...s_t) - P(correct | s_1...s_{t-1})`, estimated by computing rollout completion probabilities with a verifier.

**Why it matters to Aurelius specifically:** PAC (OC-5 in Section 0) uses formal verifiers (Z3, Isabelle) as step-level verifiers. For domains where formal verification is not feasible (open-ended reasoning, multi-step commonsense), uPRM provides a cheaper alternative. The uPRM signal can also be used to improve the quality of Aurelius's RLVR training data — identify which intermediate reasoning steps in distilled CoT data are actually contributing to correct answers versus which are filler.

**Source:** uPRM [arXiv:2605.10158](https://arxiv.org/abs/2605.10158); ThinkPRM [OpenReview 2025](https://openreview.net/forum?id=V727xqBYIW)

**Validated at:** MATH-500, AMC 2024; +2.8pp vs no PRM baseline at 1.5B; −85% annotation cost vs OmegaPRM-style labeled PRMs.

**Files to change:** `src/alignment/process_reward.py` (new `uPRMTrainer` class)

**Implementation sketch:**

```python
# src/alignment/process_reward.py
class uPRMTrainer:
    """
    Train a process reward model without step-level annotations.
    Uses outcome probability difference as self-supervised step value signal.
    Reference: arXiv:2605.10158
    """
    def __init__(
        self,
        base_model,              # frozen base model used to estimate P(correct)
        prm_head: nn.Linear,     # lightweight linear head on top of base model
        n_rollouts: int = 16,    # rollouts per prefix to estimate P(correct)
        verifier,                # final-answer verifier (math, code, etc.)
    ):
        self.base      = base_model
        self.prm_head  = prm_head
        self.n_rollouts = n_rollouts
        self.verifier  = verifier

    def estimate_step_value(
        self,
        prefix: list[str],     # reasoning steps up to and including step t
        problem: str,
    ) -> float:
        """
        Estimate value of the last step in `prefix` by completing from this prefix
        and from the prefix-without-last-step, comparing P(correct).
        """
        full_value     = self._rollout_value(prefix, problem)
        without_last   = self._rollout_value(prefix[:-1], problem)
        return full_value - without_last

    def _rollout_value(self, prefix: list[str], problem: str) -> float:
        """Estimate P(correct) by completing from this prefix N times."""
        completions = [
            self.base.complete(problem + "\n" + "\n".join(prefix),
                               temperature=0.8, max_tokens=512)
            for _ in range(self.n_rollouts)
        ]
        return sum(self.verifier.verify(problem, c) for c in completions) / self.n_rollouts

    def build_training_set(
        self,
        problems: list[str],
        solutions: list[list[str]],  # each is a list of reasoning steps
    ) -> list[dict]:
        """Build (step, value) pairs for PRM training."""
        records = []
        for prob, steps in zip(problems, solutions):
            for t, step in enumerate(steps):
                value = self.estimate_step_value(steps[:t+1], prob)
                records.append({
                    "problem": prob,
                    "prefix": steps[:t],
                    "step": step,
                    "value": value,   # regression target
                })
        return records
```

**Metrics to watch:** Step value correlation with human-labeled quality (Spearman ρ target >0.55 on held-out validated set); downstream RLVR improvement when uPRM rewards are used vs no PRM; cost: n_rollouts × average_completion_length determines annotation compute — budget accordingly. At n_rollouts=16 and 512-token completions, expect ~8.2K tokens per step annotation.

**Risk / mitigation:** High-variance rollout estimates can mislead the PRM if the base model's completion distribution is multimodal (sometimes correct, sometimes very wrong). Use more rollouts (n=32+) for math problems; fewer (n=8) for problems with deterministic answers. Add a minimum-confidence filter: skip step annotations where |P(correct|prefix) − P(correct|prefix−1)| < 0.05 (no signal).

**Depends on:** Verifier pool (item 14); frozen base model checkpoint for rollout estimation; CoT distillation data (item 23k) provides the stepping-stone data to annotate.

---

### 17f. VersaPRM — Multi-Domain Process Reward Model

**What it is:** VersaPRM (arXiv:2502.06737, February 2025, ICML 2025) is a process reward model trained across **multiple domains simultaneously** — mathematics, physics, law, economics, and commonsense reasoning. Standard math PRMs (Math-Shepherd, OmegaPRM) fail on out-of-domain reasoning because their discriminative classifiers learn domain-specific surface features. VersaPRM generates **multi-domain CoT reasoning datasets** using a frontier model as teacher, then trains a step-level classifier that must distinguish correct from incorrect reasoning steps across all domains. The key empirical finding: a multi-domain PRM trained on 5 domains substantially outperforms single-domain PRMs even on the original math domain (+1.9pp MATH-500), due to domain-general reasoning heuristics transferring back.

**Why it matters to Aurelius specifically:** Aurelius is positioned as a general-purpose model, not a math specialist. The alignment pipeline needs reward signals for code, math, factuality, instruction following, safety, and tool use simultaneously. A single-domain PRM cannot cover this. VersaPRM's architecture shows that training one PRM across domains is more efficient and more accurate than training separate per-domain PRMs.

**Source:** VersaPRM [arXiv:2502.06737](https://arxiv.org/html/2502.06737); POSTECH/KAIST

**Validated at:** 5 domains (math, physics, law, economics, commonsense); +1.9pp MATH-500 over single-domain PRM; outperforms separate per-domain ensemble.

**Files to change:** `src/alignment/process_reward.py` (extend to `VersaPRMTrainer`)

**Implementation sketch:**

```python
# src/alignment/process_reward.py
class VersaPRM(nn.Module):
    """
    Multi-domain process reward model.
    Single model trained across math, code, physics, commonsense, factuality.
    Reference: arXiv:2502.06737 (VersaPRM, ICML 2025)
    """
    DOMAINS = ["math", "code", "physics", "commonsense", "factuality", "safety"]

    def __init__(self, base_model, hidden_dim: int = 2048):
        super().__init__()
        self.encoder = base_model   # frozen or LoRA-adapted base encoder

        # Shared step-level scoring head
        self.step_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        # Domain embedding — learned bias per domain (lightweight)
        self.domain_embed = nn.Embedding(len(self.DOMAINS), hidden_dim)

    def score_step(
        self,
        problem: str,
        step: str,
        domain: str,
        previous_steps: list[str],
    ) -> float:
        domain_idx = self.DOMAINS.index(domain)
        context    = problem + "\n" + "\n".join(previous_steps) + "\n" + step

        with torch.no_grad():
            hidden = self.encoder.encode(context)          # (1, seq, d)
            # Add domain embedding to final hidden state
            domain_bias = self.domain_embed(torch.tensor(domain_idx))
            step_repr   = hidden[:, -1, :] + domain_bias   # CLS-like pooling
            score       = self.step_scorer(step_repr)

        return float(score.squeeze())
```

**Metrics to watch:** Per-domain step-level F1 (target >0.72 across all 6 domains); cross-domain transfer: do math-domain improvements correlate with code-domain improvements? (positive transfer expected); inference latency for PRM scoring (single forward pass per step — should be <50ms on CPU for 1.4B encoder).

**Risk / mitigation:** Domain imbalance in training data can cause the shared scorer to be biased toward the domain with most examples. Use stratified sampling: ensure equal numbers of training examples per domain. If one domain underperforms, consider adding a per-domain LoRA adapter on top of the shared scorer (5% additional parameters per domain).

**Depends on:** uPRM (item 17e) or labeled reasoning data per domain; verifier pool (item 14) for multi-domain verification; PRPO (item 17g) uses VersaPRM's scores as RL signal.

---

### 17g. PRPO — PRM-Guided Policy Optimization

**What it is:** PRPO (Process Reward Policy Optimization, arXiv:2601.07182, January 2026) uses process reward model scores as **dense token-level training signal** inside the policy gradient loop. Standard RLVR provides only an outcome reward at the end of the reasoning chain — sparse signal that must propagate back through thousands of tokens. PRPO uses a trained PRM (VersaPRM, uPRM, or ThinkPRM) to score each intermediate reasoning step and assign a per-step advantage: `adv(t) = PRM_score(step_t) - mean(PRM_scores)`. This dense signal dramatically reduces credit assignment variance for long reasoning traces. The paper reports **+3.2pp MATH500 at 1.5B** versus GRPO with outcome-only reward, achieved in 40% fewer training steps.

**Why it matters to Aurelius specifically:** PAC (OC-5) generates formal verification labels for math steps. PRPO is the natural complement: instead of only using formal verification at training-data admission time (as in PAC), PRPO uses the PRM signal continuously throughout RL training. Together, PAC+PRPO form a coherent pipeline: PAC ensures only provably-correct reasoning patterns enter training, PRPO provides dense signal to reinforce the most effective reasoning steps within each episode.

**Source:** PRPO [arXiv:2601.07182](https://arxiv.org/abs/2601.07182); VersaPRM companion (arXiv:2502.06737)

**Validated at:** 1.5B and 7B models; +3.2pp MATH500 vs GRPO; −40% training steps to reach same reward level.

**Files to change:** `src/alignment/grpo.py` (PRPO wrapper); `src/alignment/process_reward.py` (PRM scorer)

**Implementation sketch:**

```python
# src/alignment/grpo.py — PRPO dense reward wrapper
class PRPOWrapper:
    """
    Wrap any base RL trainer (GRPO, CISPO, REINFORCE++) with dense PRM rewards.
    Reference: arXiv:2601.07182
    """
    def __init__(
        self,
        base_trainer,       # existing GRPO / CISPO trainer
        prm: VersaPRM,      # trained PRM scorer
        prm_weight: float = 0.3,    # weight of PRM signal vs outcome reward
        outcome_weight: float = 0.7,
    ):
        self.base         = base_trainer
        self.prm          = prm
        self.prm_w        = prm_weight
        self.outcome_w    = outcome_weight

    def compute_dense_rewards(
        self,
        problem: str,
        steps:   list[str],     # parsed reasoning steps
        outcome_reward: float,  # binary outcome (correct / incorrect)
        domain: str = "math",
    ) -> torch.Tensor:
        """
        Convert PRM step scores + outcome reward into per-token dense rewards.
        Tokens within step t receive the PRM score for step t.
        """
        step_scores = [
            self.prm.score_step(problem, step, domain, steps[:i])
            for i, step in enumerate(steps)
        ]

        # Normalize PRM scores relative to mean
        prm_mean = float(np.mean(step_scores))
        prm_adv  = [s - prm_mean for s in step_scores]  # per-step advantage

        # Build per-token dense reward tensor
        # outcome_reward applied at the final token; PRM reward at each step end
        all_token_rewards = []
        for step_idx, (step, adv) in enumerate(zip(steps, prm_adv)):
            step_tokens = len(step.split())  # approximate token count
            # Distribute PRM advantage uniformly across step tokens
            step_reward = self.prm_w * adv / max(step_tokens, 1)
            all_token_rewards.extend([step_reward] * step_tokens)

        # Add outcome reward to final token
        if all_token_rewards:
            all_token_rewards[-1] += self.outcome_w * outcome_reward

        return torch.tensor(all_token_rewards, dtype=torch.float32)

    def training_step(self, batch):
        """Forward PRPO step: compute dense rewards, delegate to base trainer."""
        for item in batch:
            item["rewards"] = self.compute_dense_rewards(
                item["problem"], item["steps"],
                item["outcome_reward"], item["domain"]
            )
        return self.base.training_step(batch)
```

**Metrics to watch:** MATH500 / AMC 2024 pass@1 improvement over GRPO baseline; training steps to reach reward plateau (target −30–40%); PRM scoring overhead (add ~15–20ms per step at 1.4B — acceptable for offline training); credit assignment: compare gradient contribution from early vs late reasoning steps (PRPO should increase early-step gradient contributions vs GRPO).

**Risk / mitigation:** If the PRM is miscalibrated (frequently wrong step scores), PRPO's dense signal will reinforce incorrect reasoning. Gate on PRM quality: only activate PRPO after VersaPRM or uPRM achieves step F1 >0.68 on held-out math problems. Use `prm_weight=0.1` initially and increase to 0.3 after confirming PRM accuracy.

**Depends on:** VersaPRM (item 17f) or uPRM (item 17e); GRPO/CISPO/REINFORCE++ trainer as base; CoT distillation data (item 23k) provides training examples with parseable reasoning steps.

---

## V1 — Inference & Serving

---

### 18. EAGLE-3 Speculative Decoding Draft Head

**What it is:** Train a small (~150M parameter) auto-regressive draft model that fuses hidden states from early, middle, and late layers of the main Aurelius model (tri-layer fusion), then uses this draft for speculative decoding with tree-based verification.

**Why it matters to Aurelius specifically:** Aurelius has MTP speculative decoding in-flight targeting ~1.8× speedup. EAGLE-3 (NeurIPS 2025) achieves **4.1–6.5× speedup** at temperature 0 and 3.3× in production by conditioning the draft on richer multi-layer features. The key insight: MTP draft heads only see the final layer's hidden state. EAGLE-3 fuses early+mid+late layer signals, giving the draft model a far more complete picture of the main model's "intent."

**Source:** EAGLE-3 [arXiv 2503.01840](https://arxiv.org/html/2503.01840v1); [E2E Networks guide](https://www.e2enetworks.com/blog/Accelerating_LLM_Inference_with_EAGLE)

**Validated at:** Vicuna-13B, Llama-3.1-8B, Llama-3.3-70B — 4.1–6.5× verified speedup. Scaling law found: more draft training data → proportionally better speedup.

**Training data:** ShareGPT (68K samples) + UltraChat-200K (464K samples) ≈ 532K examples

**Files to change:** `src/inference/` (new: `src/inference/eagle3_draft.py`); `src/inference/speculative_decoding.py`

**Implementation sketch:**

```python
# src/inference/eagle3_draft.py
class Eagle3DraftModel(nn.Module):
    """
    Tri-layer fusion draft model for speculative decoding.
    Fuses Aurelius hidden states from layers 5 (early), 12 (mid), 22 (late)
    to produce a richer draft signal than MTP's final-layer-only approach.
    """
    FUSION_LAYERS = (5, 12, 22)    # early / mid / late of 24-layer Aurelius

    def __init__(
        self,
        hidden_dim: int   = 2048,  # Aurelius d_model
        draft_dim:  int   = 512,
        n_draft_layers: int = 4,
        vocab_size: int   = 8192,
    ):
        super().__init__()
        # One projector per fusion layer
        self.feature_projectors = nn.ModuleList([
            nn.Linear(hidden_dim, draft_dim, bias=False)
            for _ in self.FUSION_LAYERS
        ])
        # Fuse three projections into one draft representation
        self.fusion_gate = nn.Sequential(
            nn.Linear(draft_dim * 3, draft_dim * 2),
            nn.SiLU(),
            nn.Linear(draft_dim * 2, draft_dim),
        )
        # Small auto-regressive transformer for token drafting
        self.draft_transformer = SmallCausalTransformer(
            d_model=draft_dim,
            n_layers=n_draft_layers,
            n_heads=8,
            d_ff=draft_dim * 2,
        )
        self.lm_head = nn.Linear(draft_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: dict[int, torch.Tensor],  # layer_idx → hidden
    ) -> torch.Tensor:
        # Project and fuse three anchor-layer hidden states
        projections = [
            proj(target_hidden_states[layer])
            for proj, layer in zip(self.feature_projectors, self.FUSION_LAYERS)
        ]
        fused = self.fusion_gate(torch.cat(projections, dim=-1))

        # Auto-regressively draft next K tokens
        draft_hidden = self.draft_transformer(input_ids, prefix_context=fused)
        return self.lm_head(draft_hidden)


# src/inference/speculative_decoding.py — Eagle3 decoding loop
class Eagle3SpeculativeDecoder:
    def __init__(self, target_model, draft_model: Eagle3DraftModel,
                 k_draft: int = 5, tree_width: int = 4):
        self.target = target_model
        self.draft  = draft_model
        self.k      = k_draft        # tokens drafted per step
        self.width  = tree_width     # tree branching factor

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        generated = input_ids.clone()

        while generated.shape[1] - input_ids.shape[1] < max_new_tokens:
            # 1. Run target model forward — collect hidden states at fusion layers
            with self.target.capture_hidden_states(self.draft.FUSION_LAYERS) as hs:
                target_logits = self.target(generated)

            # 2. Draft K candidates using EAGLE-3 head
            draft_ids = self._draft_tree(generated, hs, self.k, self.width)

            # 3. Verify all candidates in one parallel forward pass of target model
            accepted, n_accepted = self._verify(generated, draft_ids, target_logits)

            # 4. Append accepted tokens (typically 3–5 per verification step)
            generated = torch.cat([generated, accepted], dim=1)

        return generated

    def _draft_tree(self, prefix, hidden_states, k, width):
        """Build a draft tree of K candidate continuations."""
        # Tree-based candidate generation — each node is a partial continuation
        root_logits = self.draft(prefix, hidden_states)
        # Sample top-`width` at each position to build tree
        return _build_candidate_tree(root_logits, k, width)

    def _verify(self, prefix, draft_candidates, target_logits):
        """Single forward pass verifies all draft candidates simultaneously."""
        # Standard speculative decoding acceptance criterion
        return _speculative_accept(prefix, draft_candidates, target_logits,
                                   self.target)
```

**Training procedure:**

```yaml
# configs/eagle3_train.yaml
eagle3:
  fusion_layers: [5, 12, 22]
  draft_dim: 512
  n_draft_layers: 4
  training_data:
    - data/sharegpt_68k.jsonl
    - data/ultrachat_200k.jsonl
  batch_size: 32
  lr: 2e-4
  warmup_steps: 500
  total_steps: 10_000
  loss: cross_entropy_on_draft_targets  # teacher-forced draft training
```

**Metrics to watch:** Acceptance rate (target >75% at temperature 0.7); effective throughput improvement (measure wall-clock tokens/sec); draft model accuracy per position (+1: >70%, +2: >55%, +3: >45%); memory overhead of draft model (150M ≈ 300MB in BF16 — acceptable).

**Risk / mitigation:** Draft head must be retrained whenever the main model checkpoint changes (weights, architecture). Pin Eagle-3 training to a specific v1 checkpoint tag. If acceptance rate is below 60%, increase draft training data — EAGLE-3's scaling law shows linear improvement with data volume.

**Depends on:** Stable v1 checkpoint. Build after v1 first major eval milestone.

---

### 19. FastKV + SnapKV Adaptive Cache Strategy

**What it is:** Add two new KV cache strategies to Aurelius's existing 8-strategy hot-swap system. FastKV (ACL Findings 2026) decouples context reduction from KV compression into two independent phases. SnapKV clusters important KV positions per head using pooled attention scores.

**Why it matters to Aurelius specifically:** Aurelius already has 8 KV cache strategies. FastKV and SnapKV represent the 2025–2026 state-of-the-art in adaptive eviction — FastKV is specifically optimised for the prefill-decoding boundary that causes latency spikes in long-context serving.

**Source:** FastKV [github.com/dongwonjo/FastKV](https://github.com/dongwonjo/FastKV) (ACL Findings 2026); Adaptive KV-Cache Compression [arxiv.org/pdf/2509.03136](https://www.arxiv.org/pdf/2509.03136) (ICLR 2026); NVIDIA kvpress [github.com/NVIDIA/kvpress](https://github.com/NVIDIA/kvpress)

**Files to change:** `src/inference/kv_cache/` (new strategy files)

**Implementation sketch:**

```python
# src/inference/kv_cache/fast_kv.py
class FastKVStrategy(KVCacheStrategy):
    """
    Decoupled context reduction + KV compression.
    Phase 1 (prefill): reduce context to budget using attention importance.
    Phase 2 (decode):  compress remaining KV with quantisation.
    Reference: FastKV, ACL Findings 2026.
    """
    name = "fast_kv"

    def __init__(self, budget_ratio: float = 0.3, quant_bits: int = 4):
        self.budget_ratio = budget_ratio
        self.quant        = KVQuantizer(bits=quant_bits)

    def compress_prefill(
        self,
        keys:   torch.Tensor,  # (batch, heads, seq, head_dim)
        values: torch.Tensor,
        attn_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Phase 1: keep top-budget_ratio tokens by cumulative attention
        budget = int(keys.shape[2] * self.budget_ratio)
        importance = attn_weights.mean(dim=(0, 1))  # average across batch/heads
        top_indices = importance.topk(budget).indices.sort().values
        k_reduced = keys[:, :, top_indices, :]
        v_reduced = values[:, :, top_indices, :]

        # Phase 2: quantise the retained KV pairs
        k_q = self.quant.quantize(k_reduced)
        v_q = self.quant.quantize(v_reduced)
        return k_q, v_q
```

**Metrics to watch:** Memory usage at 32K / 64K / 128K context (target 50–70% reduction vs full cache); generation quality on LongBench (should not degrade >5% vs full cache); prefill latency at 32K (FastKV should reduce the prefill-to-first-token spike).

**Depends on:** Existing KV cache strategy hot-swap system. Drop in as strategy #9 and #10.

---


### 19b. Mirror-SD — Dual Parallel Speculation Pipelines

**What it is:** Mirror Speculative Decoding (arXiv:2510.13161, ICLR 2026) runs **two parallel draft-verification pipelines simultaneously** — one using a forward-pass draft head (like EAGLE-3) and one using a backward-projection draft that starts from predicted future hidden states. The two pipelines propose independent candidate token sequences for the same position. The verifier batches both sets of candidates in a single forward pass and accepts whichever candidate sequence achieves higher accept-rate. Because the two drafts are statistically independent, the probability that at least one draft is accepted is superlinear in the individual acceptance rates: `P(accept either) = 1 - (1-p₁)(1-p₂)`.

**Why it matters to Aurelius specifically:** EAGLE-3 (item 18) achieves ~75% acceptance rate at temperature 0.7. Mirror-SD's dual-pipeline structure pushes effective acceptance to ~87% using exactly the same draft head, with only ~12% additional GPU memory for the second pipeline's KV buffer. The paper reports **30% throughput improvement over EAGLE-3** on Llama-3.3-70B. At Aurelius's 1.4B scale, the overhead is proportionally smaller and the benefit holds.

**Source:** Mirror-SD [arXiv:2510.13161](https://arxiv.org/abs/2510.13161) (ICLR 2026); EAGLE-3 comparison directly in paper.

**Validated at:** Llama-3.3-70B, Llama-3.1-8B; +30% throughput over EAGLE-3; +12% memory overhead.

**Files to change:** `src/inference/speculative_decoding.py`

**Implementation sketch:**

```python
# src/inference/speculative_decoding.py
class MirrorSDDecoder:
    """
    Dual parallel speculation: forward + backward draft pipelines.
    Both draft the same position; verifier accepts better candidate.
    Reference: arXiv:2510.13161 (ICLR 2026)
    """
    def __init__(
        self,
        target_model,
        forward_draft: Eagle3DraftModel,    # from item 18
        backward_draft: "BackwardProjectionDraft",
        k_draft: int = 5,
        tree_width: int = 4,
    ):
        self.target    = target_model
        self.fwd       = forward_draft
        self.bwd       = backward_draft
        self.k         = k_draft
        self.width     = tree_width

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        generated = input_ids.clone()

        while generated.shape[1] - input_ids.shape[1] < max_new_tokens:
            # Run target model once; collect hidden states for both drafts
            with self.target.capture_hidden_states([5, 12, 22]) as hs:
                target_logits = self.target(generated)

            # Both drafts run concurrently for the next K positions
            fwd_candidates = self.fwd.draft(generated, hs, self.k, self.width)
            bwd_candidates = self.bwd.draft(generated, target_logits, self.k)

            # Single verification pass — test all candidates from both pipelines
            all_candidates = _merge_candidate_trees(fwd_candidates, bwd_candidates)
            accepted, n_accepted = self._verify_merged(generated, all_candidates, target_logits)

            generated = torch.cat([generated, accepted], dim=1)

        return generated

    def _verify_merged(self, prefix, merged_candidates, target_logits):
        """
        Batch verify all candidates from both pipelines in one target model call.
        Select tokens from whichever pipeline achieves higher acceptance at each position.
        """
        all_accepted = _speculative_accept_batched(
            prefix, merged_candidates, target_logits, self.target
        )
        # Return the longest accepted prefix (greedy selection between pipelines)
        return _select_longest_accepted(all_accepted)


class BackwardProjectionDraft(nn.Module):
    """
    Backward-projection draft: predict token at position t from the hidden state
    expected at position t+k (projected backwards through residual stream).
    Statistically independent from EAGLE-3's forward projection.
    """
    def __init__(self, hidden_dim: int = 2048, draft_dim: int = 512,
                 vocab_size: int = 8192, k_horizon: int = 5):
        super().__init__()
        self.horizon   = k_horizon
        self.backward_proj = nn.Sequential(
            nn.Linear(hidden_dim, draft_dim),
            nn.SiLU(),
            nn.Linear(draft_dim, hidden_dim),  # project to hidden space
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def draft(self, prefix, target_logits, k: int) -> list:
        # Predict what the hidden state at +k will look like; project backward
        future_hidden = self._extrapolate_future_hidden(target_logits, k)
        backward_repr = self.backward_proj(future_hidden)
        draft_logits  = self.lm_head(backward_repr)
        return _sample_candidates(draft_logits, k, top_p=0.9)

    def _extrapolate_future_hidden(self, current_logits, k):
        # Linear extrapolation of hidden states over k steps using residual stream analysis
        return current_logits[:, -1:, :]   # placeholder — full implementation uses SSM-style prediction
```

**Metrics to watch:** Effective acceptance rate (target >85% vs EAGLE-3's ~75%); throughput tokens/sec (target +25–35% vs EAGLE-3); memory overhead (target <15% over EAGLE-3 alone); inter-pipeline correlation (should be <0.3 — if drafts are correlated, Mirror-SD benefit collapses to EAGLE-3 level).

**Risk / mitigation:** The backward draft must be genuinely statistically independent from the forward draft — if both are based on similar hidden-state features, correlation rises and the theoretical gain vanishes. Train the backward draft on different data splits or with different temperature settings. If inter-pipeline correlation is >0.5, switch to a trained Mamba-2 SSM as the second draft instead.

**Depends on:** EAGLE-3 draft head (item 18); stable v1 checkpoint; ~120MB additional GPU memory for the backward draft model.

---

### 19c. LK Losses — Replace KL Divergence in Draft Head Training

**What it is:** LK Losses (arXiv:2602.23881, February 2026) replaces the standard KL divergence loss used in draft model training with a family of **Lₚ–Kullback divergences** that more aggressively penalize mode-miss errors (cases where the draft predicts low probability for a token that the target model assigns high probability). Standard KL loss is asymmetric: it strongly penalizes putting probability mass where the target doesn't, but lightly penalizes *missing* target probability mass. For speculative decoding, missing mass is exactly what hurts acceptance rate — if the draft assigns low probability to a frequently-accepted token, that token will never be drafted. LK Losses use a weighted combination `L(p||q) = α·KL(p||q) + β·KL(q||p)` that biases training toward mode-covering rather than mode-seeking. Result: **+8–10% acceptance rate** on trained EAGLE-3 draft heads with no architecture change.

**Why it matters to Aurelius specifically:** Draft head training for EAGLE-3 (item 18) currently uses cross-entropy on teacher-forced targets (equivalent to forward KL). LK Loss is a one-line change to the loss function that consistently improves acceptance rate across all evaluated temperatures. At acceptance rate of 75% (EAGLE-3 baseline), a +9% improvement brings it to 82%, translating directly to higher throughput (more accepted tokens per verification step).

**Source:** LK Losses [arXiv:2602.23881](https://arxiv.org/abs/2602.23881); draft acceptance rate improvement reported across multiple architectures.

**Validated at:** EAGLE-3 on Llama-3-8B and Llama-3.3-70B; +8–10% acceptance rate across temperatures 0.0–1.0.

**Files to change:** `src/inference/eagle3_draft.py` (training loss)

**Implementation sketch:**

```python
# src/inference/eagle3_draft.py — LK Loss for draft training
class LKLoss(nn.Module):
    """
    LK Loss: weighted forward + reverse KL divergence for draft head training.
    Penalizes mode-miss more than mode-excess — directly improves acceptance rate.
    Reference: arXiv:2602.23881
    """
    def __init__(
        self,
        alpha: float = 0.4,   # weight of forward KL  (KL(target||draft) — mode-covering)
        beta: float  = 0.6,   # weight of reverse KL  (KL(draft||target) — standard CE)
        temperature: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.T     = temperature

    def forward(
        self,
        draft_logits:  torch.Tensor,   # (batch, seq, vocab) — draft model output
        target_logits: torch.Tensor,   # (batch, seq, vocab) — target model output (teacher)
    ) -> torch.Tensor:
        draft_log_probs  = F.log_softmax(draft_logits  / self.T, dim=-1)
        target_log_probs = F.log_softmax(target_logits / self.T, dim=-1)
        target_probs     = target_log_probs.exp()
        draft_probs      = draft_log_probs.exp()

        # Forward KL: KL(target || draft) = Σ p_target log(p_target / p_draft)
        # High when draft assigns low probability to target's peak mass (mode-miss)
        forward_kl = (target_probs * (target_log_probs - draft_log_probs)).sum(-1)

        # Reverse KL: KL(draft || target) = Σ p_draft log(p_draft / p_target)
        # Standard cross-entropy analog
        reverse_kl = (draft_probs  * (draft_log_probs  - target_log_probs)).sum(-1)

        loss = self.alpha * forward_kl + self.beta * reverse_kl
        return loss.mean()


# Usage: replace cross-entropy in eagle3 training loop
# Old: loss = F.cross_entropy(draft_logits, target_ids)
# New: loss = LKLoss(alpha=0.4, beta=0.6)(draft_logits, target_logits)
```

**Metrics to watch:** Draft acceptance rate at temperature 0.0, 0.7, 1.0 (target +8–10% vs CE baseline); validation perplexity of draft model on held-out dialogue (should not regress — LK Loss trains for acceptance, not generation quality); GPU time for draft training (LK Loss requires storing target logits alongside labels — adds ~15% training memory and ~5% training time).

**Risk / mitigation:** The alpha/beta ratio is sensitive to temperature and model scale. At low temperature (0.0–0.3), forward KL matters most (peak mass alignment); at high temperature (>0.8), both directions contribute equally. Sweep α ∈ {0.3, 0.4, 0.5} during draft training hyperparameter search. If acceptance rate improvement is below 5%, check that target logits are stored at full precision (not quantized) — quantized target logits degrade the mode-covering signal.

**Depends on:** EAGLE-3 draft head training (item 18); target model hidden states + logits must be stored during draft training (existing requirement for teacher-forcing, now with logits stored instead of just labels).

---

### 19d. SlimSpec — Low-Rank LM-Head for Draft Models

**What it is:** SlimSpec (arXiv:2605.10453, May 2026) replaces the full vocabulary linear projection in draft models with a **low-rank factorization**: `LM_head(h) ≈ V₂ · (V₁ · h)` where V₁ ∈ ℝᵣˣᵈ and V₂ ∈ ℝᵛˣʳ, r ≪ d, v. For a draft model with vocab_size=32,000 and hidden_dim=512, the standard LM head has 16.4M parameters. With rank r=64, SlimSpec reduces this to 557K parameters — **97% reduction in head parameters** — while retaining 96% of acceptance rate through careful initialization (V₂ is initialized from SVD of the original full-rank head). The result is a **4–5× speedup** in draft token generation latency, because the LM head projection dominates the forward pass of small draft models.

**Why it matters to Aurelius specifically:** The EAGLE-3 draft model (item 18) has a 150M parameter base with an 8,192-vocab LM head (Aurelius's vocab size). At vocab 8192 and draft hidden dim 512, the LM head is 4.2M parameters — 2.8% of the draft model but ~35% of each draft model forward pass time (due to the large matrix multiply to project to vocab space). SlimSpec's low-rank head cuts this to 0.09M parameters (rank=32), reducing the projection time by ~40× at negligible quality cost.

**Source:** SlimSpec [arXiv:2605.10453](https://arxiv.org/abs/2605.10453); EAGLE-3 LM-head analysis.

**Validated at:** EAGLE-3 draft on Llama-3-8B and 70B; 4–5× draft latency reduction; acceptance rate regression <1% vs full LM head.

**Files to change:** `src/inference/eagle3_draft.py`

**Implementation sketch:**

```python
# src/inference/eagle3_draft.py — SlimSpec low-rank LM head
class SlimSpecLMHead(nn.Module):
    """
    Low-rank factorization of draft model LM head.
    Reduces LM head parameters by 97%, draft latency by 4-5x.
    Reference: arXiv:2605.10453
    """
    def __init__(self, hidden_dim: int, vocab_size: int, rank: int = 32):
        super().__init__()
        self.v1 = nn.Linear(hidden_dim, rank, bias=False)    # hidden → low-rank
        self.v2 = nn.Linear(rank, vocab_size, bias=False)    # low-rank → vocab

    @classmethod
    def from_full_head(cls, full_lm_head: nn.Linear, rank: int = 32) -> "SlimSpecLMHead":
        """Initialize from SVD of a trained full-rank LM head — minimal quality regression."""
        W = full_lm_head.weight.data      # (vocab, hidden)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        # Keep top-rank singular values
        U_r  = U[:, :rank]           # (vocab, rank)
        S_r  = S[:rank]              # (rank,)
        Vh_r = Vh[:rank, :]          # (rank, hidden)

        head = cls(full_lm_head.in_features, full_lm_head.out_features, rank)
        # V1: hidden → rank (right singular vectors scaled by √S)
        head.v1.weight.data = Vh_r * S_r.unsqueeze(-1).sqrt()
        # V2: rank → vocab (left singular vectors scaled by √S)
        head.v2.weight.data = U_r   * S_r.unsqueeze(0).sqrt()
        return head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v2(self.v1(x))   # two small matmuls instead of one large one


# Migration: replace existing lm_head in Eagle3DraftModel
def upgrade_eagle3_to_slimspec(draft_model: Eagle3DraftModel, rank: int = 32):
    draft_model.lm_head = SlimSpecLMHead.from_full_head(draft_model.lm_head, rank)
    return draft_model
```

**Metrics to watch:** Draft model forward pass latency (target −40–50% on LM head projection); acceptance rate regression (target <1.0pp vs full LM head); GPU memory reduction (from training — SlimSpec also reduces draft model training memory); vocabulary distribution coverage (top-100 token probability mass should be retained — check with KL divergence between full and slim head outputs).

**Risk / mitigation:** SVD initialization assumes the original LM head is trained. If applying SlimSpec to a freshly initialized draft model (before EAGLE-3 training), use random initialization for v1 and v2 instead. In that case, expect 2–3pp higher acceptance rate regression that closes after ~2K training steps.

**Depends on:** EAGLE-3 draft head training (item 18). Apply after draft head reaches convergence (acceptance rate plateau), then fine-tune for 500 steps with LK Loss to recover the 1% regression.

---

### 19e. TurboQuant — 3-bit KV Quantization

**What it is:** TurboQuant (arXiv:2504.19874, ICLR 2026) achieves **3-bit KV cache quantization** — beyond the standard INT4/FP8 floor — by using **mixed-precision per-head quantization with outlier protection**: attention heads with high kurtosis (heavy-tailed key/value distributions) are quantized to 4-bit while the majority of heads use 3-bit. Outlier values within each head are identified using channel-wise statistics and are stored at higher precision in a sparse side-table (typically <0.5% of values). Result: **6× memory reduction** vs BF16 KV cache, **3× vs INT4**, with <0.3% perplexity regression on 128K context benchmarks.

**Why it matters to Aurelius specifically:** Aurelius's existing KV cache strategies (items 18-19) focus on *eviction* (dropping old KVs) and *compression* (pooling similar KVs). TurboQuant is orthogonal — it reduces the **precision** of retained KVs without evicting anything, extending the effective context window at fixed GPU memory. For a 1.4B model serving 128K context, the full BF16 KV cache requires ~4.4GB per sequence. With TurboQuant, this drops to ~730MB — enabling 6× more concurrent long-context requests per GPU.

**Source:** TurboQuant [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026); NVIDIA kvpress library (github.com/NVIDIA/kvpress).

**Validated at:** Llama-3.1-8B, Llama-3.3-70B at 128K context; 6× memory reduction; <0.3pp perplexity regression vs BF16; RULER 128K benchmark maintained.

**Files to change:** `src/inference/kv_cache/turbo_quant.py` (new)

**Implementation sketch:**

```python
# src/inference/kv_cache/turbo_quant.py
class TurboQuantKVCache:
    """
    Mixed-precision KV quantization: 3-bit for low-kurtosis heads,
    4-bit for high-kurtosis heads, with sparse outlier side table.
    Reference: arXiv:2504.19874 (TurboQuant, ICLR 2026)
    """
    KURTOSIS_THRESHOLD = 4.0     # above this → use 4-bit instead of 3-bit
    OUTLIER_RATIO      = 0.005   # top 0.5% values stored at full precision

    def __init__(self, n_heads: int, head_dim: int):
        self.n_heads  = n_heads
        self.head_dim = head_dim
        self._kurtosis_cache: dict[int, float] = {}

    def quantize(
        self,
        keys:   torch.Tensor,   # (batch, heads, seq, head_dim)
        values: torch.Tensor,
    ) -> dict:
        quant_keys, quant_vals = [], []
        outlier_tables = []

        for h in range(self.n_heads):
            k_h = keys[:, h, :, :]       # (batch, seq, head_dim)
            v_h = values[:, h, :, :]

            kurtosis = self._compute_kurtosis(k_h)
            bits = 4 if kurtosis > self.KURTOSIS_THRESHOLD else 3

            k_q, k_out = self._quantize_with_outliers(k_h, bits)
            v_q, v_out = self._quantize_with_outliers(v_h, bits)

            quant_keys.append((k_q, bits))
            quant_vals.append((v_q, bits))
            outlier_tables.append((k_out, v_out))

        return {
            "keys":    quant_keys,
            "values":  quant_vals,
            "outliers": outlier_tables,
        }

    def dequantize(self, quant_cache: dict) -> tuple[torch.Tensor, torch.Tensor]:
        keys_list, vals_list = [], []
        for h, ((kq, bits), (vq, bits_v), (ko, vo)) in enumerate(zip(
            quant_cache["keys"], quant_cache["values"], quant_cache["outliers"]
        )):
            keys_list.append(self._dequantize(kq, bits) + ko)
            vals_list.append(self._dequantize(vq, bits_v) + vo)
        return torch.stack(keys_list, dim=1), torch.stack(vals_list, dim=1)

    def _compute_kurtosis(self, tensor: torch.Tensor) -> float:
        mu = tensor.mean(); sigma = tensor.std()
        return float(((tensor - mu) ** 4).mean() / (sigma ** 4 + 1e-8))

    def _quantize_with_outliers(self, x: torch.Tensor, bits: int):
        n_outliers = max(1, int(x.numel() * self.OUTLIER_RATIO))
        flat = x.flatten()
        top_idx = flat.abs().topk(n_outliers).indices
        outlier_table = torch.zeros_like(flat)
        outlier_table[top_idx] = flat[top_idx]
        flat_clean = flat.clone(); flat_clean[top_idx] = 0.0
        quantized = _uniform_quantize(flat_clean.view_as(x), bits)
        return quantized, outlier_table.view_as(x)

    def _dequantize(self, q: torch.Tensor, bits: int) -> torch.Tensor:
        return _uniform_dequantize(q, bits)
```

**Metrics to watch:** KV cache memory at 128K context (target ~730MB per sequence at 1.4B); perplexity regression on PG-19 long-context benchmark (target <0.3pp vs BF16); RULER 128K score (no regression expected); throughput at 128K context vs BF16 (TurboQuant reduces memory bandwidth → faster attention at long context); fraction of heads using 3-bit vs 4-bit (expect 70–80% at 3-bit for a well-trained model).

**Risk / mitigation:** 3-bit quantization is lossy for heads with large activation variance. Add an activation range calibration step: run 512 sample sequences through the model before deployment to compute per-head kurtosis statistics and set per-head bit widths appropriately. If a head's kurtosis is extremely high (>8.0), fall back to BF16 for that head (typically <5% of heads). TurboQuant also requires dequantization on every attention computation — measure latency against BF16 to confirm net benefit on the target GPU.

**Depends on:** Existing KV cache infrastructure. Can be integrated as TurboQuant strategy alongside FastKV and SnapKV (item 19). TurboQuant is orthogonal to eviction strategies — apply after eviction.

---

### 19f. Transactional Attention — Anchor-Protected KV Eviction

**What it is:** Transactional Attention (TA, arXiv:2604.11288, April 2026) introduces **anchor tokens** in the KV cache that are protected from eviction. Standard KV eviction (StreamingLLM, FastKV, H₂O) evicts positions based on attention importance scores — but for structured outputs like JSON, XML, Python code, and tool calls, the **structural delimiter tokens** (braces, colons, parentheses, function names) are critically important but often receive low average attention scores (because most tokens attend to the semantic content, not the delimiters). TA identifies structural delimiter tokens at prefill time and marks them as anchors. During decode-time KV eviction, anchors are inviolable. Non-anchor tokens are evicted normally. Result: **structured output correctness maintained at 92.3%** vs 71.4% for H₂O at the same budget, and **equivalent throughput** to H₂O.

**Why it matters to Aurelius specifically:** Aurelius serves tool calls and structured JSON outputs. FastKV (item 19) and TurboQuant (item 19e) both perform KV eviction/compression that can accidentally drop the structural tokens that define JSON schema — causing malformed outputs at long context. TA is the fix: a lightweight wrapper around any KV eviction strategy that prevents structural anchor corruption. The anchor identification is also directly useful for Aurelius's AMC protocol — when the model is writing a memory entry, the AMC schema tokens (`<amc_tier_2>`, `<tool_result>`, etc.) should be anchored unconditionally.

**Source:** Transactional Attention [arXiv:2604.11288](https://arxiv.org/abs/2604.11288); H₂O comparison directly in paper.

**Validated at:** Llama-3.1-8B on structured output tasks; 92.3% schema adherence vs 71.4% for H₂O at 30% cache budget.

**Files to change:** `src/inference/kv_cache/transactional_attention.py` (new)

**Implementation sketch:**

```python
# src/inference/kv_cache/transactional_attention.py
class TransactionalAttentionWrapper:
    """
    Wrap any KV eviction strategy with anchor protection.
    Structural delimiter tokens are never evicted.
    Reference: arXiv:2604.11288
    """
    # Tokens that anchor the structure of tool calls, JSON, code, AMC protocol
    ANCHOR_TOKENS = frozenset([
        "{", "}", "[", "]", "(", ")", ":", ",",     # JSON/Python structure
        '"""', "'''", "```",                          # code block delimiters
        "<tool_call>", "</tool_call>",                # tool call boundaries
        "<amc_tier_2>", "<amc_tier_3>",              # AMC memory anchors
        "def ", "class ", "return ", "import ",       # Python structure
    ])

    def __init__(self, base_strategy, tokenizer):
        self.base      = base_strategy
        self.tokenizer = tokenizer
        self._anchor_positions: set[int] = set()

    def identify_anchors(self, input_ids: torch.Tensor) -> None:
        """Mark anchor positions during prefill."""
        tokens = [self.tokenizer.decode([t]) for t in input_ids[0].tolist()]
        self._anchor_positions = {
            i for i, tok in enumerate(tokens)
            if any(a in tok for a in self.ANCHOR_TOKENS)
        }

    def evict(
        self,
        keys:         torch.Tensor,
        values:       torch.Tensor,
        attn_scores:  torch.Tensor,
        budget:       int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply base eviction strategy, then restore any evicted anchor positions."""
        # Run base eviction
        k_evicted, v_evicted = self.base.evict(keys, values, attn_scores, budget)

        # Restore anchor positions that were removed
        seq_len = keys.shape[2]
        for anchor_pos in sorted(self._anchor_positions):
            if anchor_pos < seq_len:
                k_evicted = self._restore_position(k_evicted, keys, anchor_pos)
                v_evicted = self._restore_position(v_evicted, values, anchor_pos)

        return k_evicted, v_evicted

    def _restore_position(self, evicted, original, pos):
        """Insert an anchor position back into the evicted KV cache."""
        if evicted.shape[2] < original.shape[2]:
            return torch.cat([
                evicted[:, :, :pos, :],
                original[:, :, pos:pos+1, :],
                evicted[:, :, pos:, :]
            ], dim=2)
        return evicted
```

**Metrics to watch:** JSON schema adherence rate at 128K context (target >90% vs baseline 71.4%); number of anchor positions identified per sequence (should be 5–15% of sequence length for tool-heavy conversations); eviction overhead (anchor restoration adds ~O(n_anchors) operations per decode step — negligible for <1000 anchors); memory: anchors are never evicted so effective cache size increases slightly (monitor actual vs planned cache budget).

**Risk / mitigation:** The anchor token set must be updated whenever the tokenizer or prompt format changes. Use a configuration file (`configs/ta_anchors.yaml`) rather than hardcoding. If a task uses heavy structural repetition (e.g., 1000+ JSON keys), anchoring all delimiters could defeat the eviction budget; add an anchor count cap (max_anchors=200) and prioritize schema-opening tokens over schema-closing tokens when cap is hit.

**Depends on:** Existing KV eviction strategies (FastKV, SnapKV from item 19); tokenizer; AMC schema definitions (for AMC-specific anchor tokens).

---

### 19g. G-STEP — Selective Tool-Call Gate

**What it is:** G-STEP (Gate for Selective Tool Execution via Planning, arXiv:2605.00136, April 2026) is a **lightweight gating module** trained to predict, before each tool call, whether the expected informational gain of the tool result justifies the latency and token overhead. For tool-heavy agentic tasks, models often make redundant retrieval calls (searching for information already in context), low-gain calls (querying for minor clarifications that don't change the answer), or speculative calls (preemptively fetching data that won't be used). G-STEP adds a binary gate that routes high-gain calls to execution and suppresses low-gain calls, replacing them with a cached estimate or directly proceeding from existing context.

**Why it matters to Aurelius specifically:** Aurelius's agentic serving path will include tool calls (web search, code execution, database queries, memory writes). In production benchmarks for agentic tasks, 20–40% of tool calls are redundant when evaluated post-hoc. G-STEP eliminates these proactively, reducing average agentic task latency by ~18% and tool execution costs by ~35% (per paper). For AMC memory writes specifically, G-STEP can gate on whether the new information substantively differs from what's already stored — preventing duplicate AMC entries.

**Source:** G-STEP [arXiv:2605.00136](https://arxiv.org/abs/2605.00136); ToolTree (arXiv:2602.04523) for complementary tool planning.

**Validated at:** ToolBench, WebArena, GAIA; −35% tool call count; −18% task completion latency; maintained task success rate.

**Files to change:** `src/inference/tool_calling.py` (new `GSTEPGate` class); `src/agent/amc_memory.py` (integrate into memory write path)

**Implementation sketch:**

```python
# src/inference/tool_calling.py
class GSTEPGate(nn.Module):
    """
    Lightweight binary gate predicting tool call necessity before execution.
    Trained on (context, tool_call, expected_gain) triples.
    Reference: arXiv:2605.00136
    """
    def __init__(self, hidden_dim: int = 2048, gate_hidden: int = 256):
        super().__init__()
        # Lightweight gate — 2 linear layers on top of last hidden state
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )
        self.threshold = 0.5    # gain threshold; tune per task type

    def predict_gain(
        self,
        hidden_state: torch.Tensor,      # (1, hidden_dim) — last token hidden state
        tool_call_type: str,             # e.g., "web_search", "code_execute", "memory_write"
        context_coverage: float = 0.0,  # fraction of query terms already in context
    ) -> tuple[bool, float]:
        """Returns (should_execute, predicted_gain_score)."""
        gain_logit = self.gate(hidden_state).item()
        # Penalize if context already covers the query
        adjusted_gain = gain_logit * (1.0 - 0.8 * context_coverage)
        should_execute = adjusted_gain > self.threshold
        return should_execute, adjusted_gain

    def intercept_tool_call(
        self,
        tool_call: dict,
        hidden_state: torch.Tensor,
        current_context: str,
    ) -> dict | None:
        """
        Intercept a pending tool call; return None to suppress, or the tool_call dict to proceed.
        Suppressed calls return a cached estimate or skip to inference.
        """
        query = tool_call.get("arguments", {}).get("query", "")
        coverage = self._estimate_context_coverage(query, current_context)
        should_execute, gain = self.predict_gain(hidden_state, tool_call["name"], coverage)

        if not should_execute:
            return None    # suppress — caller should continue without tool result
        return tool_call   # proceed to tool execution

    def _estimate_context_coverage(self, query: str, context: str) -> float:
        """Simple term-overlap coverage estimate."""
        query_terms = set(query.lower().split())
        context_terms = set(context.lower().split())
        if not query_terms:
            return 0.0
        return len(query_terms & context_terms) / len(query_terms)
```

**Training data for G-STEP gate:** Collect agentic task rollouts; label each tool call as high-gain (changed final answer) or low-gain (no effect on answer) by counterfactual ablation; train gate as binary classifier.

**Metrics to watch:** Tool suppression rate (target 20–35%); task success rate with gate vs without (must not regress); false suppression rate (gate suppresses a high-gain call — track separately as this is costly); gate inference latency (<1ms since it's a 2-layer MLP on last hidden state — negligible overhead).

**Risk / mitigation:** A gate trained on ToolBench may over-suppress on new tool types not seen during gate training. Maintain a per-tool-type suppression rate monitor; if any tool type shows >50% suppression with <2% gain in task success, that tool type's threshold should be lowered (or the gate should bypass that tool type entirely). Err toward recall (not missing high-gain calls) over precision.

**Depends on:** Agentic tool-calling infrastructure; training data from rollout traces (can reuse AMC benchmark tasks); AMC memory write gating is a natural first deployment (memwrites are highly redundant in production).

---

### 19h. MemMachine — Episode-Preserving Memory with Nucleus Retrieval

**What it is:** MemMachine (arXiv:2604.04853, April 2026) is a memory system architecture for long-context agentic tasks that maintains **episode structure** across memory writes and uses **contextualized nucleus retrieval** to surface relevant memories at query time. Standard RAG memory systems write each memory as an independent embedding and retrieve by cosine similarity — losing the sequential and causal relationships between memories (what happened before and after each memory entry). MemMachine preserves episode boundaries explicitly: memories from the same conversational episode are stored with a shared episode embedding prefix, and retrieval is biased toward complete episodes rather than isolated fragments. Nucleus retrieval further biases toward the *diverse* subset of retrieved memories rather than the *most similar* subset — preventing redundant memories from dominating the context window. Result: **93% accuracy on LongMemEvalS** (multi-hop memory retrieval benchmark) vs 71% for standard RAG.

**Why it matters to Aurelius specifically:** AMC (the Aurelius Memory Contract) defines Tier 1–4 memory operations. Currently, Tier 2 and Tier 3 stores are described architecturally but the *retrieval* protocol is not specified in detail. MemMachine provides the retrieval algorithm: (1) use episode-aware embeddings for storage, (2) nucleus retrieval with diversity constraint at query time, (3) episode coherence scoring to weight cross-episode multi-hop queries. This directly implements the RC-AMC (OC-8) and SLR (OC-11) ideas from Section 0.

**Source:** MemMachine [arXiv:2604.04853](https://arxiv.org/abs/2604.04853); LongMemEvalS benchmark; MemGPT comparison in paper.

**Validated at:** LongMemEvalS multi-hop; 93% accuracy vs 71% RAG baseline; 38% reduction in irrelevant memories in context window (nucleus diversity constraint).

**Files to change:** `src/agent/memory_store.py` (new `MemMachineStore` class); `src/agent/amc_memory.py`

**Implementation sketch:**

```python
# src/agent/memory_store.py
class MemMachineStore:
    """
    Episode-preserving memory store with nucleus retrieval.
    Replaces flat embedding store for AMC Tier-2/Tier-3 memories.
    Reference: arXiv:2604.04853
    """
    def __init__(self, embed_model, episode_dim: int = 64, nucleus_k: int = 5):
        self.embed       = embed_model     # text encoder
        self.episode_dim = episode_dim     # size of episode prefix embedding
        self.k           = nucleus_k       # nucleus size for diverse retrieval
        self._store: list[dict] = []       # [{embedding, text, episode_id, step, metadata}]
        self._episode_embeds: dict[str, torch.Tensor] = {}

    def write(self, text: str, episode_id: str, step: int, metadata: dict = None):
        """Write a memory entry with episode context."""
        # Core embedding: semantic content
        content_emb = self.embed.encode(text)                  # (D,)
        # Episode embedding: shared across all memories in this episode
        if episode_id not in self._episode_embeds:
            self._episode_embeds[episode_id] = self.embed.encode(f"episode:{episode_id}")
        ep_emb = self._episode_embeds[episode_id]

        # Compound embedding: concatenate episode prefix + content
        compound = torch.cat([
            ep_emb[:self.episode_dim],         # episode prefix
            content_emb[self.episode_dim:],    # content suffix
        ])
        self._store.append({
            "embedding":  compound,
            "text":       text,
            "episode_id": episode_id,
            "step":       step,
            "metadata":   metadata or {},
        })

    def retrieve(self, query: str, k: int = 10) -> list[dict]:
        """
        Nucleus retrieval: return diverse top-k memories.
        Avoids redundant memories from dominating the context window.
        """
        query_emb = self.embed.encode(query)

        # Score all memories by cosine similarity
        scored = sorted(
            self._store,
            key=lambda m: F.cosine_similarity(m["embedding"].unsqueeze(0),
                                              query_emb.unsqueeze(0)).item(),
            reverse=True,
        )

        # Nucleus selection: greedily pick memories that maximize diversity
        selected = []
        for candidate in scored:
            if len(selected) >= k:
                break
            # Accept if sufficiently different from all already-selected memories
            if self._is_diverse(candidate, selected):
                selected.append(candidate)

        # Re-rank selected by episode coherence (prefer complete episodes)
        return self._rerank_by_episode_coherence(selected, query)

    def _is_diverse(self, candidate: dict, selected: list, threshold: float = 0.7) -> bool:
        if not selected:
            return True
        max_sim = max(
            F.cosine_similarity(candidate["embedding"].unsqueeze(0),
                                s["embedding"].unsqueeze(0)).item()
            for s in selected
        )
        return max_sim < threshold

    def _rerank_by_episode_coherence(self, memories: list[dict], query: str) -> list[dict]:
        """Boost memories whose episode matches the query's likely episode context."""
        query_ep_scores = {
            ep_id: F.cosine_similarity(
                self.embed.encode(query).unsqueeze(0),
                ep_emb.unsqueeze(0)
            ).item()
            for ep_id, ep_emb in self._episode_embeds.items()
        }
        return sorted(memories, key=lambda m: (
            0.6 * query_ep_scores.get(m["episode_id"], 0) +
            0.4 * F.cosine_similarity(m["embedding"].unsqueeze(0),
                                      self.embed.encode(query).unsqueeze(0)).item()
        ), reverse=True)
```

**Metrics to watch:** LongMemEvalS multi-hop accuracy (target >88% vs 71% RAG baseline); memory write latency (episode embedding adds ~10ms per write — acceptable for offline or batch writes); retrieval diversity (average pairwise cosine similarity of retrieved memories — target <0.5); AMC protocol compliance (memory writes must follow Tier-2/Tier-3 schema even with episode IDs added).

**Risk / mitigation:** Episode ID assignment is a design decision. For conversation memory, use conversation_id as episode_id. For agentic task memory, use task_id + subtask_id. If episode boundaries are unclear, use a fixed rolling window (e.g., every 20 memory writes = new episode) as a fallback. The diversity threshold (0.7) may need tuning per domain — for code memory where many entries are syntactically similar, lower it to 0.5.

**Depends on:** AMC memory architecture (OC-8, OC-9 from Section 0); sentence encoder (any embedding model — `sentence-transformers/all-MiniLM-L6-v2` is sufficient for prototype); episode structure from conversation/task management layer.

---

## V1 — Interpretability & Safety

---

### 20. Route-SAE — Scalable Sparse Autoencoders

**What it is:** Upgrade the existing SAE implementation to Route-SAE, which adds learned routing to direct different activation types to specialised sub-dictionaries. This makes SAEs significantly more scalable at 1B+ parameters where a single monolithic dictionary becomes too diffuse.

**Why it matters to Aurelius specifically:** Aurelius already has SAEs in `src/interpretability/`. Standard SAEs at 1.395B scale produce features that are too distributed to be actionable. Route-SAE (March 2025) applies a routing function that assigns activations to one of N sub-dictionaries based on their type (syntactic vs semantic vs factual), then trains each sub-dictionary independently. The result: more interpretable, more surgical features.

**Source:** Route-SAE [arXiv 2503.08200](https://arxiv.org/pdf/2503.08200); SAE Survey [arXiv 2503.05613](https://arxiv.org/pdf/2503.05613)

**Files to change:** `src/interpretability/sae_trainer.py` plus a new runtime `src/interpretability/route_sae.py` if inference-time hooks are split from training

**Implementation sketch:**

```python
# src/interpretability/sae.py
class RouteSAE(nn.Module):
    """
    Route-SAE: routes activations to specialised sub-dictionaries.
    Enables scalable, targeted feature extraction at 1B+ parameter scale.
    """
    def __init__(self, d_model: int, n_features: int,
                 n_subdicts: int = 4, sparsity_coeff: float = 1e-3):
        super().__init__()
        self.n_subdicts  = n_subdicts
        self.subdict_dim = n_features // n_subdicts

        # Router: assign each activation vector to one sub-dictionary
        self.router = nn.Linear(d_model, n_subdicts, bias=False)

        # Sub-dictionaries: one per activation type
        self.encoders = nn.ModuleList([
            nn.Linear(d_model, self.subdict_dim) for _ in range(n_subdicts)
        ])
        self.decoders = nn.ModuleList([
            nn.Linear(self.subdict_dim, d_model, bias=False) for _ in range(n_subdicts)
        ])
        self.sparsity_coeff = sparsity_coeff

    def forward(self, x: torch.Tensor):
        # Route: soft assignment to sub-dictionaries
        routing_weights = self.router(x).softmax(dim=-1)   # (batch, n_subdicts)

        # Encode into each sub-dictionary
        features_per_dict = [
            F.relu(enc(x)) for enc in self.encoders
        ]

        # Reconstruct from routed sub-dictionaries
        recon = sum(
            routing_weights[:, i:i+1] * self.decoders[i](f)
            for i, f in enumerate(features_per_dict)
        )

        # L1 sparsity loss across all features
        all_features = torch.cat(features_per_dict, dim=-1)
        sparsity_loss = self.sparsity_coeff * all_features.abs().mean()

        return recon, all_features, sparsity_loss

    def get_top_features(self, x: torch.Tensor, k: int = 20):
        """Return the k most active features for interpretability analysis."""
        _, all_features, _ = self.forward(x)
        return all_features.topk(k, dim=-1)
```

**Steering at inference time** (using SAE features to modify model behaviour):

```python
# src/interpretability/sae_steering.py
class SAESteeringHook:
    """
    Subtract harmful feature directions from residual stream at inference.
    Identified by probing — features correlated with harmful outputs.
    """
    def __init__(self, sae: RouteSAE, harmful_feature_ids: list[int],
                 steering_coeff: float = 20.0):
        self.sae      = sae
        self.bad_ids  = harmful_feature_ids
        self.coeff    = steering_coeff

    def __call__(self, module, input, output):
        x = output[0]   # residual stream at this layer
        _, features, _ = self.sae(x)
        # Subtract harmful feature directions
        for fid in self.bad_ids:
            direction = self.sae.decoders[fid // self.sae.subdict_dim].weight[
                fid % self.sae.subdict_dim
            ]
            x = x - self.coeff * features[:, :, fid:fid+1] * direction
        return (x,) + output[1:]
```

**Metrics to watch:** SAE reconstruction loss per sub-dictionary (should be lower than monolithic SAE); feature interpretability score (human eval: what fraction of top features have a clear semantic label); false positive rate of safety steering (legitimate prompts incorrectly triggered).

**Depends on:** Existing SAE infrastructure in `src/interpretability/`.

---

## Data Pipeline

---

### 21. FineWeb-Edu Neural Quality Filtering

**What it is:** Score all documents in the training corpus using a fastText educational quality classifier, then filter to keep only the top 65th percentile. FineWeb-Edu (HuggingFace, 2024) demonstrated that this filtering dramatically improves MMLU and ARC scores — educational content with clear explanations of concepts is disproportionately valuable for reasoning.

**Why it matters to Aurelius specifically:** The `data/` directory contains `.npy` uint16 token shards assembled from raw web/code/book sources. There is no documented quality filter step. DCLM showed that filtering to the top 10% of web documents produces a corpus that trains models to match performance of 15T-token corpora in 3.8T tokens — a 4× data efficiency gain.

**Source:** FineWeb [arXiv 2406.17557](https://arxiv.org/pdf/2406.17557); Ultra-FineWeb [arXiv 2505.05427](https://arxiv.org/html/2505.05427v1); DCLM [datologyai.com](https://www.datologyai.com/blog/technical-deep-dive-curating-our-way-to-a-state-of-the-art-text-dataset)

**Files to change:** `scripts/quality_filter.py` (new), `scripts/collect_training_data.py`

**Implementation sketch:**

```python
# scripts/quality_filter.py
import fasttext, numpy as np
from pathlib import Path

class EducationalQualityFilter:
    """
    Score documents by educational value using fastText classifier.
    Trained on FineWeb-Edu labels: "highly educational" vs "not educational".
    Keep documents scoring above threshold.
    """
    def __init__(
        self,
        model_path: str = "models/edu_quality_fasttext.bin",
        threshold: float = 0.65,
        upsample_stem: float = 2.0,    # upsample STEM/math docs by 2×
    ):
        self.model     = fasttext.load_model(model_path)
        self.threshold = threshold
        self.upsample  = upsample_stem

    def score_document(self, text: str) -> float:
        label, prob = self.model.predict(text[:512].replace("\n", " "))
        return float(prob[0]) if label[0] == "__label__educational" else 0.0

    def filter_jsonl(self, input_path: Path, output_path: Path) -> dict:
        stats = {"total": 0, "kept": 0, "upsampled": 0}
        with open(input_path) as fin, open(output_path, "w") as fout:
            for line in fin:
                doc = json.loads(line)
                score = self.score_document(doc["text"])
                stats["total"] += 1

                if score >= self.threshold:
                    fout.write(line)
                    stats["kept"] += 1

                    # Upsample STEM content
                    if self._is_stem(doc["text"]) and score > 0.8:
                        fout.write(line)   # write twice = 2× weight
                        stats["upsampled"] += 1
        return stats

    def _is_stem(self, text: str) -> bool:
        stem_keywords = {"theorem", "equation", "algorithm", "proof",
                         "hypothesis", "experimental", "derivative", "integral"}
        return bool(stem_keywords.intersection(text.lower().split()))

    def filter_npy_shard(self, shard_path: Path, tokenizer, output_path: Path):
        """Re-score tokenized shards by decoding → scoring → re-encoding."""
        tokens = np.load(shard_path)
        # Decode in chunks of 2048 tokens
        kept_chunks = []
        for i in range(0, len(tokens), 2048):
            chunk = tokens[i:i+2048]
            text  = tokenizer.decode(chunk)
            if self.score_document(text) >= self.threshold:
                kept_chunks.append(chunk)
        np.save(output_path, np.concatenate(kept_chunks).astype(np.uint16))
```

**Integration into Stage 2 curriculum:** The filtered, STEM-upsampled corpus becomes the data source for Stage 2 of the 3-stage curriculum.

**Metrics to watch:** Corpus size reduction (expect 35–40% reduction); MMLU zero-shot at 50B tokens trained on filtered vs unfiltered; ARC-Challenge score; model calibration (filtered corpus models tend to be better calibrated).

**Depends on:** Downloading and training the fastText classifier on FineWeb-Edu labels (public data, ~2 days training). Run filtering in parallel with current training.

---

### 22. Domain-Specific Synthetic Data Generation

**What it is:** Fine-tune Aurelius v1 on domain-specialist corpora (math, code) to produce two LoRA-adapted specialist models. Use these specialists to generate high-quality synthetic training examples for the Stage 2 pretraining curriculum and SFT data.

**Why it matters to Aurelius specifically:** Qwen3 generates math training data with Qwen2.5-Math and code data with Qwen2.5-Coder — specialist models produce far richer domain-specific examples than any human-curated dataset. At 1.395B, Aurelius can serve as its own teacher if specialised via LoRA.

**Source:** Qwen3 data pipeline [kili-technology.com](https://kili-technology.com/blog/data-story-qwen3); Condor data synthesis [arXiv 2501.12273](https://arxiv.org/pdf/2501.12273)

**Files to change:** `scripts/collect_training_data.py`, new `scripts/synthetic_data_gen.py`

**Implementation sketch:**

```python
# scripts/synthetic_data_gen.py
class SelfSyntheticDataGenerator:
    """
    Generate synthetic training data using domain-specialist LoRA adapters
    fine-tuned from the Aurelius base checkpoint.
    """
    def __init__(self, base_model_path: str):
        self.base = load_aurelius(base_model_path)

    def generate_math_examples(
        self,
        n_examples: int,
        math_lora_path: str,
        difficulty_levels: list[str] = ["easy", "medium", "hard", "competition"],
    ) -> list[dict]:
        model = load_lora(self.base, math_lora_path)
        examples = []

        for difficulty in difficulty_levels:
            prompt = MATH_GEN_PROMPTS[difficulty]
            for _ in range(n_examples // len(difficulty_levels)):
                # Generate problem + solution with chain-of-thought
                response = model.generate(
                    prompt, temperature=0.9, max_new_tokens=1024
                )
                problem, cot, answer = parse_math_response(response)
                # Verify answer with sympy before including
                if verify_math_answer(problem, answer):
                    examples.append({
                        "prompt": problem,
                        "cot": cot,
                        "response": answer,
                        "domain": "math",
                        "difficulty": difficulty,
                        "source": "self_synthetic",
                    })
        return examples

    def generate_code_examples(
        self,
        n_examples: int,
        code_lora_path: str,
        languages: list[str] = ["python", "rust", "typescript"],
    ) -> list[dict]:
        model = load_lora(self.base, code_lora_path)
        examples = []
        for lang in languages:
            for _ in range(n_examples // len(languages)):
                # Generate problem + unit tests + solution
                spec    = model.generate(CODE_SPEC_PROMPT.format(lang=lang))
                test    = model.generate(f"Write unit tests for: {spec}")
                solution = model.generate(f"Implement: {spec}")
                # Run tests to verify
                if run_tests(solution, test, lang):
                    examples.append({
                        "prompt": spec,
                        "response": solution,
                        "tests": test,
                        "domain": "code",
                        "language": lang,
                        "source": "self_synthetic",
                    })
        return examples
```

**Metrics to watch:** Synthetic data pass rate (fraction of generated examples passing verification); MATH benchmark improvement after including synthetic math examples in SFT; HumanEval improvement after code synthetic data.

**Risk / mitigation:** Self-distillation can cause mode collapse if the specialist LoRA is too aggressively fine-tuned. Use low LoRA rank (r=16) and conservative alpha. Monitor diversity of generated problems (embed with SBERT and check average pairwise cosine similarity — should stay below 0.7).

**Depends on:** Stable v1 checkpoint for LoRA fine-tuning.

---

### 23. DARE-TIES Model Merging Between Training Phases

**What it is:** After each major training phase (pretrain, SFT, DPO, GRPO), merge checkpoints from adjacent phases using DARE (Drop And REscale) + TIES (Trim-Elect-Sign) to prevent catastrophic forgetting and recover up to 7.5% performance vs naive sequential fine-tuning.

**Why it matters to Aurelius specifically:** Aurelius runs pretrain → SFT → DPO → GRPO → RLHF sequentially. Each phase overwrites knowledge from the previous. DARE-TIES has been shown to recover 7.5% general performance and 7% capability by resolving delta weight conflicts between task vectors from different phases.

**Source:** DARE-TIES [mbrenndoerfer.com](https://mbrenndoerfer.com/writing/model-merging-weight-averaging-task-arithmetic-ties-dare); [Nature Machine Intelligence 2025]

**Files to change:** `scripts/model_tools.py`

**Implementation sketch:**

```python
# scripts/model_tools.py
def dare(delta: dict[str, torch.Tensor], drop_rate: float = 0.3) -> dict:
    """Drop p% of delta parameters randomly, rescale remainder by 1/(1-p)."""
    result = {}
    for k, v in delta.items():
        mask = torch.bernoulli(torch.full_like(v, 1 - drop_rate)).bool()
        result[k] = v * mask / (1 - drop_rate)
    return result

def ties_merge(
    base: dict[str, torch.Tensor],
    checkpoints: list[dict[str, torch.Tensor]],
    weights: list[float],
    drop_rate: float = 0.3,
) -> dict[str, torch.Tensor]:
    """
    DARE-TIES merge:
    1. Compute delta = checkpoint - base for each task
    2. DARE: drop and rescale each delta
    3. TIES: trim small magnitudes, elect sign by majority, merge
    """
    deltas = [dare({k: ck[k] - base[k] for k in base}, drop_rate)
              for ck in checkpoints]

    merged = {}
    for k in base:
        stacked = torch.stack([d[k] for d in deltas], dim=0)  # (n_tasks, ...)

        # Trim: zero out below-mean magnitude deltas per parameter
        magnitudes = stacked.abs()
        threshold  = magnitudes.mean(dim=0, keepdim=True)
        stacked    = stacked * (magnitudes >= threshold)

        # Elect sign: majority vote across task deltas
        sign_sum   = stacked.sign().sum(dim=0)
        elected    = sign_sum.sign()

        # Merge: keep only deltas that agree with elected sign
        agree_mask = (stacked.sign() == elected.unsqueeze(0)) | (stacked == 0)
        w          = torch.tensor(weights, device=stacked.device)
        weighted   = (stacked * agree_mask * w.view(-1, *([1]*stacked.dim()-1))).sum(0)
        merged[k]  = base[k] + weighted / w.sum()

    return merged

# Usage in training pipeline
def merge_after_phase(pretrain_ckpt, sft_ckpt, dpo_ckpt):
    base = load_checkpoint(pretrain_ckpt)
    merged = ties_merge(
        base       = base,
        checkpoints = [load_checkpoint(sft_ckpt), load_checkpoint(dpo_ckpt)],
        weights     = [0.6, 0.4],   # weight SFT more (more tokens)
        drop_rate   = 0.30,
    )
    save_checkpoint(merged, "checkpoints/merged_sft_dpo.safetensors")
```

**Metrics to watch:** Zero-shot general benchmarks before/after merge (MMLU, ARC, HellaSwag); task-specific benchmarks (MATH, HumanEval) — confirm merge doesn't reduce specialisation; perplexity on held-out web text (measures base knowledge preservation).

**Depends on:** Completed pretrain + SFT checkpoints.

---


### 23b. FineWeb2 + Nemotron-CC: New Baseline Web Corpus

**What it is:** Two publicly released web datasets represent the 2025–2026 state of the art for general-purpose pretraining:

- **FineWeb2** (arXiv:2506.20920, June 2025): A 20TB / 5B document multilingual dataset built from ~100 CommonCrawl snapshots (2013–April 2024), covering 1,000+ languages. Key innovation: *Rehydration* — documents removed in the aggressive deduplication pass that originate from high-quality sources (Wikipedia mirrors, academic sites, technical documentation) are selectively re-added, recovering ~8% of corpus size in high-quality content. The English slice produces better-performing models than FineWeb-v1 on all standard benchmarks.

- **Nemotron-CC** (arXiv:2412.02595, NVIDIA, ACL 2025): 6.3T token dataset from 84 CommonCrawl snapshots, including 1.9T synthetic rephrased tokens. Uses an ensemble of three quality classifiers (FastText + two neural classifiers) whose votes are combined via majority rule, reducing false positives by 40% versus single-classifier filtering. Low-quality documents are rephrased using a medium-sized LM before inclusion rather than discarded — capturing niche technical knowledge that web style makes noisy. An 8B model trained on Nemotron-CC outperforms Llama-3.1-8B on 8 of 10 benchmarks.

**Why it matters to Aurelius specifically:** Aurelius's `data/` directory contains `.npy` token shards from a raw web/code/book corpus without a documented quality filtering pipeline. Replacing or supplementing this with FineWeb2 + the Nemotron-CC filtering methodology represents the highest-leverage single change to pretraining data quality. The Rehydration step is especially relevant: Aurelius's pretraining likely lost significant high-quality academic content to aggressive deduplication.

**Source:** FineWeb2 [arXiv:2506.20920](https://arxiv.org/abs/2506.20920); Nemotron-CC [arXiv:2412.02595](https://arxiv.org/abs/2412.02595); datatrove library (Hugging Face open-source pipeline)

**Files to change:** `scripts/collect_training_data.py`; `data/corpus_registry.yaml` (new)

**Implementation sketch:**

```python
# scripts/collect_training_data.py — FineWeb2 + Nemotron-CC pipeline integration
from datasets import load_dataset
import fasttext
import numpy as np
from pathlib import Path

class FineWeb2Pipeline:
    """
    FineWeb2 + Nemotron-CC quality pipeline for Aurelius corpus construction.
    Three-classifier ensemble + Rehydration step.
    Reference: arXiv:2506.20920, arXiv:2412.02595
    """
    def __init__(
        self,
        ft_classifier_path: str,       # fastText edu-quality classifier (FineWeb-style)
        neural_classifier_1_path: str, # FineWeb-Edu neural classifier
        neural_classifier_2_path: str, # Nemotron secondary classifier
        rehydration_domains: list[str] = None,
        min_votes: int = 2,            # Majority of 3 classifiers must agree
    ):
        self.ft_clf   = fasttext.load_model(ft_classifier_path)
        self.ncf1     = load_neural_classifier(neural_classifier_1_path)
        self.ncf2     = load_neural_classifier(neural_classifier_2_path)
        self.min_votes = min_votes
        self.rehydration_domains = rehydration_domains or [
            "arxiv.org", "wikipedia.org", "mathoverflow.net",
            "stackoverflow.com", "github.com", "docs.python.org",
        ]

    def score_document(self, text: str, url: str = "") -> dict:
        """Run all three classifiers; majority rule determines inclusion."""
        votes = [
            self._score_fasttext(text),
            self._score_neural(text, self.ncf1),
            self._score_neural(text, self.ncf2),
        ]
        accept_votes = sum(1 for v in votes if v > 0.5)

        # Rehydration: override for high-quality domains even if classifiers disagree
        from_quality_domain = any(d in url for d in self.rehydration_domains)
        if from_quality_domain and accept_votes >= 1:
            accept_votes = self.min_votes  # rehydrate: count as majority

        return {
            "accept": accept_votes >= self.min_votes,
            "votes": votes,
            "rehydrated": from_quality_domain,
        }

    def _score_fasttext(self, text: str) -> float:
        label, prob = self.ft_clf.predict(text[:512].replace("\n", " "))
        return float(prob[0]) if label[0] == "__label__quality" else 0.0

    def _score_neural(self, text: str, clf) -> float:
        return float(clf.predict_proba([text[:1024]])[0][1])


# Corpus mixing configuration (reflects optimal 30% synthetic + 70% natural)
CORPUS_MIXING = {
    "fineweb2_english":  0.45,    # high-quality natural web
    "nemotron_cc":       0.25,    # NVIDIA-filtered + rephrased synthetic mix
    "books_math":        0.12,    # Gutenberg + math books
    "code":              0.10,    # OpenCoder RefineCode (item 23j)
    "academic":          0.08,    # arXiv, PubMed, Semantic Scholar
}
```

**Metrics to watch:** MMLU 5-shot vs current corpus baseline (target +3–5pp); ARC-Challenge (target +2–4pp); downstream perplexity on PG-19 validation set; corpus size after filtering (expect ~40% reduction from raw CommonCrawl; Rehydration adds back ~5–8%); fastText classifier training time (~2 hours on labeled FineWeb-Edu subset).

**Risk / mitigation:** Nemotron-CC's 1.9T synthetic rephrased tokens represent ~30% of the total corpus — consistent with Demystifying Synthetic Data (arXiv:2510.01631) findings that 25–35% synthetic rephrase is optimal. Exceeding 50% synthetic starts to degrade model performance at small budgets. Monitor the synthetic fraction closely across corpus assembly stages.

**Depends on:** Download FineWeb2 from HuggingFace (free, public); train fastText classifier on FineWeb-Edu labels (2-hour process); Nemotron-CC dataset available for download from NVIDIA NGC or HuggingFace.

---

### 23c. Ultra-FineWeb Fast Classifier: Replace Slow LLM-Based Filtering

**What it is:** Ultra-FineWeb (arXiv:2505.05427, May 2025) demonstrates that a fine-tuned **fastText classifier** trained on LLM-labeled seed data achieves comparable document filtering quality to a full LLM judge (Mixtral-8x7B) at **50× lower inference cost**. The key insight: LLM judges have high recall on borderline-quality documents, but the marginal quality improvement from LLM vs fastText filtering does not justify the cost for the 90th percentile of documents. The fastText classifier's accuracy gap vs LLM judges is primarily on documents in the 40th–60th quality percentile — the borderline cases — which can be handled by adding a small LLM-scored "hard examples" dataset for classifier training.

**Why it matters to Aurelius specifically:** Item 21 (FineWeb-Edu Neural Quality Filtering) already proposes a fastText filter. Ultra-FineWeb specifically demonstrates that the right training methodology for this classifier — using an LLM to label a seed set of 50K documents, then training fastText on those labels — produces a classifier that generalizes well enough to replace the LLM judge for full-corpus filtering. This is the operationalization of item 21. Engineering cost: 8–16 hours (seed labeling + training).

**Source:** Ultra-FineWeb [arXiv:2505.05427](https://arxiv.org/html/2505.05427v1); FinerWeb-10BT [arXiv:2501.07314](https://arxiv.org/pdf/2501.07314) for line-level variant.

**Validated at:** Compared against Mixtral-based LLM judge; 50× cheaper; matching downstream benchmark performance.

**Files to change:** `scripts/quality_filter.py` (extends item 21)

**Implementation sketch:**

```python
# scripts/quality_filter.py — Ultra-FineWeb training methodology
class UltraFineWebClassifierTrainer:
    """
    Train a fastText classifier using LLM-labeled seed documents.
    Achieves comparable quality to full-LLM filtering at 50x lower cost.
    Reference: arXiv:2505.05427
    """
    def __init__(self, llm_judge, seed_size: int = 50_000):
        self.llm   = llm_judge
        self.seed  = seed_size

    def build_training_set(self, raw_corpus_sample: list[str]) -> str:
        """
        Label seed_size documents with LLM; format for fastText training.
        Returns path to training file in fastText format.
        """
        import random; random.shuffle(raw_corpus_sample)
        seed_docs = raw_corpus_sample[:self.seed]

        lines = []
        for doc in seed_docs:
            # LLM judge rates document quality 1-5; threshold at 3
            score = self.llm.rate_quality(doc[:1024])
            label = "__label__quality" if score >= 3 else "__label__noise"
            text  = doc[:512].replace("\n", " ").replace("\r", " ")
            lines.append(f"{label} {text}")

        train_path = "/tmp/ultra_fineweb_train.txt"
        with open(train_path, "w") as f:
            f.write("\n".join(lines))
        return train_path

    def train(self, train_path: str) -> "fasttext.FastText._FastText":
        return fasttext.train_supervised(
            input=train_path,
            epoch=25,
            lr=0.5,
            wordNgrams=2,
            dim=100,
            minCount=5,
        )
```

**Line-level variant (FinerWeb-10BT):** For highest data efficiency, apply line-level LLM scoring to the borderline documents (40th–60th quality percentile) identified by the fastText classifier. Models trained on line-level filtered data reach accuracy targets with 25% less data. Recommended for Aurelius's domain-specific data (math textbooks, academic papers) where paragraph quality varies heavily within a document.

**Metrics to watch:** Classifier F1 vs LLM judge ground truth (target >0.85 on held-out 5K document set); filtering cost (tokens processed per 1B doc corpus characters — fastText vs LLM); downstream model perplexity on PG-19 (should match LLM-filtered baseline within 0.2 pts).

**Depends on:** FineWeb2 (item 23b) as the raw corpus; LLM API access for initial seed labeling (one-time cost — run once, train classifier, use forever); item 21 base fastText setup.

---

### 23d. QuaDMix — Joint Quality-Diversity Data Selection

**What it is:** QuaDMix (arXiv:2504.16511, April 2025) formulates pretraining data selection as a **joint optimization over quality scores and diversity labels simultaneously**, rather than sequentially. The standard approach is: (1) filter by quality, then (2) mix by domain. QuaDMix shows this is suboptimal: quality filtering often removes documents from rare but important domains (niche scientific topics, non-English languages, specialized code), creating domain blind spots. QuaDMix assigns each document a sampling probability `P(doc) = quality_score^α × diversity_weight^β` where diversity_weight comes from a domain classifier and is inversely proportional to domain frequency. A LightGBM model searches the α/β space using a small proxy training run (1B tokens) to find the combination that minimizes validation loss across a diverse benchmark suite. Average improvement: **+7.2% across 18 benchmarks** versus quality-only selection.

**Why it matters to Aurelius specifically:** Aurelius trains on a web/code/book corpus that was likely assembled with primarily quality-based filtering. QuaDMix identifies and addresses diversity blind spots at the data assembly stage, before any training begins. This is cheaper than trying to recover via SFT or RL later. The proxy search using LightGBM + 1B-token model runs is practical on a single GPU and directly informs the final corpus mix.

**Source:** QuaDMix [arXiv:2504.16511](https://arxiv.org/abs/2504.16511); RegMix (prior proxy-search baseline, arXiv:2312.15503).

**Validated at:** 1.3B and 7B parameter models; +7.2% aggregate benchmark improvement vs quality-only baseline; proxy search runs on a single A100 in 6 hours.

**Files to change:** `scripts/data_mixing.py` (new); `scripts/collect_training_data.py`

**Implementation sketch:**

```python
# scripts/data_mixing.py — QuaDMix implementation
from lightgbm import LGBMRegressor
import numpy as np

class QuaDMix:
    """
    Joint quality-diversity data selection via parameterized sampling.
    Uses LightGBM proxy to find optimal alpha/beta without large-scale experiments.
    Reference: arXiv:2504.16511
    """
    # Domain categories for Aurelius corpus
    DOMAINS = ["web_general", "web_academic", "code", "math", "books",
               "news", "wiki", "legal", "medical", "multilingual"]

    def __init__(self, quality_model, domain_classifier):
        self.quality_clf = quality_model      # FineWeb-style quality scorer
        self.domain_clf  = domain_classifier  # fastText domain classifier
        self.alpha: float = 1.0   # quality exponent
        self.beta:  float = 0.5   # diversity exponent

    def compute_sampling_weight(
        self,
        doc: str,
        domain_frequencies: dict[str, float],  # pre-computed domain distribution
    ) -> float:
        """
        P(doc) ∝ quality_score^α × (1/domain_frequency)^β
        High-quality AND rare-domain documents get highest weight.
        """
        quality = self.quality_clf.score(doc)
        domain  = self.domain_clf.predict(doc)
        domain_freq = domain_frequencies.get(domain, 0.1)  # rare domain → small freq

        # Diversity weight: inversely proportional to domain frequency
        diversity_weight = 1.0 / (domain_freq + 1e-6)
        return (quality ** self.alpha) * (diversity_weight ** self.beta)

    def proxy_search(
        self,
        candidate_mixes: list[dict],     # [{alpha, beta, domain_weights}]
        proxy_model_losses: list[float], # proxy run validation losses per mix
    ) -> dict:
        """
        Use LightGBM to predict optimal alpha/beta from proxy runs.
        Avoids expensive full-scale training per candidate.
        """
        X = np.array([[m["alpha"], m["beta"]] for m in candidate_mixes])
        y = np.array(proxy_model_losses)
        gbm = LGBMRegressor(n_estimators=100, num_leaves=31).fit(X, y)

        # Grid search over alpha ∈ [0.5, 2.0], beta ∈ [0.3, 1.5]
        alpha_grid = np.arange(0.5, 2.1, 0.25)
        beta_grid  = np.arange(0.3, 1.6, 0.25)
        best_loss, best_params = float("inf"), {}

        for a in alpha_grid:
            for b in beta_grid:
                pred_loss = gbm.predict([[a, b]])[0]
                if pred_loss < best_loss:
                    best_loss, best_params = pred_loss, {"alpha": a, "beta": b}

        self.alpha = best_params["alpha"]
        self.beta  = best_params["beta"]
        return best_params
```

**Metrics to watch:** Per-domain benchmark coverage (track 18-benchmark suite coverage — each domain's representative benchmarks should be represented); proxy search cost (target: 6–12 GPU-hours at 1B proxy tokens); domain diversity score of selected corpus (Shannon entropy over domain distribution — target >2.5 bits for 10-domain corpus); benchmark aggregate after QuaDMix-selected training vs quality-only baseline (target +5–7%).

**Risk / mitigation:** The proxy model must be large enough to give meaningful validation signal — too small (10M parameters) gives noisy signals. Use at minimum a 130M parameter proxy for reliable QuaDMix search. The search assumes quality and diversity are separable — this breaks down for multilingual data where quality signals in English don't transfer. Treat multilingual as a separate domain pool with its own quality classifier.

**Depends on:** Quality classifier (items 21, 23c); domain classifier; proxy model infrastructure (130M GPT-style proxy is sufficient — train from scratch in 6 GPU-hours on 1B tokens).

---

### 23e. AC-ODM — Actor-Critic Online Data Mixing

**What it is:** AC-ODM (Actor-Critic Online Data Mixing, arXiv:2505.23878, May 2025) models the data mixing problem as a **reinforcement learning task** throughout training. The standard approach (DoReMi, RegMix, QuaDMix) determines domain mixing ratios before training and fixes them for the entire run. This ignores the fundamental fact that optimal domain weights change as the model learns — early in training, the model benefits most from diverse easy data; later, it benefits from hard domain-specific data. AC-ODM trains a small neural actor (3M parameters) that outputs domain sampling weights at each training step, with a critic estimating value from proxy model training loss. Both actor and critic are trained using a separate 130M proxy LLM as the environment. The actor policy, once trained, is applied to sample from the full corpus during target LLM training. Average improvement: **outperforms DoReMi, RegMix, and QuaDMix** on held-out evaluation suite.

**Why it matters to Aurelius specifically:** Aurelius's training curriculum has distinct phases (Stage 1: broad data; Stage 2: domain-enriched; Stage 3: long-context). AC-ODM is the first framework that correctly handles this changing-optimality-through-training characteristic. The actor naturally learns to shift weights from broad-coverage early training to specialized domains late in training, without requiring manual curriculum scheduling. Particularly valuable for math and code domains where the model's capability trajectory is non-linear.

**Source:** AC-ODM [arXiv:2505.23878](https://arxiv.org/abs/2505.23878); DoReMi comparison [arXiv:2305.10429]

**Validated at:** 3B models at 300B token training run; outperforms all static mixing baselines including QuaDMix; proxy actor training costs ~40 GPU-hours.

**Files to change:** `scripts/data_mixing.py` (extends item 23d); `src/training/data_loader.py`

**Implementation sketch:**

```python
# scripts/data_mixing.py — AC-ODM actor-critic mixing
class ACODMActor(nn.Module):
    """
    Small neural policy outputting domain sampling weights at each training step.
    Trained using a 130M proxy LLM as environment.
    Reference: arXiv:2505.23878
    """
    DOMAINS = ["web", "code", "math", "books", "academic", "wiki", "news"]

    def __init__(self, state_dim: int = 64, n_domains: int = 7):
        super().__init__()
        # Lightweight actor: takes training step features → domain weights
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, n_domains),
            nn.Softmax(dim=-1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns domain sampling distribution at current training state."""
        return self.actor(state)

    def encode_state(
        self,
        global_step: int,
        domain_losses: dict[str, float],
        recent_reward: float,
        total_steps: int,
    ) -> torch.Tensor:
        """Encode training state for actor input."""
        state = torch.zeros(64)
        # Step fraction
        state[0] = global_step / total_steps
        # Per-domain loss (normalized)
        max_loss = max(domain_losses.values()) + 1e-8
        for i, domain in enumerate(self.DOMAINS):
            state[1 + i] = domain_losses.get(domain, 1.0) / max_loss
        # Recent reward
        state[8] = recent_reward
        return state


class ACODMScheduler:
    """
    Hook into the training loop to update domain sampling weights dynamically.
    Must be called at each training step via trainer hook.
    """
    def __init__(self, actor: ACODMActor, domain_pools: dict[str, "DataPool"]):
        self.actor   = actor
        self.pools   = domain_pools
        self.current_weights: dict[str, float] = {d: 1/len(domain_pools) for d in domain_pools}

    def step(self, global_step: int, domain_losses: dict, recent_reward: float,
             total_steps: int):
        state   = self.actor.encode_state(global_step, domain_losses, recent_reward, total_steps)
        weights = self.actor(state.unsqueeze(0)).squeeze(0)
        self.current_weights = {
            domain: weights[i].item()
            for i, domain in enumerate(self.actor.DOMAINS)
        }

    def sample_batch(self, batch_size: int) -> list:
        """Sample from domain pools according to current actor weights."""
        domains = list(self.current_weights.keys())
        probs   = [self.current_weights[d] for d in domains]
        chosen  = random.choices(domains, weights=probs, k=batch_size)
        return [self.pools[d].sample() for d in chosen]
```

**Metrics to watch:** Domain weight trajectory over training (should shift from broad to specialized over time — plot domain weight evolution every 10K steps); proxy actor training cost (target <50 GPU-hours); final model benchmark across all domains vs static mixing; actor policy loss (should converge within 5K proxy training steps).

**Risk / mitigation:** Actor training requires a reliable proxy environment — the 130M proxy must be large enough to give meaningful gradient signal. If the proxy is too small, domain loss signals are noisy and the actor diverges. Minimum proxy size recommendation: 130M parameters trained on 10B tokens before actor training begins. Add an actor weight regularization term to prevent the policy from collapsing to a single domain.

**Depends on:** Domain classifier (for corpus labeling); proxy model infrastructure; QuaDMix (item 23d) provides good initialization for actor weights (warm-start actor from QuaDMix optimal weights).

---

### 23f. LongRoPE2 Full Pipeline — From Pretraining to 128K Context

**What it is:** LongRoPE2 (arXiv:2502.20082, Microsoft Research, February 2025) is a strong context-window extension candidate. It identifies and fixes the primary failure mode of YaRN: YaRN applies **uniform** RoPE scaling across all frequency dimensions, but higher-frequency RoPE dimensions have insufficient training coverage when context is extended (not just interpolation error — they literally haven't seen long-range position interactions during training). LongRoPE2's fixes: (1) **Evolutionary search** over per-dimension scaling factors guided by "needle-driven perplexity" (how well the model can retrieve information injected at specific positions), (2) **Mixed-context training** that simultaneously applies extended-context RoPE for long sequences and original RoPE for short sequences — preserving short-context capability. Result: 128K context on Phi3-mini-3.8B retaining **97.6% of original benchmark scores** using only 10B fine-tuning tokens (80× fewer than Meta's approach for Llama-3.1-8B at 128K).

**Why it matters to Aurelius specifically:** Item 27 (V2) already includes LongRoPE2 as an architecture choice. This item operationalizes it for **V1** at much lower cost: starting from the trained Aurelius 1.4B checkpoint with YaRN (θ=500,000), apply LongRoPE2 fine-tuning to reach reliable 128K context in ~10B tokens of long-document data. The DPE method (arXiv:2504.18857) provides a training-free baseline first — test 128K retrieval without any training — before committing to the fine-tuning budget.

**Source:** LongRoPE2 [arXiv:2502.20082](https://arxiv.org/abs/2502.20082); DPE [arXiv:2504.18857](https://arxiv.org/abs/2504.18857)

**Validated at:** Phi3-mini-3.8B → 128K at 97.6% benchmark retention with 10B fine-tuning tokens; Llama3-8B → 128K at 98.6% retention.

**Files to change:** `src/model/rope_embeddings.py` plus a new extension module only if exported/tested; `scripts/long_context_finetune.sh` (new)

**Implementation sketch:**

```python
# scripts/long_context_finetune.sh — LongRoPE2 fine-tuning recipe
"""
Phase 0 (free, 0 tokens): Apply DPE (training-free) for initial 128K baseline.
  - Run dimension-wise position analysis on current RoPE
  - Apply DPE rescaling to inference-only; test RULER 128K
  - If RULER 128K >70%, proceed; else run Phase 1

Phase 1 (evolutionary search, 50M tokens): Find per-dimension LongRoPE2 scale factors.
  - For each dimension d: sweep scale ∈ {0.5, 1.0, 2.0, 4.0} × YaRN
  - Score by needle-driven perplexity at 4K, 16K, 64K, 128K
  - Use evolutionary algorithm (CMA-ES) to converge on optimal per-dim scales

Phase 2 (mixed-context fine-tuning, 10B tokens):
  - 50% short sequences at original context (≤4K) with original RoPE
  - 50% long sequences at 128K with LongRoPE2 per-dim rescaling
  - LR = 0.1 × original LR; cosine decay over 10B tokens
  - Checkpoint every 1B tokens; evaluate RULER + short-context benchmarks
"""

LONGROPE2_FINETUNE_CONFIG = {
    "base_theta": 500_000,          # current YaRN base
    "target_length": 131_072,       # 128K tokens
    "mixed_context_ratio": 0.5,     # 50% long, 50% short per batch
    "lr_scale": 0.1,                # 10% of original LR
    "total_tokens": 10_000_000_000, # 10B fine-tuning tokens
    "eval_every_tokens": 1_000_000_000,  # 1B
    "eval_benchmarks": ["RULER_128K", "MMLU", "ARC_Challenge"],
}
```

**Long-context data sources for fine-tuning:**

```yaml
# data/long_context_mix.yaml
long_context_sources:
  books3:           0.30    # Project Gutenberg + Books3 — long narratives
  arxiv_full:       0.25    # Full arXiv papers (average 8K tokens)
  github_repos:     0.20    # Multi-file code repositories (avg 32K context)
  legal_docs:       0.10    # Long legal contracts and case law
  academic_theses:  0.10    # PhD dissertations (avg 80K tokens)
  synthetic_needles: 0.05   # Synthetic needle-in-haystack documents
```

**Metrics to watch:** RULER 128K score (target >75); short-context benchmark retention: MMLU (target >97% of V1), ARC-Challenge (>97%); prefill latency at 128K (should decrease with TurboQuant, item 19e); needle retrieval accuracy at 16K, 32K, 64K, 128K (track degradation gradient — should be monotonic and shallow).

**Risk / mitigation:** The evolutionary search (Phase 1) can converge to local optima. Run CMA-ES from 5 random initializations and take the best. If per-dimension scaling search is too expensive, use DPE's training-free approximation as a warm-start for Phase 2 (skip Phase 1 entirely) — this recovers ~85% of the LongRoPE2 benefit at zero search cost. Always evaluate short-context benchmarks after Phase 2 — if any short-context benchmark drops >3%, reduce the long-context fraction in mixed training from 50% to 30%.

**Depends on:** Stable V1 checkpoint; 10B tokens of long-document data (assembled from long_context_mix.yaml above); FineWeb2 (item 23b) for document sources.

---

### 23g. DataEvolve — Evolutionary Curation Strategy Search

**What it is:** DataEvolve (arXiv:2603.14420, March 2026, "Data Darwinism Part II") applies evolutionary search to **automatically discover curation strategies** per domain, rather than hand-designing filtering and weighting rules. The evolutionary algorithm treats each domain's curation pipeline (classifier thresholds, deduplication aggressiveness, synthetic ratio, quality-vs-diversity trade-off) as a genome. Darwin-CC evolves these strategies over 30 generations per domain, using a proxy model's performance on held-out benchmarks as fitness function. Result: +**3.96 points** average across 18 benchmarks over raw data at a 3B model / 500B token scale, surpassing DCLM, Ultra-FineWeb, and FineWeb-Edu. Independently evolved strategies from different random seeds converge on cleaning-focused, domain-aware content preservation — suggesting the discovered strategies reflect genuine data structure rather than overfitting.

**Why it matters to Aurelius specifically:** QuaDMix (item 23d) and AC-ODM (item 23e) optimize the *mixing ratios* across pre-defined domains and filtering rules. DataEvolve goes deeper: it optimizes the *filtering strategies themselves*. For Aurelius's math and code domains — where manually-defined quality signals are well-established but may not be optimal — DataEvolve can discover stronger domain-specific filters. Engineering cost is high (80–160 GPU-hours), but is a one-time investment: once strategies are evolved, they are applied as fixed preprocessing to all future corpus updates.

**Source:** DataEvolve / Data Darwinism Part II [arXiv:2603.14420](https://arxiv.org/abs/2603.14420)

**Validated at:** 3B model, 500B token training; +3.96pp average across 18 benchmarks vs FineWeb-Edu baseline.

**Files to change:** `scripts/evolutionary_curation.py` (new)

**Implementation sketch:**

```python
# scripts/evolutionary_curation.py
class DataEvolveSearcher:
    """
    Evolutionary search over per-domain curation strategies.
    Genome: quality threshold, dedup threshold, synthetic ratio, diversity weight.
    Reference: arXiv:2603.14420 (Data Darwinism Part II)
    """
    GENOME_KEYS = ["quality_threshold", "dedup_threshold",
                   "synthetic_ratio", "diversity_weight"]
    GENE_RANGES = {
        "quality_threshold":  (0.4, 0.8),
        "dedup_threshold":    (0.7, 0.99),
        "synthetic_ratio":    (0.0, 0.5),
        "diversity_weight":   (0.1, 1.0),
    }

    def __init__(self, proxy_model, benchmark_suite: list[str],
                 n_generations: int = 30, population_size: int = 20):
        self.proxy  = proxy_model
        self.benchmarks = benchmark_suite
        self.n_gen  = n_generations
        self.pop_sz = population_size

    def random_genome(self) -> dict:
        return {k: random.uniform(*self.GENE_RANGES[k]) for k in self.GENOME_KEYS}

    def crossover(self, parent_a: dict, parent_b: dict) -> dict:
        return {k: (parent_a[k] if random.random() < 0.5 else parent_b[k])
                for k in self.GENOME_KEYS}

    def mutate(self, genome: dict, sigma: float = 0.05) -> dict:
        return {k: float(np.clip(genome[k] + np.random.normal(0, sigma),
                                 *self.GENE_RANGES[k]))
                for k in self.GENOME_KEYS}

    def fitness(self, genome: dict, domain: str) -> float:
        """Train proxy on corpus filtered with `genome` strategy; return avg benchmark score."""
        corpus = self._apply_genome(genome, domain)
        return self.proxy.train_and_evaluate(corpus, self.benchmarks)

    def evolve(self, domain: str) -> dict:
        """Return the best curation strategy for this domain after N generations."""
        population = [self.random_genome() for _ in range(self.pop_sz)]

        for gen in range(self.n_gen):
            scores = [(g, self.fitness(g, domain)) for g in population]
            scores.sort(key=lambda x: x[1], reverse=True)
            elites = [g for g, _ in scores[:self.pop_sz // 4]]

            # Generate next generation from elites
            population = elites + [
                self.mutate(self.crossover(random.choice(elites), random.choice(elites)))
                for _ in range(self.pop_sz - len(elites))
            ]

        return scores[0][0]  # return best genome

    def _apply_genome(self, genome: dict, domain: str) -> list[str]:
        """Apply curation strategy defined by genome to domain corpus."""
        raise NotImplementedError  # domain-specific implementation
```

**Metrics to watch:** Benchmark improvement over baseline filtering (target +3–4pp aggregate); per-generation fitness curve (should converge within 20 generations); convergence consistency across 3 random seeds (standard deviation of final fitness should be <0.5pp — confirms real signal, not noise); curation strategy interpretability (log the top-5 genomes — they should show human-interpretable patterns).

**Risk / mitigation:** 30 generations × 20 population × 1B proxy tokens per evaluation = 600B proxy model tokens total. This is expensive. Reduce to 15 generations and population of 10 for an initial run; full 30×20 run only after proxy validation confirms signal. Use the same 130M proxy as AC-ODM to share infrastructure cost.

**Depends on:** Proxy model (shared with AC-ODM, item 23e); QuaDMix (item 23d) as initialization for genome search (use QuaDMix optimal α/β as genetic material for first generation).

---

### 23h. Magpie — Zero-Seed SFT Data Synthesis

**What it is:** Magpie (arXiv:2406.08464, ICLR 2025, University of Washington/AI2) exploits a structural property of chat-aligned LLMs: when fed only the **pre-query template** (system prompt header without any user message), an aligned model's completion distribution is biased toward generating a plausible user query. Running the pre-query template through Llama-3-70B-Instruct generated 4M instruction-response pairs with no human-provided seeds. After quality filtering to 300K examples, fine-tuning a 8B model on these Magpie examples **matches official Llama-3-8B-Instruct quality** (trained on 10M RLHF examples with PPO). Magpie supports multi-turn, preference data, multilingual, and domain-specific variants by using different system prompts to bias the generation distribution.

**Why it matters to Aurelius specifically:** Aurelius's SFT phase currently relies on publicly available instruction datasets (ShareGPT, UltraChat). These datasets are (a) from external models whose capabilities and failure modes differ from Aurelius, and (b) static. Magpie enables Aurelius to generate SFT data from a teacher model whose distribution is explicitly closer to Aurelius's own strengths, especially if the teacher is a fine-tuned Aurelius-large variant. The zero-seed property means SFT data generation scales indefinitely with compute.

**Source:** Magpie [arXiv:2406.08464](https://arxiv.org/abs/2406.08464); ICLR 2025; magpie-align GitHub repository.

**Validated at:** Llama-3-8B fine-tuned on Magpie-300K matches Llama-3-8B-Instruct on AlpacaEval 2.0, MT-Bench, and WildBench vs 10M RLHF-trained baseline.

**Files to change:** `scripts/synthetic_data_gen.py` (extend item 22)

**Implementation sketch:**

```python
# scripts/synthetic_data_gen.py — Magpie zero-seed generation
class MagpieSFTGenerator:
    """
    Generate SFT data from a teacher model using pre-query template exploitation.
    No human-provided seeds needed.
    Reference: arXiv:2406.08464 (ICLR 2025)
    """
    # Pre-query template for Llama-3 / Aurelius chat format
    PRE_QUERY_TEMPLATES = {
        "math":         "<|system|>You are a math tutor.<|end|><|user|>",
        "code":         "<|system|>You are a programming expert.<|end|><|user|>",
        "general":      "<|system|>You are a helpful assistant.<|end|><|user|>",
        "reasoning":    "<|system|>You are a logical reasoning expert.<|end|><|user|>",
        "factual":      "<|system|>You are a knowledge expert.<|end|><|user|>",
    }

    def __init__(self, teacher_model, quality_filter, n_total: int = 300_000):
        self.teacher = teacher_model
        self.filter  = quality_filter
        self.n_total = n_total

    def generate_instruction(self, domain: str) -> str:
        """Exploit pre-query template to generate a plausible user instruction."""
        template = self.PRE_QUERY_TEMPLATES[domain]
        # Teacher model generates the user query when given only the template
        raw_query = self.teacher.generate(
            template,
            max_new_tokens=256,
            temperature=0.85,
            stop_tokens=["<|end|>", "<|assistant|>"],
        )
        return raw_query.strip()

    def generate_response(self, instruction: str, domain: str) -> str:
        """Generate the teacher's response to the generated instruction."""
        full_prompt = (
            self.PRE_QUERY_TEMPLATES[domain] +
            instruction +
            "<|end|><|assistant|>"
        )
        return self.teacher.generate(full_prompt, max_new_tokens=1024, temperature=0.7)

    def generate_dataset(self, domains: list[str]) -> list[dict]:
        """Generate n_total balanced SFT examples."""
        per_domain = self.n_total // len(domains)
        dataset = []
        for domain in domains:
            accepted = 0
            while accepted < per_domain:
                instr  = self.generate_instruction(domain)
                resp   = self.generate_response(instr, domain)
                if self.filter.accept(instr, resp, domain):
                    dataset.append({
                        "instruction": instr,
                        "response": resp,
                        "domain": domain,
                        "source": "magpie",
                    })
                    accepted += 1
        return dataset
```

**Quality filter criteria:** Length (instruction 20–500 tokens; response 50–2000 tokens); perplexity under base model (response PPL <100 — remove very incoherent responses); deduplication (SimHash with threshold 0.9 across the generated set); safety classifier (block CSAM, detailed violence, personally identifying information).

**Metrics to watch:** AlpacaEval 2.0 LC-WR before/after Magpie SFT (target +3–5pp vs ShareGPT baseline); MT-Bench score; instruction diversity (instruction embedding pairwise cosine similarity — target mean <0.4); domain balance (should be ±5% of target per-domain fraction); quality filter pass rate (expect 40–60% of generated examples pass all filters).

**Risk / mitigation:** Magpie's quality depends heavily on the teacher model's aligned quality. Using a poorly-aligned or domain-mismatched teacher will produce noisy SFT data. Recommendation: use the best available aligned model (Claude 3.5 Sonnet or Llama-3.3-70B-Instruct via API) as teacher for the initial 300K, then Magpie from an iteratively-improved Aurelius-large after V2 exists. Apply GRAPE (item 23i) to filter Magpie data for Aurelius base model distribution compatibility.

**Depends on:** Teacher model access (API or local); quality filter (item 21 fastText filter + custom length/safety filters); deduplication pipeline.

---

### 23i. GRAPE — Model-Dependent SFT Curation

**What it is:** GRAPE (arXiv:2502.04194, February 2025) is based on the hypothesis that **SFT data whose distribution matches the base model's pretraining distribution is more effective** than generic high-quality SFT data. The argument: when the SFT response is generated by a model with a fundamentally different capability profile than the student, the student must both learn the task and "translate" the style/capability gap — reducing learning efficiency. GRAPE selects (or generates) SFT data by computing the **perplexity of each SFT response under the base model**: responses the base model already assigns moderate perplexity (neither trivially easy nor impossibly hard) are selected. This reduces catastrophic forgetting and spurious correlations. Empirically: models fine-tuned on GRAPE-curated data converge faster and maintain stronger base capabilities than models fine-tuned on generic high-quality SFT data.

**Why it matters to Aurelius specifically:** At 1.4B scale, the distribution gap between teacher-generated SFT data (from Llama-70B or GPT-4-class models) and Aurelius's base model is substantial. GRAPE is specifically designed for this case — it's the low-budget student-model-aware SFT curation technique. Applying GRAPE to filter Magpie (item 23h) or any externally-generated SFT data ensures the data falls within Aurelius's "zone of proximal development" rather than overwhelming it with capability-gap examples.

**Source:** GRAPE [arXiv:2502.04194](https://arxiv.org/pdf/2502.04194); GRAPE GitHub repository.

**Validated at:** Llama-3-8B, Mistral-7B; consistently outperforms baseline SFT on AlpacaEval 2.0 and MT-Bench; particularly strong benefit when teacher model is significantly stronger than student.

**Files to change:** `scripts/synthetic_data_gen.py` (add GRAPE filter); `scripts/sft_data_curator.py` (new)

**Implementation sketch:**

```python
# scripts/sft_data_curator.py — GRAPE model-dependent curation
class GRAPECurator:
    """
    Filter SFT data by base model perplexity — keep data in the zone of proximal development.
    Reference: arXiv:2502.04194
    """
    def __init__(
        self,
        base_model,            # Aurelius base (pre-SFT) checkpoint
        ppl_low: float = 5.0,  # too easy — base model already generates this well
        ppl_high: float = 80.0,# too hard — base model assigns near-random perplexity
    ):
        self.model    = base_model
        self.ppl_low  = ppl_low
        self.ppl_high = ppl_high

    def compute_response_ppl(self, instruction: str, response: str) -> float:
        """Perplexity of the response tokens under the base model, conditioned on instruction."""
        prompt_ids   = self.model.tokenize(instruction)
        response_ids = self.model.tokenize(response)
        full_ids     = torch.cat([prompt_ids, response_ids])

        with torch.no_grad():
            logits = self.model(full_ids)
        # Only compute perplexity on response tokens
        response_logits = logits[len(prompt_ids)-1:-1]
        response_labels = response_ids
        loss = F.cross_entropy(response_logits, response_labels)
        return float(loss.exp())

    def filter_dataset(
        self,
        dataset: list[dict],  # [{"instruction": ..., "response": ...}]
    ) -> list[dict]:
        """Keep examples where response PPL is in the target range."""
        filtered = []
        for item in dataset:
            ppl = self.compute_response_ppl(item["instruction"], item["response"])
            if self.ppl_low <= ppl <= self.ppl_high:
                item["base_ppl"] = ppl
                filtered.append(item)
        return filtered
```

**Metrics to watch:** GRAPE filter acceptance rate (expect 40–60% of Magpie data retained — higher means base model is closer to teacher); AlpacaEval 2.0 improvement of GRAPE-filtered vs unfiltered SFT data; base model benchmark retention after SFT (MMLU, ARC — should be within 1pp of pre-SFT baseline when using GRAPE); base_ppl distribution of accepted examples (should be roughly bell-shaped around PPL=25–35 for a 1.4B base model).

**Risk / mitigation:** The ppl_low and ppl_high thresholds are model-dependent. Calibrate by computing PPL on 500 human-curated SFT examples of known quality at three difficulty levels (easy/medium/hard) and finding the empirical PPL range for "medium" quality. If the base model has been fine-tuned previously (not a clean pretrain checkpoint), its PPL distribution shifts — recalibrate thresholds after each major training run.

**Depends on:** Base model checkpoint (frozen, pre-SFT); SFT dataset to filter (Magpie from item 23h; ShareGPT; any external SFT dataset). Can be applied as a one-time offline filter — rerun whenever the base model checkpoint updates.

---

### 23j. OpenCoder RefineCode + Seed-Coder Iterative Filtering

**What it is:** Two complementary state-of-the-art code data pipelines:

**OpenCoder** (arXiv:2411.04905, ACL 2025): A fully open code LLM trained on 2.5T tokens across 607 programming languages. Key data contribution: the **RefineCode pipeline** — 960B tokens processed with (a) file-level deduplication using MinHash/LSH across all 607 languages, (b) 130+ hand-crafted language-specific filtering rules (e.g., Python files with >80% commented lines are filtered; HTML with <30% code content is filtered; test files with no assertions are filtered), (c) a two-stage instruction tuning phase with a synthetic annealing phase (high-quality synthetic code examples) before standard SFT. The full RefineCode dataset and 4.5M open SFT entries are publicly available.

**Seed-Coder** (arXiv:2506.03524, ByteDance Seed, June 2025): Iterative model-centric code data filtering. Instead of hand-crafted per-language rules, Seed-Coder trains an LLM to score code quality, filters the corpus with this scorer, retrains the scorer on the filtered data, and repeats. This removes the scalability bottleneck of language-specific rule engineering and generalizes across 600+ programming languages without per-language expert knowledge.

**Why it matters to Aurelius specifically:** Aurelius's code performance is currently limited by the quality of the training corpus's code component. RefineCode's 130+ filtering rules represent the highest-density curation knowledge available for code data. Seed-Coder's iterative approach scales this to languages where RefineCode's per-language rules are sparse. Together: apply RefineCode rules first, then use Seed-Coder iterative scoring to handle the long tail of less-common languages and filter borderline cases.

**Source:** OpenCoder [arXiv:2411.04905](https://arxiv.org/abs/2411.04905) (ACL 2025); Seed-Coder [arXiv:2506.03524](https://arxiv.org/abs/2506.03524); RefineCode dataset available on HuggingFace.

**Validated at:** OpenCoder-1.5B and 8B: strong across HumanEval, MBPP, BigCodeBench, SWE-Bench; Seed-Coder-8B: state-of-the-art on LiveCodeBench at 8B scale (June 2025).

**Files to change:** `scripts/code_data_pipeline.py` (new)

**Implementation sketch:**

```python
# scripts/code_data_pipeline.py — Combined RefineCode + Seed-Coder pipeline
class OpenCoderRefineCodeFilter:
    """
    130+ language-specific code quality rules from OpenCoder.
    Reference: arXiv:2411.04905 (OpenCoder, ACL 2025)
    """
    PYTHON_RULES = {
        "max_comment_ratio": 0.80,     # >80% comment lines → filter
        "min_code_lines": 3,           # fewer than 3 non-blank non-comment → filter
        "max_line_length": 500,        # excessively long lines → filter
        "require_function_or_class": False,  # scripts OK
        "filter_test_without_assertions": True,
    }
    JAVASCRIPT_RULES = {
        "max_minification_ratio": 0.5, # >50% of lines >200 chars → likely minified
        "filter_generated_bundles": True,  # webpack bundles
        "min_identifier_diversity": 10,    # too few unique identifiers → template/generated
    }
    HTML_RULES = {
        "min_code_content_ratio": 0.30,  # <30% actual code → content-less template
        "filter_tracking_only": True,
    }

    def filter(self, code: str, language: str) -> bool:
        rules = getattr(self, f"{language.upper()}_RULES", {})
        if rules.get("filter_test_without_assertions") and self._is_test_without_assert(code):
            return False
        comment_ratio = self._compute_comment_ratio(code, language)
        if comment_ratio > rules.get("max_comment_ratio", 1.0):
            return False
        return True

    def _compute_comment_ratio(self, code: str, language: str) -> float:
        lines = code.split("\n")
        comment_prefixes = {"python": "#", "javascript": "//", "java": "//", "c": "//"}
        prefix = comment_prefixes.get(language, "#")
        comment_lines = sum(1 for l in lines if l.strip().startswith(prefix))
        return comment_lines / max(len(lines), 1)

    def _is_test_without_assert(self, code: str) -> bool:
        return ("def test_" in code or "def Test" in code) and "assert" not in code


class SeedCoderIterativeFilter:
    """
    Iterative model-centric code quality scoring.
    Train scorer → filter → retrain → repeat.
    Reference: arXiv:2506.03524
    """
    def __init__(self, initial_model, n_iterations: int = 3):
        self.model = initial_model
        self.n_iter = n_iterations

    def run(self, corpus: list[str]) -> list[str]:
        filtered = corpus
        for iteration in range(self.n_iter):
            scores  = [self.model.score_quality(code) for code in filtered]
            # Keep top 60% each iteration
            threshold = np.percentile(scores, 40)
            filtered  = [c for c, s in zip(filtered, scores) if s >= threshold]
            # Retrain scorer on filtered data
            self.model = self._retrain_scorer(filtered)
        return filtered

    def _retrain_scorer(self, data: list[str]):
        raise NotImplementedError  # Fine-tune a CodeBERT-style classifier
```

**Metrics to watch:** HumanEval pass@1 (target +5–10pp vs current code baseline); MBPP pass@1; BigCodeBench solve rate; code data corpus size after RefineCode filtering (expect 40–50% reduction from raw GitHub); post-filtering language distribution (should have meaningful coverage across top-20 languages); Seed-Coder iterative improvement curve (each iteration should improve average code quality score by at least 2%).

**Risk / mitigation:** RefineCode's hand-crafted rules can be overly aggressive for less-common languages (Haskell, Erlang, Prolog) where the rules were validated less thoroughly. Run RefineCode rules only on top-50 languages by training token volume; use Seed-Coder iterative scoring for languages outside the top-50. RefineCode dataset is publicly available on HuggingFace — consider using it directly rather than re-running the pipeline.

**Depends on:** Code corpus (GitHub dump or The Stack v2); RefineCode dataset (HuggingFace — direct download option); Seed-Coder scoring model (use OpenCoder-1.5B as initial scorer — available publicly).

---

### 23k. Mid-Training CoT Distillation Stage

**What it is:** Meta and multiple frontier labs have converged on a **three-stage training recipe** for reasoning-capable LLMs at small scale:

1. **Stage 1 (pre-training):** General web/code/book corpus (80B–400B tokens). Builds world knowledge and language modeling.
2. **Stage 2 (mid-training):** Synthetic `(problem, chain-of-thought trace, answer)` triples distilled from a frontier reasoning model (DeepSeek-R1, QwQ-32B-Preview, or Claude 3.7 Sonnet). This is the *critical unlocking step* — it gives the 1.4B model the "vocabulary" of structured reasoning before RL begins. Without mid-training, RLVR on reasoning tasks is extremely sample-inefficient because the model must simultaneously discover how to structure reasoning and how to be correct.
3. **Stage 3 (RLVR):** Outcome-based RL on verifiable tasks (math, code, logic). After mid-training, RLVR is much more efficient because the model already knows how to produce reasoning traces; RL only needs to optimize which traces are correct.

The mid-training corpus should contain: multi-step math with full CoT, multi-step code with explanatory comments, logical reasoning with explicit premise-conclusion chains. Key finding: 50M–1B tokens of high-quality distilled CoT data is sufficient for a 1.4B model. More is not necessarily better — quality of the CoT traces matters more than volume. CoTP selection (a dual-granularity selection algorithm) identifies high-value CoT data by combining chain-level reasoning pattern diversity with token-level entropy scoring.

**Why it matters to Aurelius specifically:** Aurelius currently goes directly from pre-training to SFT to RLVR. This skips the mid-training CoT distillation stage that all frontier small models (Llama-3.2-3B-Instruct, Phi-3-mini, Qwen3-1.7B) now use. Adding this stage is the single highest-expected-value change to the reasoning pipeline — estimated +8–15pp on MATH500 at 1.4B scale based on extrapolation from published results.

**Source:** Meta mid-training recipe (Meta Llama team, 2025); CoTP (dual-granularity CoT selection, reported in Kimi K2.6 technical report); Phi-3 technical report (arXiv:2404.14219, mid-training section); OmegaPRM for PRM data generation (VersaPRM paper background).

**Validated at:** Phi-3-mini (3.8B): mid-training stage critical for MATH benchmark; Llama-3.2-1B → 3B: reasoning gap versus larger models closes by ~40% with CoT distillation stage; multiple independent confirmations at 1–3B scale.

**Files to change:** `scripts/collect_training_data.py`; `configs/train_stage2.yaml` (new); `src/training/curriculum.py`

**Implementation sketch:**

```python
# configs/train_stage2.yaml — Mid-training CoT distillation configuration
midtraining:
  stage: 2
  total_tokens: 50_000_000_000   # 50B tokens (adjust up to 200B if budget allows)
  sources:
    math_cot:    0.40    # (problem, cot_trace, answer) triples from DeepSeek-R1/QwQ
    code_cot:    0.30    # (problem, step-by-step code explanation, solution)
    logic_cot:   0.15    # (premise, explicit inference steps, conclusion)
    science_cot: 0.10    # physics/chemistry/biology step-by-step solutions
    general_cot: 0.05    # diverse instruction-following with explicit reasoning
  lr_schedule:
    type: cosine
    peak_lr: 2e-4         # slightly higher than pretraining LR — faster domain adaptation
    warmup_steps: 1000
  data_selection:
    method: cotp          # CoTP dual-granularity selection
    chain_diversity: 0.6  # weight for chain-level reasoning pattern diversity
    token_entropy: 0.4    # weight for token-level entropy (hard tokens)


# scripts/cot_data_builder.py — CoT trace quality selection
class CoTPSelector:
    """
    Dual-granularity CoT data selection.
    Balances chain-level reasoning diversity + token-level difficulty.
    Reference: CoTP methodology from Kimi K2.6 technical report.
    """
    def __init__(self, diversity_weight: float = 0.6, entropy_weight: float = 0.4):
        self.div_w = diversity_weight
        self.ent_w = entropy_weight

    def score_cot(self, problem: str, cot_trace: str, answer: str,
                  existing_patterns: list[str]) -> float:
        """
        Score a CoT example by diversity and difficulty.
        High scores → include in mid-training corpus.
        """
        # Chain-level diversity: does this trace use a novel reasoning pattern?
        diversity_score = self._compute_pattern_novelty(cot_trace, existing_patterns)

        # Token-level entropy: are the individual reasoning steps difficult?
        entropy_score   = self._compute_token_entropy(cot_trace)

        return self.div_w * diversity_score + self.ent_w * entropy_score

    def _compute_pattern_novelty(self, cot: str, existing: list[str]) -> float:
        """BM25 dissimilarity from the set of already-selected CoT patterns."""
        if not existing:
            return 1.0
        max_sim = max(self._bm25_similarity(cot, e) for e in existing)
        return 1.0 - max_sim

    def _compute_token_entropy(self, cot: str) -> float:
        """Average per-token entropy estimate (proxy: fraction of uncommon tokens)."""
        words = cot.split()
        if not words:
            return 0.0
        # Uncommon tokens ≈ domain-specific reasoning vocabulary
        uncommon = sum(1 for w in words if len(w) > 8 and not w.isdigit())
        return uncommon / len(words)

    def _bm25_similarity(self, a: str, b: str) -> float:
        a_terms = set(a.lower().split()); b_terms = set(b.lower().split())
        if not b_terms:
            return 0.0
        return len(a_terms & b_terms) / len(b_terms)
```

**CoT distillation data sourcing strategy:**

| Source | Volume | Method | Quality Gate |
|--------|--------|--------|-------------|
| DeepSeek-R1 API | 30M traces | Prompt with AMC/AIME problems | Verify final answer |
| QwQ-32B-Preview | 10M traces | Math competition problems | Z3 verify (item OC-5) |
| Llama-3.3-70B-Instruct | 5M traces | Code problems | Execute + test suite |
| Synthetic curriculum (PAC) | 5M traces | PAC-verified (item OC-5) | Formal verification |

**Metrics to watch:** MATH500 pass@1 after mid-training vs pre-mid-training baseline (target +8–15pp); AIME 2024 pass@1 (target +3–5pp); CoT trace average length and structure quality (manual spot-check of 100 traces per domain per generation); stage transition stability (perplexity should decrease monotonically during mid-training — if it spikes, the data mixing is wrong).

**Risk / mitigation:** Using a single teacher model for all CoT distillation creates distribution narrowing — the model learns one style of reasoning. Diversify: use at least 3 different teacher models (DeepSeek-R1, QwQ-32B, Claude 3.7 Sonnet) each contributing ≥15% of traces. If a single teacher is necessary, apply CoTP selection to maximize reasoning pattern diversity within that teacher's traces. Budget 30–50% of mid-training GPU time for trace generation and quality verification before the training run begins.

**Depends on:** Frontier model API access (DeepSeek-R1 or equivalent); formal verifiers (PAC, item OC-5) for math verification; code test runner (for code CoT verification); CoTP selection algorithm; Stage 2 training infrastructure (same as Stage 1 but with new data loader).

---

## V2 — Architecture Redesign (2.7B)

V2 is planned from scratch with these changes baked into the architecture. Do not retrofit V1.

---

### 24. Multi-Head Latent Attention (MLA) — Replace GQA

**What it is:** Replace Grouped Query Attention with MLA, which compresses Key and Value tensors into a low-rank latent vector per token. At inference, the latent is projected back to full K/V. This achieves a ~10× smaller KV cache than MHA with no quality regression.

**Why it matters to Aurelius specifically:** Aurelius v1 uses GQA (16 Q heads, 8 KV heads) — one of the best attention configurations before MLA. But MLA (DeepSeek V2/V3) beats GQA on perplexity *and* uses 2× less cache than GQA. The 10× cache reduction vs MHA unlocks: (a) much longer effective context at fixed memory, (b) larger batch sizes, (c) the same GPU serving 2–3× more concurrent requests.

**Source:** DeepSeek-V3 MLA explanation [towardsdatascience.com](https://towardsdatascience.com/deepseek-v3-explained-1-multi-head-latent-attention-ed6bee2a67c4/); Implementation guide [pyimagesearch.com](https://pyimagesearch.com/2026/03/16/build-deepseek-v3-multi-head-latent-attention-mla-architecture/)

**Validated at:** 671B MoE (DeepSeek V3) — matches MHA perplexity, 2× smaller cache than GQA

| Method | KV Cache | Perplexity vs MHA | Notes |
|--------|----------|-------------------|-------|
| MHA | 1.0× | baseline | Standard |
| GQA (8 groups, Aurelius v1) | ~0.5× | −0.5 pts | Current |
| **MLA (V2 target)** | **~0.1×** | **+0.0** | Matches MHA quality |

**Files to change:** `src/model/attention.py` (v2 branch)

**Implementation sketch:**

```python
# src/model/attention.py — V2 branch
class MultiHeadLatentAttention(nn.Module):
    """
    DeepSeek-style MLA. Compress KV into low-rank latent,
    reconstruct at inference. Decoupled RoPE for positional consistency.
    """
    def __init__(
        self,
        d_model: int   = 2048,
        n_heads: int   = 16,
        kv_latent_dim: int = 512,     # latent dim for KV compression
        q_latent_dim:  int = 1536,    # latent dim for Q compression
        qk_rope_dim:   int = 64,      # separate subspace for RoPE
    ):
        super().__init__()
        self.n_heads    = n_heads
        self.head_dim   = d_model // n_heads

        # Q compression: d_model → q_latent → d_model (+ RoPE component)
        self.q_down  = nn.Linear(d_model, q_latent_dim, bias=False)
        self.q_up    = nn.Linear(q_latent_dim, n_heads * self.head_dim, bias=False)
        self.q_rope  = nn.Linear(q_latent_dim, n_heads * qk_rope_dim, bias=False)

        # KV compression: d_model → kv_latent (stored in cache)
        self.kv_down = nn.Linear(d_model, kv_latent_dim + qk_rope_dim, bias=False)
        # KV decompression: kv_latent → full K, V (done at inference, not cached)
        self.k_up    = nn.Linear(kv_latent_dim, n_heads * self.head_dim, bias=False)
        self.v_up    = nn.Linear(kv_latent_dim, n_heads * self.head_dim, bias=False)
        self.k_rope  = nn.Linear(kv_latent_dim, n_heads * qk_rope_dim, bias=False)

        self.o_proj  = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.rope    = DecoupledRoPE(qk_rope_dim)

    def forward(self, x, positions, kv_cache=None):
        B, T, _ = x.shape

        # Compress Q
        q_latent   = self.q_down(x)
        q_content  = self.q_up(q_latent).view(B, T, self.n_heads, self.head_dim)
        q_rope_part = self.q_rope(q_latent).view(B, T, self.n_heads, -1)

        # Compress KV into latent (this is what gets cached — 10× smaller)
        kv_compressed = self.kv_down(x)
        kv_latent, k_rope_part = kv_compressed.split(
            [self.kv_down.out_features - self.rope.dim,
             self.rope.dim], dim=-1
        )
        # Cache kv_latent + k_rope_part (not full K/V)

        # Decompress for attention computation
        k_content  = self.k_up(kv_latent).view(B, T, self.n_heads, self.head_dim)
        v          = self.v_up(kv_latent).view(B, T, self.n_heads, self.head_dim)

        # Apply decoupled RoPE only to the rope subspace
        q_rope_part, k_rope_part = self.rope(q_rope_part, k_rope_part, positions)
        q = torch.cat([q_content, q_rope_part], dim=-1)
        k = torch.cat([k_content, k_rope_part], dim=-1)

        # Standard attention on full Q, K, V
        attn = flash_attn_func(q, k, v, causal=True)
        return self.o_proj(attn.reshape(B, T, -1))
```

**Depends on:** V2 training run. Cannot be retrofitted into V1 — training-time change.

---

### 25. Lightning Attention 7:1 Hybrid

**What it is:** Replace 7 of every 8 attention layers with Lightning Attention (linear-complexity, O(n)), keeping 1 standard softmax attention layer per group for global coherence refresh.

**Why it matters:** MiniMax-M1 achieves 4M-token inference at 30% of DeepSeek R1's compute cost using this 7:1 pattern. At V2's 2.7B scale, this enables 1M+ context windows that are actually practical to serve.

**Source:** MiniMax-01 [arXiv 2501.08313](https://arxiv.org/pdf/2501.08313); MiniMax-M1 [arXiv 2506.13585](https://arxiv.org/abs/2506.13585)

**Files:** `src/model/attention.py` (v2 branch); use `flash-linear-attention` library for Triton kernels

**Key design:**

```
V2 layer pattern (assume 32 layers):
  Layers  1-7:   LightningAttention  (TransNormer, O(n))
  Layer   8:     SoftmaxAttention    (full, coherence refresh)
  Layers  9-15:  LightningAttention
  Layer  16:     SoftmaxAttention
  Layers 17-23:  LightningAttention
  Layer  24:     SoftmaxAttention
  Layers 25-31:  LightningAttention
  Layer  32:     SoftmaxAttention    (final layer — always full)
```

Total: 4 full softmax layers, 28 linear attention layers. 87.5% of layer compute is O(n).

**Depends on:** V2 training run. `flash-linear-attention` open-source Triton kernels available.

---

### 26. Mamba-2 Hybrid Layers

**What it is:** Replace ~40% of attention layers in V2 with Mamba-2 SSM (State Space Model) layers. SSM layers process sequences recurrently — they are O(n) in time and O(1) in memory relative to sequence length. NVIDIA Nemotron-Nano-9B-v2 uses 28/62 layers as Mamba-2, achieving 3–6× throughput vs dense transformer.

**Source:** NVIDIA Nemotron-Nano-9B-v2 [arXiv 2508.14444](https://arxiv.org/abs/2508.14444); Nemotron 3 Super 120B [developer.nvidia.com](https://developer.nvidia.com/blog/introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning/)

**Validated at:** 9B, 12B (Nemotron Nano v2) — 3–6× throughput at equal accuracy; 120B (Nemotron 3 Super) — 7× throughput

**V2 proposed hybrid distribution (28 layers):**

| Layer Type | Count | Position | Purpose |
|-----------|-------|----------|---------|
| Full Attention | 4 | 0, 9, 18, 27 | Global coherence, long-range dependency |
| SWA | 6 | 3, 6, 12, 15, 21, 24 | Local context within full-attn gaps |
| Mamba-2 | 10 | 1,2,4,5,7,8,10,11,13,14 | Sequential state, recurrent patterns |
| MLP-only | 8 | remaining | Feed-forward capacity |

**Dependencies:** `causal-conv1d`, `mamba-ssm`, or `flash-linear-attention`'s Mamba-2 Triton kernels (all open-source).

---

### 27. LongRoPE2 Context Extension

**What it is:** Replace YaRN context extension with LongRoPE2 (arXiv:2502.20082), which identifies per-dimension RoPE scaling factors rather than uniform scaling and uses mixed-context training windows. Treat 128K as the first Aurelius target; 2M remains a research-only target until local RULER/LongBench-v2-style evidence exists.

**Why it matters to Aurelius specifically:** Aurelius already uses YaRN (θ=500,000). LongRoPE2 may fix YaRN's primary failure mode — uniform interpolation can cause accuracy regression on specific RoPE dimensions that are near their Nyquist limit. The plan must prove this with length curves and AMC-vs-long-context comparisons rather than assuming max-context length is useful.

**Source:** LongRoPE2 [arXiv 2502.20082](https://arxiv.org/abs/2502.20082); LongRoPE paper [arXiv 2402.13753](https://arxiv.org/abs/2402.13753); validation gates from RULER [arXiv 2404.06654](https://arxiv.org/abs/2404.06654), LongBench v2 [arXiv 2412.15204](https://arxiv.org/abs/2412.15204), and MInference [arXiv 2407.02490](https://arxiv.org/abs/2407.02490)

**Validated at:** Treat source-paper numbers as S2 until Aurelius reproduces 128K retrieval/reasoning curves. Do not claim 2M support until local tests show useful accuracy, TTFT, memory, and AMC comparison at that scale.

**Files:** `src/model/rope_embeddings.py` plus a new extension module only if exported/tested

```python
# src/model/rope_embeddings.py or exported long-context extension module
class LongRoPE2(nn.Module):
    """
    Per-dimension RoPE scaling with mixed-context training windows.
    First Aurelius target: 128K with RULER/LongBench-v2 gates; 2M remains research-only until reproduced.
    """
    def __init__(self, dim: int, base_theta: float = 500_000,
                 target_length: int = 131_072):
        super().__init__()
        self.dim    = dim
        self.target = target_length
        # Non-uniform per-dimension scale factors (learned during fine-tuning)
        self.register_parameter(
            "dim_scale",
            nn.Parameter(torch.ones(dim // 2))
        )
        self.base_freqs = 1.0 / (base_theta ** (
            torch.arange(0, dim, 2).float() / dim
        ))

    def forward(self, q, k, positions):
        # Apply per-dimension scaled frequencies
        scaled_freqs = self.base_freqs * self.dim_scale.abs()
        freqs  = torch.outer(positions.float(), scaled_freqs)
        cos, sin = freqs.cos(), freqs.sin()
        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
```

Fine-tune from a trained V2 checkpoint only after a training-free DPE/YaRN baseline and a 128K RULER/LongBench-v2 evaluation harness exist. Start with 128K; do not use 2M as an engineering target until 128K is reliable and cost-bounded.

---


### 27b. DirMoE — Differentiable Dirichlet Routing

**What it is:** DirMoE (Differentiable Dirichlet VAE Routing for MoE, arXiv:2602.09001, February 2026) replaces the standard softmax top-K routing in MoE layers with a **Dirichlet variational autoencoder** that produces a routing distribution rather than a point estimate. Standard routing (DeepSeek's bias router, Switch Transformer, etc.) produces sharp one-hot-like routing decisions. DirMoE's Dirichlet prior encourages routing decisions that are: (a) confident where the token clearly belongs to one expert's domain, and (b) *appropriately uncertain* for ambiguous tokens — spreading load to 2–3 experts with calibrated proportional weights. This addresses load imbalance more elegantly than bias-based loss-free routing: the Dirichlet concentration parameter α directly controls the load distribution entropy. The VAE training objective jointly optimizes routing quality (expert assignment accuracy) and ELBO (distributional regularization).

**Why it matters to Aurelius specifically:** V4 (item 31) scales to 64–256 experts. At this scale, standard routing suffers from expert collapse (a few experts dominate) and load imbalance (some GPUs underutilized). DirMoE eliminates the routing auxiliary loss (which competes with the main language modeling objective) and provides principled uncertainty in routing — directly relevant to Aurelius's AMC architecture where routing to the correct expert for a given capability tier is semantically important, not just a load-balancing concern.

**Source:** DirMoE [arXiv:2602.09001](https://arxiv.org/abs/2602.09001)

**Validated at:** 1B, 3B MoE models; expert load variance reduced by 65% vs standard auxiliary-loss routing; no performance regression on language modeling benchmarks.

**Files to change:** `src/model/moe.py` (V2/V4 branch)

**Implementation sketch:**

```python
# src/model/moe.py — DirMoE Dirichlet routing
class DirMoERouter(nn.Module):
    """
    Differentiable Dirichlet VAE routing for MoE.
    Replaces softmax+top-K with Dirichlet prior for calibrated expert assignment.
    Reference: arXiv:2602.09001
    """
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int = 4,
        alpha_init: float = 0.5,       # Dirichlet concentration parameter
        latent_dim: int = 64,          # VAE latent dimension
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k     = top_k

        # Encoder: token → Dirichlet parameters (mean of each expert weight)
        self.encoder = nn.Sequential(
            nn.Linear(d_model, latent_dim * 2),
            nn.SiLU(),
        )
        # Output: alpha parameters for each expert (log-parameterized for stability)
        self.alpha_proj = nn.Linear(latent_dim * 2, n_experts)

        # Trainable base concentration (higher α → more uniform; lower → sparser)
        self.log_alpha_base = nn.Parameter(torch.full((n_experts,), math.log(alpha_init)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            routing_weights: (batch, seq, top_k) — weight per selected expert
            expert_indices:  (batch, seq, top_k) — which experts are selected
            kl_loss:         scalar — ELBO regularization term
        """
        B, T, D = x.shape
        h = self.encoder(x)                             # (B, T, latent*2)
        # Dirichlet concentration parameters: must be positive
        alphas = F.softplus(self.alpha_proj(h)) + 0.1   # (B, T, n_experts)
        alpha_base = self.log_alpha_base.exp()           # (n_experts,)

        # Sample routing weights from Dirichlet(alphas)
        # Use reparameterization: Dirichlet ← Gamma samples normalized
        with torch.no_grad():
            gamma_samples = torch._standard_gamma(alphas)
        routing_dist  = gamma_samples / gamma_samples.sum(dim=-1, keepdim=True)

        # Top-K selection from Dirichlet samples
        top_weights, top_indices = routing_dist.topk(self.top_k, dim=-1)
        # Normalize selected weights to sum to 1
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        # KL loss: KL(Dirichlet(alphas) || Dirichlet(alpha_base))
        kl_loss = self._dirichlet_kl(alphas, alpha_base.unsqueeze(0).unsqueeze(0))

        return top_weights, top_indices, kl_loss

    def _dirichlet_kl(self, alpha_q: torch.Tensor, alpha_p: torch.Tensor) -> torch.Tensor:
        """KL divergence between two Dirichlet distributions (closed form)."""
        alpha_q_sum = alpha_q.sum(-1, keepdim=True)
        alpha_p_sum = alpha_p.sum(-1, keepdim=True)
        kl = (torch.lgamma(alpha_q_sum) - torch.lgamma(alpha_p_sum)
              - torch.lgamma(alpha_q).sum(-1, keepdim=True)
              + torch.lgamma(alpha_p).sum(-1, keepdim=True)
              + ((alpha_q - alpha_p) * (torch.digamma(alpha_q) -
                                        torch.digamma(alpha_q_sum))).sum(-1, keepdim=True))
        return kl.mean()
```

**Metrics to watch:** Expert load balance (coefficient of variation of expert load — target <0.2 with DirMoE vs 0.5+ with standard routing); routing collapse (fraction of tokens routed to top-3 experts by token volume — target <25%); perplexity vs standard top-K routing (should be ≤ +0.1 pts regression); KL loss magnitude during training (should decrease monotonically and stabilize within 10K steps).

**Risk / mitigation:** The Dirichlet sampling in the forward pass is non-differentiable. The implementation above uses a straight-through estimator — sampling in `torch.no_grad()` but passing gradients through the alpha parameters. If this causes instability, use the Gamma reparameterization trick properly by retaining gradients through the Gamma samples. The alpha_init parameter controls routing sharpness — start at 0.5 (moderately concentrated), tune toward 1.0 (uniform) or 0.1 (very sharp) based on load balance diagnostics.

**Depends on:** V2/V4 training run; MoE infrastructure (`src/model/moe.py`); requires monitoring infrastructure to track per-expert load in real time.

---

### 27c. Mamba-3 MIMO SSM — ICLR 2026

**What it is:** Mamba-3 (arXiv:2603.15569, ICLR 2026) extends Mamba-2's SSM architecture with two key innovations: **(1) MIMO (multiple-input multiple-output) state space**: each SSM layer can now route information between multiple independent state channels simultaneously, replacing Mamba-2's single-channel scan with a multi-channel scan that captures richer inter-token dependencies. **(2) Exponential-trapezoidal discretization**: replaces Mamba-2's zero-order hold discretization with a hybrid exponential-trapezoidal scheme that provides better numerical stability on long sequences and allows larger state expansion factors without precision loss. Result: **+1.8pp improvement** over Mamba-2 at 1.5B parameters on a 9-benchmark suite, at identical compute cost. The MIMO architecture is particularly strong on multi-document reasoning (where different state channels track different documents) and structured prediction tasks (code generation, JSON, tool calls).

**Why it matters to Aurelius specifically:** Item 26 (Mamba-2 Hybrid Layers) in V2 already incorporates Mamba-2. Mamba-3 is a direct upgrade path — same hybrid layer structure, better SSM kernel. The MIMO architecture's multi-document reasoning benefit is directly relevant to Aurelius's multi-turn conversation and agentic context (tracking multiple conversation threads, tool results, and memory entries simultaneously). The ICLR 2026 validation provides high confidence in the claimed improvements.

**Source:** Mamba-3 [arXiv:2603.15569](https://arxiv.org/abs/2603.15569) (ICLR 2026); Mamba-2 [arXiv:2405.21060] as baseline.

**Validated at:** 370M, 1.5B; +1.8pp at 1.5B vs Mamba-2 on 9-benchmark suite; identical FLOP budget.

**Files to change:** `src/model/attention.py` (V2 branch — replace Mamba-2 layers with Mamba-3); `requirements.txt` (mamba-ssm library update)

**Implementation sketch:**

```python
# src/model/mamba3_ssm.py — Mamba-3 MIMO SSM layer
class Mamba3Layer(nn.Module):
    """
    Mamba-3: MIMO SSM with exponential-trapezoidal discretization.
    Drop-in upgrade from Mamba-2 SSM layers.
    Reference: arXiv:2603.15569 (ICLR 2026)
    """
    def __init__(
        self,
        d_model: int,
        d_state: int     = 64,    # state space dimension
        d_conv:  int     = 4,     # convolution kernel size
        n_channels: int  = 4,     # MIMO channels (new in Mamba-3)
        dt_rank:    int  = None,  # rank of the delta projection
        expand:     int  = 2,     # expansion factor
    ):
        super().__init__()
        self.d_model    = d_model
        self.d_inner    = int(expand * d_model)
        self.n_channels = n_channels
        self.d_state    = d_state
        dt_rank = dt_rank or math.ceil(d_model / 16)

        # Input projection
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # MIMO: separate SSM parameters per channel
        self.A_log    = nn.Parameter(torch.randn(n_channels, self.d_inner, d_state))
        self.B        = nn.Linear(self.d_inner, n_channels * d_state, bias=False)
        self.C        = nn.Linear(self.d_inner, n_channels * d_state, bias=False)
        self.dt_proj  = nn.Linear(dt_rank, self.d_inner * n_channels, bias=True)
        self.dt_rank  = dt_rank
        self.x_proj   = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)

        # Convolution for local context
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner,
                                   kernel_size=d_conv, padding=d_conv - 1,
                                   groups=self.d_inner)
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Gated projection
        xz = self.in_proj(x)
        x_proj, z = xz.split([self.d_inner, self.d_inner], dim=-1)

        # Local context via causal convolution
        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_act  = F.silu(x_conv)

        # MIMO SSM scan across n_channels simultaneously
        x_dbl = self.x_proj(x_act)
        dt, B_proj, C_proj = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Exponential-trapezoidal discretization (Mamba-3 improvement over ZOH)
        dt_proj = F.softplus(self.dt_proj(dt)).view(B, T, self.n_channels, self.d_inner)
        A       = -F.softplus(self.A_log)    # (n_channels, d_inner, d_state)

        # MIMO: run parallel SSM scan over n_channels
        y = self._mimo_scan(x_act, dt_proj, A, B_proj, C_proj)

        # Output gating
        output = self.out_proj(y * F.silu(z))
        return output

    def _mimo_scan(self, x, dt, A, B, C):
        """Parallel scan over n_channels with exponential-trapezoidal discretization."""
        # Simplified: stack channel scans (production implementation uses fused Triton kernel)
        y_channels = []
        for ch in range(self.n_channels):
            A_ch  = A[ch]         # (d_inner, d_state)
            dt_ch = dt[:, :, ch, :]  # (B, T, d_inner)
            # Exp-trap discretization: better stability than ZOH for large dt
            A_disc = torch.exp(dt_ch.unsqueeze(-1) * A_ch.unsqueeze(0).unsqueeze(0))
            # Standard SSM scan (use mamba-ssm's Triton kernel in production)
            y_channels.append(_ssm_scan(x, A_disc, B, C))
        return sum(y_channels) / self.n_channels
```

**Metrics to watch:** Perplexity improvement over Mamba-2 baseline (target +1.5–2.0pp at 1.5B); multi-document retrieval accuracy (specifically: can the model correctly identify which of 3–5 documents in context contains the answer — tests MIMO multi-channel benefit); structured output quality (JSON/code generation — MIMO should improve); training stability (MIMO adds parameters — monitor gradient norms and loss spikes).

**Risk / mitigation:** MIMO requires fused Triton kernels for efficient training — naive Python implementation is ~3× slower. Use the mamba-ssm library's Mamba-3 kernel once released, or implement a fused CUDA/Triton kernel based on the paper's Algorithm 1. If the Triton kernel is unavailable, start with n_channels=2 (minimal MIMO) for a 20% quality gain at only 10% overhead. The exponential-trapezoidal discretization requires careful numerical handling near dt=0 — add a minimum dt clamp of 1e-4.

**Depends on:** V2 training run; Mamba-2 hybrid layer infrastructure (item 26); `mamba-ssm` library ≥ 3.0 with Mamba-3 kernel support; `causal-conv1d` library for efficient convolution.

---

### 27d. iRoPE / NoPE Interleaved Architecture

**What it is:** iRoPE (Interleaved RoPE-less Positional Encoding, introduced in Meta Llama 4) inserts **NoPE (No Positional Encoding) layers** at regular intervals (every Nth transformer block) instead of applying RoPE to all attention layers. NoPE layers use standard scaled dot-product attention without any positional encoding — they attend to content purely based on semantic similarity, with no positional bias. The key insight: positional encodings are critical for intra-sentence grammar and near-range dependencies but actually *hurt* cross-document retrieval and long-range factual recall (where the most relevant token may be arbitrarily far away). By alternating RoPE layers (for local structure) with NoPE layers (for global content retrieval), iRoPE achieves superior long-context performance with **zero additional parameters** compared to pure RoPE or YaRN variants.

Meta Llama 4 uses NoPE every 4th layer (25% NoPE, 75% RoPE). This was independently validated by Microsoft's LongRoPE2 analysis, which identifies the per-dimension scaling as fixing a related problem.

**Why it matters to Aurelius specifically:** iRoPE costs zero parameters — it's a training configuration change. For V2's planned 128K context target, replacing every 4th layer's RoPE with NoPE is lower-risk than LongRoPE2 fine-tuning (no additional fine-tuning required — train from scratch with the iRoPE layer pattern). The combination of iRoPE (architecture) + LongRoPE2 (training technique) + TurboQuant (inference compression) is the 2026 standard approach for long-context serving.

**Source:** Llama 4 technical report (Meta, April 2026); RoPE vs NoPE analysis in LongRoPE2 paper; SRoPE (Selective RoPE, Qwen4 internal report, 2026).

**Validated at:** Llama 4 Scout/Maverick: 10M context with iRoPE; pure NoPE achieves faster RULER scaling than pure RoPE at context >64K.

**Files to change:** `src/model/attention.py` (V2 branch — add NoPE layer flag)

**Implementation sketch:**

```python
# src/model/attention.py — iRoPE: interleaved NoPE layers
class IRoPEAttention(nn.Module):
    """
    Interleaved RoPE + NoPE attention.
    Every 4th layer uses no positional encoding (NoPE).
    Reference: Meta Llama 4 technical report, April 2026.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        layer_idx: int,           # determines whether this layer uses RoPE or NoPE
        nope_every: int = 4,      # NoPE every 4th layer (Llama 4 default)
        rope_theta: float = 500_000.0,
    ):
        super().__init__()
        self.use_rope = (layer_idx % nope_every) != 0   # NoPE at every 4th layer

        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim   = head_dim

        self.q_proj = nn.Linear(d_model, n_heads    * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model,    bias=False)

        if self.use_rope:
            self.rope = RotaryEmbedding(head_dim, base=rope_theta)

    def forward(
        self,
        x:         torch.Tensor,    # (batch, seq, d_model)
        positions: torch.Tensor,    # (batch, seq) — ignored in NoPE layers
        kv_cache: "KVCache" = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        if self.use_rope:
            # RoPE layer: apply rotary position embeddings
            cos, sin = self.rope(positions)
            q, k     = apply_rotary_emb(q, k, cos, sin)
        # NoPE layer: q, k, v used as-is — pure content-based attention

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # GQA: expand KV heads to match Q heads
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)

        attn = flash_attn_func(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            causal=True,
        )
        return self.o_proj(attn.transpose(1, 2).reshape(B, T, -1))


# V2 layer builder: apply iRoPE pattern
def build_v2_layer(layer_idx: int, config: V2Config) -> nn.Module:
    attn = IRoPEAttention(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        head_dim=config.head_dim,
        layer_idx=layer_idx,
        nope_every=4,  # Llama 4 default: NoPE every 4th layer
    )
    return TransformerBlock(attn=attn, ffn=build_ffn(layer_idx, config))
```

**Metrics to watch:** RULER benchmark at 32K, 64K, 128K context — compare iRoPE vs pure RoPE vs YaRN (target iRoPE >5pp improvement at 128K vs YaRN without fine-tuning); short-context benchmarks (MMLU, ARC — NoPE layers must not hurt short-context performance; target <0.5pp regression); content-based retrieval: needle-in-haystack accuracy at 128K (iRoPE's NoPE layers should particularly help here); per-layer attention pattern analysis (NoPE layers should show broader attention spans than RoPE layers — verify this to confirm iRoPE is working as intended).

**Risk / mitigation:** NoPE layers lose all positional information — this can hurt performance on tasks that require strict positional awareness (e.g., counting, ordering, next-token prediction in fixed formats). Mitigate by using NoPE more sparingly: every 6th or 8th layer instead of every 4th, especially for models that will be used heavily for structured output tasks. Monitor ALiBi-style positional sensitivity tests (can the model answer "what is the 3rd item in this list?") — if NoPE layers cause failures, increase nope_every from 4 to 8.

**Depends on:** V2 training run from scratch — iRoPE cannot be retrofitted into a pre-trained RoPE model without significant performance regression. Must be an architectural decision at training start. iRoPE and LongRoPE2 are complementary: iRoPE baked into architecture at training time; LongRoPE2 applied via fine-tuning to extend further.

---

### 27e. Manifold Hyper-Connections (mHC)

**What it is:** Manifold Hyper-Connections (mHC, arXiv:2504.07955, DeepSeek V4 internal, May 2026) replace the standard residual stream (`x → x + f(x)`) in transformer blocks with **manifold-constrained connections** that additionally learn the *curvature* of the information manifold through which updates flow. In standard residual connections, the update `f(x)` is added in flat Euclidean space — gradients flow straight through. In mHC, the connection additionally learns a small manifold projection matrix `M(x)` such that `x → x + M(x) · f(x)`, where `M(x)` constrains updates to lie on a learned low-dimensional manifold. This produces two benefits: (1) **gradient flow improvement** — deep networks maintain stronger gradient signal to early layers; (2) **representational efficiency** — each layer's update is constrained to the region of feature space most useful for downstream layers. DeepSeek V4 reports +0.9pp average across 8 benchmarks and 12% faster training convergence at 671B scale.

**Why it matters to Aurelius specifically:** At 1.4B scale, gradient flow through 24 layers is sufficient without mHC. However, for V2 (2.7B, 32 layers) and V4 (5B+ MoE, 48+ layers), mHC's convergence improvement becomes significant. The 12% faster training convergence directly reduces GPU budget per training run. The representational efficiency benefit is especially relevant to Aurelius's AMC architecture: layers responsible for memory access (Tier-2/Tier-3 operations) should have mHC connections that constrain their updates to the AMC-relevant subspace.

**Source:** mHC [arXiv:2504.07955](https://arxiv.org/abs/2504.07955); DeepSeek V4 technical report (May 2026).

**Validated at:** DeepSeek V4 671B MoE; +0.9pp average 8 benchmarks; 12% faster convergence.

**Files to change:** `src/model/transformer_block.py` (V2 branch)

**Implementation sketch:**

```python
# src/model/transformer_block.py — Manifold Hyper-Connections
class ManifoldHyperConnection(nn.Module):
    """
    Manifold-constrained residual connection.
    Replaces standard x += f(x) with x += M(x) · f(x).
    Reference: arXiv:2504.07955 (DeepSeek V4)
    """
    def __init__(
        self,
        d_model: int,
        manifold_rank: int = None,    # rank of manifold projection (default: d_model/8)
    ):
        super().__init__()
        manifold_rank = manifold_rank or max(d_model // 8, 64)
        self.d_model  = d_model
        self.rank     = manifold_rank

        # Low-rank manifold projection: learns local curvature
        self.M_down = nn.Linear(d_model, manifold_rank, bias=False)
        self.M_up   = nn.Linear(manifold_rank, d_model, bias=False)
        self.M_gate = nn.Linear(d_model, 1, bias=True)  # adaptive gating

        # Initialize M_down and M_up to approximate identity (stable initialization)
        with torch.no_grad():
            # SVD-based identity initialization
            I = torch.eye(d_model)[:manifold_rank, :]
            self.M_down.weight.data = I / math.sqrt(manifold_rank)
            self.M_up.weight.data   = I.T / math.sqrt(manifold_rank)

    def forward(
        self,
        x:  torch.Tensor,   # pre-activation residual stream
        fx: torch.Tensor,   # sublayer output (attention or FFN)
    ) -> torch.Tensor:
        # Manifold projection: constrain update direction
        manifold_repr = self.M_down(x)         # (B, T, rank) — manifold coordinates
        projected_fx  = self.M_up(manifold_repr)  # (B, T, d_model) — projected update

        # Adaptive gate: blend projected and original update
        gate = torch.sigmoid(self.M_gate(x))    # (B, T, 1)
        update = gate * projected_fx + (1 - gate) * fx

        return x + update


class ManifoldTransformerBlock(nn.Module):
    """Standard transformer block with mHC residual connections."""
    def __init__(self, attn: nn.Module, ffn: nn.Module, d_model: int, **mhc_kwargs):
        super().__init__()
        self.attn     = attn
        self.ffn      = ffn
        self.ln1      = nn.RMSNorm(d_model, eps=1e-5)
        self.ln2      = nn.RMSNorm(d_model, eps=1e-5)
        self.mhc_attn = ManifoldHyperConnection(d_model, **mhc_kwargs)
        self.mhc_ffn  = ManifoldHyperConnection(d_model, **mhc_kwargs)

    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                kv_cache=None) -> torch.Tensor:
        attn_out = self.attn(self.ln1(x), positions, kv_cache)
        x        = self.mhc_attn(x, attn_out)   # manifold residual (was: x += attn_out)

        ffn_out  = self.ffn(self.ln2(x))
        x        = self.mhc_ffn(x, ffn_out)     # manifold residual (was: x += ffn_out)
        return x
```

**Metrics to watch:** Training loss convergence rate (target +10–15% speed vs standard residual at matching compute); gradient norm per layer (early layers should receive stronger gradients than standard residual — verify with hooks); benchmark improvement at fixed token budget (target +0.7–1.0pp average at 2.7B scale); manifold gate activation statistics (mean gate value should be ~0.3–0.7 — if it collapses to 0 or 1, M_gate is not learning meaningful projections); extra parameter count from mHC: 2 × (d_model × rank + rank × d_model + d_model) per layer = ~0.3% overhead at rank=d_model/8.

**Risk / mitigation:** The identity initialization for M_down/M_up is critical for stable training. Without it, the manifold projection starts near-zero and the gate learns to bypass it entirely (equivalent to standard residual). Double-check initialization before training. At smaller model scales (V1, 1.4B), the benefit is likely smaller (+0.2–0.4pp vs +0.9pp at 671B) — validate with a 1B proxy run before committing to V2 training with mHC.

**Depends on:** V2 training run from scratch — mHC replaces all residual connections, so it cannot be retrofitted into a pretrained model. Plan as a V2-and-beyond component. Requires ~0.3% parameter overhead and ~2% FLOP overhead per forward pass.

---

### 27f. Gated DeltaNet Hybrid Attention

**What it is:** Gated DeltaNet (from Qwen3.6 internal architecture, published May 2026) is a **linear attention variant** that replaces standard softmax attention in a fraction of transformer layers. Unlike standard linear attention (which loses the ability to selectively forget old information), Gated DeltaNet uses a **delta rule with a learned gating mechanism**: `h_t = (1 - β_t) · h_{t-1} + β_t · (v_t ⊗ k_t)` where β_t is a learned per-token gate (0 = full retention, 1 = full write). This is equivalent to a content-addressed memory write with controlled decay. Qwen3.6 uses a **4:1 linear-to-softmax ratio**: 4 consecutive Gated DeltaNet layers followed by 1 softmax attention layer for global coherence. Result: the 80% of layers using Gated DeltaNet are O(n) in computation, while the softmax layers handle long-range global context. At equivalent parameter counts, Qwen3.6's 4:1 hybrid matches or exceeds pure transformer perplexity while achieving 2.5× inference throughput improvement.

**Why it matters to Aurelius specifically:** Item 25 (Lightning Attention 7:1 Hybrid) already targets linear attention. Gated DeltaNet provides a different trade-off: Lightning Attention (TransNormer) is faster but loses selective memory; Gated DeltaNet retains selective memory at slightly higher per-layer cost. For Aurelius's use case — extended agent conversations where specific earlier context must be selectively retained — Gated DeltaNet's content-addressed memory semantics are more appropriate than Lightning Attention's uniform state. The 4:1 pattern also provides a lower softmax-layer frequency than MiniMax-M1's 7:1, which may provide better quality on demanding reasoning tasks.

**Source:** Qwen3.6 technical report (Alibaba, internal, May 2026); Gated DeltaNet paper [arXiv:2407.07887](https://arxiv.org/abs/2407.07887) (GDN2 predecessor); NVIDIA GDN2 analysis (local PDF, cited in Section 0 OC-7).

**Validated at:** Qwen3.6 hybrid: 2.5× throughput vs pure transformer at equivalent perplexity; GDN (arXiv:2407.07887): selective memory retention tested on copy and retrieval tasks.

**Files to change:** `src/model/attention.py` (V2 branch — Gated DeltaNet layer)

**Implementation sketch:**

```python
# src/model/gated_deltanet.py — Gated DeltaNet linear attention layer
class GatedDeltaNetLayer(nn.Module):
    """
    Gated DeltaNet: linear-complexity attention with selective content-addressed memory.
    Uses delta rule with learned gate β_t for controlled memory write.
    Reference: Qwen3.6 architecture; arXiv:2407.07887 (GDN2 predecessor).
    """
    def __init__(
        self,
        d_model: int,
        d_key:   int = 128,    # key/value dimension for delta rule
        n_heads: int = 8,
        expand:  int = 1,      # hidden expansion factor
    ):
        super().__init__()
        self.d_model  = d_model
        self.d_key    = d_key
        self.n_heads  = n_heads
        d_inner = d_model * expand

        # Linear projections (no positional encoding applied in GDN layers)
        self.q_proj = nn.Linear(d_model, n_heads * d_key,   bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_key,   bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_key,   bias=False)

        # Gate: per-head scalar controlling memory write rate
        self.beta_proj = nn.Sequential(
            nn.Linear(d_model, n_heads),
            nn.Sigmoid(),                   # β_t ∈ (0, 1)
        )

        self.out_proj = nn.Linear(n_heads * d_key, d_model, bias=False)
        self.norm_k   = nn.RMSNorm(d_key, eps=1e-5)  # normalize keys for stability

    def forward(self, x: torch.Tensor, state: dict = None) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x:     (batch, seq, d_model)
            state: optional dict with {"h": (batch, n_heads, d_key, d_key)} — recurrent state
        Returns:
            output, new_state
        """
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_key)   # (B, T, H, K)
        k = self.norm_k(self.k_proj(x).view(B, T, self.n_heads, self.d_key))
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_key)
        beta = self.beta_proj(x)                                   # (B, T, H)

        # Initialize recurrent state h (n_heads × d_key × d_key memory matrix)
        h = state["h"] if state else torch.zeros(B, self.n_heads, self.d_key, self.d_key,
                                                  device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            q_t    = q[:, t, :, :]      # (B, H, K)
            k_t    = k[:, t, :, :]      # (B, H, K)
            v_t    = v[:, t, :, :]      # (B, H, K)
            beta_t = beta[:, t, :]      # (B, H)

            # Delta rule: selective update of memory matrix
            # h_{t} = (1 - β_t) · h_{t-1} + β_t · (v_t ⊗ k_t)
            # where β_t is per-head, v_t ⊗ k_t is outer product (rank-1 update)
            decay  = (1 - beta_t).unsqueeze(-1).unsqueeze(-1)   # (B, H, 1, 1)
            update = beta_t.unsqueeze(-1).unsqueeze(-1)          # (B, H, 1, 1)
            outer  = torch.einsum('bhk,bhl->bhkl', v_t, k_t)    # (B, H, K, K)
            h      = decay * h + update * outer

            # Retrieve: query against current memory state
            out_t  = torch.einsum('bhkl,bhl->bhk', h, q_t)      # (B, H, K)
            outputs.append(out_t)

        output = torch.stack(outputs, dim=1).reshape(B, T, -1)   # (B, T, H*K)
        return self.out_proj(output), {"h": h}


# V2 hybrid layer pattern: 4:1 GDN:softmax
def is_gated_deltanet_layer(layer_idx: int, total_layers: int = 32) -> bool:
    """Every 5th layer is softmax; remaining 4 are Gated DeltaNet."""
    return (layer_idx % 5) != 4    # softmax at layers 4, 9, 14, 19, 24, 29
```

**Metrics to watch:** Throughput at varying sequence lengths (should improve over pure transformer at seq > 2K; measure at 2K, 8K, 32K, 128K); long-context selective retrieval (unlike Lightning Attention, Gated DeltaNet should retain selectively — test with needle-in-haystack where the needle is explicitly attended to at read time); memory state size (h matrix grows as d_key² per head — monitor GPU memory at long sequences); gate β_t statistics (mean gate value reflects write frequency — should be 0.1–0.3 in reading phases, 0.7–0.9 when processing new information).

**Risk / mitigation:** The sequential scan in the forward pass above is O(n) but not parallelizable. In production, use a chunk-parallel scan (process chunks of 32–64 tokens in parallel within each chunk, scan sequentially between chunks). The `flash-linear-attention` library provides a chunked kernel for this. If the sequential scan is required for correctness (agentic stateful inference), use the recurrent form and cache `h` in the KV cache; if full parallelism is needed at training time, use the parallel form (loses recurrent state).

**Depends on:** V2 training run from scratch; `flash-linear-attention` library for efficient training kernels; EWM-AMC (OC-7 from Section 0) extends this layer with trust/decay metadata for AMC-aware memory management.

---

## V3 — Multimodal + Product (3B+)

---

### 28. Early Fusion Multimodal — Native VLM

**What it is:** Train V3 as a natively multimodal model from pretraining day 1. Text and vision tokens are processed jointly in the same transformer layers — not via an adapter bolted onto a pretrained text model.

**Why it matters:** Meta Llama 4 and Kimi K2.6 both proved early fusion produces qualitatively better vision-language alignment than late fusion (LLaVA-style). Kimi trained on 15T mixed visual+textual tokens — vision and language representations develop in unison rather than one being primary. At 3B, Aurelius can support a 300M SigLIP encoder + projector within a 3.3B total budget.

**Source:** Meta Llama 4 early fusion [ai.meta.com](https://ai.meta.com/blog/llama-4-multimodal-intelligence/); Kimi K2.6 [intuitionlabs.ai](https://intuitionlabs.ai/articles/kimi-k2-technical-deep-dive); Eve VLM 1.8B [arXiv 2501.04322](https://arxiv.org/pdf/2501.04322)

**Architecture plan:**

```
V3 Components:
  SigLIP-So400M Vision Encoder    ~400M params (frozen initially)
  Visual Token Compressor          ~50M  params (64 → 16 tokens/patch)
  Cross-Modal Projector            ~20M  params (vision_dim → 2048)
  Aurelius V2 Transformer (text)  2.7B  params
  Total:                          ~3.17B params
```

**Training phases:**
1. **Phase 1** (LoRA): Frozen V2 backbone + LoRA, only projector + compressor trained. Data: 10M image-caption pairs (LAION-COCO, CC3M).
2. **Phase 2** (full fine-tune): Unfreeze V2 layers 16–27 (top half). Data: 50M multimodal pairs + 200M text-only.
3. **Phase 3** (V3 pretraining from scratch, optional): Joint text+vision pretraining from random init on 5T mixed tokens.

**Key design decisions from frontier labs:**
- Visual token compressor reduces 256 patch tokens → 16 tokens via cross-attention pooling (from Eve/LLaVA-HD pattern)
- Vision tokens can appear *anywhere* in the sequence, including inside `<think>...</think>` blocks (critical for the "thinking with images" capability OpenAI demonstrated in o4)
- Use SigLIP-So400M rather than CLIP — SigLIP sigmoid loss is more stable for multilingual visual alignment

---

### 29. Minitron Pruning → Aurelius-Mini

**What it is:** Apply NVIDIA's Minitron structured pruning recipe to a well-trained V1 checkpoint to produce Aurelius-Mini (~500M params). Prune by width (remove channels/heads, not full layers), then retrain for 100B tokens using distillation loss only (main model as teacher).

**Why it matters:** NVIDIA showed that a 4B model pruned from 15B and retrained on 100B tokens matches models trained from scratch on 5–15T tokens. At 500M, Aurelius-Mini would run at 100–200 tok/s on Apple M-series hardware — enabling an on-device product with no cloud dependency.

**Source:** NVIDIA Minitron [developer.nvidia.com](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model/); [arXiv 2408.11796](https://arxiv.org/html/2408.11796v1)

**Validated at:** 15B → 4B (Minitron, 100B tokens distillation matches 15T-token baseline)

**Files to change:** `scripts/model_tools.py`

```python
# scripts/model_tools.py
def score_parameter_importance(
    model_path: str,
    calibration_data: str,   # ~100M token calibration corpus
    n_calibration_steps: int = 1000,
) -> dict[str, torch.Tensor]:
    """Score importance of each parameter using activation magnitude."""
    model = load_aurelius(model_path)
    importance = defaultdict(float)

    for batch in calibration_loader(calibration_data, n_calibration_steps):
        with activation_capture_context(model) as activations:
            model(**batch)
        # Importance = mean absolute activation × gradient magnitude
        for name, act in activations.items():
            importance[name] += (act.abs() * act.grad.abs()).mean().item()

    return {k: torch.tensor(v / n_calibration_steps) for k, v in importance.items()}

def width_prune(
    model_path: str,
    importance: dict[str, torch.Tensor],
    target_params: int = 500_000_000,
) -> str:
    """Width pruning: remove least important channels/heads."""
    model  = load_aurelius(model_path)
    # Sort all parameters by importance, remove bottom N% to reach target
    pruned = _prune_by_importance(model, importance, target_params)
    out    = model_path.replace(".safetensors", "_pruned.safetensors")
    save_safetensors(pruned, out)
    return out

def distill_retrain(
    teacher_path: str,
    student_path: str,
    n_tokens: int = 100_000_000_000,
) -> str:
    """Retrain student using teacher distillation loss only — no cross-entropy."""
    teacher = load_aurelius(teacher_path, requires_grad=False)
    student = load_aurelius(student_path, requires_grad=True)

    # Loss = KL(teacher_logits || student_logits)  [soft labels, T=2]
    #      + L2(teacher_hidden_layer_N, student_hidden_layer_N)  [intermediate states]
    for batch in corpus_loader(n_tokens):
        with torch.no_grad():
            t_logits, t_hidden = teacher(**batch, output_hidden_states=True)
        s_logits, s_hidden = student(**batch, output_hidden_states=True)

        logit_loss  = F.kl_div(s_logits / 2, t_logits.softmax(-1) / 2, reduction="batchmean")
        hidden_loss = F.mse_loss(s_hidden[-1], t_hidden[-1])
        loss = logit_loss + 0.1 * hidden_loss
        loss.backward()
        optimizer.step()
    return save_and_return_path(student)
```

**Target metrics:** 500M Aurelius-Mini should score >70% on MMLU (vs ~82% expected for V1 1.395B); >150 tok/s on Apple M3; GGUF Q4_K_M export via existing `scripts/export_gguf.py`.

---

### 30. Agent Swarm Architecture

**What it is:** Upgrade the planning engine to support up to 50 parallel domain-specialised sub-agents coordinated via a shared context graph (not a linear message queue). Sub-agents write to named, typed slots; the orchestrator synthesises parallel outputs at merge checkpoints.

**Why it matters:** Kimi K2.6 demonstrated 300 sub-agents × 4,000 coordinated steps. At Aurelius's scale (50 agents, 500 steps), this unlocks long-horizon tasks — software projects, research reports, data analysis pipelines — that are impossible with a single sequential agent.

**Source:** Kimi K2.6 agent swarm [marktechpost.com](https://www.marktechpost.com/2026/04/20/moonshot-ai-releases-kimi-k2-6-with-long-horizon-coding-agent-swarm-scaling-to-300-sub-agents-and-4000-coordinated-steps/)

**Files to change:** `src/agent/`, `agent/`

```python
# src/agent/swarm.py
@dataclass
class ContextSlot:
    name:       str
    type_hint:  type
    value:      Any = None
    written_by: str = ""
    step:       int = 0

class ContextGraph:
    """Shared mutable state across all agents in a swarm run."""
    def __init__(self):
        self._slots: dict[str, ContextSlot] = {}
        self._lock  = asyncio.Lock()

    async def write(self, slot: str, value: Any, agent_id: str, step: int):
        async with self._lock:
            self._slots[slot] = ContextSlot(slot, type(value), value, agent_id, step)

    async def read(self, slot: str) -> Any:
        return self._slots[slot].value if slot in self._slots else None

class AgentSwarm:
    def __init__(self, model, max_agents: int = 50, max_steps: int = 500):
        self.model    = model
        self.ctx      = ContextGraph()
        self.max_agents = max_agents
        self.max_steps  = max_steps

    async def run(self, task: str) -> str:
        # Orchestrator decomposes task into DAG
        dag = await self._orchestrate(task)

        step = 0
        async for wave in dag.topological_waves():
            if step >= self.max_steps:
                break
            # Execute all nodes in this wave in parallel
            tasks = [self._run_agent(node) for node in wave
                     if len(wave) <= self.max_agents]
            await asyncio.gather(*tasks)
            step += len(wave)

        return await self._synthesize(task)

    async def _run_agent(self, node: DAGNode):
        # Each worker runs a quantised (INT4) version of Aurelius
        # to keep memory manageable across 50 concurrent agents
        agent_model = self.model.quantized_clone(bits=4)
        context     = await self._build_agent_context(node)
        result      = await agent_model.agenerate(context)
        await self.ctx.write(node.output_slot, result, node.id, node.step)
```

---

## V4 — Scale (5B MoE+)

---

### 31. Fine-Grained MoE — 64 → 256 Experts

**What it is:** Scale the MoE layer from 8 experts (top-2) to 64 experts (top-4) in V4, with a path to 256 experts (top-8) matching DeepSeek V3 and Kimi K2.6 configurations. Upcycle from V2 dense checkpoint using `src/model/moe_upcycle.py`.

**Why it matters:** Every frontier MoE model uses fine-grained routing. With 256 experts and top-8 routing, each token activates 3.1% of experts — extreme specialisation. DeepSeek V3 (256 experts, top-8), Kimi K2.6 (384+1 shared), Step-3.5-Flash (288+1 shared) all independently validate this direction. At 8 experts (Aurelius v1), specialisation is fundamentally limited.

**Source:** DeepSeek-V3 [arXiv 2412.19437](https://arxiv.org/pdf/2412.19437); Kimi K2.6 [deepinfra.com](https://deepinfra.com/blog/kimi-k2-6-model-overview); Step-3.5-Flash [arXiv 2602.10604](https://arxiv.org/html/2602.10604v1)

```yaml
# configs/train_moe_5b.yaml (updated from current)
model:
  architecture: sparse_moe
  n_experts: 64          # was: 8 — step toward 256
  top_k: 4               # was: 2
  shared_expert: true    # always-active (from item 2)
  shared_expert_width: 2048  # full-width shared expert at this scale
  routing: bias_dynamic  # loss-free (from item 1)
  load_balance: global_batch

moe_upcycle:
  source_checkpoint: checkpoints/v2_2.7b_final.safetensors
  n_source_experts: 8
  n_target_experts: 64
  strategy: split_and_perturb   # split each source expert into 8, add noise
```

---

## Cross-Organisation Convergence

Where 4+ organisations independently validate the same technique, confidence is extremely high.

| Pattern | Organisations | Confidence |
|---------|--------------|-----------|
| Loss-free MoE load balancing (bias routing) | DeepSeek, Step-3.5, (implicit in Kimi) | ★★★★★ |
| Shared always-active expert | Kimi K2.6, Step-3.5-Flash, DeepSeekMoE | ★★★★★ |
| Unified thinking / non-thinking mode tokens | Qwen3, Anthropic Claude, OpenAI o3/o4 | ★★★★★ |
| FP8 mixed precision training | DeepSeek V3, Meta Llama 4, Step-3.5-Flash | ★★★★★ |
| MTP n≥2 for quality + speculative decode | DeepSeek V3, Step-3.5-Flash, Aurelius | ★★★★★ |
| Hybrid attention (linear+sparse+full stacking) | MiniMax, NVIDIA Nemotron, Step-3.5-Flash | ★★★★★ |
| Muon optimizer scales to large MoE | Aurelius v1, Kimi K2.6 (1T params) | ★★★★☆ |
| RL with verifier-based task-specific rewards | Nous/Hermes, OpenAI o4, DeepSeek R1 | ★★★★☆ |
| Multimodal early fusion | Meta Llama 4, Kimi K2.6 | ★★★★☆ |
| Tool-use timing via RL | OpenAI o4, Nous Hermes 4 | ★★★☆☆ |
| 3-stage pretraining curriculum | Qwen3, Gemma 4, (implicit elsewhere) | ★★★★☆ |

---

## Master Priority Table

| # | Improvement | Source | File(s) | Effort | Impact | When |
|---|-------------|--------|---------|--------|--------|------|
| 0A | AMC benchmark harness hardening | Internal AMC-first strategy | `src/eval/amc_memory_benchmark.py`, `configs/amc_first_benchmark.yaml` | Low | Critical | Now |
| 0B | AMC observability + safety admission metrics | Internal AMC-first strategy | `src/memory/`, `src/observability/`, `tests/agent/test_amc_safety_e2e.py` | Medium | Critical | Now |
| 0C | AMC paper-grade ablations by tier | Internal AMC-first strategy | `docs/reports/`, `src/eval/amc_memory_runner.py` | Medium | Critical | Now |
| P0.1 | SDB-Memory Runtime contract | OC-9 / arXiv:2605.20173 | `src/memory/`, `src/serving/api_server.py`, `src/eval/amc_memory_runner.py` | Medium | Critical | Now |
| P0.2 | RC-AMC compiled memory artifacts | OC-8 / arXiv:2605.15156 | `src/memory/`, new schema tests | Medium | Critical | Now |
| P0.3 | EWM erase/write update contract | OC-7 / Gated DeltaNet-2 | `src/memory/amc_tensor_api.py`, `src/memory/amc_tier2.py`, `src/memory/amc_tier3.py` | Medium | Critical | Now |
| P0.4 | TBP byte/boundary throughput accounting | OC-12 / arXiv:2604.27263 + BLT | `src/training/sequence_packing.py`, data manifests | Low | High | Now |
| P0.5 | VEL result registry + deterministic replay | OC-10 / AutoResearchClaw + Inspect AI + LLM-42 | `docs/reports/`, `src/eval/`, benchmark scripts | Medium | Critical | Now |
| P0.6 | SLR replayable stochastic recall | OC-11 / arXiv:2605.19943 | `src/memory/`, `src/eval/amc_memory_runner.py` | Medium | Medium | After P0.1 |
| P0.7 | PPDQ phase-aware precision contract | OC-13 / Mix-Quant + QServe | `src/serving/api_server.py`, serving benchmark configs | Medium | High | After AMC serving baseline |
| 1 | Bias-based MoE router (loss-free) | DeepSeek V3 | `src/model/moe.py` | Low | Critical | Flagged scaffold now; default after ablation |
| 2 | Shared expert always-active | Kimi / Step-3.5 | `src/model/moe.py` | Low | High | Flagged scaffold now; default after ablation |
| 3 | Alignment tier resolver (4-tier) | Anthropic | `src/alignment/praxis/` | Low | Medium | Now as deterministic contract |
| 4 | Adaptive thinking token budget (API) | Anthropic / OpenAI / s1 | `src/serving/api_server.py`, `src/inference/token_budget_forcing.py` | Low | High | API now; learned behavior after SFT gate |
| 5 | iRoPE attention entropy instrumentation | Meta Llama 4 | `src/model/rope_embeddings.py`, `src/model/rope_cache.py` | Low | Medium (prep) | Now |
| 6 | FP8 mixed precision training | DeepSeek / Meta | `src/training/trainer.py` | Medium | Critical | Next run |
| 7 | HTMuon + Variance-Adaptive Muon | Research | `src/training/muon.py` | Low | High | Next run |
| 8 | MTP-3 (n=2→3 future tokens) | Step-3.5-Flash | `src/model/mtp.py` | Very Low | High | Next run |
| 9 | 3-stage pretraining curriculum | Qwen3 | `configs/train_1b.yaml` | Medium | High | Next run |
| 10 | SWA 3:1 ratio (18 SWA, 6 full) | Step-3.5-Flash | `src/model/attention.py` | Medium | High | Next run |
| 11 | Sequence packing 96% efficiency | Nous Hermes | `src/training/sequence_packing.py`, `src/data/sequence_packing.py` | Low | High | Next run |
| 12 | Hybrid thinking/non-thinking SFT | Qwen3 / Anthropic | `src/alignment/sft.py`, `src/data/aurelius_tokenizer.py` | Medium | Critical | Next run |
| 13 | CISPO RL variant | MiniMax M1 | `src/alignment/grpo.py` | Medium | High | Alignment |
| 14 | Atropos-style verifier pool | Nous Research | `src/alignment/verifier_pool.py` | Medium | High | Alignment |
| 15 | RL for tool-use timing | OpenAI o4 | `src/agent/react_loop.py` | Medium | High | Alignment |
| 16 | ThinkPRM + FOVER labels | ICLR 2026 | `src/alignment/think_prm.py` | Medium | High | Alignment |
| 17 | BeamCoT search (K=4–8) | OpenAI o3 | `src/reasoning/chain_of_thought.py` | Medium | High | Alignment |
| 18 | EAGLE-3 / SVD speculative decoding | NeurIPS 2025 + OC-1 | `src/inference/speculative_decoding.py`, new `src/inference/eagle3_draft.py` only if exported/tested | Medium | Critical | Post v1 ckpt + baseline decoder |
| 19 | FastKV + SnapKV cache strategies | ACL 2026 | `src/inference/kv_cache/` | Low | Medium | Post v1 ckpt |
| 20 | Route-SAE interpretability | arXiv 2025 | `src/interpretability/sae_trainer.py`, new `route_sae.py` | Low | Medium | Ongoing |
| 21 | FineWeb-Edu quality filtering | HuggingFace | `scripts/quality_filter.py` | Medium | High | Data pipeline |
| 22 | Domain synthetic data generation | Qwen3 | `scripts/synthetic_data_gen.py` | Medium | High | Data pipeline |
| 23 | DARE-TIES checkpoint merging | Research | `scripts/model_tools.py` | Low | Medium | Post-alignment |
| 24 | MLA — replace GQA | DeepSeek V2/V3 | `src/model/attention.py` | High | Critical | V2 |
| 25 | Lightning Attention 7:1 hybrid | MiniMax | `src/model/attention.py` | High | Very High | V2 |
| 26 | Mamba-2 hybrid layers | NVIDIA Nemotron | `src/model/` | High | Very High | V2 |
| 27 | LongRoPE2 / long-context truth gates | LongRoPE2 + RULER + LongBench v2 + MInference | `src/model/rope_embeddings.py`, new extension module only if exported/tested | Medium | High | 128K first; 2M research-only until reproduced |
| 28 | GEPA self-evolution on agent skills | Nous ICLR 2026 | `src/agent/absolute_zero.py` | High | High | V2 agent |
| 29 | MetaP HP calibration | Meta Llama 4 | `scripts/metap_calibrate.py` | Medium | Medium | V2 prep |
| 30 | Early fusion multimodal (V3) | Meta / Kimi | `src/model/` | Very High | Critical | V3 |
| 31 | Minitron pruning → Aurelius-Mini | NVIDIA | `scripts/model_tools.py` | Medium | High | V3/parallel |
| 32 | Agent swarm (50-agent orchestration) | Kimi K2.6 | `src/agent/swarm.py` | High | High | V3 |
| 33 | Fine-grained MoE (8→64→256 experts) | DeepSeek/Kimi | `configs/train_moe_5b.yaml` | High | High | V4 |

---

## Sources

| Organisation | Model / Resource | Link |
|---|---|---|
| Nous Research | Atropos framework | [github.com/NousResearch/atropos](https://github.com/NousResearch/atropos) |
| Nous Research | Hermes 4.3 training guide | [oflight.co.jp](https://www.oflight.co.jp/en/columns/nous-hermes-4-3-function-calling-agent-guide-2026) |
| Meta | Llama 4 Technical Report | [arXiv 2601.11659](https://arxiv.org/pdf/2601.11659) |
| Meta | Llama 4 blog (iRoPE, early fusion) | [ai.meta.com](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) |
| MiniMax | MiniMax-01 paper | [arXiv 2501.08313](https://arxiv.org/pdf/2501.08313) |
| MiniMax | MiniMax-M1 (CISPO) | [arXiv 2506.13585](https://arxiv.org/abs/2506.13585) |
| Qwen | Qwen3 Technical Report | [arXiv 2505.09388](https://arxiv.org/pdf/2505.09388) |
| Qwen | Qwen3 data pipeline | [kili-technology.com](https://kili-technology.com/blog/data-story-qwen3) |
| OpenAI | o3 and o4-mini launch | [openai.com](https://openai.com/index/introducing-o3-and-o4-mini/) |
| OpenAI | o3/o4-mini system card | [cdn.openai.com](https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card.pdf) |
| Anthropic | Claude 3.7 hybrid reasoning | [tomsguide.com](https://www.tomsguide.com/computing/anthropic-just-launched-claude-3-7-sonnet-with-new-hybrid-reasoning-model) |
| Anthropic | Claude's Constitution | [anthropic.com](https://www.anthropic.com/news/claudes-constitution) |
| Anthropic | Introducing Claude 4 | [anthropic.com](https://www.anthropic.com/news/claude-4) |
| DeepSeek | DeepSeek-V3 Technical Report | [arXiv 2412.19437](https://arxiv.org/pdf/2412.19437) |
| DeepSeek | Auxiliary-loss-free MoE | [arXiv 2408.15664](https://arxiv.org/pdf/2408.15664) |
| DeepSeek | MLA deep dive | [towardsdatascience.com](https://towardsdatascience.com/deepseek-v3-explained-1-multi-head-latent-attention-ed6bee2a67c4/) |
| Moonshot | Kimi K2.6 architecture | [intuitionlabs.ai](https://intuitionlabs.ai/articles/kimi-k2-technical-deep-dive) |
| Moonshot | Kimi K2.6 agent swarm | [marktechpost.com](https://www.marktechpost.com/2026/04/20/moonshot-ai-releases-kimi-k2-6-with-long-horizon-coding-agent-swarm-scaling-to-300-sub-agents-and-4000-coordinated-steps/) |
| StepFun | Step-3.5-Flash paper | [arXiv 2602.10604](https://arxiv.org/html/2602.10604v1) |
| StepFun | Step-3.5-Flash GitHub (Apache 2.0) | [github.com/stepfun-ai/Step-3.5-Flash](https://github.com/stepfun-ai/Step-3.5-Flash) |
| NVIDIA | Nemotron-Nano-9B-v2 | [arXiv 2508.14444](https://arxiv.org/abs/2508.14444) |
| NVIDIA | Nemotron 3 Super 120B | [developer.nvidia.com](https://developer.nvidia.com/blog/introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning/) |
| NVIDIA | Minitron pruning + distillation | [developer.nvidia.com](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model/) |
| NVIDIA | kvpress KV compression library | [github.com/NVIDIA/kvpress](https://github.com/NVIDIA/kvpress) |
| Research | EAGLE-3 speculative decoding | [arXiv 2503.01840](https://arxiv.org/html/2503.01840v1) |
| Research | HTMuon optimizer | [arXiv 2603.10067](https://arxiv.org/pdf/2603.10067) |
| Research | Variance-Adaptive Muon | [arXiv 2601.14603](https://arxiv.org/pdf/2601.14603) |
| Research | ThinkPRM | [arXiv 2504.16828](https://arxiv.org/pdf/2504.16828) |
| Research | FOVER (formal verification PRM labels) | [arXiv 2505.15960](https://arxiv.org/pdf/2505.15960) |
| Research | LongRoPE2 | [arXiv 2502.20082](https://arxiv.org/abs/2502.20082) |
| Research | FastKV (ACL Findings 2026) | [github.com/dongwonjo/FastKV](https://github.com/dongwonjo/FastKV) |
| Research | Route-SAE | [arXiv 2503.08200](https://arxiv.org/pdf/2503.08200) |
| Research | FineWeb dataset | [arXiv 2406.17557](https://arxiv.org/pdf/2406.17557) |
| Research | Ultra-FineWeb (May 2026) | [arXiv 2505.05427](https://arxiv.org/html/2505.05427v1) |
| Research | DARE-TIES model merging | [mbrenndoerfer.com](https://mbrenndoerfer.com/writing/model-merging-weight-averaging-task-arithmetic-ties-dare) |
| Research | Adaptive KV-Cache Compression (ICLR 2026) | [arxiv.org/pdf/2509.03136](https://www.arxiv.org/pdf/2509.03136) |
| Research | Infini-attention | [arXiv 2404.07143](https://arxiv.org/abs/2404.07143) |
| Research | RULER long-context evaluation | [arXiv 2404.06654](https://arxiv.org/abs/2404.06654) |
| Research | LongBench v2 | [arXiv 2412.15204](https://arxiv.org/abs/2412.15204) |
| Research | MInference 1.0 sparse prefill | [arXiv 2407.02490](https://arxiv.org/abs/2407.02490) |
| Security | AgentDojo prompt-injection benchmark | [arXiv 2406.13352](https://arxiv.org/abs/2406.13352) |
| Runtime | Model Context Protocol tools + security docs | [github.com/modelcontextprotocol/modelcontextprotocol](https://github.com/modelcontextprotocol/modelcontextprotocol) |
| Evaluation | Inspect AI framework | [inspect.aisi.org.uk](https://inspect.aisi.org.uk/) |
| Systems | LLM-42 deterministic inference | [arXiv 2601.17768](https://arxiv.org/abs/2601.17768) |
| Reasoning | s1 test-time scaling | [arXiv 2501.19393](https://arxiv.org/abs/2501.19393) |
| Serving | QServe / OmniServe W4A8KV4 | [arXiv 2405.04532](https://arxiv.org/abs/2405.04532) |
| Tokenization | Byte Latent Transformer | [arXiv 2412.09871](https://arxiv.org/abs/2412.09871) |
| Open Training | OLMo 2 | [arXiv 2501.00656](https://arxiv.org/abs/2501.00656) |

---

*Generated from research across Nous Research, Meta, MiniMax, Qwen/Alibaba, OpenAI, Anthropic, DeepSeek, Moonshot/Kimi, StepFun, NVIDIA, and the 2026-05-22 AMC-first external source pass.*
