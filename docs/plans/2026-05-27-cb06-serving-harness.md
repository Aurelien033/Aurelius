# CB-06 Serving-Layer Harness

> **For Hermes:** Implement inline, one tranche at a time, strict TDD. Do not push.

**Goal:** Build a CPU-runnable serving harness that exercises the full DreamBank × CascadeBank stack end-to-end on deterministic synthetic prompts, and produces proxy measurements of quality and compute cost per-routing-policy.

**Architecture:** Pure-Python greedy-decode loop over `AMCTransformer`, parameterized by:
1. Whether DreamBank is enabled.
2. Which `CascadeRouter` config drives per-prompt routing decisions.
3. Per-policy simulated-cost multipliers (baseline proxy for "skip expensive post-processing when FAST" / "escalate to CoT when THOROUGH").

**Tech Stack:** Python 3.12, PyTorch (CPU), pytest, existing `AMCTransformer` + `HLMPreferenceBank` + `CascadeRouter`.

**Depends on:** CascadeBank commit `97ccc1dd`, DreamBank commit `3a759891`.

---

## Current Verified Truth Surface

HEAD after CascadeBank commit:
```
97ccc1dd feat(cascadebank): compute routing on DreamBank alignment gate (zero new params)
3a759891 fix: harden DreamBank MVP contracts
a02469d2 feat: add DreamBank MVP
```

Non-negotiable safety rules:
- Stage only CB-06 files.
- Do not commit or modify pre-existing dirty/untracked drift.
- Do not modify any DreamBank or CascadeBank source file.

---

## What We Are and Are Not Claiming

**Claiming:** A deterministic CPU-runnable harness that proves:
1. The DreamBank + CascadeBank composition is end-to-end runnable with a real greedy decoder.
2. Different routing configs produce measurably different policy distributions.
3. Proxy compute metric scales with policy class as designed (FAST < BALANCED < THOROUGH).

**Not claiming:**
- Real MT-Bench or AlpacaEval scores.
- Real GPU FLOPs or real p50/p95 wall-clock latency.
- Chain-of-thought or self-consistency inference (the "THOROUGH" path is simulated via a cost multiplier, not via actual beam/ensemble decoding).

Real MT-Bench/AlpacaEval/GPU work is deferred to CB-07. This tranche is *the scaffold for CB-07 to slot into*.

---

## Proposed File Map

Create:
- `src/eval/serving_harness.py` — greedy decode + per-policy cost-proxy harness.
- `tests/eval/test_serving_harness.py` — unit tests for proxy metrics, deterministic routing, policy-class cost multipliers.

Modify:
- `docs/research-brief.md` — append section 7 with claim/evidence.

Defer to CB-07:
- Real MT-Bench / AlpacaEval integration.
- GPU FLOPs measurement.
- Actual chain-of-thought / self-consistency implementation (THOROUGH is currently a cost proxy).

---

## Core API Contract

### `src/eval/serving_harness.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Sequence

import torch

from src.inference.cascade_routing import CascadeRouter, ComputePolicy
from src.memory.hlm_bank import HLMPreferenceBank
from src.model.amc_transformer import AMCTransformer


POLICY_COST_MULTIPLIER: dict[ComputePolicy, float] = {
    # FAST path: skip expensive post-processing. Baseline = 1.0.
    ComputePolicy.FAST: 0.7,
    # BALANCED: default, standard cost.
    ComputePolicy.BALANCED: 1.0,
    # THOROUGH: CoT/self-consistency — simulated 2.5x cost.
    ComputePolicy.THOROUGH: 2.5,
}


@dataclass(frozen=True)
class ServingHarnessConfig:
    max_new_tokens: int = 16
    temperature: float = 0.0  # 0 = greedy (argmax)
    bank_dim: int = 8
    bank_size: int = 8
    use_bank: bool = True


@dataclass
class PromptResult:
    prompt_id: int
    generated_ids: list[int]
    routing_decision: str          # "fast" | "balanced" | "thorough"
    logit_margin: float            # proxy quality: p_best - p_second on first token
    compute_proxy: float           # tokens * policy cost multiplier
    bank_fill_at_route_time: int


@dataclass
class HarnessReport:
    total_prompts: int
    policy_distribution: dict[str, int]
    mean_logit_margin_per_policy: dict[str, float]
    total_compute_proxy: float
    results: list[PromptResult]

    def to_dict(self) -> dict:
        return {
            "total_prompts": self.total_prompts,
            "policy_distribution": self.policy_distribution,
            "mean_logit_margin_per_policy": self.mean_logit_margin_per_policy,
            "total_compute_proxy": self.total_compute_proxy,
            "num_results": len(self.results),
        }


def run_harness(
    model: AMCTransformer,
    prompts: Sequence[torch.Tensor],
    *,
    bank: HLMPreferenceBank | None = None,
    router: CascadeRouter | None = None,
    config: ServingHarnessConfig | None = None,
) -> HarnessReport: ...


def greedy_decode(
    model: AMCTransformer,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    bank: HLMPreferenceBank | None = None,
) -> tuple[list[int], float]:
    """Token-by-token greedy decode. Returns (generated_ids, first_token_logit_margin)."""
    ...
```

Mechanism:
```text
for prompt in prompts:
    # 1. Greedy decode with bank if provided
    generated, logit_margin = greedy_decode(model, prompt, max_new_tokens, bank)

    # 2. Run router once on the FINAL forward pass to get per-prompt decision
    #    (we can run the full forward pass inside greedy_decode and capture alpha/confidence)

    # 3. Compute policy-specific cost multiplier
    compute_proxy = len(generated) * POLICY_COST_MULTIPLIER[decision]

    # 4. Append to results
```

Invariants:
- `POLICY_COST_MULTIPLIER[FAST] < POLICY_COST_MULTIPLIER[BALANCED] < POLICY_COST_MULTIPLIER[THOROUGH]`.
- Greedy decode is deterministic given the same seed + weights + input.
- `HarnessReport.to_dict()` is JSON-serializable.
- When `bank is None`, router must return `fallback` (BALANCED).

---

## Tranches

### Task CB-06: Pre-flight and baseline

Verify DreamBank + CascadeBank MVP suites still pass.

```bash
cd /Users/christienantonio/aurelius && .venv/bin/python -m pytest tests/memory/test_hlm_bank.py tests/model/test_hlm_bank_adapter.py tests/model/test_amc_transformer.py tests/alignment/test_dreambank.py tests/scripts/test_run_dreambank_cycle.py tests/eval/test_dreambank_ablation.py tests/inference/test_cascade_routing.py tests/inference/test_cascade_routing_integration.py tests/eval/test_cascadebank_ablation.py -q -W ignore::DeprecationWarning 2>&1 | tail -3
```

**Acceptance:** 89 tests still green at HEAD `97ccc1dd`.

### Task CB-06-1: Greedy decode primitive

**Files:**
- Create: `src/eval/serving_harness.py` (with `greedy_decode` + `POLICY_COST_MULTIPLIER` + dataclasses)
- Create: `tests/eval/test_serving_harness.py` (greedy_decode tests only)

**TDD tests:**
- `test_greedy_decode_deterministic_same_seed`
- `test_greedy_decode_output_length_equals_max_new_tokens`
- `test_greedy_decode_returns_logit_margin_in_unit_interval`
- `test_policy_cost_multiplier_ordering_fast_balanced_thorough`

**Acceptance:** 4 tests pass.

### Task CB-06-2: Full harness + reporting

**Files:**
- Extend: `src/eval/serving_harness.py` (add `run_harness`, `HarnessReport`)
- Extend: `tests/eval/test_serving_harness.py`

**TDD tests:**
- `test_run_harness_with_empty_bank_returns_all_fallback`
- `test_run_harness_with_populated_bank_varies_policy_distribution`
- `test_run_harness_per_prompt_compute_proxy_uses_policy_multiplier`
- `test_run_harness_report_to_dict_is_json_serializable`
- `test_run_harness_total_prompts_matches_input_count`
- `test_run_harness_mean_margin_per_policy_contains_only_observed_classes`

**Acceptance:** 6 harness tests pass; combined suite = 14 tests green.

### Task CB-06-3: Documentation

- Append section 7 to `docs/research-brief.md`.

---

## Done Criteria for CB-06

CB-06 MVP is complete when all are true:
- 14 new tests green (4 greedy_decode + 6 harness + 4 from earlier? — no, only new tests in this tranche).
- Combined suite (DreamBank + CascadeBank + ServingHarness) = 89 + 10 = 99 tests green.
- `run_harness` returns deterministic, JSON-serializable reports.
- No changes to existing DreamBank/CascadeBank files.
