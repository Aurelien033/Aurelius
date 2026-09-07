"""CB-06 Serving harness — greedy decode with per-policy cost-proxy measurement.

Exercises the full DreamBank × CascadeBank stack end-to-end on deterministic
synthetic prompts, and produces proxy measurements of quality (logit margin)
and compute cost (token count × per-policy cost multiplier).

This is an MVP scaffold. Real MT-Bench / AlpacaEval / GPU FLOPs come in CB-07.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from src.inference.cascade_routing import CascadeRouter, ComputePolicy
from src.memory.hlm_bank import HLMPreferenceBank
from src.model.amc_transformer import AMCTransformer

POLICY_COST_MULTIPLIER: dict[ComputePolicy, float] = {
    # FAST path: skip expensive post-processing. 0.7x baseline.
    ComputePolicy.FAST: 0.7,
    # BALANCED: default, 1.0x baseline.
    ComputePolicy.BALANCED: 1.0,
    # THOROUGH: CoT/self-consistency — simulated 2.5x cost.
    ComputePolicy.THOROUGH: 2.5,
}


@dataclass(frozen=True)
class ServingHarnessConfig:
    max_new_tokens: int = 16
    temperature: float = 0.0  # 0 = greedy (argmax)
    use_bank: bool = True


@dataclass(frozen=True)
class PromptResult:
    prompt_id: int
    generated_ids: list[int]
    routing_decision: str  # "fast" | "balanced" | "thorough"
    logit_margin: float  # proxy quality: p_best - p_second on first generated token
    compute_proxy: float  # tokens * policy cost multiplier
    bank_fill_at_route_time: int


@dataclass
class HarnessReport:
    total_prompts: int
    policy_distribution: dict[str, int]
    mean_logit_margin_per_policy: dict[str, float]
    total_compute_proxy: float
    results: list[PromptResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "total_prompts": self.total_prompts,
            "policy_distribution": dict(self.policy_distribution),
            "mean_logit_margin_per_policy": dict(self.mean_logit_margin_per_policy),
            "total_compute_proxy": float(self.total_compute_proxy),
            "num_results": len(self.results),
        }


def greedy_decode(
    model: AMCTransformer,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    bank: HLMPreferenceBank | None = None,
) -> tuple[list[int], float]:
    """Token-by-token greedy decode.

    Args:
        prompt_ids: (1, L) input prompt tensor.
        max_new_tokens: how many tokens to generate.
        bank: optional HLMPreferenceBank to pass to the model.

    Returns:
        (generated_ids, first_token_logit_margin) where margin is
        p_best - p_second (softmax probabilities) on the FIRST generated token.
    """
    if max_new_tokens <= 0:
        return [], 0.0
    if prompt_ids.dim() != 2 or prompt_ids.shape[0] != 1:
        raise ValueError(f"prompt_ids must be shape (1, L), got {tuple(prompt_ids.shape)}")

    generated: list[int] = []
    first_logit_margin: float = 0.0
    current_ids = prompt_ids

    with torch.no_grad():
        for step in range(max_new_tokens):
            out = model(current_ids, preference_bank=bank)
            logits = out.logits  # (1, T, V)
            next_token_logits = logits[:, -1, :]  # (1, V)

            if step == 0:
                # Capture first generated token's logit margin
                probs = torch.softmax(next_token_logits, dim=-1)
                sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
                first_logit_margin = float((sorted_probs[:, 0] - sorted_probs[:, 1]).item())

            next_token = int(next_token_logits.argmax(dim=-1).item())
            generated.append(next_token)
            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_token]], device=current_ids.device)],
                dim=1,
            )

    return generated, first_logit_margin


def run_harness(
    model: AMCTransformer,
    prompts: Sequence[torch.Tensor],
    *,
    bank: HLMPreferenceBank | None = None,
    router: CascadeRouter | None = None,
    config: ServingHarnessConfig | None = None,
) -> HarnessReport:
    """Run the harness on a list of prompts.

    Routing decision is made per-prompt on the prompt-only forward pass
    (before greedy decode), so the policy class conceptually informs the
    generation cost-budget.
    """
    cfg = config or ServingHarnessConfig()
    router = router or CascadeRouter()

    passed_bank = bank if cfg.use_bank else None

    results: list[PromptResult] = []

    for prompt_id, prompt in enumerate(prompts):
        # Step 1: Route on prompt forward — captures alpha/confidence for decision
        with torch.no_grad():
            prompt_out = model(prompt, preference_bank=passed_bank)

        decision = router.decision(prompt_out.bank_alpha, prompt_out.bank_confidence)

        # Step 2: Greedy decode under the routed policy
        generated, logit_margin = greedy_decode(
            model,
            prompt,
            max_new_tokens=cfg.max_new_tokens,
            bank=passed_bank,
        )

        # Step 3: Compute proxy = tokens * policy-specific multiplier
        compute_proxy = len(generated) * POLICY_COST_MULTIPLIER[decision]

        # Step 4: Bank fill at route time (informational)
        bank_fill = (
            int(bank.strengths.gt(0).sum().item())
            if bank is not None and hasattr(bank, "strengths")
            else 0
        )

        results.append(
            PromptResult(
                prompt_id=prompt_id,
                generated_ids=generated,
                routing_decision=decision.value,
                logit_margin=logit_margin,
                compute_proxy=compute_proxy,
                bank_fill_at_route_time=bank_fill,
            )
        )

    # Aggregate
    policy_dist: dict[str, int] = {"fast": 0, "balanced": 0, "thorough": 0}
    per_class_margins: dict[str, list[float]] = defaultdict(list)
    for r in results:
        policy_dist[r.routing_decision] += 1
        per_class_margins[r.routing_decision].append(r.logit_margin)

    mean_margin_per_policy = {
        cls: float(sum(ms) / len(ms)) if ms else 0.0 for cls, ms in per_class_margins.items()
    }

    total_compute = sum(r.compute_proxy for r in results)

    return HarnessReport(
        total_prompts=len(results),
        policy_distribution=policy_dist,
        mean_logit_margin_per_policy=mean_margin_per_policy,
        total_compute_proxy=float(total_compute),
        results=results,
    )
