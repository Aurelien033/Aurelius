"""
src/alignment/fractional_reward.py — Fractional / Dense Verifiable Rewards for RLVR.

Extends the VerifiableReward interface from rlvr.py with executions-aware
scoring that produces dense signal instead of sparse 0/0.5/1 terminal rewards.

VeRPO mapping (arXiv 2511.12344):
    reward = fraction_of_tests_passed
           + all_pass_bonus * [all tests passed]

This gives GRPO a signal on *every* rollout, fixing the zero-gradient
equal-reward-group problem from v2.

REF: research_compass_2026-06-20.md, RLVR improvement lit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class FractionalRewardConfig:
    """Configuration for fractional verifiable rewards.

    Attributes:
        all_pass_bonus: Extra reward added when every test passes.
            Typical value 0.5 (so max reward = 1.0 + 0.5 = 1.5).
        penalty_scale:  Multiplier applied to (1 - pass_rate) to penalise
            completions that only pass a small fraction of tests.
            Typical value 0.0 (disabled) or small positive.
        floor: Minimum reward returned (applied after all_pass_bonus).
        ceiling: Maximum reward returned (applied after all_pass_bonus).
    """

    all_pass_bonus: float = 0.5
    penalty_scale: float = 0.0
    floor: float = -1.0
    ceiling: float = 2.0


class FractionalTestReward:
    """Fraction-of-tests-passed verifiable reward.

    Expects ``test_results`` to be a list of booleans (or ints 0/1)
    indicating pass/fail for each test case.  The reward is then:

        reward = sum(test_results) / len(test_results)
        if all_pass: reward += all_pass_bonus

    This contrasts with the parent VerifiableReward which returns only
    0 / 0.5 / 1.

    Usage::

        reward_fn = FractionalTestReward()
        score = reward_fn(prompt, completion, test_results=[True, True, False])
        # 2/3 + 0.0 = 0.666...

    For code tasks, ``test_results`` is the output of
    ``src.agent.code_execution_tool.run_tests``.
    """

    def __init__(self, config: FractionalRewardConfig | None = None) -> None:
        self.config = config or FractionalRewardConfig()

    def __call__(
        self,
        prompt: str,
        completion: str,
        test_results: list[bool] | list[int] | None = None,
    ) -> float:
        """Return fractional reward from test results.

        Args:
            prompt:       Task prompt (unused, kept for VerifiableReward compat).
            completion:   Model completion text (used only for length checks).
            test_results: List of bool/int pass indicators per test case.
                          If None or empty → returns 0.0.

        Returns:
            float in [floor, ceiling].
        """
        if not test_results:
            return self.config.floor

        n = len(test_results)
        pass_count = sum(1 for r in test_results if r)
        pass_rate = pass_count / n

        reward: float = pass_rate

        # All-pass bonus — dense signal already exists; bonus is a shape tweak
        if pass_count == n and n > 0:
            reward += self.config.all_pass_bonus

        # Optional penalty for partial passes
        if self.config.penalty_scale > 0.0:
            reward -= self.config.penalty_scale * (1.0 - pass_rate)

        # Clamp to [floor, ceiling]
        reward = max(self.config.floor, min(self.config.ceiling, reward))
        return float(reward)


class CompositeFractionalReward:
    """Weighted combination of multiple fractional reward functions.

    Unlike the parent CompositeReward which normalises by total weight,
    this variant uses learned-ish weighting that preserves the raw
    magnitude differences between reward signals — useful when one
    domain (e.g. code) has naturally higher variance than another
    (e.g. math).

    Args:
        rewards: List of (callable, weight) pairs.  Each callable must
            accept ``(prompt, completion, **kwargs)`` and return float.
        default_kwargs: Default keyword arguments forwarded to each
            reward function.
    """

    def __init__(
        self,
        rewards: list[tuple[Callable, float]],
        default_kwargs: dict | None = None,
    ) -> None:
        self.rewards = rewards
        self.default_kwargs = default_kwargs or {}

    def __call__(
        self,
        prompt: str,
        completion: str,
        **extra_kwargs,
    ) -> float:
        kwargs = {**self.default_kwargs, **extra_kwargs}
        total_weight = sum(w for _, w in self.rewards)
        if total_weight == 0.0:
            return 0.0

        weighted_sum = sum(fn(prompt, completion, **kwargs) * w for fn, w in self.rewards)
        return float(weighted_sum / total_weight)


class ExecutionGroundedReward:
    """Reward grounded in actual code execution output.

    Wraps a ``test_runner`` callable that returns
    ``(passed: int, total: int, details: list[str])`` for each test.

    This is the bridge between ``src.agent.code_execution_tool`` and
    the RLVR reward interface.

    Usage::

        from src.agent.code_execution_tool import CodeExecutionTool

        tool = CodeExecutionTool()
        reward = ExecutionGroundedReward(test_runner=tool.run_tests)
        score = reward(prompt, completion, task_id="human_eval_0")
    """

    def __init__(
        self,
        test_runner: Callable[[str, str, str | None], tuple[int, int, list[str]]] | None = None,
        config: FractionalRewardConfig | None = None,
    ) -> None:
        self.test_runner = test_runner
        self.config = config or FractionalRewardConfig()

    def __call__(
        self,
        prompt: str,
        completion: str,
        task_id: str | None = None,
        ground_truth: str | None = None,
    ) -> float:
        """Run tests and return fractional reward.

        Args:
            prompt:       Task prompt (forwarded to test_runner).
            completion:   Model completion (executed by test_runner).
            task_id:      Identifier for the task (forwarded to test_runner).
            ground_truth: Ignored by execution-based reward.

        Returns:
            float in [floor, ceiling].
        """
        if self.test_runner is None:
            return 0.0

        try:
            passed, total, _details = self.test_runner(prompt, completion, task_id)
        except Exception:
            return self.config.floor

        if total == 0:
            return self.config.floor

        pass_rate = passed / total
        reward: float = pass_rate

        if passed == total:
            reward += self.config.all_pass_bonus

        if self.config.penalty_scale > 0.0:
            reward -= self.config.penalty_scale * (1.0 - pass_rate)

        reward = max(self.config.floor, min(self.config.ceiling, reward))
        return float(reward)
