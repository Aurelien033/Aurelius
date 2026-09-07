"""
tests/training/test_multi_domain_verifier.py

Tests:
  1.  test_config_defaults
  2.  test_verify_code_passes
  3.  test_verify_code_fails
  4.  test_verify_code_no_runner_returns_zero
  5.  test_verify_math_exact
  6.  test_verify_math_near
  7.  test_verify_math_wrong
  8.  test_verify_math_missing_answer
  9.  test_verify_json_valid_with_required_keys
  10. test_verify_json_invalid_strict
  11. test_verify_json_non_strict_partial
  12. test_verify_json_missing_required_keys
  13. test_verify_reasoning_short
  14. test_verify_reasoning_long_no_answer
  15. test_verify_reasoning_with_answer_match
  16. test_verify_generic_match
  17. test_verify_unknown_domain_returns_fallback
  18. test_register_custom_domain
  19. test_reward_fn_wrap_returns_callable
  20. test_reward_fn_wrap_forwards_kwargs
  21. test_registered_domains_contains_builtins
  22. test_floor_ceiling_clamping
"""

from __future__ import annotations

import json

import pytest

from src.training.multi_domain_verifier import (
    DomainVerifierConfig,
    MultiDomainVerifier,
    REGISTERED_DOMAINS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_runner(
    passed: int = 0,
    total: int = 3,
    error: str | None = None,
):
    def _runner(prompt: str, completion: str, task_id: str | None = None):
        return passed, total, [error] if error else []

    return _runner


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = DomainVerifierConfig()
    assert cfg.fail_fast is False
    assert cfg.json_strict is True
    assert cfg.math_tolerance_pct == pytest.approx(5.0)
    assert cfg.reasoning_min_words == 15
    assert cfg.default_fallback == 0.0


# ---------------------------------------------------------------------------
# 2-4. Code verifier
# ---------------------------------------------------------------------------


def test_verify_code_passes():
    mdv = MultiDomainVerifier()
    runner = _make_fake_runner(passed=3, total=3)
    score = mdv.verify("code", "prompt", "completion", test_runner=runner, task_id="t1")
    assert score == pytest.approx(1.0)


def test_verify_code_fails():
    mdv = MultiDomainVerifier()
    runner = _make_fake_runner(passed=1, total=3)
    score = mdv.verify("code", "prompt", "completion", test_runner=runner, task_id="t1")
    assert score == pytest.approx(1.0 / 3.0)


def test_verify_code_no_runner_returns_zero():
    mdv = MultiDomainVerifier()
    score = mdv.verify("code", "prompt", "completion")
    assert score == 0.0


# ---------------------------------------------------------------------------
# 5-8. Math verifier
# ---------------------------------------------------------------------------


def test_verify_math_exact():
    mdv = MultiDomainVerifier()
    score = mdv.verify("math", "What is 2+2?", "The answer is 4.", answer="4")
    assert score == pytest.approx(1.0)


def test_verify_math_near():
    mdv = MultiDomainVerifier(config=DomainVerifierConfig(math_tolerance_pct=5.0))
    score = mdv.verify("math", "pi?", "pi is 3.15", answer="3.14159")
    assert score == pytest.approx(0.5)


def test_verify_math_wrong():
    mdv = MultiDomainVerifier()
    score = mdv.verify(
        "math", "What is 2+2?", "It is 5.", answer="4", tolerance_pct=5.0
    )
    assert score == pytest.approx(0.0)


def test_verify_math_missing_answer():
    mdv = MultiDomainVerifier()
    score = mdv.verify("math", "Solve it.", "3.")
    assert score == 0.0


# ---------------------------------------------------------------------------
# 9-12. JSON verifier
# ---------------------------------------------------------------------------


def test_verify_json_valid_with_required_keys():
    mdv = MultiDomainVerifier()
    completion = '{"name": "Aurelius", "version": 5}'
    score = mdv.verify(
        "json",
        "Return JSON",
        completion,
        required_keys=["name", "version"],
    )
    assert score == pytest.approx(1.0)


def test_verify_json_invalid_strict():
    mdv = MultiDomainVerifier()  # strict=True by default
    score = mdv.verify(
        "json",
        "Return JSON",
        "not json at all",
        required_keys=["name"],
    )
    assert score == pytest.approx(0.0)


def test_verify_json_non_strict_partial():
    mdv = MultiDomainVerifier(config=DomainVerifierConfig(json_strict=False))
    # "name" does appear as a key in JSON-like text; verifier returns 0.3
    completion = 'my name is "Aurelius" and version is 5.0'
    score = mdv.verify(
        "json",
        "Respond",
        completion,
        required_keys=["name", "version"],
    )
    # At least both key substrings appear in the text → 0.3 each (at minimum 0.3)
    assert score >= 0.3


def test_verify_json_missing_required_keys():
    mdv = MultiDomainVerifier()
    completion = '{"answer": 42}'
    score = mdv.verify(
        "json",
        "Return JSON",
        completion,
        required_keys=["name", "version"],
    )
    assert score < 1.0  # Valid JSON but missing keys → < 1.0


# ---------------------------------------------------------------------------
# 13-15. Reasoning verifier
# ---------------------------------------------------------------------------


def test_verify_reasoning_short():
    mdv = MultiDomainVerifier(config=DomainVerifierConfig(reasoning_min_words=10))
    score = mdv.verify("reasoning", "Explain", "Yes.")
    assert score == 0.0


def test_verify_reasoning_long_no_answer():
    mdv = MultiDomainVerifier()
    long_text = " ".join(["word"] * 20)
    score = mdv.verify("reasoning", "Explain", long_text)
    assert score == pytest.approx(0.5)


def test_verify_reasoning_with_answer_match():
    mdv = MultiDomainVerifier()
    # Need at least reasoning_min_words (default 15) words
    long = (
        "After carefully reasoning through the problem step by step, "
        "checking each intermediate calculation, and verifying the logic, "
        "the correct answer is \\boxed{42}."
    )
    score = mdv.verify("reasoning", "What is 6×7?", long, answer="42")
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 16. Generic verifier
# ---------------------------------------------------------------------------


def test_verify_generic_match():
    mdv = MultiDomainVerifier()
    score = mdv.verify("generic", "p", "hello", answer="hello")
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 17. Unknown domain
# ---------------------------------------------------------------------------


def test_verify_unknown_domain_returns_fallback():
    mdv = MultiDomainVerifier(config=DomainVerifierConfig(default_fallback=-0.5))
    score = mdv.verify("nonexistent", "p", "c")
    assert score == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# 18-20. Runtime registration and wrapping
# ---------------------------------------------------------------------------


def test_register_custom_domain():
    mdv = MultiDomainVerifier()
    mdv.register(
        "custom",
        lambda p, c, **kw: 0.99,
        description="Always returns 0.99",
        requires=["ignored_kwarg"],
    )
    assert "custom" in mdv.registered_domains()
    score = mdv.verify("custom", "p", "c")
    assert score == pytest.approx(0.99)


def test_reward_fn_wrap_returns_callable():
    mdv = MultiDomainVerifier()
    runner = _make_fake_runner(passed=3, total=3)
    fn = mdv.reward_fn_wrap("code", test_runner=runner, task_id="t99")
    assert callable(fn)
    score = fn("prompt", "completion")
    assert score == pytest.approx(1.0)


def test_reward_fn_wrap_forwards_kwargs():
    mdv = MultiDomainVerifier()
    runner = _make_fake_runner(passed=2, total=4)
    fn = mdv.reward_fn_wrap("code", test_runner=runner, task_id="tX")
    # Override task_id at call time
    score = fn("p", "c", task_id="t_overridden")  # type: ignore[call-arg]
    assert score == pytest.approx(2.0 / 4.0)


# ---------------------------------------------------------------------------
# 21-22. Registry and clamping
# ---------------------------------------------------------------------------


def test_registered_domains_contains_builtins():
    mdv = MultiDomainVerifier()
    domains = set(mdv.registered_domains())
    assert {"code", "math", "json", "reasoning", "generic"} <= domains


def test_floor_ceiling_clamping():
    cfg = DomainVerifierConfig(floor=-2.0, ceiling=10.0, all_pass_bonus=20.0)
    mdv = MultiDomainVerifier(config=cfg)
    runner = _make_fake_runner(passed=5, total=5)
    score = mdv.verify(
        "code", "p", "c", test_runner=runner, task_id="t", all_pass_bonus=20.0  # type: ignore[call-arg]
    )
    # 5/5 + 20.0 = 21.0 → ceiling=10.0
    assert score == pytest.approx(10.0)
