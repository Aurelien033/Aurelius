"""
src/training/multi_domain_verifier.py — Unified Multi-Domain Verifier Registry.

Provides a single interface that dispatches verification to domain-specific
verifiers: code, math, logic/reasoning, and structured/JSON outputs.

Each domain verifier returns a normalised float score in [0, 1] (or
[-1, ceiling] for hybrid rewards).  The registry supports:

    * REGISTERED_DOMAINS  — enumerate known domains with metadata
    * verify(domain, ...) — dispatch to the right verifier
    * register(domain, fn) — add custom verifiers at runtime
    * reward_fn_wrap(domain) — produce a VerifiableReward-compatible callable

Domains
-------
code       — src.agent.code_execution_tool  execution correctness
math       — regex/extraction correctness against a known numeric answer
reasoning  — chain-of-thought structural checks + known-answer comparison
json       — JSON schema compliance + known-field extraction
generic    — string-equality fallback

REF: v5 brainstorm doc, v5 synthesis, Multi-Agent Reasoning Coordination mechanism.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DomainVerifierConfig:
    """Global config for the multi-domain verifier.

    Attributes:
        fail_fast:         If True, return 0 immediately on first mismatch.
                           If False, continue and return a fractional score.
        json_strict:       If True, invalid JSON is scored 0.0. If False,
                           invalid JSON gets a partial schema score.
        math_tolerance_pct: Relative tolerance (%) for "near-correct" math.
                            Default 5% (matches existing MathReward behaviour).
        reasoning_min_words: Minimum word count for a "valid" reasoning answer.
        default_fallback:  Default score when no verifier matches a domain.
        floor:              Minimum score returned by any verifier.
        ceiling:            Maximum score returned by any verifier.
        all_pass_bonus:     Bonus added when a code task passes all tests
                            (applied before floor/ceiling clamping).
    """

    fail_fast: bool = False
    json_strict: bool = True
    math_tolerance_pct: float = 5.0
    reasoning_min_words: int = 15
    default_fallback: float = 0.0
    floor: float = 0.0
    ceiling: float = 1.0
    all_pass_bonus: float = 0.0


# ---------------------------------------------------------------------------
# Domain verifier signatures
# ---------------------------------------------------------------------------

# (prompt: str, completion: str, **kwargs) -> float
VerifierFn = Callable[..., float]


# ---------------------------------------------------------------------------
# Built-in verifiers
# ---------------------------------------------------------------------------


def _code_verifier(
    prompt: str,
    completion: str,
    test_runner: Callable[[str, str, str | None], tuple[int, int, list[str]]] | None = None,
    task_id: str | None = None,
    floor: float = 0.0,
    ceiling: float = 1.0,
    all_pass_bonus: float = 0.0,
    **_: Any,
) -> float:
    """Code execution verifier with optional floor/ceiling/all-pass bonus.

    Expects ``test_runner`` to be a callable returning
    ``(passed, total, details)`` — matches ``CodeExecutionTool.run_tests``.
    Falls back to 0.0 if no test_runner is provided.
    """
    if test_runner is None:
        return 0.0
    try:
        passed_t, total_t, _details = test_runner(prompt, completion, task_id)  # type: ignore[call-arg]
    except Exception as exc:
        logger.warning("code_verifier exception: %s", exc)
        return floor
    if total_t == 0:
        return floor
    score = passed_t / total_t
    if passed_t == total_t and all_pass_bonus:
        score += all_pass_bonus
    return max(floor, min(score, ceiling))


def _math_verifier(
    prompt: str,
    completion: str,
    answer: str | float | None = None,
    tolerance_pct: float = 5.0,
    **_: Any,
) -> float:
    """Math verifier — exact or near-match numeric extraction.

    Extracts the last number from the completion and compares to ``answer``.
    Returns 1.0 for exact match, 0.5 for within tolerance, 0.0 otherwise.
    """
    if answer is None:
        return 0.0

    try:
        target = float(answer)
    except (TypeError, ValueError):
        return 0.0

    # Extract the last number from the completion
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(completion))
    if not numbers:
        return 0.0
    try:
        predicted = float(numbers[-1])
    except (TypeError, ValueError):
        return 0.0

    if predicted == target:
        return 1.0

    rel_err = abs(predicted - target) / max(abs(target), 1e-9)
    tol = tolerance_pct / 100.0
    return 0.5 if rel_err <= tol else 0.0


def _json_verifier(
    prompt: str,
    completion: str,
    required_keys: list[str] | None = None,
    schema: dict | None = None,
    strict: bool = True,
    **_: Any,
) -> float:
    """JSON/structured-output verifier.

    Score scheme:
        1.0 — valid JSON, all required keys present
        0.7 — valid JSON, some required keys missing
        0.3 — invalid JSON, but required keys found via regex
        0.0 — nothing recoverable
    """
    required_keys = required_keys or []
    score = 0.0
    parsed: dict | None = None

    # Try parse
    text = str(completion).strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|javascript|python)?", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        parsed = json.loads(text)
        score = 0.7  # valid JSON base
    except (json.JSONDecodeError, ValueError):
        if strict:
            return 0.0
        # Non-strict: check key presence by regex word-boundary match;
        # handles "key":  (JSON-style) and bare  key  mentions in prose.
        for key in required_keys:
            keys_pat = rf'(?<!\w)(?:"{re.escape(key)}"(?:\s*:|\s)|(?<!\w){re.escape(key)}(?:\s|$|[:=]))'
            if re.search(keys_pat, completion):
                score = max(score, 0.3)

    if not isinstance(parsed, dict):
        parsed = {}

    # Grade required keys
    found = sum(1 for k in required_keys if k in parsed)
    if required_keys:
        key_score = found / len(required_keys)
        score = max(score, 0.7 + 0.3 * key_score)  # up to 1.0

    return min(score, 1.0)


def _reasoning_verifier(
    prompt: str,
    completion: str,
    answer: str | None = None,
    min_words: int = 15,
    **_: Any,
) -> float:
    """Reasoning / chain-of-thought verifier (best-effort proxy).

    A lightweight verifier for when no execution tool is available:

    1. Length check: completion must have at least ``min_words`` words.
    2. If ``answer`` is given, extract the final answer and compare.

    Score:
        0.0 — too short
        0.5 — right structure / format but wrong final answer
        1.0 — final answer matches
    """
    words = str(completion).split()
    if len(words) < min_words:
        return 0.0

    if answer is None:
        # Can't evaluate correctness — give structural credit
        return 0.5

    # Try to extract the answer — look for \boxed{...} or last line
    boxed = re.search(r"\\boxed\{([^}]*)\}", completion)
    candidate = boxed.group(1).strip() if boxed else words[-1].strip()
    if candidate.lower() == str(answer).strip().lower():
        return 1.0
    return 0.5


def _generic_verifier(
    prompt: str,
    completion: str,
    answer: str | None = None,
    **_: Any,
) -> float:
    """String-equality fallback verifier."""
    if answer is None:
        return 0.0
    return 1.0 if str(completion).strip() == str(answer).strip() else 0.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTERED_DOMAINS: dict[str, dict[str, Any]] = {
    "code": {
        "fn": _code_verifier,
        "description": "Execution-grounded: runs tests via a code executor.",
        "requires": ["test_runner"],
    },
    "math": {
        "fn": _math_verifier,
        "description": "Numeric exact/near-match against a known answer.",
        "requires": ["answer"],
    },
    "json": {
        "fn": _json_verifier,
        "description": "JSON validity + required-key presence.",
        "requires": [],
    },
    "reasoning": {
        "fn": _reasoning_verifier,
        "description": "Chain-of-thought structural + best-effort answer check.",
        "requires": [],
    },
    "generic": {
        "fn": _generic_verifier,
        "description": "String-equality fallback.",
        "requires": ["answer"],
    },
}


class MultiDomainVerifier:
    """Unified multi-domain verifier with runtime domain registration.

    Acts as a dispatcher::

        mdv = MultiDomainVerifier()
        score = mdv.verify("code", prompt, completion, task_id="...", test_runner=...)

    Register custom verifiers at runtime::

        mdv.register("custom", my_fn, requires=["special_arg"])

    Wrap as a VerifiableReward-compatible callable::

        reward_fn = mdv.reward_fn_wrap("code", test_runner=tool.run_tests)
        # matches FractionalTestReward / ExecutionGroundedReward interface

    Design: routes everything into a normalised float, so all downstream
    code (GRPO advantage, composite reward, dataset filtering) is
    domain-agnostic.
    """

    def __init__(
        self,
        config: DomainVerifierConfig | None = None,
        registry: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.config = config or DomainVerifierConfig()
        self._registry: dict[str, dict[str, Any]] = dict(REGISTERED_DOMAINS)
        if registry:
            self._registry.update(registry)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def verify(
        self,
        domain: str,
        prompt: str,
        completion: str,
        **kwargs: Any,
    ) -> float:
        """Verify completion in the given domain.

        Args:
            domain:     One of the registered domain keys (e.g. 'code', 'math').
            prompt:     Task prompt (forwarded to domain verifier).
            completion: Model output text.
            **kwargs:   Forwarded to the domain verifier.  Typical keys:
                        - test_runner: Callable for code domain.
                        - answer:      Expected answer (math, generic).
                        - task_id:     Task identifier (code).
                        - required_keys: list[str] for JSON domain.

        Returns:
            Normalised float score (domain-dependent range, typically [0,1]).
        """
        entry = self._registry.get(domain)
        if entry is None:
            logger.warning(
                "MultiDomainVerifier: unknown domain '%s' — returning fallback %.2f",
                domain,
                self.config.default_fallback,
            )
            return self.config.default_fallback

        fn: VerifierFn = entry["fn"]
        required: list[str] = entry.get("requires", [])

        # Inject config-level defaults when caller hasn't overridden them
        if domain == "json" and "strict" not in kwargs:
            kwargs["strict"] = self.config.json_strict
        if domain == "math" and "tolerance_pct" not in kwargs:
            kwargs["tolerance_pct"] = self.config.math_tolerance_pct
        if domain == "reasoning" and "min_words" not in kwargs:
            kwargs["min_words"] = self.config.reasoning_min_words
        if domain == "code":
            for key in ("floor", "ceiling", "all_pass_bonus"):
                if key not in kwargs:
                    kwargs[key] = getattr(self.config, key)

        # Warn on missing required kwargs but don't fail hard
        missing = [k for k in required if k not in kwargs]
        if missing:
            logger.warning(
                "MultiDomainVerifier: domain '%s' missing required args %s — "
                "verifier may return 0.0",
                domain,
                missing,
            )

        score = fn(prompt, completion, **kwargs)

        if self.config.fail_fast and score < 1.0:
            return 0.0

        # Global floor/ceiling clamping
        return float(max(self.config.floor, min(score, self.config.ceiling)))

    def register(
        self,
        domain: str,
        fn: VerifierFn,
        *,
        description: str = "",
        requires: list[str] | None = None,
    ) -> None:
        """Register (or override) a domain verifier at runtime.

        Args:
            domain:     Domain key string.
            fn:         Callable with signature ``(prompt, completion, **kwargs) -> float``.
            description: Human-readable description (for introspection).
            requires:   List of ``**kwargs`` keys this verifier needs.
        """
        self._registry[domain] = {
            "fn": fn,
            "description": description or f"Custom verifier for '{domain}'.",
            "requires": requires or [],
        }
        logger.info("MultiDomainVerifier: registered domain '%s'.", domain)

    def registered_domains(self) -> list[str]:
        """Return list of registered domain keys."""
        return list(self._registry.keys())

    def domain_info(self, domain: str) -> dict[str, Any]:
        """Return metadata for a registered domain."""
        return self._registry.get(domain, {})

    def reward_fn_wrap(
        self,
        domain: str,
        **fixed_kwargs: Any,
    ) -> Callable[[str, str], float]:
        """Produce a VerifiableReward-compatible (prompt, completion) -> float callable.

        The returned callable closes over ``domain`` and ``fixed_kwargs``,
        so it can be used as a drop-in reward_fn for RLVRTrainer::

            reward_fn = mdv.reward_fn_wrap(
                "code",
                test_runner=tool.run_tests,
                task_id="human_eval_0",
            )
            # matches FractionalTestReward / ExecutionGroundedReward signature

        Args:
            domain:      Domain key to dispatch to.
            **fixed_kwargs: Keyword arguments forwarded to the domain verifier
                            on every call.  Override per-call by passing kwargs
                            when calling the returned fn.

        Returns:
            Callable ``(prompt, completion, **override_kwargs) -> float``.
        """
        def _reward_fn(
            prompt: str,
            completion: str,
            **override_kwargs: Any,
        ) -> float:
            merged = {**fixed_kwargs, **override_kwargs}
            return self.verify(domain, prompt, completion, **merged)

        return _reward_fn
