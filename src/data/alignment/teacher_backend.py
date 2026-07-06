"""Clean-teacher backend for alignment-data generation.

The FIREWALL lives here in code, not just in docs: data generated for a
release-bound artifact may only come from open-licensed ("clean") teachers.
This module fails CLOSED — a model id that is not on the verified-clean
allowlist raises unless the caller explicitly opts out with a logged warning.

Transport is injectable so the pipeline is unit-testable with no network/keys:

    teacher = CleanTeacher("qwen/qwen3-coder-next", transport=my_stub)
    text = teacher.complete("...prompt...")

Real use wires the default OpenAI-compatible transport (OpenRouter etc.),
mirroring docs/training/make_code_traces.py.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable

# Verified open-licensed teacher families (hub-checked: Apache/MIT).
# Prefix match on the provider/model id. gpt-oss is Apache (open) and allowed;
# closed OpenAI/Anthropic/Google ids are NOT on this list -> rejected.
CLEAN_PREFIXES: tuple[str, ...] = (
    "qwen/", "qwen3", "qwen2", "z-ai/", "zai-org/", "glm",
    "deepseek/", "deepseek-ai/", "bytedance-seed/", "seed-",
    "openai/gpt-oss", "gpt-oss", "mistralai/", "meta-llama/llama-3",
    "moonshotai/kimi-linear",  # note: most Kimi are license "other" -> not blanket-clean
)
# Explicit closed denylist (belt-and-suspenders; these must never feed a release).
CLOSED_MARKERS: tuple[str, ...] = (
    "claude", "anthropic", "gpt-4", "gpt-5", "gpt-4o", "openai/o1", "openai/o3",
    "openai/o4", "gemini", "google/gemini", "grok",
)


def is_clean_model(model_id: str) -> bool:
    """True iff `model_id` is a verified open-licensed teacher (release-safe)."""
    m = model_id.strip().lower()
    if any(mark in m for mark in CLOSED_MARKERS):
        return False
    return any(m.startswith(p) or p in m for p in CLEAN_PREFIXES)


class FirewallError(RuntimeError):
    """Raised when a non-clean teacher is used for release-bound generation."""


def _default_transport(api_base: str) -> Callable[[str, str, float, int], str]:
    """Return a transport(model, prompt, temperature, max_tokens) -> text using
    an OpenAI-compatible chat endpoint. Key from env; matches make_code_traces."""
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    url = api_base.rstrip("/") + "/chat/completions"

    def _call(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
        if not key:
            raise RuntimeError("set OPENROUTER_API_KEY (or OPENAI_API_KEY)")
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"]

    return _call


class CleanTeacher:
    """A firewall-checked completion source for alignment-data generation."""

    def __init__(
        self,
        model: str,
        api_base: str = "https://openrouter.ai/api/v1",
        transport: Callable[[str, str, float, int], str] | None = None,
        allow_unverified: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        if not is_clean_model(model):
            msg = (f"teacher {model!r} is NOT on the verified-clean allowlist. "
                   "Release-bound alignment data must use open-licensed teachers "
                   "(Qwen/GLM/DeepSeek/Seed/gpt-oss). ")
            if not allow_unverified:
                raise FirewallError(msg + "Pass allow_unverified=True only for "
                                    "PRIVATE, non-release experiments.")
            print("⚠ FIREWALL OVERRIDE: " + msg + "Output must NOT enter a release pipeline.")
        self.model = model
        self.clean = is_clean_model(model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._transport = transport or _default_transport(api_base)

    def complete(self, prompt: str, temperature: float | None = None,
                 max_tokens: int | None = None) -> str:
        return self._transport(
            self.model, prompt,
            self.temperature if temperature is None else temperature,
            self.max_tokens if max_tokens is None else max_tokens,
        )
