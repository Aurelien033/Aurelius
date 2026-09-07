"""LLM-backed debate voices for the Memory Debate protocol.

Provides `LLMDebateVoices`, a class that wraps a configured LLM API
(e.g., OpenRouter, Alibaba, Anthropic) to provide `ProposerFn`,
`SkepticFn`, and `JudgeFn` callables for the `MemoryDebateController`.

This allows the debate protocol to use real reasoning models instead of
rule-based mocks.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from src.memory.amc_runtime_cache import AMCMemoryBlock
from src.memory.memory_debate import (
    DebateDecision,
    DebateVerdict,
)

# Type aliases for clarity
LLMCallable = Callable[[list[dict[str, str]]], str]


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for the LLM API used in debate voices."""

    base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3.6-plus"
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = 30.0

    def __post_init__(self) -> None:
        api_key = self.api_key
        if api_key is None:
            api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError(
                "api_key must be provided or set via DASHSCOPE_API_KEY / OPENROUTER_API_KEY env var"
            )
        object.__setattr__(self, "api_key", api_key)


class LLMDebateVoices:
    """Provides LLM-backed debate functions for the Memory Debate protocol.

    Usage:
        voices = LLMDebateVoices()
        ctrl = MemoryDebateController()
        verdict = ctrl.run_debate(
            block,
            propose_fn=voices.propose,
            skeptic_fn=voices.skeptic,
            judge_fn=voices.judge,
        )
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.cfg = config or LLMConfig()
        self._client = httpx.Client(timeout=self.cfg.timeout)

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Make a single LLM API call and return the response text."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.api_key}",
        }
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        resp = self._client.post(
            f"{self.cfg.base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    # ── Injectable debate voices ─────────────────────────────────────────────

    def propose(self, block: AMCMemoryBlock) -> str:
        """Argue why this block should be admitted to Tier-3 memory."""
        prompt = (
            f"You are a memory proposer. Your task is to argue why the "
            f"following memory block should be admitted to the high-trust "
            f"(Tier-3) memory of an AI system.\n\n"
            f"Block ID: {block.block_id}\n"
            f"Tokens: {block.tokens[:50]}{'...' if len(block.tokens) > 50 else ''}\n"
            f"Provenance: {block.provenance}\n"
            f"Trust State: {block.trust_state}\n"
            f"Salience: {block.salience}\n\n"
            f"Provide a concise argument (max 3 sentences) for admission."
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        return self._call_llm(messages)

    def skeptic(self, block: AMCMemoryBlock, proposer_argument: str) -> str:
        """Argue against the block's admission, countering the proposer."""
        prompt = (
            f"You are a memory skeptic. A proposer has argued that a memory "
            f"block should be admitted. Your task is to critique that argument "
            f"and identify any risks (e.g., poisoning, staleness, contradiction, "
            f"over-specificity, hallucination).\n\n"
            f"Block ID: {block.block_id}\n"
            f"Tokens: {block.tokens[:50]}{'...' if len(block.tokens) > 50 else ''}\n"
            f"Provenance: {block.provenance}\n\n"
            f"Proposer's Argument:\n{proposer_argument}\n\n"
            f"Provide a concise counterargument (max 3 sentences)."
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        return self._call_llm(messages)

    def judge(
        self,
        block: AMCMemoryBlock,
        proposer_argument: str,
        skeptic_argument: str,
    ) -> DebateVerdict:
        """Render a verdict based on both arguments."""
        prompt = (
            f"You are a memory judge. Based on the arguments below, decide "
            f"whether the memory block should be ADMITTED, QUARANTINED, or REJECTED.\n\n"
            f"Block ID: {block.block_id}\n"
            f"Provenance: {block.provenance}\n\n"
            f"Proposer: {proposer_argument}\n"
            f"Skeptic: {skeptic_argument}\n\n"
            f"Respond in exactly this JSON format:\n"
            f'{{"decision": "admit"|"quarantine"|"reject", '
            f'"reason": "...", "judge_confidence": 0.0 to 1.0}}'
        )
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Respond ONLY with valid JSON.",
            },
            {"role": "user", "content": prompt},
        ]
        text = self._call_llm(messages)
        try:
            data = json.loads(text)
            decision_map = {
                "admit": DebateDecision.ADMIT,
                "quarantine": DebateDecision.QUARANTINE,
                "reject": DebateDecision.REJECT,
            }
            return DebateVerdict(
                decision=decision_map[data["decision"].lower()],
                reason=data["reason"],
                proposer_argument=proposer_argument,
                skeptic_argument=skeptic_argument,
                judge_confidence=float(data["judge_confidence"]),
                block_id=block.block_id,
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback verdict if LLM responds with malformed JSON
            return DebateVerdict(
                decision=DebateDecision.QUARANTINE,
                reason="LLM response was malformed; defaulting to quarantine",
                proposer_argument=proposer_argument,
                skeptic_argument=skeptic_argument,
                judge_confidence=0.0,
                block_id=block.block_id,
            )
