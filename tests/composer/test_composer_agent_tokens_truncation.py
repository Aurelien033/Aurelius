"""Tests for composer_agent R1 (token counting) + R2 (conversation truncation) fixes."""
from __future__ import annotations

from pathlib import Path

from src.composer.composer_agent import AgentConfig, ComposerAgent


def _agent(**cfg) -> ComposerAgent:
    return ComposerAgent(repo_root=Path("."), model_fn=lambda p: "ok",
                         config=AgentConfig(**cfg))


# --------------------------- R1: token counting ---------------------------
def test_count_tokens_char_estimate_not_word_count():
    a = _agent()
    # 40 chars -> ~10 tokens via /4; word-count would give ~8. The point: it is
    # NOT the old .split() word count, and it scales with chars.
    text = "a b c d e f g h " * 5   # 80 chars, 40 words
    est = a._count_tokens(text)
    assert est == len(text) // 4
    assert est != len(text.split())          # different from the old word count
    assert a._count_tokens("") == 1          # floor


def test_count_tokens_uses_injected_counter():
    a = _agent(token_counter=lambda t: 999)
    assert a._count_tokens("anything") == 999
    # a broken counter falls back to the char estimate, doesn't raise
    def boom(t):
        raise RuntimeError("nope")
    a2 = _agent(token_counter=boom)
    assert a2._count_tokens("abcdefgh") == 2   # 8//4


# --------------------------- R2: truncation ---------------------------
def test_truncation_keeps_system_and_recent_window():
    a = _agent(max_tokens_per_turn=25)          # budget = 100 chars
    convo = [{"role": "system", "content": "S" * 20}]
    for i in range(10):
        convo.append({"role": "assistant", "content": f"A{i}" * 10})  # ~30 chars
        convo.append({"role": "user", "content": f"U{i}" * 10})
    out = a._truncate_conversation(convo)
    # system always retained
    assert out[0]["role"] == "system"
    # total chars within budget (+ allowance for the one guaranteed recent msg)
    assert sum(len(m["content"]) for m in out) <= 100 + 30
    # the retained tail includes the MOST RECENT messages (last user turn)
    assert out[-1]["content"] == convo[-1]["content"]


def test_truncation_preserves_assistant_turns_in_window():
    # the audit's R2 alarm was "drops assistant turns"; verify they're retained
    a = _agent(max_tokens_per_turn=1000)        # generous budget -> keep all
    convo = [{"role": "system", "content": "sys"},
             {"role": "user", "content": "do X"},
             {"role": "assistant", "content": "I edited file Y"},
             {"role": "user", "content": "result: ok"}]
    out = a._truncate_conversation(convo)
    roles = [m["role"] for m in out]
    assert "assistant" in roles                 # agent's own action retained
    assert out == convo                          # everything fits -> unchanged


def test_truncation_marker_collision_safe():
    # an assistant payload containing a literal "<user>" must NOT corrupt anything
    a = _agent(max_tokens_per_turn=1000)
    convo = [{"role": "system", "content": "sys"},
             {"role": "user", "content": "task"},
             {"role": "assistant", "content": "diff:\n<user>\nnot a real turn\n</user>"},
             {"role": "user", "content": "ok"}]
    out = a._truncate_conversation(convo)
    # message boundaries are preserved as list items (no string-split on marker)
    assert len(out) == 4
    assert out[2]["content"] == "diff:\n<user>\nnot a real turn\n</user>"


def test_truncation_empty():
    a = _agent()
    assert a._truncate_conversation([]) == []
