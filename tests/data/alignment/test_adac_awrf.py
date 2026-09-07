"""Tests for the ADAC/AWRF alignment-data generator + clean-teacher firewall.

No network/keys: a StubTeacher returns canned reason-rich text so the pipeline
runs deterministically. Covers the firewall (fail-closed), schema validation,
scenario parsing, the write->review->rewrite->why-filter loop, and jsonl output.
"""

from __future__ import annotations

import json

import pytest

from src.data.alignment.adac_schema import (
    ADACScenario,
    parse_scenarios,
    validate_scenario,
)
from src.data.alignment.awrf_pipeline import (
    corpus_stats,
    generate_adac_corpus,
    make_example,
    write_sft_jsonl,
)
from src.data.alignment.teacher_backend import (
    CleanTeacher,
    FirewallError,
    is_clean_model,
)


# --------------------------- firewall ---------------------------
def test_is_clean_model():
    assert is_clean_model("qwen/qwen3-coder-next")
    assert is_clean_model("z-ai/glm-5.2")
    assert is_clean_model("deepseek/deepseek-v3.2")
    assert is_clean_model("ByteDance-Seed/Seed-OSS-36B-Instruct")
    assert is_clean_model("openai/gpt-oss-120b")  # Apache open — allowed
    assert not is_clean_model("anthropic/claude-3.5")  # closed
    assert not is_clean_model("openai/gpt-5")  # closed
    assert not is_clean_model("google/gemini-2.0")  # closed
    assert not is_clean_model("some-random/model")  # unknown -> not clean


def test_clean_teacher_fails_closed():
    with pytest.raises(FirewallError):
        CleanTeacher("anthropic/claude-3.5", transport=lambda *a: "x")
    # explicit override is allowed but flagged (no raise)
    t = CleanTeacher("anthropic/claude-3.5", transport=lambda *a: "ok", allow_unverified=True)
    assert t.clean is False
    assert t.complete("hi") == "ok"


def test_clean_teacher_runs_with_stub():
    calls = []

    def transport(model, prompt, temp, mx):
        calls.append((model, prompt))
        return "RESPONSE"

    t = CleanTeacher("qwen/qwen3-coder-next", transport=transport)
    assert t.clean is True
    assert t.complete("prompt") == "RESPONSE"
    assert calls and calls[0][0] == "qwen/qwen3-coder-next"


# --------------------------- schema + parsing ---------------------------
def _valid_scenario_dict(domain="coding"):
    return {
        "domain": domain,
        "aurelius_principle": ["test-integrity"],
        "user_goal": "ship a feature by Friday",
        "tempting_shortcut": "delete the failing test",
        "why_shortcut_is_instrumental": "green CI unblocks the release",
        "violated_norm_or_policy": "test-integrity; hides real defects",
        "safe_alternative": "fix or quarantine the test with a tracking ticket",
        "human_checkpoint_needed": False,
    }


def test_validate_scenario():
    assert validate_scenario(_valid_scenario_dict()) == []
    bad = _valid_scenario_dict()
    bad["domain"] = "not_a_domain"
    assert any("unknown domain" in p for p in validate_scenario(bad))
    bad2 = _valid_scenario_dict()
    del bad2["user_goal"]
    assert any("user_goal" in p for p in validate_scenario(bad2))
    bad3 = _valid_scenario_dict()
    bad3["human_checkpoint_needed"] = "yes"
    assert any("bool" in p for p in validate_scenario(bad3))


def test_parse_scenarios_from_teacher_json():
    payload = json.dumps([_valid_scenario_dict(), _valid_scenario_dict("security")])
    text = "Sure, here you go:\n" + payload + "\nHope that helps!"
    scs = parse_scenarios(text, "coding")
    assert len(scs) == 2
    assert all(isinstance(s, ADACScenario) for s in scs)
    assert scs[0].scenario_id and scs[1].domain == "security"
    # malformed -> empty, no crash
    assert parse_scenarios("no json here", "coding") == []
    assert parse_scenarios("[ {broken", "coding") == []


# --------------------------- AWRF loop ---------------------------
class StubTeacher:
    """Deterministic: draft is thin, rewrite is reason-rich (simulates AWRF lift)."""

    REASON_RICH = (
        "I understand you want to hit the deadline. The tempting shortcut is to "
        "delete the failing test, but that is wrong because it violates test-"
        "integrity and hides a real defect from oversight; the harm is that a "
        "broken path ships to users. Instead, the safer and reversible option is "
        "to quarantine the test with a tracking ticket and fix the root cause, "
        "and if the risk is high you should get a maintainer's sign-off first."
    )

    def __init__(self):
        self.n = 0

    def complete(self, prompt: str) -> str:
        self.n += 1
        # first call in make_example is the draft, second is the rewrite
        return "Just delete the test." if self.n % 2 == 1 else self.REASON_RICH


def test_make_example_applies_why_gate():
    sc = ADACScenario(**{**_valid_scenario_dict(), "scenario_id": "s1"})
    ex = make_example(StubTeacher(), sc, min_why=1.5)
    assert ex.response == StubTeacher.REASON_RICH
    assert ex.why_density > 1.5 and ex.kept is True

    # a low-why teacher gets rejected
    class Thin:
        def complete(self, p):
            return "Option B. Done."

    ex2 = make_example(Thin(), sc, min_why=1.5)
    assert ex2.kept is False


def test_generate_corpus_and_write(tmp_path):
    exs = generate_adac_corpus(
        StubTeacher(),
        domains=["coding", "security"],
        principles=["test-integrity", "honesty"],
        per_cell=1,
    )
    # StubTeacher.complete doesn't emit JSON scenarios -> generate_scenarios yields
    # none; corpus is empty but the call path is exercised without error.
    stats = corpus_stats(exs)
    assert stats["n"] == len(exs)

    # now feed a teacher that DOES emit scenarios then reason-rich responses
    class FullTeacher:
        def __init__(self):
            self.mode = "scenario"
            self.k = 0

        def complete(self, prompt):
            if "JSON array" in prompt:
                return json.dumps([_valid_scenario_dict()])
            self.k += 1
            return "delete it" if self.k % 2 == 1 else StubTeacher.REASON_RICH

    exs2 = generate_adac_corpus(
        FullTeacher(), domains=["coding"], principles=["test-integrity"], per_cell=1
    )
    assert len(exs2) >= 1 and any(e.kept for e in exs2)
    out = tmp_path / "adac.jsonl"
    n = write_sft_jsonl(exs2, str(out))
    assert n >= 1
    rows = [json.loads(line) for line in open(out)]
    assert rows and set(rows[0]) >= {"prompt", "response", "why_density", "domain", "source"}
    assert rows[0]["source"] == "adac_awrf"
