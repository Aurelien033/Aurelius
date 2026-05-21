"""Tests for agent_registry.AgentMemoryBridgeContract._verify_write_shape.

Covers happy paths, all failure modes, tag validation, tool_name guard,
agent_type cross-validation, and the WriteShapeReport summary helpers.
"""

from __future__ import annotations

import pytest

from agent.agent_registry import (
    AGENT_REGISTRY,
    AgentMemoryBridgeContract,
    AgentType,
    WriteShapeReport,
)

# Canonical registry lookup for use in cross-validation tests
CODE_AGENT = AGENT_REGISTRY["code"]


# ── helpers ──────────────────────────────────────────────────────────────

def ok_item(**kwargs) -> dict:
    """Minimal valid write-shape dict."""
    defaults = {"memory_type": "observation", "content": "test"}
    defaults.update(kwargs)
    return defaults


# ── WriteShapeReport behaviour ───────────────────────────────────────────

class TestWriteShapeReport:
    def test_bool_true_when_valid(self):
        r = WriteShapeReport(valid=True, total_items=3, valid_items=3)
        assert bool(r) is True

    def test_bool_false_when_invalid(self):
        r = WriteShapeReport(valid=False, total_items=3, valid_items=2)
        assert bool(r) is False

    def test_invalid_count(self):
        r = WriteShapeReport(total_items=5, valid_items=3)
        assert r.invalid_count == 2

    def test_repr(self):
        r = WriteShapeReport(valid=False, total_items=2, valid_items=0, violations=["bad"])
        assert "valid=False" in repr(r)
        assert "2" in repr(r)

    def test_empty_violations_default(self):
        r = WriteShapeReport()
        assert r.violations == []

    def test_existing_agent_type_with_capabilities(self):
        assert len(CODE_AGENT.capabilities) > 0


# ── Happy path ───────────────────────────────────────────────────────────

class TestHappyPath:
    def setup_method(self):
        self.contract = AgentMemoryBridgeContract()

    def test_empty_batch_is_valid(self):
        r = self.contract._verify_write_shape([], agent_type=CODE_AGENT)
        assert r.valid
        assert r.valid_items == 0

    def test_single_ok_batch(self):
        r = self.contract._verify_write_shape([ok_item()], agent_type=CODE_AGENT)
        assert r.valid
        assert r.valid_items == 1

    def test_multiple_ok_items(self):
        batch = [ok_item(memory_type=t, content=f"item-{i}")
                 for i, t in enumerate(["observation", "decision", "reflection", "fact", "plan"])]
        r = self.contract._verify_write_shape(batch)
        assert r.valid
        assert r.valid_items == 5

    def test_importance_zero_and_one_accepted(self):
        for imp in (0.0, 1.0):
            r = self.contract._verify_write_shape([ok_item(importance=imp)])
            assert r.valid, f"importance={imp} should be accepted"

    def test_tags_list_accepted(self):
        r = self.contract._verify_write_shape([ok_item(tags=["important"])])
        assert r.valid

    def test_without_agent_type(self):
        r = self.contract._verify_write_shape([ok_item()], agent_type=None)
        assert r.valid


# ── Invalid memory_type ──────────────────────────────────────────────────

class TestInvalidMemoryType:
    def setup_method(self):
        self.contract = AgentMemoryBridgeContract()

    def test_unknown_memory_type(self):
        r = self.contract._verify_write_shape([ok_item(memory_type="wild_guess")])
        assert not r.valid
        assert any("unrecognised" in v for v in r.violations)

    def test_empty_memory_type(self):
        r = self.contract._verify_write_shape([ok_item(memory_type="")])
        assert not r.valid


# ── Invalid / missing keys ────────────────────────────────────────────────

class TestMissingKeys:
    def setup_method(self):
        self.contract = AgentMemoryBridgeContract()

    def test_missing_content(self):
        bad = {"memory_type": "observation"}
        r = self.contract._verify_write_shape([bad])
        assert not r.valid
        assert any("content" in v for v in r.violations)

    def test_missing_memory_type(self):
        bad = {"content": "x"}
        r = self.contract._verify_write_shape([bad])
        assert not r.valid
        assert any("memory_type" in v for v in r.violations)

    def test_item_not_dict(self):
        r = self.contract._verify_write_shape(["not a dict"])  # type: ignore[list-item]
        assert not r.valid
        assert any("not a dict" in v for v in r.violations)


# ── importance range ─────────────────────────────────────────────────────

class TestImportanceRange:
    def setup_method(self):
        self.contract = AgentMemoryBridgeContract()

    @pytest.mark.parametrize("bad", [-0.1, 1.1, 999.0, -1.0])
    def test_out_of_range(self, bad: float):
        r = self.contract._verify_write_shape([ok_item(importance=bad)])
        assert not r.valid
        assert any("outside [0, 1]" in v for v in r.violations)

    def test_non_numeric_importance(self):
        r = self.contract._verify_write_shape([ok_item(importance="high")])  # type: ignore[dict-item]
        assert not r.valid
        assert any("not numeric" in v for v in r.violations)


# ── tags validation ──────────────────────────────────────────────────────

class TestTagsValidation:
    def setup_method(self):
        self.contract = AgentMemoryBridgeContract()

    def test_tags_must_be_list(self):
        r = self.contract._verify_write_shape([ok_item(tags="oops")])  # type: ignore[dict-item]
        assert not r.valid
        assert any("tags" in v for v in r.violations)

    def test_tags_must_contain_strings(self):
        r = self.contract._verify_write_shape([ok_item(tags=[1, 2])])  # type: ignore[dict-item]
        assert not r.valid
        assert any("tags" in v for v in r.violations)


# ── tool_name cross-validation ───────────────────────────────────────────

class TestToolNameValidation:
    def setup_method(self):
        self.contract = AgentMemoryBridgeContract()

    def test_known_tool_accepted(self):
        # "python" is in CODE_AGENT.tools
        r = self.contract._verify_write_shape(
            [ok_item(tool_name="python")],
            agent_type=CODE_AGENT,
        )
        assert r.valid, f"known tool 'python' should be accepted: {r.violations}"

    def test_unknown_tool_rejected(self):
        r = self.contract._verify_write_shape(
            [ok_item(tool_name="totally_fake_tool_xyz")],
            agent_type=CODE_AGENT,
        )
        assert not r.valid
        assert any("tool_name" in v and "not in" in v for v in r.violations)

    def test_no_agent_type_skips_tool_check(self):
        r = self.contract._verify_write_shape(
            [ok_item(tool_name="totally_fake_tool_xyz")],
            agent_type=None,
        )
        assert r.valid  # no agent_type → no cross-validation


# ── Mixed batch ──────────────────────────────────────────────────────────

class TestMixedBatch:
    def setup_method(self):
        self.contract = AgentMemoryBridgeContract()

    def test_all_good_items(self):
        batch = [ok_item(memory_type=t, content=f"item-{t}")
                 for t in ["observation", "plan", "fact", "decision"]]
        r = self.contract._verify_write_shape(batch)
        assert r.valid
        assert r.valid_items == 4

    def test_single_bad_item_in_good_batch(self):
        good = ok_item(memory_type="observation", content="good")
        bad = ok_item(memory_type="not_a_type", content="bad")
        r = self.contract._verify_write_shape([good, bad])
        assert not r.valid
        assert r.valid_items == 1
        assert r.invalid_count == 1

    def test_multiple_bad_items(self):
        batch = [ok_item(memory_type="observation"), {"content": "no memory_type"}, ok_item(tags="x")]  # type: ignore[dict-item]
        r = self.contract._verify_write_shape(batch)
        assert not r.valid
        assert r.invalid_count == 2
        assert r.valid_items == 1
