"""Unit and contract tests for src.memory.amc_tensor_api.

Dependencies
------------
Only stdlib + pytest.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

# ── Load amc_tensor_api directly (bypass memory/__init__.py which imports
#    unified_orchestrator → relative-import beyond top-level-package error) ──
import importlib.util, pathlib as _pl

_src = _pl.Path(__file__).resolve().parents[2] / "src"
import sys as _sys
_mod_name = "memory.amc_tensor_api"
_mod_src   = _src / "memory" / "amc_tensor_api.py"
_spec = importlib.util.spec_from_file_location(
    _mod_name,
    _mod_src,
    submodule_search_locations=[str(_src / "memory")],
)
assert _spec is not None and _spec.loader is not None, "could not build spec"
_mod = importlib.util.module_from_spec(_spec)
_sys.modules[_mod_name] = _mod   # register before exec so @dataclass can find it
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

AdmissionAction            = _mod.AdmissionAction
AMCBenchmarkConfig         = _mod.AMCBenchmarkConfig
AMCMemoryModes             = _mod.AMCMemoryModes
AMCBenchmarkResult         = _mod.AMCBenchmarkResult
AMCLayerMemory             = _mod.AMCLayerMemory
AMCMemoryController        = _mod.AMCMemoryController
AMCMemoryConsolidate       = _mod.AMCMemoryConsolidate
AMCMemoryRead              = _mod.AMCMemoryRead
AMCMemoryWrite             = _mod.AMCMemoryWrite
AMCTensorState             = _mod.AMCTensorState
AMCWriteDecision           = _mod.AMCWriteDecision
MemoryTier                 = _mod.MemoryTier
build_benchmark_result     = _mod.build_benchmark_result
score_ablation             = _mod.score_ablation


# ── helpers ─────────────────────────────────────────────────────────────────

def _stub_state(layer_index: int = 0, token_count: int = 4, **kw: Any) -> AMCTensorState:
    """Return an AMCTensorState with lightweight shape-tagged stub objects."""
    k: Any = {"_shape": (1, token_count, 16), "_stub": True}
    v: Any = {"_shape": (1, token_count, 16), "_stub": True}
    return AMCTensorState(
        layer_index=layer_index,
        token_count=token_count,
        kvs=(k, v),
        **kw,
    )


# ─────────────────────────────────────────────────────────────────────────────
# AMCTensorState
# ─────────────────────────────────────────────────────────────────────────────

class TestAMCTensorState:
    def test_is_frozen(self):
        with pytest.raises(FrozenInstanceError):
            _stub_state().layer_index = 99  # type: ignore[misc]

    def test_construction(self):
        s = _stub_state(layer_index=2, token_count=128)
        assert s.layer_index == 2
        assert s.token_count == 128

    def test_rejects_negative_layer_index(self):
        with pytest.raises(ValueError, match="layer_index"):
            AMCTensorState(layer_index=-1, token_count=0, kvs=(None, None))

    def test_rejects_negative_token_count(self):
        with pytest.raises(ValueError, match="token_count"):
            AMCTensorState(layer_index=0, token_count=-1, kvs=(None, None))

    def test_kvs_must_be_tuple(self):
        with pytest.raises(TypeError, match="kvs"):
            AMCTensorState(layer_index=0, token_count=0, kvs=[None, None])  # type: ignore[arg-type]

    def test_equality(self):
        a = _stub_state(layer_index=1, token_count=10)
        b = _stub_state(layer_index=1, token_count=10)
        assert a == b

    def test_optional_fields(self):
        s = _stub_state(
            layer_index=0, token_count=4,
            dtype="float32", device_index=0, metadata={"step": 1},
        )
        assert s.dtype == "float32"
        assert s.device_index == 0
        assert s.metadata["step"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Tier-1 result shapes (AMCReadResult / AMCWriteDecision)
# ─────────────────────────────────────────────────────────────────────────────

class TestTier1Results:
    def test_write_decision_is_frozen(self):
        with pytest.raises(FrozenInstanceError):
            AMCWriteDecision(admitted=True).admitted = False  # type: ignore[misc]

    def test_write_decision_defaults(self):
        d = AMCWriteDecision(admitted=False)
        assert d.admitted is False
        assert d.action == AdmissionAction.ALLOW
        assert d.reason == ""
        assert d.surprise_score == pytest.approx(0.0)
        assert d.surprise_adjusted == pytest.approx(0.0)
        assert d.tier2_write_promoted is False

    def test_write_decision_all_fields(self):
        d = AMCWriteDecision(
            admitted=True,
            action=AdmissionAction.WARN,
            reason="suspicious token",
            surprise_score=0.72,
            surprise_adjusted=0.55,
            tier2_write_promoted=True,
        )
        assert d.admitted
        assert d.action == AdmissionAction.WARN
        assert "suspicious" in d.reason
        assert d.surprise_score == pytest.approx(0.72)
        assert d.surprise_adjusted == pytest.approx(0.55)

    def test_write_decision_rejects_bad_surprise_score(self):
        with pytest.raises(ValueError):
            AMCWriteDecision(admitted=True, surprise_score=1.5)

    def test_amc_read_result_is_frozen(self):
        with pytest.raises(FrozenInstanceError):
            _mod.AMCReadResult(token_count=0, state=_stub_state()).token_count = 1  # type: ignore[misc]

    def test_read_result_defaults(self):
        s = _stub_state()
        r = _mod.AMCReadResult(token_count=4, state=s)
        assert r.token_count == 4
        assert r.state == s
        assert r.scores is None
        assert r.metadata == {}

    def test_read_result_with_scores(self):
        s = _stub_state()
        r = _mod.AMCReadResult(token_count=4, state=s, scores=(0.1, 0.2, 0.3))
        assert r.scores == (0.1, 0.2, 0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Tier-2 / Tier-3 operation shapes
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryOpShapes:
    def test_memory_read_default_tier2(self):
        op = AMCMemoryRead(query="deployment region")
        assert op.tier == MemoryTier.TIER_2
        assert op.limit == 5
        assert op.layer_index is None

    def test_memory_read_tier3(self):
        op = AMCMemoryRead(query="past session", tier=MemoryTier.TIER_3, limit=3)
        assert op.tier == MemoryTier.TIER_3
        assert op.limit == 3

    def test_memory_read_empty_query_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            AMCMemoryRead(query="   ")

    def test_memory_read_zero_limit_rejected(self):
        with pytest.raises(ValueError, match="limit"):
            AMCMemoryRead(query="x", limit=0)

    def test_memory_write_default_tier2(self):
        op = AMCMemoryWrite(content="observed", surprise=0.7)
        assert op.tier == MemoryTier.TIER_2
        assert op.surprise == pytest.approx(0.7)
        assert op.importance is None

    def test_memory_write_empty_content_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            AMCMemoryWrite(content="", surprise=0.5)

    def test_memory_write_bad_surprise_rejected(self):
        with pytest.raises(ValueError, match="surprise"):
            AMCMemoryWrite(content="ok", surprise=1.5)

    def test_memory_consolidate_default(self):
        op = AMCMemoryConsolidate()
        assert op.tiers == (MemoryTier.TIER_3,)
        assert op.max_entries is None

    def test_memory_consolidate_tier2_plus_tier3(self):
        op = AMCMemoryConsolidate(
            tiers=(MemoryTier.TIER_2, MemoryTier.TIER_3), max_entries=500
        )
        assert op.tiers == (MemoryTier.TIER_2, MemoryTier.TIER_3)
        assert op.max_entries == 500


# ─────────────────────────────────────────────────────────────────────────────
# AMCLayerMemory protocol
# ─────────────────────────────────────────────────────────────────────────────

class _ConcreteLayerMemory:
    """Minimal concrete that satisfies the AMCLayerMemory protocol."""

    def __init__(self, layer_index: int = 0) -> None:
        self.layer_index = layer_index
        self._state: AMCTensorState | None = None

    def read(self, layer_index: int, step: int, *, seq_len: int) -> _mod.AMCReadResult:
        if self._state is None:
            raise IndexError("no state stored yet")
        return _mod.AMCReadResult(token_count=self._state.token_count, state=self._state)

    def write(
        self, state: AMCTensorState, *, step: int, surprise: float = 0.0
    ) -> AMCWriteDecision:
        self._state = state
        action = (
            AdmissionAction.QUARANTINE if surprise > 0.9 else AdmissionAction.ALLOW
        )
        return AMCWriteDecision(admitted=True, action=action, surprise_score=surprise)

    def write_observation(
        self, layer_index: int, step: int, content: str,
        *, surprise: float, importance: float | None = None,
    ) -> AMCWriteDecision:
        return AMCWriteDecision(admitted=True, surprise_score=surprise)

    def read_memory(
        self, query: str, *, tier: MemoryTier = MemoryTier.TIER_2,
        limit: int = 5, layer_index: int | None = None, step: int | None = None,
    ) -> list[dict[str, Any]]:
        return [{"content": "stub", "source": "test", "trust_level": "trusted"}]

    def consolidate(
        self, *, tiers: tuple[MemoryTier, ...] = (MemoryTier.TIER_3,),
        max_entries: int | None = None,
    ) -> dict[str, Any]:
        return {"promoted": 0, "quarantined": 0, "expired_pruned": 0, "errors": []}

    def reset(self, layer_index: int | None = None) -> None:
        self._state = None

    def stats(self) -> dict[str, Any]:
        return {"layer_index": self.layer_index, "episodic_entries": 0}


# ─────────────────────────────────────────────────────────────────────────────
# AMCMemoryController protocol
# ─────────────────────────────────────────────────────────────────────────────

class _ConcreteMemoryController:
    """Minimal concrete that satisfies AMCMemoryController protocol."""

    def __init__(self) -> None:
        self._obs: list[Any] = []

    def read(self, layer_index: int, step: int, *, seq_len: int) -> _mod.AMCReadResult:
        return _mod.AMCReadResult(
            token_count=seq_len,
            state=_stub_state(layer_index=layer_index, token_count=seq_len),
        )

    def write(self, state: AMCTensorState, *, step: int, surprise: float = 0.0) -> AMCWriteDecision:
        return AMCWriteDecision(admitted=True, surprise_score=surprise)

    def write_observation(
        self, layer_index: int, step: int, content: str,
        *, surprise: float, importance: float | None = None,
    ) -> AMCWriteDecision:
        return AMCWriteDecision(admitted=True, surprise_score=surprise)

    def reset(self, layer_index: int | None = None) -> None:
        self._obs.clear()

    def observe(
        self, role: str, content: str, *, surprise: float,
        importance: float | None = None,
    ) -> Any | None:
        return {"role": role, "content": content, "surprise": surprise}

    def retrieve(self, query: str, *, limit: int | None = None) -> list[Any]:
        return [{"content": "stub-retrieved"}]

    def promote(self, *, key: str, value: Any, **kw: Any) -> Any | None:
        return {"key": key, "value": value}

    def quarantine(self, *, key: str, value: Any, **kw: Any) -> Any:
        return {"key": key, "quarantined": True}

    def tier3_promote(self, tier2_id: Any, *, confidence: float = 1.0) -> Any | None:
        return {"tier2_id": tier2_id}

    def consolidate(self) -> Any:
        return {"promoted": 0, "quarantined": 0}

    def prior_consolidation_memories(self, m: Any, *, current_at: Any = None) -> list[Any]:
        return []

    def prioritize(self, limit: int = 5) -> list[Any]:
        return []

    def verify_and_promote(self, key: str, confidence: float = 1.0, **kw: Any) -> Any | None:
        return {"key": key, "verified": True, "confidence": confidence}

    def stats(self) -> dict[str, Any]:
        return {"component": "stub", "observed": len(self._obs)}


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark scaffold
# ─────────────────────────────────────────────────────────────────────────────

ALL_MODES = {"no_memory", "tier2_context", "tier1_only", "tier1_tier2", "full"}


class TestAMCMemoryModes:
    def test_all_modes_present(self):
        assert {m.value for m in AMCMemoryModes} == ALL_MODES

    def test_values_are_non_empty_strings(self):
        for m in AMCMemoryModes:
            assert isinstance(m.value, str)
            assert m.value

    def test_specific_identities(self):
        assert AMCMemoryModes.NO_MEMORY.value == "no_memory"
        assert AMCMemoryModes.FULL.value == "full"


class TestAMCBenchmarkConfig:
    def test_default_config(self):
        cfg = AMCBenchmarkConfig()
        assert cfg.mode == AMCMemoryModes.NO_MEMORY
        assert cfg.context_tokens == 512
        assert cfg.samples_per == 3
        assert cfg.tier1_layers == 6
        assert cfg.tier2_max_retrieved == 5
        assert cfg.tier2_surprise_threshold == pytest.approx(0.5)
        assert cfg.tier3_min_confidence == pytest.approx(0.6)

    def test_full_pipeline_config(self):
        cfg = AMCBenchmarkConfig(mode=AMCMemoryModes.FULL, tier1_layers=12)
        assert cfg.mode == AMCMemoryModes.FULL
        assert cfg.tier1_layers == 12

    def test_zero_context_tokens_rejected(self):
        with pytest.raises(ValueError, match="context_tokens"):
            AMCBenchmarkConfig(context_tokens=0)

    def test_zero_samples_per_rejected(self):
        with pytest.raises(ValueError, match="samples_per"):
            AMCBenchmarkConfig(samples_per=0)

    def test_bad_surprise_threshold_rejected(self):
        with pytest.raises(ValueError, match="tier2_surprise_threshold"):
            AMCBenchmarkConfig(tier2_surprise_threshold=2.0)

    def test_bad_confidence_rejected(self):
        with pytest.raises(ValueError, match="tier3_min_confidence"):
            AMCBenchmarkConfig(tier3_min_confidence=-0.1)


class TestAMCBenchmarkResult:
    def test_to_dict_serialisable(self):
        r = AMCBenchmarkResult(
            mode=AMCMemoryModes.TIER2_CONTEXT,
            context_tokens=1024, samples_per=5,
            overall_score=0.85,
            per_task_scores={"t": 1.0},
            results={}, elapsed_seconds=12.3,
        )
        d = r.to_dict()
        assert d["mode"] == "tier2_context"
        assert d["overall_score"] == pytest.approx(0.85)
        assert d["elapsed_seconds"] == pytest.approx(12.3)

    def test_is_frozen(self):
        with pytest.raises(FrozenInstanceError):
            AMCBenchmarkResult(
                mode=AMCMemoryModes.NO_MEMORY,
                context_tokens=512, samples_per=1,
                overall_score=0.0, per_task_scores={},
                results={}, elapsed_seconds=0.0,
            ).overall_score = 1.0  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# score_ablation / build_benchmark_result
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreAblation:
    def setup_method(self):
        self.results = {
            "t1": {"pass_rate": 1.0, "n": 3, "cells": []},
            "t2": {"pass_rate": 0.67, "n": 3, "cells": []},
        }

    def test_returns_benchmark_result(self):
        r = score_ablation(
            self.results,
            mode=AMCMemoryModes.TIER1_TIER2,
            context_tokens=512, samples_per=3,
        )
        assert isinstance(r, AMCBenchmarkResult)
        assert r.mode == AMCMemoryModes.TIER1_TIER2
        assert r.overall_score == pytest.approx(0.835, abs=1e-6)

    def test_injects_metadata(self):
        r = score_ablation(
            self.results,
            mode=AMCMemoryModes.NO_MEMORY,
            context_tokens=512, samples_per=1,
            metadata={"run_id": "test"},
        )
        assert r.metadata["run_id"] == "test"


class TestBuildBenchmarkResult:
    def test_matches_score_ablation_when_scores_equal(self):
        scores = {"t1": 1.0, "t2": 0.6}
        overall = sum(scores.values()) / len(scores)
        r1 = score_ablation(
            {k: {"pass_rate": v, "n": 1, "cells": []} for k, v in scores.items()},
            mode=AMCMemoryModes.TIER2_CONTEXT,
            context_tokens=512, samples_per=1,
        )
        r2 = build_benchmark_result(
            mode=AMCMemoryModes.TIER2_CONTEXT,
            context_tokens=512, samples_per=1,
            scores=scores, overall_score=overall, elapsed_seconds=7.5,
        )
        assert r1.overall_score == pytest.approx(r2.overall_score)
        assert r1.per_task_scores == r2.per_task_scores
