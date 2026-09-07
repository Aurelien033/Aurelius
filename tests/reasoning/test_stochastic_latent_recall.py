"""Tests for P0.6 SLR replayable stochastic recall (OC-11)."""

from __future__ import annotations

import importlib
import inspect
import json
import math
from pathlib import Path

import pytest

from src.reasoning.stochastic_latent_recall import (
    REDACTED,
    SLRCandidate,
    SLRCandidateStatus,
    SLRCommitPolicy,
    SLRConfig,
    SLRDisabledError,
    SLRQuery,
    SLRRecallMode,
    SLRSelectionMetric,
    SLRSelectionRecord,
    build_slr_replay_record,
    default_slr_config,
    generate_replayable_perturbation,
    generate_slr_candidates,
    run_slr_recall_trial,
    sanitize_slr_payload,
    select_slr_candidate,
    stable_json_dumps,
    stable_sha256,
)


def _query(query_id: str = "q-1") -> SLRQuery:
    return SLRQuery(
        query_id=query_id,
        query_text="What is the capital of France?",
        memory_scope=("session-1",),
        baseline_candidate_ids=("base-1",),
    )


def _enabled_config(**kwargs: object) -> SLRConfig:
    base = default_slr_config(enabled=True, k=3, replay_seed=42, noise_sigma=0.1)
    if not kwargs:
        return base
    return SLRConfig(
        enabled=bool(kwargs.get("enabled", base.enabled)),
        k=int(kwargs.get("k", base.k)),
        noise_sigma=float(kwargs.get("noise_sigma", base.noise_sigma)),
        replay_seed=int(kwargs.get("replay_seed", base.replay_seed)),
        selection_metric=kwargs.get("selection_metric", base.selection_metric),
        commit_policy=kwargs.get("commit_policy", base.commit_policy),
        recall_mode=kwargs.get("recall_mode", base.recall_mode),
        max_candidates=int(kwargs.get("max_candidates", base.max_candidates)),
        metadata=dict(kwargs.get("metadata", base.metadata)),
    )


# ── 1–5: config validation ───────────────────────────────────────────────────


def test_default_config_is_disabled() -> None:
    cfg = default_slr_config()
    assert cfg.enabled is False
    assert cfg.commit_policy == SLRCommitPolicy.PROPOSAL_ONLY
    with pytest.raises(SLRDisabledError):
        generate_slr_candidates(config=cfg, query=_query())


def test_config_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="k"):
        _enabled_config(k=0)
    with pytest.raises(ValueError, match="k"):
        _enabled_config(k=5, max_candidates=3)


def test_config_rejects_max_candidates_over_32() -> None:
    with pytest.raises(ValueError, match="max_candidates"):
        SLRConfig(
            enabled=True,
            k=1,
            noise_sigma=0.0,
            replay_seed=1,
            selection_metric=SLRSelectionMetric.SCORE,
            commit_policy=SLRCommitPolicy.PROPOSAL_ONLY,
            recall_mode=SLRRecallMode.QUERY_PERTURBATION,
            max_candidates=33,
        )


def test_config_rejects_negative_noise_sigma() -> None:
    with pytest.raises(ValueError, match="noise_sigma"):
        _enabled_config(noise_sigma=-0.01)


# ── 6–8: perturbation ────────────────────────────────────────────────────────


def test_zero_noise_produces_zero_perturbation() -> None:
    assert generate_replayable_perturbation(
        replay_seed=99,
        trajectory_index=1,
        width=4,
        noise_sigma=0.0,
    ) == (0.0, 0.0, 0.0, 0.0)


def test_same_seed_produces_identical_candidates() -> None:
    cfg = _enabled_config(replay_seed=7, k=3)
    q = _query("same")
    a = generate_slr_candidates(config=cfg, query=q)
    b = generate_slr_candidates(config=cfg, query=q)
    assert len(a) == len(b) == 3
    for left, right in zip(a, b, strict=True):
        assert left.perturbation == right.perturbation
        assert left.recall_keys == right.recall_keys
        assert left.score == right.score


def test_different_trajectory_index_differs_with_noise() -> None:
    p0 = generate_replayable_perturbation(replay_seed=1, trajectory_index=0, width=6, noise_sigma=0.5)
    p1 = generate_replayable_perturbation(replay_seed=1, trajectory_index=1, width=6, noise_sigma=0.5)
    assert p0 != p1


# ── 9–11: candidate generation / validation ──────────────────────────────────


def test_generate_produces_exactly_k_candidates() -> None:
    candidates = generate_slr_candidates(config=_enabled_config(k=4), query=_query())
    assert len(candidates) == 4


def test_candidate_scores_finite() -> None:
    for cand in generate_slr_candidates(config=_enabled_config(k=2), query=_query()):
        assert math.isfinite(cand.score)
        assert math.isfinite(cand.confidence)


def test_confidence_bounds_enforced() -> None:
    with pytest.raises(ValueError, match="confidence"):
        SLRCandidate(
            candidate_id="c",
            trajectory_index=0,
            replay_seed=1,
            perturbation=(0.0,),
            recall_keys=("k",),
            recalled_entry_ids=(),
            score=0.5,
            confidence=1.5,
            verifier_score=None,
            cost=None,
            status=SLRCandidateStatus.GENERATED,
        )


# ── 12–14: deterministic selection ───────────────────────────────────────────


def _candidates_for_selection() -> tuple[SLRCandidate, ...]:
    return tuple(
        SLRCandidate(
            candidate_id=f"cand-{i}",
            trajectory_index=i,
            replay_seed=42,
            perturbation=(float(i),),
            recall_keys=(f"key-{i}",),
            recalled_entry_ids=(f"entry-{i}",),
            score=float(i),
            confidence=0.5,
            verifier_score=None,
            cost=None,
            status=SLRCandidateStatus.GENERATED,
        )
        for i in range(3)
    )


def test_selection_by_score_is_deterministic() -> None:
    cands = _candidates_for_selection()
    sel = select_slr_candidate(
        candidates=cands,
        metric=SLRSelectionMetric.SCORE,
        query_id="q",
        replay_seed=42,
    )
    assert sel.selected_candidate_id == "cand-2"


def test_selection_tiebreaker_by_candidate_id() -> None:
    tied = (
        SLRCandidate(
            candidate_id="b",
            trajectory_index=0,
            replay_seed=1,
            perturbation=(),
            recall_keys=(),
            recalled_entry_ids=(),
            score=1.0,
            confidence=0.5,
            verifier_score=None,
            cost=None,
            status=SLRCandidateStatus.GENERATED,
        ),
        SLRCandidate(
            candidate_id="a",
            trajectory_index=1,
            replay_seed=1,
            perturbation=(),
            recall_keys=(),
            recalled_entry_ids=(),
            score=1.0,
            confidence=0.5,
            verifier_score=None,
            cost=None,
            status=SLRCandidateStatus.GENERATED,
        ),
    )
    sel = select_slr_candidate(
        candidates=tied,
        metric=SLRSelectionMetric.SCORE,
        query_id="q",
        replay_seed=1,
    )
    assert sel.selected_candidate_id == "a"


def test_empty_candidate_selection_fails() -> None:
    with pytest.raises(ValueError, match="candidate"):
        select_slr_candidate(
            candidates=(),
            metric=SLRSelectionMetric.SCORE,
            query_id="q",
            replay_seed=1,
        )


# ── 15–16: replay record stability ───────────────────────────────────────────


def test_replay_record_stable_across_reruns() -> None:
    trial_a = run_slr_recall_trial(config=_enabled_config(), query=_query("stable"))
    trial_b = run_slr_recall_trial(config=_enabled_config(), query=_query("stable"))
    assert trial_a.replay.config_hash == trial_b.replay.config_hash
    assert trial_a.replay.selection_hash == trial_b.replay.selection_hash
    assert trial_a.replay.candidate_hashes == trial_b.replay.candidate_hashes


def test_replay_record_changes_when_selection_changes() -> None:
    cfg = _enabled_config(k=2, replay_seed=99)
    q = _query("change")
    trial = run_slr_recall_trial(config=cfg, query=q)
    alt_selection = SLRSelectionRecord(
        selection_id="sel-alt",
        query_id=q.query_id,
        selected_candidate_id=trial.candidates[-1].candidate_id,
        candidate_ids=tuple(c.candidate_id for c in trial.candidates),
        selection_metric=cfg.selection_metric,
        replay_seed=cfg.replay_seed,
        deterministic_tiebreaker="manual",
        selected_score=trial.candidates[-1].score,
        rejected_candidate_ids=tuple(
            c.candidate_id for c in trial.candidates[:-1]
        ),
        created_at=trial.selection.created_at,
    )
    replay_alt = build_slr_replay_record(
        config=cfg,
        query=q,
        candidates=trial.candidates,
        selection=alt_selection,
    )
    assert replay_alt.selection_hash != trial.replay.selection_hash


# ── 17–20: security / import / proposal-only ────────────────────────────────


def test_metadata_secret_keys_redacted() -> None:
    cfg = _enabled_config(metadata={"api_key": "secret", "nested": {"token": "t"}})
    blob = json.dumps(cfg.to_dict())
    assert REDACTED in blob
    assert "secret" not in blob


def test_module_imports_without_heavy_deps() -> None:
    mod = importlib.import_module("src.reasoning.stochastic_latent_recall")
    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import numpy" not in source
    assert mod.SCHEMA_VERSION == "slr.v1"


def test_no_direct_commit_api() -> None:
    import src.reasoning.stochastic_latent_recall as slr

    names = {
        name
        for name, obj in inspect.getmembers(slr)
        if inspect.isfunction(obj) and "commit" in name.lower()
    }
    assert names == set()


def test_replay_seed_present_everywhere() -> None:
    trial = run_slr_recall_trial(config=_enabled_config(replay_seed=123), query=_query())
    assert trial.config.replay_seed == 123
    assert all(c.replay_seed == 123 for c in trial.candidates)
    assert trial.selection.replay_seed == 123
    assert trial.replay.replay_seed == 123


def test_stable_json_and_hash_deterministic() -> None:
    payload_a = {"b": 1, "a": 2}
    payload_b = {"a": 2, "b": 1}
    assert stable_json_dumps(payload_a) == stable_json_dumps(payload_b)
    assert stable_sha256(payload_a) == stable_sha256(payload_b)


def test_sanitize_slr_payload() -> None:
    out = sanitize_slr_payload({"password": "x", "note": "ok"})
    assert out["password"] == REDACTED
    assert out["note"] == "ok"
