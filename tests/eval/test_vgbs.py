"""Tests for VGBS (verifier-guided beam search) — torch-free, injected callables."""

from __future__ import annotations


from src.eval.vgbs import (
    beam_cost,
    best_of_n,
    bon_cost,
    cost_ratio,
    verified_best_of_n,
    verifier_guided_beam_search,
)


# A tiny synthetic problem: build the string "AAAA"; each step appends a char.
# The PRM rewards prefixes that are all 'A' (correct path); 'B' steps score low.
def _expand(prompt, prefix):
    return ["A", "B"]  # two candidate next steps


def _score(prompt, prefix):
    if not prefix:
        return 0.5
    good = sum(1 for c in prefix if c == "A")
    return good / len(prefix)  # fraction-correct in [0,1]


def _complete(prefix):
    return len(prefix) >= 4


def test_vgbs_finds_correct_path():
    r = verifier_guided_beam_search(
        "make AAAA",
        expand=_expand,
        score_step=_score,
        is_complete=_complete,
        beam_width=4,
        max_depth=6,
    )
    assert r.completed
    assert r.best == "AAAA"  # PRM steered to the all-A path
    assert r.best_score == 1.0
    assert r.depth_reached == 4
    assert r.n_expansions > 0


def test_vgbs_beam_width_one_is_greedy():
    # width-1 = greedy by PRM; still finds AAAA here (A always scores >= B)
    r = verifier_guided_beam_search(
        "x", expand=_expand, score_step=_score, is_complete=_complete, beam_width=1, max_depth=6
    )
    assert r.best == "AAAA" and r.completed


def test_vgbs_empty_pool_graceful():
    # expand returns nothing -> no candidates; returns init gracefully
    r = verifier_guided_beam_search(
        "x",
        expand=lambda p, s: [],
        score_step=_score,
        is_complete=lambda s: False,
        beam_width=2,
        max_depth=3,
    )
    assert r.completed is False and r.n_expansions == 0


def test_vgbs_bad_width_raises():
    import pytest

    with pytest.raises(ValueError):
        verifier_guided_beam_search(
            "x", expand=_expand, score_step=_score, is_complete=_complete, beam_width=0
        )


def test_best_of_n_baseline():
    # sampler always returns "AAAB" (score 0.75); best-of-n picks it
    best, score, n = best_of_n("x", sample=lambda p: "AAAB", score_final=_score, n=8)
    assert best == "AAAB" and abs(score - 0.75) < 1e-9 and n == 8


def test_cost_accounting_matches_roadmap():
    # the 14.5x claim: b=4,d=30,N=64
    assert beam_cost(4, 30, 0.1) == 132.0
    assert bon_cost(64, 30) == 1920
    assert abs(cost_ratio(4, 30, 64, 0.1) - 1920 / 132) < 1e-9
    assert cost_ratio(4, 30, 64, 0.1) > 14 and cost_ratio(4, 30, 64, 0.1) < 15


def test_verified_best_of_n_code_domain():
    # execution verifier (terminal): pick the candidate that passes tests
    cands = ["def f(): return 0", "def f(): return 42", "broken"]

    def exec_verify(prompt, code):  # 1.0 iff it's the passing solution
        return 1.0 if code == "def f(): return 42" else 0.0

    best, score, scores = verified_best_of_n("p", cands, exec_verify)
    assert best == "def f(): return 42" and score == 1.0
    assert scores == [0.0, 1.0, 0.0]
    # empty pool safe
    assert verified_best_of_n("p", [], exec_verify) == ("", 0.0, [])
