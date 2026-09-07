"""Why-density scorer (Teaching-Claude-Why: reasons >> actions).

Deterministic, dependency-free proxy for "does this training example explain the
PRINCIPLE behind the safe action, or just show the action?" Anthropic found
reason-rich data cut misalignment ~22%->3% vs action-only ~22%->15%; the final
reason-grounded rewrite carried ~19x of the quality. So a data GATE that rejects
action-only synthetic safety data is high-leverage.

This is a lexical proxy, not a judge: it counts principle-causal markers,
normalizes by length, and flags examples that decide/act without reasoning. Use
it to FILTER a generated alignment corpus and to score a held-out sample before
training. It is intentionally conservative and transparent.
"""

from __future__ import annotations

import re

# markers of causal/principled deliberation (why), not just what was done
_WHY_MARKERS = (
    "because",
    "since",
    "the reason",
    "in order to",
    "so that",
    "otherwise",
    "the principle",
    "this violates",
    "this would",
    "the risk is",
    "which could",
    "is wrong because",
    "is unsafe because",
    "the harm",
    "trade-off",
    "tradeoff",
    "on one hand",
    "on the other hand",
    "however",
    "whereas",
    "even though",
    "the constitution",
    "the policy",
    "reversible",
    "blast radius",
    "oversight",
    "instead",
    "rather than",
    "the safer",
    "i should not",
    "we should not",
    "consent",
    "honest",
    "deception",
    "integrity",
    "accountab",
)
# markers of ethical/normative framing (principle present)
_PRINCIPLE_MARKERS = (
    "honest",
    "deceiv",
    "deception",
    "transparen",
    "consent",
    "harm",
    "safety",
    "safe",
    "unsafe",
    "oversight",
    "accountab",
    "integrity",
    "trust",
    "manipulat",
    "coerc",
    "blackmail",
    "sabotage",
    "reversible",
    "irreversible",
    "escalate",
    "disclose",
    "principle",
    "policy",
    "constitution",
    "boundary",
    "boundaries",
)
_WORD_RE = re.compile(r"[a-z']+")


def _count_markers(text_lc: str, markers: tuple[str, ...]) -> int:
    return sum(text_lc.count(m) for m in markers)


def why_density_score(text: str) -> float:
    """Return principle-causal marker density per 100 words (0.0+).

    Combines why-markers (weight 1.0) and principle-markers (weight 0.5, they
    signal the ethical frame even without an explicit connective)."""
    if not text:
        return 0.0
    lc = text.lower()
    n_words = max(len(_WORD_RE.findall(lc)), 1)
    score = _count_markers(lc, _WHY_MARKERS) + 0.5 * _count_markers(lc, _PRINCIPLE_MARKERS)
    return 100.0 * score / n_words


def is_action_only(text: str, min_why: float = 1.5) -> bool:
    """True if the example decides/acts without enough reasoning density.
    `min_why` = markers per 100 words below which we call it action-only."""
    return why_density_score(text) < min_why


def score_dataset(texts: list[str], min_why: float = 1.5) -> dict:
    """Score a corpus. Returns density stats + the fraction that would be kept by
    the why-gate. Use `keep_frac` and `mean_density` as data-quality signals."""
    if not texts:
        return {
            "n": 0,
            "mean_density": 0.0,
            "median_density": 0.0,
            "keep_frac": 0.0,
            "action_only_frac": 0.0,
        }
    ds = sorted(why_density_score(t) for t in texts)
    n = len(ds)
    kept = sum(d >= min_why for d in ds)
    return {
        "n": n,
        "mean_density": sum(ds) / n,
        "median_density": ds[n // 2],
        "keep_frac": kept / n,
        "action_only_frac": (n - kept) / n,
    }


def passes_why_gate(texts: list[str], min_keep_frac: float = 0.7, min_why: float = 1.5) -> bool:
    """Dataset-level gate: at least `min_keep_frac` of examples must be reason-rich.
    Reject an alignment corpus that is mostly action-only."""
    return score_dataset(texts, min_why=min_why)["keep_frac"] >= min_keep_frac
