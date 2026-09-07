"""V3 TruthSurface — the measurement backbone + promotion gate.

Every Aurelius experiment emits a claim row here; a claim may only be PROMOTED to
`confirmed` if it names a metric, baseline, falsifier, and rollback AND clears a
statistical gate. This is the honesty engine the whole roadmap rests on: it turns
"we ran something and it looked better" into "this delta cleared its pre-registered
gate at this n, at this cost."

Contains the statistical gate math (all hand-verified against the arc's receipts):
  * mcnemar(b, c)            — paired-delta significance (replicates the p≈0.63 result)
  * min_n_for_delta(...)     — sample size to resolve a delta (n≈2070 for +2pp @ p=.88)
  * ci_halfwidth(p, n)       — Wald half-width (164 → 5.0pp = "HE is a weak instrument")
  * capacity_verdict(...)    — 2SE gate for base-vs-base capacity deltas
  * vb_mca(...)              — $/verified-solution (the controller objective)
  * promotion_gate(claim)    — the enforced no-metric/baseline/falsifier/rollback -> no promote rule

Pure-python (math + optional pyyaml). Append-only JSONL ledger = event-sourced.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Statistical gates (hand-verified; see calc-brief §5)
# ---------------------------------------------------------------------------

_Z = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600}


def _chi2_1_sf(x: float) -> float:
    """Survival function P(chi^2_1 > x) = erfc(sqrt(x/2))."""
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def mcnemar(b: int, c: int, continuity: bool = True) -> dict:
    """Paired McNemar test on discordant counts (b = A-pass/B-fail gains,
    c = A-fail/B-pass, or vice-versa). Returns chi2 + approx two-sided p.

    Example (aggregation sweep b=1,c=3): chi2=0.25, p≈0.62 — NOT significant,
    replicating the logged ≈0.63.  Ship rule: p<0.05 to call a paired delta real.
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p": 1.0, "significant": False}
    diff = abs(b - c)
    if continuity:
        diff = max(diff - 1, 0)
    chi2 = (diff * diff) / n
    p = _chi2_1_sf(chi2)
    return {"b": b, "c": c, "chi2": round(chi2, 4), "p": round(p, 4),
            "significant": p < 0.05}


def min_n_for_delta(p: float, delta: float, power: float = 0.80,
                    alpha_two_sided: float = 0.05) -> int:
    """Sample size to resolve a `delta` (in proportion, e.g. 0.02) at base rate
    `p`. n = (z_alpha + z_beta)^2 * p(1-p) / delta^2.

    p=0.88, delta=0.02 -> ~2070 ; delta=0.05 -> ~331. This is why capacity
    claims need n>=4000 or the less-saturated MBPP.
    """
    if delta <= 0:
        raise ValueError("delta must be > 0")
    z_a = _Z[1 - alpha_two_sided / 2]
    z_b = _Z[power]
    return math.ceil((z_a + z_b) ** 2 * p * (1 - p) / (delta * delta))


def ci_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    """Wald CI half-width for a proportion. n=164,p=.88 -> 0.0497 (~5pp = 2SE)."""
    if n <= 0:
        return 1.0
    return z * math.sqrt(p * (1 - p) / n)


def capacity_verdict(new_pp: float, base_pp: float, n: int) -> dict:
    """Is `new` > `base` beyond 2 SE of the base rate? (base-vs-base, no training)."""
    se = ci_halfwidth(base_pp / 100.0, n, z=1.0) * 100
    d = new_pp - base_pp
    if d > 2 * se:
        verdict = "CONFIRMED"
    elif d > 0:
        verdict = "SUGGESTIVE (<2SE; use a less-saturated bench)"
    else:
        verdict = "NO GAIN (likely saturation)"
    return {"delta_pp": round(d, 2), "two_se_pp": round(2 * se, 2), "verdict": verdict}


def vb_mca(delta_pp: float, n_tasks: int, cost_usd: float) -> dict:
    """Verified-solved-tasks-per-dollar (inverted): $/verified-solution.
    dSolved = delta_pp/100 * n_tasks. Lower $/solution = better. The controller
    objective — capacity steps dominate this vs sub-1pp post-train tricks.
    """
    dsolved = (delta_pp / 100.0) * n_tasks
    cost_per = (cost_usd / dsolved) if dsolved > 0 else float("inf")
    return {"d_solved": round(dsolved, 2), "cost_usd": cost_usd,
            "usd_per_solution": round(cost_per, 2) if cost_per != float("inf") else None}


# ---------------------------------------------------------------------------
# Claim record + ledger
# ---------------------------------------------------------------------------

_STATUSES = ("proposed", "confirmed", "refuted", "noise")


@dataclass
class ClaimRecord:
    """One measured claim. metric/baseline/falsifier/rollback are REQUIRED for
    promotion (the completeness-blueprint Phase-0 exit gate)."""
    claim_id: str
    statement: str
    metric: str = ""              # e.g. "HumanEval pass@1"
    baseline: str = ""            # e.g. "Qwen3-8B base 84.1%"
    falsifier: str = ""           # the pre-registered kill condition
    rollback: str = ""            # what to revert to if it fails in prod
    value: float | None = None    # measured metric value (pp or rate)
    baseline_value: float | None = None
    n: int | None = None
    cost_usd: float | None = None
    status: str = "proposed"
    bench: str = ""
    notes: str = ""
    ts: float = field(default_factory=time.time)

    def missing_fields(self) -> list[str]:
        return [f for f in ("metric", "baseline", "falsifier", "rollback")
                if not getattr(self, f)]


def promotion_gate(claim: ClaimRecord) -> dict:
    """Can this claim be promoted to `confirmed`? Enforces the four required
    fields + a significance check (delta beyond 2*CI half-width) when values +n
    are present. Returns {ok, reasons}."""
    reasons: list[str] = []
    miss = claim.missing_fields()
    if miss:
        reasons.append(f"missing required field(s): {', '.join(miss)}")
    sig = None
    if claim.value is not None and claim.baseline_value is not None and claim.n:
        d = claim.value - claim.baseline_value
        hw = ci_halfwidth(claim.baseline_value / 100.0, claim.n, z=1.96) * 100
        sig = abs(d) >= hw
        if not sig:
            reasons.append(f"delta {d:+.2f}pp within CI half-width ±{hw:.2f}pp "
                           f"(n={claim.n}) — not significant; raise n or use MBPP")
    else:
        reasons.append("no value/baseline_value/n -> significance unprovable")
    return {"ok": not reasons, "significant": sig, "reasons": reasons}


class TruthSurface:
    """Append-only claim ledger with the promotion gate."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.claims: list[ClaimRecord] = []
        if self.path and self.path.exists():
            self._load()

    def add(self, claim: ClaimRecord, *, promote: bool = False) -> dict:
        """Append a claim. If promote=True, run the gate and set status
        confirmed/noise accordingly. Returns the gate result."""
        gate = promotion_gate(claim)
        if promote:
            claim.status = "confirmed" if gate["ok"] else "noise"
        self.claims.append(claim)
        if self.path:
            self._append(claim)
        return gate

    def refute(self, claim_id: str, note: str = "") -> None:
        """Mark an existing claim refuted (e.g. the RLVR-beat-base withdrawal)."""
        for c in self.claims:
            if c.claim_id == claim_id:
                c.status = "refuted"
                if note:
                    c.notes = (c.notes + " | " if c.notes else "") + note
        if self.path:
            self._rewrite()

    def by_status(self, status: str) -> list[ClaimRecord]:
        return [c for c in self.claims if c.status == status]

    def summary(self) -> dict:
        out = {s: 0 for s in _STATUSES}
        for c in self.claims:
            out[c.status] = out.get(c.status, 0) + 1
        return {"total": len(self.claims), **out}

    # -- persistence (JSONL, event-sourced) --
    def _append(self, claim: ClaimRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(claim)) + "\n")

    def _rewrite(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as f:
            for c in self.claims:
                f.write(json.dumps(asdict(c)) + "\n")

    def _load(self) -> None:
        self.claims = [ClaimRecord(**json.loads(l)) for l in self.path.read_text().splitlines() if l.strip()]

    # -- YAML export (human-readable claims ledger) --
    def to_yaml(self, path: str | Path) -> None:
        import yaml
        Path(path).write_text(yaml.safe_dump([asdict(c) for c in self.claims], sort_keys=False))
