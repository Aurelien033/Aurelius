"""Unit tests for src.analysis.time_series.

Covers the four primary analyses, error paths, edge values, and the
stdlib-only import constraint (statistics + math → no numpy).
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from src.analysis.time_series import (
    ChangepointResult,
    DEFAULT_ANALYZER,
    MovingAverageResult,
    SeasonalityResult,
    SeriesSummary,
    TimeSeriesAnalyzer,
    TrendResult,
)

# stdlib imports used by this module
_STDLIB = {"statistics", "math", "dataclasses", "typing", '__future__'}

_SRC = pathlib.Path(__file__).parents[2] / "src" / "analysis" / "time_series.py"


def test_no_third_party_imports() -> None:
    tree = ast.parse(_SRC.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported - _STDLIB, f"Non-stdlib: {imported - _STDLIB}"


# ── helpers used internally ──────────────────────────────────────────────────

class TestHelpers:
    def setup_method(self):
        self.ta = TimeSeriesAnalyzer()

    def test_clean_removes_none(self):
        assert self.ta._clean([1.0, None, 2.0]) == [1.0, 2.0]

    def test_clean_removes_nan(self):
        assert self.ta._clean([1.0, float("nan"), 3.0]) == [1.0, 3.0]

    def test_quantile_single(self):
        assert self.ta._quantile([5.0], 0.0) == 5.0

    def test_quantile_middle(self):
        assert self.ta._quantile([1, 2, 3], 0.5) == 2.0

    def test_quantile_fractional(self):
        r = self.ta._quantile([1, 2], 0.5)
        assert 1.5 <= r <= 2.5

    def test_acf_at_zero(self):
        v = self.ta._acf([1, 2, 3, 4, 5], 0)
        assert v == 0.0

    def test_acf_not_enough_data(self):
        assert self.ta._acf([1], 5) == 0.0


# ── moving_average ───────────────────────────────────────────────────────────

class TestMovingAverage:
    def setup_method(self):
        self.ta = TimeSeriesAnalyzer()

    def test_basic(self):
        r = self.ta.moving_average([1, 2, 3, 4, 5], window=3)
        assert r.window == 3
        assert r.original_length == 5
        assert abs(r.values[0] - 2.0) < 1e-9
        assert abs(r.values[-1] - 4.0) < 1e-9

    def test_window_one_identity(self):
        r = self.ta.moving_average([10, 20, 30], window=1)
        assert r.values == [10.0, 20.0, 30.0]

    def test_window_equals_length(self):
        r = self.ta.moving_average([5, 10, 15], window=3)
        assert len(r.values) == 1
        assert abs(r.values[0] - 10.0) < 1e-9

    def test_window_too_large_raises(self):
        with pytest.raises(ValueError):
            self.ta.moving_average([1, 2, 3], window=5)

    def test_window_zero_raises(self):
        with pytest.raises(ValueError):
            self.ta.moving_average([1, 2, 3], window=0)

    def test_empty_series_raises(self):
        with pytest.raises(ValueError):
            self.ta.moving_average([], window=3)

    def test_padded_alignment(self):
        r = self.ta.moving_average([1, 2, 3, 4, 5], window=3)
        padded = r.padded
        assert padded[0] is None
        assert padded[1] is not None
        assert padded[2] is not None
        assert padded[3] is not None
        assert padded[4] is None

    def test_decimal_values(self):
        r = self.ta.moving_average([0.1, 0.2, 0.3, 0.4], window=2)
        assert abs(r.values[0] - 0.15) < 1e-9


# ── trend ────────────────────────────────────────────────────────────────────

class TestTrend:
    def setup_method(self):
        self.ta = TimeSeriesAnalyzer()

    def test_perfect_up(self):
        r = self.ta.detect_trend([1, 2, 3, 4, 5])
        assert r.slope > 0
        assert r.direction == "up"
        assert r.r_squared == pytest.approx(1.0, abs=1e-9)

    def test_perfect_down(self):
        r = self.ta.detect_trend([10, 8, 6, 4, 2])
        assert r.slope < 0
        assert r.direction == "down"
        assert r.r_squared == pytest.approx(1.0, abs=1e-9)

    def test_flat(self):
        r = self.ta.detect_trend([5, 5, 5, 5])
        assert r.slope == 0.0
        assert r.direction == "flat"
        assert r.r_squared == 0.0

    def test_intercept(self):
        r = self.ta.detect_trend([0, 5, 10, 15])
        assert abs(r.intercept) < 0.5  # near zero

    def test_single_point(self):
        r = self.ta.detect_trend([42.0])
        assert r.slope == 0.0
        assert r.direction == "flat"
        assert r.r_squared == 0.0

    def test_empty_series(self):
        r = self.ta.detect_trend([])
        assert r.slope == 0.0
        assert r.direction == "flat"
        assert r.data == []

    def test_strength_values_preserved(self):
        r = self.ta.detect_trend([1, 2, 3])
        assert 0.0 <= r.strength <= 2.0

    def test_data_field_populated(self):
        r = self.ta.detect_trend([3, 1, 4, 1, 5])
        assert r.data == [3.0, 1.0, 4.0, 1.0, 5.0]


# ── seasonality ──────────────────────────────────────────────────────────────

class TestSeasonality:
    def setup_method(self):
        self.ta = TimeSeriesAnalyzer()

    def _sine(self, period: int, n: int) -> list[float]:
        import math
        return [math.sin(2 * math.pi * i / period) for i in range(n)]

    def test_strong_periodic_signal(self):
        s = self.ta.detect_seasonality(self._sine(4, 200), max_lag=20)
        assert s.period == 4

    def test_no_seasonality(self):
        import random
        random.seed(42)
        flat = [random.gauss(0, 1) for _ in range(100)]
        s = self.ta.detect_seasonality(flat, max_lag=20)
        assert s.period == 0
        assert s.period_strength == 0.0

    def test_short_series(self):
        s = self.ta.detect_seasonality([1, 2, 3, 4], max_lag=10)
        assert s.period == 0
        assert s.acf_peaks == []

    def test_too_short(self):
        s = self.ta.detect_seasonality([])
        assert s.period == 0

    def test_acf_peaks_populated(self):
        s = self.ta.detect_seasonality(self._sine(7, 500), max_lag=30)
        assert len(s.acf_peaks) > 0


# ── changepoints ─────────────────────────────────────────────────────────────

class TestChangepoints:
    def setup_method(self):
        self.ta = TimeSeriesAnalyzer()

    def _step_series(self, step_at: int, before: float, after: float, n: int) -> list[float]:
        return [before] * step_at + [after] * (n - step_at)

    def test_clear_step_detected(self):
        data = self._step_series(50, 10, 90, 100)
        cps = self.ta.detect_changepoints(data, threshold=3.0)
        assert len(cps) >= 1
        # Nearest the step
        assert any(48 <= cp.index <= 52 for cp in cps)

    def test_no_constant_series(self):
        cps = self.ta.detect_changepoints([5] * 30, threshold=2.0)
        assert cps == []

    def test_delta_direction(self):
        before, after = 5.0, 15.0
        data = self._step_series(50, before, after, 100)
        # The step should trigger a positive delta
        cp = next(c for c in self.ta.detect_changepoints(data, threshold=2.0))
        assert cp.after_mean > cp.before_mean

    def test_threshold_controls_sensitivity(self):
        data = self._step_series(50, 10, 20, 100)
        strict = self.ta.detect_changepoints(data, threshold=5.0)
        loose = self.ta.detect_changepoints(data, threshold=1.0)
        assert len(strict) <= len(loose)

    def test_short_series(self):
        assert self.ta.detect_changepoints([1, 2, 3], threshold=2.0) == []


# ── summary ──────────────────────────────────────────────────────────────────

class TestSummary:
    def setup_method(self):
        self.ta = TimeSeriesAnalyzer()

    def test_basic(self):
        s = self.ta.summary([1, 2, 3, 4, 5])
        assert s.count == 5
        assert s.minimum == 1.0
        assert s.maximum == 5.0
        assert s.mean_val == 3.0
        assert s.q1 == 2.0
        assert s.q3 == 4.0

    def test_empty(self):
        s = self.ta.summary([])
        assert s.count == 0
        assert s.mean_val == 0.0

    def test_single_value(self):
        s = self.ta.summary([42.0])
        assert s.count == 1
        assert s.minimum == 42.0
        assert s.maximum == 42.0
        assert s.mean_val == 42.0

    def test_iqr_calculation(self):
        s = self.ta.summary(list(range(1, 21)))
        assert abs(s.iqr - 9.5) < 0.5


# ── singleton ────────────────────────────────────────────────────────────────

def test_default_analyzer(self=None):
    assert isinstance(DEFAULT_ANALYZER, TimeSeriesAnalyzer)
