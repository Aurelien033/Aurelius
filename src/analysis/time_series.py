"""Time series analysis using pure stdlib math.

Inspired by timesfm (Google Time Series Forecasting Model).  Provides moving
averages, trend detection, seasonality estimation, and change-point detection
with zero third-party imports — ``statistics`` and ``math`` only.

Pattern:
    ``TimeSeriesAnalyzer.moving_average(data, window)``
    ``TimeSeriesAnalyzer.detect_trend(data)``
    ``TimeSeriesAnalyzer.detect_seasonality(data, max_lag)``
    ``TimeSeriesAnalyzer.detect_changepoints(data, threshold)``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median, stdev as pop_stdev
from typing import Any

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class MovingAverageResult:
    """Windowed moving average over a series."""

    values: list[float] = field(default_factory=list)
    window: int = 0
    original_length: int = 0

    @property
    def padded(self) -> list[float | None]:
        """Return list aligned with *original* — ``None`` where the window was short.

        ``values[i]`` = ``mean(original[i : i + window])``.
        The value is placed at ``i + window // 2`` (pandas convention).
        """
        half = self.window // 2
        result: list[float | None] = [None] * self.original_length
        for i, v in enumerate(self.values):
            pos = i + half
            if pos < self.original_length:
                result[pos] = v
        return result


@dataclass
class TrendResult:
    """Linear trend fit to a series."""

    slope: float = 0.0
    intercept: float = 0.0
    r_squared: float = 0.0
    direction: str = "flat"  # "up" / "down" / "flat"
    strength: float = 0.0  # 0..1, |slope| / mean(|y|)
    data: list[float] = field(default_factory=list)


@dataclass
class SeasonalityResult:
    """Dominant seasonal period detected in a series."""

    period: int = 0  # dominant lag in samples
    period_strength: float = 0.0  # how clearly periodic
    acf_peaks: list[int] = field(default_factory=list)  # all significant lags


@dataclass
class ChangepointResult:
    """Single detected change-point."""

    index: int = 0
    before_mean: float = 0.0
    after_mean: float = 0.0
    delta: float = 0.0


@dataclass
class SeriesSummary:
    """Overall statistical summary of a time series."""

    count: int = 0
    minimum: float = 0.0
    maximum: float = 0.0
    mean_val: float = 0.0
    median_val: float = 0.0
    std_dev: float = 0.0
    q1: float = 0.0
    q3: float = 0.0
    iqr: float = 0.0


# ---------------------------------------------------------------------------
# Core analyser
# ---------------------------------------------------------------------------


class TimeSeriesAnalyzer:
    """Pure-stdlib time series toolkit.  Operates on ``list[float]``."""

    # -- helpers ----------------------------------------------------------------

    @staticmethod
    def _clean(data: list[float]) -> list[float]:
        return [float(x) for x in data if x is not None and x == x]  # NaN check

    @staticmethod
    def _quantile(sorted_vals: list[float], q: float) -> float:
        if not sorted_vals:
            return 0.0
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        pos = q * (len(sorted_vals) - 1)
        lo = int(pos)
        frac = pos - lo
        hi = min(lo + 1, len(sorted_vals) - 1)
        return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])

    @staticmethod
    def _acf(values: list[float], lag: int) -> float:
        """Autocorrelation at *lag* (returns 0.0 if not enough data)."""
        n = len(values)
        if lag <= 0 or lag >= n:
            return 0.0
        m = mean(values)
        var = sum((x - m) ** 2 for x in values) / n
        if var == 0.0:
            return 0.0
        cov = sum(
            (values[i] - m) * (values[i + lag] - m) for i in range(n - lag)
        ) / n
        return cov / var

    # -- public API -------------------------------------------------------------

    def moving_average(
        self,
        data: list[float],
        window: int = 3,
    ) -> MovingAverageResult:
        """Compute the simple moving average over *data* with the given *window*.

        Returns a :class:`MovingAverageResult` that exposes ``.values`` and
        ``.padded`` (position-aligned with the original series).
        """
        clean = self._clean(data)
        if window > len(clean):
            raise ValueError(
                f"window ({window}) larger than series length ({len(clean)})"
            )
        avgs = [mean(clean[i: i + window]) for i in range(len(clean) - window + 1)]
        return MovingAverageResult(
            values=avgs, window=window, original_length=len(clean)
        )

    def detect_trend(self, data: list[float]) -> TrendResult:
        """Ordinary least-squares linear trend."""
        clean = self._clean(data)
        n = len(clean)
        if n < 2:
            return TrendResult(
                slope=0.0,
                intercept=clean[0] if clean else 0.0,
                r_squared=0.0,
                direction="flat",
                strength=0.0,
                data=clean,
            )
        x_mean = mean(range(n))
        y_mean = mean(clean)
        numer = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(clean))
        denom = sum((i - x_mean) ** 2 for i in range(n))
        slope = numer / denom if denom else 0.0
        intercept = y_mean - slope * x_mean
        ss_res = sum((clean[i] - slope * i - intercept) ** 2 for i in range(n))
        ss_tot = sum((y - y_mean) ** 2 for y in clean)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot else 0.0
        avg_abs_y = mean(abs(y) for y in clean) if clean else 0.0
        strength = abs(slope) / avg_abs_y if avg_abs_y else 0.0
        direction = "up" if slope > 1e-9 else ("down" if slope < -1e-9 else "flat")
        return TrendResult(
            slope=round(slope, 6),
            intercept=round(intercept, 6),
            r_squared=round(r_squared, 6),
            direction=direction,
            strength=round(strength, 6),
            data=clean,
        )

    def detect_seasonality(
        self,
        data: list[float],
        max_lag: int = 60,
    ) -> SeasonalityResult | None:
        """Detect the fundamental seasonality period via ACF peak analysis."""
        clean = self._clean(data)
        n = len(clean)
        if n < 4:
            return SeasonalityResult()
            

        peaks_raw: list[tuple[float, int]] = []
        for lag in range(2, min(max_lag, n // 2) + 1):
            acf_val = abs(self._acf(clean, lag))
            if acf_val > 0.3:
                peaks_raw.append((acf_val, lag))

        if not peaks_raw:
            return SeasonalityResult()

        # Use the shortest candidate per ACF peak quartile as the base period.
        # For a pure sine, ACF hits repeat at all half-period multiples;
        # doubling the shortest candidate avoids returning exactly P/2.
        acf_sorted: list[tuple[float, int]] = sorted(peaks_raw, reverse=True)
        top_quartile_size = max(1, len(acf_sorted) // 4)
        top_quartile: list[int] = [lag for _, lag in acf_sorted[:top_quartile_size]]
        base_candidate = min(top_quartile)

        # If the double of base_candidate also appears, prefer the double.
        top_set: set[int] = {lag for _, lag in peaks_raw}
        if base_candidate * 2 in top_set:
            dominant = base_candidate * 2
        else:
            dominant = base_candidate

        period_strength = round(
            sum(1.0 for _, l in peaks_raw if abs(l - dominant) <= base_candidate)
            / len(peaks_raw),
            4,
        )

        return SeasonalityResult(
            period=dominant,
            period_strength=period_strength,
            acf_peaks=sorted(lag for _, lag in peaks_raw),
        )
    def detect_changepoints(
        self,
        data: list[float],
        threshold: float = 2.0,
    ) -> list[ChangepointResult]:
        """Detect points where the local mean shifts by z_threshold σ."""
        clean = self._clean(data)
        n = len(clean)
        if n < 4:
            return []
        cps: list[tuple[float, int]] = []
        for i in range(1, n - 1):
            before = clean[:i]
            after = clean[i:]
            b_mean = mean(before)
            a_mean = mean(after)
            pooled_pop = pop_stdev(before) if len(before) > 1 else 0.0
            pooled_a = pop_stdev(after) if len(after) > 1 else 0.0
            pooled_std = (pooled_pop + pooled_a) / 2.0
            if pooled_std == 0.0:
                continue
            delta = (a_mean - b_mean) / pooled_std
            if abs(delta) >= threshold:
                cps.append(
                    ChangepointResult(
                        index=i,
                        before_mean=round(b_mean, 4),
                        after_mean=round(a_mean, 4),
                        delta=round(delta, 4),
                    )
                )
        return cps

    def summary(self, data: list[float]) -> SeriesSummary:
        """Five-number summary + mean + stdev for *data*."""
        clean = self._clean(data)
        if not clean:
            return SeriesSummary()
        sorted_v = sorted(clean)
        n = len(sorted_v)
        return SeriesSummary(
            count=n,
            minimum=round(sorted_v[0], 4),
            maximum=round(sorted_v[-1], 4),
            mean_val=round(mean(clean), 4),
            median_val=round(median(clean), 4),
            std_dev=round(pop_stdev(clean) if n > 1 else 0.0, 4),
            q1=round(self._quantile(sorted_v, 0.25), 4),
            q3=round(self._quantile(sorted_v, 0.75), 4),
            iqr=round(
                self._quantile(sorted_v, 0.75) - self._quantile(sorted_v, 0.25), 4
            ),
        )


DEFAULT_ANALYZER = TimeSeriesAnalyzer()
