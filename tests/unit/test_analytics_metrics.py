"""Unit tests for alphaforge.analytics.metrics (execDesign.md §10.1).

Covers: hand-computed Sharpe/Sortino/CAGR/Calmar/MaxDD on tiny fixtures to
1e-12 (constant returns, single-dip drawdown with known depth and dates),
the UTC-midnight daily aggregation boundary (the headline basis per
leakageCritique.md finding 29), turnover bucketing arithmetic, exposure
fractions, win metrics, CAGR vs exact compounding, summarize field wiring,
and input-shape validation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from alphaforge.analytics import (
    DAYS_PER_YEAR,
    HOURS_PER_YEAR,
    MS_PER_YEAR,
    cagr,
    calmar,
    daily_equity,
    daily_returns,
    drawdown_series,
    exposure_series,
    max_drawdown,
    sharpe,
    sortino,
    summarize,
    to_returns,
    turnover,
    win_metrics,
)

HOUR = 3_600_000
DAY = 24 * HOUR
T0 = 1_704_067_200_000  # 2024-01-01T00:00:00Z


def eq(vals: list[float], start: int = T0, step: int = HOUR) -> pd.Series:
    idx = pd.Index([start + i * step for i in range(len(vals))], name="ts")
    return pd.Series([float(v) for v in vals], index=idx, dtype=float)


class TestToReturns:
    def test_hand_computed_values_and_index(self) -> None:
        e = eq([100.0, 110.0, 99.0])
        r = to_returns(e)
        assert list(r.index) == [T0 + HOUR, T0 + 2 * HOUR]
        assert r.iloc[0] == pytest.approx(0.1, abs=1e-12)
        assert r.iloc[1] == pytest.approx(99.0 / 110.0 - 1.0, abs=1e-12)

    def test_validation_rejects_bad_input(self) -> None:
        with pytest.raises(ValueError, match=">= 2 points"):
            to_returns(eq([100.0]))
        bad_idx = pd.Series([100.0, 110.0], index=pd.Index([T0 + HOUR, T0]))
        with pytest.raises(ValueError, match="strictly increasing"):
            to_returns(bad_idx)
        with pytest.raises(ValueError, match="finite and > 0"):
            to_returns(eq([100.0, -5.0]))
        with pytest.raises(ValueError, match="finite and > 0"):
            to_returns(eq([100.0, math.nan]))


class TestDailyAggregation:
    def test_utc_midnight_boundary_last_of_day(self) -> None:
        # Points at 22:00, 23:00 of day 1 and 00:00, 01:00 of day 2.
        # A point exactly AT midnight belongs to the NEW day (bar-close stamping),
        # so day1 last = 110 (23:00) and day2 last = 130 (01:00).
        start = T0 + 22 * HOUR
        e = eq([100.0, 110.0, 120.0, 130.0], start=start)
        de = daily_equity(e)
        assert list(de.index) == [T0, T0 + DAY]  # day-start labels
        assert de.iloc[0] == pytest.approx(110.0, abs=0.0)
        assert de.iloc[1] == pytest.approx(130.0, abs=0.0)
        dr = daily_returns(e)
        assert list(dr.index) == [T0 + DAY]
        assert dr.iloc[0] == pytest.approx(130.0 / 110.0 - 1.0, abs=1e-12)

    def test_missing_days_dropped_not_filled(self) -> None:
        e = pd.Series(
            [100.0, 105.0, 120.0],
            index=pd.Index([T0 + HOUR, T0 + 2 * HOUR, T0 + 3 * DAY], name="ts"),
        )
        de = daily_equity(e)
        assert list(de.index) == [T0, T0 + 3 * DAY]  # day 2/3 absent, not padded

    def test_single_day_yields_empty_returns(self) -> None:
        dr = daily_returns(eq([100.0, 101.0, 102.0]))
        assert len(dr) == 0


class TestSharpe:
    def test_hand_computed_to_1e12(self) -> None:
        r = np.array([0.01, -0.02, 0.03])
        m = (0.01 - 0.02 + 0.03) / 3.0
        var = ((0.01 - m) ** 2 + (-0.02 - m) ** 2 + (0.03 - m) ** 2) / 2.0  # ddof=1
        expected = m / math.sqrt(var) * math.sqrt(HOURS_PER_YEAR)
        assert sharpe(r, HOURS_PER_YEAR) == pytest.approx(expected, abs=1e-12)

    def test_accepts_series(self) -> None:
        r = to_returns(eq([100.0, 101.0, 99.0, 103.0]))
        assert sharpe(r, DAYS_PER_YEAR) == pytest.approx(
            sharpe(r.to_numpy(), DAYS_PER_YEAR), abs=0.0
        )

    def test_constant_returns_zero_std_is_nan(self) -> None:
        assert math.isnan(sharpe(np.array([0.01, 0.01, 0.01]), HOURS_PER_YEAR))

    def test_too_few_observations_is_nan(self) -> None:
        assert math.isnan(sharpe(np.array([0.01]), HOURS_PER_YEAR))
        assert math.isnan(sharpe(np.array([], dtype=float), HOURS_PER_YEAR))


class TestSortino:
    def test_hand_computed_full_t_denominator(self) -> None:
        r = np.array([0.02, -0.01, 0.03, -0.02])
        m = (0.02 - 0.01 + 0.03 - 0.02) / 4.0
        sigma_down = math.sqrt((0.01**2 + 0.02**2) / 4.0)  # full-T, target 0
        expected = m / sigma_down * math.sqrt(HOURS_PER_YEAR)
        assert sortino(r, HOURS_PER_YEAR) == pytest.approx(expected, abs=1e-12)

    def test_no_downside_is_nan(self) -> None:
        assert math.isnan(sortino(np.array([0.01, 0.02, 0.0]), HOURS_PER_YEAR))
        assert math.isnan(sortino(np.array([], dtype=float), HOURS_PER_YEAR))


class TestCagr:
    def test_single_period_hand_value(self) -> None:
        # (121/100)^(2/1) - 1 = 1.21^2 - 1 = 0.4641
        assert cagr(eq([100.0, 121.0]), 2.0) == pytest.approx(0.4641, abs=1e-12)

    def test_matches_exact_compounding(self) -> None:
        g = 0.01
        vals = [100.0 * (1.0 + g) ** k for k in range(101)]  # T = 100 periods
        # A = 100 -> CAGR must equal the exact 100-period compound growth.
        assert cagr(eq(vals), 100.0) == pytest.approx((1.0 + g) ** 100 - 1.0, rel=1e-12)

    def test_negative_growth(self) -> None:
        assert cagr(eq([100.0, 81.0]), 2.0) == pytest.approx(0.81**2 - 1.0, abs=1e-12)


class TestMaxDrawdown:
    def test_single_dip_known_depth_and_dates(self) -> None:
        e = eq([100.0, 110.0, 99.0, 104.5, 121.0])
        depth, peak_ts, trough_ts = max_drawdown(e)
        assert depth == pytest.approx(1.0 - 99.0 / 110.0, abs=1e-12)  # 10%
        assert peak_ts == T0 + HOUR
        assert trough_ts == T0 + 2 * HOUR

    def test_monotone_curve_zero_dd(self) -> None:
        depth, peak_ts, trough_ts = max_drawdown(eq([100.0, 101.0, 102.0]))
        assert depth == 0.0
        assert peak_ts == T0
        assert trough_ts == T0

    def test_drawdown_series_values(self) -> None:
        dd = drawdown_series(eq([100.0, 110.0, 99.0, 104.5, 121.0]))
        expected = [0.0, 0.0, 1.0 - 99.0 / 110.0, 1.0 - 104.5 / 110.0, 0.0]
        np.testing.assert_allclose(dd.to_numpy(), expected, atol=1e-12)
        assert list(dd.index) == [T0 + i * HOUR for i in range(5)]


class TestCalmar:
    def test_cagr_over_abs_max_dd(self) -> None:
        e = eq([100.0, 110.0, 99.0, 104.5, 121.0])
        expected = cagr(e, HOURS_PER_YEAR) / (1.0 - 99.0 / 110.0)
        assert calmar(e, HOURS_PER_YEAR) == pytest.approx(expected, abs=1e-12)

    def test_zero_dd_is_nan(self) -> None:
        assert math.isnan(calmar(eq([100.0, 101.0]), HOURS_PER_YEAR))


class TestTurnover:
    def test_bucketing_and_annualization(self) -> None:
        e = eq([100.0, 100.0, 100.0, 100.0])
        fills = pd.Series(
            [50.0, 25.0, 25.0],
            index=pd.Index([T0 + HOUR, T0 + HOUR + 1, T0 + 2 * HOUR], name="ts"),
        )
        # Bucket sums: ts1 -> 50 (exact match), ts2 -> 25 (between stamps, rolls
        # forward to next close) + 25 (exact) = 50. tau = [0, .5, .5, 0], T = 4.
        expected = (0.5 + 0.5) / 4.0 * HOURS_PER_YEAR
        assert turnover(fills, e, HOURS_PER_YEAR) == pytest.approx(expected, abs=1e-9)

    def test_fill_after_last_equity_point_ignored(self) -> None:
        e = eq([100.0, 100.0])
        fills = pd.Series([999.0], index=pd.Index([T0 + 5 * HOUR], name="ts"))
        assert turnover(fills, e, HOURS_PER_YEAR) == 0.0

    def test_empty_fills_zero(self) -> None:
        fills = pd.Series([], index=pd.Index([], name="ts"), dtype=float)
        assert turnover(fills, eq([100.0, 100.0]), HOURS_PER_YEAR) == 0.0

    def test_negative_notional_rejected(self) -> None:
        fills = pd.Series([-1.0], index=pd.Index([T0], name="ts"))
        with pytest.raises(ValueError, match="absolute notional"):
            turnover(fills, eq([100.0, 100.0]), HOURS_PER_YEAR)


class TestExposure:
    def test_gross_net_fractions(self) -> None:
        e = eq([100.0, 200.0])
        positions = pd.DataFrame(
            {
                "ts": [T0, T0, T0 + HOUR],
                "qty": [2.0, -1.0, 1.0],
                "mark": [30.0, 40.0, 50.0],
            }
        )
        out = exposure_series(positions, e)
        # ts0: notionals +60, -40 -> gross 100/100 = 1.0, net 20/100 = 0.2
        assert out.loc[T0, "gross"] == pytest.approx(1.0, abs=1e-12)
        assert out.loc[T0, "net"] == pytest.approx(0.2, abs=1e-12)
        # ts1: +50 on equity 200 -> 0.25 / 0.25
        assert out.loc[T0 + HOUR, "gross"] == pytest.approx(0.25, abs=1e-12)
        assert out.loc[T0 + HOUR, "net"] == pytest.approx(0.25, abs=1e-12)

    def test_flat_timestamps_zero_and_unknown_ts_rejected(self) -> None:
        e = eq([100.0, 100.0])
        empty = pd.DataFrame({"ts": [], "qty": [], "mark": []})
        out = exposure_series(empty, e)
        assert (out["gross"] == 0.0).all()
        assert (out["net"] == 0.0).all()
        rogue = pd.DataFrame({"ts": [T0 + 7 * HOUR], "qty": [1.0], "mark": [10.0]})
        with pytest.raises(ValueError, match="not on the equity index"):
            exposure_series(rogue, e)


class TestWinMetrics:
    def test_hand_computed(self) -> None:
        wm = win_metrics(np.array([0.02, -0.01, 0.01, -0.02, 0.0]))
        assert wm.win_rate == pytest.approx(2.0 / 5.0, abs=1e-12)  # zero is not a win
        assert wm.avg_win == pytest.approx(0.015, abs=1e-12)
        assert wm.avg_loss == pytest.approx(-0.015, abs=1e-12)
        assert wm.profit_factor == pytest.approx(1.0, abs=1e-12)

    def test_degenerate_cases(self) -> None:
        all_up = win_metrics(np.array([0.01, 0.02]))
        assert math.isnan(all_up.avg_loss)
        assert math.isnan(all_up.profit_factor)  # no losses -> undefined, not inf
        empty = win_metrics(np.array([], dtype=float))
        assert math.isnan(empty.win_rate)


class TestSummarize:
    @staticmethod
    def hourly_equity(n_days: int) -> pd.Series:
        # Deterministic (seed-free), strictly positive, alternating up/down DAYS
        # (sin period = 48 bars = 2 days) so daily win/loss metrics are defined.
        n = n_days * 24
        rets = 0.0001 + 0.002 * np.sin(2.0 * np.pi * np.arange(n) / 48.0)
        vals = 100_000.0 * np.cumprod(1.0 + rets)
        return eq([100_000.0, *vals.tolist()])

    def test_headline_is_daily_based(self) -> None:
        e = self.hourly_equity(5)
        s = summarize(e)
        d_rets = daily_returns(e)
        d_eq = daily_equity(e)
        assert s.sharpe == pytest.approx(sharpe(d_rets, DAYS_PER_YEAR), abs=1e-12)
        assert s.sortino == pytest.approx(sortino(d_rets, DAYS_PER_YEAR), abs=1e-12)
        # Headline CAGR is time-anchored at the run start (full elapsed wall
        # time): the daily-resampled curve's first point is already end-of-
        # day-1, so a resampled CAGR would silently drop the partial first
        # day's PnL and can even flip sign versus total_return.
        elapsed_ms = float(e.index[-1] - e.index[0])
        expected_cagr = (float(e.iloc[-1]) / float(e.iloc[0])) ** (MS_PER_YEAR / elapsed_ms) - 1.0
        assert s.cagr == pytest.approx(expected_cagr, abs=1e-12)
        assert s.max_dd == pytest.approx(max_drawdown(d_eq)[0], abs=1e-12)
        assert s.max_dd_bar == pytest.approx(max_drawdown(e)[0], abs=1e-12)
        assert s.max_dd_bar >= s.max_dd  # full grid can only be as deep or deeper
        assert s.n_days == len(d_eq)
        assert s.n_periods == len(e) - 1
        assert s.start_ts == int(e.index[0])
        assert s.end_ts == int(e.index[-1])
        assert s.total_return == pytest.approx(
            float(e.iloc[-1]) / float(e.iloc[0]) - 1.0, abs=1e-12
        )

    def test_hourly_diagnostics_use_bar_annualization(self) -> None:
        e = self.hourly_equity(3)
        s = summarize(e)
        r = to_returns(e)
        assert s.sharpe_hourly == pytest.approx(sharpe(r, HOURS_PER_YEAR), abs=1e-12)
        assert s.vol_ann_hourly == pytest.approx(
            float(np.std(r.to_numpy(), ddof=1)) * math.sqrt(HOURS_PER_YEAR), abs=1e-12
        )

    def test_optional_inputs_missing_are_nan_not_zero(self) -> None:
        s = summarize(self.hourly_equity(2))
        assert math.isnan(s.turnover_ann)
        assert math.isnan(s.fees_paid)
        assert math.isnan(s.funding_net)
        assert math.isnan(s.gross_mean)
        assert math.isnan(s.net_mean)

    def test_fills_funding_positions_wiring(self) -> None:
        e = eq([100.0, 100.0, 100.0, 100.0])
        fills = pd.DataFrame(
            {"ts": [T0 + HOUR, T0 + 2 * HOUR], "notional": [50.0, 50.0], "fee": [0.2, 0.3]}
        )
        funding = pd.Series([1.5, -0.5], index=pd.Index([T0 + HOUR, T0 + 3 * HOUR], name="ts"))
        positions = pd.DataFrame({"ts": [T0 + HOUR], "qty": [1.0], "mark": [50.0]})
        s = summarize(e, fills=fills, funding=funding, positions=positions)
        assert s.turnover_ann == pytest.approx((1.0 / 4.0) * HOURS_PER_YEAR, abs=1e-9)
        assert s.fees_paid == pytest.approx(0.5, abs=1e-12)
        assert s.funding_net == pytest.approx(1.0, abs=1e-12)
        assert s.gross_mean == pytest.approx(0.5 / 4.0, abs=1e-12)  # one of four stamps at 0.5
        assert s.net_mean == pytest.approx(0.5 / 4.0, abs=1e-12)

    def test_fills_without_fee_column_yields_nan_fees(self) -> None:
        e = eq([100.0, 100.0])
        fills = pd.DataFrame({"ts": [T0 + HOUR], "notional": [10.0]})
        assert math.isnan(summarize(e, fills=fills).fees_paid)

    def test_fills_missing_required_columns_rejected(self) -> None:
        e = eq([100.0, 100.0])
        with pytest.raises(ValueError, match="notional"):
            summarize(e, fills=pd.DataFrame({"ts": [T0]}))
