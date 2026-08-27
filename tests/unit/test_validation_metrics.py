"""Unit tests for alphaforge.validation.metrics (alphaDesign.md §7.2).

Covers: hand-computed Spearman rank IC fixtures (including average tie ranks),
universe masking (a planted extreme non-member cannot move the members' IC),
the min_members NaN rule, the Newey-West HAC t-stat against a fully hand-computed
2-lag example to 1e-12 (plus the lags=0 plain-t reduction), ic_summary fields and
per-year bucketing, and non-overlapping grid subsampling.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from alphaforge.validation import (
    ic_summary,
    newey_west_tstat,
    non_overlapping,
    rank_ic,
)

HOUR = 3_600_000
T0 = 1_704_153_600_000  # 2024-01-02T00:00:00Z
T1 = T0 + HOUR

TS_2021 = 1_609_459_200_000  # 2021-01-01T00:00:00Z
TS_2022 = 1_640_995_200_000  # 2022-01-01T00:00:00Z


def panel(rows: dict[tuple[int, str], float]) -> pd.Series:
    index = pd.MultiIndex.from_tuples(list(rows), names=["ts_open", "instrument_id"])
    return pd.Series(list(rows.values()), index=index, dtype=float)


def full_mask(s: pd.Series) -> pd.Series:
    return pd.Series(True, index=s.index)


class TestRankIc:
    def test_hand_computed_spearman_with_ties(self) -> None:
        # factor [1, 2, 2, 4] → ranks [1, 2.5, 2.5, 4]; fwd [10, 30, 20, 40] → [1, 3, 2, 4]
        # Pearson on ranks: Σdxdy = 4.5, Σdx² = 4.5, Σdy² = 5 → 4.5/√22.5
        factor = panel({(T0, f"I{i}"): v for i, v in enumerate([1.0, 2.0, 2.0, 4.0])})
        fwd = panel({(T0, f"I{i}"): v for i, v in enumerate([10.0, 30.0, 20.0, 40.0])})
        out = rank_ic(factor, fwd, full_mask(factor), min_members=4)
        assert out.name == "rank_ic"
        assert out.loc[T0] == pytest.approx(4.5 / math.sqrt(22.5), abs=1e-12)

    def test_perfect_monotone_and_inverted(self) -> None:
        factor = panel({(T0, f"I{i}"): float(i) for i in range(6)})
        fwd_up = panel({(T0, f"I{i}"): math.exp(i) for i in range(6)})  # monotone ↑
        fwd_dn = panel({(T0, f"I{i}"): -math.exp(i) for i in range(6)})  # monotone ↓
        assert rank_ic(factor, fwd_up, full_mask(factor)).loc[T0] == pytest.approx(1.0)
        assert rank_ic(factor, fwd_dn, full_mask(factor)).loc[T0] == pytest.approx(-1.0)

    def test_non_member_outlier_cannot_move_members_ic(self) -> None:
        members = {(T0, f"M{i}"): float(i) for i in range(5)}
        fwd_members = {(T0, f"M{i}"): float([3, 1, 4, 0, 2][i]) for i in range(5)}
        factor = panel({**members, (T0, "X"): 1e9})
        fwd = panel({**fwd_members, (T0, "X"): -1e9})
        mask = pd.Series([iid.startswith("M") for _, iid in factor.index], index=factor.index)
        with_x = rank_ic(factor, fwd, mask)
        without_x = rank_ic(panel(members), panel(fwd_members), full_mask(panel(members)))
        assert with_x.loc[T0] == pytest.approx(without_x.loc[T0], abs=1e-12)

    def test_min_members_emits_nan_not_a_value(self) -> None:
        factor = panel({(T0, f"I{i}"): float(i) for i in range(3)})  # n=3 < 5
        fwd = panel({(T0, f"I{i}"): float(i) for i in range(3)})
        out = rank_ic(factor, fwd, full_mask(factor))
        assert T0 in out.index and math.isnan(out.loc[T0])

    def test_nan_fwd_rows_do_not_count_as_members(self) -> None:
        factor = panel({(T0, f"I{i}"): float(i) for i in range(5)})
        fwd_rows = {(T0, f"I{i}"): float(i) for i in range(5)}
        fwd_rows[(T0, "I4")] = np.nan  # 4 usable < 5
        out = rank_ic(factor, panel(fwd_rows), full_mask(factor))
        assert math.isnan(out.loc[T0])

    def test_rows_absent_from_mask_are_fail_closed(self) -> None:
        factor = panel({(T0, f"I{i}"): float(i) for i in range(5)})
        fwd = panel({(T0, f"I{i}"): float(i) for i in range(5)})
        partial_mask = pd.Series(True, index=factor.index[:4])  # I4 missing → non-member
        assert math.isnan(rank_ic(factor, fwd, partial_mask).loc[T0])

    def test_constant_factor_cross_section_is_nan(self) -> None:
        factor = panel({(T0, f"I{i}"): 1.0 for i in range(5)})
        fwd = panel({(T0, f"I{i}"): float(i) for i in range(5)})
        assert math.isnan(rank_ic(factor, fwd, full_mask(factor)).loc[T0])

    def test_one_entry_per_timestamp_sorted(self) -> None:
        rows = {(t, f"I{i}"): float(i * (1 if t == T0 else -1)) for t in (T1, T0) for i in range(5)}
        factor = panel(rows)
        fwd = panel({k: float(i) for i, k in enumerate(rows)})
        out = rank_ic(factor, fwd, full_mask(factor))
        assert list(out.index) == [T0, T1]


class TestNeweyWestTstat:
    def test_hand_computed_two_lag_example_to_1e12(self) -> None:
        # x = [1..5]: x̄=3, d=[-2,-1,0,1,2]
        # gamma0 = 10/5 = 2; gamma1 = (2+0+0+2)/5 = 0.8; gamma2 = (0-1+0)/5 = -0.2
        # w1 = 2/3, w2 = 1/3 → S = 2 + 2(2/3·0.8 + 1/3·(-0.2)) = 44/15
        # t = 3 / sqrt((44/15)/5)
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = 3.0 / math.sqrt((44.0 / 15.0) / 5.0)
        assert newey_west_tstat(x, 2) == pytest.approx(expected, abs=1e-12)

    def test_lags_zero_reduces_to_plain_tstat(self) -> None:
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = 3.0 / math.sqrt((10.0 / 5.0) / 5.0)  # mean / sqrt(gamma0/n)
        assert newey_west_tstat(x, 0) == pytest.approx(expected, abs=1e-12)

    def test_nans_are_dropped(self) -> None:
        x = pd.Series([1.0, np.nan, 2.0, 3.0, np.nan, 4.0, 5.0])
        assert newey_west_tstat(x, 2) == pytest.approx(
            newey_west_tstat(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), 2), abs=1e-15
        )

    def test_degenerate_inputs(self) -> None:
        assert math.isnan(newey_west_tstat(pd.Series([1.0]), 2))  # n < 2
        assert math.isnan(newey_west_tstat(pd.Series([2.0, 2.0, 2.0]), 1))  # zero variance
        with pytest.raises(ValueError, match="lags"):
            newey_west_tstat(pd.Series([1.0, 2.0]), -1)

    def test_lags_beyond_sample_are_capped(self) -> None:
        x = pd.Series([1.0, 2.0, 3.0])
        assert math.isfinite(newey_west_tstat(x, 100))


class TestIcSummary:
    def test_fields_and_by_year_bucketing(self) -> None:
        ic = pd.Series(
            [0.1, 0.2, -0.1, 0.3, np.nan],
            index=pd.Index(
                [TS_2021, TS_2021 + HOUR, TS_2022, TS_2022 + HOUR, TS_2022 + 2 * HOUR],
                name="ts_open",
            ),
        )
        summary = ic_summary(ic, lags=0)
        assert summary.n == 4  # NaN dropped
        assert summary.mean == pytest.approx(0.125, abs=1e-12)
        assert summary.std == pytest.approx(
            float(pd.Series([0.1, 0.2, -0.1, 0.3]).std(ddof=1)), abs=1e-12
        )
        assert summary.hit_rate == pytest.approx(0.75, abs=1e-12)
        assert summary.ic_ir == pytest.approx(summary.mean / summary.std, abs=1e-12)
        assert summary.t_nw == pytest.approx(
            newey_west_tstat(pd.Series([0.1, 0.2, -0.1, 0.3]), 0), abs=1e-12
        )
        assert summary.by_year == {
            2021: pytest.approx(0.15, abs=1e-12),
            2022: pytest.approx(0.1, abs=1e-12),
        }

    def test_empty_series(self) -> None:
        summary = ic_summary(pd.Series([np.nan], index=pd.Index([TS_2021])), lags=3)
        assert summary.n == 0
        assert math.isnan(summary.mean)
        assert summary.by_year == {}


class TestNonOverlapping:
    def test_keeps_every_hth_grid_point(self) -> None:
        ts = [T0 + i * HOUR for i in range(10)]
        s = pd.Series(range(10), index=pd.Index(ts, name="ts_open"), dtype=float)
        out = non_overlapping(s, 3)
        assert list(out.index) == [ts[0], ts[3], ts[6], ts[9]]
        assert out.tolist() == [0.0, 3.0, 6.0, 9.0]

    def test_multiindex_panel_keeps_all_instruments_at_kept_ts(self) -> None:
        rows = {(T0 + i * HOUR, iid): float(i) for i in range(6) for iid in ("A", "B")}
        s = panel(rows)
        out = non_overlapping(s, 2)
        kept_ts = sorted(set(out.index.get_level_values(0)))
        assert kept_ts == [T0, T0 + 2 * HOUR, T0 + 4 * HOUR]
        assert len(out) == 6  # both instruments survive at each kept ts

    def test_grid_is_unique_timestamps_not_row_positions(self) -> None:
        # duplicate timestamps across instruments must count once on the grid
        rows = {(T0 + i * HOUR, iid): 1.0 for i in range(4) for iid in ("A", "B", "C")}
        out = non_overlapping(panel(rows), 2)
        assert sorted(set(out.index.get_level_values(0))) == [T0, T0 + 2 * HOUR]

    def test_h_one_is_identity_and_bad_h_raises(self) -> None:
        s = pd.Series([1.0, 2.0], index=pd.Index([T0, T1]))
        pd.testing.assert_series_equal(non_overlapping(s, 1), s)
        with pytest.raises(ValueError, match="h_bars"):
            non_overlapping(s, 0)


# --------------------------------------------------------------------------- #
# Property tests (A5-1). These complement the hand-computed fixtures above:    #
# the invariants must hold across a wide input distribution, not just the      #
# planted examples. Examples are capped to keep the suite fast.                #
# --------------------------------------------------------------------------- #


def _distinct_floats(min_size: int, max_size: int) -> st.SearchStrategy[list[float]]:
    """Lists of pairwise-distinct finite floats (no ties → a non-degenerate ranking)."""
    return st.lists(
        st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        ),
        min_size=min_size,
        max_size=max_size,
        unique=True,
    )


class TestRankIcProperties:
    @given(
        factor_vals=_distinct_floats(5, 40),
        data=st.data(),
    )
    @settings(max_examples=80, deadline=None)
    def test_ic_is_bounded_in_unit_interval(
        self, factor_vals: list[float], data: st.DataObject
    ) -> None:
        # An arbitrary (possibly correlated, possibly not) cross-section at one ts.
        n = len(factor_vals)
        fwd_vals = data.draw(
            st.lists(
                st.floats(
                    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False, width=64
                ),
                min_size=n,
                max_size=n,
            )
        )
        factor = panel({(T0, f"I{i}"): v for i, v in enumerate(factor_vals)})
        fwd = panel({(T0, f"I{i}"): v for i, v in enumerate(fwd_vals)})
        ic = rank_ic(factor, fwd, full_mask(factor), min_members=2).loc[T0]
        if math.isnan(ic):  # degenerate (constant-rank) cross-section is allowed to be NaN
            return
        assert -1.0 - 1e-12 <= ic <= 1.0 + 1e-12

    @given(factor_vals=_distinct_floats(5, 40), increasing=st.booleans())
    @settings(max_examples=80, deadline=None)
    def test_strict_monotone_relationship_is_plus_or_minus_one(
        self, factor_vals: list[float], increasing: bool
    ) -> None:
        # fwd is a strictly monotone transform of factor: ranks coincide (↑) or
        # reverse (↓) exactly, so RankIC must hit the +1 / -1 boundary.
        order = np.argsort(np.asarray(factor_vals))
        ranks = np.empty(len(factor_vals), dtype=float)
        ranks[order] = np.arange(len(factor_vals), dtype=float)
        sign = 1.0 if increasing else -1.0
        fwd_vals = (sign * ranks).tolist()  # strictly monotone in the factor's order
        factor = panel({(T0, f"I{i}"): v for i, v in enumerate(factor_vals)})
        fwd = panel({(T0, f"I{i}"): v for i, v in enumerate(fwd_vals)})
        ic = rank_ic(factor, fwd, full_mask(factor), min_members=2).loc[T0]
        assert ic == pytest.approx(sign, abs=1e-12)


class TestNeweyWestTstatProperties:
    @given(
        vals=st.lists(
            st.floats(
                min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False, width=64
            ),
            min_size=2,
            max_size=60,
        ),
        lags=st.integers(min_value=0, max_value=20),
    )
    @settings(max_examples=120, deadline=None)
    def test_finite_and_sign_consistent_with_mean(self, vals: list[float], lags: int) -> None:
        x = pd.Series(vals)
        t = newey_west_tstat(x, lags)
        if math.isnan(t):  # tiny-sample truncated-Bartlett var ≤ 0 is allowed to be NaN
            return
        assert math.isfinite(t)
        mean = float(np.mean(vals))
        # t = mean / sqrt(positive var) ⇒ sign(t) == sign(mean) (both zero handled together).
        assert math.copysign(1.0, t) == math.copysign(1.0, mean) or mean == 0.0

    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
        n=st.integers(min_value=24, max_value=120),
        phi=st.floats(min_value=0.3, max_value=0.9, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=80, deadline=None)
    def test_magnitude_non_increasing_while_added_autocovariance_stays_positive(
        self, seed: int, n: int, phi: float
    ) -> None:
        # Variance-inflation guarantee (§7.2): S(L) = gamma_0 + 2·Σ_{l<=L} w_l·gamma_l with
        # Bartlett w_l = 1 - l/(L+1). Growing L from L to L+1 (a) raises w_l for every
        # already-included lag (the L+1 denominator grows) AND (b) adds a fresh
        # w_{L+1}·gamma_{L+1} term. When the relevant sample autocovariances are POSITIVE
        # both effects strictly inflate S, so |t_NW| is non-increasing. We build a
        # persistent AR(1)-with-drift series (positive serial correlation by construction)
        # and assert the monotone step only across the contiguous lag prefix whose sample
        # autocovariance is positive — the exact regime where the theorem applies.
        rng = np.random.default_rng(seed)
        e = rng.normal(0.0, 1.0, n)
        levels = np.empty(n, dtype=float)
        levels[0] = e[0]
        for k in range(1, n):
            levels[k] = phi * levels[k - 1] + e[k]
        levels += 5.0  # non-zero mean so the t-stat is well away from zero
        x = pd.Series(levels)

        demeaned = levels - levels.mean()
        # Largest prefix of lags 1..L whose sample autocovariance is strictly positive.
        max_pos_lag = 0
        for lag in range(1, min(8, n - 1) + 1):
            gamma = float(demeaned[lag:] @ demeaned[:-lag]) / n
            if gamma <= 0.0:
                break
            max_pos_lag = lag

        mags: list[float] = []
        for lag in range(0, max_pos_lag + 1):
            t = newey_west_tstat(x, lag)
            if not math.isfinite(t):
                return  # a non-positive truncated variance breaks the chain; skip
            mags.append(abs(t))
        for prev, cur in pairwise(mags):
            assert cur <= prev + 1e-9


class TestNonOverlappingProperties:
    @given(
        n=st.integers(min_value=1, max_value=60),
        h=st.integers(min_value=1, max_value=12),
    )
    @settings(max_examples=100, deadline=None)
    def test_kept_grid_points_are_spaced_at_least_h_bars(self, n: int, h: int) -> None:
        ts = [T0 + i * HOUR for i in range(n)]
        s = pd.Series(range(n), index=pd.Index(ts, name="ts_open"), dtype=float)
        out = non_overlapping(s, h)
        kept = sorted(set(out.index))
        # Kept timestamps are a subset of the input grid, strictly increasing, and any
        # two consecutive kept points are >= h bars apart (non-overlapping by construction).
        assert set(kept) <= set(ts)
        positions = [(t - T0) // HOUR for t in kept]
        for prev, cur in pairwise(positions):
            assert cur - prev >= h

    @given(
        n=st.integers(min_value=1, max_value=60),
        h=st.integers(min_value=1, max_value=12),
    )
    @settings(max_examples=100, deadline=None)
    def test_subsampling_never_invents_or_duplicates_rows(self, n: int, h: int) -> None:
        ts = [T0 + i * HOUR for i in range(n)]
        s = pd.Series(np.arange(n, dtype=float), index=pd.Index(ts, name="ts_open"))
        out = non_overlapping(s, h)
        assert len(out) <= len(s)
        # values are carried verbatim from the source rows (no mutation / interpolation)
        source = dict(zip(s.index.to_numpy().tolist(), s.to_numpy().tolist(), strict=True))
        for idx, val in zip(out.index.to_numpy().tolist(), out.to_numpy().tolist(), strict=True):
            assert source[idx] == val


class TestIcSummaryProperties:
    @given(
        ic_vals=st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=60,
        ),
        lags=st.integers(min_value=0, max_value=12),
    )
    @settings(max_examples=100, deadline=None)
    def test_aggregates_are_internally_consistent(self, ic_vals: list[float], lags: int) -> None:
        n = len(ic_vals)
        index = pd.Index([T0 + i * HOUR for i in range(n)], name="ts_open")
        ic = pd.Series(ic_vals, index=index, dtype=float)
        summary = ic_summary(ic, lags=lags)

        assert summary.n == n  # no NaNs were planted ⇒ none dropped
        # mean of values that all lie in [-1, 1] stays in [-1, 1]; matches numpy.
        assert -1.0 <= summary.mean <= 1.0
        assert summary.mean == pytest.approx(float(np.mean(ic_vals)), abs=1e-12)
        # hit_rate is a share in [0, 1] and equals the empirical fraction of strictly-positive ICs.
        assert 0.0 <= summary.hit_rate <= 1.0
        assert summary.hit_rate == pytest.approx(
            float(np.mean([v > 0.0 for v in ic_vals])), abs=1e-12
        )
        if n >= 2 and math.isfinite(summary.std) and summary.std > 0.0:
            assert summary.ic_ir == pytest.approx(summary.mean / summary.std, abs=1e-9)
        else:
            assert math.isnan(summary.ic_ir)
        # by_year means are themselves bounded by the global IC range.
        for year_mean in summary.by_year.values():
            assert -1.0 <= year_mean <= 1.0
