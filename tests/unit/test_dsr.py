"""Unit tests for alphaforge.validation.dsr (alphaDesign.md section 7.4).

Probabilistic and Deflated Sharpe Ratio (Bailey and Lopez de Prado). Every
reference value is hand-derived from the section 7.4 formulas and checked against
scipy's standard-normal CDF/PPF so the maths is pinned exactly:

    PSR(SR*) = Phi( ( (SR - SR*) * sqrt(T - 1) )
                    / sqrt( 1 - g3*SR + ((g4 - 1)/4)*SR^2 ) )
    SR*      = sqrt(V[SR]) * ( (1 - gamma)*PPF(1 - 1/N) + gamma*PPF(1 - 1/(N*e)) )

Covered: the Gaussian PSR reduction to Phi(SR*sqrt(T-1)) for small SR (to 1e-6)
and the exact Gaussian denominator sqrt(1 + SR^2/2) (to 1e-12); monotonicity of
the expected-max Sharpe in the number of trials; deflation of a
high-Sharpe-many-trials configuration below 0.5; the sign effect of skewness and
kurtosis on PSR; the daily-return entry point's per-period vs annualized
bookkeeping; and the error guards.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import norm
from scipy.stats import skew as scipy_skew

from alphaforge.validation import (
    EULER_MASCHERONI,
    DSRReport,
    deflated_sharpe_ratio,
    dsr_from_returns,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)

GAMMA = 0.5772156649
E = math.e


def _psr_reference(sr: float, n: int, g3: float, g4: float, bench: float) -> float:
    """Independent re-implementation of the section 7.4 PSR formula."""
    denom = math.sqrt(1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr)
    z = (sr - bench) * math.sqrt(n - 1.0) / denom
    return float(norm.cdf(z))


def _ems_reference(n_trials: int, var: float) -> float:
    """Independent re-implementation of the section 7.4 expected-max Sharpe."""
    if var == 0.0:
        return 0.0
    q = (1.0 - GAMMA) * float(norm.ppf(1.0 - 1.0 / n_trials)) + GAMMA * float(
        norm.ppf(1.0 - 1.0 / (n_trials * E))
    )
    return math.sqrt(var) * q


class TestEulerMascheroni:
    def test_constant_value(self) -> None:
        assert EULER_MASCHERONI == 0.5772156649


class TestProbabilisticSharpeRatio:
    def test_gaussian_reduction_small_sr(self) -> None:
        # skew=0, kurtosis=3 -> denominator sqrt(1 + sr^2/2); for tiny sr this is
        # ~1 so PSR -> Phi(sr*sqrt(n-1)) to 1e-6.
        sr, n = 0.001, 1000
        psr = probabilistic_sharpe_ratio(sr, n, skew=0.0, kurtosis=3.0)
        reduced = float(norm.cdf(sr * math.sqrt(n - 1.0)))
        assert psr == pytest.approx(reduced, abs=1e-6)

    def test_exact_gaussian_denominator(self) -> None:
        # For skew=0, kurtosis=3 the variance term is exactly 1 + sr^2/2 even at a
        # non-trivial sr; pin the function to that closed form to 1e-12.
        sr, n = 0.1, 101
        denom = math.sqrt(1.0 + 0.5 * sr * sr)
        expected = float(norm.cdf(sr * math.sqrt(n - 1.0) / denom))
        psr = probabilistic_sharpe_ratio(sr, n, skew=0.0, kurtosis=3.0)
        assert psr == pytest.approx(expected, abs=1e-12)
        assert psr == pytest.approx(0.8407413278013518, abs=1e-12)

    def test_matches_full_formula_with_moments(self) -> None:
        sr, n, g3, g4, bench = 0.08, 400, -0.4, 6.0, 0.02
        out = probabilistic_sharpe_ratio(sr, n, skew=g3, kurtosis=g4, sr_benchmark=bench)
        assert out == pytest.approx(_psr_reference(sr, n, g3, g4, bench), abs=1e-12)

    def test_zero_edge_against_zero_benchmark_is_half(self) -> None:
        # sr == sr_benchmark -> z = 0 -> Phi(0) = 0.5 regardless of moments.
        assert probabilistic_sharpe_ratio(0.0, 500, skew=-1.0, kurtosis=9.0) == pytest.approx(
            0.5, abs=1e-12
        )

    def test_negative_skew_lowers_psr(self) -> None:
        # Negative skew (g3 < 0) shrinks the denominator (-g3*sr > 0 in it... but
        # the term is 1 - g3*sr, so g3<0 ADDS to the denominator), inflating the
        # estimator's std and LOWERING PSR vs the symmetric Gaussian. Positive
        # skew does the opposite. Check the ordering explicitly.
        sr, n = 0.1, 500
        base = probabilistic_sharpe_ratio(sr, n, skew=0.0, kurtosis=3.0)
        neg = probabilistic_sharpe_ratio(sr, n, skew=-1.0, kurtosis=3.0)
        pos = probabilistic_sharpe_ratio(sr, n, skew=1.0, kurtosis=3.0)
        assert neg < base < pos

    def test_higher_kurtosis_lowers_psr(self) -> None:
        # Fatter tails (larger g4) enlarge the denominator -> lower PSR for sr > 0.
        sr, n = 0.1, 500
        low = probabilistic_sharpe_ratio(sr, n, skew=0.0, kurtosis=3.0)
        high = probabilistic_sharpe_ratio(sr, n, skew=0.0, kurtosis=12.0)
        assert high < low

    def test_more_observations_sharpen_psr(self) -> None:
        # Same positive sr, more data -> more confident the Sharpe is positive.
        few = probabilistic_sharpe_ratio(0.05, 50, skew=0.0, kurtosis=3.0)
        many = probabilistic_sharpe_ratio(0.05, 5000, skew=0.0, kurtosis=3.0)
        assert 0.5 < few < many < 1.0

    def test_raises_for_too_few_observations(self) -> None:
        with pytest.raises(ValueError, match="n_obs must be >= 2"):
            probabilistic_sharpe_ratio(0.1, 1, skew=0.0, kurtosis=3.0)

    def test_raises_for_nonpositive_variance_term(self) -> None:
        # Choose moments so 1 - g3*sr + ((g4-1)/4)*sr^2 <= 0: huge positive skew
        # with a moderate sr drives the term negative.
        with pytest.raises(ValueError, match="non-positive PSR denominator"):
            probabilistic_sharpe_ratio(0.5, 100, skew=5.0, kurtosis=1.0)


class TestExpectedMaxSharpe:
    def test_matches_reference_formula(self) -> None:
        assert expected_max_sharpe(10, 0.04) == pytest.approx(_ems_reference(10, 0.04), abs=1e-12)
        assert expected_max_sharpe(10, 0.04) == pytest.approx(0.3149196602689944, abs=1e-12)

    def test_increases_with_n_trials(self) -> None:
        var = 0.04
        values = [expected_max_sharpe(n, var) for n in (2, 5, 10, 50, 100, 500, 1000)]
        assert all(b > a for a, b in pairwise(values))

    def test_scales_with_sqrt_variance(self) -> None:
        # SR* is linear in sqrt(V[SR]); doubling sqrt(var) (4x var) doubles SR*.
        small = expected_max_sharpe(100, 0.01)
        big = expected_max_sharpe(100, 0.04)
        assert big == pytest.approx(2.0 * small, abs=1e-12)

    def test_zero_variance_gives_zero_benchmark(self) -> None:
        assert expected_max_sharpe(100, 0.0) == 0.0

    def test_raises_for_single_trial(self) -> None:
        with pytest.raises(ValueError, match="n_trials must be >= 2"):
            expected_max_sharpe(1, 0.04)

    def test_raises_for_negative_variance(self) -> None:
        with pytest.raises(ValueError, match="sr_trials_variance must be >= 0"):
            expected_max_sharpe(10, -0.01)


class TestDeflatedSharpeRatio:
    def test_equals_psr_against_expected_max_benchmark(self) -> None:
        sr, n, g3, g4 = 0.1, 500, -0.2, 5.0
        n_trials, var = 200, 0.03
        bench = expected_max_sharpe(n_trials, var)
        dsr = deflated_sharpe_ratio(sr, n, g3, g4, n_trials, var)
        expected = probabilistic_sharpe_ratio(sr, n, skew=g3, kurtosis=g4, sr_benchmark=bench)
        assert dsr == pytest.approx(expected, abs=1e-12)

    def test_high_sharpe_many_trials_deflates_below_half(self) -> None:
        # A respectable per-period Sharpe (0.12) becomes worthless once you admit
        # it was the best of 2000 searched configs with healthy cross-trial spread.
        dsr = deflated_sharpe_ratio(0.12, 500, 0.0, 3.0, n_trials=2000, sr_trials_variance=0.05)
        assert dsr < 0.5

    def test_deflates_monotonically_in_n_trials(self) -> None:
        # More trials -> higher benchmark -> lower DSR for a fixed observed Sharpe.
        def dsr(n_trials: int) -> float:
            return deflated_sharpe_ratio(
                sr=0.1,
                n_obs=500,
                skew=0.0,
                kurtosis=3.0,
                n_trials=n_trials,
                sr_trials_variance=0.02,
            )

        assert dsr(5) > dsr(50) > dsr(500)

    def test_zero_trial_variance_reduces_to_zero_benchmark_psr(self) -> None:
        # With no cross-trial dispersion SR* = 0, so DSR == PSR(0).
        dsr = deflated_sharpe_ratio(0.08, 300, 0.0, 3.0, n_trials=1000, sr_trials_variance=0.0)
        psr0 = probabilistic_sharpe_ratio(0.08, 300, skew=0.0, kurtosis=3.0, sr_benchmark=0.0)
        assert dsr == pytest.approx(psr0, abs=1e-12)


class TestDsrFromReturns:
    @staticmethod
    def _daily_series(rets: np.ndarray) -> pd.Series:
        day = 86_400_000
        start = 1_704_153_600_000  # 2024-01-02T00:00:00Z
        idx = pd.Index([start + i * day for i in range(rets.size)], name="ts")
        return pd.Series(rets, index=idx, name="ret")

    def test_moments_and_annualization(self) -> None:
        rng = np.random.default_rng(7)
        rets = rng.normal(0.0008, 0.01, size=600)
        series = self._daily_series(rets)
        report = dsr_from_returns(series, n_trials=50, sr_trials_variance=0.02)

        sr = float(np.mean(rets)) / float(np.std(rets, ddof=1))
        g3 = float(scipy_skew(rets, bias=True))
        g4 = float(scipy_kurtosis(rets, fisher=False, bias=True))

        assert isinstance(report, DSRReport)
        assert report.n_obs == 600
        assert report.sr_per_period == pytest.approx(sr, abs=1e-12)
        assert report.skew == pytest.approx(g3, abs=1e-12)
        assert report.kurtosis == pytest.approx(g4, abs=1e-12)
        # default periods_per_year = 365 (24/7 crypto day count).
        assert report.sr_ann == pytest.approx(sr * math.sqrt(365.0), abs=1e-12)
        assert report.expected_max_sr == pytest.approx(expected_max_sharpe(50, 0.02), abs=1e-12)
        assert report.psr == pytest.approx(
            probabilistic_sharpe_ratio(sr, 600, skew=g3, kurtosis=g4), abs=1e-12
        )
        assert report.dsr == pytest.approx(
            probabilistic_sharpe_ratio(
                sr, 600, skew=g3, kurtosis=g4, sr_benchmark=report.expected_max_sr
            ),
            abs=1e-12,
        )

    def test_dsr_below_psr_when_trials_present(self) -> None:
        # A positive benchmark must pull DSR strictly below the zero-benchmark PSR.
        rng = np.random.default_rng(11)
        rets = rng.normal(0.0010, 0.008, size=730)
        report = dsr_from_returns(self._daily_series(rets), n_trials=300, sr_trials_variance=0.03)
        assert report.expected_max_sr > 0.0
        assert report.dsr < report.psr

    def test_custom_periods_per_year(self) -> None:
        rng = np.random.default_rng(3)
        rets = rng.normal(0.0005, 0.012, size=400)
        report = dsr_from_returns(
            self._daily_series(rets), n_trials=10, sr_trials_variance=0.01, periods_per_year=252.0
        )
        assert report.sr_ann == pytest.approx(report.sr_per_period * math.sqrt(252.0), abs=1e-12)

    def test_drops_non_finite_returns(self) -> None:
        rets = np.array([0.01, np.nan, -0.02, 0.015, np.inf, 0.005, -0.01, 0.02])
        report = dsr_from_returns(self._daily_series(rets), n_trials=5, sr_trials_variance=0.01)
        finite = np.array([0.01, -0.02, 0.015, 0.005, -0.01, 0.02])
        assert report.n_obs == finite.size
        sr = float(np.mean(finite)) / float(np.std(finite, ddof=1))
        assert report.sr_per_period == pytest.approx(sr, abs=1e-12)

    def test_raises_for_too_few_returns(self) -> None:
        with pytest.raises(ValueError, match="need >= 2 return observations"):
            dsr_from_returns(
                self._daily_series(np.array([0.01])), n_trials=5, sr_trials_variance=0.01
            )

    def test_raises_for_zero_variance_series(self) -> None:
        with pytest.raises(ValueError, match="zero variance"):
            dsr_from_returns(
                self._daily_series(np.full(20, 0.001)), n_trials=5, sr_trials_variance=0.01
            )


# --------------------------------------------------------------------- property tests
#
# Hypothesis @given idiom (mirrors tests/property/test_hmm_props.py): capped examples,
# deadline disabled so a slow scipy import on the first draw cannot flake the run. Every
# strategy is hand-bounded to the regime where the PSR/DSR formulas are analytically
# well-behaved, so each property holds EXACTLY rather than statistically:
#
# * ``skew`` in [-0.5, 0.5] and ``kurtosis`` >= 3 with |sr| <= 1.5 keep the denominator
#   variance term ``1 - skew*sr + ((kurtosis-1)/4)*sr^2`` >= 0.875 > 0 across the whole
#   envelope, so the function never hits its degenerate-moment ``ValueError`` guard.
# * The "increasing in observed Sharpe" properties FIX ``skew = 0`` and keep
#   ``sr >= benchmark >= 0``: there the z-score derivative ``(1 + b*bench*sr) /
#   (1 + b*sr^2)^{3/2}`` (with ``b = (kurtosis-1)/4 > 0``) is strictly positive, so PSR/DSR
#   rise with ``sr``. Outside that regime (very negative sr against a positive benchmark)
#   the growing denominator can reverse the order, so we do NOT assert monotonicity there.
# * Strict deflation needs ``sr_trials_variance > 0`` and strictly increasing ``n_trials``;
#   with zero cross-trial variance the benchmark is pinned at 0 and DSR == PSR.

# A bounded-moment regime that keeps the PSR denominator variance term strictly positive.
_SR = st.floats(min_value=-1.5, max_value=1.5, allow_nan=False, allow_infinity=False)
_SKEW = st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False)
_KURTOSIS = st.floats(min_value=3.0, max_value=15.0, allow_nan=False, allow_infinity=False)
_N_OBS = st.integers(min_value=2, max_value=10_000)
_N_TRIALS = st.integers(min_value=2, max_value=5_000)
_POS_VAR = st.floats(min_value=1e-4, max_value=0.2, allow_nan=False, allow_infinity=False)


class TestPSRDSRProperties:
    @given(sr=_SR, n_obs=_N_OBS, skew=_SKEW, kurtosis=_KURTOSIS, bench=_SR)
    @settings(max_examples=200, deadline=None)
    def test_psr_is_a_probability(
        self, sr: float, n_obs: int, skew: float, kurtosis: float, bench: float
    ) -> None:
        # PSR = Phi(z): a standard-normal CDF lies in [0, 1]. For an extreme z (a large
        # Sharpe gap over many observations) the float CDF saturates exactly at the
        # 0.0 / 1.0 boundary, so the honest invariant is the CLOSED interval.
        psr = probabilistic_sharpe_ratio(
            sr, n_obs, skew=skew, kurtosis=kurtosis, sr_benchmark=bench
        )
        assert 0.0 <= psr <= 1.0

    @given(
        sr=_SR,
        n_obs=_N_OBS,
        skew=_SKEW,
        kurtosis=_KURTOSIS,
        n_trials=_N_TRIALS,
        var=_POS_VAR,
    )
    @settings(max_examples=200, deadline=None)
    def test_dsr_is_a_probability(
        self,
        sr: float,
        n_obs: int,
        skew: float,
        kurtosis: float,
        n_trials: int,
        var: float,
    ) -> None:
        dsr = deflated_sharpe_ratio(sr, n_obs, skew, kurtosis, n_trials, var)
        assert 0.0 <= dsr <= 1.0

    @given(
        sr=_SR,
        n_obs=_N_OBS,
        skew=_SKEW,
        kurtosis=_KURTOSIS,
        n_trials=_N_TRIALS,
        var=_POS_VAR,
    )
    @settings(max_examples=200, deadline=None)
    def test_dsr_never_exceeds_zero_benchmark_psr(
        self,
        sr: float,
        n_obs: int,
        skew: float,
        kurtosis: float,
        n_trials: int,
        var: float,
    ) -> None:
        # Deflation can only LOWER the verdict: a strictly positive expected-max
        # benchmark (var > 0) pulls DSR at or below the zero-benchmark PSR.
        psr = probabilistic_sharpe_ratio(sr, n_obs, skew=skew, kurtosis=kurtosis)
        dsr = deflated_sharpe_ratio(sr, n_obs, skew, kurtosis, n_trials, var)
        assert dsr <= psr

    @given(
        sr=_SR,
        n_obs=_N_OBS,
        skew=_SKEW,
        kurtosis=_KURTOSIS,
        var=_POS_VAR,
        n_low=st.integers(min_value=2, max_value=2_500),
        bump=st.integers(min_value=1, max_value=2_500),
    )
    @settings(max_examples=200, deadline=None)
    def test_dsr_monotone_decreasing_in_n_trials(
        self,
        sr: float,
        n_obs: int,
        skew: float,
        kurtosis: float,
        var: float,
        n_low: int,
        bump: int,
    ) -> None:
        # More trials -> higher expected-max benchmark -> higher SR* -> smaller z ->
        # lower DSR, holding everything else fixed (requires var > 0 so the benchmark
        # actually moves). The deflation benchmark itself is STRICTLY increasing in
        # n_trials; the DSR is non-increasing always (and strictly so wherever the
        # float CDF has not yet saturated to the 0.0 / 1.0 boundary).
        n_high = n_low + bump
        bench_low = expected_max_sharpe(n_low, var)
        bench_high = expected_max_sharpe(n_high, var)
        assert bench_high > bench_low  # the mechanism: SR* strictly rises with N
        dsr_low = deflated_sharpe_ratio(sr, n_obs, skew, kurtosis, n_low, var)
        dsr_high = deflated_sharpe_ratio(sr, n_obs, skew, kurtosis, n_high, var)
        assert dsr_high <= dsr_low

    @given(
        sr_low=st.floats(min_value=0.0, max_value=1.4, allow_nan=False, allow_infinity=False),
        bump=st.floats(min_value=1e-3, max_value=0.5, allow_nan=False, allow_infinity=False),
        n_obs=_N_OBS,
        kurtosis=_KURTOSIS,
    )
    @settings(max_examples=200, deadline=None)
    def test_psr_monotone_increasing_in_observed_sharpe(
        self, sr_low: float, bump: float, n_obs: int, kurtosis: float
    ) -> None:
        # Fixing skew=0 and benchmark=0 with sr >= 0, a larger observed Sharpe is never
        # less convincing. >= (not >) tolerates the fp plateau as PSR saturates to 1.0.
        psr_low = probabilistic_sharpe_ratio(sr_low, n_obs, skew=0.0, kurtosis=kurtosis)
        psr_high = probabilistic_sharpe_ratio(sr_low + bump, n_obs, skew=0.0, kurtosis=kurtosis)
        assert psr_high >= psr_low

    @given(
        sr_above_bench=st.floats(
            min_value=1e-3, max_value=1.4, allow_nan=False, allow_infinity=False
        ),
        bump=st.floats(min_value=1e-3, max_value=0.5, allow_nan=False, allow_infinity=False),
        n_obs=_N_OBS,
        kurtosis=_KURTOSIS,
        n_trials=_N_TRIALS,
        var=_POS_VAR,
    )
    @settings(max_examples=200, deadline=None)
    def test_dsr_monotone_increasing_in_observed_sharpe(
        self,
        sr_above_bench: float,
        bump: float,
        n_obs: int,
        kurtosis: float,
        n_trials: int,
        var: float,
    ) -> None:
        # Same property against the deflated benchmark: measure both Sharpes RELATIVE to
        # the (fixed) expected-max benchmark so the numerator stays non-negative, where
        # the z-score is strictly increasing in sr. skew=0 keeps the denominator even.
        bench = expected_max_sharpe(n_trials, var)
        sr_low = bench + sr_above_bench
        dsr_low = deflated_sharpe_ratio(sr_low, n_obs, 0.0, kurtosis, n_trials, var)
        dsr_high = deflated_sharpe_ratio(sr_low + bump, n_obs, 0.0, kurtosis, n_trials, var)
        assert dsr_high >= dsr_low

    @given(skew=_SKEW, kurtosis=_KURTOSIS, sr=_SR)
    @settings(max_examples=120, deadline=None)
    def test_small_n_regime_is_sane(self, skew: float, kurtosis: float, sr: float) -> None:
        # The smallest admissible sample (n_obs == 2) collapses sqrt(n_obs - 1) to 1 but
        # still yields a valid probability; n_obs == 1 must raise. A zero edge against a
        # zero benchmark is exactly 0.5 regardless of the (finite) moments.
        psr = probabilistic_sharpe_ratio(sr, 2, skew=skew, kurtosis=kurtosis)
        assert 0.0 < psr < 1.0
        assert probabilistic_sharpe_ratio(0.0, 2, skew=skew, kurtosis=kurtosis) == pytest.approx(
            0.5, abs=1e-12
        )
        with pytest.raises(ValueError, match="n_obs must be >= 2"):
            probabilistic_sharpe_ratio(sr, 1, skew=skew, kurtosis=kurtosis)


class TestWithinPathDeflationIsMoreConservative:
    """CPCV per-path V[SR] is the HONESTER deflation: it can only push DSR DOWN.

    The upgrade replaces the between-config Sharpe variance (the spread of ONE mean
    estimate per config) with the within-CPCV-path variance (the spread of a config's
    own resampled OOS backtest paths, pooled). The mechanism is the LAW OF TOTAL
    VARIANCE: the *population* variance of the pooled per-path Sharpes decomposes as

        Var_pop(pooled) = mean_g Var_pop(paths in group g)  +  Var_pop(group means),

    so the pooled (within-path) population variance is ALWAYS >= the population variance
    of the per-config means — strictly so whenever any group has a non-zero internal
    spread. A larger ``V[SR]`` raises the expected-max benchmark ``SR*``, which lowers
    DSR for a fixed observed Sharpe (the ``test_deflates_monotonically`` mechanism). So
    on the same data the within-path DSR is never ABOVE the between-config DSR — exactly
    the conservative direction the design intends (alphaDesign.md §7.4).

    NOTE on ddof: the production ``trial_sharpe_variance`` uses sample (ddof=1) variance.
    The clean inequality ``var_within >= var_between`` is a POPULATION identity; with
    ddof=1 over tiny group counts the Bessel correction can locally reverse it, so the
    property here is stated on the *population* (ddof=0) variances that the law of total
    variance governs, then the production ddof=1 path is pinned separately by
    ``tests/unit/test_cpcv_path_sharpes.py``.
    """

    @staticmethod
    def _between_config_var_pop(groups: list[list[float]]) -> float:
        """Population variance of the per-group MEAN Sharpes (between-config V[SR])."""
        means = np.asarray([float(np.mean(g)) for g in groups], dtype=float)
        return float(np.var(means, ddof=0))

    @staticmethod
    def _within_path_var_pop(groups: list[list[float]]) -> float:
        """Population variance over the FLATTENED per-path Sharpes (within-path V[SR])."""
        flat = np.asarray([s for g in groups for s in g], dtype=float)
        return float(np.var(flat, ddof=0))

    @given(
        base=st.lists(
            st.floats(min_value=-0.4, max_value=0.4, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=6,
        ),
        spread=st.floats(min_value=0.0, max_value=0.3, allow_nan=False, allow_infinity=False),
        sr=st.floats(min_value=0.0, max_value=1.2, allow_nan=False, allow_infinity=False),
        n_obs=st.integers(min_value=30, max_value=5000),
        n_trials=st.integers(min_value=2, max_value=2000),
        kurtosis=st.floats(min_value=3.0, max_value=12.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None)
    def test_within_path_dsr_not_above_between_config_dsr(
        self,
        base: list[float],
        spread: float,
        sr: float,
        n_obs: int,
        n_trials: int,
        kurtosis: float,
    ) -> None:
        # Equal-size CPCV path groups (3 paths each), centred on each config's own mean
        # +/- a shared spread. The law of total variance then governs the two variances.
        groups = [[m - spread, m, m + spread] for m in base]
        var_between = self._between_config_var_pop(groups)
        var_within = self._within_path_var_pop(groups)
        # The mechanism (law of total variance): the within-path pool is at least as
        # dispersed as the collapsed per-config means.
        assert var_within >= var_between - 1e-12

        dsr_between = deflated_sharpe_ratio(sr, n_obs, 0.0, kurtosis, n_trials, var_between)
        dsr_within = deflated_sharpe_ratio(sr, n_obs, 0.0, kurtosis, n_trials, var_within)
        # Wider V[SR] -> higher SR* -> DSR can only fall (<=, tolerating the fp plateau).
        assert dsr_within <= dsr_between + 1e-12

    def test_worked_example_within_path_strictly_more_conservative(self) -> None:
        # A concrete pair where the within-path population variance is strictly larger
        # (every group has internal spread), so the within-path DSR is strictly lower.
        groups = [[0.02, 0.05, -0.01], [0.10, 0.13, 0.07], [-0.05, 0.00, -0.10]]
        var_between = self._between_config_var_pop(groups)
        var_within = self._within_path_var_pop(groups)
        assert var_within > var_between
        dsr_between = deflated_sharpe_ratio(0.15, 500, 0.0, 4.0, 50, var_between)
        dsr_within = deflated_sharpe_ratio(0.15, 500, 0.0, 4.0, 50, var_within)
        assert dsr_within < dsr_between
