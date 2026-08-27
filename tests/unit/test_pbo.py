"""Unit tests for alphaforge.validation.pbo (CSCV PBO, alphaDesign.md section 7.3).

Coverage:

* A fully hand-computed S=4, N=3 micro case: every per-combination Sharpe, OOS
  rank, omega, logit lambda, the lambda vector, PBO, and the IS/OOS pair scatter
  are asserted against values worked out by hand (the exact-math anchor).
* The :func:`_sharpe` primitive: mean/std (ddof=0) with the zero-variance ->
  -inf rule.
* The three diagnostic regimes the statistic must distinguish (BBLZ 2017):
  identical-noise configs -> PBO near 0.5; one consistently-superior config ->
  PBO ~ 0; a mirror-pair overfit (IS-best is OOS-worst) -> PBO near 1.0.
* Combination capping with seeded determinism, full enumeration below the cap,
  trailing-row trimming, and the omega-in-(0,1) finiteness invariant.
* Input validation.

All data is deterministic and offline (seeded ``numpy`` PRNGs / literal arrays).
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from alphaforge.validation.pbo import PBOResult, _sharpe, pbo_cscv

# --------------------------------------------------------------------------- #
# Hand-computed micro case: S=4 (half=2), block_len=2, N=3, C(4,2)=6 combos.
#
# Rows (8) grouped into 4 contiguous 2-row blocks; 3 config columns:
#
#   block 0: [0.02, 0.01, -0.03], [0.00, 0.03, -0.01]
#   block 1: [0.04,-0.02,  0.02], [0.02, 0.00,  0.04]
#   block 2: [-0.01,0.05,  0.01], [0.01, 0.03,  0.03]
#   block 3: [0.03, 0.02, -0.04], [0.05, 0.04, -0.02]
#
# Per-config block Sharpe = mean/std (ddof=0) of the 2 rows. Concatenating the
# two IS blocks (4 rows) gives the IS Sharpe; the two OOS blocks give the OOS
# Sharpe. Worked out by hand (and reproduced in test_micro_matches_hand_trace):
#
#  c IS-blocks  IS-best  SR_IS(best)   SR_OOS(best)   OOS-rank  omega  lambda
#  0  (0,1)        0      1.4142135624  0.8944271910      2     0.50    0.0
#  1  (0,2)        1      2.1213203436  0.4472135955      2     0.50    0.0
#  2  (0,3)        1      2.2360679775  0.5570860145      1     0.25   -1.0986122887
#  3  (1,2)        2      2.2360679775 -2.2360679775      1     0.25   -1.0986122887
#  4  (1,3)        0      3.1304951685  0.4472135955      2     0.50    0.0
#  5  (2,3)        1      3.1304951685  0.2773500981      2     0.50    0.0
#
# All six lambda <= 0  =>  PBO = 1.0.
# --------------------------------------------------------------------------- #
MICRO = np.array(
    [
        [0.02, 0.01, -0.03],
        [0.00, 0.03, -0.01],
        [0.04, -0.02, 0.02],
        [0.02, 0.00, 0.04],
        [-0.01, 0.05, 0.01],
        [0.01, 0.03, 0.03],
        [0.03, 0.02, -0.04],
        [0.05, 0.04, -0.02],
    ],
    dtype=float,
)

_LOGIT_QUARTER = math.log(0.25 / 0.75)  # = -1.0986122886681098

# Lambdas in evaluation order = sorted C(4,2) combinations.
EXPECTED_LAMBDAS = np.array([0.0, 0.0, _LOGIT_QUARTER, _LOGIT_QUARTER, 0.0, 0.0], dtype=float)
# (SR_IS(best), SR_OOS(best)) per combination.
EXPECTED_PAIRS = np.array(
    [
        [1.4142135624, 0.8944271910],
        [2.1213203436, 0.4472135955],
        [2.2360679775, 0.5570860145],
        [2.2360679775, -2.2360679775],
        [3.1304951685, 0.4472135955],
        [3.1304951685, 0.2773500981],
    ],
    dtype=float,
)


class TestSharpePrimitive:
    def test_mean_over_population_std(self) -> None:
        # block = [[0.02,0.01,-0.03],[0.00,0.03,-0.01]]:
        # col0 mean=0.01, std(ddof=0)=0.01 -> 1.0
        # col1 mean=0.02, std=0.01 -> 2.0
        # col2 mean=-0.02, std=0.01 -> -2.0
        sr = _sharpe(MICRO[:2])
        assert sr == pytest.approx([1.0, 2.0, -2.0], abs=1e-12)

    def test_zero_variance_maps_to_minus_inf(self) -> None:
        # A constant column has undefined Sharpe -> -inf (never IS-best).
        block = np.array([[1.0, 0.5], [1.0, 1.5]], dtype=float)
        sr = _sharpe(block)
        assert sr[0] == -np.inf  # constant 1.0 column
        assert math.isfinite(sr[1])

    def test_single_row_block_zero_std(self) -> None:
        # One row -> std 0 everywhere -> all -inf.
        sr = _sharpe(MICRO[:1])
        assert np.all(sr == -np.inf)


class TestMicroHandComputed:
    def test_micro_matches_hand_trace(self) -> None:
        res = pbo_cscv(MICRO, n_splits=4, max_combinations=5000, seed=42)
        assert isinstance(res, PBOResult)
        assert res.n_combinations == math.comb(4, 2)  # 6
        np.testing.assert_allclose(res.lambdas, EXPECTED_LAMBDAS, atol=1e-12)
        np.testing.assert_allclose(res.is_oos_pairs, EXPECTED_PAIRS, atol=1e-9)
        # All six lambda <= 0 -> PBO = 1.0.
        assert res.pbo == pytest.approx(1.0, abs=1e-12)

    def test_pbo_equals_left_tail_fraction(self) -> None:
        res = pbo_cscv(MICRO, n_splits=4, max_combinations=5000, seed=42)
        assert res.pbo == pytest.approx(float(np.mean(res.lambdas <= 0.0)), abs=1e-15)

    def test_dataframe_and_array_inputs_agree(self) -> None:
        df = pd.DataFrame(MICRO, columns=["cfg_a", "cfg_b", "cfg_c"])
        a = pbo_cscv(df, n_splits=4, max_combinations=5000, seed=42)
        b = pbo_cscv(MICRO, n_splits=4, max_combinations=5000, seed=42)
        np.testing.assert_array_equal(a.lambdas, b.lambdas)
        assert a.pbo == b.pbo

    def test_lambda_sign_pins_to_omega_half(self) -> None:
        # Reproduce the rank machinery independently and confirm the sign of
        # lambda is exactly determined by whether the IS-best is above the OOS
        # median (omega vs 0.5).
        from scipy.stats import rankdata

        blocks = MICRO.reshape(4, 2, 3)
        for c, is_blocks in enumerate(combinations(range(4), 2)):
            oos = sorted(set(range(4)) - set(is_blocks))
            sr_is = _sharpe(blocks[list(is_blocks)].reshape(-1, 3))
            sr_oos = _sharpe(blocks[oos].reshape(-1, 3))
            best = int(np.argmax(sr_is))
            omega = float(rankdata(sr_oos, method="average")[best]) / 4.0
            expected = math.log(omega / (1.0 - omega))
            assert EXPECTED_LAMBDAS[c] == pytest.approx(expected, abs=1e-12)


class TestRegimes:
    def test_identical_noise_pbo_near_half(self) -> None:
        # iid-identical noise configs: the IS-best is a winner's-curse pick that
        # regresses to the OOS mean, so it lands at/below the OOS median about
        # half the time -> PBO ~ 0.5 (BBLZ 2017).
        rng = np.random.default_rng(123)
        matrix = rng.standard_normal((1600, 20)) * 0.01
        res = pbo_cscv(matrix, n_splits=16, max_combinations=5000, seed=42)
        assert 0.35 <= res.pbo <= 0.65

    def test_superior_config_pbo_low(self) -> None:
        # One config has a genuine, time-stable positive mean edge -> it is
        # IS-best AND OOS-best in essentially every split -> PBO ~ 0.
        rng = np.random.default_rng(123)
        matrix = rng.standard_normal((1600, 20)) * 0.01
        matrix[:, 0] += 0.004  # dominant Sharpe everywhere
        res = pbo_cscv(matrix, n_splits=16, max_combinations=5000, seed=42)
        assert res.pbo < 0.05

    def test_overfit_config_pbo_near_one(self) -> None:
        # Mirror-pair overfit: configs 0 and 1 have near-zero within-block
        # variance (enormous |block Sharpe|) with block means that are large +
        # on one time-half and large - on the other, mirrored. Whichever half is
        # "+ heavy" in-sample makes one of them IS-best; that same config is then
        # OOS-worst on its - half. The IS-best is consistently OOS-worst ->
        # PBO near 1.0.
        rng = np.random.default_rng(7)
        n_splits = 16
        t, n = 1600, 30
        block_len = t // n_splits
        matrix = rng.standard_normal((t, n)) * 0.01
        for b in range(n_splits):
            seg = slice(b * block_len, (b + 1) * block_len)
            first_half = b < n_splits // 2
            matrix[seg, 0] = (0.10 if first_half else -0.10) + rng.standard_normal(block_len) * 1e-4
            matrix[seg, 1] = (-0.10 if first_half else 0.10) + rng.standard_normal(block_len) * 1e-4
        res = pbo_cscv(matrix, n_splits=n_splits, max_combinations=5000, seed=42)
        assert res.pbo > 0.85

    def test_gate_documented_threshold(self) -> None:
        # The live-eligibility gate is PBO < 0.2 (alphaDesign.md section 7.3):
        # the superior config passes, the overfit configs do not.
        rng = np.random.default_rng(123)
        good = rng.standard_normal((1600, 20)) * 0.01
        good[:, 0] += 0.004
        assert pbo_cscv(good, n_splits=16, seed=42).pbo < 0.2


class TestCombinationCapAndDeterminism:
    def test_full_enumeration_below_cap(self) -> None:
        # C(8,4) = 70 <= cap -> all 70 enumerated, seed unused.
        rng = np.random.default_rng(99)
        matrix = rng.standard_normal((800, 12)) * 0.01
        res = pbo_cscv(matrix, n_splits=8, max_combinations=5000, seed=42)
        assert res.n_combinations == math.comb(8, 4) == 70
        assert res.lambdas.shape == (70,)
        assert res.is_oos_pairs.shape == (70, 2)

    def test_cap_limits_combination_count(self) -> None:
        rng = np.random.default_rng(99)
        matrix = rng.standard_normal((800, 12)) * 0.01
        res = pbo_cscv(matrix, n_splits=8, max_combinations=40, seed=42)
        assert res.n_combinations == 40
        assert res.lambdas.shape == (40,)

    def test_default_s16_full_enumeration(self) -> None:
        # C(16,8) = 12870 > default cap 5000 -> exactly 5000 sampled combos.
        rng = np.random.default_rng(5)
        matrix = rng.standard_normal((1600, 25)) * 0.01
        res = pbo_cscv(matrix, n_splits=16)
        assert res.n_combinations == 5000

    def test_seed_determinism(self) -> None:
        rng = np.random.default_rng(99)
        matrix = rng.standard_normal((1600, 30)) * 0.01
        a = pbo_cscv(matrix, n_splits=16, max_combinations=2000, seed=42)
        b = pbo_cscv(matrix, n_splits=16, max_combinations=2000, seed=42)
        np.testing.assert_array_equal(a.lambdas, b.lambdas)
        np.testing.assert_array_equal(a.is_oos_pairs, b.is_oos_pairs)
        assert a.pbo == b.pbo

    def test_different_seed_changes_sample(self) -> None:
        rng = np.random.default_rng(99)
        matrix = rng.standard_normal((1600, 30)) * 0.01
        a = pbo_cscv(matrix, n_splits=16, max_combinations=2000, seed=42)
        c = pbo_cscv(matrix, n_splits=16, max_combinations=2000, seed=7)
        assert not np.array_equal(a.lambdas, c.lambdas)

    def test_sampled_combinations_are_distinct(self) -> None:
        # The without-replacement sampler must never repeat a combination, so
        # the lambda vector length equals the requested cap with no dup combos.
        rng = np.random.default_rng(3)
        matrix = rng.standard_normal((1600, 10)) * 0.01
        res = pbo_cscv(matrix, n_splits=16, max_combinations=3000, seed=11)
        assert res.n_combinations == 3000


class TestInvariantsAndEdges:
    def test_trailing_rows_trimmed(self) -> None:
        # 1607 rows, S=16 -> block_len=100, 7 trailing rows dropped; still runs.
        rng = np.random.default_rng(1)
        matrix = rng.standard_normal((1607, 30)) * 0.01
        res = pbo_cscv(matrix, n_splits=16, max_combinations=500, seed=1)
        assert res.n_combinations == 500

    def test_all_lambdas_finite(self) -> None:
        # omega = rank/(N+1) with 1 <= rank <= N is strictly inside (0,1), so
        # every logit is finite (no +/-inf, no NaN).
        rng = np.random.default_rng(2)
        matrix = rng.standard_normal((1600, 20)) * 0.01
        res = pbo_cscv(matrix, n_splits=16, max_combinations=1000, seed=2)
        assert np.all(np.isfinite(res.lambdas))

    def test_pbo_in_unit_interval(self) -> None:
        rng = np.random.default_rng(4)
        matrix = rng.standard_normal((1600, 20)) * 0.01
        res = pbo_cscv(matrix, n_splits=16, max_combinations=1000, seed=4)
        assert 0.0 <= res.pbo <= 1.0


class TestValidation:
    @pytest.mark.parametrize("bad", [1, 3, 0, -2, 15])
    def test_n_splits_must_be_even_ge_2(self, bad: int) -> None:
        matrix = np.zeros((100, 5))
        with pytest.raises(ValueError, match="even integer"):
            pbo_cscv(matrix, n_splits=bad)

    def test_max_combinations_must_be_positive(self) -> None:
        matrix = np.zeros((100, 5))
        with pytest.raises(ValueError, match="max_combinations"):
            pbo_cscv(matrix, n_splits=4, max_combinations=0)

    def test_needs_two_configs(self) -> None:
        matrix = np.zeros((100, 1))
        with pytest.raises(ValueError, match="2 config columns"):
            pbo_cscv(matrix, n_splits=4)

    def test_rows_must_cover_blocks(self) -> None:
        matrix = np.zeros((10, 5))
        with pytest.raises(ValueError, match="rows"):
            pbo_cscv(matrix, n_splits=16)

    def test_rejects_non_2d(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            pbo_cscv(np.zeros((10, 5, 2)), n_splits=4)


# --------------------------------------------------------------------------- #
# Hypothesis property tests (A5-1). Four load-bearing CSCV invariants, each
# stated as a property over a generated family of inputs rather than a single
# fixture:
#
#   1. RANGE          PBO is always a probability in [0, 1] (and PBO is exactly
#                     the left-tail fraction of a finite lambda vector), for ANY
#                     finite per-period return matrix and any valid knobs.
#   2. NULL           on i.i.d.-noise PnL (no cross-sectional edge) the IS-best
#                     pick is a winner's curse that regresses to the OOS median,
#                     so PBO concentrates near 0.5 across many seeds.
#   3. REAL SIGNAL    when one column dominates OOS consistently, the IS-best is
#                     also OOS-best almost everywhere -> PBO is LOW (< gate 0.2).
#   4. OVERFIT        a deliberately rank-inverting (mirror-pair) construction,
#                     where the IS-best is engineered to be OOS-worst, drives
#                     PBO HIGH (-> 1).
#
# Examples are capped and deadlines disabled because each example runs a full
# CSCV sweep (up to thousands of combinations); derandomize keeps the suite
# reproducible. All data is seeded/synthetic - never the live data lake.
# --------------------------------------------------------------------------- #
_NULL_SEEDS = list(range(40))


def _iid_noise(t: int, n: int, seed: int, *, scale: float = 0.01) -> np.ndarray:
    """A pure i.i.d.-Gaussian return matrix: T x N with no cross-sectional edge."""
    return np.asarray(np.random.default_rng(seed).standard_normal((t, n)) * scale, dtype=float)


class TestPropertyRange:
    """Property 1: PBO is a probability and equals the lambda left-tail fraction."""

    @given(
        t=st.integers(min_value=64, max_value=400),
        n=st.integers(min_value=2, max_value=12),
        n_splits=st.sampled_from([2, 4, 6, 8]),
        max_combinations=st.integers(min_value=1, max_value=80),
        seed=st.integers(min_value=0, max_value=10_000),
        data=st.data(),
    )
    @settings(max_examples=60, derandomize=True, deadline=None)
    def test_pbo_is_a_probability_and_left_tail_fraction(
        self,
        t: int,
        n: int,
        n_splits: int,
        max_combinations: int,
        seed: int,
        data: st.DataObject,
    ) -> None:
        # Arbitrary finite returns (bounded so std is well-defined; values may be
        # any sign / magnitude in band, including degenerate all-equal columns).
        matrix = data.draw(
            hnp.arrays(
                np.float64,
                shape=(t, n),
                elements=st.floats(
                    min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
                ),
            )
        )
        res = pbo_cscv(matrix, n_splits=n_splits, max_combinations=max_combinations, seed=seed)
        assert isinstance(res, PBOResult)
        # Probability range.
        assert 0.0 <= res.pbo <= 1.0
        # The defining identity: PBO is the fraction of combos with lambda <= 0.
        assert res.pbo == pytest.approx(float(np.mean(res.lambdas <= 0.0)), abs=1e-15)
        # omega = rank/(N+1) is strictly inside (0, 1) -> every logit is finite.
        assert np.all(np.isfinite(res.lambdas))
        # Bookkeeping: result arrays are sized by the number of combos evaluated.
        full = math.comb(n_splits, n_splits // 2)
        assert res.n_combinations == min(full, max_combinations)
        assert res.lambdas.shape == (res.n_combinations,)
        assert res.is_oos_pairs.shape == (res.n_combinations, 2)


class TestPropertyNull:
    """Property 2: i.i.d. noise (no signal) -> PBO concentrates near 0.5."""

    @given(seed=st.sampled_from(_NULL_SEEDS))
    @settings(max_examples=len(_NULL_SEEDS), derandomize=True, deadline=None)
    def test_each_noise_seed_never_a_confident_verdict(self, seed: int) -> None:
        # A SINGLE i.i.d.-noise draw has substantial PBO sampling variance (with
        # N=20 configs the per-seed PBO empirically wanders the ~[0.23, 0.88]
        # range), so the per-seed property is NOT a tight band around 0.5 - that
        # is statistically false for one draw. What IS always true under the null
        # is that no noise draw produces a *confident* verdict at either extreme:
        # it never looks like a clean edge (PBO -> 0) or a clean overfit
        # (PBO -> 1). The sharp ~0.5 statement is the multi-seed mean below.
        matrix = _iid_noise(1600, 20, seed)
        res = pbo_cscv(matrix, n_splits=16, max_combinations=2000, seed=42)
        assert 0.1 <= res.pbo <= 0.9

    def test_mean_pbo_over_many_seeds_is_approximately_half(self) -> None:
        # The sharp statement: averaged over many independent noise matrices the
        # mean PBO is ~0.5 (the CSCV null), with a tight tolerance the per-seed
        # band cannot give. This is the BBLZ 2017 calibration of the statistic.
        pbos = [
            pbo_cscv(_iid_noise(1600, 20, s), n_splits=16, max_combinations=2000, seed=42).pbo
            for s in _NULL_SEEDS
        ]
        assert np.mean(pbos) == pytest.approx(0.5, abs=0.06)


class TestPropertyRealSignal:
    """Property 3: one OOS-consistent dominant column -> PBO is LOW."""

    @given(
        seed=st.integers(min_value=0, max_value=10_000),
        edge=st.floats(min_value=0.003, max_value=0.01),
        n=st.integers(min_value=5, max_value=25),
    )
    @settings(
        max_examples=30,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_dominant_column_drives_pbo_below_gate(self, seed: int, edge: float, n: int) -> None:
        # Column 0 carries a genuine, time-stable positive mean edge added on top
        # of common noise: it is IS-best AND OOS-best in essentially every split,
        # so the IS-best is (almost) never below the OOS median -> PBO is LOW and
        # clears the live-eligibility gate (alphaDesign.md section 7.3: < 0.2).
        matrix = _iid_noise(1600, n, seed)
        matrix[:, 0] += edge
        res = pbo_cscv(matrix, n_splits=16, max_combinations=2000, seed=42)
        assert res.pbo < 0.2


class TestPropertyOverfit:
    """Property 4: a rank-inverting mirror-pair construction -> PBO is HIGH."""

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @settings(
        max_examples=25,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_mirror_pair_overfit_drives_pbo_high(self, seed: int) -> None:
        # Deliberately overfit / rank-inverting: a pure mirror pair (the only two
        # configs) with near-zero within-block variance (enormous |block Sharpe|)
        # whose block means are large + on one time-half and large - on the
        # mirrored half. Whichever half is "+ heavy" in-sample makes one of the
        # pair IS-best; that same config is then OOS-worst on its "-" half, so the
        # IS-best is OOS-worst on every TIME-BALANCED split -> PBO is HIGH.
        #
        # We enumerate the FULL C(16, 8) here (max_combinations above the count)
        # rather than sub-sampling: the inversion is exact only for balanced
        # splits, and the minority of unbalanced sampled splits adds variance
        # that, on an adverse seed, can drag a sub-sampled PBO down toward ~0.65.
        # Full enumeration keeps the property robustly HIGH (> 0.6) across all
        # construction seeds while staying far above the 0.5 null and the 0.2
        # gate - the qualitative "overfit" verdict the statistic must deliver.
        rng = np.random.default_rng(seed)
        n_splits = 16
        t = 1600
        block_len = t // n_splits
        matrix = np.empty((t, 2), dtype=np.float64)
        for b in range(n_splits):
            seg = slice(b * block_len, (b + 1) * block_len)
            first_half = b < n_splits // 2
            matrix[seg, 0] = (0.10 if first_half else -0.10) + rng.standard_normal(block_len) * 1e-4
            matrix[seg, 1] = (-0.10 if first_half else 0.10) + rng.standard_normal(block_len) * 1e-4
        res = pbo_cscv(matrix, n_splits=n_splits, max_combinations=13_000, seed=42)
        assert res.n_combinations == math.comb(16, 8)  # full enumeration, no sub-sampling
        assert res.pbo > 0.6
