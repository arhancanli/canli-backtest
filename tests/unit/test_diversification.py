from __future__ import annotations

import numpy as np
import pytest

from alphaforge.validation.diversification import diversification_report


def fixture() -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(9)
    n = 756
    factor = rng.normal(0.0002, 0.006, n)
    sleeves = {
        "alpha": factor + rng.normal(0.0, 0.004, n),
        "beta": -0.2 * factor + rng.normal(0.0001, 0.006, n),
    }
    book = 0.5 * sleeves["alpha"] + 0.5 * sleeves["beta"]
    candidate = rng.normal(0.0008, 0.005, n)
    stress = book <= np.quantile(book, 0.10)
    periods = [str(2023 + index // 252) for index in range(n)]
    return candidate, sleeves, book, stress, periods


def report(candidate: np.ndarray | None = None):
    base, sleeves, book, stress, periods = fixture()
    return diversification_report(
        base if candidate is None else candidate,
        sleeves,
        book,
        stress_mask=stress,
        period_labels=periods,
        candidate_weight=0.10,
        bootstrap_samples=300,
    )


def test_report_is_deterministic_aligned_and_complete() -> None:
    first = report()
    second = report()
    assert first == second
    assert first.observations == 756
    assert set(first.pairwise_correlations) == {"alpha", "beta"}
    assert set(first.leave_one_period_out_book_sharpe_deltas) == {"2023", "2024", "2025"}
    assert first.max_pairwise_correlation_upper_95 >= first.max_pairwise_correlation


def test_common_stress_exposure_is_visible_when_full_sample_is_muted() -> None:
    candidate, sleeves, _, stress, _ = fixture()
    candidate[stress] = 1.5 * sleeves["alpha"][stress]
    result = report(candidate)
    assert result.max_stressed_pairwise_correlation > result.max_pairwise_correlation
    assert result.max_stressed_pairwise_correlation > 0.5
    assert result.stressed_joint_loss_rate > 0.5
    assert result.candidate_mean_on_book_es_days < 0.0


def test_book_damage_and_leave_period_failure_are_reported() -> None:
    candidate, sleeves, book, stress, periods = fixture()
    candidate[:252] -= 0.002
    result = diversification_report(
        candidate,
        sleeves,
        book,
        stress_mask=stress,
        period_labels=periods,
        candidate_weight=0.20,
        bootstrap_samples=300,
    )
    assert result.minimum_leave_one_period_out_book_sharpe_delta < result.book_sharpe_delta
    assert result.book_expected_shortfall_delta != 0.0
    assert result.book_max_drawdown_delta != 0.0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0])
def test_missing_nonfinite_and_total_loss_returns_fail_closed(bad: float) -> None:
    candidate, sleeves, book, stress, periods = fixture()
    candidate[10] = bad
    with pytest.raises(ValueError):
        diversification_report(
            candidate,
            sleeves,
            book,
            stress_mask=stress,
            period_labels=periods,
            candidate_weight=0.10,
            bootstrap_samples=300,
        )


def test_alignment_and_stress_masks_cannot_be_repaired_silently() -> None:
    candidate, sleeves, book, stress, periods = fixture()
    with pytest.raises(ValueError, match="length"):
        diversification_report(
            candidate[:-1],
            sleeves,
            book,
            stress_mask=stress,
            period_labels=periods,
            candidate_weight=0.10,
            bootstrap_samples=300,
        )
    with pytest.raises(ValueError, match="boolean"):
        diversification_report(
            candidate,
            sleeves,
            book,
            stress_mask=stress.astype(int),
            period_labels=periods,
            candidate_weight=0.10,
            bootstrap_samples=300,
        )


def test_sparse_valid_returns_retry_degenerate_bootstrap_draws() -> None:
    rng = np.random.default_rng(21)
    observations = 300
    candidate = np.zeros(observations)
    candidate[::17] = rng.normal(0.001, 0.01, len(candidate[::17]))
    sleeves = {
        "alpha": rng.normal(0.0002, 0.006, observations),
        "beta": rng.normal(0.0001, 0.007, observations),
    }
    book = 0.5 * sleeves["alpha"] + 0.5 * sleeves["beta"]
    stress = book <= np.quantile(book, 0.10)
    result = diversification_report(
        candidate,
        sleeves,
        book,
        stress_mask=stress,
        period_labels=[str(2023 + index // 100) for index in range(observations)],
        candidate_weight=0.10,
        bootstrap_samples=300,
        bootstrap_block_size=7,
    )
    assert np.isfinite(result.max_pairwise_correlation_upper_95)
    assert np.isfinite(result.max_stressed_pairwise_correlation_upper_95)
