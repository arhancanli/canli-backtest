"""Deterministic diversification evidence for ALPHAC sleeve admission.

Inputs are pre-aligned OOS simple returns. This module never joins, drops missing rows, discovers
a stress regime, or chooses a weight. Correlation uncertainty uses circular moving-block bootstrap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["DiversificationReport", "diversification_report"]


@dataclass(frozen=True, slots=True)
class DiversificationReport:
    observations: int
    stressed_observations: int
    candidate_weight: float
    pairwise_correlations: dict[str, float]
    stressed_pairwise_correlations: dict[str, float]
    pairwise_correlation_upper_95: dict[str, float]
    stressed_pairwise_correlation_upper_95: dict[str, float]
    average_pairwise_correlation: float
    average_pairwise_correlation_upper_95: float
    max_pairwise_correlation: float
    max_stressed_pairwise_correlation: float
    max_pairwise_correlation_upper_95: float
    max_stressed_pairwise_correlation_upper_95: float
    candidate_mean_on_book_es_days: float
    stressed_joint_loss_rate: float
    book_sharpe_delta: float
    minimum_leave_one_period_out_book_sharpe_delta: float
    leave_one_period_out_book_sharpe_deltas: dict[str, float]
    book_max_drawdown_delta: float
    book_expected_shortfall_delta: float
    bootstrap_samples: int
    bootstrap_block_size: int
    bootstrap_seed: int
    return_data_opened: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _vector(name: str, values: ArrayLike, expected: int | None = None) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if expected is not None and vector.size != expected:
        raise ValueError(f"{name} length {vector.size} != {expected}")
    if vector.size < 3:
        raise ValueError(f"{name} requires at least three observations")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains missing or non-finite observations")
    if np.any(vector <= -1.0):
        raise ValueError(f"{name} contains a simple return at or below -100%")
    return vector


def _correlation(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    if left.size < 3 or np.std(left) == 0.0 or np.std(right) == 0.0:
        raise ValueError("correlation requires three non-constant paired observations")
    value = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(value):
        raise ValueError("correlation is non-finite")
    return value


def _bootstrap_upper(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    *,
    samples: int,
    block_size: int,
    rng: np.random.Generator,
) -> float:
    if samples < 200:
        raise ValueError("bootstrap_samples must be at least 200")
    if not 1 <= block_size <= left.size:
        raise ValueError("bootstrap_block_size must be between one and sample length")
    block_count = int(np.ceil(left.size / block_size))
    offsets = np.arange(block_size, dtype=np.int64)
    estimates = np.empty(samples, dtype=np.float64)
    accepted = 0
    attempts = 0
    max_attempts = samples * 20
    while accepted < samples:
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(
                "bootstrap could not draw enough non-constant paired resamples"
            )
        starts = rng.integers(0, left.size, size=block_count)
        indices = ((starts[:, None] + offsets) % left.size).reshape(-1)[: left.size]
        try:
            estimate = _correlation(left[indices], right[indices])
        except ValueError:
            continue
        estimates[accepted] = estimate
        accepted += 1
    return float(np.quantile(estimates, 0.95))


def _bootstrap_average_upper(
    candidate: NDArray[np.float64],
    sleeves: Sequence[NDArray[np.float64]],
    *,
    samples: int,
    block_size: int,
    rng: np.random.Generator,
) -> float:
    """One-sided upper 95% bound on the AVERAGE pairwise correlation.

    The average pairwise correlation is the quantity the book's Sharpe ceiling is a function of
    (``s_bar / sqrt(rho_bar)``), so admitting a candidate on its point estimate alone admits it on
    a number whose standard error can exceed the effect being gated: at 252 observations the
    sampling error of a correlation is about 0.063, while the correlation objective that separates
    a reachable book from an unreachable one is -0.03. This bound is what makes the average
    gateable rather than merely reportable.

    EVERY SLEEVE IS RESAMPLED ON THE SAME DRAWN INDEX SET within a bootstrap sample. That is what
    makes this the bootstrap distribution *of the average*: it preserves the cross-sleeve
    dependence that the average is taken over. Averaging the per-pair upper bounds returned by
    ``_bootstrap_upper`` would NOT give this number and must not be substituted for it -- the mean
    of upper bounds is the upper bound of no statistic at all, and because averaging reduces
    variance it is systematically too wide, which would reject candidates the evidence supports.

    Called after the per-pair loop so the shared ``rng`` stream reaching the existing
    ``_bootstrap_upper`` calls is byte-for-byte unchanged by this addition.
    """
    if samples < 200:
        raise ValueError("bootstrap_samples must be at least 200")
    size = candidate.size
    if not 1 <= block_size <= size:
        raise ValueError("bootstrap_block_size must be between one and sample length")
    if not sleeves:
        raise ValueError("at least one current sleeve is required")
    block_count = int(np.ceil(size / block_size))
    offsets = np.arange(block_size, dtype=np.int64)
    estimates = np.empty(samples, dtype=np.float64)
    accepted = 0
    attempts = 0
    max_attempts = samples * 20
    while accepted < samples:
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(
                "average bootstrap could not draw enough non-constant resamples"
            )
        starts = rng.integers(0, size, size=block_count)
        indices = ((starts[:, None] + offsets) % size).reshape(-1)[:size]
        drawn = candidate[indices]
        try:
            estimate = float(
                np.mean([_correlation(drawn, sleeve[indices]) for sleeve in sleeves])
            )
        except ValueError:
            continue
        estimates[accepted] = estimate
        accepted += 1
    return float(np.quantile(estimates, 0.95))


def _sharpe(returns: NDArray[np.float64], annualization: int) -> float:
    deviation = float(np.std(returns, ddof=1))
    if deviation == 0.0:
        raise ValueError("Sharpe ratio is undefined for constant returns")
    return float(np.mean(returns) / deviation * np.sqrt(annualization))


def _max_drawdown(returns: NDArray[np.float64]) -> float:
    wealth = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[:-1]
    return float(np.min(wealth / peaks - 1.0))


def _expected_shortfall(returns: NDArray[np.float64], alpha: float) -> float:
    count = max(1, int(np.ceil(alpha * returns.size)))
    return float(np.mean(np.partition(returns, count - 1)[:count]))


def diversification_report(
    candidate_returns: ArrayLike,
    sleeve_returns: Mapping[str, ArrayLike],
    book_returns: ArrayLike,
    *,
    stress_mask: ArrayLike,
    period_labels: Sequence[str],
    candidate_weight: float,
    annualization: int = 252,
    expected_shortfall_alpha: float = 0.05,
    bootstrap_samples: int = 2_000,
    bootstrap_block_size: int = 21,
    bootstrap_seed: int = 20260816,
) -> DiversificationReport:
    """Compute one frozen candidate's aligned, confidence-bound diversification evidence."""
    candidate = _vector("candidate_returns", candidate_returns)
    observations = int(candidate.size)
    book = _vector("book_returns", book_returns, observations)
    if not sleeve_returns:
        raise ValueError("at least one current sleeve is required")
    sleeves = {
        str(name): _vector(f"sleeve_returns[{name}]", values, observations)
        for name, values in sleeve_returns.items()
    }
    if len(period_labels) != observations:
        raise ValueError("period_labels must align exactly with returns")
    periods = np.asarray(period_labels, dtype=str)
    unique_periods = sorted(set(periods.tolist()))
    if len(unique_periods) < 2 or any(not period for period in unique_periods):
        raise ValueError("at least two non-empty predeclared period labels are required")
    stress = np.asarray(stress_mask)
    if stress.ndim != 1 or stress.size != observations or stress.dtype != np.bool_:
        raise ValueError("stress_mask must be an exactly aligned boolean vector")
    stressed_observations = int(np.sum(stress))
    if stressed_observations < 3:
        raise ValueError("stress_mask requires at least three observations")
    if not 0.0 < candidate_weight < 1.0:
        raise ValueError("candidate_weight must be strictly between zero and one")
    if annualization <= 0:
        raise ValueError("annualization must be positive")
    if not 0.0 < expected_shortfall_alpha < 0.5:
        raise ValueError("expected_shortfall_alpha must be between zero and 0.5")

    ordinary: dict[str, float] = {}
    stressed: dict[str, float] = {}
    ordinary_upper: dict[str, float] = {}
    stressed_upper: dict[str, float] = {}
    rng = np.random.default_rng(bootstrap_seed)
    stressed_block_size = min(bootstrap_block_size, stressed_observations)
    for name in sorted(sleeves):
        sleeve = sleeves[name]
        ordinary[name] = _correlation(candidate, sleeve)
        stressed[name] = _correlation(candidate[stress], sleeve[stress])
        ordinary_upper[name] = _bootstrap_upper(
            candidate,
            sleeve,
            samples=bootstrap_samples,
            block_size=bootstrap_block_size,
            rng=rng,
        )
        stressed_upper[name] = _bootstrap_upper(
            candidate[stress],
            sleeve[stress],
            samples=bootstrap_samples,
            block_size=stressed_block_size,
            rng=rng,
        )

    average_upper = _bootstrap_average_upper(
        candidate,
        [sleeves[name] for name in sorted(sleeves)],
        samples=bootstrap_samples,
        block_size=bootstrap_block_size,
        rng=rng,
    )

    blended = (1.0 - candidate_weight) * book + candidate_weight * candidate
    book_sharpe = _sharpe(book, annualization)
    blended_sharpe = _sharpe(blended, annualization)
    leave_out: dict[str, float] = {}
    for period in unique_periods:
        retained = periods != period
        if int(np.sum(retained)) < 3:
            raise ValueError(f"leaving period {period!r} out leaves fewer than three observations")
        leave_out[period] = _sharpe(blended[retained], annualization) - _sharpe(
            book[retained], annualization
        )

    es_count = max(1, int(np.ceil(expected_shortfall_alpha * observations)))
    book_es_indices = np.argpartition(book, es_count - 1)[:es_count]
    joint_loss_rate = float(np.mean((candidate[stress] < 0.0) & (book[stress] < 0.0)))
    return DiversificationReport(
        observations=observations,
        stressed_observations=stressed_observations,
        candidate_weight=candidate_weight,
        pairwise_correlations=ordinary,
        stressed_pairwise_correlations=stressed,
        pairwise_correlation_upper_95=ordinary_upper,
        stressed_pairwise_correlation_upper_95=stressed_upper,
        average_pairwise_correlation=float(np.mean(list(ordinary.values()))),
        average_pairwise_correlation_upper_95=average_upper,
        max_pairwise_correlation=max(ordinary.values()),
        max_stressed_pairwise_correlation=max(stressed.values()),
        max_pairwise_correlation_upper_95=max(ordinary_upper.values()),
        max_stressed_pairwise_correlation_upper_95=max(stressed_upper.values()),
        candidate_mean_on_book_es_days=float(np.mean(candidate[book_es_indices])),
        stressed_joint_loss_rate=joint_loss_rate,
        book_sharpe_delta=blended_sharpe - book_sharpe,
        minimum_leave_one_period_out_book_sharpe_delta=min(leave_out.values()),
        leave_one_period_out_book_sharpe_deltas=leave_out,
        book_max_drawdown_delta=_max_drawdown(blended) - _max_drawdown(book),
        book_expected_shortfall_delta=_expected_shortfall(
            blended, expected_shortfall_alpha
        )
        - _expected_shortfall(book, expected_shortfall_alpha),
        bootstrap_samples=bootstrap_samples,
        bootstrap_block_size=bootstrap_block_size,
        bootstrap_seed=bootstrap_seed,
    )
