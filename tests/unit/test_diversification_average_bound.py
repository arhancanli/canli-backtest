"""The average pairwise correlation must carry a confidence bound, resampled jointly.

The book's Sharpe ceiling is a function of the AVERAGE pairwise correlation, so that average is
the number the whole fourteen-sleeve objective turns on. Admitting a candidate on its point
estimate alone admits it on a statistic whose sampling error at 252 observations (~0.063) is
twice the size of the effect the objective distinguishes (-0.03).

The subtle part is that the bound must come from ONE index set shared across sleeves within each
bootstrap sample. Averaging the per-pair upper bounds is a different and wrong quantity. The
identity tests below hold if and only if the index set really is shared, so they fail loudly if
someone later "simplifies" this into independent per-pair draws.
"""

from __future__ import annotations

import numpy as np

from alphaforge.validation.diversification import (
    _bootstrap_average_upper,
    _bootstrap_upper,
    diversification_report,
)

SEED = 20260816
SAMPLES = 400
BLOCK = 21


def _returns(n: int = 900) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    candidate = rng.normal(0.0004, 0.01, n)
    sleeve = rng.normal(0.0003, 0.012, n)
    return candidate, sleeve


def test_one_sleeve_reduces_exactly_to_the_pairwise_bound() -> None:
    """With a single sleeve the average IS the pair, so the two paths must agree bit for bit."""
    candidate, sleeve = _returns()
    average = _bootstrap_average_upper(
        candidate,
        [sleeve],
        samples=SAMPLES,
        block_size=BLOCK,
        rng=np.random.default_rng(SEED),
    )
    pair = _bootstrap_upper(
        candidate,
        sleeve,
        samples=SAMPLES,
        block_size=BLOCK,
        rng=np.random.default_rng(SEED),
    )
    assert average == pair


def test_duplicate_sleeves_prove_the_index_set_is_shared() -> None:
    """Two identical sleeves must give the identical bound -- only true under a shared draw.

    On any resample the two pairwise correlations are equal, so their average equals either one
    and the whole bootstrap distribution is unchanged. Draw the indices independently per sleeve
    and the two correlations differ, their average has strictly lower variance, and the bound
    moves. This is the mutation detector for the shared-index property.
    """
    candidate, sleeve = _returns()
    one = _bootstrap_average_upper(
        candidate, [sleeve], samples=SAMPLES, block_size=BLOCK, rng=np.random.default_rng(SEED)
    )
    duplicated = _bootstrap_average_upper(
        candidate,
        [sleeve, sleeve.copy()],
        samples=SAMPLES,
        block_size=BLOCK,
        rng=np.random.default_rng(SEED),
    )
    assert one == duplicated


def test_bound_is_deterministic_and_exceeds_the_point_estimate() -> None:
    n = 900
    rng = np.random.default_rng(11)
    candidate = rng.normal(0.0004, 0.01, n)
    sleeves = {name: rng.normal(0.0003, 0.011, n) for name in ("a", "b", "c")}
    book = rng.normal(0.0005, 0.009, n)
    stress = np.zeros(n, dtype=bool)
    stress[:120] = True
    periods = [f"p{index // 300}" for index in range(n)]

    kwargs = {
        "stress_mask": stress,
        "period_labels": periods,
        "candidate_weight": 0.2,
        "bootstrap_samples": SAMPLES,
    }
    first = diversification_report(candidate, sleeves, book, **kwargs)
    second = diversification_report(candidate, sleeves, book, **kwargs)

    assert first.average_pairwise_correlation_upper_95 == (
        second.average_pairwise_correlation_upper_95
    )
    assert first.average_pairwise_correlation_upper_95 > first.average_pairwise_correlation
    # An upper bound on an average cannot exceed the largest per-pair upper bound by construction
    # of the underlying correlations; a value above it would mean the average is being computed
    # over something other than these pairs.
    assert first.average_pairwise_correlation_upper_95 <= first.max_pairwise_correlation_upper_95
    assert "average_pairwise_correlation_upper_95" in first.to_dict()
