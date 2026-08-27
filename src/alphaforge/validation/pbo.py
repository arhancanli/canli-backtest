"""Probability of Backtest Overfitting via CSCV (alphaDesign.md section 7.3).

Combinatorially Symmetric Cross-Validation (Bailey, Borwein, Lopez de Prado,
Zhu 2017, "The Probability of Backtest Overfitting", J. Computational Finance).
Given a performance matrix ``M`` of per-period returns with ``T`` rows
(observations on a common time grid) and ``N`` columns (strategy/config
variants - e.g. a hyperparameter or feature-selection sweep), CSCV measures how
often the variant that looks best IN-SAMPLE is mediocre-or-worse OUT-OF-SAMPLE.
That frequency IS the probability of backtest overfitting.

Exact algorithm (alphaDesign.md section 7.3, BBLZ 2017 eq. 1-5):

1. Partition the ``T`` rows into ``S`` contiguous, equal-size submatrices
   ``M_1, ..., M_S`` (``S`` even; default ``S = 16``). Trailing rows that do
   not divide evenly are dropped so every block has exactly ``T // S`` rows.
2. For each of the ``C(S, S/2)`` ways to choose half the blocks as the
   in-sample (IS) set ``J`` (the other half forming the OOS complement
   ``J_bar``): concatenate the chosen blocks into the IS matrix and the rest
   into the OOS matrix (cap at ``max_combinations`` uniformly-sampled, seeded
   combinations when ``C(S, S/2) > max_combinations``).
3. Rank the ``N`` configs by IS performance and take the IS-best config
   ``n* = argmax_n SR_IS(n)``. Performance ``SR`` of a config on a (sub)matrix
   is the per-period Sharpe ratio ``mean / std`` of that config's returns
   (population std, ddof=0; see :func:`_sharpe`).
4. Find ``n*``'s rank among the ``N`` configs OOS, ranked ASCENDING so that
   ``1 = worst`` and ``N = best`` (average ranks for ties), and form the
   relative rank::

       omega_c = rank_OOS(n*) / (N + 1)        in (0, 1)

   ``omega_c = 0.5`` is exactly the OOS median; ``omega_c -> 1`` means the
   IS-best variant was also OOS-best (no overfit); ``omega_c -> 0`` means the
   IS-best variant was OOS-worst (severe overfit).
5. Logit transform (BBLZ 2017 eq. 4)::

       lambda_c = ln( omega_c / (1 - omega_c) )

   so ``lambda_c <= 0`` exactly when the IS-best config landed at or below the
   OOS median (``omega_c <= 0.5``).
6. The probability of backtest overfitting (BBLZ 2017 eq. 5) is the fraction of
   combinations whose IS-best config was below-median OOS::

       PBO = (1 / C) * sum_c  1{ lambda_c <= 0 }

Deployment gate (alphaDesign.md section 7.3): a configuration family is eligible
for live only when ``PBO < 0.2``. This module computes the statistic; it does
NOT gate - it is an instrument of judgement, applied identically to edges and
non-edges (Phase 10 is not gated on a positive result).

Everything is pure, deterministic, and offline: the same matrix, ``n_splits``,
``max_combinations`` and ``seed`` always yield the same :class:`PBOResult`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import comb
from scipy.stats import rankdata

__all__ = ["PBOResult", "pbo_cscv"]


@dataclass(frozen=True, slots=True, kw_only=True)
class PBOResult:
    """Result of a CSCV probability-of-backtest-overfitting computation.

    Attributes:
        pbo: The probability of backtest overfitting in ``[0, 1]`` - the
            fraction of evaluated combinations whose IS-best config was at or
            below the OOS median (``lambda <= 0``). Gate: ``< 0.2`` for a config
            family to be live-eligible (alphaDesign.md section 7.3).
        lambdas: The logit ``lambda_c = ln(omega_c / (1 - omega_c))`` for every
            evaluated combination, in evaluation order; ``float64`` array of
            length :attr:`n_combinations`. The distribution (its left tail mass)
            is the PBO; its spread is the degradation diagnostic.
        is_oos_pairs: ``(n_combinations, 2)`` ``float64`` array; row ``c`` is
            ``(SR_IS(n*_c), SR_OOS(n*_c))`` - the IS-best config's in-sample and
            out-of-sample Sharpe. The scatter of OOS vs IS is the
            performance-degradation plot (BBLZ 2017 fig. 3).
        n_combinations: Number of combinations actually evaluated - either the
            full ``C(S, S/2)`` or ``max_combinations`` when the full count
            exceeds it.
    """

    pbo: float
    lambdas: npt.NDArray[np.float64]
    is_oos_pairs: npt.NDArray[np.float64]
    n_combinations: int


def _sharpe(block: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Per-period Sharpe ratio ``mean / std`` of each column of ``block``.

    ``block`` is ``(rows, N)`` of per-period returns; returns a length-``N``
    vector of ``mean_t / std_t`` per config, with population std (``ddof=0``) so
    a single-row block still yields a finite mean and a ``0`` std handled below.
    A config with zero return variance (e.g. an all-constant column) has an
    undefined Sharpe; we map ``std == 0`` to ``-inf`` so such a config can never
    be selected IS-best and always ranks worst OOS - the conservative choice
    (it carries no realized edge to overfit on).
    """
    mean = block.mean(axis=0)
    std = block.std(axis=0, ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(std > 0.0, mean / std, -np.inf)
    return np.asarray(sr, dtype=np.float64)


def pbo_cscv(
    perf_matrix: pd.DataFrame | npt.NDArray[np.float64],
    *,
    n_splits: int = 16,
    max_combinations: int = 5000,
    seed: int = 42,
) -> PBOResult:
    """Probability of Backtest Overfitting via CSCV (module docstring; section 7.3).

    Args:
        perf_matrix: ``T x N`` per-period returns - rows are observations on a
            common time grid, columns are the ``N`` strategy/config variants.
            A :class:`pandas.DataFrame` (columns are config ids) or a 2-D
            ``float`` array. Rows are partitioned in given (time) order, so the
            row order must be chronological for the IS/OOS blocks to be
            time-contiguous.
        n_splits: Number of contiguous equal submatrices ``S`` (must be even and
            ``>= 2``; default 16, giving ``C(16, 8) = 12870`` combinations).
        max_combinations: Cap on combinations evaluated. When
            ``C(S, S/2) > max_combinations``, that many distinct combinations are
            drawn uniformly without replacement using ``seed``; otherwise all
            combinations are enumerated and ``seed`` is unused.
        seed: PRNG seed for combination sampling - determinism guarantee.

    Returns:
        A :class:`PBOResult`.

    Raises:
        ValueError: ``n_splits`` not an even integer ``>= 2``; ``max_combinations
            < 1``; fewer than 2 config columns (a single config has no
            cross-sectional rank); or fewer than ``n_splits`` observation rows
            (cannot form ``S`` non-empty blocks).
    """
    if n_splits < 2 or n_splits % 2 != 0:
        raise ValueError(f"n_splits must be an even integer >= 2; got {n_splits}")
    if max_combinations < 1:
        raise ValueError(f"max_combinations must be >= 1; got {max_combinations}")

    if isinstance(perf_matrix, pd.DataFrame):
        matrix = perf_matrix.to_numpy(dtype=np.float64)
    else:
        matrix = np.asarray(perf_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"perf_matrix must be 2-D (T x N); got ndim={matrix.ndim}")

    n_rows, n_configs = matrix.shape
    if n_configs < 2:
        raise ValueError(
            f"perf_matrix needs >= 2 config columns to rank cross-sectionally; got {n_configs}"
        )
    if n_rows < n_splits:
        raise ValueError(
            f"perf_matrix has {n_rows} rows < n_splits={n_splits}; "
            "cannot form that many non-empty contiguous blocks"
        )

    # Step 1: contiguous equal blocks; drop the trailing remainder so every
    # block has exactly block_len rows (BBLZ require equal partitions).
    block_len = n_rows // n_splits
    usable = block_len * n_splits
    blocks = matrix[:usable].reshape(n_splits, block_len, n_configs)

    # Step 2: choose which combinations of half the blocks form the IS set.
    half = n_splits // 2
    total = int(comb(n_splits, half, exact=True))
    rng = np.random.default_rng(seed)
    block_indices = np.arange(n_splits)
    combos: list[tuple[int, ...]]
    if total > max_combinations:
        # Sample distinct combinations uniformly without replacement, seeded.
        seen: set[tuple[int, ...]] = set()
        while len(seen) < max_combinations:
            choice = tuple(sorted(rng.choice(block_indices, size=half, replace=False).tolist()))
            seen.add(choice)
        combos = sorted(seen)
    else:
        combos = list(combinations(range(n_splits), half))

    n_combinations = len(combos)
    lambdas = np.empty(n_combinations, dtype=np.float64)
    is_oos_pairs = np.empty((n_combinations, 2), dtype=np.float64)
    all_blocks = set(range(n_splits))

    for c, is_blocks in enumerate(combos):
        oos_blocks = sorted(all_blocks - set(is_blocks))
        # Concatenate the chosen blocks (time order within IS/OOS is irrelevant
        # to a mean/std Sharpe, but we keep block order for clarity).
        is_mat = blocks[list(is_blocks)].reshape(-1, n_configs)
        oos_mat = blocks[oos_blocks].reshape(-1, n_configs)

        # Step 3: IS-best config.
        sr_is = _sharpe(is_mat)
        best = int(np.argmax(sr_is))

        # Step 4: OOS rank of the IS-best config, ascending (1 = worst), average
        # ties; relative rank omega in (0, 1).
        sr_oos = _sharpe(oos_mat)
        oos_ranks = rankdata(sr_oos, method="average")  # 1..N ascending
        omega = float(oos_ranks[best]) / (n_configs + 1.0)

        # Step 5: logit. omega is strictly inside (0, 1) since 1 <= rank <= N
        # and the denominator is N + 1, so the log is always finite.
        lambdas[c] = float(np.log(omega / (1.0 - omega)))
        is_oos_pairs[c, 0] = float(sr_is[best])
        is_oos_pairs[c, 1] = float(sr_oos[best])

    # Step 6: PBO = fraction with lambda <= 0 (IS-best at/below OOS median).
    pbo = float(np.mean(lambdas <= 0.0))

    return PBOResult(
        pbo=pbo,
        lambdas=lambdas,
        is_oos_pairs=is_oos_pairs,
        n_combinations=n_combinations,
    )
