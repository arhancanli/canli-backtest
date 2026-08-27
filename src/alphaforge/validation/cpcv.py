"""Combinatorially Purged Cross-Validation (alphaDesign.md §7, the
``CombinatorialPurgedCV`` spec; Lopez de Prado, *Advances in Financial Machine
Learning* ch. 12, "The Combinatorial Purged Cross-Validation Method").

CPCV exists to produce a *distribution* of out-of-sample (OOS) paths for a
selection statistic (Sharpe, IC, ...), not a single point estimate. The dense
forward-chaining :class:`~alphaforge.validation.splits.PurgedWalkForward` stays
the deployment-faithful FINAL evaluation: it tests strictly forward in time the
way the strategy will actually trade. CPCV is the *research-protocol* sibling
used to judge whether an edge survives across many resamplings of the same
history (alphaDesign.md §7.1 research protocol) and feeds the path Sharpe
variance ``V[SR]`` consumed by the Deflated Sharpe Ratio (§7.4).

Construction (uniform decision-bar grid ``t_0 < t_1 < ... < t_{n-1}`` of
``n`` positions, spacing ``Δ`` -- asserted, because purge/embargo are *bar
counts* and equal time offsets only on a uniform grid):

* Partition the ``n`` grid positions into ``N = n_groups`` CONTIGUOUS groups.
  Group sizes are as equal as possible; when ``N`` does not divide ``n`` the
  first ``n mod N`` groups get one extra position (the ``numpy.array_split``
  convention). The groups TILE ``[t_0, t_{n-1}]`` exactly once.

* Each split chooses ``k = k_test`` of the ``N`` groups as the test set,
  enumerated by :func:`itertools.combinations` in lexicographic group-index
  order -- so there are exactly ``C(N, k)`` splits and the enumeration is
  deterministic. A test set is, in general, several DISJOINT contiguous blocks.

* Train = the remaining ``N - k`` groups' positions, MINUS purged samples,
  MINUS embargoed samples, with respect to EVERY test block:

  - **Purge** (alphaDesign.md §7.1): drop a train position ``p`` whose label
    interval ``[p, p + purge_bars·Δ]`` overlaps a test block spanning grid
    positions ``[a, b)`` -- i.e. ``p + purge_bars >= a`` and ``p <= b - 1``.
    Because test blocks can lie on either side of a train group, purging binds
    in BOTH directions here (unlike forward-chaining WF where train precedes
    test and only the trailing edge is purged).
  - **Embargo**: additionally drop a train position ``p`` in
    ``[b, b + embargo_bars)`` immediately AFTER a test block -- train decisions
    whose features still carry serially-correlated test-period information.
    Default ``embargo_bars = 168`` (7d > 2x the 72-bar max label horizon).

The purge/embargo boolean arithmetic is the SAME contract implemented in
:class:`~alphaforge.validation.splits.PurgedWalkForward.split`
(``candidate + purge_bars >= a) & (candidate <= b - 1`` for purge;
``(candidate >= b) & (candidate < b + embargo_bars)`` for embargo). It is shared
here through :func:`_purge_embargo_mask` and exercised against
``PurgedWalkForward`` for equivalence in the tests, rather than duplicated as
loose copy-paste.

Path accounting (alphaDesign.md §7): across all ``C(N, k)`` splits every group
is used as a test group in exactly ``C(N-1, k-1) = k·C(N, k)/N`` splits, so the
per-group OOS predictions assemble into

    phi = n_backtest_paths = k · C(N, k) / N

complete, non-overlapping backtest PATHS over the full history -- the
distribution of OOS statistics the protocol selects on. (Default N=10, k=2:
C(10,2)=45 splits, phi=9 paths.)

Everything is pure and deterministic: the same grid yields the same splits in
the same order, always.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import combinations
from math import comb
from typing import TYPE_CHECKING, Self

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from alphaforge.config.settings import Settings
    from alphaforge.core.time import Ms

__all__ = ["CombinatorialPurgedCV"]


def _purge_embargo_mask(
    candidates: npt.NDArray[np.int64],
    test_start: int,
    test_end_excl: int,
    *,
    purge_bars: int,
    embargo_bars: int,
) -> npt.NDArray[np.bool_]:
    """Positions to DROP for one test block ``[test_start, test_end_excl)``.

    The exact purge/embargo predicate of
    :meth:`alphaforge.validation.splits.PurgedWalkForward.split`, lifted so CPCV
    reuses it verbatim instead of re-deriving it:

    * purge: label interval ``[p, p + purge_bars]`` overlaps the test block
      ``<=> (p + purge_bars >= test_start) and (p <= test_end_excl - 1)``;
    * embargo: ``p in [test_end_excl, test_end_excl + embargo_bars)``.

    All quantities are GRID POSITIONS (bar counts), not timestamps.
    """
    purged = (candidates + purge_bars >= test_start) & (candidates <= test_end_excl - 1)
    embargoed = (candidates >= test_end_excl) & (candidates < test_end_excl + embargo_bars)
    return purged | embargoed


class CombinatorialPurgedCV:
    """Combinatorial purged cross-validation over a uniform bar grid (module docstring).

    Args:
        n_groups: Number ``N`` of contiguous groups the grid is partitioned
            into. Must be ``>= 2`` and ``> k_test`` (at least one train group
            must remain).
        k_test: Number ``k`` of groups used as the test set per split. Must be
            ``>= 1`` and ``< n_groups``.
        purge_bars: Label-interval length in grid points; a train position
            whose interval ``[p, p + purge_bars]`` touches any test block is
            dropped. MUST be ``>= horizon_bars`` when the latter is given.
        embargo_bars: Train positions within this many grid points AFTER a test
            block are dropped. Default 168.
        horizon_bars: The label horizon ``h`` (``settings.signals.horizon_bars``)
            this splitter must purge for. When provided,
            ``purge_bars < horizon_bars`` raises -- an under-purged CV leaks
            label information into training (alphaDesign.md §7.1).

    Raises:
        ValueError: ``n_groups < 2``; ``k_test`` not in ``[1, n_groups)``;
            negative ``purge_bars``/``embargo_bars``; or
            ``purge_bars < horizon_bars``.
    """

    __slots__ = ("embargo_bars", "k_test", "n_groups", "purge_bars")

    def __init__(
        self,
        n_groups: int = 10,
        k_test: int = 2,
        *,
        purge_bars: int,
        embargo_bars: int = 168,
        horizon_bars: int | None = None,
    ) -> None:
        if n_groups < 2:
            raise ValueError(f"n_groups must be >= 2, got {n_groups}")
        if not 1 <= k_test < n_groups:
            raise ValueError(
                f"k_test must satisfy 1 <= k_test < n_groups ({n_groups}), got {k_test}; "
                "at least one train group must remain"
            )
        if purge_bars < 0 or embargo_bars < 0:
            raise ValueError(
                f"purge_bars/embargo_bars must be >= 0, got {purge_bars}/{embargo_bars}"
            )
        if horizon_bars is not None and purge_bars < horizon_bars:
            raise ValueError(
                f"purge_bars ({purge_bars}) must be >= the label horizon h = "
                f"{horizon_bars} (settings.signals.horizon_bars): a train sample "
                "decided inside the last h bars before a test block has a label "
                "that overlaps it -- training on it is leakage (alphaDesign §7.1)"
            )
        self.n_groups = int(n_groups)
        self.k_test = int(k_test)
        self.purge_bars = int(purge_bars)
        self.embargo_bars = int(embargo_bars)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        n_groups: int = 10,
        k_test: int = 2,
        embargo_bars: int = 168,
    ) -> Self:
        """The blessed constructor: ``purge_bars = settings.signals.horizon_bars``
        -- the ONE shared horizon constant (buildability §3.10), asserted, never
        duplicated as a literal at a call site."""
        h = settings.signals.horizon_bars
        return cls(
            n_groups,
            k_test,
            purge_bars=h,
            embargo_bars=embargo_bars,
            horizon_bars=h,
        )

    @property
    def n_splits(self) -> int:
        """Number of splits :meth:`split` yields: ``C(n_groups, k_test)``."""
        return comb(self.n_groups, self.k_test)

    @property
    def n_backtest_paths(self) -> int:
        """Number of complete OOS backtest paths: ``phi = k·C(N, k)/N``.

        Each group is a test group in exactly ``C(N-1, k-1) = k·C(N, k)/N``
        splits; concatenating one test-group prediction per path over all
        groups yields that many full-history OOS paths. The value is always an
        integer (``= C(n_groups - 1, k_test - 1)``).
        """
        return comb(self.n_groups - 1, self.k_test - 1)

    @staticmethod
    def _group_bounds(n: int, n_groups: int) -> list[tuple[int, int]]:
        """Half-open ``[start, end)`` grid-position bounds of each contiguous group.

        Sizes follow the ``numpy.array_split`` convention: the first
        ``n mod n_groups`` groups get ``ceil(n / n_groups)`` positions, the rest
        get ``floor``. Groups tile ``[0, n)`` exactly once with no gaps.
        """
        base, extra = divmod(n, n_groups)
        bounds: list[tuple[int, int]] = []
        start = 0
        for g in range(n_groups):
            size = base + (1 if g < extra else 0)
            bounds.append((start, start + size))
            start += size
        return bounds

    def split(
        self, ts_grid: Sequence[Ms] | npt.NDArray[np.int64]
    ) -> Iterator[tuple[npt.NDArray[np.int64], tuple[npt.NDArray[np.int64], ...]]]:
        """Yield ``(train_ts, test_ts_groups)`` per combinatorial split (module docstring).

        ``ts_grid`` is the full decision-bar grid (epoch-ms UTC, strictly
        increasing, UNIFORM spacing -- asserted). For each of the ``C(N, k)``
        test-group combinations (lexicographic group-index order) yields:

        * ``train_ts``: int64 epoch-ms of the surviving train positions
          (remaining groups minus purge minus embargo), ascending;
        * ``test_ts_groups``: a tuple of ``k_test`` int64 epoch-ms arrays, one
          per chosen test group, in ascending group order -- the per-group
          structure is preserved so each group can be routed to its backtest
          path.

        All arrays are fresh copies (safe to mutate). The minimum grid length
        is ``n_groups`` (one position per group); shorter grids would yield
        empty groups.

        Raises:
            ValueError: empty / non-1-D / non-increasing / non-uniform grid, or
                ``len(ts_grid) < n_groups``.
        """
        grid = np.asarray(ts_grid, dtype=np.int64)
        if grid.ndim != 1:
            raise ValueError(f"ts_grid must be 1-D, got ndim={grid.ndim}")
        n = int(grid.size)
        if n < self.n_groups:
            raise ValueError(
                f"ts_grid has {n} points < n_groups={self.n_groups}; "
                "every group must hold at least one bar"
            )
        spacing = np.diff(grid)
        if not bool(np.all(spacing > 0)):
            raise ValueError("ts_grid must be strictly increasing")
        if n >= 2 and not bool(np.all(spacing == spacing[0])):
            raise ValueError(
                "ts_grid must be uniformly spaced (purge/embargo are bar counts; "
                f"got spacings {sorted(set(spacing.tolist()))})"
            )

        bounds = self._group_bounds(n, self.n_groups)
        all_groups = range(self.n_groups)

        for test_ids in combinations(all_groups, self.k_test):
            test_set = set(test_ids)
            test_blocks = [bounds[g] for g in test_ids]  # ascending group order

            # Candidate train positions: every position in the NON-test groups.
            train_pos_chunks = [
                np.arange(lo, hi, dtype=np.int64)
                for g, (lo, hi) in enumerate(bounds)
                if g not in test_set
            ]
            candidates = (
                np.concatenate(train_pos_chunks)
                if train_pos_chunks
                else np.empty(0, dtype=np.int64)
            )

            # Drop = union of purge+embargo over every test block.
            drop = np.zeros(candidates.shape, dtype=np.bool_)
            for a, b in test_blocks:
                drop |= _purge_embargo_mask(
                    candidates,
                    a,
                    b,
                    purge_bars=self.purge_bars,
                    embargo_bars=self.embargo_bars,
                )
            train_pos = candidates[~drop]

            test_ts_groups = tuple(grid[a:b].copy() for a, b in test_blocks)
            yield grid[train_pos].copy(), test_ts_groups
