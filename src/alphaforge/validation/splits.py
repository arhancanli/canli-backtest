"""THE purged walk-forward splitter (alphaDesign.md §7.1; leakageCritique.md finding 16).

This is the ONE splitter library in AlphaForge. The Phase-6 walk-forward
runner consumes it for leg layout; Phase 9/10 model cross-validation imports
:class:`PurgedWalkForward` from here — a second splitter implementation
anywhere else is a leakage bug by decree (finding 16: sklearn
``TimeSeriesSplit`` and hand-rolled fold loops are how labels bleed into
training sets).

Definitions, on a uniform decision-bar grid ``t_0 < t_1 < ... < t_{n-1}``
with spacing ``Δ`` (asserted — purge/embargo are *bar counts*, so the
position arithmetic below is exact only on a uniform grid):

* A sample decided at grid point ``t`` carries the label interval
  ``[t, t + purge_bars·Δ]`` — the execution-aware forward return
  ``y_h(t)`` is not fully realized until ``purge_bars >= h`` bars later
  (alphaDesign.md §5.1), which is why ``purge_bars`` must be at least the
  label horizon ``h = settings.signals.horizon_bars`` (THE single horizon
  constant, buildabilityCritique.md §3.10 — :meth:`PurgedWalkForward.
  from_settings` wires it and the constructor refuses anything smaller).

* **Purge** (alphaDesign.md §7.1): drop every train sample whose label
  interval overlaps the test span ``[T_a, T_b]`` — for a train sample at
  ``t < T_a`` that is exactly ``t + purge_bars·Δ >= T_a``.

* **Embargo**: additionally drop train samples with
  ``t in [T_b + Δ, T_b + embargo_bars·Δ]`` — train decisions *after* the
  test block whose features still carry serially-correlated test-period
  information. In forward-chaining walk-forward the train window always
  precedes the test span, so the embargo never bites here; it is implemented
  (not merely documented) so the identical purge/embargo arithmetic is
  reused verbatim when CombinatorialPurgedCV arrives in the ML phase, where
  train blocks DO follow test blocks. Default ``embargo_bars = 168`` (7d
  > 2x the 72-bar max label horizon, headroom for feature serial
  correlation).

Split layout (forward-chaining, rolling train window):

* Split ``k`` tests grid positions ``[train_bars + k·test_bars,
  train_bars + (k+1)·test_bars)`` — the final split may be shorter (the
  grid tail). Test spans therefore TILE ``[t_{train_bars}, t_{n-1}]``
  without overlap or gap.
* Train candidates are the rolling ``train_bars`` positions immediately
  before the test span; purging then removes the last ``purge_bars`` of
  them, leaving exactly ``train_bars - purge_bars`` train samples per split.

Everything is pure and deterministic: the same grid yields the same splits,
always.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Self

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from alphaforge.config.settings import Settings
    from alphaforge.core.time import Ms

__all__ = ["PurgedWalkForward"]


class PurgedWalkForward:
    """Forward-chaining walk-forward splits with purging + embargo (module docstring).

    Args:
        train_bars: Rolling train-window length in grid points (pre-purge).
        test_bars: Test-span length in grid points; splits roll forward by
            this amount, so consecutive test spans tile the grid.
        purge_bars: Label-interval length in grid points; the last
            ``purge_bars`` train candidates before each test span are
            dropped. MUST be >= the label horizon ``h`` — pass
            ``horizon_bars`` (or use :meth:`from_settings`) and the
            constructor enforces it.
        embargo_bars: Train samples within this many grid points AFTER a
            test span are dropped (CPCV reuse; a no-op for forward
            chaining — see module docstring). Default 168.
        horizon_bars: The label horizon ``h`` this splitter must purge for
            (``settings.signals.horizon_bars``). When provided,
            ``purge_bars < horizon_bars`` raises — an under-purged splitter
            leaks label information into training and must never construct.

    Raises:
        ValueError: non-positive ``train_bars``/``test_bars``, negative
            ``purge_bars``/``embargo_bars``, ``purge_bars >= train_bars``
            (no train sample would ever survive), or
            ``purge_bars < horizon_bars``.
    """

    __slots__ = ("embargo_bars", "purge_bars", "test_bars", "train_bars")

    def __init__(
        self,
        train_bars: int,
        test_bars: int,
        *,
        purge_bars: int,
        embargo_bars: int = 168,
        horizon_bars: int | None = None,
    ) -> None:
        if train_bars <= 0 or test_bars <= 0:
            raise ValueError(f"train_bars/test_bars must be > 0, got {train_bars}/{test_bars}")
        if purge_bars < 0 or embargo_bars < 0:
            raise ValueError(
                f"purge_bars/embargo_bars must be >= 0, got {purge_bars}/{embargo_bars}"
            )
        if purge_bars >= train_bars:
            raise ValueError(
                f"purge_bars ({purge_bars}) must be < train_bars ({train_bars}); "
                "otherwise every train sample is purged and the splits are empty"
            )
        if horizon_bars is not None and purge_bars < horizon_bars:
            raise ValueError(
                f"purge_bars ({purge_bars}) must be >= the label horizon h = "
                f"{horizon_bars} (settings.signals.horizon_bars): a train sample "
                "decided inside the last h bars before the test span has a label "
                "that overlaps it — training on it is leakage (alphaDesign §7.1)"
            )
        self.train_bars = int(train_bars)
        self.test_bars = int(test_bars)
        self.purge_bars = int(purge_bars)
        self.embargo_bars = int(embargo_bars)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        train_bars: int,
        test_bars: int,
        embargo_bars: int = 168,
    ) -> Self:
        """The blessed constructor: ``purge_bars = settings.signals.horizon_bars``
        — the ONE shared horizon constant (buildability §3.10), asserted, never
        duplicated as a literal at a call site."""
        h = settings.signals.horizon_bars
        return cls(
            train_bars,
            test_bars,
            purge_bars=h,
            embargo_bars=embargo_bars,
            horizon_bars=h,
        )

    def n_splits(self, n_grid: int) -> int:
        """Number of splits :meth:`split` will yield over a grid of ``n_grid`` points."""
        remaining = n_grid - self.train_bars
        if remaining <= 0:
            return 0
        return -(-remaining // self.test_bars)  # ceil division

    def split(
        self, ts_grid: Sequence[Ms] | npt.NDArray[np.int64]
    ) -> Iterator[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
        """Yield ``(train_ts, test_ts)`` epoch-ms arrays per split (module docstring).

        ``ts_grid`` is the full decision-bar grid (epoch-ms UTC, strictly
        increasing). Spacing need NOT be uniform: purge/embargo are bar COUNTS
        applied positionally (grid-index offsets), so the splitter is correct on
        both the contiguous H1 crypto grid and the gappy XNYS session grid (a bar
        is a session for D1). Yields arrays of grid *timestamps* (int64 copies).

        Raises:
            ValueError: empty/non-1-D/non-increasing grid, or a grid too short for
                even one split (``len(ts_grid) <= train_bars``).
        """
        grid = np.asarray(ts_grid, dtype=np.int64)
        if grid.ndim != 1:
            raise ValueError(f"ts_grid must be 1-D, got ndim={grid.ndim}")
        n = int(grid.size)
        if n <= self.train_bars:
            raise ValueError(
                f"ts_grid has {n} points <= train_bars={self.train_bars}; "
                "not a single test span fits"
            )
        spacing = np.diff(grid)
        if not bool(np.all(spacing > 0)):
            raise ValueError("ts_grid must be strictly increasing")
        # No uniform-spacing requirement: the split below is PURELY POSITIONAL (purge/embargo
        # are bar COUNTS applied as grid-index offsets), so it is correct on any strictly
        # increasing decision grid — the contiguous H1 crypto grid AND the gappy XNYS session
        # grid alike (a "bar" is a session for D1). On a uniform grid the result is identical
        # to before (the assertion only ever passed there); on a session grid the bar counts
        # are session counts, which is exactly the intended purge/embargo semantics.

        for k in range(self.n_splits(n)):
            a = self.train_bars + k * self.test_bars  # test start position
            b = min(a + self.test_bars, n)  # test end position (exclusive)
            candidates = np.arange(a - self.train_bars, a, dtype=np.int64)
            # Purge: label interval [p, p + purge_bars] overlaps [a, b-1].
            purged = (candidates + self.purge_bars >= a) & (candidates <= b - 1)
            # Embargo: train decision inside (b-1, b-1 + embargo] after the test
            # block (general form for CPCV reuse; vacuous when train < test).
            embargoed = (candidates >= b) & (candidates < b + self.embargo_bars)
            train_pos = candidates[~(purged | embargoed)]
            yield grid[train_pos].copy(), grid[a:b].copy()
