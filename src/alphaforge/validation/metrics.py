"""IC metrics with overlap-honest inference (alphaDesign.md §7.2).

Per-timestamp Spearman rank IC of a factor against execution-aware forward returns,
summarized with a Newey-West HAC t-statistic. Because an ``h``-bar label observed on
a 1-bar grid overlaps its ``h - 1`` neighbours, the dense IC series is serially
correlated by construction; honest inference either subsamples to a non-overlapping
grid (:func:`non_overlapping`) or uses the HAC t-stat with ``lags = h``
(:func:`newey_west_tstat`). RankIC is primary — robust to crypto tails.

Direction convention: factors are **direction-adjusted upstream** (multiplied by
``FeatureSpec.direction``) before being passed to :func:`rank_ic`, so a healthy
alpha always shows a *positive* IC here; this module never re-signs anything.

Universe masking: IC at time T is computed only over the point-in-time universe
members at T — evaluating a factor on names selected with future information is
selection bias. The mask must come from ``UniverseStore.membership_asof``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "ICSummary",
    "ic_summary",
    "newey_west_tstat",
    "non_overlapping",
    "rank_ic",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ICSummary:
    """Summary statistics of a per-timestamp IC series.

    - ``n``: number of finite IC observations
    - ``mean`` / ``std``: sample mean and std (ddof=1) of IC_t
    - ``t_nw``: Newey-West HAC t-statistic of the mean (see :func:`newey_west_tstat`)
    - ``hit_rate``: share of timestamps with IC_t > 0
    - ``ic_ir``: mean / std (information ratio of the IC series, per observation)
    - ``by_year``: calendar-year (UTC) → mean IC, for stability inspection
    """

    n: int
    mean: float
    std: float
    t_nw: float
    hit_rate: float
    ic_ir: float
    by_year: Mapping[int, float]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation = Pearson correlation of average-tie ranks."""
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    dx = rx - rx.mean()
    dy = ry - ry.mean()
    denom = float(np.sqrt((dx @ dx) * (dy @ dy)))
    if denom == 0.0:  # a constant ranking carries no cross-sectional information
        return float("nan")
    return float((dx @ dy) / denom)


def rank_ic(
    factor: pd.Series,
    fwd: pd.Series,
    universe_mask: pd.Series,
    *,
    min_members: int = 5,
) -> pd.Series:
    """Per-timestamp Spearman rank IC of ``factor`` vs ``fwd`` over universe members.

    For each ``ts_open`` group of ``factor``'s ``(ts_open, instrument_id)``
    MultiIndex::

        RankIC_t = SpearmanCorr_i( factor_{i,t}, fwd_{i,t} )

    computed over rows that are (a) universe members per ``universe_mask`` at that
    timestamp (rows absent from the mask are non-members — fail-closed) and (b)
    finite in both series after aligning ``fwd`` and the mask to ``factor.index``.
    Timestamps with fewer than ``min_members`` usable members emit NaN (too few
    names for a meaningful cross-section), as do degenerate (constant-rank) groups.

    The factor is assumed direction-adjusted upstream (x ``FeatureSpec.direction``);
    this function never re-signs. Returns a float Series named ``"rank_ic"`` indexed
    by sorted ``ts_open`` (epoch-ms int64), one entry per timestamp present in
    ``factor``.
    """
    if not isinstance(factor.index, pd.MultiIndex) or factor.index.nlevels != 2:
        raise ValueError("factor must be indexed by a 2-level (ts_open, instrument_id) MultiIndex")
    if min_members < 2:
        raise ValueError(f"min_members must be >= 2; got {min_members}")

    factor_values = factor.to_numpy(dtype=float)
    fwd_values = fwd.reindex(factor.index).to_numpy(dtype=float)
    member = universe_mask.reindex(factor.index, fill_value=False).astype(bool).to_numpy()

    grouped = factor.groupby(level=0, sort=True).indices
    timestamps: list[int] = []
    ics: list[float] = []
    for ts, positions in grouped.items():
        usable = (
            member[positions]
            & np.isfinite(factor_values[positions])
            & np.isfinite(fwd_values[positions])
        )
        timestamps.append(int(ts))  # type: ignore[call-overload]
        if int(usable.sum()) < min_members:
            ics.append(float("nan"))
            continue
        chosen = positions[usable]
        ics.append(_spearman(factor_values[chosen], fwd_values[chosen]))

    index = pd.Index(np.asarray(timestamps, dtype=np.int64), name="ts_open")
    return pd.Series(ics, index=index, name="rank_ic", dtype=float)


def newey_west_tstat(x: pd.Series, lags: int) -> float:
    """Newey-West (HAC, Bartlett kernel) t-statistic of the mean of ``x``.

    With ``n`` finite observations, sample mean ``x̄``, demeaned ``d_t = x_t - x̄``,
    and lag-``l`` autocovariance ``gamma_l = (1/n) Σ_{t=l+1..n} d_t · d_{t-l}``::

        S        = gamma_0 + 2 · Σ_{l=1..L} w_l · gamma_l,   w_l = 1 - l/(L+1)   (Bartlett)
        Var(x̄)  = S / n
        t_NW     = x̄ / sqrt(S / n)

    where ``L = lags`` (capped at ``n - 1``; empty-sum lags contribute nothing).
    ``lags = 0`` reduces to the plain t-stat with the population (ddof=0) variance.
    For ``h``-bar forward-return ICs computed on the dense 1-bar grid use
    ``lags = h`` — the labels overlap ``h - 1`` neighbours, inducing exactly that
    much serial correlation under the null.

    NaNs are dropped first. Returns NaN when fewer than 2 finite observations remain
    or the kernel-weighted variance is non-positive (possible in tiny samples since
    the truncated Bartlett sum is only guaranteed PSD asymptotically).
    """
    if lags < 0:
        raise ValueError(f"lags must be >= 0; got {lags}")
    values = x.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 2:
        return float("nan")

    mean = float(values.mean())
    demeaned = values - mean
    long_run_var = float(demeaned @ demeaned) / n  # gamma_0
    max_lag = min(lags, n - 1)
    for lag in range(1, max_lag + 1):
        gamma = float(demeaned[lag:] @ demeaned[:-lag]) / n
        weight = 1.0 - lag / (lags + 1.0)
        long_run_var += 2.0 * weight * gamma

    if long_run_var <= 0.0:
        return float("nan")
    return mean / float(np.sqrt(long_run_var / n))


def ic_summary(ic: pd.Series, *, lags: int) -> ICSummary:
    """Summarize a per-timestamp IC series (index = ``ts_open`` epoch-ms int64).

    NaN ICs (thin cross-sections) are dropped. ``t_nw`` uses
    :func:`newey_west_tstat` with the supplied ``lags`` (use the label horizon ``h``
    on a dense grid; 0 on a non-overlapping grid). ``hit_rate`` is the share of
    finite ICs > 0; ``ic_ir = mean/std`` (NaN when ``std`` is 0 or undefined);
    ``by_year`` buckets by UTC calendar year of the timestamp.
    """
    clean = ic[np.isfinite(ic.to_numpy(dtype=float))]
    n = len(clean)
    if n == 0:
        return ICSummary(
            n=0,
            mean=float("nan"),
            std=float("nan"),
            t_nw=float("nan"),
            hit_rate=float("nan"),
            ic_ir=float("nan"),
            by_year={},
        )

    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if n >= 2 else float("nan")
    ic_ir = mean / std if np.isfinite(std) and std > 0.0 else float("nan")
    hit_rate = float((clean > 0.0).mean())

    years = pd.to_datetime(clean.index.to_numpy(dtype=np.int64), unit="ms", utc=True).year
    by_year = {int(year): float(group.mean()) for year, group in clean.groupby(np.asarray(years))}

    return ICSummary(
        n=n,
        mean=mean,
        std=std,
        t_nw=newey_west_tstat(clean, lags),
        hit_rate=hit_rate,
        ic_ir=ic_ir,
        by_year=by_year,
    )


def non_overlapping(series: pd.Series, h_bars: int, *, delta_ms: int | None = None) -> pd.Series:
    """Subsample a timestamp-indexed series to a non-overlapping ``h``-bar grid.

    ``h``-bar labels observed every bar overlap their ``h - 1`` neighbours; keeping
    every ``h``-th grid point makes consecutive observations non-overlapping, so
    plain i.i.d. inference (lags=0) is honest. The grid is the sorted unique
    timestamps of the index (level 0 for a MultiIndexed panel series).

    Two anchoring modes:

    * ``delta_ms is None`` (default, back-compatible): keep grid POSITIONS
      ``0, h, 2h, …`` of the slice's own sorted unique timestamps. The kept set is
      anchored at the slice's first timestamp, so subsampling a sub-window of a
      series can keep a DIFFERENT phase than subsampling the full series -- fine for
      a one-shot batch IC report, WRONG when the same IC must be sliced many times
      (the walk-forward, where a per-leg and a global pass would diverge).
    * ``delta_ms`` set to the bar width (epoch-ms): keep timestamps on an ABSOLUTE,
      epoch-fixed bar grid -- exactly the bars with ``(ts // delta_ms) % h == 0``.
      The kept phase depends ONLY on the venue-fixed bar grid (the epoch), NEVER on
      which window is being subsampled, so this is SLICE-INVARIANT:
      ``non_overlapping(full, h, delta_ms=D)`` restricted to any sub-window equals
      ``non_overlapping(window, h, delta_ms=D)`` for that window -- and in
      particular two overlapping windows keep the SAME timestamps on their overlap.
      This is what lets the walk-forward compute the IC once globally and have each
      leg's slice carry the identical kept-IC phase a per-leg pass would (only the
      EWMA warm-up depth differs). Kept points stay uniformly spaced by ``h*delta``
      (so :func:`estimate_blend_weights`'s uniform-spacing assertion still holds).
      The epoch phase differs from the slice-first phase the default mode uses, so
      the two modes are NOT interchangeable -- ``delta_ms`` is for the rolling
      walk-forward weights; ``None`` for one-shot batch IC reports.

    Returns a new (filtered) Series; the input is not mutated.
    """
    if h_bars <= 0:
        raise ValueError(f"h_bars must be > 0; got {h_bars}")
    level0 = (
        series.index.get_level_values(0)
        if isinstance(series.index, pd.MultiIndex)
        else series.index
    )
    if delta_ms is None:
        grid = np.unique(level0.to_numpy())
        keep = grid[::h_bars]
        return series[level0.isin(keep)]
    if delta_ms <= 0:
        raise ValueError(f"delta_ms must be > 0; got {delta_ms}")
    grid = np.unique(level0.to_numpy())
    if grid.size == 0:
        return series[level0.isin(grid)]
    bar_idx = grid.astype(np.int64) // np.int64(delta_ms)
    keep = grid[(bar_idx % h_bars) == 0]
    return series[level0.isin(keep)]
