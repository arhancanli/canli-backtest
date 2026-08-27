"""Volatility estimators (alphaDesign.md §2.2) — built first; labeling and the
vol-normalized factors depend on the sigma_hat helpers exported here.

Helpers operate on the sanctioned wide layout (:meth:`FeatureContext.panel`): a
float64 frame on the COMPLETE expected bar grid, one column per instrument. Row
position == time slot, so ``shift``/``rolling``/``ewm`` are exact *time* operations
and a missing bar (NaN grid slot) propagates NaN naturally — gaps are never bridged
and inputs are never forward-filled or mutated.

Estimators (exact formulas — see each function docstring):

* :func:`ewma_vol` / :func:`ewma_vol_from_returns` — zero-mean EWMA sigma_hat, span 168,
  ``min_periods = span``. THE workhorse vol reused by labels, TS-momentum
  (:mod:`alphaforge.features.library.momentum`) and residual reversal
  (:mod:`alphaforge.features.library.mean_reversion`).
* :func:`parkinson` — Parkinson (1980) range estimator, window 168.
* :func:`yang_zhang` — Yang-Zhang (2000) drift-independent OHLC estimator,
  windows 168 and 720.

Registered features (all ``direction=0`` — risk/regime *features*, not alphas; not
cross-sectional): ``vol_yz_168``, ``vol_yz_720``, ``vol_pk_168`` (annualized), and
``vol_ratio_168_720 = sigma_YZ(168)/sigma_YZ(720)`` (vol-of-vol regime proxy; the
annualization constant cancels). All are finite-window (parity atol = 0):
Yang-Zhang over ``n`` bars needs ``n + 1`` bars of history (it consumes
``C_{t-1}``), Parkinson needs exactly ``n``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from alphaforge.core.time import Timeframe
from alphaforge.features.context import long_series
from alphaforge.features.registry import feature
from alphaforge.features.spec import Family, FeatureSpec

if TYPE_CHECKING:
    from alphaforge.features.context import FeatureContext

__all__ = [
    "BARS_PER_YEAR_H1",
    "EWMA_HEADROOM_SPANS",
    "EWMA_VOL_SPAN",
    "ewma_vol",
    "ewma_vol_from_returns",
    "ewma_vol_halflife",
    "log_returns",
    "parkinson",
    "vol_pk_168",
    "vol_ratio_168_720",
    "vol_yz_168",
    "vol_yz_720",
    "yang_zhang",
]

BARS_PER_YEAR_H1: Final[float] = Timeframe.H1.bars_per_year
"""Annualization base for per-bar 1h vol: ``sigma_ann = sigma_bar · sqrt(8760)``."""

EWMA_VOL_SPAN: Final[int] = 168
"""Default EWMA sigma_hat span (7d of 1h bars) — alphaDesign.md §2.2."""

EWMA_HEADROOM_SPANS: Final[int] = 15
"""Lookback headroom (in spans) this library grants EWMA-consuming specs.

The machinery's :data:`~alphaforge.features.spec.EWMA_LOOKBACK_SPANS` (= 10) is a
*floor*; at exactly 10 spans the truncated live window drops weight
``(1-λ)^(10·span) ≈ e^-20 ≈ 2e-9`` of the variance recursion — the same order as
the 1e-9 parity rtol, so vol-regime drift in the dropped tail can (and, measured,
does) breach tolerance. buildabilityCritique.md §6.1 asks for a *documented
truncation error < 1e-12*: at 15 spans the dropped weight is
``(1-λ)^(15·span) ≈ e^-30 ≈ 9e-14`` of the variance (≈ 5e-14 relative on sigma),
which stays below 1e-12 even if the dropped tail's variance level deviates from
the local level by an order of magnitude. Library specs therefore derive their
EWMA lookback as ``EWMA_HEADROOM_SPANS · span``.
"""


def _log(frame: pd.DataFrame) -> pd.DataFrame:
    """Elementwise natural log of a wide panel; NaN-preserving, float64 out.

    Package-internal: the one place panel logs are taken (momentum and
    mean-reversion import it). Inputs are never mutated.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.log(frame.to_numpy(dtype="float64"))
    return pd.DataFrame(values, index=frame.index, columns=frame.columns)


# ----------------------------------------------- prefix-independent window reductions
#
# pandas' rolling aggregations (mean/var/cov) run on STREAMING accumulators: the
# value at a given window depends, in the last ulps, on everything that came
# before it (measured drift ~1e-13 between a long and a truncated history). That
# silently breaks the finite-window parity contract (atol = 0: compute_asof sees
# exactly ``lookback_bars`` bars, compute_history sees years). The reducers below
# evaluate every output from ONLY its own window (numpy ``sliding_window_view``),
# so identical window content gives bit-identical results regardless of preceding
# history. Any NaN inside a window yields NaN — exactly ``min_periods = window``
# semantics and the no-gap-bridging guarantee. Package-internal: mean_reversion
# imports them for the beta/residual windows.


def _roll_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling sum over axis 0; rows before ``window - 1`` and NaN-touched windows NaN.

    The input is normalized to FORTRAN-ordered float64 first. With an F-ordered
    base, the sliding window's reduction axis is unit-stride, so numpy reduces
    every window with the same contiguous pairwise loop — the result depends ONLY
    on the window's contents, never on the caller's memory layout, the panel's
    column count, or the window's position in the array. A C-ordered base makes
    the reduction axis strided, and numpy's buffered strided reduction couples
    neighbouring windows into shared buffer chunks, so the same window contents
    can reduce to different last-ulp results — breaking the atol = 0 parity
    contract this reducer exists to uphold (engine panels arrive C-ordered,
    DataFrame-arithmetic outputs F-ordered).
    """
    arr = np.asfortranarray(arr, dtype="float64")
    out = np.full(arr.shape, np.nan, dtype="float64")
    if arr.shape[0] >= window:
        view = np.lib.stride_tricks.sliding_window_view(arr, window, axis=0)
        out[window - 1 :] = view.sum(axis=-1)
    return out


def _roll_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean over axis 0 (see :func:`_roll_sum` for NaN/warmup semantics)."""
    return _roll_sum(arr, window) / float(window)


def _roll_var(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling sample variance (ddof=1) over axis 0, prefix-independent.

    Computational form ``(Σx² - (Σx)²/n) / (n-1)`` — the exact ``1/(n-1)``
    sample variance; clamped at 0 against catastrophic cancellation on
    near-constant windows (NaN propagates through the clamp).
    """
    s1 = _roll_sum(arr, window)
    s2 = _roll_sum(arr * arr, window)
    var = (s2 - s1 * s1 / float(window)) / float(window - 1)
    clamped: np.ndarray = np.maximum(var, 0.0)
    return clamped


def log_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Per-bar log returns ``r_t = ln(C_t / C_{t-1})`` columnwise on the grid.

    Computed positionally on the complete expected bar grid, so a missing bar
    yields NaN both at the gap slot and at the slot after it — a return is never
    silently taken across a gap.
    """
    return _log(close / close.shift(1))


def ewma_vol_from_returns(returns: pd.DataFrame, span: int = EWMA_VOL_SPAN) -> pd.DataFrame:
    """Zero-mean EWMA volatility of a pre-computed return panel.

    Formula (alphaDesign.md §2.2), with ``λ = 2/(span+1)``::

        sigma_hat2_t = λ·r_t² + (1-λ)·sigma_hat2_{t-1}        (zero-mean assumption)
        sigma_hat_t  = sqrt(sigma_hat2_t)                      # per-bar vol

    Implemented as ``r².ewm(span=span, adjust=False, min_periods=span).mean()`` —
    ``adjust=False`` is exactly the recursion above with ``alpha = λ``, and
    ``min_periods = span`` keeps the estimate NaN until ``span`` valid returns
    exist. This is an infinite-memory recursion: any spec consuming it must opt
    into the EWMA-family convention (``lookback_bars >= 10·span``; parity rtol
    1e-9, buildabilityCritique.md §6.1).
    """
    if span <= 0:
        raise ValueError(f"ewma_vol span must be > 0, got {span}")
    var = returns.pow(2).ewm(span=span, adjust=False, min_periods=span).mean()
    return var.pow(0.5)


def ewma_vol(close: pd.DataFrame, span: int = EWMA_VOL_SPAN) -> pd.DataFrame:
    """Per-bar EWMA volatility of log returns of ``close`` (zero-mean, min_periods=span).

    ``sigma_hat_t = sqrt(λ·r_t² + (1-λ)·sigma_hat2_{t-1})`` with ``r_t = ln(C_t/C_{t-1})`` and
    ``λ = 2/(span+1)`` — see :func:`ewma_vol_from_returns` for the exact
    implementation and the EWMA-family lookback convention.
    """
    return ewma_vol_from_returns(log_returns(close), span=span)


def ewma_vol_halflife(returns: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """Zero-mean EWMA volatility of a return panel, halflife-parameterized.

    Same recursion as :func:`ewma_vol_from_returns` with ``alpha`` expressed via a
    halflife ``h`` instead of a span::

        alpha        = 1 - exp(ln(0.5) / h)            # weight halves every h bars
        sigma_hat2_t = alpha·r_t² + (1-alpha)·sigma_hat2_{t-1}
        sigma_hat_t  = sqrt(sigma_hat2_t)

    Implemented as ``r².ewm(halflife=h, adjust=False, min_periods=int(h)).mean()``.
    This is the ONE halflife EWMA-vol body in the codebase (``sigma_daily`` in
    :mod:`alphaforge.features.library.market` consumes it). Infinite-memory: any
    spec consuming it must opt into the EWMA-family convention (equivalent span
    ``2/alpha - 1``; ``lookback_bars >= 10·span`` floor, parity rtol 1e-9,
    buildabilityCritique.md §6.1).
    """
    if halflife <= 0:
        raise ValueError(f"ewma_vol halflife must be > 0, got {halflife}")
    var = returns.pow(2).ewm(halflife=halflife, adjust=False, min_periods=int(halflife)).mean()
    return var.pow(0.5)


def parkinson(
    high: pd.DataFrame,
    low: pd.DataFrame,
    *,
    window: int = 168,
    annualize: bool = False,
) -> pd.DataFrame:
    """Parkinson (1980) range volatility over ``window`` bars.

    Formula (alphaDesign.md §2.2)::

        sigma2_P = (1 / (4·n·ln 2)) · Σ_{t=1..n} [ln(H_t / L_t)]²
        sigma_P  = sqrt(sigma2_P)                       # per-bar
        sigma_ann = sigma_P · sqrt(8760)                # annualize=True (1h bars)

    Needs exactly ``n`` bars (no previous close): the first finite value appears
    once ``window`` consecutive grid slots have data, and any gap inside the
    window yields NaN. Windows are reduced independently (prefix-independent —
    see the reducer note above), so batch/live parity is exact (atol = 0).
    """
    if window < 1:
        raise ValueError(f"parkinson window must be >= 1, got {window}")
    hl_sq = _log(high / low).pow(2)
    var_p = _roll_mean(hl_sq.to_numpy(dtype="float64"), window) / (4.0 * math.log(2.0))
    vol = np.sqrt(var_p)
    if annualize:
        vol = vol * math.sqrt(BARS_PER_YEAR_H1)
    return pd.DataFrame(vol, index=high.index, columns=high.columns)


def yang_zhang(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    *,
    window: int = 168,
    annualize: bool = False,
) -> pd.DataFrame:
    """Yang-Zhang (2000) drift-independent OHLC volatility over ``window`` bars.

    Formula (alphaDesign.md §2.2), per bar ``t`` with ``n = window``::

        o_t = ln(O_t / C_{t-1})                 # gap return
        c_t = ln(C_t / O_t)                     # open-to-close
        u_t = ln(H_t / O_t),  d_t = ln(L_t / O_t)

        sigma2_o  = (1/(n-1)) Σ (o_t - ō)²
        sigma2_c  = (1/(n-1)) Σ (c_t - c̄)²
        sigma2_RS = (1/n) Σ [ u_t·(u_t - c_t) + d_t·(d_t - c_t) ]   # Rogers-Satchell
        k     = 0.34 / (1.34 + (n+1)/(n-1))
        sigma2_YZ = sigma2_o + k·sigma2_c + (1-k)·sigma2_RS

    Per-bar ``sigma_YZ = sqrt(sigma2_YZ)``; ``annualize=True`` multiplies by
    ``sqrt(8760)`` (1h bars). Sample variances are the exact ``1/(n-1)`` (ddof=1)
    form. Because ``o_t`` consumes ``C_{t-1}``, the estimator needs ``window + 1``
    bars: NaN until then, and any NaN inside a window (a data gap) yields NaN —
    never bridged. Windows are reduced independently (prefix-independent — see
    the reducer note above), so batch/live parity is exact (atol = 0).
    """
    if window < 2:
        raise ValueError(f"yang_zhang window must be >= 2 (uses 1/(n-1)), got {window}")
    o = _log(open_ / close.shift(1)).to_numpy(dtype="float64")
    c = _log(close / open_).to_numpy(dtype="float64")
    u = _log(high / open_).to_numpy(dtype="float64")
    d = _log(low / open_).to_numpy(dtype="float64")
    var_o = _roll_var(o, window)
    var_c = _roll_var(c, window)
    rs = u * (u - c) + d * (d - c)
    var_rs = _roll_mean(rs, window)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    var_yz = var_o + k * var_c + (1.0 - k) * var_rs
    vol = np.sqrt(var_yz)
    if annualize:
        vol = vol * math.sqrt(BARS_PER_YEAR_H1)
    return pd.DataFrame(vol, index=close.index, columns=close.columns)


# -------------------------------------------------------------------- feature fns


def _vol_yz_fn(ctx: FeatureContext, spec: FeatureSpec) -> pd.Series:
    """Registered body for ``vol_yz_{n}``: annualized Yang-Zhang sigma over n bars.

    ``sigma_YZ,ann(t) = sqrt(sigma2_o + k·sigma2_c + (1-k)·sigma2_RS) · sqrt(8760)`` — see
    :func:`yang_zhang` for the full formula. Pure function of bars available at
    the decision close ``t + Δ``; NaN until ``n + 1`` bars of history exist.
    """
    window = int(spec.params["window"])
    vol = yang_zhang(
        ctx.panel("open"),
        ctx.panel("high"),
        ctx.panel("low"),
        ctx.panel("close"),
        window=window,
        annualize=bool(spec.params["annualize"]),
    )
    return long_series(vol, name=spec.name)


def _vol_pk_fn(ctx: FeatureContext, spec: FeatureSpec) -> pd.Series:
    """Registered body for ``vol_pk_{n}``: annualized Parkinson sigma over n bars.

    ``sigma_P,ann(t) = sqrt((1/(4·n·ln 2))·Σ ln(H/L)²) · sqrt(8760)`` — see
    :func:`parkinson`. NaN until ``n`` bars of history exist.
    """
    window = int(spec.params["window"])
    vol = parkinson(
        ctx.panel("high"),
        ctx.panel("low"),
        window=window,
        annualize=bool(spec.params["annualize"]),
    )
    return long_series(vol, name=spec.name)


def _vol_ratio_fn(ctx: FeatureContext, spec: FeatureSpec) -> pd.Series:
    """Registered body for ``vol_ratio_168_720``: vol-of-vol regime proxy.

    ``RATIO(t) = sigma_YZ(fast)(t) / sigma_YZ(slow)(t)`` (alphaDesign.md §2.2) — per-bar
    Yang-Zhang at both windows; the annualization constant cancels. NaN until the
    slow window (``slow + 1`` bars) is warm, and NaN (never ±inf) where the slow
    sigma is zero.
    """
    fast = int(spec.params["fast_window"])
    slow = int(spec.params["slow_window"])
    open_ = ctx.panel("open")
    high = ctx.panel("high")
    low = ctx.panel("low")
    close = ctx.panel("close")
    fast_vol = yang_zhang(open_, high, low, close, window=fast)
    slow_vol = yang_zhang(open_, high, low, close, window=slow)
    ratio = fast_vol / slow_vol.where(slow_vol > 0.0)
    return long_series(ratio, name=spec.name)


# ------------------------------------------------------------ registered factories


@feature
def vol_yz_168() -> FeatureSpec:
    """Annualized Yang-Zhang vol, 168 bars (7d). Feature, not alpha (direction 0)."""
    window = 168
    return FeatureSpec(
        name="vol_yz_168",
        family=Family.VOLATILITY,
        direction=0,
        cross_sectional=False,
        lookback_bars=window + 1,  # o_t consumes C_{t-1}
        params={"window": window, "annualize": True},
        fn=_vol_yz_fn,
    )


@feature
def vol_yz_720() -> FeatureSpec:
    """Annualized Yang-Zhang vol, 720 bars (30d). Feature, not alpha (direction 0)."""
    window = 720
    return FeatureSpec(
        name="vol_yz_720",
        family=Family.VOLATILITY,
        direction=0,
        cross_sectional=False,
        lookback_bars=window + 1,  # o_t consumes C_{t-1}
        params={"window": window, "annualize": True},
        fn=_vol_yz_fn,
    )


@feature
def vol_pk_168() -> FeatureSpec:
    """Annualized Parkinson vol, 168 bars (7d). Feature, not alpha (direction 0)."""
    window = 168
    return FeatureSpec(
        name="vol_pk_168",
        family=Family.VOLATILITY,
        direction=0,
        cross_sectional=False,
        lookback_bars=window,  # range-only: no previous close
        params={"window": window, "annualize": True},
        fn=_vol_pk_fn,
    )


@feature
def vol_ratio_168_720() -> FeatureSpec:
    """sigma_YZ(168)/sigma_YZ(720) — vol-of-vol regime proxy (direction 0, not CS)."""
    fast, slow = 168, 720
    return FeatureSpec(
        name="vol_ratio_168_720",
        family=Family.VOLATILITY,
        direction=0,
        cross_sectional=False,
        lookback_bars=slow + 1,  # bound by the slow Yang-Zhang window
        params={"fast_window": fast, "slow_window": slow},
        fn=_vol_ratio_fn,
    )
