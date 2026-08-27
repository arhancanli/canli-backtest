"""Performance metrics over equity curves — exact formulas from execDesign.md §10.1.

All functions are pure: they take an equity ``pd.Series`` indexed by :data:`Ms`
(UTC epoch milliseconds, strictly increasing) with float64 values in quote units
(USDT in v1), and return plain floats / Series. Risk-free rate is 0 by crypto
convention. When an optional financing provider is enabled, realized cash financing is already
embedded in the equity curve; it is not subtracted again as a Sharpe benchmark.

Headline basis — DAILY aggregation (leakageCritique.md finding 29)
-------------------------------------------------------------------
Hourly Sharpe assumes iid bars; positive autocorrelation in hourly returns
inflates the ``sqrt(8760)`` annualization (Lo 2002). The headline Sharpe /
Sortino / vol therefore use **UTC-daily aggregated** returns (resample to
UTC-day last equity, then ``pct_change``) annualized with A = 365 (24/7 crypto
calendar, no business-day convention). The per-bar (hourly) estimators are
reported alongside as diagnostics — if the two disagree wildly, distrust the
hourly one.

Exact formulas (A = periods_per_year; T = number of return observations)::

    r_t      = E_t / E_{t-1} - 1
    CAGR     = (E_T / E_0)^(A/T) - 1
    Sharpe   = mean(r) / std(r, ddof=1) * sqrt(A)
    Sortino  = mean(r) / sigma_down * sqrt(A),  sigma_down = sqrt((1/T) * sum(min(r_t, 0)^2))
    MaxDD    = max_t (1 - E_t / max_{s<=t} E_s)
    Calmar   = CAGR / |MaxDD|
    Turnover = (1/T) * sum_t tau_t * A,  tau_t = sum_i |traded notional_{i,t}| / E_t  # one-way

Rounding policy: all internal computation is full-precision float64; rounding
happens only at report rendering (:func:`alphaforge.analytics.tearsheet.render_text`).

Undefined-metric convention: where a formula's denominator is zero or there are
too few observations, functions return ``math.nan`` (never raise) so summaries
stay total; input *shape* violations (non-increasing index, non-positive
equity, < 2 points) raise ``ValueError`` because they indicate caller bugs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from alphaforge.core.time import Ms

__all__ = [
    "DAYS_PER_YEAR",
    "HOURS_PER_YEAR",
    "MS_PER_YEAR",
    "PerfSummary",
    "WinMetrics",
    "cagr",
    "calmar",
    "daily_equity",
    "daily_returns",
    "drawdown_series",
    "exposure_series",
    "max_drawdown",
    "sharpe",
    "sortino",
    "summarize",
    "to_returns",
    "turnover",
    "win_metrics",
]

DAYS_PER_YEAR: Final[float] = 365.0
"""Annualization day count — 24/7 crypto calendar, not 252 business days."""

HOURS_PER_YEAR: Final[float] = 8760.0
"""365 * 24; A for 1h bars per execDesign.md §10.1."""

MS_PER_YEAR: Final[int] = 365 * 86_400_000
"""Milliseconds per (365-day) year; used to infer per-bar A from index spacing."""

_MS_PER_DAY: Final[int] = 86_400_000


# ---------------------------------------------------------------------------
# input validation / index helpers
# ---------------------------------------------------------------------------


def _validate_equity(equity: pd.Series) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Validate an equity curve and return ``(ts_ms, values)`` arrays.

    Requirements: >= 2 points, strictly increasing integer epoch-ms index,
    finite strictly-positive float values (equity is in quote units and a
    non-positive account equity would make every ratio metric meaningless).
    """
    if len(equity) < 2:
        raise ValueError(f"equity needs >= 2 points, got {len(equity)}")
    idx = np.asarray(equity.index.to_numpy(), dtype=np.int64)
    if not bool(np.all(np.diff(idx) > 0)):
        raise ValueError("equity index must be strictly increasing epoch-ms ints")
    vals = np.asarray(equity.to_numpy(), dtype=np.float64)
    if not bool(np.all(np.isfinite(vals))) or bool(np.any(vals <= 0.0)):
        raise ValueError("equity values must be finite and > 0")
    return idx, vals


def _as_array(returns: pd.Series | npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Coerce a returns vector (Series or ndarray) to a float64 ndarray."""
    if isinstance(returns, pd.Series):
        return np.asarray(returns.to_numpy(), dtype=np.float64)
    return np.asarray(returns, dtype=np.float64)


def _dt_index(ts_ms: npt.NDArray[np.int64]) -> pd.DatetimeIndex:
    """Epoch-ms int64 array -> tz-aware UTC DatetimeIndex (presentation/resample only)."""
    return pd.DatetimeIndex(pd.to_datetime(ts_ms, unit="ms", utc=True))


def _ms_index(dt: pd.DatetimeIndex) -> pd.Index:
    """Tz-aware UTC DatetimeIndex -> epoch-ms int64 Index (back to the API convention)."""
    naive = dt.tz_convert(None)  # UTC wall time, tz dropped
    ms = naive.to_numpy(dtype="datetime64[ms]").astype(np.int64)
    return pd.Index([int(v) for v in ms], name="ts")


# ---------------------------------------------------------------------------
# returns
# ---------------------------------------------------------------------------


def to_returns(equity: pd.Series) -> pd.Series:
    """Per-period simple returns ``r_t = E_t / E_{t-1} - 1``.

    The first observation (undefined return) is dropped; the result keeps the
    epoch-ms index of the *later* point of each pair, so ``r_t`` is stamped at
    the bar close where it materialized.
    """
    idx, vals = _validate_equity(equity)
    rets = vals[1:] / vals[:-1] - 1.0
    return pd.Series(rets, index=pd.Index(idx[1:], name="ts"), name="ret")


def daily_equity(equity: pd.Series) -> pd.Series:
    """Resample equity to UTC-day **last** observation; index = epoch-ms of day start.

    A point with timestamp exactly at UTC midnight belongs to the *new* day
    (bars are stamped at close; the 00:00 close is the first observation of
    the new UTC day). Days with no observations are dropped, never filled.
    """
    idx, vals = _validate_equity(equity)
    s = pd.Series(vals, index=_dt_index(idx))
    daily = s.resample("1D").last().dropna()
    return pd.Series(
        np.asarray(daily.to_numpy(), dtype=np.float64),
        index=_ms_index(pd.DatetimeIndex(daily.index)),
        name="equity",
    )


def daily_returns(equity: pd.Series) -> pd.Series:
    """UTC-daily simple returns — THE headline basis (leakageCritique.md finding 29).

    Hourly per-bar returns carry positive autocorrelation, which inflates the
    ``sqrt(8760)`` annualization of Sharpe-type ratios (Lo 2002). Aggregating
    to UTC-day-last equity first and computing ``pct_change`` on that grid is
    the honest headline estimator; per-bar metrics are diagnostics only.

    Returns an empty float Series if the equity curve spans fewer than 2 UTC days.
    """
    daily = daily_equity(equity)
    if len(daily) < 2:
        return pd.Series(np.array([], dtype=np.float64), index=pd.Index([], name="ts"), name="ret")
    return to_returns(daily)


# ---------------------------------------------------------------------------
# ratio metrics
# ---------------------------------------------------------------------------


def sharpe(returns: pd.Series | npt.NDArray[np.float64], periods_per_year: float) -> float:
    """Annualized Sharpe ratio ``mean(r) / std(r, ddof=1) * sqrt(A)`` (rf = 0).

    Risk-free = 0 by convention. Any configured realized financing cashflows are already
    reflected in the input equity curve.
    Returns ``nan`` for < 2 observations or zero sample std.
    """
    r = _as_array(returns)
    if r.size < 2:
        return math.nan
    sd = float(np.std(r, ddof=1))
    if sd == 0.0:
        return math.nan
    return float(np.mean(r)) / sd * math.sqrt(periods_per_year)


def sortino(returns: pd.Series | npt.NDArray[np.float64], periods_per_year: float) -> float:
    """Annualized Sortino ``mean(r) / sigma_down * sqrt(A)`` with target 0.

    ``sigma_down = sqrt((1/T) * sum_t min(r_t, 0)^2)`` — **full-T denominator**
    (not downside-count), per execDesign.md §10.1. Returns ``nan`` for empty
    input or when there are no negative returns (sigma_down = 0).
    """
    r = _as_array(returns)
    if r.size == 0:
        return math.nan
    downside = np.minimum(r, 0.0)
    sigma_down = math.sqrt(float(np.mean(downside * downside)))
    if sigma_down == 0.0:
        return math.nan
    return float(np.mean(r)) / sigma_down * math.sqrt(periods_per_year)


def cagr(equity: pd.Series, periods_per_year: float) -> float:
    """Compound annual growth rate ``(E_T / E_0)^(A/T) - 1``.

    ``T = len(equity) - 1`` return periods on the curve's own grid; ``A``
    converts to a per-year rate. Exactly consistent with per-period compounding.
    """
    _, vals = _validate_equity(equity)
    t = len(vals) - 1
    return float((vals[-1] / vals[0]) ** (periods_per_year / t) - 1.0)


def max_drawdown(equity: pd.Series) -> tuple[float, Ms, Ms]:
    """Maximum drawdown ``max_t (1 - E_t / HWM_t)`` -> ``(depth_frac, peak_ts, trough_ts)``.

    ``depth_frac`` is a non-negative fraction (0.10 = -10%). ``peak_ts`` is the
    timestamp of the high-water mark preceding the trough (first occurrence);
    ``trough_ts`` is the deepest point (first occurrence). For a curve that
    never draws down, returns ``(0.0, first_ts, first_ts)``.
    """
    idx, vals = _validate_equity(equity)
    hwm = np.maximum.accumulate(vals)
    dd = 1.0 - vals / hwm
    trough = int(np.argmax(dd))
    depth = float(dd[trough])
    if depth == 0.0:
        first = int(idx[0])
        return 0.0, first, first
    peak = int(np.argmax(vals[: trough + 1]))
    return depth, int(idx[peak]), int(idx[trough])


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Underwater series ``dd_t = 1 - E_t / max_{s<=t} E_s`` (>= 0), epoch-ms index.

    Sign convention matches the §10.1 MaxDD formula (depth as a positive
    fraction); plot ``-dd`` for a conventional underwater chart.
    """
    idx, vals = _validate_equity(equity)
    dd = 1.0 - vals / np.maximum.accumulate(vals)
    return pd.Series(dd, index=pd.Index(idx, name="ts"), name="drawdown")


def calmar(equity: pd.Series, periods_per_year: float) -> float:
    """Calmar ratio ``CAGR / |MaxDD|`` on the curve's own grid; ``nan`` if MaxDD = 0."""
    depth, _, _ = max_drawdown(equity)
    if depth == 0.0:
        return math.nan
    return cagr(equity, periods_per_year) / depth


# ---------------------------------------------------------------------------
# trading-activity metrics
# ---------------------------------------------------------------------------


def turnover(fills_notional: pd.Series, equity: pd.Series, periods_per_year: float) -> float:
    """Annualized one-way turnover ``(1/T) * sum_t tau_t * A`` (execDesign.md §10.1).

    ``tau_t = sum |traded notional| / E_t`` per equity period. ``fills_notional``
    is indexed by fill epoch-ms with values = **absolute** traded notional in
    quote units; fills are bucketed to the first equity timestamp ``>=`` their
    own (the bar close at/after execution). Fills after the last equity point
    are ignored (their bar never closed inside the curve). ``T = len(equity)``
    periods including zero-trade ones. One-way: buys and sells both count once.
    """
    idx, vals = _validate_equity(equity)
    if len(fills_notional) == 0:
        return 0.0
    fill_ts = np.asarray(fills_notional.index.to_numpy(), dtype=np.int64)
    notional = np.asarray(fills_notional.to_numpy(), dtype=np.float64)
    if bool(np.any(~np.isfinite(notional)) or np.any(notional < 0.0)):
        raise ValueError("fills_notional values must be finite and >= 0 (absolute notional)")
    pos = np.searchsorted(idx, fill_ts, side="left")
    keep = pos < len(idx)
    per_period = np.zeros(len(idx), dtype=np.float64)
    np.add.at(per_period, pos[keep], notional[keep])
    tau = per_period / vals
    return float(np.sum(tau) / len(idx) * periods_per_year)


def exposure_series(positions: pd.DataFrame, equity: pd.Series) -> pd.DataFrame:
    """Gross/net exposure fractions per equity timestamp from a positions frame.

    ``positions`` columns (execDesign.md §4.5 ``positions.parquet`` subset):
    ``ts`` (epoch-ms, must be a subset of the equity index), ``qty`` (signed
    base units), ``mark`` (mark price, quote). Signed notional = ``qty * mark``;
    ``gross_t = sum_i |notional_i| / E_t``, ``net_t = sum_i notional_i / E_t``.
    Equity timestamps with no position rows are flat (0.0). Returns a frame
    indexed by the full equity epoch-ms index with float columns gross/net.
    """
    idx, vals = _validate_equity(equity)
    eq_index = pd.Index(idx, name="ts")
    out = pd.DataFrame({"gross": 0.0, "net": 0.0}, index=eq_index)
    if len(positions) > 0:
        required = {"ts", "qty", "mark"}
        if not required.issubset(positions.columns):
            raise ValueError(f"positions frame must have columns {sorted(required)}")
        ts = np.asarray(positions["ts"].to_numpy(), dtype=np.int64)
        unknown = ~np.isin(ts, idx)
        if bool(np.any(unknown)):
            raise ValueError("positions.ts contains timestamps not on the equity index")
        signed = np.asarray((positions["qty"] * positions["mark"]).to_numpy(), dtype=np.float64)
        grouped_g = pd.Series(np.abs(signed)).groupby(pd.Series(ts)).sum()
        grouped_n = pd.Series(signed).groupby(pd.Series(ts)).sum()
        out.loc[grouped_g.index, "gross"] = grouped_g.to_numpy()
        out.loc[grouped_n.index, "net"] = grouped_n.to_numpy()
        out["gross"] = out["gross"] / vals
        out["net"] = out["net"] / vals
    return out


@dataclass(frozen=True, slots=True, kw_only=True)
class WinMetrics:
    """Win/loss statistics over a returns vector (daily basis for the headline).

    ``win_rate`` counts strictly positive returns over all T (zeros are not
    wins); ``avg_win`` / ``avg_loss`` are conditional means (``nan`` if no
    qualifying observations); ``profit_factor = sum(wins) / |sum(losses)|``
    (``nan`` if there are no losses).
    """

    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float


def win_metrics(returns: pd.Series | npt.NDArray[np.float64]) -> WinMetrics:
    """Compute :class:`WinMetrics` over a returns vector; all-``nan`` if empty."""
    r = _as_array(returns)
    if r.size == 0:
        return WinMetrics(
            win_rate=math.nan, avg_win=math.nan, avg_loss=math.nan, profit_factor=math.nan
        )
    wins = r[r > 0.0]
    losses = r[r < 0.0]
    win_rate = float(wins.size) / float(r.size)
    avg_win = float(np.mean(wins)) if wins.size else math.nan
    avg_loss = float(np.mean(losses)) if losses.size else math.nan
    loss_sum = float(np.sum(losses))
    profit_factor = float(np.sum(wins)) / abs(loss_sum) if loss_sum != 0.0 else math.nan
    return WinMetrics(
        win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss, profit_factor=profit_factor
    )


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class PerfSummary:
    """Full performance summary; headline ratios are DAILY-based (finding 29).

    Headline fields (``sharpe`` .. ``max_dd``) are computed on UTC-daily
    aggregated equity with A = 365; ``*_hourly`` fields are the same
    estimators on the curve's native per-bar grid (A inferred from median bar
    spacing — 8760 for 1h) and are diagnostics only. ``max_dd_bar`` is the
    full-resolution drawdown (>= the daily one; reported so intraday damage is
    never hidden). Monetary fields are quote units, full precision (rounding
    is a rendering concern). Optional inputs missing at :func:`summarize` time
    surface as ``nan`` ("not provided"), never silently as 0.
    """

    start_ts: Ms
    end_ts: Ms
    n_periods: int
    n_days: int
    initial_equity: float
    final_equity: float
    total_return: float
    # -- headline (UTC-daily basis, A = 365) --
    cagr: float
    vol_ann: float
    sharpe: float
    sortino: float
    calmar: float
    max_dd: float
    max_dd_peak_ts: Ms
    max_dd_trough_ts: Ms
    win_rate_d: float
    avg_win_d: float
    avg_loss_d: float
    profit_factor_d: float
    # -- per-bar diagnostics (A inferred from bar spacing; 8760 for 1h) --
    sharpe_hourly: float
    sortino_hourly: float
    vol_ann_hourly: float
    max_dd_bar: float
    # -- activity / cashflows --
    turnover_ann: float
    fees_paid: float
    funding_net: float
    gross_mean: float
    net_mean: float


def _annualized_vol(returns: pd.Series | npt.NDArray[np.float64], periods_per_year: float) -> float:
    """``std(r, ddof=1) * sqrt(A)``; ``nan`` for < 2 observations."""
    r = _as_array(returns)
    if r.size < 2:
        return math.nan
    return float(np.std(r, ddof=1)) * math.sqrt(periods_per_year)


def summarize(
    equity: pd.Series,
    *,
    fills: pd.DataFrame | None = None,
    funding: pd.Series | None = None,
    positions: pd.DataFrame | None = None,
) -> PerfSummary:
    """Build a :class:`PerfSummary` from an equity curve and optional artifacts.

    - ``equity``: epoch-ms indexed quote-unit curve (>= 2 points).
    - ``fills``: frame with ``ts`` (epoch-ms), ``notional`` (absolute traded
      quote notional) and optionally ``fee`` (quote) — execDesign.md §4.5
      ``trades.parquet`` subset. Drives ``turnover_ann`` / ``fees_paid``.
    - ``funding``: epoch-ms indexed Series of **signed** funding cashflows in
      quote units (positive = received); ``funding_net`` is its sum. These rows
      come from the stored funding-events table upstream — analytics never
      assumes a funding schedule.
    - ``positions``: frame per :func:`exposure_series`; drives mean gross/net.

    Omitted optional inputs yield ``nan`` for their fields (never silent 0).
    """
    idx, vals = _validate_equity(equity)
    bar_ms = float(np.median(np.diff(idx)))
    bar_a = MS_PER_YEAR / bar_ms

    bar_rets = to_returns(equity)
    d_eq = daily_equity(equity)
    d_rets = daily_returns(equity)

    # Headline CAGR compounds the FULL run over its actual elapsed wall time:
    # (E_T / E_0)^(year / elapsed) - 1, anchored at the run-start equity. The
    # daily-resampled curve drops the partial first day's PnL (its first point
    # is already end-of-day-1), which can flip the sign of a resampled CAGR
    # versus total_return on a strong day 1 — the time-anchored form is always
    # consistent with total_return. Drawdown stays daily-resampled per the
    # headline convention (leakage finding 29); full-resolution depth is
    # exposed as ``max_dd_bar``.
    elapsed_ms = float(idx[-1] - idx[0])
    cagr_d = float((vals[-1] / vals[0]) ** (MS_PER_YEAR / elapsed_ms) - 1.0)
    if len(d_eq) >= 2:
        max_dd_d, dd_peak, dd_trough = max_drawdown(d_eq)
        calmar_d = cagr_d / max_dd_d if max_dd_d > 0.0 else math.nan
    else:
        max_dd_d, dd_peak, dd_trough = math.nan, int(idx[0]), int(idx[0])
        calmar_d = math.nan

    wm = win_metrics(d_rets)
    max_dd_b, _, _ = max_drawdown(equity)

    if fills is not None:
        if not {"ts", "notional"}.issubset(fills.columns):
            raise ValueError("fills frame must have columns ['ts', 'notional']")
        notional = pd.Series(
            np.asarray(fills["notional"].to_numpy(), dtype=np.float64),
            index=pd.Index(np.asarray(fills["ts"].to_numpy(), dtype=np.int64), name="ts"),
        )
        turnover_ann = turnover(notional, equity, bar_a)
        fees_paid = float(fills["fee"].sum()) if "fee" in fills.columns else math.nan
    else:
        turnover_ann = math.nan
        fees_paid = math.nan

    funding_net = float(funding.sum()) if funding is not None else math.nan

    if positions is not None:
        expo = exposure_series(positions, equity)
        gross_mean = float(expo["gross"].mean())
        net_mean = float(expo["net"].mean())
    else:
        gross_mean = math.nan
        net_mean = math.nan

    return PerfSummary(
        start_ts=int(idx[0]),
        end_ts=int(idx[-1]),
        n_periods=len(bar_rets),
        n_days=len(d_eq),
        initial_equity=float(vals[0]),
        final_equity=float(vals[-1]),
        total_return=float(vals[-1] / vals[0] - 1.0),
        cagr=cagr_d,
        vol_ann=_annualized_vol(d_rets, DAYS_PER_YEAR),
        sharpe=sharpe(d_rets, DAYS_PER_YEAR),
        sortino=sortino(d_rets, DAYS_PER_YEAR),
        calmar=calmar_d,
        max_dd=max_dd_d,
        max_dd_peak_ts=dd_peak,
        max_dd_trough_ts=dd_trough,
        win_rate_d=wm.win_rate,
        avg_win_d=wm.avg_win,
        avg_loss_d=wm.avg_loss,
        profit_factor_d=wm.profit_factor,
        sharpe_hourly=sharpe(bar_rets, bar_a),
        sortino_hourly=sortino(bar_rets, bar_a),
        vol_ann_hourly=_annualized_vol(bar_rets, bar_a),
        max_dd_bar=max_dd_b,
        turnover_ann=turnover_ann,
        fees_paid=fees_paid,
        funding_net=funding_net,
        gross_mean=gross_mean,
        net_mean=net_mean,
    )
