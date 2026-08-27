"""Tearsheet rendering — one multi-panel PNG + aligned text summary (execDesign.md §10.3).

The same :func:`build_tearsheet` serves backtests and the live daily job: it
accepts any object satisfying :class:`ResultLike` (the Phase-5 backtester's
result container does; so will the live ledger snapshot adapter).

Headless discipline: the ``Agg`` backend is forced **at import time, before
pyplot is imported** — this module must never try to open Cocoa/GUI windows
(it runs inside the 24/7 live process and CI).

Determinism: rendering is a pure function of its inputs — fixed figsize/dpi,
no RNG, no wall-clock reads, and PNG metadata pinned (``Software`` only, no
timestamps) — so re-rendering the same result is **byte-stable** for a given
matplotlib version (regression-tested). Text output rounds for display only;
all underlying numbers stay full-precision float64 (metrics.py policy).
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Protocol

import matplotlib
import numpy as np
import pandas as pd

from alphaforge.analytics.metrics import (
    DAYS_PER_YEAR,
    PerfSummary,
    daily_returns,
    drawdown_series,
    exposure_series,
    summarize,
)
from alphaforge.core.time import from_ms

matplotlib.use("Agg", force=True)

# Imported only after the backend is forced (headless contract; see module docstring).
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = ["ResultLike", "build_tearsheet", "render_text"]

_FIGSIZE: tuple[float, float] = (16.0, 14.0)
_DPI: int = 110
_ROLLING_SHARPE_DAYS: int = 30
_PNG_METADATA: dict[str, str] = {"Software": "alphaforge"}  # no timestamps -> byte-stable


class ResultLike(Protocol):
    """Duck-typed backtest/live result consumed by :func:`build_tearsheet`.

    ``equity``: epoch-ms indexed quote-unit Series (required, >= 2 points).
    ``fills``: §4.5 trades frame subset (``ts``, ``notional``, optional ``fee``) or None.
    ``funding``: epoch-ms indexed signed funding cashflow Series or None.
    ``positions``: §4.5 positions frame subset (``ts``, ``qty``, ``mark``) or None.
    """

    @property
    def equity(self) -> pd.Series: ...

    @property
    def fills(self) -> pd.DataFrame | None: ...

    @property
    def funding(self) -> pd.Series | None: ...

    @property
    def positions(self) -> pd.DataFrame | None: ...


# ---------------------------------------------------------------------------
# text rendering
# ---------------------------------------------------------------------------


def _fmt_value(name: str, value: object) -> str:
    """Render one PerfSummary field for the text block (display rounding only)."""
    if name.endswith("_ts"):
        ms = int(value)  # type: ignore[call-overload]
        return f"{ms}  ({from_ms(ms).strftime('%Y-%m-%d %H:%M:%S')}Z)"
    if isinstance(value, bool) or not isinstance(value, int | float):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def render_text(summary: PerfSummary) -> str:
    """Render every :class:`PerfSummary` field as an aligned text block.

    One ``name  value`` line per dataclass field, names left-justified to a
    common column — greppable and diff-friendly. Floats are rounded to 6
    decimals for display only (internal values stay full precision).
    """
    fields = dataclasses.fields(summary)
    width = max(len(f.name) for f in fields)
    lines = ["AlphaForge performance summary", "=" * (width + 28)]
    for f in fields:
        value = getattr(summary, f.name)
        lines.append(f"{f.name:<{width}}  {_fmt_value(f.name, value)}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# panel helpers (each takes a prepared Axes; all pure given their inputs)
# ---------------------------------------------------------------------------


def _dt(ms_index: pd.Index) -> pd.DatetimeIndex:
    """Epoch-ms Index -> UTC DatetimeIndex for plotting."""
    return pd.DatetimeIndex(
        pd.to_datetime(np.asarray(ms_index.to_numpy(), dtype=np.int64), unit="ms", utc=True)
    )


def _panel_equity(ax: Axes, equity: pd.Series) -> None:
    """Equity curve on log-y with high-water mark overlay."""
    x = _dt(equity.index)
    vals = np.asarray(equity.to_numpy(), dtype=np.float64)
    ax.plot(x, vals, color="#1f77b4", lw=1.0, label="equity")
    ax.plot(x, np.maximum.accumulate(vals), color="#aaaaaa", lw=0.8, ls="--", label="HWM")
    ax.set_yscale("log")
    ax.set_title("Equity (log)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)


def _panel_drawdown(ax: Axes, equity: pd.Series) -> None:
    """Underwater plot: -drawdown in percent (metrics.drawdown_series sign convention)."""
    dd = drawdown_series(equity)
    x = _dt(dd.index)
    y = -np.asarray(dd.to_numpy(), dtype=np.float64) * 100.0
    ax.fill_between(x, y, 0.0, color="#d62728", alpha=0.4)
    ax.plot(x, y, color="#d62728", lw=0.6)
    ax.set_title("Drawdown (%)")
    ax.grid(True, alpha=0.3)


def _panel_histogram(ax: Axes, d_rets: pd.Series) -> None:
    """Histogram of UTC-daily returns (the headline basis), in percent."""
    if len(d_rets) == 0:
        ax.text(0.5, 0.5, "insufficient data (< 2 UTC days)", ha="center", va="center")
    else:
        ax.hist(
            np.asarray(d_rets.to_numpy(), dtype=np.float64) * 100.0,
            bins=50,
            color="#2ca02c",
            alpha=0.8,
        )
        ax.axvline(0.0, color="#333333", lw=0.8)
    ax.set_title("Daily returns (%)")
    ax.grid(True, alpha=0.3)


def _panel_rolling_sharpe(ax: Axes, d_rets: pd.Series) -> None:
    """Rolling 30-day Sharpe on daily returns (full 30-day window required)."""
    if len(d_rets) >= _ROLLING_SHARPE_DAYS:
        roll = d_rets.rolling(_ROLLING_SHARPE_DAYS)
        rs = roll.mean() / roll.std(ddof=1) * math.sqrt(DAYS_PER_YEAR)
        rs = rs.dropna()
    else:
        rs = pd.Series(np.array([], dtype=np.float64), index=pd.Index([], name="ts"))
    if len(rs) == 0:
        ax.text(
            0.5, 0.5, f"insufficient data (< {_ROLLING_SHARPE_DAYS} days)", ha="center", va="center"
        )
    else:
        ax.plot(_dt(rs.index), np.asarray(rs.to_numpy(), dtype=np.float64), color="#9467bd", lw=1.0)
        ax.axhline(0.0, color="#333333", lw=0.8)
    ax.set_title(f"Rolling {_ROLLING_SHARPE_DAYS}d Sharpe (daily basis)")
    ax.grid(True, alpha=0.3)


def _panel_exposure(ax: Axes, equity: pd.Series, positions: pd.DataFrame | None) -> None:
    """Gross/net exposure as fractions of equity; placeholder if positions absent."""
    if positions is None:
        ax.text(0.5, 0.5, "no positions data", ha="center", va="center")
    else:
        expo = exposure_series(positions, equity)
        x = _dt(expo.index)
        ax.plot(
            x,
            np.asarray(expo["gross"].to_numpy(), dtype=np.float64),
            color="#ff7f0e",
            lw=0.9,
            label="gross",
        )
        ax.plot(
            x,
            np.asarray(expo["net"].to_numpy(), dtype=np.float64),
            color="#1f77b4",
            lw=0.9,
            label="net",
        )
        ax.axhline(0.0, color="#333333", lw=0.8)
        ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Exposure (x equity)")
    ax.grid(True, alpha=0.3)


def _monthly_table(d_rets: pd.Series) -> pd.DataFrame:
    """Compound daily returns into a year x month (1..12) table of monthly returns."""
    if len(d_rets) == 0:
        return pd.DataFrame()
    s = pd.Series(np.asarray(d_rets.to_numpy(), dtype=np.float64), index=_dt(d_rets.index))
    monthly = (1.0 + s).resample("1ME").prod() - 1.0
    monthly = monthly.dropna()
    midx = pd.DatetimeIndex(monthly.index)
    frame = pd.DataFrame(
        {
            "year": midx.year,
            "month": midx.month,
            "ret": np.asarray(monthly.to_numpy(), dtype=np.float64),
        }
    )
    return frame.pivot(index="year", columns="month", values="ret")


def _panel_monthly_heatmap(ax: Axes, d_rets: pd.Series) -> None:
    """Monthly-return heatmap (RdYlGn, symmetric scale) with % annotations."""
    table = _monthly_table(d_rets)
    if table.empty:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center")
        ax.set_title("Monthly returns (%)")
        return
    data = table.to_numpy(dtype=np.float64)
    finite = data[np.isfinite(data)]
    bound = float(np.max(np.abs(finite))) if finite.size else 1.0
    bound = bound if bound > 0.0 else 1.0
    masked = np.ma.masked_invalid(data)
    ax.imshow(masked, cmap="RdYlGn", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(range(data.shape[1]), [str(m) for m in table.columns])
    ax.set_yticks(range(data.shape[0]), [str(y) for y in table.index])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j] * 100.0:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("Monthly returns (%)")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def build_tearsheet(result: ResultLike, out_dir: Path) -> Path:
    """Render the 6-panel tearsheet PNG + text summary; return the PNG path.

    Writes ``tearsheet.png`` (equity log-y + HWM, underwater drawdown,
    daily-return histogram, rolling 30d Sharpe, gross/net exposure, monthly
    return heatmap) and ``tearsheet.txt`` (:func:`render_text` of the full
    :class:`PerfSummary`) into ``out_dir`` (created if missing). Byte-stable
    given identical inputs and matplotlib version (no timestamps, fixed
    figsize/dpi, Agg backend).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    equity = result.equity
    summary = summarize(
        equity, fills=result.fills, funding=result.funding, positions=result.positions
    )
    d_rets = daily_returns(equity)

    fig: Figure
    fig, axes = plt.subplots(3, 2, figsize=_FIGSIZE, dpi=_DPI)
    try:
        _panel_equity(axes[0][0], equity)
        _panel_drawdown(axes[0][1], equity)
        _panel_histogram(axes[1][0], d_rets)
        _panel_rolling_sharpe(axes[1][1], d_rets)
        _panel_exposure(axes[2][0], equity, result.positions)
        _panel_monthly_heatmap(axes[2][1], d_rets)
        span = (
            f"{from_ms(summary.start_ts).strftime('%Y-%m-%d')} -> "
            f"{from_ms(summary.end_ts).strftime('%Y-%m-%d')}"
        )
        fig.suptitle(
            f"AlphaForge tearsheet  |  {span}  |  Sharpe(d) {summary.sharpe:.2f}  |  "
            f"MaxDD {summary.max_dd * 100.0:.1f}%",
            fontsize=12,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        png_path = out_dir / "tearsheet.png"
        fig.savefig(png_path, dpi=_DPI, metadata=_PNG_METADATA)
    finally:
        plt.close(fig)

    txt_path = out_dir / "tearsheet.txt"
    txt_path.write_text(render_text(summary), encoding="utf-8")
    return png_path
