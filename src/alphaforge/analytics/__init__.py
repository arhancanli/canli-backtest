"""AlphaForge performance analytics (execDesign.md §10).

Curated public surface: exact §10.1 metrics over epoch-ms-indexed equity
Series (:func:`sharpe`, :func:`sortino`, :func:`cagr`, :func:`calmar`,
:func:`max_drawdown`, :func:`turnover`, ...), the :class:`PerfSummary`
container built by :func:`summarize`, and the §10.3 tearsheet renderer
(:func:`build_tearsheet` / :func:`render_text`, ``Agg``-only matplotlib).

Conventions inherited from the metrics module (load-bearing): headline ratios
are UTC-daily based with A = 365 (leakageCritique.md finding 29 — hourly
autocorrelation inflates sqrt(8760) annualization); rf = 0; money is float64
quote units at full internal precision with rounding only at rendering.
"""

from alphaforge.analytics.metrics import (
    DAYS_PER_YEAR,
    HOURS_PER_YEAR,
    MS_PER_YEAR,
    PerfSummary,
    WinMetrics,
    cagr,
    calmar,
    daily_equity,
    daily_returns,
    drawdown_series,
    exposure_series,
    max_drawdown,
    sharpe,
    sortino,
    summarize,
    to_returns,
    turnover,
    win_metrics,
)
from alphaforge.analytics.tearsheet import ResultLike, build_tearsheet, render_text

__all__ = [
    "DAYS_PER_YEAR",
    "HOURS_PER_YEAR",
    "MS_PER_YEAR",
    "PerfSummary",
    "ResultLike",
    "WinMetrics",
    "build_tearsheet",
    "cagr",
    "calmar",
    "daily_equity",
    "daily_returns",
    "drawdown_series",
    "exposure_series",
    "max_drawdown",
    "render_text",
    "sharpe",
    "sortino",
    "summarize",
    "to_returns",
    "turnover",
    "win_metrics",
]
