"""BacktestResult — the persisted truth of one engine run (execDesign.md §4.5).

One immutable container holding everything a run produced: the equity curve,
the fills / funding / corporate-actions / financing / orders / positions frames, the
:class:`PerfSummary`
(headline metrics on the UTC-daily basis, leakageCritique.md finding 29), a
config echo for reproducibility, and the engine's bookkeeping counters
(dropped / unfilled / forced-flat orders — silence is never an option).

Artifact layout written by :meth:`BacktestResult.save` (all under one run
directory; parquet via pyarrow, full float64 precision — rounding happens only
in ``summary.txt`` / ``tearsheet.txt`` at render time):

==================  ==========================================================
``equity.parquet``     ``ts`` (epoch-ms int64), ``equity`` (float64, quote)
``fills.parquet``      ``ts, instrument_id, side, qty, price, notional, fee,
                       liquidity, realized_pnl_quote, client_order_id, reason``
``funding.parquet``    ``ts_funding, instrument_id, rate, mark_price,
                       position_qty, payment_quote`` (stored-events table rows
                       that actually settled against a position)
``corporate_actions.parquet`` applied split conversions and dividend cashflows
``financing.parquet``  cash, margin-debit, and short-collateral accrual intervals
``orders.parquet``     ``decision_ts, instrument_id, side, qty,
                       decision_price, status, client_order_id`` — every order
                       decision incl. skipped/dropped ones
``positions.parquet``  ``ts, instrument_id, qty, mark, weight, unreal_pnl``
                       per bar-close snapshot
``run_meta.json``      config echo + counters (sorted keys, reproducible)
``summary.txt``        rendered :class:`PerfSummary`
``tearsheet.png/.txt`` 6-panel tearsheet (``alphaforge.analytics``)
==================  ==========================================================

:meth:`BacktestResult.load` round-trips a saved directory; the summary is
*recomputed* from the artifacts (never trusted from disk), so a load can only
ever disagree with a save if the artifacts themselves were tampered with.

The class structurally satisfies :class:`alphaforge.analytics.ResultLike`
(``equity`` / ``fills`` / ``funding`` / ``positions``) so the same tearsheet
code serves backtests and the live daily job.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pandas as pd

from alphaforge.analytics import PerfSummary, build_tearsheet, render_text, summarize

__all__ = ["BacktestResult"]

_EQUITY_FILE: Final[str] = "equity.parquet"
_FILLS_FILE: Final[str] = "fills.parquet"
_FUNDING_FILE: Final[str] = "funding.parquet"
_CORPORATE_ACTIONS_FILE: Final[str] = "corporate_actions.parquet"
_FINANCING_FILE: Final[str] = "financing.parquet"
_ORDERS_FILE: Final[str] = "orders.parquet"
_POSITIONS_FILE: Final[str] = "positions.parquet"
_META_FILE: Final[str] = "run_meta.json"
_SUMMARY_FILE: Final[str] = "summary.txt"

FILLS_COLUMNS: Final[tuple[str, ...]] = (
    "ts",
    "instrument_id",
    "side",
    "qty",
    "price",
    "notional",
    "fee",
    "liquidity",
    "realized_pnl_quote",
    "client_order_id",
    "reason",
)

FUNDING_COLUMNS: Final[tuple[str, ...]] = (
    "ts_funding",
    "instrument_id",
    "rate",
    "mark_price",
    "position_qty",
    "payment_quote",
)

CORPORATE_ACTION_COLUMNS: Final[tuple[str, ...]] = (
    "action_ts",
    "instrument_id",
    "action_type",
    "position_qty_before",
    "ratio",
    "cash_amount",
    "cashflow_quote",
)

FINANCING_COLUMNS: Final[tuple[str, ...]] = (
    "start_ts",
    "end_ts",
    "currency",
    "cash_balance",
    "unrestricted_credit_base",
    "short_proceeds_base",
    "debit_base",
    "credit_rate_bps",
    "debit_rate_bps",
    "short_proceeds_rate_bps",
    "day_count",
    "payment_quote",
    "source",
)

ORDERS_COLUMNS: Final[tuple[str, ...]] = (
    "decision_ts",
    "instrument_id",
    "side",
    "qty",
    "decision_price",
    "status",
    "client_order_id",
)

POSITIONS_COLUMNS: Final[tuple[str, ...]] = (
    "ts",
    "instrument_id",
    "qty",
    "mark",
    "weight",
    "unreal_pnl",
)


def _require_columns(name: str, frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Validate (and column-order) ``frame`` against its declared schema."""
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} frame missing columns {sorted(missing)}")
    return frame.loc[:, list(columns)]


def _json_default(value: object) -> object:
    """JSON fallback for config echo values (Paths, enums, NaN-free floats)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return str(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class BacktestResult:
    """Everything one event-driven backtest run produced (full precision).

    ``equity`` is a float64 Series in quote units indexed by epoch-ms bar
    closes; the frames follow the module-docstring schemas. ``config`` echoes
    the run parameters (span, instrument ids, cost params, fill model) and
    ``counters`` the engine's bookkeeping (fills applied, orders skipped by
    the no-trade band, dropped on gaps / missing cost inputs, unfilled on the
    final bar, forced-flat delistings, funding events applied).

    The dataclass is frozen; the contained pandas objects are shared, not
    copied — treat them as read-only.
    """

    equity: pd.Series
    fills: pd.DataFrame
    funding_events: pd.DataFrame
    orders: pd.DataFrame
    positions: pd.DataFrame
    summary: PerfSummary
    corporate_actions: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=list(CORPORATE_ACTION_COLUMNS))
    )
    financing_events: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=list(FINANCING_COLUMNS))
    )
    config: dict[str, object] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fills", _require_columns("fills", self.fills, FILLS_COLUMNS))
        object.__setattr__(
            self,
            "funding_events",
            _require_columns("funding", self.funding_events, FUNDING_COLUMNS),
        )
        object.__setattr__(
            self,
            "financing_events",
            _require_columns("financing", self.financing_events, FINANCING_COLUMNS),
        )
        object.__setattr__(self, "orders", _require_columns("orders", self.orders, ORDERS_COLUMNS))
        object.__setattr__(
            self,
            "corporate_actions",
            _require_columns(
                "corporate_actions", self.corporate_actions, CORPORATE_ACTION_COLUMNS
            ),
        )
        object.__setattr__(
            self, "positions", _require_columns("positions", self.positions, POSITIONS_COLUMNS)
        )

    # ------------------------------------------------------- ResultLike views

    @property
    def funding(self) -> pd.Series:
        """Signed funding cashflows (quote units) indexed by settlement epoch-ms.

        ``payment_quote`` per settled event — positive means the book
        *received* funding. Empty Series when no event ever met a position.
        Satisfies the :class:`~alphaforge.analytics.ResultLike` ``funding``
        member; analytics only ever sums it.
        """
        return pd.Series(
            self.funding_events["payment_quote"].to_numpy(dtype="float64"),
            index=pd.Index(
                self.funding_events["ts_funding"].to_numpy(dtype="int64"), name="ts_funding"
            ),
            name="payment_quote",
        )

    # ------------------------------------------------------------ persistence

    def save(self, out_dir: Path) -> Path:
        """Write all artifacts (module docstring layout) into ``out_dir``; return it.

        Parquet frames carry full float64 precision; only ``summary.txt`` and
        the tearsheet round for display. The tearsheet is rendered through
        :func:`alphaforge.analytics.build_tearsheet` — the same function the
        live daily job uses.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        equity_frame = pd.DataFrame(
            {
                "ts": self.equity.index.to_numpy(dtype="int64"),
                "equity": self.equity.to_numpy(dtype="float64"),
            }
        )
        equity_frame.to_parquet(out_dir / _EQUITY_FILE, index=False)
        self.fills.to_parquet(out_dir / _FILLS_FILE, index=False)
        self.funding_events.to_parquet(out_dir / _FUNDING_FILE, index=False)
        self.corporate_actions.to_parquet(out_dir / _CORPORATE_ACTIONS_FILE, index=False)
        self.financing_events.to_parquet(out_dir / _FINANCING_FILE, index=False)
        self.orders.to_parquet(out_dir / _ORDERS_FILE, index=False)
        self.positions.to_parquet(out_dir / _POSITIONS_FILE, index=False)
        meta = {"config": self.config, "counters": self.counters}
        (out_dir / _META_FILE).write_text(
            json.dumps(meta, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        (out_dir / _SUMMARY_FILE).write_text(render_text(self.summary), encoding="utf-8")
        build_tearsheet(self, out_dir)
        return out_dir

    @classmethod
    def load(cls, out_dir: Path) -> BacktestResult:
        """Reconstruct a result from a :meth:`save` directory (round-trip).

        The :class:`PerfSummary` is **recomputed** from the loaded artifacts
        via :func:`alphaforge.analytics.summarize` — identical inputs give
        identical metrics, and a stale/tampered summary file can never
        misrepresent the artifacts (the text files are render-only outputs).

        Raises ``FileNotFoundError`` on a missing artifact and ``ValueError``
        on schema drift — never silently fills gaps.
        """
        equity_frame = pd.read_parquet(out_dir / _EQUITY_FILE)
        equity = pd.Series(
            equity_frame["equity"].to_numpy(dtype="float64"),
            index=pd.Index(equity_frame["ts"].to_numpy(dtype="int64"), name="ts"),
            name="equity",
        )
        fills = pd.read_parquet(out_dir / _FILLS_FILE)
        funding_events = pd.read_parquet(out_dir / _FUNDING_FILE)
        corporate_actions_path = out_dir / _CORPORATE_ACTIONS_FILE
        corporate_actions = (
            pd.read_parquet(corporate_actions_path)
            if corporate_actions_path.exists()
            else pd.DataFrame(columns=list(CORPORATE_ACTION_COLUMNS))
        )
        financing_path = out_dir / _FINANCING_FILE
        financing_events = (
            pd.read_parquet(financing_path)
            if financing_path.exists()
            else pd.DataFrame(columns=list(FINANCING_COLUMNS))
        )
        orders = pd.read_parquet(out_dir / _ORDERS_FILE)
        positions = pd.read_parquet(out_dir / _POSITIONS_FILE)
        meta_raw = json.loads((out_dir / _META_FILE).read_text(encoding="utf-8"))
        if not isinstance(meta_raw, dict):
            raise ValueError(f"{out_dir / _META_FILE}: top level must be a JSON object")
        config_raw = meta_raw.get("config", {})
        counters_raw = meta_raw.get("counters", {})
        if not isinstance(config_raw, dict) or not isinstance(counters_raw, dict):
            raise ValueError(f"{out_dir / _META_FILE}: 'config'/'counters' must be objects")
        funding_series = pd.Series(
            funding_events["payment_quote"].to_numpy(dtype="float64"),
            index=pd.Index(funding_events["ts_funding"].to_numpy(dtype="int64"), name="ts_funding"),
            name="payment_quote",
        )
        summary = summarize(equity, fills=fills, funding=funding_series, positions=positions)
        return cls(
            equity=equity,
            fills=fills,
            funding_events=funding_events,
            orders=orders,
            positions=positions,
            summary=summary,
            corporate_actions=corporate_actions,
            financing_events=financing_events,
            config=dict(config_raw),
            counters={str(k): int(v) for k, v in counters_raw.items()},
        )
