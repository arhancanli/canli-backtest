"""EventDrivenBacktester — the truth engine (execDesign.md §4.1/§4.4).

THE no-lookahead contract (leakageCritique.md findings 4/5), enforced in code,
never by convention:

* A strategy decides at the **close** of bar ``t`` and its orders fill at the
  **open** of bar ``t+1`` — never the same bar. The fill model receives only
  the ``t+1`` bar and raises :class:`~alphaforge.core.errors.LookaheadError`
  if that bar opened before the decision; the engine re-checks every returned
  fill as defense in depth.
* The strategy sees history through :class:`StrategyContext`, whose ``bars``
  accessor is a true PIT read (``as_of = decision close``) — the engine
  slices, the strategy *cannot* ask for more.
* Funding cashflows are applied by iterating the **stored funding-events
  table** per instrument (leakageCritique.md finding 6): each event carries
  its own ``ts_funding``, so 8h/4h/1h funding intervals are honoured
  automatically — there is no funding clock anywhere in this engine.
* Equity splits and cash dividends are replayed from the point-in-time
  corporate-actions table before ex-date fills. Splits transform held and queued
  quantities; dividends accrue against the signed position entering the ex date.
* All pricing/fee arithmetic is delegated to the ONE shared
  :class:`~alphaforge.costs.TransactionCostModel` (buildabilityCritique.md
  §3.7) via the fill model — the engine itself never computes a cost.

Event loop per bar close ``t_k`` over the union grid of all instruments'
closes (execDesign.md §4.1)::

    0. fill orders queued at t_{k-1} against each instrument's bar opening at
       t_{k-1} (missing bar => order DROPPED and counted, never filled at a
       synthetic price)
    1. apply stored funding events with ts_funding in (t_{k-1}, t_k], mark
       price proxied by the instrument's close at t_k (documented
       approximation, execDesign.md §4.3)
    2. mark the ledger at close[t_k]; record the equity point
    3. force-flat lifecycle-confirmed delisted instruments at their last lake
       close with taker fee; a terminal history with no delisting metadata halts
       instead of being converted into a fabricated exit
    4. strategy.on_bar_close(ctx at t_k) -> target weights (fraction of
       equity, signed; ids absent from the mapping are LEFT UNTOUCHED — return
       an explicit 0.0 to flatten)
    5. discretize targets to orders (no-trade band, lot rounding, min-qty /
       min-notional filters) and queue them for t_{k+1}

Risk guards enforced in the backtest path (execDesign.md §7.1) — the engine
is the BACKTEST half of the capability matrix; the live loop's full
:class:`PreTradeChecker` (hard w/gross/net caps, price collar, margin, kill
switch) is the Phase-7 LIVE responsibility and is NOT replicated here:

* **Reduce-only exemption (curve correctness).** An order that REDUCES the
  held position toward zero (its signed delta opposes the position, shrinking
  ``|qty|``) must ALWAYS execute, however small — it bypasses the no-trade
  band AND the ``min_qty``/``min_notional`` skip and is emitted with
  ``reduce_only=True`` (execDesign.md §7.1 check 7: you must always be able to
  de-risk). Such forced reducers are counted ``flatten_residual_forced``;
  ``skipped_below_min`` is then reserved for OPENING/INCREASING orders only.
  Without this an absorbing flatten (all-zero targets, a drawdown-ladder halt)
  could orphan a sub-min residual that keeps accruing PnL/funding — an
  optimistic, dishonest curve. (Sub-LOT reducers still cannot trade: a residual
  below one ``lot_size`` is an exchange-grid limit, not a min-filter defect, and
  stays open — documented.)
* **1% ADV participation cap (cost correctness, execDesign.md §3.2/§7.1).**
  Before execution, a non-reduce-only order whose fill notional
  (``qty * open_{t+1}``) exceeds ``max_order_adv_frac * adv_quote`` (default
  0.01 = 1% of the decision bar's 30d-median ADV) is CLAMPED down to 1% of ADV
  (lot-floored) and counted ``orders_adv_clamped`` — institutional behaviour:
  trade what fits the sqrt-impact law's valid band, never skip the whole order
  or price a fill where the law is extrapolated. This keeps every impact
  evaluation inside the regime the cost model's 5%-of-ADV ``CostModelMisuse``
  tripwire bounds. Reduce-only orders bypass the cap (de-risking is never
  blocked, §7.1 check 7).

Orders decided on the final bar have no ``t+1``: they are counted as
``unfilled_final_bar`` and produce **zero fills and no exception**.

Phase 6's SignalService + optimizer will implement the :class:`Strategy`
protocol; Phase 5 ships :class:`ScriptedStrategy` (predefined ts -> weights)
for tests and :class:`RankStrategy` as a CLI demo (NOT the product).

Money is float64 in quote units at full internal precision (rounding only at
report rendering); all timestamps are epoch-ms UTC (:data:`Ms`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Final, Protocol, cast

import numpy as np
import pandas as pd
import pyarrow as pa

from alphaforge.analytics import summarize
from alphaforge.backtest.fills import BarView, FillModel, NextOpenFill, ParticipationCappedFill
from alphaforge.backtest.ledger import Ledger
from alphaforge.backtest.result import (
    FILLS_COLUMNS,
    ORDERS_COLUMNS,
    POSITIONS_COLUMNS,
    BacktestResult,
)
from alphaforge.core.calendar import TradingCalendar, calendar_for
from alphaforge.core.errors import FillUnavailableError, LookaheadError
from alphaforge.core.instruments import Instrument, InstrumentStore
from alphaforge.core.time import Ms, Timeframe
from alphaforge.core.types import (
    AssetClass,
    Fill,
    Liquidity,
    MarketType,
    OrderRequest,
    OrderType,
    Side,
)
from alphaforge.costs import MAX_ADV_PARTICIPATION, TransactionCostModel
from alphaforge.data.store.reader import PITDataReader
from alphaforge.execution.borrow import (
    BorrowDataProvider,
    RecallNotice,
    accrue_borrow_charge,
    evaluate_locate,
    incremental_short_qty,
)
from alphaforge.execution.corporate_actions import CorporateAction, CorporateActionType
from alphaforge.execution.financing import FinancingDataProvider, accrue_financing
from alphaforge.execution.market_status import MarketStatusProvider, execution_block_reason
from alphaforge.features.library.vol import ewma_vol_halflife, log_returns

__all__ = [
    "CostInputProvider",
    "EventDrivenBacktester",
    "LakeCostInputs",
    "RankStrategy",
    "ScriptedStrategy",
    "StaticCostInputs",
    "Strategy",
    "StrategyContext",
]

_LOG = logging.getLogger(__name__)

_DAY_MS: Final[int] = 86_400_000

# Corporate-action split sanity guard (2026-07-18 marking-fix campaign): a stored split
# record is applied at its ex boundary ONLY if the actual price move agrees with the
# stored ratio — |log((open_ex / close_pre) * ratio)| must not exceed this bound
# (post-split price ≈ pre-split price / ratio, so the product is ≈ 1 for a genuine
# record). 0.30 tolerates a ±~35% genuine same-day move stacked on the split while
# still rejecting a bogus 2:1 / 1:2 record whose price did not move (|log 2| ≈ 0.69 —
# the HON ratio-0.5 ghost row). An exposed record that fails aborts the run. Continuing
# with an unconverted position would treat the mechanical price discontinuity as return.
_SPLIT_SANITY_MAX_ABS_LOG: Final[float] = 0.30

# A plain cash dividend cannot economically exceed the security's observable pre-ex value per
# share. Rows above that boundary require a different lifecycle model (liquidation, redemption,
# rights, mixed consideration, or a vendor-unit correction), none of which can be represented by
# `cash_amount` alone. Fail closed instead of minting cash from a malformed vendor value. The
# 1.0 ceiling intentionally permits a full-value liquidating distribution only when the source has
# already classified it as a cash dividend; anything larger needs independently bound semantics.
_DIVIDEND_SANITY_MAX_PRE_CLOSE_MULTIPLE: Final[float] = 1.0

#: Engine bookkeeping counter names (all always present in the result, 0 if unused).
COUNTER_NAMES: Final[tuple[str, ...]] = (
    "bars_processed",
    "orders_decided",
    "fills_applied",
    "skipped_no_trade_band",
    "skipped_below_min",
    "flatten_residual_forced",
    "orders_adv_clamped",
    "orders_reduce_adv_clamped",
    "dropped_no_decision_bar",
    "dropped_missing_next_bar",
    "dropped_no_cost_inputs",
    "unfilled_final_bar",
    "forced_flat",
    "funding_events_applied",
    "borrow_charges_applied",
    "corporate_actions_applied",
    "cash_dividends_applied",
    "financing_intervals_applied",
)


# ---------------------------------------------------------------------------
# strategy surface
# ---------------------------------------------------------------------------


class StrategyContext:
    """The strategy's window onto the world at one bar close — PIT by construction.

    ``ts`` is the decision timestamp (a bar close, epoch-ms UTC). ``equity``
    is the marked equity at ``ts``; ``positions`` the open book; and
    ``instruments`` the full run universe (its keys are the tradable ids).
    History access goes through :meth:`bars`, which reads the lake with
    ``as_of = ts`` — bars closing after ``ts`` are structurally invisible, so
    a strategy cannot look ahead even by bug.

    ``asset_class`` selects the sleeve's :attr:`calendar` (the grid + annualization
    source); the default ``CRYPTO_PERP`` resolves the 24/7 calendar, byte-identical to
    the bare ``core.time`` arithmetic the spine used before the calendar was threaded.
    """

    __slots__ = ("_asset_class", "_equity", "_instruments", "_positions", "_reader", "_tf", "_ts")

    def __init__(
        self,
        *,
        reader: PITDataReader,
        tf: Timeframe,
        ts: Ms,
        equity: float,
        positions: Mapping[str, float],
        instruments: Mapping[str, Instrument],
        asset_class: AssetClass = AssetClass.CRYPTO_PERP,
    ) -> None:
        self._reader = reader
        self._tf = tf
        self._ts = ts
        self._equity = equity
        self._positions = dict(positions)
        self._instruments = dict(instruments)
        self._asset_class = asset_class

    @property
    def ts(self) -> Ms:
        """Decision timestamp — the bar close this callback is for (epoch-ms UTC)."""
        return self._ts

    @property
    def tf(self) -> Timeframe:
        """The engine's bar timeframe."""
        return self._tf

    @property
    def asset_class(self) -> AssetClass:
        """The sleeve's asset class — selects the trading :attr:`calendar`."""
        return self._asset_class

    @property
    def calendar(self) -> TradingCalendar:
        """The trading calendar for this sleeve — the ONE grid/annualization source.

        ``calendar_for(self.asset_class)``: the 24/7 calendar for crypto (whose grid
        and ``periods_per_year`` delegate to the bare ``core.time`` kernel /
        ``tf.bars_per_year``, hence byte-identical), the XNYS session calendar for
        equities (252-day basis, weekend/holiday gaps).
        """
        return calendar_for(self._asset_class)

    @property
    def equity(self) -> float:
        """Account equity marked at :attr:`ts` (quote units, full precision)."""
        return self._equity

    @property
    def positions(self) -> Mapping[str, float]:
        """Open positions as signed base-unit quantities keyed by instrument id."""
        return dict(self._positions)

    @property
    def instruments(self) -> Mapping[str, Instrument]:
        """The run universe; keys are the ids a strategy may emit targets for."""
        return dict(self._instruments)

    def bars(self, instrument_ids: Sequence[str], lookback_bars: int) -> pa.Table:
        """PIT-read the trailing ``lookback_bars`` bars ending at :attr:`ts`.

        Returns lake rows with ``ts_open in [ts - lookback_bars*Δ, ts)`` and
        ``as_of = ts`` — exactly the bars whose close is at or before the
        decision. The newest visible bar is the one closing at ``ts`` itself.
        """
        if lookback_bars <= 0:
            raise ValueError(f"lookback_bars must be > 0, got {lookback_bars}")
        start = self._ts - lookback_bars * self._tf.ms
        return self._reader.ohlcv(
            instrument_ids, start=start, end=self._ts, as_of=self._ts, tf=self._tf
        )


class Strategy(Protocol):
    """A target-weight strategy driven by the engine, one callback per bar close.

    Returns target weights as a mapping ``instrument_id -> w`` (fraction of
    current equity, signed; -0.10 = 10% short). **Ids absent from the mapping
    are left untouched** — return an explicit ``0.0`` to flatten. Returning
    ``{}`` means "hold everything". Implementations may be stateful; the
    engine calls strictly in increasing time order within one run.

    Phase 6's SignalService + portfolio optimizer implement this protocol;
    the engine does not change when they arrive.
    """

    def on_bar_close(self, ctx: StrategyContext) -> Mapping[str, float]:
        """Decide target weights at ``ctx.ts`` using only PIT data from ``ctx``."""
        ...


class ScriptedStrategy:
    """Replays a predefined ``decision_ts -> {instrument_id: weight}`` script.

    The test/bench workhorse: deterministic, no data access. Timestamps absent
    from the script return ``{}`` (hold). The script is copied at construction.
    """

    __slots__ = ("_script",)

    def __init__(self, script: Mapping[Ms, Mapping[str, float]]) -> None:
        self._script: dict[Ms, dict[str, float]] = {
            int(ts): dict(targets) for ts, targets in script.items()
        }

    def on_bar_close(self, ctx: StrategyContext) -> Mapping[str, float]:
        """Return the scripted targets for ``ctx.ts`` (empty mapping = hold)."""
        return self._script.get(ctx.ts, {})


class RankStrategy:
    """DEMO: equal-weight long the top-K by trailing return, periodic rebalance.

    THIS IS NOT THE PRODUCT — it exists so ``af backtest run`` exercises the
    full engine on real lake data before Phase 6 lands the real signal stack.
    Every ``rebalance_bars`` bars it ranks the run universe by the trailing
    ``lookback_bars``-bar close-to-close return (PIT via ``ctx.bars``), goes
    long the top ``top_k`` at ``1/top_k`` each, and explicitly zeroes every
    other ranked instrument. Instruments without a full lookback window are
    unranked (and untouched). Stateful: tracks its next rebalance time.
    """

    __slots__ = ("_lookback_bars", "_next_rebalance_ts", "_rebalance_bars", "_top_k")

    def __init__(
        self, *, top_k: int = 10, lookback_bars: int = 168, rebalance_bars: int = 168
    ) -> None:
        if top_k <= 0 or lookback_bars <= 0 or rebalance_bars <= 0:
            raise ValueError(
                f"top_k/lookback_bars/rebalance_bars must be > 0, got "
                f"{top_k}/{lookback_bars}/{rebalance_bars}"
            )
        self._top_k = top_k
        self._lookback_bars = lookback_bars
        self._rebalance_bars = rebalance_bars
        self._next_rebalance_ts: Ms | None = None

    def on_bar_close(self, ctx: StrategyContext) -> Mapping[str, float]:
        """Rebalance when due: long top-K trailing-return names, zero the rest."""
        if self._next_rebalance_ts is not None and ctx.ts < self._next_rebalance_ts:
            return {}
        self._next_rebalance_ts = ctx.ts + self._rebalance_bars * ctx.tf.ms

        ids = sorted(ctx.instruments)
        window = self._lookback_bars + 1  # need close at t and close 168 bars earlier
        tbl = ctx.bars(ids, window)
        if tbl.num_rows == 0:
            return {}
        frame = pd.DataFrame(
            {
                "instrument_id": tbl.column("instrument_id").to_pylist(),
                "ts_open": tbl.column("ts_open").cast(pa.int64()).to_pylist(),
                "close": tbl.column("close").to_pylist(),
            }
        )
        newest_open = ctx.ts - ctx.tf.ms
        oldest_open = ctx.ts - window * ctx.tf.ms
        returns: dict[str, float] = {}
        for iid, grp in frame.groupby("instrument_id", sort=True):
            by_open = dict(zip(grp["ts_open"], grp["close"], strict=True))
            c_now = by_open.get(newest_open)
            c_then = by_open.get(oldest_open)
            if c_now is None or c_then is None or c_then <= 0.0:
                continue  # incomplete window: unranked, position untouched
            returns[str(iid)] = float(c_now) / float(c_then) - 1.0
        if not returns:
            return {}
        ranked = sorted(returns, key=lambda i: (-returns[i], i))
        top = ranked[: self._top_k]
        w = 1.0 / len(top)
        return {iid: (w if iid in top else 0.0) for iid in ranked}


# ---------------------------------------------------------------------------
# cost inputs (ADV / sigma_daily as of the decision bar)
# ---------------------------------------------------------------------------


class CostInputProvider(Protocol):
    """Supplies the cost model's per-order inputs as of a decision timestamp."""

    def cost_inputs(self, instrument_id: str, decision_ts: Ms) -> tuple[float, float]:
        """Return ``(adv_quote, sigma_daily)`` as of ``decision_ts`` (NaN = unknown)."""
        ...


class StaticCostInputs:
    """Fixed ``(adv_quote, sigma_daily)`` for every instrument and timestamp.

    For tests and quick what-ifs where hand-computable fills matter more than
    realism. Production runs use :class:`LakeCostInputs`.
    """

    __slots__ = ("_adv_quote", "_sigma_daily")

    def __init__(self, *, adv_quote: float, sigma_daily: float) -> None:
        if not math.isfinite(adv_quote) or adv_quote <= 0.0:
            raise ValueError(f"adv_quote must be finite and > 0, got {adv_quote!r}")
        if not math.isfinite(sigma_daily) or sigma_daily <= 0.0:
            raise ValueError(f"sigma_daily must be finite and > 0, got {sigma_daily!r}")
        self._adv_quote = adv_quote
        self._sigma_daily = sigma_daily

    def cost_inputs(self, instrument_id: str, decision_ts: Ms) -> tuple[float, float]:
        """Return the fixed pair regardless of instrument/timestamp."""
        return (self._adv_quote, self._sigma_daily)


class LakeCostInputs:
    """ADV / sigma_daily computed from the lake with THE library formulas.

    Same definitions as the registered features in
    :mod:`alphaforge.features.library.market` (which need a full
    ``FeatureContext``/registry pipeline — deliberately not dragged into the
    backtester; the formulas are shared via the sanctioned primitive bodies
    :func:`~alphaforge.features.library.vol.ewma_vol_halflife` /
    :func:`~alphaforge.features.library.vol.log_returns`, and the ADV
    mechanics below mirror ``market._adv_quote_fn`` line for line):

    * ``adv_quote`` — 30-day rolling MEDIAN of daily quote volume (USDT/day);
      daily = 1h ``quote_volume`` summed per complete UTC day; the value at
      decision time ``t`` covers the last 30 UTC days fully completed by ``t``.
    * ``sigma_daily`` — zero-mean EWMA vol of 1h log returns (halflife 240
      bars = 10 days) scaled by ``sqrt(24)``.

    Everything is computed ONCE over ``[start - warmup, end)`` at
    construction; per-decision lookups are then positional. The bulk read uses
    ``as_of = end`` — PIT-safe for OHLCV because bar *contents* are immutable
    (only quality-flag visibility evolves, and flags are not consumed here);
    the per-decision truncation is positional (only rows with
    ``ts_open <= decision_ts - Δ`` ever feed a lookup at ``decision_ts``).

    Sleeve-aware: the per-bar→daily sigma scale (``sqrt(bars_per_day)`` =
    ``_DAY_MS // anchor_tf.ms``: 24 for H1, 1 for D1), the grid, and the warmup are
    resolved from the sleeve ``anchor_tf`` + ``calendar``. On H1/CRYPTO_PERP every value
    is byte-identical to the prior hard-coded 1h math (calendar passthrough, sqrt(24),
    33-day warmup); D1/EQUITY walks XNYS sessions and warms up the 240-SESSION sigma.
    """

    ADV_DAYS: Final[int] = 30
    SIGMA_HALFLIFE_BARS: Final[float] = 240.0
    #: H1 warmup: 30 complete UTC days (ADV) + partial decision day + sigma headroom. The D1
    #: warmup is far longer (the 240-session sigma); it is derived in __init__ per calendar.
    WARMUP_MS: Final[int] = 33 * _DAY_MS

    __slots__ = ("_adv", "_cal", "_sigma", "_tf")

    def __init__(
        self,
        reader: PITDataReader,
        instrument_ids: Sequence[str],
        *,
        start: Ms,
        end: Ms,
        anchor_tf: Timeframe = Timeframe.H1,
        calendar: TradingCalendar | None = None,
    ) -> None:
        self._tf = anchor_tf
        cal = calendar if calendar is not None else calendar_for(AssetClass.CRYPTO_PERP)
        self._cal = cal  # retained for the calendar-aware decision-bar lookup (cost_inputs)
        # bars per calendar day for the sigma per-bar -> daily scaling: 24 (H1), 1 (D1).
        bars_per_day = _DAY_MS // anchor_tf.ms
        # Warmup must cover the EWMA-sigma halflife (240 BARS) + the 30-bar/day ADV window in
        # CALENDAR time. For H1 the 30-day ADV (33 days) dominates the 10-day sigma -> 33 days
        # (UNCHANGED, crypto byte-identical). For D1 the 240-SESSION sigma dominates (~1 year),
        # so size the warmup to ~(240 + 30 + buffer) sessions of calendar time via the calendar.
        if anchor_tf is Timeframe.H1:
            warmup_ms = self.WARMUP_MS
        else:
            warmup_bars = self.SIGMA_HALFLIFE_BARS + self.ADV_DAYS + 30.0
            cal_days_per_bar = 365.0 / cal.periods_per_year(anchor_tf)
            warmup_ms = math.ceil(warmup_bars * cal_days_per_bar) * _DAY_MS
        read_start = start - warmup_ms
        tbl = reader.ohlcv(instrument_ids, start=read_start, end=end, as_of=end, tf=self._tf)
        grid = cal.expected_bar_opens(read_start, end, self._tf)
        index = pd.Index(grid, dtype="int64", name="ts_open")
        ids = sorted(dict.fromkeys(instrument_ids))
        if tbl.num_rows == 0:
            nan = pd.DataFrame(np.nan, index=index, columns=ids, dtype="float64")
            self._sigma = nan
            self._adv = nan.copy()
            return
        frame = pd.DataFrame(
            {
                "instrument_id": tbl.column("instrument_id").to_pylist(),
                "ts_open": tbl.column("ts_open").cast(pa.int64()).to_pylist(),
                "close": tbl.column("close").to_pylist(),
                "quote_volume": tbl.column("quote_volume").to_pylist(),
            }
        )
        closes = (
            frame.pivot(index="ts_open", columns="instrument_id", values="close")
            .reindex(index=index, columns=ids)
            .astype("float64")
        )
        qv = (
            frame.pivot(index="ts_open", columns="instrument_id", values="quote_volume")
            .reindex(index=index, columns=ids)
            .astype("float64")
        )

        # sigma_daily = EWMA_halflife240(per-bar log returns) * sqrt(bars_per_day): sqrt(24)
        # for H1 (per-market.py), sqrt(1)=1 for D1 (a session's return IS already daily).
        self._sigma = ewma_vol_halflife(log_returns(closes), self.SIGMA_HALFLIFE_BARS) * math.sqrt(
            float(bars_per_day)
        )

        # adv_quote = rolling 30-day MEDIAN of complete-UTC-day quote volume,
        # mapped to each bar's last fully-completed day (market.py mechanics).
        grid_arr = np.asarray(grid, dtype="int64")
        day_key = (grid_arr // _DAY_MS) * _DAY_MS
        daily = qv.groupby(day_key).sum(min_count=1)
        first_full_day = -(-read_start // _DAY_MS) * _DAY_MS  # ceil to next UTC day
        day_index = np.asarray(daily.index, dtype="int64")
        complete = (day_index >= first_full_day) & (day_index + _DAY_MS <= end)
        daily = daily.loc[complete]
        med = daily.rolling(self.ADV_DAYS, min_periods=self.ADV_DAYS).median()
        last_day = ((grid_arr + self._tf.ms) // _DAY_MS) * _DAY_MS - _DAY_MS
        adv = med.reindex(last_day)
        adv.index = index
        self._adv = adv.astype("float64")

    def cost_inputs(self, instrument_id: str, decision_ts: Ms) -> tuple[float, float]:
        """``(adv_quote, sigma_daily)`` from the decision bar (the bar that CLOSED at
        the decision instant — its open is ``floor_bar(decision_ts - 1, tf)``).

        The prior-bar open is resolved through the CALENDAR, not naive
        ``decision_ts - tf.ms`` arithmetic: on the contiguous 24/7 crypto grid the two
        are byte-identical (every hour is a grid point), but on the gappy XNYS session
        grid ``decision_ts - 1 day`` of a Monday/post-holiday decision lands on a
        non-session (a weekend/holiday) and would spuriously miss the index — silently
        dropping every Monday and post-holiday trade. ``floor_bar`` hops to the true
        prior session (mirrors the engine loop's ``prev_open`` at the decision instant).

        Returns ``(nan, nan)`` for unknown instruments, off-grid timestamps,
        or insufficient history — the engine drops such orders loudly rather
        than guessing a cost.
        """
        ts_open = self._cal.floor_bar(decision_ts - 1, self._tf)
        if instrument_id not in self._sigma.columns or ts_open not in self._sigma.index:
            return (math.nan, math.nan)
        adv = cast(float, self._adv.at[ts_open, instrument_id])
        sigma = cast(float, self._sigma.at[ts_open, instrument_id])
        return (float(adv), float(sigma))


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------


def _finite_positive(value: float) -> bool:
    """True iff ``value`` is a finite float > 0."""
    return math.isfinite(value) and value > 0.0


class EventDrivenBacktester:
    """Bar-close event loop over PIT lake data (module docstring = the contract).

    Args:
        reader: PIT lake reader; the only data access path (strategy included).
        instruments: SCD2 instrument store. Static trading attributes
            (lot/min filters, funding interval, market type) are resolved
            ``as_of = end`` of each run — a documented v1 simplification
            (per-bar SCD2 resolution arrives with the execution layer).
        cost_model: THE shared cost authority; consumed only through the fill
            model, never reimplemented (buildabilityCritique.md §3.7).
        tf: Bar timeframe (default 1h). Must be the sleeve's anchor TF — the grid
            and bar-arithmetic step the engine values on.
        asset_class: Selects the trading ``calendar`` (when ``calendar`` is not
            passed explicitly) and is forwarded to :class:`StrategyContext`. The
            default ``CRYPTO_PERP`` resolves the 24/7 calendar, so the grid /
            next-bar / floor-bar arithmetic reduces to the bare ``ts_open ± tf.ms``
            steps the engine used before the calendar was threaded — byte-identical.
        calendar: The :class:`~alphaforge.core.calendar.TradingCalendar` whose
            sessions define the bar grid, the next-bar fill target, and the
            prior-bar mark/funding lookup. Defaults to ``calendar_for(asset_class)``;
            crypto resolves the 24/7 calendar (grid step ``+tf.ms``, prior open
            ``-tf.ms``), equities the XNYS session calendar (weekend/holiday hops).
        fill_model: Defaults to :class:`NextOpenFill` over ``cost_model``.
        cost_inputs: ADV/sigma provider; defaults to a per-run
            :class:`LakeCostInputs` (1h-only — other timeframes must inject).
        borrow_data: Optional point-in-time security-lending provider. When present,
            incremental short orders require quantity-bounded locates and open shorts
            accrue the security-level fee covering each interval. Known recalls create
            retryable cover orders and escalate at their deadline. When absent, the existing
            general-collateral path is unchanged.
        market_status: Optional coverage-complete point-in-time venue/instrument status
            provider. Halts, outages and auction-only states block fills; close-only
            permits only reduce-only orders.
        financing_data: Optional PIT cash-financing schedule. When enabled, every
            holding interval requires one quote that covers the full interval. Positive
            unrestricted cash earns the credit rate, negative cash pays the debit rate,
            and cash collateralized by short market value receives the explicit
            short-proceeds rate. Missing coverage halts the run.
        no_trade_band_frac: Suppress orders with ``|Δnotional| <
            band * equity`` (default 0.001 = 10 bps of equity; mirrors
            ``portfolio.no_trade_band``). Risk-REDUCING orders bypass this band
            (you must always be able to de-risk). Full exchange-filter
            discretization is Phase 6; v1 applies lot-size rounding toward zero
            plus ``min_qty``/``min_notional`` filters from the instrument record
            (reducing orders bypass those too — see the module docstring).
        max_order_adv_frac: Per-order ADV participation cap as a fraction of
            the decision bar's 30d-median ADV (default 0.01 = 1%; matches
            ``RiskCfg.max_order_adv_frac`` / ``RiskLimits.max_adv_participation``
            and the sqrt-impact validity guard, execDesign.md §3.2/§7.1). A
            non-reduce-only order whose fill notional exceeds this is CLAMPED to
            the cap (lot-floored) and counted ``orders_adv_clamped``, keeping
            the impact law in its valid regime; reduce-only orders are exempt.
        config_echo: Extra key/values copied verbatim into the result config
            (callers pass cost parameters etc. for full reproducibility — the
            engine cannot introspect the cost model's internals and will not
            guess).
        verified_split_events: Optional exact-event verification map keyed by
            ``(instrument_id, ex_date_ms)`` with the issuer-confirmed ratio. A
            matching tuple may bypass only the price-gap heuristic; every other
            split retains the fail-closed sanity guard.
    """

    __slots__ = (
        "_asset_class",
        "_borrow_data",
        "_calendar",
        "_clamp_reduce_only_adv",
        "_config_echo",
        "_cost_inputs",
        "_cost_model",
        "_fill_model",
        "_financing_data",
        "_instruments",
        "_market_status",
        "_max_order_adv_frac",
        "_no_trade_band_frac",
        "_reader",
        "_tf",
        "_verified_split_events",
    )

    def __init__(
        self,
        reader: PITDataReader,
        instruments: InstrumentStore,
        cost_model: TransactionCostModel,
        *,
        tf: Timeframe = Timeframe.H1,
        asset_class: AssetClass = AssetClass.CRYPTO_PERP,
        calendar: TradingCalendar | None = None,
        fill_model: FillModel | None = None,
        cost_inputs: CostInputProvider | None = None,
        borrow_data: BorrowDataProvider | None = None,
        market_status: MarketStatusProvider | None = None,
        financing_data: FinancingDataProvider | None = None,
        no_trade_band_frac: float = 0.001,
        max_order_adv_frac: float = 0.01,
        clamp_reduce_only_adv: bool = False,
        config_echo: Mapping[str, object] | None = None,
        verified_split_events: Mapping[tuple[str, Ms], float] | None = None,
    ) -> None:
        if not math.isfinite(no_trade_band_frac) or no_trade_band_frac < 0.0:
            raise ValueError(
                f"no_trade_band_frac must be finite and >= 0, got {no_trade_band_frac!r}"
            )
        if not math.isfinite(max_order_adv_frac) or not (
            0.0 < max_order_adv_frac <= MAX_ADV_PARTICIPATION
        ):
            raise ValueError(
                "max_order_adv_frac must be finite and in (0, "
                f"{MAX_ADV_PARTICIPATION}], got {max_order_adv_frac!r}"
            )
        if cost_inputs is None and tf not in (Timeframe.H1, Timeframe.D1):
            raise ValueError(
                "the default LakeCostInputs provider supports H1 (crypto) and D1 (equity) "
                f"sleeves; inject cost_inputs for tf={tf.value}"
            )
        self._reader = reader
        self._instruments = instruments
        self._cost_model = cost_model
        self._tf = tf
        self._asset_class = asset_class
        self._calendar: TradingCalendar = (
            calendar_for(asset_class) if calendar is None else calendar
        )
        self._fill_model: FillModel = NextOpenFill(cost_model) if fill_model is None else fill_model
        self._cost_inputs = cost_inputs
        self._borrow_data = borrow_data
        self._market_status = market_status
        self._financing_data = financing_data
        self._no_trade_band_frac = no_trade_band_frac
        self._max_order_adv_frac = max_order_adv_frac
        # Opt-in (default OFF = byte-identical, reduce-only bypasses the cap as the golden master
        # locks). ON for thin-universe profiles (equity/managed-futures), where a large flatten on
        # a low-ADV instrument would otherwise crash the sqrt-impact model: it clamps the de-risk
        # at the 5%-ADV validity edge and closes the remainder over the next bars.
        self._clamp_reduce_only_adv = clamp_reduce_only_adv
        self._config_echo: dict[str, object] = dict(config_echo or {})
        self._verified_split_events: dict[tuple[str, Ms], float] = dict(
            verified_split_events or {}
        )
        for (iid, ex_date), ratio in self._verified_split_events.items():
            if not iid or not isinstance(ex_date, int) or not _finite_positive(float(ratio)):
                raise ValueError(
                    "verified_split_events requires non-empty instrument ids, integer ex-date "
                    "milliseconds, and finite positive ratios"
                )
        if self._verified_split_events:
            canonical = json.dumps(
                sorted(
                    (iid, ex_date, float(ratio))
                    for (iid, ex_date), ratio in self._verified_split_events.items()
                ),
                separators=(",", ":"),
            ).encode()
            self._config_echo["verified_split_events_count"] = len(
                self._verified_split_events
            )
            self._config_echo["verified_split_events_sha256"] = hashlib.sha256(
                canonical
            ).hexdigest()

    # ------------------------------------------------------------------- run

    def run(
        self,
        strategy: Strategy,
        instrument_ids: Sequence[str],
        *,
        start: Ms,
        end: Ms,
        initial_cash: float = 100_000.0,
    ) -> BacktestResult:
        """Run ``strategy`` over ``[start, end)`` and return the full result.

        ``start``/``end`` must be ``tf``-aligned epoch-ms UTC; bars with
        ``ts_open in [start, end)`` define the run. Raises ``ValueError`` on
        unknown instruments, an empty/too-short bar set (< 2 bar closes), or
        malformed bounds. Propagates :class:`LookaheadError` (contract breach
        — a bug, never a data condition) and
        :class:`~alphaforge.core.errors.CostModelMisuse` (an order escaped
        the sizing guards) — both must halt the run, not be smoothed over.
        """
        ids = list(dict.fromkeys(instrument_ids))
        if not ids:
            raise ValueError("instrument_ids must be non-empty")
        if end <= start:
            raise ValueError(f"end ({end}) must be > start ({start})")
        if start % self._tf.ms or end % self._tf.ms:
            raise ValueError(
                f"start/end must be {self._tf.value}-aligned epoch-ms, got {start}/{end}"
            )
        if not _finite_positive(initial_cash):
            raise ValueError(f"initial_cash must be finite and > 0, got {initial_cash!r}")

        insts: dict[str, Instrument] = {}
        for iid in ids:
            inst = self._instruments.get(iid, as_of=end)
            if inst is None:
                # SCD2 validity starts at first INGESTION time, so a run that
                # ends before the store's first record resolves no version at
                # as_of=end. Fall back to the EARLIEST recorded version (the
                # nearest-in-time record for a historical run) — a documented
                # v1 simplification consistent with the static as_of=end
                # attribute resolution above; truly unknown ids still raise.
                history = self._instruments.history(iid)
                if not history:
                    raise ValueError(f"unknown instrument {iid!r} (no SCD2 record)")
                inst = history[0][2]
            insts[iid] = inst
        market_status_coverage_hash: str | None = None
        if self._market_status is not None:
            market_status_coverage_hash = self._market_status.require_coverage(
                tuple(ids), start=start, end=end
            )
        financing_currency: str | None = None
        if self._financing_data is not None:
            currencies = {inst.quote for inst in insts.values()}
            if len(currencies) != 1:
                raise ValueError(
                    "financing replay requires one ledger currency; got "
                    f"{sorted(currencies)!r}"
                )
            financing_currency = next(iter(currencies))

        bars = self._load_bars(ids, start=start, end=end)
        # The decision grid is the set of bar *closes*, each modelled as the open
        # of the bar an order would fill into: the close of bar ``o`` is
        # ``calendar.next_bar_open(o, tf)`` — for the 24/7 crypto calendar this is
        # exactly ``o + tf.ms`` (byte-identical to the prior bare-step grid), and for
        # an XNYS session calendar a Friday close maps to the following Monday open so
        # the fill target ``bars[iid].get(decision_ts)`` lands on a real session bar
        # instead of a phantom weekend slot.
        grid = sorted(
            {
                self._calendar.next_bar_open(ts_open, self._tf)
                for rows in bars.values()
                for ts_open in rows
            }
        )
        if len(grid) < 2:
            raise ValueError(
                f"need at least 2 bar closes in [{start}, {end}) for {ids}, got {len(grid)}"
            )
        last_close_of: dict[str, Ms] = {
            iid: self._calendar.next_bar_open(max(rows), self._tf)
            for iid, rows in bars.items()
            if rows
        }
        funding_events = self._load_funding(insts, start=start, end=end)
        funding_ptr: dict[str, int] = dict.fromkeys(funding_events, 0)
        # Corporate-action splits (EQUITY sleeve only; 2026-07-18 marking-fix campaign):
        # a position held across a split ex-date must convert (qty·ratio, avg/ratio) or
        # the raw-close mark fabricates a 1/ratio-sized phantom P&L jump (the ALIT
        # 1-for-20 defect). Crypto/MF paths are untouched: the load is gated on the
        # asset class AND their lakes carry no corporate_actions partitions.
        corporate_actions = (
            self._load_corporate_actions(ids, start=start, end=end)
            if self._asset_class is AssetClass.EQUITY
            else {}
        )
        corporate_action_ptr: dict[str, int] = dict.fromkeys(corporate_actions, 0)
        provider: CostInputProvider = (
            LakeCostInputs(
                self._reader,
                ids,
                start=start,
                end=end,
                anchor_tf=self._tf,
                calendar=self._calendar,
            )
            if self._cost_inputs is None
            else self._cost_inputs
        )

        ledger = Ledger(initial_cash, insts)
        counters: dict[str, int] = dict.fromkeys(COUNTER_NAMES, 0)
        if isinstance(self._fill_model, ParticipationCappedFill):
            counters.update(
                {
                    "dropped_no_bar_liquidity": 0,
                    "partial_fill_orders": 0,
                    "partial_fill_residual_canceled": 0,
                }
            )
        if self._borrow_data is not None:
            counters.update(
                {
                    "borrow_locates_denied": 0,
                    "borrow_locates_partial": 0,
                    "borrow_quotes_missing": 0,
                    "borrow_recall_fills_applied": 0,
                    "borrow_recalls_queued": 0,
                    "dynamic_borrow_charges_applied": 0,
                    "forced_buy_ins_queued": 0,
                }
            )
        if self._market_status is not None:
            counters.update(
                {
                    "market_status_blocks": 0,
                    "market_status_close_only_blocks": 0,
                }
            )
        order_records: list[dict[str, object]] = []
        position_records: list[dict[str, object]] = []
        reason_by_order: dict[str, str] = {}
        record_by_order: dict[str, dict[str, object]] = {}
        last_close: dict[str, float] = {}
        pending: list[OrderRequest] = []
        recall_remaining: dict[tuple[str, Ms, Ms, Ms], float] = {}
        recall_keys_by_order: dict[str, tuple[tuple[str, Ms, Ms, Ms], ...]] = {}
        final_close = grid[-1]
        # Short-borrow carry: a single general-collateral per-day rate (0 for a crypto
        # perp book -> the accrual below is skipped, byte-identical). Accrued on the SHORT
        # leg over the CALENDAR days between consecutive bars (weekends included).
        borrow_rate = self._cost_model.borrow_frac_per_day()
        prev_bar_ts: Ms | None = None

        for t in grid:
            counters["bars_processed"] += 1
            # The bar that just closed at decision instant ``t`` is the session bar
            # whose open is ``floor_bar(t - 1, tf)``: for the 24/7 calendar this is
            # exactly ``t - tf.ms`` (byte-identical), and for an XNYS session calendar
            # the prior open of a Monday decision is the preceding Friday session
            # (never a phantom weekend slot at ``t - tf.ms``).
            prev_open = self._calendar.floor_bar(t - 1, self._tf)

            # (0a-0) cash financing for the interval held into this bar. The
            # quote selected at the interval start must cover the complete span;
            # no overnight/weekend rate is forward-filled across an invalidity edge.
            if (
                self._financing_data is not None
                and financing_currency is not None
                and prev_bar_ts is not None
            ):
                financing_quote = self._financing_data.quote(
                    financing_currency, as_of=prev_bar_ts
                )
                if financing_quote is None:
                    raise ValueError(
                        f"missing PIT financing quote for {financing_currency!r} "
                        f"at {prev_bar_ts}"
                    )
                short_market_value = sum(
                    abs(pos.qty) * last_close[pos.instrument_id]
                    for pos in ledger.positions().values()
                    if pos.qty < 0.0 and pos.instrument_id in last_close
                )
                accrual = accrue_financing(
                    financing_quote,
                    cash_balance=ledger.cash,
                    short_market_value=short_market_value,
                    start_ts=prev_bar_ts,
                    end_ts=t,
                    decision_ts=prev_bar_ts,
                )
                ledger.apply_financing(accrual)
                counters["financing_intervals_applied"] += 1

            # (0a) short-borrow carry for the interval [prev_bar_ts, t] on the position
            #      held INTO this bar (BEFORE today's fills change it), marked at the prior
            #      close. days is the CALENDAR gap (1 over a weekday, 3 across a weekend),
            #      so borrow accrues over wall-clock time. Skipped entirely when borrow_rate
            #      is 0 (every crypto book) -> byte-identical.
            if self._borrow_data is not None and prev_bar_ts is not None:
                for iid, pos in ledger.positions().items():
                    if pos.qty >= 0.0:
                        continue
                    mark = last_close.get(iid)
                    if mark is None:
                        continue
                    quote = self._borrow_data.quote(iid, as_of=prev_bar_ts)
                    if quote is None:
                        raise ValueError(
                            f"missing PIT borrow quote for open short {iid!r} at {prev_bar_ts}"
                        )
                    accrue_borrow_charge(
                        quote,
                        short_qty=abs(pos.qty),
                        mark_price=mark,
                        start_ts=prev_bar_ts,
                        end_ts=t,
                        decision_ts=prev_bar_ts,
                    )
                    days = (t - prev_bar_ts) / _DAY_MS
                    rate_per_day = quote.annual_fee_bps * 1e-4 / 365.0
                    ledger.apply_borrow(iid, t, rate_per_day, days, mark)
                    counters["borrow_charges_applied"] += 1
                    counters["dynamic_borrow_charges_applied"] += 1
            elif borrow_rate > 0.0 and prev_bar_ts is not None:
                days = (t - prev_bar_ts) / _DAY_MS
                for iid, pos in ledger.positions().items():
                    if pos.qty < 0.0:
                        mark = last_close.get(iid)
                        if mark is not None:
                            ledger.apply_borrow(iid, t, borrow_rate, days, mark)
                            counters["borrow_charges_applied"] += 1
            prev_bar_ts = t

            # (0b) corporate-action split conversion at the ex boundary (EQUITY sleeve
            #      only — `splits` is empty otherwise). Runs AFTER the borrow accrual
            #      (which correctly used the pre-split qty x pre-split mark) and BEFORE
            #      today's fills (which price at the ex bar's post-split open). Each
            #      record is sanity-checked against the actual price move; a record
            #      whose price did not move (the bogus HON ratio-0.5 row) is logged
            #      and skipped, never applied.
            if corporate_actions:
                pending = self._apply_due_corporate_actions(
                    corporate_actions,
                    corporate_action_ptr,
                    bars,
                    ledger,
                    last_close,
                    pending,
                    record_by_order,
                    counters,
                    prev_open,
                    t,
                )

            # (0) fill orders queued at the previous close against the bar
            #     opening at that close (ts_open == decision_ts).
            for order in pending:
                filled_qty = self._execute(
                    order, insts, bars, provider, ledger, counters, record_by_order
                )
                recall_keys = recall_keys_by_order.pop(order.client_order_id, ())
                qty_to_allocate = filled_qty
                for key in recall_keys:
                    allocated = min(recall_remaining[key], qty_to_allocate)
                    recall_remaining[key] -= allocated
                    qty_to_allocate -= allocated
                    if allocated > 0.0:
                        counters["borrow_recall_fills_applied"] += 1
                    if qty_to_allocate <= 0.0:
                        break
            pending = []

            # (1) stored funding events with ts_funding in (prev_close, t];
            #     pointer per instrument is monotone, so "<= t" is exact.
            for iid, events in funding_events.items():
                ptr = funding_ptr[iid]
                while ptr < len(events) and events[ptr][0] <= t:
                    ts_funding, rate = events[ptr]
                    ptr += 1
                    bar = bars[iid].get(prev_open)
                    mark = bar.close if bar is not None else last_close.get(iid)
                    if mark is not None:  # no close ever seen => provably flat: no-op
                        ledger.apply_funding(iid, ts_funding, rate, mark)
                funding_ptr[iid] = ptr

            # (2) mark at close[t] (last-known closes carry across per-
            #     instrument gaps so open positions always mark — documented).
            for iid, rows in bars.items():
                bar = rows.get(prev_open)
                if bar is not None:
                    last_close[iid] = bar.close
            state = ledger.mark(
                {iid: last_close[iid] for iid in ledger.positions()},
                t,
            )

            # (3) administratively close only lifecycle-confirmed delistings. A terminal
            #     history for an instrument whose metadata still says active is a stale/
            #     missing-price defect, not permission to fabricate a same-bar exit.
            if t < final_close:
                for pos in list(ledger.positions().values()):
                    if last_close_of.get(pos.instrument_id) != t:
                        continue
                    inst = insts[pos.instrument_id]
                    if inst.delisted_ts is None or inst.delisted_ts > final_close:
                        raise ValueError(
                            f"terminal price history for active instrument "
                            f"{pos.instrument_id!r} ends at {t}; refusing to infer a "
                            "delisting or fabricate an exit without lifecycle metadata"
                        )
                    self._force_flat(
                        pos.instrument_id,
                        inst,
                        ledger,
                        last_close[pos.instrument_id],
                        t,
                        counters,
                        order_records,
                        reason_by_order,
                    )
                state = ledger.mark(
                    {iid: last_close[iid] for iid in ledger.positions()},
                    t,
                )
            equity = state.equity_quote

            # snapshot positions at this close (post administrative actions)
            for pos in state.positions:
                mark = last_close[pos.instrument_id]
                position_records.append(
                    {
                        "ts": t,
                        "instrument_id": pos.instrument_id,
                        "qty": pos.qty,
                        "mark": mark,
                        "weight": pos.qty * mark / equity,
                        "unreal_pnl": pos.qty * (mark - pos.avg_entry_price),
                    }
                )

            # (4) strategy decision at close t
            ctx = StrategyContext(
                reader=self._reader,
                tf=self._tf,
                ts=t,
                equity=equity,
                positions={p.instrument_id: p.qty for p in state.positions},
                instruments=insts,
                asset_class=self._asset_class,
            )
            targets = strategy.on_bar_close(ctx)

            # (5) discretize targets -> orders queued for the next close
            pending = self._discretize(
                targets,
                insts,
                bars,
                ledger,
                equity,
                t,
                prev_open,
                counters,
                order_records,
                reason_by_order,
                record_by_order,
            )
            if self._borrow_data is not None:
                pending = self._queue_borrow_recalls(
                    pending,
                    ledger,
                    insts,
                    t,
                    last_close,
                    recall_remaining,
                    recall_keys_by_order,
                    counters,
                    order_records,
                    reason_by_order,
                    record_by_order,
                )

        # Orders decided on the final bar have no t+1: zero fills, loud counter.
        for order in pending:
            counters["unfilled_final_bar"] += 1
            record_by_order[order.client_order_id]["status"] = "unfilled_final_bar"

        return self._build_result(
            ledger,
            insts,
            ids,
            start=start,
            end=end,
            initial_cash=initial_cash,
            counters=counters,
            order_records=order_records,
            position_records=position_records,
            reason_by_order=reason_by_order,
            market_status_coverage_hash=market_status_coverage_hash,
        )

    # ----------------------------------------------------------- loop pieces

    def _execute(
        self,
        order: OrderRequest,
        insts: Mapping[str, Instrument],
        bars: Mapping[str, dict[Ms, BarView]],
        provider: CostInputProvider,
        ledger: Ledger,
        counters: dict[str, int],
        record_by_order: dict[str, dict[str, object]],
    ) -> float:
        """Fill one queued order against the bar opening at its decision close.

        Missing next bar (per-instrument gap or delisting): the order is
        DROPPED and counted — never filled at a synthetic price. Missing
        ADV/sigma: dropped and counted — never a guessed cost.

        ADV participation cap (execDesign.md §3.2/§7.1): an order whose fill notional
        (``qty * open_{t+1}``) exceeds its ADV cap is CLAMPED down (lot-floored) so the
        sqrt-impact law is only ever evaluated inside its valid band. Build (non-reduce) orders
        cap at ``max_order_adv_frac * adv_quote`` (1%, counted ``orders_adv_clamped``); reduce-only
        de-risking orders get a wider cap at the impact-law edge (5% of ADV, counted
        ``orders_reduce_adv_clamped``) so a flatten exits fast but a thin-instrument flatten closes
        over a few bars instead of crashing the cost model.
        """
        record = record_by_order[order.client_order_id]
        if self._market_status is not None:
            status = self._market_status.status(order.instrument_id, as_of=order.decision_ts)
            if status is None:
                raise ValueError(
                    f"missing PIT market-status coverage for {order.instrument_id!r} "
                    f"at {order.decision_ts}"
                )
            block_reason = execution_block_reason(status, order)
            if block_reason is not None:
                counters["market_status_blocks"] += 1
                if block_reason == "close_only_blocks_risk_increase":
                    counters["market_status_close_only_blocks"] += 1
                record["status"] = f"blocked_{block_reason}"
                return 0.0
        next_bar = bars[order.instrument_id].get(order.decision_ts)
        if next_bar is None:
            counters["dropped_missing_next_bar"] += 1
            record["status"] = "dropped_missing_next_bar"
            return 0.0
        adv_quote, sigma_daily = provider.cost_inputs(order.instrument_id, order.decision_ts)
        if not _finite_positive(adv_quote) or not _finite_positive(sigma_daily):
            counters["dropped_no_cost_inputs"] += 1
            record["status"] = "dropped_no_cost_inputs"
            return 0.0
        inst = insts[order.instrument_id]
        order = self._clamp_to_adv(order, inst, next_bar.open, adv_quote, counters, record)
        try:
            fill = self._fill_model.fill(
                order,
                inst,
                next_bar,
                adv_quote=adv_quote,
                sigma_daily=sigma_daily,
            )
        except FillUnavailableError:
            counters["dropped_no_bar_liquidity"] = (
                counters.get("dropped_no_bar_liquidity", 0) + 1
            )
            record["status"] = "dropped_no_bar_liquidity"
            return 0.0
        if fill.ts < order.decision_ts:  # defense in depth against custom fill models
            raise LookaheadError(
                f"fill model returned ts {fill.ts} before decision {order.decision_ts} "
                f"for {order.instrument_id!r}"
            )
        ledger.apply_fill(fill)
        counters["fills_applied"] += 1
        if fill.qty < order.qty:
            counters["partial_fill_orders"] = counters.get("partial_fill_orders", 0) + 1
            counters["partial_fill_residual_canceled"] = (
                counters.get("partial_fill_residual_canceled", 0) + 1
            )
            record["qty"] = fill.qty
            record["status"] = "partial_fill_residual_canceled"
        else:
            record["status"] = "filled"
        return fill.qty

    def _queue_borrow_recalls(
        self,
        pending: list[OrderRequest],
        ledger: Ledger,
        insts: Mapping[str, Instrument],
        t: Ms,
        last_close: Mapping[str, float],
        recall_remaining: dict[tuple[str, Ms, Ms, Ms], float],
        recall_keys_by_order: dict[str, tuple[tuple[str, Ms, Ms, Ms], ...]],
        counters: dict[str, int],
        order_records: list[dict[str, object]],
        reason_by_order: dict[str, str],
        record_by_order: dict[str, dict[str, object]],
    ) -> list[OrderRequest]:
        """Replace conflicting targets with retryable recall-cover orders."""
        assert self._borrow_data is not None
        notices_by_instrument: dict[
            str, list[tuple[tuple[str, Ms, Ms, Ms], RecallNotice]]
        ] = {}
        for notice in self._borrow_data.recalls_known(as_of=t):
            key = (
                notice.instrument_id,
                notice.event_ts,
                notice.available_at,
                notice.cover_deadline,
            )
            recall_remaining.setdefault(key, notice.recalled_qty)
            if recall_remaining[key] > 0.0:
                notices_by_instrument.setdefault(notice.instrument_id, []).append((key, notice))

        positions = ledger.positions()
        out = list(pending)
        for iid, keyed_notices in sorted(notices_by_instrument.items()):
            pos = positions.get(iid)
            current_short = 0.0 if pos is None else max(-pos.qty, 0.0)
            if current_short == 0.0:
                for key, _ in keyed_notices:
                    recall_remaining[key] = 0.0
                continue
            if iid not in insts or iid not in last_close:
                raise ValueError(f"borrow recall references unmarkable instrument {iid!r}")
            keys = tuple(key for key, _ in keyed_notices)
            recalled_qty = min(sum(recall_remaining[key] for key in keys), current_short)
            existing = next((order for order in out if order.instrument_id == iid), None)
            existing_buy_qty = (
                existing.qty if existing is not None and existing.side is Side.BUY else 0.0
            )
            qty = max(recalled_qty, existing_buy_qty)
            steps = math.floor(qty / insts[iid].lot_size + 1e-9)
            qty = steps * insts[iid].lot_size
            if qty <= 0.0:
                continue
            if existing is not None:
                out.remove(existing)
                record_by_order[existing.client_order_id]["status"] = "superseded_by_borrow_recall"
            forced = any(notice.cover_deadline <= t for _, notice in keyed_notices)
            reason = "forced_buy_in" if forced else "borrow_recall"
            client_order_id = f"bt-{t}-{iid}-{reason}"
            order = OrderRequest(
                client_order_id=client_order_id,
                instrument_id=iid,
                side=Side.BUY,
                qty=qty,
                order_type=OrderType.MARKET,
                reduce_only=True,
                decision_ts=t,
                decision_price=last_close[iid],
                reason=reason,
            )
            record: dict[str, object] = {
                "decision_ts": t,
                "instrument_id": iid,
                "side": Side.BUY.value,
                "qty": qty,
                "decision_price": last_close[iid],
                "status": "queued",
                "client_order_id": client_order_id,
            }
            out.append(order)
            order_records.append(record)
            record_by_order[client_order_id] = record
            reason_by_order[client_order_id] = reason
            recall_keys_by_order[client_order_id] = keys
            counters["borrow_recalls_queued"] += 1
            if forced:
                counters["forced_buy_ins_queued"] += 1
        return out

    def _clamp_to_adv(
        self,
        order: OrderRequest,
        inst: Instrument,
        ref_price: float,
        adv_quote: float,
        counters: dict[str, int],
        record: dict[str, object],
    ) -> OrderRequest:
        """Clamp an order's qty so its fill notional stays within 1% of ADV.

        Institutional behaviour (execDesign.md §3.2/§7.1): rather than skip an
        oversized order or price a fill where the sqrt-impact law is
        extrapolated, trade exactly the cap. The fill notional is sized at the
        fill reference (``qty * ref_price``, ``ref_price = open_{t+1}``), so the
        clamped notional ``qty_cap * ref_price <= max_order_adv_frac * adv_quote``
        by construction and the impact term is always evaluated in-band.

        Reduce-only (de-risking) orders are never BLOCKED, but they get a wider cap, not an
        exemption: they clamp at the sqrt-impact law's validity edge (``MAX_ADV_PARTICIPATION`` =
        5% of ADV, ~5x the 1% build cap) so a flatten exits fast but the cost model still only
        ever prices in-band. A flatten larger than 5% of ADV closes over the next bars (realistic:
        you exit fast, not instantly) instead of raising ``CostModelMisuse``. This is byte-
        identical for any run whose reduce-only orders are already <=5% of ADV (e.g. the crypto
        golden master on its liquid universe); only oversized flattens on a thin instrument — the
        case that used to crash the run — are changed.

        Either cap that floors below one ``lot_size`` is left UNCLAMPED so the cost model's
        5%-of-ADV ``CostModelMisuse`` tripwire fires loudly (an ADV too small to host even one lot
        is a broken-guard condition, never silently under-traded to nothing).
        """
        if order.reduce_only:
            if not self._clamp_reduce_only_adv:
                return order  # default: de-risking bypasses the cap (golden-master contract)
            # De-risk priority: clamp at the 5%-ADV impact-law edge, not the 1% build cap.
            reduce_cap = MAX_ADV_PARTICIPATION * adv_quote
            if order.qty * ref_price <= reduce_cap:
                return order
            # No epsilon: remain strictly inside the impact-law validity band.
            steps = math.floor(reduce_cap / (ref_price * inst.lot_size))
            qty_cap = steps * inst.lot_size
            if qty_cap <= 0.0 or qty_cap >= order.qty:
                return order  # sub-lot at 5%: let the tripwire halt loudly (broken-guard)
            counters["orders_reduce_adv_clamped"] += 1
            record["qty"] = qty_cap
            return replace(order, qty=qty_cap)
        cap_notional = self._max_order_adv_frac * adv_quote
        if order.qty * ref_price <= cap_notional:
            return order
        steps = math.floor(cap_notional / ref_price / inst.lot_size + 1e-9)
        qty_cap = steps * inst.lot_size
        if qty_cap <= 0.0 or qty_cap >= order.qty:
            return order  # sub-lot cap: let the 5%-ADV tripwire halt loudly
        counters["orders_adv_clamped"] += 1
        record["qty"] = qty_cap  # artifact reflects the size that actually traded
        return replace(order, qty=qty_cap)

    def _force_flat(
        self,
        iid: str,
        inst: Instrument,
        ledger: Ledger,
        close: float,
        t: Ms,
        counters: dict[str, int],
        order_records: list[dict[str, object]],
        reason_by_order: dict[str, str],
    ) -> None:
        """Administratively close a delisted instrument at its last close.

        Price = last available close (the same print used for the final mark,
        so the position value is unchanged) with the taker fee from the shared
        cost model — strictly conservative versus holding an unmarkable
        position, and loud (counted + logged with status ``forced_flat``).
        This is NOT a strategy decision, so the same-bar timing is exempt from
        the t+1 fill rule by design (documented).
        """
        pos = ledger.positions()[iid]
        side = Side.SELL if pos.qty > 0.0 else Side.BUY
        qty = abs(pos.qty)
        fee = self._cost_model.fee_frac(inst, Liquidity.TAKER) * qty * close
        client_order_id = f"bt-{t}-{iid}-forceflat"
        ledger.apply_fill(
            Fill(
                client_order_id=client_order_id,
                instrument_id=iid,
                side=side,
                qty=qty,
                price=close,
                fee_quote=fee,
                liquidity=Liquidity.TAKER,
                ts=t,
            )
        )
        counters["forced_flat"] += 1
        reason_by_order[client_order_id] = "forced_flat"
        order_records.append(
            {
                "decision_ts": t,
                "instrument_id": iid,
                "side": side.value,
                "qty": qty,
                "decision_price": close,
                "status": "forced_flat",
                "client_order_id": client_order_id,
            }
        )

    def _discretize(
        self,
        targets: Mapping[str, float],
        insts: Mapping[str, Instrument],
        bars: Mapping[str, dict[Ms, BarView]],
        ledger: Ledger,
        equity: float,
        t: Ms,
        prev_open: Ms,
        counters: dict[str, int],
        order_records: list[dict[str, object]],
        reason_by_order: dict[str, str],
        record_by_order: dict[str, dict[str, object]],
    ) -> list[OrderRequest]:
        """Turn target weights into queued orders (band, lot, min filters).

        Per instrument: ``Δnotional = w*equity - qty*close_t``; quantity rounded
        toward zero to ``lot_size`` steps. ``prev_open`` is the open of the bar
        whose close IS the decision instant ``t`` (``calendar.floor_bar(t-1, tf)``;
        ``t - tf.ms`` on the 24/7 calendar) — the bar carrying ``close_t``.

        Risk-REDUCING orders (signed ``Δnotional`` opposes the held position, so
        ``|qty|`` shrinks toward zero) bypass BOTH the no-trade band and the
        ``min_qty``/``min_notional`` skip and are emitted ``reduce_only=True``
        (execDesign.md §7.1 check 7): a flatten or de-risk must always execute,
        however small, so an absorbing flatten (all-zero targets / a halt) can
        never orphan a sub-min residual that keeps accruing PnL & funding. Such
        forced reducers are counted ``flatten_residual_forced``;
        ``skipped_below_min`` is reserved for OPENING/INCREASING orders. (A
        residual below one ``lot_size`` still cannot trade — an exchange-grid
        limit, not a min-filter defect — and stays open: documented.)

        OPENING/INCREASING orders keep the v1 band + lot + ``min_qty`` /
        ``min_notional`` discretization (full exchange-filter handling is the
        portfolio layer's :func:`~alphaforge.portfolio.discretize.weights_to_orders`).
        """
        positions = ledger.positions()
        orders: list[OrderRequest] = []
        for iid in sorted(targets):
            weight = float(targets[iid])
            if iid not in insts:
                raise ValueError(f"strategy emitted target for unknown instrument {iid!r}")
            if not math.isfinite(weight):
                raise ValueError(f"strategy emitted non-finite weight for {iid!r}: {weight!r}")
            counters["orders_decided"] += 1
            inst = insts[iid]
            bar = bars[iid].get(prev_open)

            def _skip(status: str, *, price: float, side: str, qty: float, _iid: str = iid) -> None:
                order_records.append(
                    {
                        "decision_ts": t,
                        "instrument_id": _iid,
                        "side": side,
                        "qty": qty,
                        "decision_price": price,
                        "status": status,
                        "client_order_id": "",
                    }
                )

            if bar is None:
                counters["dropped_no_decision_bar"] += 1
                _skip("dropped_no_decision_bar", price=math.nan, side="", qty=0.0)
                continue
            price = bar.close
            pos = positions.get(iid)
            held_qty = 0.0 if pos is None else pos.qty
            delta_notional = weight * equity - held_qty * price
            side = Side.BUY if delta_notional > 0.0 else Side.SELL
            # A REDUCING order moves |held_qty| toward zero: its signed delta
            # opposes the held position. (A target that crosses zero would both
            # reduce and re-open; the engine emits one netting order, which the
            # ledger flips internally — the reduce leg is what de-risks, so the
            # whole order is treated as reducing for the exemption.)
            reducing = held_qty != 0.0 and (delta_notional > 0.0) != (held_qty > 0.0)
            if not reducing and abs(delta_notional) < self._no_trade_band_frac * equity:
                counters["skipped_no_trade_band"] += 1
                _skip(
                    "skipped_no_trade_band",
                    price=price,
                    side=side.value,
                    qty=abs(delta_notional) / price,
                )
                continue
            steps = math.floor(abs(delta_notional) / price / inst.lot_size + 1e-9)
            qty = steps * inst.lot_size
            if qty <= 0.0:
                # Sub-lot residual: nothing trades on the exchange grid (a lot-
                # size limit, not a min-filter defect) — reducers included.
                counters["skipped_below_min"] += 1
                _skip("skipped_below_min", price=price, side=side.value, qty=qty)
                continue
            if self._borrow_data is not None and side is Side.SELL:
                locate_required = incremental_short_qty(
                    current_signed_qty=held_qty, sell_qty=qty
                )
                if locate_required > 0.0:
                    quote = self._borrow_data.quote(iid, as_of=t)
                    granted = 0.0
                    if quote is None:
                        counters["borrow_quotes_missing"] += 1
                    elif inst.can_short:
                        locate = evaluate_locate(
                            quote, requested_qty=locate_required, decision_ts=t
                        )
                        granted = locate.granted_qty
                    allowed_qty = qty - locate_required + granted
                    allowed_steps = math.floor(allowed_qty / inst.lot_size + 1e-9)
                    allowed_qty = allowed_steps * inst.lot_size
                    if allowed_qty < qty:
                        if allowed_qty <= 0.0:
                            counters["borrow_locates_denied"] += 1
                            _skip(
                                "dropped_borrow_unavailable",
                                price=price,
                                side=side.value,
                                qty=qty,
                            )
                            continue
                        counters["borrow_locates_partial"] += 1
                        qty = allowed_qty
            below_min = qty < inst.min_qty or qty * price < inst.min_notional
            if below_min and not reducing:
                counters["skipped_below_min"] += 1
                _skip("skipped_below_min", price=price, side=side.value, qty=qty)
                continue
            if below_min:  # reducing order forced through the min filters
                counters["flatten_residual_forced"] += 1
            client_order_id = f"bt-{t}-{iid}"
            order = OrderRequest(
                client_order_id=client_order_id,
                instrument_id=iid,
                side=side,
                qty=qty,
                order_type=OrderType.MARKET,
                reduce_only=reducing,
                decision_ts=t,
                decision_price=price,
                reason="target_weight",
            )
            reason_by_order[client_order_id] = "target_weight"
            record: dict[str, object] = {
                "decision_ts": t,
                "instrument_id": iid,
                "side": side.value,
                "qty": qty,
                "decision_price": price,
                "status": "queued",
                "client_order_id": client_order_id,
            }
            order_records.append(record)
            record_by_order[client_order_id] = record
            orders.append(order)
        return orders

    # -------------------------------------------------------------- plumbing

    def _load_bars(self, ids: Sequence[str], *, start: Ms, end: Ms) -> dict[str, dict[Ms, BarView]]:
        """One bulk PIT read -> per-instrument ``ts_open -> BarView`` maps.

        ``as_of = end`` is PIT-safe here: bar contents are immutable (only
        quality-flag *visibility* evolves with as_of, and the engine does not
        consume flags); the per-decision truncation is enforced positionally
        by the event loop and, for strategies, by ``StrategyContext.bars``.
        """
        tbl = self._reader.ohlcv(ids, start=start, end=end, as_of=end, tf=self._tf)
        out: dict[str, dict[Ms, BarView]] = {iid: {} for iid in ids}
        if tbl.num_rows == 0:
            return out
        iid_col = tbl.column("instrument_id").to_pylist()
        ts_col = tbl.column("ts_open").cast(pa.int64()).to_pylist()
        cols = {
            name: tbl.column(name).to_pylist()
            for name in ("open", "high", "low", "close", "volume", "quote_volume")
        }
        for i, iid in enumerate(iid_col):
            qv = cols["quote_volume"][i]
            out[str(iid)][int(ts_col[i])] = BarView(
                ts_open=int(ts_col[i]),
                open=float(cols["open"][i]),
                high=float(cols["high"][i]),
                low=float(cols["low"][i]),
                close=float(cols["close"][i]),
                volume=float(cols["volume"][i]),
                quote_volume=math.nan if qv is None else float(qv),
            )
        return out

    def _load_funding(
        self, insts: Mapping[str, Instrument], *, start: Ms, end: Ms
    ) -> dict[str, list[tuple[Ms, float]]]:
        """Stored funding events per perp instrument, ``start < ts_funding <= end``.

        Read once up front (funding history is immutable; ``as_of = end``
        bounds visibility — an event published within its lag of the run end
        is conservatively excluded). The events ARE the schedule: applying
        them one by one is automatically 8h/4h/1h interval-aware
        (leakageCritique.md finding 6).
        """
        perp_ids = sorted(iid for iid, inst in insts.items() if inst.market_type is MarketType.PERP)
        out: dict[str, list[tuple[Ms, float]]] = {iid: [] for iid in perp_ids}
        if not perp_ids:
            return out
        tbl = self._reader.funding(perp_ids, start=start + 1, end=end + 1, as_of=end)
        iid_col = tbl.column("instrument_id").to_pylist()
        ts_col = tbl.column("ts_funding").cast(pa.int64()).to_pylist()
        rate_col = tbl.column("rate").to_pylist()
        for i, iid in enumerate(iid_col):
            out[str(iid)].append((int(ts_col[i]), float(rate_col[i])))
        return out  # reader returns sorted (instrument_id, ts_funding)

    def _load_corporate_actions(
        self, ids: Sequence[str], *, start: Ms, end: Ms
    ) -> dict[str, list[CorporateAction]]:
        """Load validated PIT split/dividend events for ``[start, end)``.

        The read is bounded by ``as_of=end`` but every returned event must also have
        ``available_at <= ex_date``. A late record cannot be replayed at its economic
        boundary without future data, so the run fails closed rather than retroactively
        rewriting holdings. Crypto paths never call this method.
        """
        tbl = self._reader.corporate_actions(list(ids), start=start, end=end, as_of=end)
        out: dict[str, list[CorporateAction]] = {}
        if tbl.num_rows == 0:
            return out
        iid_col = tbl.column("instrument_id").to_pylist()
        type_col = tbl.column("action_type").to_pylist()
        ex_col = tbl.column("ex_date").cast(pa.int64()).to_pylist()
        available_col = tbl.column("available_at").cast(pa.int64()).to_pylist()
        ratio_col = tbl.column("ratio").to_pylist()
        cash_col = tbl.column("cash_amount").to_pylist()
        for i, iid in enumerate(iid_col):
            action = CorporateAction(
                instrument_id=str(iid),
                action_type=CorporateActionType(str(type_col[i])),
                ex_date=int(ex_col[i]),
                available_at=int(available_col[i]),
                ratio=float(ratio_col[i]),
                # The lake encodes "this split carries no cash" as NaN, not NULL, so
                # `cash_col[i] is None` is False and a NaN reached CorporateAction, which
                # correctly refuses it ("split cash_amount must be null"). Every one of the
                # 4,804 splits in lake_sharadar and 4,512 of 5,156 in lake carry NaN, so this
                # aborted any run whose window contained a split -- it killed the AlphaLedger
                # pinned re-run 26 minutes in. NULL and NaN are the same intent here and the
                # boundary is where they must be reconciled. Normalising non-finite to None is
                # safe in BOTH directions: a split becomes valid, while a CASH_DIVIDEND whose
                # amount is genuinely missing still fails loudly on "requires a finite
                # cash_amount" rather than being silently swallowed. Both asserted in
                # tests/unit/test_corporate_action_nan_cash.py.
                cash_amount=(
                    None
                    if cash_col[i] is None or not math.isfinite(float(cash_col[i]))
                    else float(cash_col[i])
                ),
            )
            action.require_known_before_boundary()
            out.setdefault(action.instrument_id, []).append(action)
        for events in out.values():
            events.sort(key=lambda event: (event.ex_date, event.action_type.value))
        return out

    def _apply_due_corporate_actions(
        self,
        actions: Mapping[str, list[CorporateAction]],
        action_ptr: dict[str, int],
        bars: Mapping[str, dict[Ms, BarView]],
        ledger: Ledger,
        last_close: dict[str, float],
        pending: list[OrderRequest],
        record_by_order: dict[str, dict[str, object]],
        counters: dict[str, int],
        prev_open: Ms,
        t: Ms,
    ) -> list[OrderRequest]:
        """Apply due ex-date groups before fills and return the converted order queue.

        Cash dividends accrue against the position held before the ex-date fill, so an
        order buying at the ex open receives no prior entitlement. Splits convert both
        open positions and pre-ex queued quantities. Mixed split/dividend events on the
        same boundary fail closed while exposed because the lake lacks deliverable-basis
        metadata needed to order them safely.
        """
        for iid, events in actions.items():
            ptr = action_ptr[iid]
            while ptr < len(events) and events[ptr].ex_date <= prev_open:
                ex_date = events[ptr].ex_date
                end_ptr = ptr
                while end_ptr < len(events) and events[end_ptr].ex_date == ex_date:
                    events[end_ptr].require_known_by(t)
                    end_ptr += 1
                group = events[ptr:end_ptr]
                split_events = [
                    event for event in group if event.action_type is CorporateActionType.SPLIT
                ]
                dividends = [
                    event
                    for event in group
                    if event.action_type is CorporateActionType.CASH_DIVIDEND
                ]
                pos = ledger.positions().get(iid)
                has_pending = any(order.instrument_id == iid for order in pending)
                if split_events and dividends and (pos is not None or has_pending):
                    raise ValueError(
                        f"mixed split/dividend boundary for exposed instrument {iid!r} "
                        f"at {ex_date} lacks deliverable-basis ordering"
                    )
                if split_events and (pos is not None or has_pending):
                    bar = bars[iid].get(prev_open)
                    if bar is None:
                        break  # defer conversion until a post-boundary bar is observable
                    ratio = math.prod(event.ratio for event in split_events)
                    pre_close = last_close.get(iid)
                    if (
                        pre_close is None
                        or not _finite_positive(pre_close)
                        or not _finite_positive(bar.open)
                    ):
                        raise ValueError(
                            f"split for {iid!r} at {ex_date} has no verifiable price boundary"
                        )
                    log_err = abs(math.log((bar.open / pre_close) * ratio))
                    verified_ratio = self._verified_split_events.get((iid, ex_date))
                    exact_event_verified = verified_ratio is not None and math.isclose(
                        ratio,
                        verified_ratio,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    if log_err > _SPLIT_SANITY_MAX_ABS_LOG and not exact_event_verified:
                        raise ValueError(
                            f"split record(s) for {iid} at ex={ex_date} failed the sanity "
                            f"guard: stored ratio {ratio:.6g} disagrees with the actual price "
                            f"move {pre_close:.6g} -> {bar.open:.6g} (|log err| {log_err:.3f} "
                            f"> {_SPLIT_SANITY_MAX_ABS_LOG:.2f}); refusing to continue with "
                            "an unconverted exposed position or order"
                        )
                    if log_err > _SPLIT_SANITY_MAX_ABS_LOG:
                        _LOG.info(
                            "issuer-verified split bypassed price-gap heuristic: "
                            "instrument=%s ex_date=%s ratio=%.12g log_error=%.6f",
                            iid,
                            ex_date,
                            ratio,
                            log_err,
                        )
                    if pos is not None:
                        ledger.apply_split(iid, t, ratio)
                    last_close[iid] = pre_close / ratio
                    converted: list[OrderRequest] = []
                    for order in pending:
                        if order.instrument_id == iid:
                            order = replace(order, qty=order.qty * ratio)
                            record = record_by_order.get(order.client_order_id)
                            if record is not None:
                                record["qty"] = order.qty
                        converted.append(order)
                    pending = converted
                    counters["corporate_actions_applied"] += len(split_events)
                if dividends and pos is not None:
                    pre_close = last_close.get(iid)
                    if pre_close is None or not _finite_positive(pre_close):
                        raise ValueError(
                            f"cash dividend for {iid!r} at {ex_date} has no verifiable "
                            "pre-ex close"
                        )
                    total_cash = sum(event.cash_amount or 0.0 for event in dividends)
                    multiple = total_cash / pre_close
                    if multiple > _DIVIDEND_SANITY_MAX_PRE_CLOSE_MULTIPLE:
                        raise ValueError(
                            f"cash dividend for {iid!r} at {ex_date} is {total_cash:.12g} "
                            f"per share, {multiple:.6g}x the pre-ex close {pre_close:.12g}; "
                            "refusing unverified dividend units/lifecycle semantics"
                        )
                for dividend in dividends:
                    if ledger.apply_cash_dividend(iid, t, dividend.cash_amount or 0.0) != 0.0:
                        counters["corporate_actions_applied"] += 1
                        counters["cash_dividends_applied"] += 1
                ptr = end_ptr
                action_ptr[iid] = ptr
        return pending

    def _build_result(
        self,
        ledger: Ledger,
        insts: Mapping[str, Instrument],
        ids: Sequence[str],
        *,
        start: Ms,
        end: Ms,
        initial_cash: float,
        counters: dict[str, int],
        order_records: list[dict[str, object]],
        position_records: list[dict[str, object]],
        reason_by_order: dict[str, str],
        market_status_coverage_hash: str | None,
    ) -> BacktestResult:
        """Assemble frames + :func:`summarize` into a :class:`BacktestResult`."""
        equity = ledger.equity_curve()
        raw_fills = ledger.fills_log()
        fills = pd.DataFrame(
            {
                "ts": raw_fills["ts"],
                "instrument_id": raw_fills["instrument_id"],
                "side": raw_fills["side"],
                "qty": raw_fills["qty"],
                "price": raw_fills["price"],
                "notional": raw_fills["qty"] * raw_fills["price"],
                "fee": raw_fills["fee_quote"],
                "liquidity": raw_fills["liquidity"],
                "realized_pnl_quote": raw_fills["realized_pnl_quote"],
                "client_order_id": raw_fills["client_order_id"],
                "reason": raw_fills["client_order_id"].map(
                    lambda coid: reason_by_order.get(str(coid), "")
                ),
            },
            columns=list(FILLS_COLUMNS),
        )
        funding_frame = ledger.funding_log()
        corporate_action_frame = ledger.corporate_actions_log()
        financing_frame = ledger.financing_log()
        counters["funding_events_applied"] = len(funding_frame)
        orders = pd.DataFrame(order_records, columns=list(ORDERS_COLUMNS))
        positions = pd.DataFrame(position_records, columns=list(POSITIONS_COLUMNS))
        funding_series = pd.Series(
            funding_frame["payment_quote"].to_numpy(dtype="float64"),
            index=pd.Index(funding_frame["ts_funding"].to_numpy(dtype="int64"), name="ts_funding"),
            name="payment_quote",
        )
        summary = summarize(equity, fills=fills, funding=funding_series, positions=positions)
        config: dict[str, object] = {
            "start": start,
            "end": end,
            "tf": self._tf.value,
            "asset_class": self._asset_class.value,
            "instrument_ids": list(ids),
            "initial_cash": initial_cash,
            "borrow_total": ledger.total_borrow,
            "corporate_action_cashflow_total": ledger.total_corporate_actions,
            "financing_cashflow_total": ledger.total_financing,
            "no_trade_band_frac": self._no_trade_band_frac,
            "max_order_adv_frac": self._max_order_adv_frac,
            "fill_model": type(self._fill_model).__name__,
            "perp_instruments": sorted(
                iid for iid, inst in insts.items() if inst.market_type is MarketType.PERP
            ),
            **self._config_echo,
        }
        if self._market_status is not None:
            config["market_status_coverage"] = "run_interval_preflight"
            if market_status_coverage_hash is not None:
                config["market_status_coverage_hash"] = market_status_coverage_hash
        return BacktestResult(
            equity=equity,
            fills=fills,
            funding_events=funding_frame,
            orders=orders,
            positions=positions,
            summary=summary,
            corporate_actions=corporate_action_frame,
            financing_events=financing_frame,
            config=config,
            counters=counters,
        )
