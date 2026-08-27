"""FeatureContext — the PIT-windowed data surface handed to feature functions.

The engine constructs a context whose window ``[start, end)`` (on ``ts_open``) ends at
the decision availability cutoff ``end``: every table the context serves was readable
by a live trader at ``end``, because all reads go through
:class:`~alphaforge.data.store.reader.PITDataReader` with ``as_of = end``. Feature
functions therefore *cannot* request future data — the context simply does not
contain it (dataDesign.md §7.2).

Per-timestamp PIT inside the window (the batch-history subtlety):

* **Prices/volume** are uniformly available: a bar labeled ``ts_open`` is available at
  ``ts_open + Δ`` and forever after, so a single window read at ``as_of = end`` is
  PIT-correct for *every* evaluation timestamp inside the window.
* **Funding** rows are pre-filtered to ``available_at <= end``; per-evaluation-ts
  correctness inside the window is the feature's responsibility and is solved by the
  one sanctioned helper :meth:`FeatureContext.funding_asof_join`, which joins on the
  stored ``available_at`` — NEVER on ``ts_funding`` (leakage findings 6/18).
* **Quality flags** are the exception: a bar's flag bits differ by *evaluation* time
  (bit lags in :data:`FLAG_AVAILABILITY_LAG_BARS`; leakage finding 5). The flags
  column served here is masked at ``as_of = end`` only — i.e. it is the flag state at
  the window end, not at each row's own decision time. A feature that consumes flags
  MUST lag each bit per-row itself using :data:`FLAG_AVAILABILITY_LAG_BARS` (bit ``b``
  on the bar at ``ts_open`` is decision-visible from ``ts_open + (1 + lag_b)·Δ``).
  v1 library features do not consume flags, so the only PIT surfaces on the deployed
  path are prices/volume/funding — which are handled above.

Mutation safety: served frames are copy-on-write shallow copies of internal caches
(pandas >= 3 semantics) — a feature that writes into them mutates only its own view.
Feature code must still NEVER mutate inputs (pure-function contract).

The sanctioned layout for window math is :meth:`FeatureContext.panel`: a wide frame on
the COMPLETE expected bar grid, where row position == time slot. Positional ops
(``shift``/``rolling``/``ewm``) over stored-rows-only layouts silently change meaning
across data gaps and break batch/as-of parity; over the full grid they are exact time
operations, which is what makes ``lookback_bars`` a *time* guarantee.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

from alphaforge.config.sleeve import sleeve_for
from alphaforge.core.calendar import TradingCalendar, calendar_for
from alphaforge.core.errors import LookaheadError
from alphaforge.core.time import Ms, Timeframe
from alphaforge.core.types import AssetClass
from alphaforge.data.schemas import FLAG_AVAILABILITY_LAG_BARS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from alphaforge.core.instruments import Instrument, InstrumentStore
    from alphaforge.data.store.reader import PITDataReader
    from alphaforge.data.universe.store import UniverseStore

__all__ = [
    "FLAG_AVAILABILITY_LAG_BARS",
    "FeatureContext",
    "long_series",
]

#: Bar columns served by :meth:`FeatureContext.bars` (``ingested_at`` is audit-only and
#: ``n_trades`` is not a v1 feature input — withholding them here enforces that).
_BAR_COLUMNS: tuple[str, ...] = (
    "instrument_id",
    "ts_open",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "quality_flags",
)

_FUNDING_COLUMNS: tuple[str, ...] = ("instrument_id", "ts_funding", "rate", "available_at")

#: Corporate-action columns served by :meth:`FeatureContext.corporate_actions` — the
#: projection the equity adjusted-close kernel consumes (the ``_CA_COLUMNS`` of
#: :mod:`alphaforge.features.library.equity_price`). ``ex_date``/``available_at`` are
#: cast to int64 ms (both are in :data:`_TS_COLUMNS`); ALL adjustment joins gate on
#: ``available_at``, never ``ex_date`` (leakage finding 18).
_CA_COLUMNS: tuple[str, ...] = (
    "instrument_id",
    "ex_date",
    "available_at",
    "action_type",
    "ratio",
    "cash_amount",
)

#: Fundamentals columns served to the equity value/quality factors. ``period_end`` and
#: ``available_at`` are timestamps cast to int64 ms (both in :data:`_TS_COLUMNS`); ALL
#: factor joins gate on ``available_at`` (the SEC filing date), NEVER ``period_end`` —
#: a quarter is unknowable until filed (the fundamentals analogue of finding 18).
_FUND_COLUMNS: tuple[str, ...] = (
    "instrument_id",
    "period_end",
    "available_at",
    "fiscal_period",
    "fiscal_year",
    "revenues",
    "cost_of_revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "op_cash_flow",
    "invest_cash_flow",
    "capex",
    "free_cash_flow",
    "net_common_issued",
    "shares_basic",
    "share_factor",
    "assets_avg",
    "equity",
    "assets",
    "diluted_shares",
)

_TS_COLUMNS: frozenset[str] = frozenset(
    {"ts_open", "ts_funding", "available_at", "ex_date", "period_end"}
)


def long_series(wide: pd.DataFrame, *, name: str | None = None) -> pd.Series:
    """Flatten a wide (ts_open x instrument_id) frame to the canonical long Series.

    The result is float64, indexed by the 2-level MultiIndex
    ``(ts_open, instrument_id)`` = the Cartesian product of ``wide.index`` and
    ``wide.columns`` in row-major order — exactly the index contract of
    :data:`~alphaforge.features.spec.FeatureFn`. This is the one sanctioned
    wide-to-long conversion; it is deterministic and preserves NaN.
    """
    index = pd.MultiIndex.from_product(
        [wide.index, wide.columns], names=("ts_open", "instrument_id")
    )
    values = wide.to_numpy(dtype="float64").ravel()
    return pd.Series(values, index=index, dtype="float64", name=name)


class FeatureContext:
    """PIT-windowed data access for feature functions (see module docstring).

    Constructed by :class:`~alphaforge.features.engine.FeatureEngine`; feature code
    receives it ready-made and treats it as read-only. ``end`` is the decision
    availability cutoff: all reads use ``as_of = end``.
    """

    def __init__(
        self,
        *,
        reader: PITDataReader,
        instruments: InstrumentStore,
        universe: UniverseStore,
        instrument_ids: Sequence[str],
        start: Ms,
        end: Ms,
        calendar: TradingCalendar | None = None,
        asset_class: AssetClass = AssetClass.CRYPTO_PERP,
        anchor_tf: Timeframe | None = None,
    ) -> None:
        if end <= start:
            raise ValueError(f"FeatureContext window requires end > start, got [{start}, {end})")
        ids = tuple(dict.fromkeys(instrument_ids))
        if not ids:
            raise ValueError("FeatureContext requires at least one instrument_id")
        self._reader = reader
        self._instruments = instruments
        self._universe = universe
        self._instrument_ids = ids
        self._start = start
        self._end = end
        # The grid source for :meth:`panel` and the asset class for the corp-actions
        # read. ``asset_class`` defaults CRYPTO_PERP ⇒ the 24/7 calendar, whose
        # ``expected_bar_opens`` delegates to the same ``core.time`` kernel the panel
        # used before the calendar was threaded (byte-identical). The calendar may be
        # passed explicitly (the engine threads its sleeve calendar) or resolved from
        # ``asset_class`` here — both must agree; the explicit form avoids a re-resolve.
        self._asset_class = asset_class
        self._calendar = calendar if calendar is not None else calendar_for(asset_class)
        # The sleeve's anchor TF is the DEFAULT timeframe for bars()/panel(): factor bodies
        # call ctx.panel("close") with no tf and must get H1 on crypto, D1 on equities (an
        # XNYS calendar rejects H1). Defaults from the sleeve of ``asset_class`` so the
        # crypto path stays H1 (byte-identical); the engine threads its own anchor_tf.
        self._anchor_tf = anchor_tf if anchor_tf is not None else sleeve_for(asset_class).anchor_tf
        self._bars_cache: dict[Timeframe, pd.DataFrame] = {}
        self._panel_cache: dict[tuple[Timeframe, str], pd.DataFrame] = {}
        self._funding_cache: pd.DataFrame | None = None
        self._ca_cache: pd.DataFrame | None = None
        self._fund_cache: pd.DataFrame | None = None

    # ------------------------------------------------------------- identity

    @property
    def instrument_ids(self) -> Sequence[str]:
        """Requested instruments, de-duplicated, in request order."""
        return self._instrument_ids

    @property
    def start(self) -> Ms:
        """Window start on ``ts_open`` (inclusive, epoch ms UTC) — includes the
        engine's lookback headroom; rows before the engine's requested output range
        exist purely as warm-up history."""
        return self._start

    @property
    def end(self) -> Ms:
        """Window end on ``ts_open`` (exclusive, epoch ms UTC) == the decision
        availability cutoff used as ``as_of`` for every read."""
        return self._end

    # ------------------------------------------------------------- raw tables

    def bars(self, tf: Timeframe | None = None) -> pd.DataFrame:
        """OHLCV bars of timeframe ``tf`` (default: the sleeve anchor TF) in the window.

        Columns: ``ts_open`` (int64 epoch ms), ``instrument_id``, ``open``, ``high``,
        ``low``, ``close``, ``volume``, ``quote_volume``, ``quality_flags``; sorted by
        ``(instrument_id, ts_open)``. Only bars fully available at the window end are
        present (``ts_open + tf.ms <= end``).

        NaN policy (v1, declared): rows whose masked flags include
        ``BAD_PRINT_SUSPECT`` (or any other bit) keep their stored prices AS-IS; the
        flags column is exposed and each feature decides its own policy. Nothing is
        NaN-ed or forward-filled here (leakage finding 5 / critique 28: silent
        repair in research erases adverse prints that live trading eats).

        Flag caveat: ``quality_flags`` is masked at ``as_of = end`` — features that
        consume flags must additionally lag bits per evaluation row via
        :data:`FLAG_AVAILABILITY_LAG_BARS` (module docstring). Converted from Arrow
        once and cached; the returned frame is a copy-on-write shallow copy.
        """
        tf = tf if tf is not None else self._anchor_tf
        cached = self._bars_cache.get(tf)
        if cached is None:
            tbl = self._reader.ohlcv(
                list(self._instrument_ids),
                start=self._start,
                end=self._end,
                as_of=self._end,
                tf=tf,
            )
            cached = self._to_pandas(tbl, _BAR_COLUMNS)
            self._bars_cache[tf] = cached
        return cached.copy(deep=False)

    def funding(self) -> pd.DataFrame:
        """Funding settlements in the window, pre-filtered to ``available_at <= end``.

        Columns: ``ts_funding`` (int64 epoch ms), ``instrument_id``, ``rate``,
        ``available_at`` (int64 epoch ms); sorted by ``(instrument_id, ts_funding)``.
        Per-evaluation-timestamp correctness inside the window is the consumer's
        responsibility — use :meth:`funding_asof_join`, never a join on
        ``ts_funding`` (leakage finding 18).
        """
        if self._funding_cache is None:
            tbl = self._reader.funding(
                list(self._instrument_ids),
                start=self._start,
                end=self._end,
                as_of=self._end,
            )
            self._funding_cache = self._to_pandas(tbl, _FUNDING_COLUMNS)
        return self._funding_cache.copy(deep=False)

    def corporate_actions(self) -> pd.DataFrame:
        """Corporate-action rows in the window, pre-filtered to ``available_at <= end``.

        The equity analogue of :meth:`funding` — splits/dividends instead of perp
        settlements. Columns: ``instrument_id``, ``ex_date`` (int64 epoch ms),
        ``available_at`` (int64 epoch ms), ``action_type`` (``'split'``/``'dividend'``),
        ``ratio`` (split factor, 1.0 for a dividend), ``cash_amount`` (per-share
        dividend cash, null for splits); sorted by ``(instrument_id, ex_date)``.

        Masked on the stored ``available_at`` (never ``ex_date``; leakage finding 18) —
        the rows served are those knowable by the window end. Per-decision-row PIT
        inside the window is the consumer's responsibility and is solved by the
        adjusted-close kernel
        (:func:`~alphaforge.features.library.equity_price.adjusted_close`), which folds
        each action only into rows whose own decision could know it
        (``available_at <= ts_open + Δ``). Served to the equity price factors; the
        crypto path never calls it (no corporate actions exist for perps, so the read
        is empty even if it did). Converted once and cached; returns a copy-on-write
        shallow copy.
        """
        if self._ca_cache is None:
            tbl = self._reader.corporate_actions(
                list(self._instrument_ids),
                start=self._start,
                end=self._end,
                as_of=self._end,
            )
            self._ca_cache = self._to_pandas(tbl, _CA_COLUMNS)
        return self._ca_cache.copy(deep=False)

    def fundamentals(self) -> pd.DataFrame:
        """Quarterly fundamentals rows in the window, pre-filtered to ``available_at <= end``.

        The fundamentals analogue of :meth:`funding`/:meth:`corporate_actions`: one row per
        ``(instrument, fiscal quarter)`` carrying the income-statement + balance-sheet line
        items (columns: :data:`_FUND_COLUMNS`, with ``period_end`` / ``available_at`` as
        int64 epoch ms). Masked on the stored ``available_at`` (the SEC filing date) — NEVER
        ``period_end`` — so the served rows are the quarters knowable by the window end.
        Per-decision-row PIT inside the window is the consumer's responsibility and is solved
        by :meth:`fundamentals_asof_join` (each quarter folds only into bars whose own
        decision could know it, ``available_at <= ts_open + Δ``). The crypto path reads empty
        (perps have no fundamentals partitions), so it never perturbs a crypto factor —
        byte-identical. Converted once and cached; returns a copy-on-write shallow copy.
        """
        if self._fund_cache is None:
            tbl = self._reader.fundamentals(
                list(self._instrument_ids),
                start=self._start,
                end=self._end,
                as_of=self._end,
            )
            self._fund_cache = self._to_pandas(tbl, _FUND_COLUMNS)
        return self._fund_cache.copy(deep=False)

    def fundamentals_asof_join(
        self,
        bars_index: pd.MultiIndex,
        *,
        column: str,
        frame: pd.DataFrame | None = None,
        tf: Timeframe | None = None,
    ) -> pd.Series:
        """Last-known fundamentals ``column`` per ``(ts_open, instrument_id)``, PIT per row.

        For each index entry the decision time is the bar close ``ts_open + tf.ms``; the
        joined value is the latest fundamentals row of that instrument with
        ``available_at <= ts_open + tf.ms`` (backward as-of merge on the stored
        ``available_at`` — a quarter filed at ``available_at`` is invisible to decisions
        before it). This forward-fills the latest FILED quarter across every daily session
        until the next filing. NaN where no quarter is known yet.

        ``frame`` defaults to :meth:`fundamentals`; a caller may pass a frame with DERIVED
        columns (e.g. per-instrument TTM rolling sums computed on the sparse quarterly grid
        BEFORE this join) as long as it carries ``instrument_id``, ``available_at``, and
        ``column``. Returns a float64 Series named ``column`` aligned exactly to
        ``bars_index``. Lookback discipline matches :meth:`funding_asof_join` — a consuming
        spec must declare ``lookback_bars`` covering its filing-recency + TTM horizon or the
        truncation harness fails it.
        """
        tf = tf if tf is not None else self._anchor_tf
        if bars_index.nlevels != 2:
            raise ValueError(
                f"bars_index must be a 2-level (ts_open, instrument_id) MultiIndex, "
                f"got {bars_index.nlevels} level(s)"
            )
        fund = self.fundamentals() if frame is None else frame
        if fund.empty or column not in fund.columns:
            return pd.Series(float("nan"), index=bars_index, dtype="float64", name=column)
        left = pd.DataFrame(
            {
                "ts_open": bars_index.get_level_values(0).to_numpy(dtype="int64"),
                "instrument_id": bars_index.get_level_values(1),
            }
        )
        left["decision_ts"] = left["ts_open"] + tf.ms
        right = fund[["instrument_id", "available_at", column]].copy()
        right["available_at"] = right["available_at"].to_numpy(dtype="int64")
        merged = pd.merge_asof(
            left.sort_values("decision_ts", kind="stable"),
            right.sort_values("available_at", kind="stable"),
            left_on="decision_ts",
            right_on="available_at",
            by="instrument_id",
            direction="backward",
            allow_exact_matches=True,
        )
        out = pd.Series(
            merged[column].to_numpy(dtype="float64"),
            index=pd.MultiIndex.from_arrays(
                [merged["ts_open"], merged["instrument_id"]],
                names=("ts_open", "instrument_id"),
            ),
            dtype="float64",
            name=column,
        )
        return out.reindex(bars_index).rename(column)

    # ------------------------------------------------------------- derived views

    def panel(self, column: str = "close", tf: Timeframe | None = None) -> pd.DataFrame:
        """Wide float64 frame of ``column`` on the COMPLETE expected bar grid.

        ``tf`` defaults to the sleeve anchor TF (H1 crypto / D1 equity).

        Index: every aligned ``ts_open`` (int64 epoch ms) in ``[start, end)`` per the
        sleeve calendar's
        :meth:`~alphaforge.core.calendar.TradingCalendar.expected_bar_opens` —
        including grid slots with no stored bar (NaN rows). Columns: all requested
        ``instrument_ids`` (NaN columns for instruments without data). Because the grid
        is complete, row position == time slot, so ``shift``/``rolling``/``ewm`` are
        exact *time* operations and batch/as-of parity holds across data gaps. This is
        THE sanctioned layout for feature window math. Cached per ``(tf, column)``;
        returns a copy-on-write shallow copy.

        The grid is the calendar grid: for the default crypto sleeve the 24/7 calendar
        delegates to the same ``core.time`` kernel the panel used before the calendar
        was threaded (byte-identical); for the equity D1 sleeve the XNYS calendar emits
        only session opens, so weekend/holiday slots are correctly absent (not NaN-padded).
        """
        tf = tf if tf is not None else self._anchor_tf
        key = (tf, column)
        cached = self._panel_cache.get(key)
        if cached is None:
            bars = self._bars_cache.get(tf)
            if bars is None:
                self.bars(tf)
                bars = self._bars_cache[tf]
            if column not in bars.columns or column in ("ts_open", "instrument_id"):
                raise ValueError(
                    f"panel column must be one of "
                    f"{sorted(set(_BAR_COLUMNS) - {'ts_open', 'instrument_id'})}, "
                    f"got {column!r}"
                )
            grid = self._calendar.expected_bar_opens(self._start, self._end, tf)
            wide = bars.pivot(index="ts_open", columns="instrument_id", values=column)
            wide = wide.reindex(index=grid, columns=list(self._instrument_ids))
            wide = wide.astype("float64")
            wide.index.name = "ts_open"
            wide.columns.name = "instrument_id"
            cached = wide
            self._panel_cache[key] = cached
        return cached.copy(deep=False)

    def funding_asof_join(
        self, bars_index: pd.MultiIndex, *, tf: Timeframe | None = None
    ) -> pd.Series:
        """Last-known funding rate per ``(ts_open, instrument_id)``, PIT-correct per row.

        For each index entry the decision time is the bar close ``ts_open + tf.ms``;
        the joined value is the latest funding row of that instrument with
        ``available_at <= ts_open + tf.ms`` (backward as-of merge on the stored
        ``available_at`` — leakage findings 6/18: a rate published at
        ``settlement + lag`` is invisible to decisions taken before publication).
        NaN where no settlement is known yet. Returns a float64 Series named
        ``"rate"`` aligned exactly to ``bars_index``.

        Lookback discipline: "last known" reaches back through the context window
        only — a consuming spec must declare ``lookback_bars`` covering its funding
        recency horizon (>= funding_interval_hours + publication lag + 1 bar, plus
        whatever settlement history the formula itself averages), otherwise the
        truncation harness (:func:`alphaforge.features.parity.verify_truncation`)
        will fail it.
        """
        tf = tf if tf is not None else self._anchor_tf
        if bars_index.nlevels != 2:
            raise ValueError(
                f"bars_index must be a 2-level (ts_open, instrument_id) MultiIndex, "
                f"got {bars_index.nlevels} level(s)"
            )
        left = pd.DataFrame(
            {
                "ts_open": bars_index.get_level_values(0).to_numpy(dtype="int64"),
                "instrument_id": bars_index.get_level_values(1),
            }
        )
        left["decision_ts"] = left["ts_open"] + tf.ms
        funding = self.funding()
        if funding.empty:
            return pd.Series(float("nan"), index=bars_index, dtype="float64", name="rate")
        merged = pd.merge_asof(
            left.sort_values("decision_ts", kind="stable"),
            funding[["instrument_id", "rate", "available_at"]].sort_values(
                "available_at", kind="stable"
            ),
            left_on="decision_ts",
            right_on="available_at",
            by="instrument_id",
            direction="backward",
            allow_exact_matches=True,
        )
        out = pd.Series(
            merged["rate"].to_numpy(dtype="float64"),
            index=pd.MultiIndex.from_arrays(
                [merged["ts_open"], merged["instrument_id"]],
                names=("ts_open", "instrument_id"),
            ),
            dtype="float64",
            name="rate",
        )
        return out.reindex(bars_index).rename("rate")

    # ------------------------------------------------------------- reference data

    def instrument(self, instrument_id: str) -> Instrument:
        """SCD2 instrument version as-of the window end (decision cutoff).

        This is where interval-aware funding metadata lives
        (``funding_interval_hours``; leakage finding 6) — carry annualization must
        read it here, never hard-code 8h.

        Earliest-version fallback: the SCD2 store records when *we* learned facts
        (``valid_from`` = seed/refresh time), not when they became true on the
        venue. A historical replay window therefore predates every stored version
        for instruments seeded after the fact. Falling back to the EARLIEST known
        version is a documented, bounded anachronism — venue metadata (tick size,
        fees, funding interval) drifts slowly and was genuinely public at the
        time; refusing to replay history would be strictly worse. Raises
        ``KeyError`` only if the instrument is entirely unknown to the store.
        """
        inst = self._instruments.get(instrument_id, as_of=self._end)
        if inst is None:
            versions = self._instruments.history(instrument_id)
            if versions:
                return versions[0][2]
            raise KeyError(
                f"instrument {instrument_id!r} is unknown to the SCD2 store "
                f"(no version valid at as_of={self._end} and no history to fall back to)"
            )
        return inst

    def universe_asof(self, ts: Ms) -> frozenset[str]:
        """Point-in-time universe membership at ``ts`` (for market/breadth features).

        Membership intervals are knowable from their ``effective_from``, so this is
        PIT by construction. ``ts`` must not exceed the window end — asking about
        the future from inside a PIT context raises
        :class:`~alphaforge.core.errors.LookaheadError`.
        """
        if ts > self._end:
            raise LookaheadError(
                f"universe_asof({ts}) exceeds the context decision cutoff {self._end}"
            )
        return self._universe.membership_asof(ts)

    # ------------------------------------------------------------- internals

    @staticmethod
    def _to_pandas(tbl: pa.Table, columns: tuple[str, ...]) -> pd.DataFrame:
        """Arrow → pandas, once: select ``columns``, cast timestamp cols to int64 ms."""
        tbl = tbl.select(list(columns))
        for name in columns:
            if name in _TS_COLUMNS:
                idx = tbl.schema.get_field_index(name)
                tbl = tbl.set_column(idx, name, tbl.column(name).cast(pa.int64()))
        df: pd.DataFrame = tbl.to_pandas()
        return df
