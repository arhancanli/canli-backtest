"""Fill models — how queued orders become executions (execDesign.md §4.2).

THE no-lookahead contract (execDesign.md §4.1; leakageCritique.md findings 4/5):
a decision taken at the close of bar ``t`` fills at the **open of bar t+1**,
never inside bar ``t``. A fill model therefore receives *only* the next bar; it
mechanically verifies ``next_bar.ts_open >= order.decision_ts`` and raises
:class:`~alphaforge.core.errors.LookaheadError` otherwise (the decision bar's
open is strictly before the decision close, so any attempt to fill on the
decision bar — or any earlier bar — trips the guard).

Pricing is delegated entirely to the ONE :class:`TransactionCostModel`
instance (buildabilityCritique.md §3.7 — costs are never reimplemented here)::

    P_fill = open_{t+1} * (1 + s * (half_spread + impact + latency)),  s = side.sign
    fee    = fee_frac(taker) * qty * P_fill                            # quote units

The fee is charged on the *executed* notional (``qty * P_fill``) and travels on
the :class:`~alphaforge.core.types.Fill` as ``fee_quote`` — the ledger debits
it as a separate cash line so commissions and price slippage stay separately
attributable in artifacts.

v1 scope notes (documented, deliberate):

* All fills are TAKER (market orders at next open — matches live execution
  timing; maker/limit fills are an execution-layer concern, Phase 8+).
* :class:`ParticipationCappedFill` is the opt-in institutional path for partial
  fills. It caps execution to a configurable share of the next bar's observed
  quote volume and fails closed when that liquidity is missing or below one lot.
  :class:`NextOpenFill` remains the compatibility default.

All money is float64 in quote units, full precision internally — rounding is a
report-rendering concern only. Timestamps are epoch-ms UTC (:data:`Ms`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol

from alphaforge.core.errors import FillUnavailableError, LookaheadError
from alphaforge.core.instruments import Instrument
from alphaforge.core.time import Ms
from alphaforge.core.types import Fill, Liquidity, OrderRequest, Side
from alphaforge.costs import TransactionCostModel

__all__ = ["BarView", "FillModel", "MakerFill", "NextOpenFill", "ParticipationCappedFill"]


@dataclass(frozen=True, slots=True, kw_only=True)
class BarView:
    """A read-only view of one OHLCV bar handed to a fill model.

    ``ts_open`` is the bar's open time (epoch-ms UTC; the bar covers
    ``[ts_open, ts_open + Δ)``). Prices are quote units; ``volume`` is base
    units; ``quote_volume`` is quote units and may be NaN for sources that
    lack it. The engine constructs these from PIT-read lake rows — a fill
    model never touches the reader and therefore can never widen its window.
    """

    ts_open: Ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float

    def __post_init__(self) -> None:
        for name in ("open", "high", "low", "close"):
            value: float = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"BarView.{name} must be finite and > 0, got {value!r}")
        if not math.isfinite(self.volume) or self.volume < 0.0:
            raise ValueError(f"BarView.volume must be finite and >= 0, got {self.volume!r}")


class FillModel(Protocol):
    """Protocol for turning a queued order plus the next bar into a fill.

    Implementations MUST raise :class:`LookaheadError` when ``next_bar`` is not
    strictly after the decision (``next_bar.ts_open < order.decision_ts``) —
    the engine relies on this as the last line of the no-lookahead contract.
    ``adv_quote`` / ``sigma_daily`` are the cost-model inputs computed from PIT
    data through the decision bar (never the fill bar).
    """

    def fill(
        self,
        order: OrderRequest,
        inst: Instrument,
        next_bar: BarView,
        *,
        adv_quote: float,
        sigma_daily: float,
    ) -> Fill:
        """Execute ``order`` against ``next_bar``; returns the resulting fill."""
        ...


class NextOpenFill:
    """Default fill model: market order at the next bar's open (execDesign.md §4.2).

    Matches live execution timing — the live loop wakes after a bar closes,
    computes, and submits market orders that land around the next open.
    Pricing and fees come exclusively from the shared
    :class:`TransactionCostModel` (the sole cost authority): the latency
    add-on lives inside ``cost_model.fill_price`` and only there, so it shifts
    the recorded fill price without being double-counted in any optimizer
    penalty (see ``TransactionCostModel.oneway_cost_frac``).
    """

    __slots__ = ("_cost_model",)

    def __init__(self, cost_model: TransactionCostModel) -> None:
        self._cost_model = cost_model

    @property
    def cost_model(self) -> TransactionCostModel:
        """The shared cost authority this model prices with (read-only)."""
        return self._cost_model

    def fill(
        self,
        order: OrderRequest,
        inst: Instrument,
        next_bar: BarView,
        *,
        adv_quote: float,
        sigma_daily: float,
    ) -> Fill:
        """Fill ``order`` at the open of ``next_bar`` with full cost treatment.

        Exact arithmetic (full float64, no rounding)::

            ref      = next_bar.open
            notional = order.qty * ref                      # impact sizing input
            P_fill   = cost_model.fill_price(inst, side, ref, notional, adv, sigma)
                     = ref * (1 + s*(half_spread + impact + latency)),  s = side.sign
            fee      = cost_model.fee_frac(inst, TAKER) * order.qty * P_fill

        ``Fill.ts = next_bar.ts_open`` — the execution prints at the bar open,
        which is exactly the decision close for a contiguous grid.

        Raises:
            LookaheadError: if ``next_bar.ts_open < order.decision_ts`` — the
                bar opened before the decision existed, i.e. a same-bar (or
                earlier) fill was attempted. ``ts_open == decision_ts`` is the
                legal contiguous case.
            CostModelMisuse: propagated from the cost model on out-of-regime
                inputs (notional > 5% ADV, non-finite adv/sigma) — never
                swallowed; the engine decides whether to drop or halt.
        """
        if next_bar.ts_open < order.decision_ts:
            raise LookaheadError(
                f"fill bar opening at {next_bar.ts_open} precedes the decision at "
                f"{order.decision_ts} for {order.instrument_id!r} "
                f"(order {order.client_order_id!r}): decisions at the close of bar t "
                "fill at the open of bar t+1, never the same bar"
            )
        if inst.instrument_id != order.instrument_id:
            raise ValueError(
                f"instrument mismatch: order is for {order.instrument_id!r} but "
                f"fill was offered {inst.instrument_id!r}"
            )
        ref = next_bar.open
        notional = order.qty * ref
        price = self._cost_model.fill_price(
            inst,
            order.side,
            ref,
            notional,
            adv_quote,
            sigma_daily,
        )
        fee = self._cost_model.fee_frac(inst, Liquidity.TAKER) * order.qty * price
        return Fill(
            client_order_id=order.client_order_id,
            instrument_id=order.instrument_id,
            side=order.side,
            qty=order.qty,
            price=price,
            fee_quote=fee,
            liquidity=Liquidity.TAKER,
            ts=next_bar.ts_open,
        )


class ParticipationCappedFill:
    """Next-open taker fill capped by observed bar liquidity.

    Executable quantity is at most ``max_bar_participation`` of the next bar's
    ``quote_volume``, converted at the open and floored to the instrument lot. Any residual is
    left unfilled; the event-driven engine records it as a partial fill with residual canceled.
    Missing/non-positive quote volume or a cap below one valid lot raises
    :class:`FillUnavailableError`, so no volume is invented.

    Price formation and fees remain delegated to :class:`NextOpenFill`, preserving the shared
    spread, square-root impact, latency and commission authority.
    """

    __slots__ = ("_max_bar_participation", "_next_open")

    def __init__(
        self,
        cost_model: TransactionCostModel,
        *,
        max_bar_participation: float = 0.10,
    ) -> None:
        if (
            not math.isfinite(max_bar_participation)
            or max_bar_participation <= 0.0
            or max_bar_participation > 1.0
        ):
            raise ValueError(
                "max_bar_participation must be finite and in (0, 1], got "
                f"{max_bar_participation!r}"
            )
        self._next_open = NextOpenFill(cost_model)
        self._max_bar_participation = max_bar_participation

    @property
    def cost_model(self) -> TransactionCostModel:
        """The shared cost authority used after quantity capping."""
        return self._next_open.cost_model

    @property
    def max_bar_participation(self) -> float:
        """Maximum observed next-bar quote-volume share consumed by one order."""
        return self._max_bar_participation

    def fill(
        self,
        order: OrderRequest,
        inst: Instrument,
        next_bar: BarView,
        *,
        adv_quote: float,
        sigma_daily: float,
    ) -> Fill:
        """Execute the lot-floored participation-capped quantity at the next open."""
        quote_volume = next_bar.quote_volume
        if not math.isfinite(quote_volume) or quote_volume <= 0.0:
            raise FillUnavailableError(
                f"missing/non-positive quote volume for {order.instrument_id!r} at "
                f"{next_bar.ts_open}: {quote_volume!r}"
            )
        cap_quote = self._max_bar_participation * quote_volume
        cap_steps = math.floor(cap_quote / next_bar.open / inst.lot_size + 1e-12)
        cap_qty = cap_steps * inst.lot_size
        executable_qty = min(order.qty, cap_qty)
        if executable_qty < inst.min_qty or executable_qty * next_bar.open < inst.min_notional:
            raise FillUnavailableError(
                f"bar-liquidity cap for {order.instrument_id!r} at {next_bar.ts_open} "
                f"permits {executable_qty!r}, below min_qty/min_notional"
            )
        return self._next_open.fill(
            replace(order, qty=executable_qty),
            inst,
            next_bar,
            adv_quote=adv_quote,
            sigma_daily=sigma_daily,
        )


class MakerFill:
    """Post-only / maker-first fill model with CONSERVATIVE non-fill modeling.

    RESEARCH / OPT-IN ONLY. The deployed path and the golden master keep using
    :class:`NextOpenFill` (pure taker); this model is injected explicitly (e.g. via
    ``WalkForwardRunner(..., fill_model_factory=...)``) to *measure* the maker-execution
    uplift estimated by the EV screen ``scripts/exp3_maker_exec_screen.py``. It is never the
    engine default, so its existence cannot perturb a single existing artifact.

    A post-only limit is posted at the next bar's **open**. Whether it fills passively is read
    from that bar's own geometry — the order genuinely rests *during* bar ``t+1``, so using
    that bar's close is execution realism, **not signal lookahead** (the signal was frozen at
    the decision close ``t``; the no-lookahead guard below still rejects a fill bar that
    precedes the decision). The fill rule is deliberately the CONSERVATIVE close-direction
    proxy, NOT ``low < open`` (which is true on almost every hourly bar and would imply a ~95%
    fill rate — optimistic, fake-alpha-manufacturing). Instead a passive fill is counted only
    when the bar closed in the direction that brings liquidity to a resting order::

        BUY  fills passively iff next_bar.close < open  (the bar sold off  -> sellers reach the bid)
        SELL fills passively iff next_bar.close > open  (the bar rallied   -> buyers  reach the ask)

    This yields a ~50% fill rate — a deliberate LOWER BOUND (a patient order over a full hour
    likely fills more often); the genuine rate must come from **live post-only logs**. The two
    branches:

    * **Passive fill** — execute at the open, **MAKER** fee, **no half-spread crossed** (you
      provided liquidity). Both maker and taker enter at the same open, so the differential
      vs taker on this branch is exactly the fee + spread saved (~5.5 bps).
    * **Non-fill → chase** — the bar moved *with* the order (an up-bar for a buy), so the
      post-only missed and is chased as a **TAKER** at the full cost-model price (half-spread
      + impact + latency) **plus** an explicit ``adverse_selection_bps`` penalty in the trade
      direction. You only miss when price ran away, so the chase is *systematically* adverse —
      modelling it as anything cheaper would invent free alpha.

    The net differential per side is therefore ``p*(fee+spread) - (1-p)*adverse_selection`` —
    exactly the EV-screen formula, now with ``p`` driven empirically by the close direction.
    ``adverse_selection_bps`` defaults to a deliberately punitive 5 bps (the screen's central case).

    Pricing is still delegated to the ONE :class:`TransactionCostModel`; this model only
    chooses maker-vs-taker liquidity and adds the adverse-selection penalty on the chase.
    """

    __slots__ = ("_adverse_selection_frac", "_cost_model")

    def __init__(
        self, cost_model: TransactionCostModel, *, adverse_selection_bps: float = 5.0
    ) -> None:
        if not math.isfinite(adverse_selection_bps) or adverse_selection_bps < 0.0:
            raise ValueError(
                f"adverse_selection_bps must be finite and >= 0, got {adverse_selection_bps!r}"
            )
        self._cost_model = cost_model
        self._adverse_selection_frac = adverse_selection_bps * 1e-4

    @property
    def cost_model(self) -> TransactionCostModel:
        """The shared cost authority this model prices with (read-only)."""
        return self._cost_model

    def fill(
        self,
        order: OrderRequest,
        inst: Instrument,
        next_bar: BarView,
        *,
        adv_quote: float,
        sigma_daily: float,
    ) -> Fill:
        """Fill ``order`` maker-first against ``next_bar``; chase as taker on a non-fill.

        Raises:
            LookaheadError: if ``next_bar.ts_open < order.decision_ts`` (same contract as
                :class:`NextOpenFill` — a decision at the close of bar t fills in bar t+1).
            CostModelMisuse: propagated from the cost model on the taker-chase branch for
                out-of-regime inputs (notional > 5% ADV, non-finite adv/sigma).
        """
        if next_bar.ts_open < order.decision_ts:
            raise LookaheadError(
                f"fill bar opening at {next_bar.ts_open} precedes the decision at "
                f"{order.decision_ts} for {order.instrument_id!r} "
                f"(order {order.client_order_id!r}): decisions at the close of bar t "
                "fill at the open of bar t+1, never the same bar"
            )
        if inst.instrument_id != order.instrument_id:
            raise ValueError(
                f"instrument mismatch: order is for {order.instrument_id!r} but "
                f"fill was offered {inst.instrument_id!r}"
            )
        ref = next_bar.open
        is_buy = order.side is Side.BUY
        # CONSERVATIVE close-direction proxy (see class docstring): count a passive fill only
        # when the bar closed in the direction that brings liquidity to a resting order. ~50%
        # fill rate, a deliberate lower bound; an adverse-direction bar is a miss that chases.
        passive_filled = (next_bar.close < ref) if is_buy else (next_bar.close > ref)
        if passive_filled:
            price = ref  # executed at the open with no spread crossed — pure maker
            fee = self._cost_model.fee_frac(inst, Liquidity.MAKER) * order.qty * price
            liquidity = Liquidity.MAKER
        else:
            # post-only missed -> chase at taker, with a systematic adverse-selection penalty
            base = self._cost_model.fill_price(
                inst, order.side, ref, order.qty * ref, adv_quote, sigma_daily
            )
            price = base + order.side.sign * ref * self._adverse_selection_frac
            fee = self._cost_model.fee_frac(inst, Liquidity.TAKER) * order.qty * price
            liquidity = Liquidity.TAKER
        return Fill(
            client_order_id=order.client_order_id,
            instrument_id=order.instrument_id,
            side=order.side,
            qty=order.qty,
            price=price,
            fee_quote=fee,
            liquidity=liquidity,
            ts=next_bar.ts_open,
        )
