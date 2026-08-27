"""The transaction cost model — single source of truth (execDesign §3.2).

One :class:`TransactionCostModel` instance is shared by the vectorized
backtester, the event-driven backtester, ``PaperBroker`` and the optimizer's
turnover penalty. It is **never reimplemented** elsewhere
(buildabilityCritique §3.7): if any layer needs a cost number, it asks this
object.

Cost anatomy for an order of quote notional ``Q`` in instrument ``i``:

1. **Commission** — venue fee schedule by market type (§3.1),
   ``fee_frac = fee_bps * 1e-4``.
2. **Half-spread** — ``hs_i``: per-instrument override in bps if observed,
   else a deliberately conservative 2.5 bps default for a mixed top-30
   universe (BTC/ETH perps actually run ~0.5-1 bp).
3. **Square-root market impact** —
   ``impact_frac = Y * sigma_daily * sqrt(Q / ADV)`` with ``Y = 1.0`` default
   (Tóth et al. 2011, Almgren 2005: empirical Y clusters in 0.5-1.5).
   Valid only for small participation: pre-trade checks reject ``Q > 1% ADV``
   upstream (execDesign §7.1); this model hard-fails at ``Q > 5% ADV`` with
   :class:`~alphaforge.core.errors.CostModelMisuse` rather than extrapolate a
   law that underestimates cost outside its regime.
4. **Latency add-on** — flat 2 bps placeholder for the decision→fill delay
   (leakageCritique finding 7). Applied to the **fill price only**, never to
   the optimizer's cost penalty: see :meth:`TransactionCostModel.fill_price`
   vs :meth:`TransactionCostModel.oneway_cost_frac` for why the split exists.

Units: all prices/notionals are float64 in quote units (USDT); all rate
outputs are dimensionless fractions of notional; timestamps elsewhere are
epoch-ms UTC. Internal math runs at full float64 precision — rounding is a
reporting concern only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from alphaforge.core.errors import CostModelMisuse
from alphaforge.core.types import AssetClass, Liquidity, MarketType, Side
from alphaforge.costs.fees import (
    BINANCE_VIP0_PERP,
    BINANCE_VIP0_SPOT,
    US_EQUITY_DEFAULT,
    FeeSchedule,
)

_DAYS_PER_YEAR: float = 365.0

if TYPE_CHECKING:
    from alphaforge.config.settings import Settings
    from alphaforge.core.instruments import Instrument

__all__ = ["MAX_ADV_PARTICIPATION", "TransactionCostModel"]

MAX_ADV_PARTICIPATION: float = 0.05
"""Hard validity ceiling for the sqrt impact law as a fraction of ADV.

Pre-trade checks reject orders above 1% of ADV (execDesign §7.1), so the model
normally never sees anything close to this. Reaching 5% means an upstream
guard is broken; the model raises instead of returning a number it knows is
wrong."""


def _require_positive_finite(name: str, value: float) -> None:
    """Reject non-finite or non-positive runtime inputs with CostModelMisuse."""
    if not math.isfinite(value) or value <= 0.0:
        raise CostModelMisuse(f"{name} must be finite and > 0, got {value!r}")


def _require_param(name: str, value: float) -> None:
    """Reject non-finite or negative constructor parameters (programming error)."""
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"TransactionCostModel: {name} must be finite and >= 0, got {value!r}")


class TransactionCostModel:
    """All trading frictions; one instance shared system-wide.

    Constructor parameters mirror ``CostsCfg`` (configs/base.yaml ``costs:``);
    prefer :meth:`from_settings` so config remains the single dial. Rates are
    accepted in basis points and converted to fractions internally exactly
    once. The instance is immutable in practice (no public mutators) so it is
    safe to share across the backtester, broker and optimizer.
    """

    def __init__(
        self,
        spot_fees: FeeSchedule = BINANCE_VIP0_SPOT,
        perp_fees: FeeSchedule = BINANCE_VIP0_PERP,
        equity_fees: FeeSchedule = US_EQUITY_DEFAULT,
        impact_coef: float = 1.0,
        default_half_spread_bps: float = 2.5,
        latency_bps: float = 2.0,
        borrow_bps_annual: float = 0.0,
        half_spread_overrides: Mapping[str, float] | None = None,
    ) -> None:
        _require_param("impact_coef", impact_coef)
        _require_param("default_half_spread_bps", default_half_spread_bps)
        _require_param("latency_bps", latency_bps)
        _require_param("borrow_bps_annual", borrow_bps_annual)
        overrides: dict[str, float] = dict(half_spread_overrides or {})
        for instrument_id, bps in overrides.items():
            _require_param(f"half_spread_overrides[{instrument_id!r}]", bps)
        self._spot_fees = spot_fees
        self._perp_fees = perp_fees
        self._equity_fees = equity_fees
        self._impact_coef = impact_coef
        self._default_half_spread_frac = default_half_spread_bps * 1e-4
        self._latency_frac = latency_bps * 1e-4
        # Short-borrow as a per-DAY fraction of notional (annual GC rate / 365); the ledger
        # multiplies it by holding days x short notional, mirroring the perp funding leg.
        self._borrow_frac_per_day = (borrow_bps_annual * 1e-4) / _DAYS_PER_YEAR
        self._half_spread_frac_overrides: Mapping[str, float] = MappingProxyType(
            {iid: bps * 1e-4 for iid, bps in overrides.items()}
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> TransactionCostModel:
        """Build the model from validated ``Settings`` (configs ``costs:`` section).

        ``CostsCfg`` carries the perp fee tier, impact coefficient, default
        half-spread and the latency add-on. Spot fees are pinned to
        :data:`~alphaforge.costs.fees.BINANCE_VIP0_SPOT` (spot is not traded
        in the current system; the constant exists so the routing is total).
        Half-spread overrides start empty — they are populated from observed
        spreads, not config guesses.

        SLEEVE-AWARE: for the EQUITY sleeve (``data.asset_class == EQUITY``) the
        commission schedule, the half-spread default and the short-borrow rate are
        taken from the equity ``CostsCfg`` fields, so an equity backtest is NOT priced
        on the Binance crypto schedule. Crypto profiles resolve byte-identically (no
        equity field touched, borrow 0).
        """
        cfg = settings.costs
        is_equity = settings.data.asset_class is AssetClass.EQUITY
        return cls(
            spot_fees=BINANCE_VIP0_SPOT,
            perp_fees=FeeSchedule(maker_bps=cfg.perp_maker_bps, taker_bps=cfg.perp_taker_bps),
            equity_fees=FeeSchedule(
                maker_bps=cfg.equity_commission_bps, taker_bps=cfg.equity_commission_bps
            ),
            impact_coef=cfg.impact_coef,
            default_half_spread_bps=(
                cfg.equity_half_spread_bps if is_equity else cfg.default_half_spread_bps
            ),
            latency_bps=cfg.latency_addon_bps,
            borrow_bps_annual=cfg.equity_borrow_bps_annual if is_equity else 0.0,
            half_spread_overrides=None,
        )

    # ------------------------------------------------------------------ rates

    def fee_frac(self, inst: Instrument, liquidity: Liquidity) -> float:
        """Commission as a fraction of quote notional.

        Schedule routed by ``inst.market_type``: PERP → perp schedule, CASH → equity,
        SPOT → spot. Dated futures and options fail closed until product/venue-specific
        commission models exist; borrowing a crypto schedule would fabricate implementation cost.
        """
        if inst.market_type is MarketType.PERP:
            schedule = self._perp_fees
        elif inst.market_type is MarketType.CASH:
            schedule = self._equity_fees
        elif inst.market_type is MarketType.SPOT:
            schedule = self._spot_fees
        else:
            raise CostModelMisuse(
                f"no transaction-fee schedule registered for market type "
                f"{inst.market_type.value!r} ({inst.instrument_id!r})"
            )
        return schedule.fee_frac(liquidity)

    def borrow_frac_per_day(self) -> float:
        """Short-borrow cost as a per-DAY fraction of the short notional.

        The general-collateral annual borrow rate / 365 (0 for crypto perps, which have
        no borrow leg). The ledger accrues ``borrow_frac_per_day * holding_days *
        short_notional`` on the SHORT side of an equity book, mirroring the per-interval
        funding leg of a perp. A continuous carry, NOT a per-trade friction — separate from
        :meth:`oneway_cost_frac`."""
        return self._borrow_frac_per_day

    def half_spread_frac(self, inst: Instrument) -> float:
        """Half the bid-ask spread as a fraction of price.

        Per-instrument override (keyed by canonical ``instrument_id``,
        populated from observed spreads) wins; otherwise the conservative
        default (2.5 bps) applies.
        """
        override = self._half_spread_frac_overrides.get(inst.instrument_id)
        return self._default_half_spread_frac if override is None else override

    def impact_frac(self, notional: float, adv_quote: float, sigma_daily: float) -> float:
        """Square-root-law market impact as a fraction of price.

            impact_frac = Y * sigma_daily * sqrt(notional / adv_quote)

        ``sigma_daily`` is the EWMA daily return vol (``sigma_1h * sqrt(24)``,
        halflife 240 bars); ``adv_quote`` is the 30-day rolling *median* daily
        quote volume (median resists wash/news spikes). Both are supplied
        as-of the decision bar by the data layer.

        Raises :class:`CostModelMisuse` if ``notional > 0.05 * adv_quote``:
        the sqrt law is only trustworthy for participation ≲1% and
        *underestimates* cost beyond that, so the model refuses to
        extrapolate. Pre-trade checks reject at 1% of ADV upstream
        (execDesign §7.1); 5% here is a tripwire for broken guards, never a
        knob to loosen.
        """
        _require_positive_finite("impact_frac: notional", notional)
        _require_positive_finite("impact_frac: adv_quote", adv_quote)
        _require_positive_finite("impact_frac: sigma_daily", sigma_daily)
        if notional > MAX_ADV_PARTICIPATION * adv_quote:
            raise CostModelMisuse(
                f"sqrt impact law invalid: notional {notional!r} exceeds "
                f"{MAX_ADV_PARTICIPATION:.0%} of ADV {adv_quote!r} "
                "(pre-trade checks should have rejected this at 1% of ADV)"
            )
        return self._impact_coef * sigma_daily * math.sqrt(notional / adv_quote)

    def latency_frac(self, inst: Instrument) -> float:
        """Decision→fill latency slippage as a fraction of price.

        Flat 2 bps placeholder, deliberately conservative (leakageCritique
        finding 7: the next-open fill already models the bar delay, this adds
        the intra-second arrival drift live fills will see). Replaced by a
        per-instrument calibrated arrival-slippage estimate from paper-trading
        fills in Phase 12; the signature already takes ``inst`` so that swap
        changes no call sites.
        """
        del inst  # placeholder is flat; parameter kept for the Phase 12 calibrated version
        return self._latency_frac

    # ------------------------------------------------------------------ totals

    def oneway_cost_frac(
        self,
        inst: Instrument,
        notional: float,
        adv_quote: float,
        sigma_daily: float,
        liquidity: Liquidity = Liquidity.TAKER,
    ) -> float:
        """Total one-way cost as a fraction of notional (execDesign §3.2):

            cost_frac = fee_frac + half_spread_frac + impact_frac

        Used by the optimizer's turnover penalty and the vectorized engine.
        **Deliberately excludes the latency add-on**: latency is a *price*
        phenomenon — it shifts the fill price the ledger records
        (:meth:`fill_price`) — not an economic penalty the optimizer should
        double-count when deciding whether a trade clears its cost hurdle.
        Including it here and in the fill price would charge it twice.
        """
        return (
            self.fee_frac(inst, liquidity)
            + self.half_spread_frac(inst)
            + self.impact_frac(notional, adv_quote, sigma_daily)
        )

    def fill_price(
        self,
        inst: Instrument,
        side: Side,
        ref_price: float,
        notional: float,
        adv_quote: float,
        sigma_daily: float,
    ) -> float:
        """Execution price after spread, impact and latency slippage:

            P = ref_price * (1 + s * (half_spread + impact + latency)),
            s = +1 for BUY, -1 for SELL

        ``ref_price`` is the next bar's open (the engine enforces the
        decide-at-close-t, fill-at-open-t+1 contract; this model never sees
        the decision bar). The commission is **not** baked into the price —
        the ledger charges it as a separate cash debit via the fee schedule,
        keeping price slippage and commissions separately attributable in
        artifacts. Latency lives here and *only* here (see
        :meth:`oneway_cost_frac` for the other half of that contract).
        Full float64 precision; rounding is for reports only.
        """
        _require_positive_finite("fill_price: ref_price", ref_price)
        slip = (
            self.half_spread_frac(inst)
            + self.impact_frac(notional, adv_quote, sigma_daily)
            + self.latency_frac(inst)
        )
        return ref_price * (1.0 + float(side.sign) * slip)
