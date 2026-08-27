"""Exchange fee schedules (execDesign §3.1).

This module is part of ``alphaforge.costs`` — the single source of truth for
every trading friction in the system. Fee rates are declared exactly once,
here; the vectorized backtester, the event-driven backtester, ``PaperBroker``
and the optimizer all consume them through
:class:`~alphaforge.costs.model.TransactionCostModel` and never re-declare a
basis point anywhere else (buildabilityCritique §3.7).

Units and rounding policy
-------------------------
Rates are stored in **basis points of quote notional** (1 bp = 1e-4). The fee
charged on a fill is

    ``fee_quote = |qty| * price * fee_bps * 1e-4``

with ``qty`` in base units, ``price`` in quote units, hence ``fee_quote`` in
quote units (USDT). All money math is float64 at full precision; rounding
happens only at the reporting boundary, never inside the ledger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from alphaforge.core.types import Liquidity

__all__ = [
    "BINANCE_VIP0_PERP",
    "BINANCE_VIP0_SPOT",
    "US_EQUITY_DEFAULT",
    "FeeSchedule",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class FeeSchedule:
    """Maker/taker commission rates in basis points of quote notional.

    Immutable by design: a schedule is venue/tier metadata, not mutable state.
    Rates must be finite and non-negative (rebate tiers, if ever needed, get
    their own explicit type rather than a sneaky negative number here).
    """

    maker_bps: float
    taker_bps: float

    def __post_init__(self) -> None:
        for name, value in (("maker_bps", self.maker_bps), ("taker_bps", self.taker_bps)):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"FeeSchedule.{name} must be finite and >= 0, got {value!r}")

    def bps(self, liquidity: Liquidity) -> float:
        """Commission rate in basis points for the given liquidity flag."""
        return self.maker_bps if liquidity is Liquidity.MAKER else self.taker_bps

    def fee_frac(self, liquidity: Liquidity) -> float:
        """Commission as a fraction of quote notional: ``fee_bps * 1e-4``."""
        return self.bps(liquidity) * 1e-4

    def fee_quote(self, qty: float, price: float, liquidity: Liquidity) -> float:
        """Fee in quote units on a fill: ``|qty| * price * fee_bps * 1e-4``.

        ``qty`` is base units (sign ignored — fees are paid on both sides),
        ``price`` is the fill price in quote units. Full float64 precision;
        the caller rounds for display only.
        """
        if not math.isfinite(qty):
            raise ValueError(f"FeeSchedule.fee_quote: qty must be finite, got {qty!r}")
        if not math.isfinite(price) or price <= 0.0:
            raise ValueError(f"FeeSchedule.fee_quote: price must be finite and > 0, got {price!r}")
        return abs(qty) * price * self.fee_frac(liquidity)


BINANCE_VIP0_SPOT = FeeSchedule(maker_bps=10.0, taker_bps=10.0)
"""Binance spot VIP0, no BNB discount: 0.10% maker and taker (execDesign §3.1)."""

BINANCE_VIP0_PERP = FeeSchedule(maker_bps=2.0, taker_bps=5.0)
"""Binance USDⓈ-M perp VIP0: 0.02% maker / 0.05% taker (execDesign §3.1)."""

US_EQUITY_DEFAULT = FeeSchedule(maker_bps=1.0, taker_bps=1.0)
"""US cash-equity commission default: a conservative 1 bp per side (EQUITIES_SLEEVE §6).

US equity commissions are near-zero at retail but an institutional book pays exchange/SEC/
TAF fees plus a small per-share commission; 1 bp all-in (maker == taker — there is no perp
maker-rebate analogue here) is a deliberately conservative round number. The real equity
frictions are the bid-ask spread and impact, not commission. Overridable via
``CostsCfg.equity_commission_bps``; the borrow leg for shorts is separate (see
``TransactionCostModel.borrow_frac_per_day``)."""
