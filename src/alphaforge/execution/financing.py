"""Point-in-time cash, margin-debit, and short-collateral financing primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from alphaforge.core.errors import LookaheadError
from alphaforge.core.time import Ms

__all__ = [
    "DayCountBasis",
    "FinancingAccrual",
    "FinancingDataProvider",
    "FinancingQuote",
    "StaticFinancingDataProvider",
    "accrue_financing",
]


class DayCountBasis(StrEnum):
    ACT_360 = "ACT/360"
    ACT_365 = "ACT/365"

    @property
    def days_per_year(self) -> float:
        return 360.0 if self is DayCountBasis.ACT_360 else 365.0


def _finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class FinancingQuote:
    """One effective financing schedule with publication lineage.

    Rates are annual basis points. ``credit_rate_bps`` applies to unrestricted
    positive cash, ``debit_rate_bps`` to negative cash, and
    ``short_proceeds_rate_bps`` to positive cash collateralized by short market
    value. Credit and short-proceeds rates may be negative; debit rates may not.
    """

    currency: str
    observed_ts: Ms
    available_at: Ms
    valid_from: Ms
    valid_until: Ms
    credit_rate_bps: float
    debit_rate_bps: float
    short_proceeds_rate_bps: float
    day_count: DayCountBasis = DayCountBasis.ACT_360
    source: str = ""

    def __post_init__(self) -> None:
        if not self.currency or self.currency != self.currency.upper():
            raise ValueError("currency must be non-empty uppercase text")
        if self.available_at < self.observed_ts:
            raise ValueError("available_at cannot precede observed_ts")
        if self.valid_from >= self.valid_until:
            raise ValueError("valid_from must precede valid_until")
        _finite("credit_rate_bps", self.credit_rate_bps)
        _finite("debit_rate_bps", self.debit_rate_bps)
        _finite("short_proceeds_rate_bps", self.short_proceeds_rate_bps)
        if self.debit_rate_bps < 0.0:
            raise ValueError("debit_rate_bps must be >= 0")
        if not self.source:
            raise ValueError("financing source cannot be empty")

    def require_effective(self, *, start_ts: Ms, end_ts: Ms, decision_ts: Ms) -> None:
        if start_ts >= end_ts:
            raise ValueError("financing start_ts must precede end_ts")
        if self.available_at > decision_ts:
            raise LookaheadError(
                f"financing quote for {self.currency} available at {self.available_at} "
                f"exceeds decision {decision_ts}"
            )
        if start_ts < self.valid_from or end_ts > self.valid_until:
            raise ValueError("financing quote does not cover the complete accrual interval")


@dataclass(frozen=True, slots=True, kw_only=True)
class FinancingAccrual:
    currency: str
    start_ts: Ms
    end_ts: Ms
    cash_balance: float
    unrestricted_credit_base: float
    short_proceeds_base: float
    debit_base: float
    credit_rate_bps: float
    debit_rate_bps: float
    short_proceeds_rate_bps: float
    day_count: DayCountBasis
    payment_quote: float
    source: str


class FinancingDataProvider(Protocol):
    def quote(self, currency: str, *, as_of: Ms) -> FinancingQuote | None:
        """Return the effective schedule known at ``as_of``, or ``None``."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class StaticFinancingDataProvider:
    """Deterministic in-memory provider for replay and adapter tests."""

    quotes: tuple[FinancingQuote, ...]

    def quote(self, currency: str, *, as_of: Ms) -> FinancingQuote | None:
        candidates = [
            quote
            for quote in self.quotes
            if quote.currency == currency
            and quote.available_at <= as_of
            and quote.valid_from <= as_of < quote.valid_until
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda quote: (quote.valid_from, quote.available_at))


def accrue_financing(
    quote: FinancingQuote,
    *,
    cash_balance: float,
    short_market_value: float,
    start_ts: Ms,
    end_ts: Ms,
    decision_ts: Ms,
) -> FinancingAccrual:
    """Accrue one fully covered interval without crediting segregated short proceeds.

    Positive cash up to current short market value is treated as restricted
    collateral and receives the explicit short-proceeds rate. Remaining positive
    cash receives the ordinary credit rate. Negative cash pays the debit rate.
    """
    _finite("cash_balance", cash_balance)
    _finite("short_market_value", short_market_value)
    if short_market_value < 0.0:
        raise ValueError("short_market_value must be >= 0")
    quote.require_effective(start_ts=start_ts, end_ts=end_ts, decision_ts=decision_ts)

    positive_cash = max(cash_balance, 0.0)
    short_base = min(positive_cash, short_market_value)
    unrestricted_base = positive_cash - short_base
    debit_base = max(-cash_balance, 0.0)
    elapsed_years = (end_ts - start_ts) / (
        quote.day_count.days_per_year * 86_400_000.0
    )
    payment = elapsed_years * 1e-4 * (
        unrestricted_base * quote.credit_rate_bps
        + short_base * quote.short_proceeds_rate_bps
        - debit_base * quote.debit_rate_bps
    )
    _finite("financing payment", payment)
    return FinancingAccrual(
        currency=quote.currency,
        start_ts=start_ts,
        end_ts=end_ts,
        cash_balance=cash_balance,
        unrestricted_credit_base=unrestricted_base,
        short_proceeds_base=short_base,
        debit_base=debit_base,
        credit_rate_bps=quote.credit_rate_bps,
        debit_rate_bps=quote.debit_rate_bps,
        short_proceeds_rate_bps=quote.short_proceeds_rate_bps,
        day_count=quote.day_count,
        payment_quote=payment,
        source=quote.source,
    )
