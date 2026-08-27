"""Point-in-time securities-borrow, locate, recall, and buy-in primitives.

General-collateral assumptions are not sufficient for security-level short execution.
This module keeps availability, quantity, fee, locate expiry, and recall deadlines explicit.
It does not infer historical borrow from today's broker flags or fabricate fills after a recall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from alphaforge.core.errors import LookaheadError
from alphaforge.core.time import Ms

__all__ = [
    "BorrowDataProvider",
    "BorrowQuote",
    "BorrowStatus",
    "LocateDecision",
    "LocateStatus",
    "RecallInstruction",
    "RecallNotice",
    "StaticBorrowDataProvider",
    "accrue_borrow_charge",
    "evaluate_locate",
    "incremental_short_qty",
    "recall_instruction",
]

_MS_PER_YEAR = 365.0 * 86_400_000.0


def _finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _nonnegative_finite(name: str, value: float) -> None:
    _finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")


class BorrowStatus(StrEnum):
    EASY = "easy"
    HARD = "hard"
    UNAVAILABLE = "unavailable"


class LocateStatus(StrEnum):
    GRANTED = "granted"
    PARTIAL = "partial"
    DENIED = "denied"


class BorrowDataProvider(Protocol):
    """PIT security-lending data seam consumed by simulation and live adapters."""

    def quote(self, instrument_id: str, *, as_of: Ms) -> BorrowQuote | None:
        """Latest quote known and effective at ``as_of``, or ``None``."""
        ...

    def recalls_known(self, *, as_of: Ms) -> tuple[RecallNotice, ...]:
        """All recall notices available by ``as_of`` in deterministic order."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class BorrowQuote:
    """Security-level borrow observation and its exact effective interval."""

    instrument_id: str
    observed_ts: Ms
    available_at: Ms
    valid_from: Ms
    valid_until: Ms
    status: BorrowStatus
    available_qty: float
    annual_fee_bps: float
    utilization: float | None = None

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id cannot be empty")
        if self.available_at < self.observed_ts:
            raise ValueError("available_at cannot precede observed_ts")
        if self.valid_from >= self.valid_until:
            raise ValueError("valid_from must precede valid_until")
        _nonnegative_finite("available_qty", self.available_qty)
        _nonnegative_finite("annual_fee_bps", self.annual_fee_bps)
        if self.utilization is not None:
            _finite("utilization", self.utilization)
            if not 0.0 <= self.utilization <= 1.0:
                raise ValueError("utilization must be in [0, 1]")
        if self.status is BorrowStatus.UNAVAILABLE and self.available_qty != 0.0:
            raise ValueError("unavailable borrow must report available_qty=0")

    def require_effective(self, decision_ts: Ms) -> None:
        if self.available_at > decision_ts:
            raise LookaheadError(
                f"borrow quote for {self.instrument_id!r} available at {self.available_at} "
                f"exceeds decision {decision_ts}"
            )
        if not self.valid_from <= decision_ts < self.valid_until:
            raise ValueError("borrow quote is not effective at the decision timestamp")


@dataclass(frozen=True, slots=True, kw_only=True)
class LocateDecision:
    """Deterministic quantity-bounded locate outcome."""

    status: LocateStatus
    instrument_id: str
    requested_qty: float
    granted_qty: float
    annual_fee_bps: float | None
    expires_at: Ms | None
    reason: str


def evaluate_locate(
    quote: BorrowQuote,
    *,
    requested_qty: float,
    decision_ts: Ms,
) -> LocateDecision:
    """Grant no more than the quantity explicitly available at decision time."""
    _nonnegative_finite("requested_qty", requested_qty)
    quote.require_effective(decision_ts)
    if requested_qty == 0.0:
        return LocateDecision(
            status=LocateStatus.GRANTED,
            instrument_id=quote.instrument_id,
            requested_qty=0.0,
            granted_qty=0.0,
            annual_fee_bps=quote.annual_fee_bps,
            expires_at=quote.valid_until,
            reason="no_incremental_short",
        )
    if quote.status is BorrowStatus.UNAVAILABLE or quote.available_qty == 0.0:
        return LocateDecision(
            status=LocateStatus.DENIED,
            instrument_id=quote.instrument_id,
            requested_qty=requested_qty,
            granted_qty=0.0,
            annual_fee_bps=None,
            expires_at=None,
            reason="borrow_unavailable",
        )
    granted = min(requested_qty, quote.available_qty)
    status = LocateStatus.GRANTED if granted == requested_qty else LocateStatus.PARTIAL
    return LocateDecision(
        status=status,
        instrument_id=quote.instrument_id,
        requested_qty=requested_qty,
        granted_qty=granted,
        annual_fee_bps=quote.annual_fee_bps,
        expires_at=quote.valid_until,
        reason="quantity_available" if status is LocateStatus.GRANTED else "quantity_capped",
    )


def incremental_short_qty(*, current_signed_qty: float, sell_qty: float) -> float:
    """Locate needed for a sell after accounting for long inventory and existing shorts."""
    _finite("current_signed_qty", current_signed_qty)
    _nonnegative_finite("sell_qty", sell_qty)
    existing_short = max(-current_signed_qty, 0.0)
    resulting_short = max(-(current_signed_qty - sell_qty), 0.0)
    return resulting_short - existing_short


def accrue_borrow_charge(
    quote: BorrowQuote,
    *,
    short_qty: float,
    mark_price: float,
    start_ts: Ms,
    end_ts: Ms,
    decision_ts: Ms,
) -> float:
    """Positive ACT/365 borrow charge when one quote covers the entire accrual interval."""
    _nonnegative_finite("short_qty", short_qty)
    _nonnegative_finite("mark_price", mark_price)
    if mark_price == 0.0:
        raise ValueError("mark_price must be > 0")
    if start_ts >= end_ts:
        raise ValueError("start_ts must precede end_ts")
    quote.require_effective(decision_ts)
    if start_ts < quote.valid_from or end_ts > quote.valid_until:
        raise ValueError("borrow quote does not cover the full accrual interval")
    elapsed_years = (end_ts - start_ts) / _MS_PER_YEAR
    return short_qty * mark_price * quote.annual_fee_bps * 1e-4 * elapsed_years


@dataclass(frozen=True, slots=True, kw_only=True)
class RecallNotice:
    """Point-in-time broker/lender recall with a mandatory cover deadline."""

    instrument_id: str
    recalled_qty: float
    event_ts: Ms
    available_at: Ms
    cover_deadline: Ms

    def __post_init__(self) -> None:
        _nonnegative_finite("recalled_qty", self.recalled_qty)
        if self.recalled_qty == 0.0:
            raise ValueError("recalled_qty must be > 0")
        if self.available_at < self.event_ts:
            raise ValueError("recall available_at cannot precede event_ts")
        if self.cover_deadline < self.available_at:
            raise ValueError("cover_deadline cannot precede notice availability")


@dataclass(frozen=True, slots=True, kw_only=True)
class RecallInstruction:
    """Cover quantity and urgency emitted from an already-known recall notice."""

    instrument_id: str
    buy_qty: float
    submit_by: Ms
    forced_buy_in: bool
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class StaticBorrowDataProvider:
    """Deterministic in-memory provider for locked fixtures and artifact replay."""

    quotes: tuple[BorrowQuote, ...]
    recalls: tuple[RecallNotice, ...] = ()

    def quote(self, instrument_id: str, *, as_of: Ms) -> BorrowQuote | None:
        eligible = [
            quote
            for quote in self.quotes
            if quote.instrument_id == instrument_id
            and quote.available_at <= as_of
            and quote.valid_from <= as_of < quote.valid_until
        ]
        if not eligible:
            return None
        return max(eligible, key=lambda quote: (quote.available_at, quote.observed_ts))

    def recalls_known(self, *, as_of: Ms) -> tuple[RecallNotice, ...]:
        return tuple(
            sorted(
                (notice for notice in self.recalls if notice.available_at <= as_of),
                key=lambda notice: (
                    notice.available_at,
                    notice.event_ts,
                    notice.instrument_id,
                ),
            )
        )


def recall_instruction(
    notice: RecallNotice,
    *,
    current_short_qty: float,
    decision_ts: Ms,
) -> RecallInstruction:
    """Cover the recalled amount, capped by the actual outstanding short position."""
    _nonnegative_finite("current_short_qty", current_short_qty)
    if notice.available_at > decision_ts:
        raise LookaheadError(
            f"recall for {notice.instrument_id!r} available at {notice.available_at} "
            f"exceeds decision {decision_ts}"
        )
    buy_qty = min(current_short_qty, notice.recalled_qty)
    forced = buy_qty > 0.0 and decision_ts >= notice.cover_deadline
    return RecallInstruction(
        instrument_id=notice.instrument_id,
        buy_qty=buy_qty,
        submit_by=notice.cover_deadline,
        forced_buy_in=forced,
        reason=(
            "forced_buy_in_deadline_reached"
            if forced
            else "recall_cover_required" if buy_qty > 0.0 else "no_short_remaining"
        ),
    )
