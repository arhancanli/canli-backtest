"""Point-in-time equity corporate-action primitives.

The lake stores raw prices, so holdings must be transformed by explicit lifecycle
events rather than inferred from price jumps.  This module validates the event
shape and its availability lineage; the backtest engine owns event ordering and
the ledger owns cash/position accounting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from alphaforge.core.errors import LookaheadError
from alphaforge.core.time import Ms

__all__ = ["CorporateAction", "CorporateActionType"]


class CorporateActionType(StrEnum):
    SPLIT = "split"
    CASH_DIVIDEND = "dividend"


@dataclass(frozen=True, slots=True, kw_only=True)
class CorporateAction:
    """One source-bound split or cash-dividend event.

    ``ratio`` follows the lake convention ``split_to / split_from``.  A cash
    dividend carries ``ratio=1`` and a positive per-share ``cash_amount``.
    ``available_at`` is retained separately from ``ex_date`` so simulation can
    reject a record that was not knowable before its economic boundary.
    """

    instrument_id: str
    action_type: CorporateActionType
    ex_date: Ms
    available_at: Ms
    ratio: float
    cash_amount: float | None

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id cannot be empty")
        if self.ex_date < 0 or self.available_at < 0:
            raise ValueError("corporate-action timestamps must be nonnegative")
        if not math.isfinite(self.ratio) or self.ratio <= 0.0:
            raise ValueError("corporate-action ratio must be finite and > 0")
        if self.action_type is CorporateActionType.SPLIT:
            if self.cash_amount is not None:
                raise ValueError("split cash_amount must be null")
        else:
            if self.ratio != 1.0:
                raise ValueError("cash-dividend ratio must equal 1")
            if self.cash_amount is None or not math.isfinite(self.cash_amount):
                raise ValueError("cash dividend requires a finite cash_amount")
            if self.cash_amount <= 0.0:
                raise ValueError("cash dividend cash_amount must be > 0")

    def require_known_by(self, decision_ts: Ms) -> None:
        """Raise when this event was unavailable at ``decision_ts``."""
        if self.available_at > decision_ts:
            raise LookaheadError(
                f"corporate action for {self.instrument_id!r} available at "
                f"{self.available_at} exceeds decision {decision_ts}"
            )

    def require_known_before_boundary(self) -> None:
        """Fail closed when a lifecycle event arrived after its ex boundary."""
        if self.available_at > self.ex_date:
            raise LookaheadError(
                f"corporate action for {self.instrument_id!r} at {self.ex_date} "
                f"was not available until {self.available_at}; replay cannot "
                "apply it at the economic boundary without future data"
            )
