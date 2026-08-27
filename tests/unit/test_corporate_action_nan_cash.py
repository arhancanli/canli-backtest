"""The lake writes NaN, not NULL, for a split that carries no cash.

`CorporateAction.__post_init__` requires `cash_amount is None` for a SPLIT. The lake stores
"no cash" as NaN, and `NaN is None` is False, so every split reaching the engine raised
`ValueError("split cash_amount must be null")` and aborted the whole run. This was not rare:
all 4,804 splits in `data/lake_sharadar/corporate_actions` and 4,512 of 5,156 in
`data/lake/corporate_actions` carry NaN. It killed the AlphaLedger pinned re-run 26 minutes in,
and any equity backtest whose window contains a split hits it.

These tests pin the CALLER, not the constructor's intention. A test that instantiates
CorporateAction directly would pass both before and after the fix and prove nothing about the
path that actually runs -- the same shape as the funding-carry defect, where nine tests pinned
the intention while production booked funding zero times in 44 days.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

from alphaforge.backtest.engine import EventDrivenBacktester
from alphaforge.execution.corporate_actions import CorporateAction, CorporateActionType

_IID = "XUSE:CASH:AAONUSD"


def _table(action_type: str, ratio: float, cash: float | None) -> pa.Table:
    return pa.table(
        {
            "instrument_id": pa.array([_IID], pa.string()),
            "action_type": pa.array([action_type], pa.string()),
            "ex_date": pa.array([1_000_000], pa.int64()),
            "available_at": pa.array([500_000], pa.int64()),
            "ratio": pa.array([ratio], pa.float64()),
            "cash_amount": pa.array([cash], pa.float64()),
        }
    )


class _StubReader:
    """Returns one corp-action row, the way the real reader hands it to the engine."""

    def __init__(self, table: pa.Table) -> None:
        self._table = table

    def corporate_actions(self, instrument_ids, *, start, end, as_of):
        return self._table


def _load(table: pa.Table) -> dict[str, list[CorporateAction]]:
    """Drive the real `_load_corporate_actions` with a stub reader."""
    engine = object.__new__(EventDrivenBacktester)
    engine._reader = _StubReader(table)  # type: ignore[attr-defined]
    return EventDrivenBacktester._load_corporate_actions(
        engine, [_IID], start=0, end=2_000_000
    )


def test_split_with_nan_cash_loads() -> None:
    """The regression. Before the fix this raised ValueError and aborted the run."""
    out = _load(_table("split", 2.0, math.nan))
    (action,) = out[_IID]
    assert action.action_type is CorporateActionType.SPLIT
    assert action.cash_amount is None, "NaN must be normalised to None at the boundary"


def test_split_with_null_cash_still_loads() -> None:
    """The 644 rows that already stored a true NULL must be unaffected."""
    out = _load(_table("split", 2.0, None))
    assert out[_IID][0].cash_amount is None


def test_cash_dividend_with_nan_still_fails_loudly() -> None:
    """The fix must not become a blanket NaN swallow.

    A dividend whose amount is genuinely missing is a real data defect and must keep failing.
    If this test ever passes-by-not-raising, the fix has been widened into a bug.
    """
    with pytest.raises(ValueError, match="finite cash_amount"):
        _load(_table("dividend", 1.0, math.nan))


def test_cash_dividend_with_real_amount_is_preserved() -> None:
    """Guards the other direction: the coercion must not eat a legitimate value."""
    out = _load(_table("dividend", 1.0, 0.42))
    assert out[_IID][0].cash_amount == pytest.approx(0.42)


def test_this_check_can_fail() -> None:
    """A check that cannot fail is worse than no check.

    Reproduces the pre-fix expression verbatim and asserts it still produces the NaN that
    broke the run -- so if someone 'simplifies' the boundary back, the suite notices.
    """
    cash = math.nan
    pre_fix = None if cash is None else float(cash)
    assert pre_fix is not None and math.isnan(pre_fix)
    with pytest.raises(ValueError, match="split cash_amount must be null"):
        CorporateAction(
            instrument_id=_IID,
            action_type=CorporateActionType.SPLIT,
            ex_date=1_000_000,
            available_at=500_000,
            ratio=2.0,
            cash_amount=pre_fix,
        )
