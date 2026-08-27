from __future__ import annotations

import pytest

from alphaforge.execution.financing import (
    DayCountBasis,
    FinancingQuote,
    StaticFinancingDataProvider,
    accrue_financing,
)

DAY = 86_400_000


def quote(**changes: object) -> FinancingQuote:
    values: dict[str, object] = {
        "currency": "USD",
        "observed_ts": 0,
        "available_at": 0,
        "valid_from": 0,
        "valid_until": 10 * DAY,
        "credit_rate_bps": 400.0,
        "debit_rate_bps": 700.0,
        "short_proceeds_rate_bps": 100.0,
        "day_count": DayCountBasis.ACT_360,
        "source": "synthetic-locked-test",
    }
    values.update(changes)
    return FinancingQuote(**values)  # type: ignore[arg-type]


def test_positive_cash_separates_short_collateral() -> None:
    accrual = accrue_financing(
        quote(),
        cash_balance=120_000.0,
        short_market_value=20_000.0,
        start_ts=0,
        end_ts=DAY,
        decision_ts=0,
    )
    assert accrual.unrestricted_credit_base == 100_000.0
    assert accrual.short_proceeds_base == 20_000.0
    assert accrual.debit_base == 0.0
    assert accrual.payment_quote == pytest.approx((100_000 * 0.04 + 20_000 * 0.01) / 360)


def test_negative_cash_pays_debit_rate() -> None:
    accrual = accrue_financing(
        quote(),
        cash_balance=-50_000.0,
        short_market_value=0.0,
        start_ts=0,
        end_ts=3 * DAY,
        decision_ts=0,
    )
    assert accrual.debit_base == 50_000.0
    assert accrual.payment_quote == pytest.approx(-50_000 * 0.07 * 3 / 360)


def test_quote_must_cover_complete_interval() -> None:
    with pytest.raises(ValueError, match="complete accrual interval"):
        accrue_financing(
            quote(valid_until=DAY),
            cash_balance=1.0,
            short_market_value=0.0,
            start_ts=0,
            end_ts=2 * DAY,
            decision_ts=0,
        )


def test_static_provider_never_returns_future_known_quote() -> None:
    provider = StaticFinancingDataProvider(
        quotes=(quote(available_at=DAY, valid_from=0, valid_until=2 * DAY),)
    )
    assert provider.quote("USD", as_of=DAY - 1) is None
    assert provider.quote("USD", as_of=DAY) is not None
