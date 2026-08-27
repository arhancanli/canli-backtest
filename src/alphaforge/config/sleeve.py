"""Per-asset-class sleeve registry — the single source of (anchor TF, calendar).

The engine spine (features, backtest, labeling, portfolio, signals, walk-forward)
is valued on a per-sleeve **anchor timeframe** and reads its grid/annualization from
the matching :class:`~alphaforge.core.calendar.TradingCalendar`. This module is the ONE
place that maps an :class:`~alphaforge.core.types.AssetClass` to its anchor TF; the
calendar itself is the stateless shared singleton owned by
:func:`~alphaforge.core.calendar.calendar_for` (keyed off ``asset_class`` so there is
one source of truth — the MIC re-key, audit G11/P18, is a later arc).

THE BYTE-IDENTITY RULE: the crypto sleeves anchor :data:`~alphaforge.core.time.Timeframe.H1`
on the 24/7 :class:`~alphaforge.core.calendar.Always24x7Calendar` — *exactly* the current
hard-coded behavior. Routing the spine through this registry is therefore an identity
refactor on the crypto path: ``sleeve_for(AssetClass.CRYPTO_PERP)`` resolves H1 + 24/7,
whose ``expected_bar_opens``/``periods_per_year`` delegate to the same ``core.time``
kernel / ``tf.bars_per_year`` the spine calls today. The EQUITY sleeve anchors
:data:`~alphaforge.core.time.Timeframe.D1` on the XNYS session calendar (252-day basis).
DATED FUTURES and OPTIONS are intentionally absent: taxonomy and execution-domain support
do not imply a research sleeve, and each must name a product-specific exchange calendar
rather than inherit a generic default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from alphaforge.core.calendar import TradingCalendar, calendar_for
from alphaforge.core.errors import ConfigError
from alphaforge.core.time import Timeframe
from alphaforge.core.types import AssetClass

__all__ = [
    "CRYPTO_PERP_SLEEVE",
    "CRYPTO_SPOT_SLEEVE",
    "EQUITY_SLEEVE",
    "Sleeve",
    "sleeve_for",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class Sleeve:
    """One asset-class sleeve: its anchor timeframe + the calendar key.

    The anchor timeframe is the grid features/signals/sizing are valued on and the
    grid ``lookback_bars`` counts. CRYPTO sleeves anchor :data:`Timeframe.H1` (the
    current behavior); the EQUITY sleeve anchors :data:`Timeframe.D1`. The calendar is
    :func:`~alphaforge.core.calendar.calendar_for` of :attr:`asset_class` — not stored
    on the sleeve, because the calendar is a stateless shared singleton owned by
    ``core/calendar.py`` and keying off ``asset_class`` keeps one source of truth.
    """

    asset_class: AssetClass
    anchor_tf: Timeframe
    #: Walk-forward / CPCV embargo in ANCHOR-TF BARS (= sessions for D1, hours for H1). Must
    #: be >= the sleeve's deepest active factor lookback so a train sample adjacent to a test
    #: fold cannot share feature-window data with it. Crypto: 168 (the 7-day-in-hours default).
    #: Equity: 274 (> eq_beta_252 / eq_rev_resid_21's ~254-274-session lookbacks).
    embargo_bars: int = 168

    @property
    def calendar(self) -> TradingCalendar:
        """The shared trading calendar for this sleeve's asset class."""
        return calendar_for(self.asset_class)

    def periods_per_year(self) -> float:
        """Annualization factor for the anchor TF — the ONE source for this sleeve.

        Equals :meth:`TradingCalendar.periods_per_year` of the anchor TF: ``8760.0``
        for the crypto H1 sleeves (24/7), ``252.0`` for the equity D1 sleeve (XNYS).
        Sharpe/vol/covariance annualizers read this, never a bare ``tf.bars_per_year``.
        """
        return self.calendar.periods_per_year(self.anchor_tf)


CRYPTO_PERP_SLEEVE: Final = Sleeve(asset_class=AssetClass.CRYPTO_PERP, anchor_tf=Timeframe.H1)
CRYPTO_SPOT_SLEEVE: Final = Sleeve(asset_class=AssetClass.CRYPTO_SPOT, anchor_tf=Timeframe.H1)
EQUITY_SLEEVE: Final = Sleeve(
    asset_class=AssetClass.EQUITY, anchor_tf=Timeframe.D1, embargo_bars=274
)

_REGISTRY: Final[dict[AssetClass, Sleeve]] = {
    s.asset_class: s for s in (CRYPTO_PERP_SLEEVE, CRYPTO_SPOT_SLEEVE, EQUITY_SLEEVE)
}


def sleeve_for(asset_class: AssetClass) -> Sleeve:
    """Return the registered :class:`Sleeve` for ``asset_class``.

    The single resolution helper every spine entry point calls to get its
    ``(anchor_tf, calendar)``. CRYPTO_PERP/CRYPTO_SPOT resolve to the H1 + 24/7 sleeve
    (byte-identical to today); EQUITY resolves to the D1 + XNYS sleeve. Dated futures
    remain unregistered until a product-specific calendar and research path exist.

    Raises:
        ConfigError: if ``asset_class`` has no registered sleeve.
    """
    try:
        return _REGISTRY[asset_class]
    except KeyError:
        raise ConfigError(f"no sleeve registered for {asset_class.value!r}") from None
