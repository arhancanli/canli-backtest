"""Transaction costs — the single fee/cost authority (execDesign §3).

Every layer that needs a cost number (vectorized backtester, event-driven
backtester, ``PaperBroker``, optimizer turnover penalty, pre-trade checks)
imports from this package and shares **one**
:class:`~alphaforge.costs.model.TransactionCostModel` instance. Fee rates and
cost formulas are never re-declared elsewhere (buildabilityCritique §3.7).
"""

from alphaforge.costs.fees import BINANCE_VIP0_PERP, BINANCE_VIP0_SPOT, FeeSchedule
from alphaforge.costs.model import MAX_ADV_PARTICIPATION, TransactionCostModel

__all__ = [
    "BINANCE_VIP0_PERP",
    "BINANCE_VIP0_SPOT",
    "MAX_ADV_PARTICIPATION",
    "FeeSchedule",
    "TransactionCostModel",
]
