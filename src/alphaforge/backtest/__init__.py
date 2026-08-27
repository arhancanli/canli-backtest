"""AlphaForge event-driven backtester — the truth engine (execDesign.md §4).

Phase 5 surface: the :class:`Ledger` (accounting truth), fill models
(:class:`NextOpenFill` over :class:`BarView`), the :class:`EventDrivenBacktester`
with its :class:`Strategy`/:class:`StrategyContext` protocol (plus
:class:`ScriptedStrategy` for tests and the :class:`RankStrategy` demo), cost-input
providers, and :class:`BacktestResult` artifact persistence.

Non-negotiables enforced across this package:

* Decisions at the close of bar ``t`` fill at the open of bar ``t+1`` — never the
  same bar (``LookaheadError`` otherwise; leakageCritique.md findings 4/5).
* Funding cashflows come from the stored funding-events table per instrument,
  interval-aware (8h/4h/1h) — never a hard-coded clock (finding 6).
* One ``TransactionCostModel`` instance (``alphaforge.costs``) is the sole
  fee/cost authority (buildabilityCritique.md §3.7).
"""

from alphaforge.backtest.engine import (
    CostInputProvider,
    EventDrivenBacktester,
    LakeCostInputs,
    RankStrategy,
    ScriptedStrategy,
    StaticCostInputs,
    Strategy,
    StrategyContext,
)
from alphaforge.backtest.fills import (
    BarView,
    FillModel,
    MakerFill,
    NextOpenFill,
    ParticipationCappedFill,
)
from alphaforge.backtest.ledger import Ledger
from alphaforge.backtest.result import BacktestResult

__all__ = [
    "BacktestResult",
    "BarView",
    "CostInputProvider",
    "EventDrivenBacktester",
    "FillModel",
    "LakeCostInputs",
    "Ledger",
    "MakerFill",
    "NextOpenFill",
    "ParticipationCappedFill",
    "RankStrategy",
    "ScriptedStrategy",
    "StaticCostInputs",
    "Strategy",
    "StrategyContext",
]
