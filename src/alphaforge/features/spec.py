"""FeatureSpec — identity, contract, and data requirements of one registered feature.

This is THE merged factor abstraction (buildabilityCritique.md ruling 3.4): Design A's
``FeatureSpec`` machinery absorbs Design B's ``FactorMeta`` fields (``family``,
``direction``, ``cross_sectional``). There is exactly one feature framework in this
codebase and it is this one — research (``FeatureEngine.compute_history``) and live
(``FeatureEngine.compute_asof``) both execute the *same* registered function object
(leakage finding 3), so train/serve skew reduces to window-boundary bugs, which
:mod:`alphaforge.features.parity` catches mechanically.

Load-bearing invariants enforced at construction:

* ``lookback_bars`` is the EXACT maximum history (in anchor-timeframe bars, INCLUDING
  the bar being valued) the function touches. It sizes the live path's minimal window
  and the truncation-invariance harness; an understated value is caught by
  :func:`alphaforge.features.parity.verify_truncation`, an unstated one cannot exist
  (``> 0`` enforced here). Per leakage finding 20 it must be *derived from* ``params``
  inside the spec factory, never hand-typed independently of them — the registry's
  factory pattern (:func:`alphaforge.features.registry.feature`) keeps the derivation
  and the params at a single construction site.
* EWMA-family features (infinite-memory recursions) cannot be bit-identical between a
  truncated live window and a full-history pass. Convention (buildabilityCritique.md
  §6.1): declare ``params["ewma_family"] = True`` plus the ``span`` used, and set
  ``lookback_bars >= EWMA_LOOKBACK_SPANS * span`` so the truncation error is far below
  the parity tolerance (1e-9 relative; see :func:`alphaforge.features.parity.parity_tolerance`).
  Finite-window features keep the exact contract: parity atol = 0.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    import pandas as pd

    from alphaforge.features.context import FeatureContext

__all__ = [
    "EWMA_LOOKBACK_SPANS",
    "Family",
    "FeatureFn",
    "FeatureSpec",
]


EWMA_LOOKBACK_SPANS: Final[int] = 10
"""Minimum ``lookback_bars / span`` ratio for EWMA-family features.

With smoothing factor ``alpha = 2/(span+1)``, the total weight of observations older
than ``L = 10*span`` bars is ``(1-alpha)^L ~= e^-20 ~= 2e-9``; because those observations
deviate from the local level by a bounded fraction of it, the induced *relative*
truncation error is well below the 1e-9 parity tolerance for EWMA-family specs.
"""


class Family(StrEnum):
    """Factor family taxonomy (alphaDesign.md §2). Drives blending/reporting grouping."""

    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    CARRY = "carry"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    MARKET = "market"
    REGIME = "regime"
    VALUE = "value"
    QUALITY = "quality"


type FeatureFn = Callable[[FeatureContext, FeatureSpec], pd.Series]
"""The registered pure feature function.

Contract: given a PIT-windowed :class:`~alphaforge.features.context.FeatureContext`
and its own :class:`FeatureSpec` (for params), return a float-convertible
:class:`pandas.Series` indexed by a 2-level MultiIndex ``(ts_open: int64 epoch ms,
instrument_id: str)`` covering ALL requested instruments/timestamps in the context
window, with NaN wherever history is insufficient. Pure function discipline: no I/O,
no state, NEVER mutate context-provided tables (they are shared, copy-on-write
shallow views). The value labeled ``ts_open`` is the decision made at that bar's
close ``ts_open + Δ`` — it may use the bar itself, never anything after it.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class FeatureSpec:
    """Immutable identity + contract of one feature (see module docstring).

    Attributes:
        name: Unique registry key, e.g. ``"mom_xs_504_48"``.
        family: Factor family (taxonomy only; no behavior attached here).
        direction: ``+1`` higher value ⇒ higher expected forward return, ``-1``
            inverse, ``0`` not-an-alpha (risk/regime/utility feature).
        cross_sectional: ``True`` ⇒ the cross-sectional pipeline post-processing
            (winsorize/neutralize/zscore) applies before the value is consumed.
        lookback_bars: EXACT max history needed, in anchor bars including the bar
            being valued; must be ``> 0`` and derived from ``params`` (finding 20).
        params: Feature parameters; stored as an immutable mapping. Reserved keys:
            ``ewma_family`` (bool) marks infinite-memory recursions and relaxes the
            parity tolerance; it requires an integer ``span`` and
            ``lookback_bars >= EWMA_LOOKBACK_SPANS * span``.
        fn: The registered pure function (see :data:`FeatureFn`).
    """

    name: str
    family: Family
    direction: int
    cross_sectional: bool
    lookback_bars: int
    params: Mapping[str, Any]
    fn: FeatureFn

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FeatureSpec.name must be non-empty")
        if self.direction not in (-1, 0, 1):
            raise ValueError(
                f"FeatureSpec.direction must be -1, 0 or +1, got {self.direction!r} "
                f"for {self.name!r}"
            )
        if not isinstance(self.lookback_bars, int) or self.lookback_bars <= 0:
            raise ValueError(
                f"FeatureSpec.lookback_bars must be an int > 0, got "
                f"{self.lookback_bars!r} for {self.name!r} (leakage finding 20: derive "
                "it from params in the spec factory)"
            )
        if not callable(self.fn):
            raise ValueError(f"FeatureSpec.fn must be callable for {self.name!r}")
        # Freeze params against mutation; the spec is shared process-wide.
        frozen: Mapping[str, Any] = MappingProxyType(dict(self.params))
        object.__setattr__(self, "params", frozen)
        if frozen.get("ewma_family"):
            span = frozen.get("span")
            if not isinstance(span, int) or span <= 0:
                raise ValueError(
                    f"EWMA-family spec {self.name!r} must declare an int params['span'] "
                    f"> 0, got {span!r} (buildabilityCritique.md §6.1)"
                )
            min_lookback = EWMA_LOOKBACK_SPANS * span
            if self.lookback_bars < min_lookback:
                raise ValueError(
                    f"EWMA-family spec {self.name!r} declares lookback_bars="
                    f"{self.lookback_bars} < {EWMA_LOOKBACK_SPANS}*span={min_lookback}; "
                    "the truncation-error convention requires lookback_bars >= "
                    f"{EWMA_LOOKBACK_SPANS}x span (buildabilityCritique.md §6.1)"
                )

    @property
    def is_ewma_family(self) -> bool:
        """True iff this spec opted into the EWMA warmup/tolerance convention."""
        return bool(self.params.get("ewma_family"))
