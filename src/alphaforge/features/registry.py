"""FeatureRegistry — the single process-wide name → FeatureSpec mapping.

One registry, one function body, two callers (dataDesign.md §7.1): library modules
register spec *factories* at import time via the :func:`feature` decorator; research
(``compute_history``) and live (``compute_asof``) resolve the SAME
:class:`~alphaforge.features.spec.FeatureSpec` (hence the same ``fn`` object) from
here. The factory pattern — not bare specs — is deliberate (leakage finding 20): the
factory is the single construction site where ``lookback_bars`` is *derived from* the
params it is registered with, so a window parameter can never drift away from the
history the live path loads.

This module registers exactly two trivial reference features (``ret_log_1``,
``close_px``) used by the engine/parity test suites and as the canonical registration
example; the real factor library lives in ``alphaforge.features.library`` modules and
registers itself on import through the same decorator.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Final

import numpy as np

from alphaforge.features.context import FeatureContext, long_series
from alphaforge.features.spec import Family, FeatureSpec

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "FeatureRegistry",
    "close_px",
    "default_registry",
    "feature",
    "ret_log_1",
]


class FeatureRegistry:
    """Name → :class:`FeatureSpec` mapping with duplicate-name protection.

    Specs are stored in registration order (stable, import-order deterministic).
    Instances are mutable only through :meth:`register`; everything handed out is
    the frozen spec itself.
    """

    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    def register(self, spec_factory: Callable[..., FeatureSpec]) -> FeatureSpec:
        """Build the spec by calling ``spec_factory()`` and register it by name.

        The factory must be callable with no arguments and is invoked exactly once,
        eagerly — a malformed spec fails at import time, not at first compute.
        Returns the constructed spec. Raises ``ValueError`` on a duplicate name
        (names are global identities; silently shadowing a factor is how two
        versions of a formula end up in one backtest).
        """
        spec = spec_factory()
        if spec.name in self._specs:
            raise ValueError(
                f"feature {spec.name!r} is already registered; names are unique "
                "registry keys (bump the name, not the body)"
            )
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> FeatureSpec:
        """Return the spec registered under ``name``; ``KeyError`` with candidates if absent."""
        try:
            return self._specs[name]
        except KeyError:
            raise KeyError(
                f"no feature registered under {name!r}; known: {sorted(self._specs)}"
            ) from None

    def all_specs(self) -> list[FeatureSpec]:
        """All registered specs, in registration order (a fresh list each call)."""
        return list(self._specs.values())

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[FeatureSpec]:
        return iter(self._specs.values())


_DEFAULT_REGISTRY: Final[FeatureRegistry] = FeatureRegistry()


def default_registry() -> FeatureRegistry:
    """The process-global registry populated by importing the library modules."""
    return _DEFAULT_REGISTRY


def feature(spec_factory: Callable[..., FeatureSpec]) -> Callable[..., FeatureSpec]:
    """Decorator: register ``spec_factory()`` into the default registry at import.

    Usage::

        @feature
        def mom_xs_504_48() -> FeatureSpec:
            lookback, skip = 504, 48
            return FeatureSpec(
                name="mom_xs_504_48", ..., lookback_bars=lookback + 1,
                params={"lookback": lookback, "skip": skip}, fn=_xs_momentum,
            )

    The factory body is the single place where ``lookback_bars`` is derived from the
    params (leakage finding 20). Returns the factory unchanged so it stays
    importable/testable.
    """
    default_registry().register(spec_factory)
    return spec_factory


# ----------------------------------------------------------------- reference features


def _ret_log_1_fn(ctx: FeatureContext, spec: FeatureSpec) -> pd.Series:
    """1-bar log return.

    Formula (alphaDesign.md conventions): with ``w = params["window"]``::

        r_t = ln(C_t / C_{t-w})

    computed on the complete expected bar grid (gaps stay NaN; a return is never
    silently taken across a missing bar — both the gap slot and the bar after it are
    NaN because ``C_{t-w}`` is absent from the grid). Value labeled ``ts_open`` is
    decision-usable at the bar close ``ts_open + Δ``. NaN until ``w`` prior grid
    slots exist.
    """
    window = int(spec.params["window"])
    close = ctx.panel("close")
    out: pd.DataFrame = np.log(close / close.shift(window))  # type: ignore[assignment]
    return long_series(out, name=spec.name)


def _close_px_fn(ctx: FeatureContext, spec: FeatureSpec) -> pd.Series:
    """Identity close price: ``v_t = C_t`` on the complete expected bar grid.

    Not an alpha (direction 0) — a plumbing/reference feature: its batch/as-of
    parity isolates pure window-alignment bugs from formula bugs.
    """
    return long_series(ctx.panel("close"), name=spec.name)


@feature
def ret_log_1() -> FeatureSpec:
    """Reference feature: 1-bar log return (lookback derived: window + 1 bars)."""
    window = 1
    return FeatureSpec(
        name="ret_log_1",
        family=Family.MARKET,
        direction=0,
        cross_sectional=False,
        lookback_bars=window + 1,
        params={"window": window},
        fn=_ret_log_1_fn,
    )


@feature
def close_px() -> FeatureSpec:
    """Reference feature: identity close price (lookback 1 bar)."""
    return FeatureSpec(
        name="close_px",
        family=Family.MARKET,
        direction=0,
        cross_sectional=False,
        lookback_bars=1,
        params={},
        fn=_close_px_fn,
    )
