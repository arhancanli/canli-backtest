"""Unit tests for alphaforge.features.spec and alphaforge.features.registry.

Covers: FeatureSpec construction invariants (name/direction/lookback validation,
params immutability, the EWMA-family lookback convention), registry registration via
spec factories, duplicate-name rejection, lookup errors, registration order, the
``feature`` decorator, and the two pre-registered reference features.
"""

from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.features.context import FeatureContext
from alphaforge.features.registry import FeatureRegistry, default_registry, feature
from alphaforge.features.spec import EWMA_LOOKBACK_SPANS, Family, FeatureSpec


def _noop_fn(ctx: FeatureContext, spec: FeatureSpec) -> pd.Series:
    raise AssertionError("reference fn should not be invoked in spec tests")


def make_spec(**overrides: object) -> FeatureSpec:
    kwargs: dict[str, object] = {
        "name": "test_feature",
        "family": Family.MOMENTUM,
        "direction": 1,
        "cross_sectional": True,
        "lookback_bars": 10,
        "params": {"window": 10},
        "fn": _noop_fn,
    }
    kwargs.update(overrides)
    return FeatureSpec(**kwargs)  # type: ignore[arg-type]


# ------------------------------------------------------------------ FeatureSpec


class TestFeatureSpec:
    def test_valid_spec_constructs(self) -> None:
        spec = make_spec()
        assert spec.name == "test_feature"
        assert spec.family is Family.MOMENTUM
        assert spec.direction == 1
        assert spec.cross_sectional is True
        assert spec.lookback_bars == 10
        assert spec.params["window"] == 10
        assert spec.fn is _noop_fn

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name"):
            make_spec(name="")

    @pytest.mark.parametrize("direction", [-2, 2, 5])
    def test_direction_outside_unit_rejected(self, direction: int) -> None:
        with pytest.raises(ValueError, match="direction"):
            make_spec(direction=direction)

    @pytest.mark.parametrize("direction", [-1, 0, 1])
    def test_direction_unit_values_accepted(self, direction: int) -> None:
        assert make_spec(direction=direction).direction == direction

    @pytest.mark.parametrize("lookback", [0, -1, -100])
    def test_nonpositive_lookback_rejected(self, lookback: int) -> None:
        with pytest.raises(ValueError, match="lookback_bars"):
            make_spec(lookback_bars=lookback)

    def test_non_int_lookback_rejected(self) -> None:
        with pytest.raises(ValueError, match="lookback_bars"):
            make_spec(lookback_bars=2.5)

    def test_non_callable_fn_rejected(self) -> None:
        with pytest.raises(ValueError, match="callable"):
            make_spec(fn="not a function")

    def test_params_are_immutable(self) -> None:
        spec = make_spec(params={"window": 10})
        with pytest.raises(TypeError):
            spec.params["window"] = 99  # type: ignore[index]

    def test_params_copied_from_caller_dict(self) -> None:
        raw = {"window": 10}
        spec = make_spec(params=raw)
        raw["window"] = 99
        assert spec.params["window"] == 10

    def test_spec_is_frozen(self) -> None:
        spec = make_spec()
        with pytest.raises(AttributeError):
            spec.name = "other"  # type: ignore[misc]

    def test_is_ewma_family_default_false(self) -> None:
        assert make_spec().is_ewma_family is False


class TestEwmaFamilyConvention:
    def test_valid_ewma_spec(self) -> None:
        spec = make_spec(
            params={"span": 8, "ewma_family": True},
            lookback_bars=EWMA_LOOKBACK_SPANS * 8,
        )
        assert spec.is_ewma_family is True

    def test_ewma_requires_span(self) -> None:
        with pytest.raises(ValueError, match="span"):
            make_spec(params={"ewma_family": True}, lookback_bars=100)

    def test_ewma_rejects_non_int_span(self) -> None:
        with pytest.raises(ValueError, match="span"):
            make_spec(params={"ewma_family": True, "span": 8.0}, lookback_bars=100)

    def test_ewma_lookback_below_10x_span_rejected(self) -> None:
        with pytest.raises(ValueError, match="10"):
            make_spec(
                params={"span": 8, "ewma_family": True},
                lookback_bars=EWMA_LOOKBACK_SPANS * 8 - 1,
            )


# ------------------------------------------------------------------ registry


class TestFeatureRegistry:
    def test_register_returns_spec_and_stores_it(self) -> None:
        registry = FeatureRegistry()
        spec = registry.register(lambda: make_spec(name="alpha_one"))
        assert spec.name == "alpha_one"
        assert "alpha_one" in registry
        assert registry.get("alpha_one") is spec
        assert len(registry) == 1

    def test_duplicate_name_rejected(self) -> None:
        registry = FeatureRegistry()
        registry.register(lambda: make_spec(name="dup"))
        with pytest.raises(ValueError, match="dup"):
            registry.register(lambda: make_spec(name="dup"))

    def test_invalid_spec_fails_at_registration_time(self) -> None:
        registry = FeatureRegistry()
        with pytest.raises(ValueError, match="lookback_bars"):
            registry.register(lambda: make_spec(name="bad", lookback_bars=0))
        assert "bad" not in registry

    def test_get_unknown_raises_keyerror_with_candidates(self) -> None:
        registry = FeatureRegistry()
        registry.register(lambda: make_spec(name="known"))
        with pytest.raises(KeyError, match="known"):
            registry.get("unknown")

    def test_all_specs_registration_order(self) -> None:
        registry = FeatureRegistry()
        registry.register(lambda: make_spec(name="zzz"))
        registry.register(lambda: make_spec(name="aaa"))
        assert [s.name for s in registry.all_specs()] == ["zzz", "aaa"]
        assert [s.name for s in registry] == ["zzz", "aaa"]

    def test_all_specs_returns_fresh_list(self) -> None:
        registry = FeatureRegistry()
        registry.register(lambda: make_spec(name="one"))
        listing = registry.all_specs()
        listing.clear()
        assert len(registry) == 1


class TestDefaultRegistryAndDecorator:
    def test_default_registry_is_a_singleton(self) -> None:
        assert default_registry() is default_registry()

    def test_reference_features_registered(self) -> None:
        registry = default_registry()
        ret = registry.get("ret_log_1")
        px = registry.get("close_px")
        assert ret.lookback_bars == 2  # window + 1, derived from params
        assert ret.params["window"] == 1
        assert ret.direction == 0
        assert ret.cross_sectional is False
        assert px.lookback_bars == 1
        assert px.direction == 0

    def test_feature_decorator_registers_and_returns_factory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The decorator targets the process-global default registry. Swap in a
        # fresh registry for the duration of the test so the throwaway spec never
        # pollutes the real registry (which downstream consumers — the IC report,
        # the registry-wide invariant sweep — enumerate as "all real factors").
        import alphaforge.features.registry as registry_module

        scratch = FeatureRegistry()
        monkeypatch.setattr(registry_module, "_DEFAULT_REGISTRY", scratch)
        name = "test_decorator_feature_spec"
        assert name not in default_registry()

        @feature
        def throwaway() -> FeatureSpec:
            return make_spec(name=name)

        assert name in default_registry()
        assert default_registry() is scratch
        assert callable(throwaway)
        assert throwaway().name == name
