from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from alphaforge.validation.dsr import expected_max_sharpe, probabilistic_sharpe_ratio

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/build_dsr_calculator_contract.py"
ARTIFACT = ROOT / "artifacts/engineering/deflated_sharpe_calculator_contract.json"


def _module():
    spec = importlib.util.spec_from_file_location("build_dsr_calculator_contract", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_persisted_contract_matches_builder_and_source_hashes() -> None:
    module = _module()
    persisted = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert persisted == module.build_contract()
    assert persisted["content_hash"] == module._content_hash(persisted)
    assert persisted["source_bindings"]["implementation"]["sha256"] == module._sha256(
        module.DSR_SOURCE
    )
    assert persisted["source_bindings"]["admission_contract"]["sha256"] == module._sha256(
        module.ADMISSION_SOURCE
    )


@pytest.mark.parametrize("vector_index", [0, 1, 2])
def test_golden_vectors_reproduce_production_formula(vector_index: int) -> None:
    vector = _module().build_contract()["test_vectors"][vector_index]
    inputs = vector["inputs"]
    outputs = vector["outputs"]
    scale = math.sqrt(inputs["periods_per_year"])
    sr = inputs["observed_sharpe_annualized"] / scale
    variance = (inputs["cross_trial_sharpe_sd_annualized"] / scale) ** 2
    benchmark = expected_max_sharpe(inputs["effective_independent_trials"], variance)

    assert outputs["observed_sharpe_per_period"] == pytest.approx(sr, abs=1e-15)
    assert outputs["cross_trial_sharpe_variance_per_period"] == pytest.approx(variance, abs=1e-15)
    assert outputs["expected_max_sharpe_per_period"] == pytest.approx(benchmark, abs=1e-15)
    assert outputs["psr_against_zero"] == pytest.approx(
        probabilistic_sharpe_ratio(
            sr,
            inputs["observations"],
            inputs["skew"],
            inputs["non_excess_kurtosis"],
        ),
        abs=1e-15,
    )
    assert outputs["deflated_sharpe_ratio"] == pytest.approx(
        probabilistic_sharpe_ratio(
            sr,
            inputs["observations"],
            inputs["skew"],
            inputs["non_excess_kurtosis"],
            sr_benchmark=benchmark,
        ),
        abs=1e-15,
    )
