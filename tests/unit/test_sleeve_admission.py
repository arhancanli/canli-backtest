from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from alphaforge.validation.sleeve_admission import (
    evaluate_sleeve_evidence,
    load_admission_contract,
)

CONTRACT_PATH = Path(__file__).parents[2] / "config/sleeve_admission_contract.json"


def _tested_execution(dimension: str) -> dict:
    manifest = []
    for case in ("base", "stress", "extreme"):
        assumptions = {"dimension": dimension, "case": case}
        result = {"passed": True, "observations": 100}
        manifest.append(
            {
                "scenario_id": f"{dimension}_{case}",
                "status": "PASS",
                "assumptions": assumptions,
                "assumptions_sha256": _canonical_hash(assumptions),
                "result": result,
                "result_sha256": _canonical_hash(result),
            }
        )
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return {
        "status": "TESTED_PASS",
        "evidence_sha256": hashlib.sha256(canonical).hexdigest(),
        "scenarios": len(manifest),
        "scenario_manifest": manifest,
    }


def _canonical_hash(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def passing_evidence() -> dict:
    contract = load_admission_contract(CONTRACT_PATH)
    digest = "a" * 64
    return {
        "lineage": {
            "preregistration_sha256": digest,
            "data_manifest_sha256": digest,
            "diversification_report_sha256": digest,
            "code_commit": "abcdef1234567890",
            "family_trial_account": "family_alpha",
            "return_identity_id": "family_alpha_v1",
            "point_in_time_data": True,
            "survivorship_control": True,
            "corporate_action_control": True,
            "direction_locked": True,
            "parameters_locked": True,
        },
        "robustness": dict.fromkeys(contract["required_robustness"], True),
        "statistics": {
            "oos_observations": 756,
            "stressed_oos_observations": 126,
            "net_sharpe": 0.8,
            "stressed_sharpe": 0.55,
            "newey_west_t": 1.2,
            "deflated_sharpe": 0.97,
            "book_deflated_sharpe": 0.96,
            "pbo": {"status": "MEASURED", "value": 0.12, "family_variants": 6},
        },
        "diversification": {
            "absolute_beta": 0.04,
            "average_pairwise_correlation": -0.03,
            "candidate_average_correlation_to_existing_book": -0.02,
            "book_average_pairwise_correlation_delta": -0.005,
            "average_pairwise_correlation_upper_95": 0.04,
            "max_pairwise_correlation": 0.22,
            "max_stressed_pairwise_correlation": 0.38,
            "max_pairwise_correlation_upper_95": 0.31,
            "max_stressed_pairwise_correlation_upper_95": 0.47,
            "correlation_observations": 756,
        },
        "portfolio": {
            "book_sharpe_delta": 0.03,
            "minimum_leave_one_period_out_book_sharpe_delta": 0.01,
            "book_max_drawdown_delta": -0.005,
            "book_expected_shortfall_delta": -0.002,
            "book_sharpe_delta_lower_95": 0.008,
            "book_expected_max_drawdown": 0.102,
        },
        "overlay": {
            "covariance_halflife_days": 21,
            "realized_vol_halflife_days": 240,
            "realized_vol_leg_is_unlevered": True,
        },
        "capacity": {
            "capacity_usd": 8_000_000,  # far above the floor; the curve is what reconciles it
            "minimum_stressed_fill_ratio": 0.96,
            "curve": [
                {
                    "capital_usd": 1_000_000,
                    "net_sharpe": 0.8,
                    "fill_ratio": 1.0,
                    "stressed_cost_bps": 5.0,
                },
                {
                    "capital_usd": 5_000_000,
                    "net_sharpe": 0.7,
                    "fill_ratio": 0.98,
                    "stressed_cost_bps": 9.0,
                },
                {
                    "capital_usd": 8_000_000,
                    "net_sharpe": 0.55,
                    "fill_ratio": 0.96,
                    "stressed_cost_bps": 14.0,
                },
            ],
        },
        "execution": {
            dimension: _tested_execution(dimension)
            for dimension in contract["execution_dimensions"]
        },
    }


def test_complete_evidence_is_technically_eligible() -> None:
    contract = load_admission_contract(CONTRACT_PATH)
    report = evaluate_sleeve_evidence(passing_evidence(), contract)

    assert report.eligible
    assert report.failures == ()
    # Derived from the contract, not typed: a hand-maintained count silently goes stale the
    # moment a gate is added, and the drift is invisible because the assertion still passes on
    # the old number until someone notices. The contract already publishes the figure.
    assert report.checks_evaluated == contract["evidence_checks_per_candidate"]


def test_missing_execution_dimension_fails_closed() -> None:
    contract = load_admission_contract(CONTRACT_PATH)
    evidence = passing_evidence()
    del evidence["execution"]["partial_fills"]

    report = evaluate_sleeve_evidence(evidence, contract)
    assert not report.eligible
    assert "missing_execution_dimension:partial_fills" in report.failures


def test_target_numbers_cannot_rescue_failed_evidence() -> None:
    contract = load_admission_contract(CONTRACT_PATH)
    evidence = passing_evidence()
    evidence["statistics"]["net_sharpe"] = -1.0
    evidence["portfolio_target_sharpe"] = 2.5

    report = evaluate_sleeve_evidence(evidence, contract)
    assert not report.eligible
    assert any(failure.startswith("threshold:statistics.net_sharpe") for failure in report.failures)


def test_single_identity_pbo_is_explicitly_undefined_never_zero() -> None:
    contract = load_admission_contract(CONTRACT_PATH)
    evidence = passing_evidence()
    evidence["statistics"]["pbo"] = {
        "status": "NOT_DEFINED_SINGLE_IDENTITY",
        "value": None,
        "family_variants": 1,
    }
    report = evaluate_sleeve_evidence(evidence, contract)
    assert report.eligible
    assert report.warnings

    dishonest = copy.deepcopy(evidence)
    dishonest["statistics"]["pbo"]["value"] = 0.0
    assert not evaluate_sleeve_evidence(dishonest, contract).eligible


def test_not_applicable_execution_requires_real_reason() -> None:
    contract = load_admission_contract(CONTRACT_PATH)
    evidence = passing_evidence()
    evidence["execution"]["option_surfaces_assignment_and_gaps"] = {
        "status": "NOT_APPLICABLE",
        "reason": "Cash equities do not contain option contracts.",
        "applicability_evidence_sha256": "b" * 64,
    }
    assert evaluate_sleeve_evidence(evidence, contract).eligible

    evidence["execution"]["option_surfaces_assignment_and_gaps"]["reason"] = "n/a"
    assert not evaluate_sleeve_evidence(evidence, contract).eligible


def test_nan_bool_and_placeholder_hashes_fail_closed() -> None:
    contract = load_admission_contract(CONTRACT_PATH)
    for path, value in (
        (("statistics", "net_sharpe"), float("nan")),
        (("statistics", "net_sharpe"), True),
        (("diversification", "max_pairwise_correlation_upper_95"), float("nan")),
    ):
        evidence = passing_evidence()
        evidence[path[0]][path[1]] = value
        assert not evaluate_sleeve_evidence(evidence, contract).eligible

    evidence = passing_evidence()
    evidence["lineage"]["preregistration_sha256"] = True
    assert not evaluate_sleeve_evidence(evidence, contract).eligible


def test_capacity_must_reconcile_to_real_strictly_ordered_curve() -> None:
    contract = load_admission_contract(CONTRACT_PATH)
    evidence = passing_evidence()
    evidence["capacity"]["curve"] = [{}, {}, {}]
    assert not evaluate_sleeve_evidence(evidence, contract).eligible

    evidence = passing_evidence()
    evidence["capacity"]["capacity_usd"] = 9_000_000
    assert not evaluate_sleeve_evidence(evidence, contract).eligible

    evidence = passing_evidence()
    evidence["capacity"]["curve"].reverse()
    assert not evaluate_sleeve_evidence(evidence, contract).eligible

    evidence = passing_evidence()
    evidence["capacity"]["curve"][0]["fill_ratio"] = 0.99
    evidence["capacity"]["curve"][1]["fill_ratio"] = 1.0
    assert not evaluate_sleeve_evidence(evidence, contract).eligible

    evidence = passing_evidence()
    evidence["capacity"]["curve"][1]["stressed_cost_bps"] = 4.0
    assert not evaluate_sleeve_evidence(evidence, contract).eligible

    evidence = passing_evidence()
    evidence["capacity"]["minimum_stressed_fill_ratio"] = 0.99
    assert not evaluate_sleeve_evidence(evidence, contract).eligible


def test_point_correlation_cannot_override_confidence_or_tail_gate() -> None:
    contract = load_admission_contract(CONTRACT_PATH)
    evidence = passing_evidence()
    evidence["diversification"]["max_pairwise_correlation"] = 0.05
    evidence["diversification"]["max_pairwise_correlation_upper_95"] = 0.51
    assert not evaluate_sleeve_evidence(evidence, contract).eligible

    evidence = passing_evidence()
    evidence["portfolio"]["book_expected_shortfall_delta"] = 0.001
    assert not evaluate_sleeve_evidence(evidence, contract).eligible


def test_execution_pass_requires_hash_bound_scenario_evidence() -> None:
    contract = load_admission_contract(CONTRACT_PATH)
    evidence = passing_evidence()
    del evidence["execution"]["latency"]["evidence_sha256"]
    assert not evaluate_sleeve_evidence(evidence, contract).eligible

    evidence = passing_evidence()
    evidence["execution"]["latency"]["scenarios"] = 2
    assert not evaluate_sleeve_evidence(evidence, contract).eligible

    evidence = passing_evidence()
    evidence["execution"]["latency"]["scenario_manifest"][0]["result_sha256"] = "b" * 64
    report = evaluate_sleeve_evidence(evidence, contract)
    assert not report.eligible
    assert "execution_manifest_hash_mismatch:latency" in report.failures

    evidence = passing_evidence()
    manifest = evidence["execution"]["latency"]["scenario_manifest"]
    manifest[1]["scenario_id"] = manifest[0]["scenario_id"]
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    evidence["execution"]["latency"]["evidence_sha256"] = hashlib.sha256(canonical).hexdigest()
    report = evaluate_sleeve_evidence(evidence, contract)
    assert "execution_manifest_duplicate_ids:latency" in report.failures

    evidence = passing_evidence()
    manifest = evidence["execution"]["latency"]["scenario_manifest"]
    manifest[2]["status"] = "FAIL"
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    evidence["execution"]["latency"]["evidence_sha256"] = hashlib.sha256(canonical).hexdigest()
    report = evaluate_sleeve_evidence(evidence, contract)
    assert "execution_pass_contains_failed_scenario:latency" in report.failures


def test_execution_scenario_payloads_are_individually_hash_bound() -> None:
    contract = load_admission_contract(CONTRACT_PATH)
    evidence = passing_evidence()
    manifest = evidence["execution"]["latency"]["scenario_manifest"]
    manifest[0]["assumptions"]["case"] = "silently_mutated"
    evidence["execution"]["latency"]["evidence_sha256"] = _canonical_hash(manifest)
    report = evaluate_sleeve_evidence(evidence, contract)
    assert "execution_manifest_assumptions_hash_mismatch:latency:0" in report.failures

    evidence = passing_evidence()
    manifest = evidence["execution"]["latency"]["scenario_manifest"]
    manifest[0]["result"]["passed"] = False
    manifest[0]["result_sha256"] = _canonical_hash(manifest[0]["result"])
    evidence["execution"]["latency"]["evidence_sha256"] = _canonical_hash(manifest)
    report = evaluate_sleeve_evidence(evidence, contract)
    assert "execution_manifest_status_result_mismatch:latency:0" in report.failures
