"""The arithmetic the contract publishes must be the arithmetic the identity gives.

canlicapital.com publishes the fourteen-sleeve objective and the admission gates side by side.
For most of this programme's life it published both WITHOUT publishing the relationship between
them, and that relationship was the whole story: the v4 correlation ceiling of 0.15 capped the
book below the objective at every plausible per-sleeve quality, so the target sat next to a gate
that forbade it. ``frontier_arithmetic`` exists to state that relationship in public.

A block of numbers copied into a config drifts the moment a threshold moves. This re-derives every
published figure from the governing identity -- independently, not by calling the builder, which
would only confirm the builder agrees with itself.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from alphaforge.validation.sleeve_admission import load_admission_contract

CONTRACT_PATH = Path(__file__).parents[2] / "config/sleeve_admission_contract.json"
TOLERANCE = 1e-12


@pytest.fixture(scope="module")
def contract() -> dict:
    return load_admission_contract(CONTRACT_PATH)


def _book_sharpe(s_bar: float, rho_bar: float, n: int) -> float:
    return s_bar * math.sqrt(n / (1.0 + (n - 1) * rho_bar))


def test_block_is_present_and_scoped_to_the_incremental_gates(contract: dict) -> None:
    frontier = contract["frontier_arithmetic"]
    thresholds = contract["thresholds"]
    assert frontier["incremental_candidate_average_correlation_gate"] == thresholds[
        "candidate_average_correlation_to_existing_book_max"
    ]
    assert frontier["incremental_book_average_correlation_delta_gate_exclusive"] == thresholds[
        "book_average_pairwise_correlation_delta_max_exclusive"
    ]
    assert "correlation_gate_in_force" not in frontier
    assert frontier["target_sleeve_count"] == contract["objective"]["target_total_sleeves"]


def test_psd_floor_matches_the_identity(contract: dict) -> None:
    frontier = contract["frontier_arithmetic"]
    n = frontier["target_sleeve_count"]
    assert frontier["psd_floor_at_target_n"] == pytest.approx(-1.0 / (n - 1), abs=TOLERANCE)


def test_zero_global_correlation_reference_matches_the_identity(contract: dict) -> None:
    frontier = contract["frontier_arithmetic"]
    n = frontier["target_sleeve_count"]
    gate = frontier["legacy_global_zero_correlation_reference"]
    quality = frontier["quality_precondition_at_the_gate"]
    ceiling = frontier["book_sharpe_ceiling_at_zero_global_correlation"]

    assert ceiling["s_bar_traded_basis"] == pytest.approx(
        _book_sharpe(quality["s_bar_measured_traded_basis"], gate, n), abs=TOLERANCE
    )
    assert ceiling["s_bar_four_curve_basis"] == pytest.approx(
        _book_sharpe(quality["s_bar_measured_four_curve_basis"], gate, n), abs=TOLERANCE
    )


def test_incremental_gates_do_not_claim_to_establish_the_objective(contract: dict) -> None:
    frontier = contract["frontier_arithmetic"]
    assert frontier["incremental_gates_alone_establish_objective_floor"] is False
    assert frontier["incremental_gates_alone_establish_objective_ceiling"] is False


def test_quality_preconditions_invert_the_identity(contract: dict) -> None:
    frontier = contract["frontier_arithmetic"]
    n = frontier["target_sleeve_count"]
    gate = frontier["legacy_global_zero_correlation_reference"]
    low, high = contract["objective"]["portfolio_sharpe_target"]
    quality = frontier["quality_precondition_at_the_gate"]

    for target in (low, high):
        published = quality[f"s_bar_required_for_{target}"]
        assert _book_sharpe(published, gate, n) == pytest.approx(target, abs=1e-9), (
            f"s_bar of {published} does not actually produce {target} at the gate"
        )


def test_correlation_requirements_invert_the_identity(contract: dict) -> None:
    frontier = contract["frontier_arithmetic"]
    n = frontier["target_sleeve_count"]
    low, high = contract["objective"]["portfolio_sharpe_target"]
    quality = frontier["quality_precondition_at_the_gate"]
    required = frontier["correlation_required_at_measured_quality"]

    bases = {
        "traded_basis": quality["s_bar_measured_traded_basis"],
        "four_curve_basis": quality["s_bar_measured_four_curve_basis"],
    }
    for basis, s_bar in bases.items():
        for target in (low, high):
            rho = required[basis][f"rho_bar_required_for_{target}"]
            assert _book_sharpe(s_bar, rho, n) == pytest.approx(target, abs=1e-9)
            assert rho >= frontier["psd_floor_at_target_n"], (
                f"{basis} requires an average correlation below the PSD floor, which no real "
                "correlation matrix can produce -- that target would be unreachable at any "
                "sleeve count, and the contract must not imply otherwise"
            )
