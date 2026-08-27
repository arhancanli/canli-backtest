"""Fail-closed, asset-agnostic ALPHAC sleeve-admission evidence contract."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, TypeGuard, cast

__all__ = ["AdmissionReport", "evaluate_sleeve_evidence", "load_admission_contract"]

SUPPORTED_SCHEMAS = frozenset(
    {
        "canli.alphac-sleeve-admission-contract.v4",
        "canli.alphac-sleeve-admission-contract.v5-proposed",
        "canli.alphac-sleeve-admission-contract.v6",
        "canli.alphac-sleeve-admission-contract.v7-proposed",
        "canli.alphac-sleeve-admission-contract.v7",
    }
)

# Trading days per year. The one place the Sharpe <-> t-statistic relationship is annualized,
# used both by the contract's own consistency check and by the autocorrelation-inflation gate.
ANNUALIZATION_DAYS = 252.0

# Pooled cross-trial Sharpe variance measured across this project's own ledger. It is the honest
# V[SR] that replaced the ~80x-too-small value one sleeve had been graded against, and it is used
# here only to test whether the contract's own thresholds can coexist.
_POOLED_CROSS_TRIAL_SHARPE_VARIANCE = 7.96e-04

# Gates a contract may or may not declare. Each applies if and only if the contract declares its
# threshold, which keeps the contract the single source of truth for what is enforced: an older
# contract is unaffected, and a newer one cannot declare a gate that quietly does nothing.
# `tests/unit/test_admission_contract_is_fully_enforced.py` pins that no declared threshold in
# any contract goes unread by the evaluator.
OPTIONAL_NUMERIC_GATES: dict[str, tuple[str, str]] = {
    "average_pairwise_correlation_max": (
        "diversification.average_pairwise_correlation",
        "<=",
    ),
    "candidate_average_correlation_to_existing_book_max": (
        "diversification.candidate_average_correlation_to_existing_book",
        "<=",
    ),
    "book_average_pairwise_correlation_delta_max_exclusive": (
        "diversification.book_average_pairwise_correlation_delta",
        "<",
    ),
    "book_deflated_sharpe_min": ("statistics.book_deflated_sharpe", ">="),
    "book_sharpe_delta_lower_95_min_exclusive": ("portfolio.book_sharpe_delta_lower_95", ">"),
    "book_expected_max_drawdown_max": ("portfolio.book_expected_max_drawdown", "<="),
    "covariance_halflife_days_max": ("overlay.covariance_halflife_days", "<="),
    "realized_vol_halflife_days_max": ("overlay.realized_vol_halflife_days", "<="),
    "average_pairwise_correlation_upper_95_max": (
        "diversification.average_pairwise_correlation_upper_95",
        "<=",
    ),
    # Optional since v6: see OPTIONAL_MEASUREMENT_GATES for why the per-sleeve deflated Sharpe is
    # measured and published rather than gated at 0.95.
    "deflated_sharpe_min": ("statistics.deflated_sharpe", ">="),
    # Separated from minimum_oos_observations in v6. A candidate's own sample length is a
    # property of the CANDIDATE; the correlation window is the intersection of the candidate and
    # the BOOK, so tying them together means a book with a short common history can admit
    # nothing, however long the candidate's record. See minimum_correlation_observations in the
    # contract for what actually protects precision here.
    "minimum_correlation_observations": ("diversification.correlation_observations", ">="),
}

# Gates that require a figure to be MEASURED AND PRESENT without imposing a threshold on its
# value. A quantity can be load-bearing evidence without being a pass/fail bar, and pretending
# otherwise is how a bar nobody can clear ends up in a contract.
OPTIONAL_MEASUREMENT_GATES: dict[str, str] = {
    "deflated_sharpe_must_be_measured": "statistics.deflated_sharpe",
    "book_deflated_sharpe_must_be_measured": "statistics.book_deflated_sharpe",
}

# Gates computed from several evidence fields rather than read from one path. They follow the same
# declare-to-enforce rule as the tables above: named here, applied if and only if the contract
# declares the key, and counted in `evidence_checks_per_candidate` by `load_admission_contract`.
OPTIONAL_DERIVED_GATES: frozenset[str] = frozenset({"newey_west_t_ratio_min"})

OPTIONAL_BOOLEAN_GATES: dict[str, str] = {
    "realized_vol_leg_must_be_unlevered": "overlay.realized_vol_leg_is_unlevered",
}

# Checks that are neither lineage, robustness, execution dimensions nor optional gates: the
# always-present numeric gates plus the capacity-curve reconciliation.
_BASE_CHECKS = 18


@dataclass(frozen=True, slots=True)
class AdmissionReport:
    decision: str
    checks_evaluated: int
    failures: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def eligible(self) -> bool:
        return self.decision == "TECHNICALLY_ELIGIBLE"


def _minimum_sharpe_for_deflated(target: float, observations: int) -> float:
    """Smallest annualized Sharpe that can reach `target` DSR, under the MOST favourable search.

    Solved at ``n_trials = 2``, the fewest the deflation formula accepts and therefore the least
    deflation any selection unit can attract. If a candidate sitting on the contract's Sharpe
    floor cannot reach the DSR gate even here, it cannot reach it under any selection unit, and
    the two thresholds are unsatisfiable together regardless of how trials are accounted.

    Zero skew and Gaussian kurtosis are assumed because they are the neutral case; real skew and
    fat tails move the requirement further away, never closer, so this stays a lower bound.
    """
    from alphaforge.validation.dsr import deflated_sharpe_ratio

    low, high = 0.0, 20.0
    for _ in range(90):
        mid = (low + high) / 2.0
        achieved = deflated_sharpe_ratio(
            sr=mid / math.sqrt(ANNUALIZATION_DAYS),
            n_obs=observations,
            n_trials=2,
            sr_trials_variance=_POOLED_CROSS_TRIAL_SHARPE_VARIANCE,
            skew=0.0,
            kurtosis=3.0,
        )
        if achieved < target:
            low = mid
        else:
            high = mid
    return high


def _reject_unsatisfiable_significance_floors(contract: Mapping[str, Any], path: Path) -> None:
    """Refuse a contract whose Sharpe floor and t-statistic floor cannot both be met.

    A t-statistic on a mean return is approximately ``Sharpe * sqrt(years)``, so the three
    thresholds ``net_sharpe_min``, ``newey_west_t_min`` and ``minimum_oos_observations`` are not
    independent: fixing any two determines whether the third is reachable. A candidate sitting
    exactly on the Sharpe floor, measured over exactly the minimum sample the contract demands,
    attains at most ``net_sharpe_min * sqrt(minimum_oos_observations / 252)``. If the t floor sits
    above that value, then the Sharpe floor is not the binding gate and never was -- the t floor
    silently replaces it with a much higher effective Sharpe requirement that nobody chose and
    that no reader of the config can see.

    That is exactly what the v4 contract did. Its floors of Sharpe 0.40, t 2.0 and 252 observations
    require 25 YEARS of out-of-sample data from a candidate sitting at the Sharpe floor; at the
    252-observation minimum the pair demands a standalone Sharpe of 2.0 outright, five times the
    declared floor. The v5 proposal lowered the Sharpe floor to 0.15 while leaving the t floor at
    2.0, which raised the hidden requirement to 178 years. No sleeve in the atlas was ever going to
    pass, and the config gave no hint of it.

    This is a load-time structural check rather than a per-candidate gate because it is a property
    of the CONTRACT, not of any evidence: a contract that cannot be satisfied is not a strict
    contract, it is a broken one, and it should fail before it is ever used to judge anything.
    """
    thresholds = contract["thresholds"]
    sharpe_floor = thresholds["net_sharpe_min"]
    t_floor = thresholds["newey_west_t_min"]
    minimum_observations = thresholds["minimum_oos_observations"]
    if not all(
        _finite_number(value) for value in (sharpe_floor, t_floor, minimum_observations)
    ):
        raise ValueError(f"non-numeric significance floors in {path}")
    if minimum_observations <= 0:
        raise ValueError(f"minimum_oos_observations must be positive in {path}")
    years = float(minimum_observations) / ANNUALIZATION_DAYS
    attainable_t = float(sharpe_floor) * math.sqrt(years)
    # Collect EVERY violated pair rather than raising on the first. Both superseded contracts
    # violate both pairs, and a report that names one of them invites fixing that one and
    # rediscovering the other.
    problems: list[str] = []

    if float(t_floor) > attainable_t:
        required_years = (float(t_floor) / float(sharpe_floor)) ** 2 if sharpe_floor > 0 else None
        detail = (
            f"{required_years:.1f} years" if required_years is not None else "an infinite sample"
        )
        problems.append(
            f"newey_west_t_min={t_floor} exceeds the {attainable_t:.3f} a candidate at "
            f"net_sharpe_min={sharpe_floor} attains over {minimum_observations} observations "
            f"({years:.2f} years); it would need {detail} of out-of-sample data, so the t floor "
            "and not the declared Sharpe floor is the real gate. Lower newey_west_t_min, raise "
            "net_sharpe_min, or raise minimum_oos_observations until the three agree."
        )

    deflated_floor = thresholds.get("deflated_sharpe_min")
    if deflated_floor is not None and _finite_number(deflated_floor) and sharpe_floor > 0:
        required = _minimum_sharpe_for_deflated(float(deflated_floor), int(minimum_observations))
        if required > float(sharpe_floor):
            problems.append(
                f"deflated_sharpe_min={deflated_floor} needs an annualized Sharpe of at least "
                f"{required:.3f} over {minimum_observations} observations even at the most "
                f"favourable possible trial count, but net_sharpe_min is only {sharpe_floor} -- "
                f"about {required / float(sharpe_floor):.0f}x lower. Raise net_sharpe_min, lower "
                "deflated_sharpe_min, or stop gating the per-sleeve deflated Sharpe and gate the "
                "book's instead: the published claim is the book's."
            )

    if problems:
        joined = "".join(f"\n  - {problem}" for problem in problems)
        raise ValueError(f"self-contradictory significance floors in {path}:{joined}")


def load_admission_contract(path: Path) -> dict[str, Any]:
    contract = cast(dict[str, Any], json.loads(path.read_text()))
    if contract.get("schema") not in SUPPORTED_SCHEMAS:
        raise ValueError(f"unsupported sleeve admission contract: {path}")
    if contract.get("objective", {}).get("targets_are_admission_evidence") is not False:
        raise ValueError("portfolio targets must never count as sleeve-admission evidence")
    _reject_unsatisfiable_significance_floors(contract, path)
    # Whether every declared threshold is actually read is pinned by
    # tests/unit/test_admission_contract_is_fully_enforced.py, which records real fetches during
    # an evaluation rather than comparing against a hand-maintained list here that would drift.
    declared = set(contract["thresholds"])
    optional_declared = len(
        declared
        & (
            set(OPTIONAL_NUMERIC_GATES)
            | set(OPTIONAL_BOOLEAN_GATES)
            | set(OPTIONAL_MEASUREMENT_GATES)
            | OPTIONAL_DERIVED_GATES
        )
    )
    expected_checks = (
        len(contract["required_lineage"])
        + len(contract["required_robustness"])
        + len(contract["execution_dimensions"])
        + _BASE_CHECKS
        + optional_declared
    )
    if contract.get("evidence_checks_per_candidate") != expected_checks:
        raise ValueError(
            "stale evidence_checks_per_candidate: "
            f"expected {expected_checks}, got {contract.get('evidence_checks_per_candidate')}"
        )
    return contract


def _nested(evidence: Mapping[str, Any], path: str) -> Any:
    value: Any = evidence
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _finite_number(value: Any) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", value) is not None


def _git_commit(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{7,64}", value) is not None


def _canonical_sha256(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _matches_canonical_sha256(digest: Any, payload: Any) -> bool:
    if not _sha256(digest):
        return False
    try:
        expected = _canonical_sha256(payload)
    except (TypeError, ValueError):
        return False
    return str(digest).removeprefix("sha256:") == expected


def evaluate_sleeve_evidence(
    evidence: Mapping[str, Any], contract: Mapping[str, Any]
) -> AdmissionReport:
    """Evaluate one sealed result; missing or failed evidence always fails closed."""
    failures: list[str] = []
    warnings: list[str] = []
    checks = 0

    lineage_types = contract["lineage_types"]
    sha_fields = set(lineage_types["sha256_fields"])
    identifier_fields = set(lineage_types["identifier_fields"])
    boolean_fields = set(lineage_types["boolean_fields"])
    code_commit_field = lineage_types["code_commit_field"]
    for field in contract["required_lineage"]:
        checks += 1
        value = _nested(evidence, f"lineage.{field}")
        if field in sha_fields and not _sha256(value):
            failures.append(f"invalid_sha256:lineage.{field}")
        elif field == code_commit_field and not _git_commit(value):
            failures.append(f"invalid_git_commit:lineage.{field}")
        elif field in identifier_fields and (not isinstance(value, str) or len(value.strip()) < 3):
            failures.append(f"invalid_identifier:lineage.{field}")
        elif field in boolean_fields and value is not True:
            failures.append(f"missing_or_false:lineage.{field}")

    for field in contract["required_robustness"]:
        checks += 1
        if _nested(evidence, f"robustness.{field}") is not True:
            failures.append(f"not_passed:robustness.{field}")

    thresholds = contract["thresholds"]
    numeric_gates = {
        "statistics.oos_observations": (">=", thresholds["minimum_oos_observations"]),
        "statistics.stressed_oos_observations": (
            ">=",
            thresholds["minimum_stressed_oos_observations"],
        ),
        "statistics.net_sharpe": (">=", thresholds["net_sharpe_min"]),
        "statistics.stressed_sharpe": (">=", thresholds["stressed_sharpe_min"]),
        "statistics.newey_west_t": (">=", thresholds["newey_west_t_min"]),
        "diversification.absolute_beta": ("<=", thresholds["absolute_beta_max"]),
        "diversification.max_pairwise_correlation": (
            "<=",
            thresholds["ordinary_pairwise_correlation_max"],
        ),
        "diversification.max_stressed_pairwise_correlation": (
            "<=",
            thresholds["stressed_pairwise_correlation_max"],
        ),
        "diversification.max_pairwise_correlation_upper_95": (
            "<=",
            thresholds["pairwise_correlation_upper_95_max"],
        ),
        "diversification.max_stressed_pairwise_correlation_upper_95": (
            "<=",
            thresholds["stressed_pairwise_correlation_upper_95_max"],
        ),
        "portfolio.book_sharpe_delta": (
            ">",
            thresholds["book_sharpe_delta_min_exclusive"],
        ),
        "portfolio.minimum_leave_one_period_out_book_sharpe_delta": (
            ">",
            thresholds["minimum_leave_one_period_out_book_sharpe_delta_exclusive"],
        ),
        "portfolio.book_max_drawdown_delta": (
            "<=",
            thresholds["book_max_drawdown_delta_max"],
        ),
        "portfolio.book_expected_shortfall_delta": (
            "<=",
            thresholds["book_expected_shortfall_delta_max"],
        ),
        "capacity.capacity_usd": (">=", thresholds["capacity_usd_min"]),
        "capacity.minimum_stressed_fill_ratio": (
            ">=",
            thresholds["capacity_minimum_stressed_fill_ratio"],
        ),
    }
    # Declare-to-enforce: see OPTIONAL_NUMERIC_GATES at module level.
    for key, (path, operator) in OPTIONAL_NUMERIC_GATES.items():
        if key not in thresholds:
            continue
        numeric_gates[path] = (operator, thresholds[key])

    for path, (operator, threshold) in numeric_gates.items():
        checks += 1
        value = _nested(evidence, path)
        if not _finite_number(value):
            failures.append(f"missing_numeric:{path}")
        elif operator == ">=" and value < threshold:
            failures.append(f"threshold:{path}:{value}<{threshold}")
        elif operator == "<=" and value > threshold:
            failures.append(f"threshold:{path}:{value}>{threshold}")
        elif operator == ">" and value <= threshold:
            failures.append(f"threshold:{path}:{value}<={threshold}")
        elif operator == "<" and value >= threshold:
            failures.append(f"threshold:{path}:{value}>={threshold}")

    # Declare-to-enforce: see OPTIONAL_DERIVED_GATES at module level.
    #
    # WHAT THIS GATE IS FOR, because it is not another significance bar. A Newey-West t corrects
    # for autocorrelation in the return stream. A sleeve priced off stale or appraised marks --
    # catastrophe bonds, municipal basis, freight derivatives, anything thinly traded, which is
    # precisely the corner of the atlas where the anti-correlated sources live -- shows an
    # inflated naive Sharpe because its measured volatility is smoothed. The tell is not a low t;
    # it is a t far BELOW the one its own reported Sharpe implies. Gating that ratio catches
    # smoothed pricing at any Sharpe level, which a flat t floor cannot do: a flat floor waves
    # through a stale-priced sleeve with a high headline Sharpe and rejects an honest one with a
    # modest Sharpe, getting both cases backwards.
    #
    # Uses only evidence the contract already requires, so it adds a check and no new burden.
    if "newey_west_t_ratio_min" in thresholds:
        checks += 1
        ratio_floor = thresholds["newey_west_t_ratio_min"]
        newey_west_t = _nested(evidence, "statistics.newey_west_t")
        net_sharpe = _nested(evidence, "statistics.net_sharpe")
        observations = _nested(evidence, "statistics.oos_observations")
        if not all(
            _finite_number(value) for value in (newey_west_t, net_sharpe, observations)
        ):
            failures.append("missing_numeric:statistics.newey_west_t_ratio")
        elif net_sharpe <= 0 or observations <= 0:
            # A non-positive Sharpe has no meaningful implied t to compare against; the sleeve is
            # already rejected by net_sharpe_min, and dividing here would invent a verdict.
            failures.append("undefined:statistics.newey_west_t_ratio")
        else:
            implied_t = float(net_sharpe) * math.sqrt(float(observations) / ANNUALIZATION_DAYS)
            ratio = float(newey_west_t) / implied_t
            if ratio < ratio_floor:
                failures.append(
                    f"threshold:statistics.newey_west_t_ratio:{ratio:.4f}<{ratio_floor}"
                )

    # Measurement gates: the figure must be present and finite. Absent fails closed, exactly as a
    # threshold gate would, because unmeasured is not the same as unremarkable.
    for key, path in OPTIONAL_MEASUREMENT_GATES.items():
        if key not in thresholds:
            continue
        checks += 1
        if thresholds[key] is not True:
            continue
        if not _finite_number(_nested(evidence, path)):
            failures.append(f"missing_numeric:{path}")

    # Boolean gates follow the same declare-to-enforce rule. `is True` rather than truthiness:
    # a placeholder string or a 1 must not satisfy a safety flag.
    for key, path in OPTIONAL_BOOLEAN_GATES.items():
        if key not in thresholds:
            continue
        checks += 1
        if thresholds[key] is not True:
            continue
        if _nested(evidence, path) is not True:
            failures.append(f"missing_or_false:{path}")

    checks += 1
    curve = _nested(evidence, "capacity.curve")
    if not isinstance(curve, list) or len(curve) < thresholds["capacity_curve_min_points"]:
        failures.append("capacity_curve_too_short")
    else:
        required_curve_fields = set(contract["capacity_curve_required_fields"])
        capitals: list[float] = []
        sharpes: list[float] = []
        fill_ratios: list[float] = []
        stressed_costs: list[float] = []
        qualifying_capitals: list[float] = []
        curve_invalid = False
        for index, point in enumerate(curve):
            if not isinstance(point, Mapping) or not required_curve_fields.issubset(point):
                failures.append(f"capacity_curve_invalid_point:{index}")
                curve_invalid = True
                continue
            capital = point["capital_usd"]
            sharpe = point["net_sharpe"]
            fill_ratio = point["fill_ratio"]
            stressed_cost = point["stressed_cost_bps"]
            if not all(_finite_number(v) for v in (capital, sharpe, fill_ratio, stressed_cost)):
                failures.append(f"capacity_curve_nonfinite_point:{index}")
                curve_invalid = True
                continue
            if capital <= 0 or not 0 <= fill_ratio <= 1 or stressed_cost < 0:
                failures.append(f"capacity_curve_out_of_range_point:{index}")
                curve_invalid = True
                continue
            capitals.append(float(capital))
            sharpes.append(float(sharpe))
            fill_ratios.append(float(fill_ratio))
            stressed_costs.append(float(stressed_cost))
            if (
                sharpe >= thresholds["net_sharpe_min"]
                and fill_ratio >= thresholds["capacity_minimum_stressed_fill_ratio"]
            ):
                qualifying_capitals.append(float(capital))
        if capitals != sorted(set(capitals)):
            failures.append("capacity_curve_capital_not_strictly_increasing")
            curve_invalid = True
        if any(right > left for left, right in pairwise(fill_ratios)):
            failures.append("capacity_curve_fill_ratio_improves_with_capital")
            curve_invalid = True
        if any(right < left for left, right in pairwise(stressed_costs)):
            failures.append("capacity_curve_cost_falls_with_capital")
            curve_invalid = True
        if any(right > left for left, right in pairwise(sharpes)):
            failures.append("capacity_curve_sharpe_improves_with_capital")
            curve_invalid = True
        reported_capacity = _nested(evidence, "capacity.capacity_usd")
        if not curve_invalid and (
            not qualifying_capitals
            or not _finite_number(reported_capacity)
            or float(reported_capacity) != max(qualifying_capitals)
        ):
            failures.append("capacity_usd_not_reconciled_to_curve")
        reported_minimum_fill = _nested(evidence, "capacity.minimum_stressed_fill_ratio")
        if not curve_invalid and _finite_number(reported_capacity):
            fills_through_capacity = [
                fill
                for capital, fill in zip(capitals, fill_ratios, strict=False)
                if capital <= float(reported_capacity)
            ]
            if (
                not fills_through_capacity
                or not _finite_number(reported_minimum_fill)
                or not math.isclose(
                    float(reported_minimum_fill),
                    min(fills_through_capacity),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                failures.append("minimum_stressed_fill_ratio_not_reconciled_to_curve")

    checks += 1
    pbo = _nested(evidence, "statistics.pbo")
    if not isinstance(pbo, Mapping):
        failures.append("missing:statistics.pbo")
    elif pbo.get("status") == contract["pbo_policy"]["measured_status"]:
        value = pbo.get("value")
        if not _finite_number(value) or not 0.0 <= value <= thresholds["pbo_max"]:
            failures.append("pbo_missing_or_above_gate")
    elif pbo.get("status") == contract["pbo_policy"]["single_identity_status"]:
        if pbo.get("value") is not None or pbo.get("family_variants") != 1:
            failures.append("invalid_single_identity_pbo")
        else:
            warnings.append("PBO is undefined for one identity and is not represented as zero")
    else:
        failures.append("invalid_pbo_status")

    execution = _nested(evidence, "execution")
    allowed = set(contract["execution_statuses"])
    execution_policy = contract["execution_evidence_policy"]
    required_scenario_fields = set(execution_policy["scenario_manifest_required_fields"])
    scenario_statuses = set(execution_policy["scenario_statuses"])
    for dimension in contract["execution_dimensions"]:
        checks += 1
        item = execution.get(dimension) if isinstance(execution, Mapping) else None
        if not isinstance(item, Mapping) or item.get("status") not in allowed:
            failures.append(f"missing_execution_dimension:{dimension}")
            continue
        if item["status"] in {"TESTED_PASS", "TESTED_FAIL"}:
            if item["status"] == "TESTED_FAIL":
                failures.append(f"execution_failed:{dimension}")
            if not _sha256(item.get("evidence_sha256")):
                failures.append(f"execution_test_without_hash:{dimension}")
            scenario_count = item.get("scenarios")
            manifest = item.get("scenario_manifest")
            if (
                not isinstance(scenario_count, int)
                or isinstance(scenario_count, bool)
                or scenario_count < thresholds["minimum_execution_scenarios_per_dimension"]
            ):
                failures.append(f"execution_test_without_scenarios:{dimension}")
            if not isinstance(manifest, list):
                failures.append(f"execution_test_without_manifest:{dimension}")
                continue
            if (
                not isinstance(scenario_count, int)
                or isinstance(scenario_count, bool)
                or (scenario_count != len(manifest))
            ):
                failures.append(f"execution_scenario_count_mismatch:{dimension}")
            scenario_ids: list[str] = []
            manifest_statuses: list[str] = []
            for index, scenario in enumerate(manifest):
                if not isinstance(scenario, Mapping) or not required_scenario_fields.issubset(
                    scenario
                ):
                    failures.append(f"execution_manifest_invalid_scenario:{dimension}:{index}")
                    continue
                scenario_id = scenario["scenario_id"]
                scenario_status = scenario["status"]
                if not isinstance(scenario_id, str) or len(scenario_id.strip()) < 3:
                    failures.append(f"execution_manifest_invalid_id:{dimension}:{index}")
                else:
                    scenario_ids.append(scenario_id)
                if scenario_status not in scenario_statuses:
                    failures.append(f"execution_manifest_invalid_status:{dimension}:{index}")
                else:
                    manifest_statuses.append(str(scenario_status))
                assumptions = scenario["assumptions"]
                result = scenario["result"]
                if not isinstance(assumptions, Mapping) or not assumptions:
                    failures.append(f"execution_manifest_invalid_assumptions:{dimension}:{index}")
                elif not _matches_canonical_sha256(scenario["assumptions_sha256"], assumptions):
                    failures.append(
                        f"execution_manifest_assumptions_hash_mismatch:{dimension}:{index}"
                    )
                if not isinstance(result, Mapping) or not result:
                    failures.append(f"execution_manifest_invalid_result:{dimension}:{index}")
                elif not _matches_canonical_sha256(scenario["result_sha256"], result):
                    failures.append(f"execution_manifest_result_hash_mismatch:{dimension}:{index}")
                passed = result.get("passed") if isinstance(result, Mapping) else None
                if not isinstance(passed, bool):
                    failures.append(f"execution_manifest_result_missing_passed:{dimension}:{index}")
                elif (scenario_status == "PASS") != passed:
                    failures.append(
                        f"execution_manifest_status_result_mismatch:{dimension}:{index}"
                    )
            if len(scenario_ids) != len(set(scenario_ids)):
                failures.append(f"execution_manifest_duplicate_ids:{dimension}")
            evidence_hash = str(item.get("evidence_sha256", "")).removeprefix("sha256:")
            if evidence_hash != _canonical_sha256(manifest):
                failures.append(f"execution_manifest_hash_mismatch:{dimension}")
            if item["status"] == "TESTED_PASS" and any(
                status != "PASS" for status in manifest_statuses
            ):
                failures.append(f"execution_pass_contains_failed_scenario:{dimension}")
            if item["status"] == "TESTED_FAIL" and not any(
                status == "FAIL" for status in manifest_statuses
            ):
                failures.append(f"execution_fail_without_failed_scenario:{dimension}")
        elif item["status"] == "NOT_APPLICABLE":
            reason = item.get("reason")
            if not isinstance(reason, str) or len(reason.strip()) < 12:
                failures.append(f"execution_na_without_reason:{dimension}")
            if not _sha256(item.get("applicability_evidence_sha256")):
                failures.append(f"execution_na_without_hash:{dimension}")

    decision = "TECHNICALLY_ELIGIBLE" if not failures else "REJECT_OR_REMEDIATE"
    return AdmissionReport(decision, checks, tuple(failures), tuple(warnings))
