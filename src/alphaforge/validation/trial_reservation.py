"""Pre-result evidence reservation required before a new return hypothesis is logged."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Final

from alphaforge.validation.experiments import ExperimentLog, hypothesis_hash

SCHEMA: Final[str] = "canli.alphac-forward-trial-reservation.v1"
STATUS: Final[str] = "RETURN_IDENTITY_RESERVED"
IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9_]{2,95}$")
SHA256: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_OUTCOME_KEYS: Final[frozenset[str]] = frozenset(
    {
        "admission",
        "capacity",
        "drawdown",
        "dsr",
        "max_drawdown",
        "metrics",
        "pnl",
        "psr",
        "result",
        "returns",
        "sharpe",
        "stress",
        "verdict",
    }
)
REQUIRED_EVIDENCE: Final[frozenset[str]] = frozenset(
    {"preregistration", "input_data_manifest", "runner", "python_project", "locked_environment"}
)
PACKET_MANIFEST: Final[Path] = Path("artifacts/research/trial_packet_manifest.json")
LEGACY_EPOCH_CLOSURE: Final[Path] = Path("artifacts/research/legacy_research_epoch_closure.json")
LEGACY_EPOCH_SCHEMA: Final[str] = "canli.alphac-legacy-research-epoch-closure.v1"
LEGACY_EPOCH_STATUS: Final[str] = "LEGACY_EPOCH_RETIRED_FAIL_CLOSED"
IDENTITY_PACKET_DIR: Final[Path] = Path("artifacts/research/trial_packets")
ACTIVE_ADMISSION_CONTRACT: Final[Path] = Path("config/sleeve_admission_contract.json")
ACTIVE_TRIAL_POLICY: Final[Path] = Path("config/trial_accounting.json")
ADMISSION_PROMOTION_RECEIPT: Final[Path] = Path("config/admission_v7_promotion.json")


class ReservationError(ValueError):
    """The proposed forward identity is not safely reserved."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_file(repo: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ReservationError("evidence path must be a non-empty repository-relative string")
    candidate = (repo / relative).resolve()
    try:
        candidate.relative_to(repo.resolve())
    except ValueError as error:
        raise ReservationError(f"evidence path escapes repository: {relative}") from error
    if not candidate.is_file():
        raise ReservationError(f"reserved evidence is missing: {relative}")
    return candidate


def _parse_utc(value: object) -> dt.datetime:
    if not isinstance(value, str):
        raise ReservationError("reserved_at must be an ISO-8601 timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ReservationError("reserved_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ReservationError("reserved_at must include UTC timezone information")
    return parsed


def _observed_content_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "content_hash"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _validate_legacy_epoch_closure(
    repo: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    closure_path = repo / LEGACY_EPOCH_CLOSURE
    if not closure_path.is_file():
        raise ReservationError(
            "new return identity blocked: historical trial-packet coverage is incomplete and "
            "the fail-closed legacy epoch closure is missing"
        )
    try:
        closure = json.loads(closure_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ReservationError(
            "new return identity blocked: fail-closed legacy epoch closure is unreadable"
        ) from error
    if closure.get("schema") != LEGACY_EPOCH_SCHEMA or closure.get("status") != LEGACY_EPOCH_STATUS:
        raise ReservationError("new return identity blocked: legacy epoch closure schema mismatch")
    if closure.get("content_hash") != _observed_content_hash(closure):
        raise ReservationError(
            "new return identity blocked: legacy epoch closure content hash mismatch"
        )

    bindings = closure.get("source_bindings")
    required_bindings = {
        "trial_packet_manifest",
        "identity_packet_index",
        "recoverability_audit",
        "historical_curve_index",
    }
    if not isinstance(bindings, dict) or set(bindings) != required_bindings:
        raise ReservationError("new return identity blocked: legacy source bindings are incomplete")
    for name in sorted(required_bindings):
        binding = bindings[name]
        if not isinstance(binding, dict) or set(binding) != {
            "path",
            "sha256",
            "content_hash",
        }:
            raise ReservationError(
                f"new return identity blocked: legacy {name} binding is malformed"
            )
        source = _repo_file(repo, binding["path"])
        if _sha256(source) != binding["sha256"]:
            raise ReservationError(f"new return identity blocked: legacy {name} file hash mismatch")
        try:
            source_payload = json.loads(source.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise ReservationError(
                f"new return identity blocked: legacy {name} source is unreadable"
            ) from error
        if source_payload.get("content_hash") != binding["content_hash"] or binding[
            "content_hash"
        ] != _observed_content_hash(source_payload):
            raise ReservationError(
                f"new return identity blocked: legacy {name} content hash mismatch"
            )
    manifest_binding = bindings["trial_packet_manifest"]
    if (
        Path(manifest_binding["path"]) != PACKET_MANIFEST
        or manifest_binding["sha256"] != _sha256(manifest_path)
        or manifest_binding["content_hash"] != manifest["content_hash"]
    ):
        raise ReservationError(
            "new return identity blocked: closure does not bind the current packet manifest"
        )

    manifest_summary = manifest["summary"]
    identity_count = manifest_summary.get("distinct_hypothesis_identities")
    complete = manifest_summary.get("complete_trial_packets")
    incomplete = manifest_summary.get("incomplete_trial_packets")
    published = manifest_summary.get("published_identity_packets")
    if (
        not isinstance(identity_count, int)
        or identity_count < 0
        or published != identity_count
        or not isinstance(complete, int)
        or not isinstance(incomplete, int)
        or complete + incomplete != identity_count
        or manifest_summary.get("incomplete_not_yet_audited") != 0
    ):
        raise ReservationError(
            "new return identity blocked: legacy epoch does not account for every identity"
        )
    closure_summary = closure.get("summary")
    if closure_summary != {
        "retired_identities": identity_count,
        "retired_complete_evidenced_kills": complete,
        "retired_incomplete_evidence_debt": incomplete,
        "retired_unassessed_identities": 0,
        "eligible_for_admission": 0,
        "identity_reuse_permitted": 0,
    }:
        raise ReservationError("new return identity blocked: legacy retirement counts mismatch")

    manifest_identities = {item["hypothesis_key"]: item for item in manifest.get("identities", [])}
    closure_identities = closure.get("identities")
    if (
        len(manifest_identities) != identity_count
        or not isinstance(closure_identities, list)
        or len(closure_identities) != identity_count
    ):
        raise ReservationError("new return identity blocked: legacy identity inventory mismatch")
    seen: set[str] = set()
    for item in closure_identities:
        identity = item.get("hypothesis_key") if isinstance(item, dict) else None
        manifest_item = manifest_identities.get(identity)
        complete_identity = (
            manifest_item is not None and manifest_item.get("coverage_status") == "COMPLETE"
        )
        expected_disposition = (
            "RETIRED_COMPLETE_EVIDENCED_KILL"
            if complete_identity
            else "RETIRED_INCOMPLETE_EVIDENCE_DEBT"
        )
        if (
            not isinstance(identity, str)
            or identity in seen
            or manifest_item is None
            or item.get("config_hash") != manifest_item.get("config_hash")
            or item.get("packet_content_hash") != manifest_item.get("identity_packet_content_hash")
            or item.get("packet_complete") is not complete_identity
            or item.get("disposition") != expected_disposition
            or item.get("eligible_for_admission") is not False
            or item.get("identity_reuse_permitted") is not False
        ):
            raise ReservationError(
                "new return identity blocked: a legacy identity is not fail-closed"
            )
        seen.add(identity)
    policy = closure.get("forward_epoch_policy")
    if (
        not isinstance(policy, dict)
        or not policy
        or any(value is not True for value in policy.values())
    ):
        raise ReservationError("new return identity blocked: forward epoch policy is incomplete")
    return {
        "coverage_mode": "FAIL_CLOSED_LEGACY_EPOCH_RETIREMENT",
        "manifest_path": str(PACKET_MANIFEST),
        "manifest_content_hash": manifest["content_hash"],
        "legacy_epoch_closure_path": str(LEGACY_EPOCH_CLOSURE),
        "legacy_epoch_closure_content_hash": closure["content_hash"],
        "historical_identities": identity_count,
        "complete_trial_packets": complete,
        "retired_incomplete_trial_packets": incomplete,
        "historical_identities_eligible_for_admission": 0,
    }


def _validate_historical_packet_coverage(repo: Path) -> dict[str, Any]:
    path = repo / PACKET_MANIFEST
    if not path.is_file():
        raise ReservationError(
            "new return identity blocked: canonical trial-packet manifest is missing"
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ReservationError(
            "new return identity blocked: canonical trial-packet manifest is unreadable"
        ) from error
    if manifest.get("schema") != "canli.alphac-trial-packet-manifest.v2":
        raise ReservationError("new return identity blocked: trial-packet manifest schema mismatch")
    expected_content_hash = manifest.get("content_hash")
    observed_content_hash = _observed_content_hash(manifest)
    if expected_content_hash != observed_content_hash:
        raise ReservationError(
            "new return identity blocked: trial-packet manifest content hash mismatch"
        )
    summary = manifest.get("summary")
    if not isinstance(summary, dict):
        raise ReservationError("new return identity blocked: trial-packet summary is missing")
    identities = summary.get("distinct_hypothesis_identities")
    complete = summary.get("complete_trial_packets")
    incomplete = summary.get("incomplete_trial_packets")
    published = summary.get("published_identity_packets")
    coverage_complete = (
        isinstance(identities, int)
        and identities >= 0
        and complete == identities
        and published == identities
        and incomplete == 0
        and summary.get("coverage_status") == "COMPLETE"
    )
    if not coverage_complete:
        return _validate_legacy_epoch_closure(repo, manifest, path)
    return {
        "coverage_mode": "ALL_HISTORICAL_PACKETS_COMPLETE",
        "manifest_path": str(PACKET_MANIFEST),
        "manifest_content_hash": observed_content_hash,
        "historical_identities": identities,
        "complete_trial_packets": complete,
    }


def _validate_forward_epoch_serial_completion(
    repo: Path,
    *,
    reserved_hypothesis_identity: str,
) -> dict[str, Any]:
    closure_path = repo / LEGACY_EPOCH_CLOSURE
    legacy_keys: set[str] = set()
    if closure_path.is_file():
        closure = json.loads(closure_path.read_text(encoding="utf-8"))
        legacy_keys = {
            item["hypothesis_key"]
            for item in closure.get("identities", [])
            if isinstance(item, dict) and isinstance(item.get("hypothesis_key"), str)
        }

    ledger_paths = sorted(
        {
            *repo.glob("var*/experiments.jsonl"),
            *repo.glob("artifacts/**/experiments.jsonl"),
        }
    )
    first_records: dict[str, tuple[int, str]] = {}
    for ledger_path in ledger_paths:
        if any("archive" in part.casefold() for part in ledger_path.relative_to(repo).parts):
            continue
        ledger = ExperimentLog(ledger_path)
        for record in ledger.all():
            key = ledger._hypothesis_key(record.config)
            ordering = (record.now_ms, record.config_hash)
            if key not in first_records or ordering < first_records[key]:
                first_records[key] = ordering
    ordered_forward_keys = [
        key
        for key, _ in sorted(first_records.items(), key=lambda item: (*item[1], item[0]))
        if key not in legacy_keys
    ]
    if reserved_hypothesis_identity in ordered_forward_keys:
        target_index = ordered_forward_keys.index(reserved_hypothesis_identity)
        forward_keys = ordered_forward_keys[:target_index]
    else:
        forward_keys = ordered_forward_keys
    verified_packets: list[dict[str, str]] = []
    for identity in forward_keys:
        path = repo / IDENTITY_PACKET_DIR / f"{identity}.json"
        if not path.is_file():
            raise ReservationError(
                "new return identity blocked: prior forward identity has no complete packet: "
                f"{identity}"
            )
        try:
            packet = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise ReservationError(
                f"new return identity blocked: prior forward packet is unreadable: {identity}"
            ) from error
        if (
            packet.get("schema") != "canli.alphac-identity-trial-packet.v2"
            or packet.get("hypothesis_key") != identity
            or packet.get("complete") is not True
            or packet.get("missing_sections") != []
            or packet.get("content_hash") != _observed_content_hash(packet)
        ):
            raise ReservationError(
                "new return identity blocked: prior forward packet is incomplete or invalid: "
                f"{identity}"
            )
        verified_packets.append(
            {
                "hypothesis_key": identity,
                "packet_path": str(IDENTITY_PACKET_DIR / f"{identity}.json"),
                "packet_content_hash": packet["content_hash"],
            }
        )
    return {
        "policy": "SERIAL_COMPLETE_PACKET_BEFORE_NEXT_FORWARD_IDENTITY",
        "forward_identities_already_logged": len(forward_keys),
        "complete_forward_packets_verified": len(verified_packets),
        "verified_packets": verified_packets,
    }


def _effective_contract_hash(contract: dict[str, Any]) -> str:
    normalized = json.loads(json.dumps(contract))
    normalized["prospective_scope"]["effective_contract_content_hash"] = None
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _validate_governance_epoch(
    payload: dict[str, Any],
    *,
    repo: Path,
    historical_identities: int,
    forward_identities_already_logged: int,
) -> dict[str, Any]:
    """Bind a new identity to the exact in-force gate and staged trial-budget epoch."""
    governance = payload.get("governance_epoch")
    required = {
        "admission_contract_path": ACTIVE_ADMISSION_CONTRACT.as_posix(),
        "trial_policy_path": ACTIVE_TRIAL_POLICY.as_posix(),
        "promotion_receipt_path": ADMISSION_PROMOTION_RECEIPT.as_posix(),
    }
    if not isinstance(governance, dict):
        raise ReservationError("governance_epoch must bind the active v7 contract and trial policy")
    expected_keys = {
        *required,
        "admission_contract_sha256",
        "trial_policy_sha256",
        "promotion_receipt_sha256",
        "effective_contract_hash",
        "reservation_ordinal",
    }
    if set(governance) != expected_keys:
        raise ReservationError("governance_epoch fields are incomplete or unexpected")
    for key, expected_path in required.items():
        if governance.get(key) != expected_path:
            raise ReservationError(f"governance_epoch must use canonical {key}")

    paths = {
        "admission_contract": _repo_file(repo, governance["admission_contract_path"]),
        "trial_policy": _repo_file(repo, governance["trial_policy_path"]),
        "promotion_receipt": _repo_file(repo, governance["promotion_receipt_path"]),
    }
    for name, path in paths.items():
        claimed = governance.get(f"{name}_sha256")
        if not isinstance(claimed, str) or SHA256.fullmatch(claimed) is None:
            raise ReservationError(f"governance_epoch {name}_sha256 is invalid")
        if _sha256(path) != claimed:
            raise ReservationError(f"governance_epoch {name} hash mismatch")

    contract = json.loads(paths["admission_contract"].read_text())
    policy = json.loads(paths["trial_policy"].read_text())
    receipt = json.loads(paths["promotion_receipt"].read_text())
    if (
        contract.get("schema") != "canli.alphac-sleeve-admission-contract.v7"
        or contract.get("status") != "IN_FORCE"
        or contract.get("prospective_scope", {}).get("effective") is not True
    ):
        raise ReservationError("new return identity requires the in-force v7 admission contract")
    effective_hash = contract["prospective_scope"].get("effective_contract_content_hash")
    if effective_hash != _effective_contract_hash(contract):
        raise ReservationError("active admission contract effective hash mismatch")
    if governance.get("effective_contract_hash") != effective_hash:
        raise ReservationError("reservation does not bind the active effective contract hash")
    receipt_body = {key: value for key, value in receipt.items() if key != "content_hash"}
    if (
        receipt.get("schema") != "canli.alphac-admission-v7-promotion.v1"
        or receipt.get("content_hash") != _observed_content_hash(receipt_body)
        or receipt.get("active_contract") != contract
        or receipt.get("effective_contract_hash") != effective_hash
    ):
        raise ReservationError(
            "v7 promotion receipt is invalid or does not bind the active contract"
        )
    if (
        policy.get("schema") != "alphac.trial-accounting-policy.v2"
        or policy.get("research_status") != "ACTIVE_STAGED_PROSPECTIVE_BUDGET"
        or policy.get("prospective_v7_review", {}).get("admission_v7_effective_contract_hash")
        != effective_hash
    ):
        raise ReservationError("active trial policy does not bind the v7 contract")

    ordinal = governance.get("reservation_ordinal")
    expected_ordinal = historical_identities + forward_identities_already_logged + 1
    first_effective = contract["prospective_scope"]["effective_on_or_after_reservation_ordinal"]
    budget = policy.get("hypothesis_identity_budget")
    if ordinal != expected_ordinal or ordinal < first_effective:
        raise ReservationError(
            f"reservation_ordinal must be the next governed identity: {expected_ordinal}"
        )
    if not isinstance(budget, int) or ordinal > budget:
        raise ReservationError("staged hypothesis-identity budget is exhausted")
    return {
        "admission_contract_schema": contract["schema"],
        "admission_contract_sha256": _sha256(paths["admission_contract"]),
        "effective_contract_hash": effective_hash,
        "trial_policy_schema": policy["schema"],
        "trial_policy_sha256": _sha256(paths["trial_policy"]),
        "hypothesis_identity_budget": budget,
        "staged_hard_reviews": policy["prospective_v7_review"]["staged_hard_reviews"],
        "promotion_receipt_sha256": _sha256(paths["promotion_receipt"]),
        "reservation_ordinal": ordinal,
    }


def validate_reservation(
    payload: dict[str, Any],
    *,
    trial_config: dict[str, Any],
    repo: Path,
) -> dict[str, Any]:
    """Validate one pre-result reservation and return its canonical audit summary."""
    if payload.get("schema") != SCHEMA or payload.get("status") != STATUS:
        raise ReservationError(f"reservation must declare {SCHEMA} / {STATUS}")
    for key in ("family_trial_account", "return_identity_id"):
        value = payload.get(key)
        if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
            raise ReservationError(f"{key} must be a stable snake-case identifier")
    if payload.get("hypotheses_spent") != 1:
        raise ReservationError("a forward reservation must spend exactly one hypothesis")
    _parse_utc(payload.get("reserved_at"))

    forbidden = FORBIDDEN_OUTCOME_KEYS.intersection(payload)
    if forbidden:
        raise ReservationError(
            "pre-result reservation contains outcome fields: " + ", ".join(sorted(forbidden))
        )
    if payload.get("trial_config") != trial_config:
        raise ReservationError(
            "reservation trial_config does not exactly match the attempted trial"
        )
    observed_identity = hypothesis_hash(trial_config)
    if payload.get("hypothesis_identity") != observed_identity:
        raise ReservationError("reservation hypothesis_identity does not match trial_config")
    packet_coverage = _validate_historical_packet_coverage(repo)
    forward_epoch_seriality = _validate_forward_epoch_serial_completion(
        repo,
        reserved_hypothesis_identity=payload["hypothesis_identity"],
    )
    governance_epoch = _validate_governance_epoch(
        payload,
        repo=repo,
        historical_identities=packet_coverage["historical_identities"],
        forward_identities_already_logged=forward_epoch_seriality[
            "forward_identities_already_logged"
        ],
    )

    packet_path = payload.get("packet_public_path")
    paper_path = payload.get("paper_public_path")
    if not isinstance(packet_path, str) or not packet_path.startswith("/glassbox/trial-packets/"):
        raise ReservationError("packet_public_path must reserve a stable trial-packet URL")
    if not isinstance(paper_path, str) or not paper_path.startswith("/research/"):
        raise ReservationError("paper_public_path must reserve a stable research URL")

    evidence = payload.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != REQUIRED_EVIDENCE:
        raise ReservationError(
            "evidence must contain exactly: " + ", ".join(sorted(REQUIRED_EVIDENCE))
        )
    validated: dict[str, dict[str, str]] = {}
    for key in sorted(REQUIRED_EVIDENCE):
        item = evidence.get(key)
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ReservationError(f"{key} evidence must contain exactly path and sha256")
        expected = item.get("sha256")
        if not isinstance(expected, str) or SHA256.fullmatch(expected) is None:
            raise ReservationError(f"{key} evidence sha256 must be 64 lowercase hex characters")
        source = _repo_file(repo, item.get("path"))
        observed = _sha256(source)
        if observed != expected:
            relative = source.relative_to(repo)
            raise ReservationError(
                f"{key} evidence hash mismatch: {relative}: {observed} != {expected}"
            )
        validated[key] = {"path": str(source.relative_to(repo)), "sha256": observed}

    return {
        "schema": SCHEMA,
        "status": "VALIDATED_BEFORE_RETURN_COMPUTE",
        "return_identity_id": payload["return_identity_id"],
        "family_trial_account": payload["family_trial_account"],
        "hypothesis_identity": observed_identity,
        "packet_public_path": packet_path,
        "paper_public_path": paper_path,
        "historical_packet_coverage": packet_coverage,
        "forward_epoch_seriality": forward_epoch_seriality,
        "governance_epoch": governance_epoch,
        "evidence": validated,
    }
