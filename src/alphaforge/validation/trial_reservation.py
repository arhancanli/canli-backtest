"""Pre-result evidence reservation required before a new return hypothesis is logged."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Final

from alphaforge.validation.experiments import ExperimentLog, hypothesis_hash
from alphaforge.validation.sleeve_admission import _matches_canonical_sha256
from alphaforge.validation.trial_budget import BudgetAmendmentError, effective_budget

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
SERIALITY_WAIVER_DIR: Final[Path] = Path("artifacts/research/seriality_waivers")
SERIALITY_WAIVER_SCHEMA: Final[str] = "canli.alphac-seriality-waiver.v1"
EVIDENCE_CLASSES_POLICY: Final[Path] = Path("config/trial_accounting_evidence_classes.json")
EVIDENCE_CLASSES_SCHEMA: Final[str] = "canli.alphac-trial-evidence-classes.v1"
IDENTITY_BATCH_DIR: Final[Path] = Path("artifacts/research/identity_batches")
IDENTITY_BATCH_SCHEMA: Final[str] = "canli.alphac-identity-batch.v1"
RESERVATION_DIR: Final[Path] = Path("artifacts/research/preregistrations")
RESERVATION_FILENAME: Final[str] = "return_identity_reservation.json"
IDENTITY_BATCH_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "batch_id",
        "family_trial_account",
        "identity_configs",
        "identity_hashes",
        "batch_content_hash",
        "registry_path",
        "all_identities_reserved_before_first_return",
        "interim_result_access_forbidden",
        "pbo_columns",
    }
)
DECLARED_DIAGNOSTIC_FIELDS: Final[frozenset[str]] = frozenset(
    {"scenario_id", "assumptions", "assumptions_sha256"}
)


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


def _identity_is_decided(repo: Path, identity: str) -> bool:
    """True when the identity has a complete packet whose closure is ADMIT, KILL or waived."""
    path = repo / IDENTITY_PACKET_DIR / f"{identity}.json"
    if not path.is_file():
        return False
    try:
        packet = json.loads(path.read_text(encoding="utf-8"))
        if (
            packet.get("schema") != "canli.alphac-identity-trial-packet.v2"
            or packet.get("complete") is not True
            or packet.get("content_hash") != _observed_content_hash(packet)
        ):
            return False
        _validate_prior_identity_admission_disposition(repo, identity, packet)
    except (json.JSONDecodeError, OSError, ReservationError, KeyError):
        return False
    return True


def _open_identity_batches(repo: Path) -> dict[str, list[str]]:
    """Sealed batches with at least one member not yet decided, keyed by batch id.

    A batch is atomic: while any member is undecided, no reservation outside the batch may be
    made (protocol: "block any unrelated reservation until every batch packet is complete").
    """
    registry_dir = repo / IDENTITY_BATCH_DIR
    if not registry_dir.is_dir():
        return {}
    open_batches: dict[str, list[str]] = {}
    for path in sorted(registry_dir.glob("*.json")):
        try:
            registry = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise ReservationError(f"identity batch registry is unreadable: {path.name}") from error
        if registry.get("schema") != IDENTITY_BATCH_SCHEMA:
            raise ReservationError(f"identity batch registry schema mismatch: {path.name}")
        if registry.get("content_hash") != _observed_content_hash(registry):
            raise ReservationError(f"identity batch registry content hash mismatch: {path.name}")
        members = registry.get("identity_hashes")
        if not isinstance(members, list) or not members:
            raise ReservationError(f"identity batch registry has no members: {path.name}")
        undecided = [str(m) for m in members if not _identity_is_decided(repo, str(m))]
        if undecided:
            open_batches[str(registry["batch_id"])] = undecided
    return open_batches


def _validate_forward_epoch_serial_completion(
    repo: Path,
    *,
    reserved_hypothesis_identity: str,
    batch_id: str | None = None,
    batch_members: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    for open_batch_id, undecided in _open_identity_batches(repo).items():
        if open_batch_id != batch_id:
            raise ReservationError(
                "new return identity blocked: identity batch "
                f"{open_batch_id} is open with undecided members "
                f"({', '.join(undecided)}); unrelated reservations wait for the batch"
            )
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
    same_batch_pending: list[str] = []
    for identity in forward_keys:
        if identity in batch_members:
            # Atomic batch: a sibling reserved in the same sealed batch is allowed to be
            # undecided (the batch decides together); it is never allowed to be unreserved,
            # which _validate_identity_batch enforces against the registry and the disk.
            same_batch_pending.append(identity)
            continue
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
        # Wired 2026-09-14 (plan task 4, owner-delegated): a complete packet is not a decided
        # one. The prior identity's sealed closure must say ADMIT or KILL, or the owner must
        # have waived it against this exact packet's content hash.
        disposition = _validate_prior_identity_admission_disposition(repo, identity, packet)
        verified_packets.append(
            {
                "hypothesis_key": identity,
                "packet_path": str(IDENTITY_PACKET_DIR / f"{identity}.json"),
                "packet_content_hash": packet["content_hash"],
                "closure_disposition": disposition,
            }
        )
    return {
        "policy": (
            "SERIAL_COMPLETE_PACKET_BEFORE_NEXT_FORWARD_IDENTITY"
            if batch_id is None
            else "ATOMIC_BATCH_SIBLINGS_PENDING_TOGETHER_UNRELATED_IDENTITIES_BLOCKED"
        ),
        "prior_identities_must_be_decided": True,
        "forward_identities_already_logged": len(forward_keys),
        "forward_identity_keys_logged": list(forward_keys),
        "complete_forward_packets_verified": len(verified_packets),
        "verified_packets": verified_packets,
        "same_batch_pending": same_batch_pending,
    }


def _closure_path_from_packet(packet: dict[str, Any]) -> str:
    """Find the admission-or-kill closure this packet's evidence points at.

    Convention observed in `artifacts/research/trial_packets/da5f5f47f99f9bd2.json`: the
    `admission_or_kill_decision` evidence list carries the sealed result receipt (keyed by
    `source_path`/`public_path`) alongside the closure (keyed by `path`, ending
    `_admission_closure.json`). Returns the first evidence entry whose `path` matches that
    convention.
    """
    evidence = packet["required_sections"]["admission_or_kill_decision"]["evidence"]
    for item in evidence:
        path = item.get("path") if isinstance(item, dict) else None
        if isinstance(path, str) and path.endswith("_admission_closure.json"):
            return path
    raise ReservationError("packet does not reference an admission-or-kill closure")


def _validate_prior_identity_admission_disposition(
    repo: Path,
    identity: str,
    packet: dict[str, Any],
) -> str:
    """Close the real seriality gap: a complete packet is not the same as a decided one.

    Wired into `_validate_forward_epoch_serial_completion` on 2026-09-14 (plan task 4). Packet
    completion records evidence accounting, not a gate outcome, per that packet's own
    `governance_finding.seriality_interaction`. This requires, for the prior identity's sealed
    closure, a disposition of ADMIT or KILL, or a signed owner waiver bound to this exact
    packet's `content_hash` (so a re-seal invalidates the waiver). Returns the disposition that
    let the reservation proceed: ``ADMIT``, ``KILL`` or ``WAIVED``.
    """
    closure_path = _repo_file(repo, _closure_path_from_packet(packet))
    try:
        closure = json.loads(closure_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ReservationError(
            f"prior identity admission closure is unreadable: {identity}"
        ) from error
    if closure.get("content_hash") != _observed_content_hash(closure):
        raise ReservationError(
            f"prior identity admission closure content hash mismatch: {identity}"
        )
    disposition = closure.get("decision", {}).get("disposition")
    if disposition in {"ADMIT", "KILL"}:
        return str(disposition)

    waiver_path = repo / SERIALITY_WAIVER_DIR / f"{identity}.json"
    if not waiver_path.is_file():
        raise ReservationError(
            f"prior forward identity {identity} closure disposition is neither ADMIT nor KILL "
            "and no seriality waiver is on file"
        )
    try:
        waiver = json.loads(waiver_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ReservationError(f"seriality waiver is unreadable: {identity}") from error
    if waiver.get("schema") != SERIALITY_WAIVER_SCHEMA:
        raise ReservationError(f"seriality waiver schema mismatch: {identity}")
    if waiver.get("waived_hypothesis_key") != identity:
        raise ReservationError(f"seriality waiver does not name this identity: {identity}")
    if waiver.get("waived_packet_content_hash") != packet["content_hash"]:
        raise ReservationError(f"seriality waiver does not match the sealed packet: {identity}")
    if waiver.get("content_hash") != _observed_content_hash(waiver):
        raise ReservationError(f"seriality waiver content hash mismatch: {identity}")
    reason = waiver.get("reason")
    if not isinstance(reason, str) or len(reason.strip()) < 12:
        raise ReservationError(f"seriality waiver reason is missing or too short: {identity}")
    authorized_by = waiver.get("authorized_by")
    if not isinstance(authorized_by, str) or not authorized_by.startswith("Arhan Canli, owner,"):
        raise ReservationError(f"seriality waiver is not authorized by the owner: {identity}")
    return "WAIVED"


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
    reserved_unlogged_predecessors: int = 0,
) -> dict[str, Any]:
    """Bind a new identity to the exact in-force gate and staged trial-budget epoch.

    ``reserved_unlogged_predecessors`` counts batch siblings sealed before this identity that
    have not yet reached the ledger: each holds an ordinal, so the next governed ordinal moves
    past them even though nothing has run.
    """
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
    # An ordinal above the policy's own budget also binds the owner's budget amendment
    # (alphaforge.validation.trial_budget); at or below it the block is exactly as before.
    if "budget_amendment_sha256" in governance:
        expected_keys.add("budget_amendment_sha256")
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
    expected_ordinal = (
        historical_identities
        + forward_identities_already_logged
        + reserved_unlogged_predecessors
        + 1
    )
    first_effective = contract["prospective_scope"]["effective_on_or_after_reservation_ordinal"]
    policy_budget = policy.get("hypothesis_identity_budget")
    if ordinal != expected_ordinal or ordinal < first_effective:
        raise ReservationError(
            f"reservation_ordinal must be the next governed identity: {expected_ordinal}"
        )
    if not isinstance(policy_budget, int) or not isinstance(ordinal, int):
        raise ReservationError("staged hypothesis-identity budget is exhausted")
    try:
        in_force = effective_budget(repo, paths["trial_policy"])
    except BudgetAmendmentError as error:
        raise ReservationError(str(error)) from error
    amendment_claim = governance.get("budget_amendment_sha256")
    if ordinal > policy_budget:
        if in_force.amendment_sha256 is None or amendment_claim is None:
            raise ReservationError("staged hypothesis-identity budget is exhausted")
        if f"sha256:{amendment_claim}" != in_force.amendment_sha256:
            raise ReservationError("governance_epoch budget amendment hash mismatch")
        if ordinal > in_force.ceiling:
            raise ReservationError("staged hypothesis-identity budget is exhausted")
    elif amendment_claim is not None:
        raise ReservationError(
            "governance_epoch binds a budget amendment the ordinal does not need"
        )
    out: dict[str, Any] = {
        "admission_contract_schema": contract["schema"],
        "admission_contract_sha256": _sha256(paths["admission_contract"]),
        "effective_contract_hash": effective_hash,
        "trial_policy_schema": policy["schema"],
        "trial_policy_sha256": _sha256(paths["trial_policy"]),
        "hypothesis_identity_budget": policy_budget,
        "staged_hard_reviews": policy["prospective_v7_review"]["staged_hard_reviews"],
        "promotion_receipt_sha256": _sha256(paths["promotion_receipt"]),
        "reservation_ordinal": ordinal,
    }
    if ordinal > policy_budget:
        out["hypothesis_identity_ceiling"] = in_force.ceiling
        out["staged_hard_reviews"] = list(in_force.staged_hard_reviews)
        out["budget_amendment_sha256"] = amendment_claim
    return out


DIAGNOSTIC_SCENARIO_CLASSES: Final[frozenset[str]] = frozenset(
    {"cost_stress_scenarios", "execution_stress_scenarios", "capacity_scenarios"}
)
DIAGNOSTIC_SCENARIO_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "scenario_id",
        "assumptions",
        "assumptions_sha256",
        "result",
        "result_sha256",
        "primary_decision_path_sha256",
    }
)


def _validate_diagnostic_scenarios(
    payload: dict[str, Any],
    *,
    sealed_primary_decision_path_sha256: str,
) -> dict[str, Any]:
    """A diagnostic reuses the sealed primary decision path; it never forks it (spec 2(a)).

    Unwired: nothing in `validate_reservation` calls this yet (plan task 1). A diagnostic
    scenario computes cost, execution, or capacity outcomes on top of the exact decisions already
    sealed in a primary return path (same instrument weights, same entry and exit times); it
    cannot alter which decisions were made and cannot spend a new hypothesis identity. Every
    scenario must therefore carry a `primary_decision_path_sha256` bound to the one sealed path
    supplied by the caller -- a mismatch means the scenario belongs to a different decision path
    and must be classified as a new identity, not accepted as a diagnostic of this one.
    """
    diagnostic_scenarios = payload.get("diagnostic_scenarios")
    if diagnostic_scenarios is None:
        return {"diagnostic_scenario_count": 0}
    if not isinstance(diagnostic_scenarios, dict) or not set(diagnostic_scenarios).issubset(
        DIAGNOSTIC_SCENARIO_CLASSES
    ):
        raise ReservationError(
            "diagnostic_scenarios must be keyed by a subset of: "
            + ", ".join(sorted(DIAGNOSTIC_SCENARIO_CLASSES))
        )
    scenario_count = 0
    for scenario_class, scenarios in diagnostic_scenarios.items():
        if not isinstance(scenarios, list):
            raise ReservationError(f"diagnostic scenario class must be a list: {scenario_class}")
        for index, scenario in enumerate(scenarios):
            has_required_fields = (
                isinstance(scenario, dict) and set(scenario) == DIAGNOSTIC_SCENARIO_REQUIRED_FIELDS
            )
            if not has_required_fields:
                raise ReservationError(
                    f"diagnostic scenario {scenario_class}[{index}] must declare exactly: "
                    + ", ".join(sorted(DIAGNOSTIC_SCENARIO_REQUIRED_FIELDS))
                )
            assumptions = scenario["assumptions"]
            result = scenario["result"]
            if not isinstance(assumptions, dict) or not assumptions:
                raise ReservationError(
                    f"diagnostic scenario {scenario_class}[{index}] assumptions must be a "
                    "non-empty object"
                )
            if not _matches_canonical_sha256(scenario["assumptions_sha256"], assumptions):
                raise ReservationError(
                    f"diagnostic scenario {scenario_class}[{index}] assumptions_sha256 does not "
                    "match the frozen assumptions"
                )
            if not isinstance(result, dict) or not result:
                raise ReservationError(
                    f"diagnostic scenario {scenario_class}[{index}] result must be a non-empty "
                    "object"
                )
            if not _matches_canonical_sha256(scenario["result_sha256"], result):
                raise ReservationError(
                    f"diagnostic scenario {scenario_class}[{index}] result_sha256 does not match "
                    "the frozen result"
                )
            if scenario["primary_decision_path_sha256"] != sealed_primary_decision_path_sha256:
                raise ReservationError(
                    f"diagnostic scenario {scenario_class}[{index}] primary_decision_path_sha256 "
                    "does not match the sealed primary decision path"
                )
            scenario_count += 1
    return {"diagnostic_scenario_count": scenario_count}


def _validate_declared_diagnostic_scenarios(payload: dict[str, Any]) -> dict[str, Any]:
    """Pre-result shape of the diagnostics: frozen assumptions, no results, unique ids.

    Wired 2026-09-14 (plan task 1, owner-delegated). Before the first return a diagnostic
    scenario is an assumptions manifest bound by hash; its result and its binding to the sealed
    primary decision path arrive only when the run seals, where `_validate_diagnostic_scenarios`
    checks them. A result field here is an outcome field and is refused.
    """
    diagnostic_scenarios = payload.get("diagnostic_scenarios")
    if diagnostic_scenarios is None:
        return {"declared_diagnostic_scenarios": 0, "classes": {}}
    if not isinstance(diagnostic_scenarios, dict) or not set(diagnostic_scenarios).issubset(
        DIAGNOSTIC_SCENARIO_CLASSES
    ):
        raise ReservationError(
            "diagnostic_scenarios must be keyed by a subset of: "
            + ", ".join(sorted(DIAGNOSTIC_SCENARIO_CLASSES))
        )
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for scenario_class, scenarios in diagnostic_scenarios.items():
        if not isinstance(scenarios, list):
            raise ReservationError(f"diagnostic scenario class must be a list: {scenario_class}")
        for index, scenario in enumerate(scenarios):
            if not isinstance(scenario, dict) or set(scenario) != DECLARED_DIAGNOSTIC_FIELDS:
                raise ReservationError(
                    f"declared diagnostic {scenario_class}[{index}] must carry exactly: "
                    + ", ".join(sorted(DECLARED_DIAGNOSTIC_FIELDS))
                    + " (a result before the run is an outcome field)"
                )
            scenario_id = scenario["scenario_id"]
            if not isinstance(scenario_id, str) or IDENTIFIER.fullmatch(scenario_id) is None:
                raise ReservationError(
                    f"declared diagnostic {scenario_class}[{index}] scenario_id must be a "
                    "stable snake-case identifier"
                )
            if scenario_id in seen:
                raise ReservationError(f"duplicate diagnostic scenario_id: {scenario_id}")
            seen.add(scenario_id)
            assumptions = scenario["assumptions"]
            if not isinstance(assumptions, dict) or not assumptions:
                raise ReservationError(
                    f"declared diagnostic {scenario_id} assumptions must be a non-empty object"
                )
            if not _matches_canonical_sha256(scenario["assumptions_sha256"], assumptions):
                raise ReservationError(
                    f"declared diagnostic {scenario_id} assumptions_sha256 does not match"
                )
        counts[scenario_class] = len(scenarios)
    return {"declared_diagnostic_scenarios": len(seen), "classes": counts}


def batch_content_hash(batch_id: str, family_trial_account: str, identity_hashes: list[str]) -> str:
    """The hash that seals a batch: its id, family and ordered member identities, nothing else."""
    body = {
        "batch_id": batch_id,
        "family_trial_account": family_trial_account,
        "identity_hashes": list(identity_hashes),
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _reservations_on_disk(repo: Path) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    root = repo / RESERVATION_DIR
    if not root.is_dir():
        return found
    for path in sorted(root.glob(f"*/{RESERVATION_FILENAME}")):
        try:
            found.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as error:
            raise ReservationError(f"reservation on disk is unreadable: {path}") from error
    return found


def _validate_identity_batch(payload: dict[str, Any], repo: Path) -> dict[str, Any] | None:
    """An atomic, predeclared batch of selectable identities (spec: batch accounting).

    Every member is a counted hypothesis identity with its own reservation; the batch registry
    (`artifacts/research/identity_batches/<batch_id>.json`) is sealed before the first return
    and this reservation must match it exactly: no member added, removed or reordered. Every
    sibling sealed BEFORE this one must already be reserved on disk under the same batch hash,
    so an identity cannot claim membership of a batch that never reserved it.
    """
    batch = payload.get("identity_batch")
    if batch is None:
        return None
    if not isinstance(batch, dict) or set(batch) != IDENTITY_BATCH_REQUIRED_FIELDS:
        raise ReservationError(
            "identity_batch must carry exactly: "
            + ", ".join(sorted(IDENTITY_BATCH_REQUIRED_FIELDS))
        )
    batch_id = batch["batch_id"]
    if not isinstance(batch_id, str) or IDENTIFIER.fullmatch(batch_id) is None:
        raise ReservationError("identity_batch.batch_id must be a stable snake-case identifier")
    if batch["family_trial_account"] != payload.get("family_trial_account"):
        raise ReservationError("identity_batch.family_trial_account must equal the reservation's")
    configs = batch["identity_configs"]
    if (
        not isinstance(configs, list)
        or not configs
        or not all(isinstance(c, dict) and c for c in configs)
    ):
        raise ReservationError(
            "identity_batch.identity_configs must be a non-empty list of configs"
        )
    hashes = [hypothesis_hash(config) for config in configs]
    if batch["identity_hashes"] != hashes:
        raise ReservationError(
            "identity_batch.identity_hashes must be the hypothesis hashes of identity_configs, "
            "in order"
        )
    if len(set(hashes)) != len(hashes):
        raise ReservationError("identity_batch contains the same hypothesis identity twice")
    reserved = payload.get("hypothesis_identity")
    if reserved not in hashes:
        raise ReservationError("reserved identity is not a member of its own identity_batch")
    if batch["all_identities_reserved_before_first_return"] is not True:
        raise ReservationError("identity_batch must reserve every identity before the first return")
    if batch["interim_result_access_forbidden"] is not True:
        raise ReservationError("identity_batch must forbid interim result access")
    if batch["pbo_columns"] != len(hashes):
        raise ReservationError("identity_batch.pbo_columns must equal the number of members")
    expected_hash = batch_content_hash(batch_id, str(batch["family_trial_account"]), hashes)
    if batch["batch_content_hash"] != expected_hash:
        raise ReservationError("identity_batch.batch_content_hash does not seal its members")
    registry_relative = batch["registry_path"]
    if registry_relative != (IDENTITY_BATCH_DIR / f"{batch_id}.json").as_posix():
        raise ReservationError("identity_batch.registry_path must be the canonical registry file")
    registry = json.loads(_repo_file(repo, registry_relative).read_text(encoding="utf-8"))
    if (
        registry.get("schema") != IDENTITY_BATCH_SCHEMA
        or registry.get("content_hash") != _observed_content_hash(registry)
        or registry.get("batch_id") != batch_id
        or registry.get("family_trial_account") != batch["family_trial_account"]
        or registry.get("identity_configs") != configs
        or registry.get("identity_hashes") != hashes
        or registry.get("batch_content_hash") != expected_hash
        or registry.get("pbo_columns") != len(hashes)
        or registry.get("all_identities_reserved_before_first_return") is not True
        or registry.get("interim_result_access_forbidden") is not True
    ):
        raise ReservationError(
            "identity_batch does not match its sealed registry: a member was added, removed or "
            "reordered after the batch was sealed, or the registry is invalid"
        )
    _parse_utc(registry.get("sealed_at"))
    position = hashes.index(str(reserved))
    predecessors = hashes[:position]
    on_disk = {
        str(r.get("hypothesis_identity")): r
        for r in _reservations_on_disk(repo)
        if isinstance(r.get("identity_batch"), dict)
        and r["identity_batch"].get("batch_content_hash") == expected_hash
    }
    for sibling in predecessors:
        if sibling not in on_disk:
            raise ReservationError(
                f"identity batch member {sibling} precedes this identity in the sealed batch "
                "but has no reservation on disk under the same batch hash"
            )
    return {
        "batch_id": batch_id,
        "batch_content_hash": expected_hash,
        "registry_path": registry_relative,
        "members": len(hashes),
        "position": position + 1,
        "pbo_columns": len(hashes),
        "predecessors": predecessors,
        "pbo_defined": len(hashes) >= 2,
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
    declared_diagnostics = _validate_declared_diagnostic_scenarios(payload)
    identity_batch = _validate_identity_batch(payload, repo)
    packet_coverage = _validate_historical_packet_coverage(repo)
    forward_epoch_seriality = _validate_forward_epoch_serial_completion(
        repo,
        reserved_hypothesis_identity=payload["hypothesis_identity"],
        batch_id=None if identity_batch is None else identity_batch["batch_id"],
        batch_members=(
            frozenset()
            if identity_batch is None
            else frozenset(payload["identity_batch"]["identity_hashes"])
        ),
    )
    logged = set(forward_epoch_seriality["forward_identity_keys_logged"])
    reserved_unlogged_predecessors = (
        0
        if identity_batch is None
        else sum(1 for sibling in identity_batch["predecessors"] if sibling not in logged)
    )
    governance_epoch = _validate_governance_epoch(
        payload,
        repo=repo,
        historical_identities=packet_coverage["historical_identities"],
        forward_identities_already_logged=forward_epoch_seriality[
            "forward_identities_already_logged"
        ],
        reserved_unlogged_predecessors=reserved_unlogged_predecessors,
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
        "declared_diagnostic_scenarios": declared_diagnostics,
        "identity_batch": identity_batch,
        "evidence": validated,
    }
