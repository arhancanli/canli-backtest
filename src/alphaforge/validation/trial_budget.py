"""The hypothesis-identity ceiling in force: the policy's budget, raised only by an owner amendment.

``config/trial_accounting.json`` is the POLICY. It is embedded byte-for-byte in the admission v7
promotion receipt and hash-bound by every sealed reservation, so it is never edited to record a
decision (2026-09-14: recording one review inside it drifted five sealed bindings). A budget raise
is recorded beside it, in ``config/trial_accounting_budget_amendments.json``, the way held reviews
are recorded in ``trial_accounting_reviews.json``.

An amendment is valid only if its content hash matches, it binds the policy's exact bytes, and its
ceilings rise. Reservations at or below the policy's own budget are validated exactly as before;
an ordinal above it must bind the amendment file's hash, so no identity past 400 can exist
without the recorded decision that allowed it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

AMENDMENTS: Final[Path] = Path("config/trial_accounting_budget_amendments.json")
AMENDMENTS_SCHEMA: Final = "canli.alphac-trial-budget-amendments.v1"


class BudgetAmendmentError(ValueError):
    """The amendment record is present but cannot be trusted."""


@dataclass(frozen=True)
class EffectiveBudget:
    ceiling: int
    staged_hard_reviews: tuple[int, ...]
    amendment_sha256: str | None
    policy_budget: int


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def content_hash(body: dict[str, Any]) -> str:
    payload = {k: v for k, v in body.items() if k != "content_hash"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def effective_budget(repo: Path, policy_path: Path) -> EffectiveBudget:
    """The ceiling and staged reviews in force for ``policy_path`` under ``repo``."""
    policy = json.loads(policy_path.read_text())
    base = int(policy["hypothesis_identity_budget"])
    reviews = tuple(
        int(x) for x in (policy.get("prospective_v7_review") or {}).get("staged_hard_reviews", [])
    )
    path = repo / AMENDMENTS
    if not path.is_file():
        return EffectiveBudget(base, reviews, None, base)
    doc = json.loads(path.read_text())
    if doc.get("schema") != AMENDMENTS_SCHEMA:
        raise BudgetAmendmentError(f"{AMENDMENTS}: unexpected schema {doc.get('schema')!r}")
    if doc.get("content_hash") != content_hash(doc):
        raise BudgetAmendmentError(f"{AMENDMENTS}: content hash does not match the file")
    ceiling = base
    staged = list(reviews)
    for amendment in doc.get("amendments", []):
        if amendment.get("policy_sha256") != _sha256_file(policy_path):
            raise BudgetAmendmentError(
                f"{AMENDMENTS}: amendment {amendment.get('id')!r} binds a different policy"
            )
        new = int(amendment["hypothesis_identity_ceiling"])
        if new <= ceiling:
            raise BudgetAmendmentError(f"{AMENDMENTS}: ceilings must rise ({new} <= {ceiling})")
        added = [int(x) for x in amendment.get("staged_hard_reviews", [])]
        if not added or max(added) != new or any(x <= ceiling for x in added):
            raise BudgetAmendmentError(
                f"{AMENDMENTS}: staged reviews must lie above {ceiling} and end at {new}"
            )
        ceiling = new
        staged.extend(added)
    return EffectiveBudget(ceiling, tuple(sorted(set(staged))), _sha256_file(path), base)


def governance_epoch_fields(
    repo: Path,
    *,
    admission_contract: Path,
    trial_policy: Path,
    promotion_receipt: Path,
    effective_contract_hash: str,
    ordinal: int,
) -> dict[str, Any]:
    """The ``governance_epoch`` block a new reservation must carry at ``ordinal``.

    Adds ``budget_amendment_sha256`` exactly when the ordinal is above the policy's own budget,
    which is when the validator requires it.
    """
    budget = effective_budget(repo, repo / trial_policy)
    fields: dict[str, Any] = {
        "admission_contract_path": admission_contract.as_posix(),
        "trial_policy_path": trial_policy.as_posix(),
        "promotion_receipt_path": promotion_receipt.as_posix(),
        "admission_contract_sha256": _sha256_file(repo / admission_contract).removeprefix(
            "sha256:"
        ),
        "trial_policy_sha256": _sha256_file(repo / trial_policy).removeprefix("sha256:"),
        "promotion_receipt_sha256": _sha256_file(repo / promotion_receipt).removeprefix("sha256:"),
        "effective_contract_hash": effective_contract_hash,
        "reservation_ordinal": ordinal,
    }
    if ordinal > budget.policy_budget:
        if budget.amendment_sha256 is None:
            raise BudgetAmendmentError(
                f"ordinal {ordinal} is above the policy budget {budget.policy_budget} and no "
                "budget amendment is recorded"
            )
        fields["budget_amendment_sha256"] = budget.amendment_sha256.removeprefix("sha256:")
    return fields
