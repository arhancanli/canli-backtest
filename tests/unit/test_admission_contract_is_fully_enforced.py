"""Every threshold a contract declares must be read by the code that enforces it.

A contract is a promise about what will be checked. A threshold that sits in the JSON but is
never read by ``evaluate_sleeve_evidence`` is not a weaker check -- it is *no* check, wearing the
costume of one, and it reads as enforced to anyone auditing the config. This repository has hit
that failure mode before (a published law whose guard could not see the offender), so it is
pinned here rather than trusted.

The test does not grep for key names; a reader that constructs keys dynamically would defeat
that. It substitutes a recording mapping for the contract's ``thresholds`` and asserts, against
evidence that *passes*, that every declared key was actually fetched on the path that runs.

It covers every ``config/sleeve_admission_contract*.json``, so a proposed contract cannot be
promoted with decorative thresholds still in it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from test_sleeve_admission import passing_evidence

from alphaforge.validation.sleeve_admission import (
    evaluate_sleeve_evidence,
    load_admission_contract,
)

CONFIG_DIR = Path(__file__).parents[2] / "config"
CONTRACTS = sorted(CONFIG_DIR.glob("sleeve_admission_contract*.json"))


class _RecordingThresholds(Mapping[str, Any]):
    """A read-through mapping that records which keys the evaluator actually fetches."""

    def __init__(self, wrapped: Mapping[str, Any]) -> None:
        self._wrapped = wrapped
        self.read: set[str] = set()

    def __getitem__(self, key: str) -> Any:
        self.read.add(key)
        return self._wrapped[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._wrapped)

    def __len__(self) -> int:
        return len(self._wrapped)


def test_contract_files_exist() -> None:
    """A glob that silently matches nothing would make every test below vacuously pass."""
    assert CONTRACTS, f"no sleeve_admission_contract*.json found under {CONFIG_DIR}"


@pytest.mark.parametrize("contract_path", CONTRACTS, ids=lambda p: p.name)
def test_contract_loads_through_the_real_loader(contract_path: Path) -> None:
    """A contract nobody can load is not a proposal, it is a document.

    ``load_admission_contract`` enforces the supported schema set and derives
    ``evidence_checks_per_candidate`` from the contract's own contents, so this also catches a
    stale check count -- including one that drifts when a new optional gate is declared.
    """
    contract = load_admission_contract(contract_path)
    assert contract["thresholds"]


@pytest.mark.parametrize("contract_path", CONTRACTS, ids=lambda p: p.name)
def test_every_declared_threshold_is_read_by_the_evaluator(contract_path: Path) -> None:
    import json

    contract = json.loads(contract_path.read_text())
    declared = set(contract["thresholds"])
    recorder = _RecordingThresholds(contract["thresholds"])
    instrumented = dict(contract)
    instrumented["thresholds"] = recorder

    evaluate_sleeve_evidence(passing_evidence(), instrumented)

    unenforced = sorted(declared - recorder.read)
    assert not unenforced, (
        f"{contract_path.name} declares thresholds that evaluate_sleeve_evidence never reads: "
        f"{unenforced}. A declared-but-unread threshold is not a weaker gate, it is no gate. "
        "Either wire it into the evaluator or remove it from the contract."
    )


def test_the_guard_itself_can_fail() -> None:
    """A check that cannot fail is worse than no check, so prove this one bites."""
    import json

    contract = json.loads(CONTRACTS[0].read_text())
    contract["thresholds"] = dict(contract["thresholds"])
    contract["thresholds"]["a_threshold_no_evaluator_reads"] = 0.5

    declared = set(contract["thresholds"])
    recorder = _RecordingThresholds(contract["thresholds"])
    instrumented = dict(contract)
    instrumented["thresholds"] = recorder
    evaluate_sleeve_evidence(passing_evidence(), instrumented)

    assert "a_threshold_no_evaluator_reads" in declared - recorder.read
