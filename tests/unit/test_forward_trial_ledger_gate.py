from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import alphaforge.validation.experiments as experiments
from alphaforge.validation.experiments import ExperimentLog, ExperimentUnion, hypothesis_hash

ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _governance_fixture(repo: Path) -> dict[str, object]:
    bindings = {}
    for relative in (
        "config/sleeve_admission_contract.json",
        "config/trial_accounting.json",
        "config/admission_v7_promotion.json",
    ):
        source = ROOT / relative
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        key = {
            "config/sleeve_admission_contract.json": "admission_contract",
            "config/trial_accounting.json": "trial_policy",
            "config/admission_v7_promotion.json": "promotion_receipt",
        }[relative]
        bindings[f"{key}_path"] = relative
        bindings[f"{key}_sha256"] = _sha(target)
    contract = json.loads((repo / "config/sleeve_admission_contract.json").read_text())
    bindings["effective_contract_hash"] = contract["prospective_scope"][
        "effective_contract_content_hash"
    ]
    bindings["reservation_ordinal"] = 229
    return bindings


def _reservation(repo: Path, trial: dict[str, object]) -> Path:
    manifest = {
        "schema": "canli.alphac-trial-packet-manifest.v2",
        "summary": {
            "distinct_hypothesis_identities": 228,
            "complete_trial_packets": 228,
            "incomplete_trial_packets": 0,
            "published_identity_packets": 228,
            "coverage_status": "COMPLETE",
        },
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["content_hash"] = "sha256:" + hashlib.sha256(canonical).hexdigest()
    manifest_path = repo / "artifacts" / "research" / "trial_packet_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    evidence: dict[str, dict[str, str]] = {}
    for name in (
        "preregistration",
        "input_data_manifest",
        "runner",
        "python_project",
        "locked_environment",
    ):
        path = repo / f"{name}.txt"
        path.write_text(name, encoding="utf-8")
        evidence[name] = {"path": path.name, "sha256": _sha(path)}
    payload = {
        "schema": "canli.alphac-forward-trial-reservation.v1",
        "status": "RETURN_IDENTITY_RESERVED",
        "family_trial_account": "new_family",
        "return_identity_id": "new_family_v1",
        "hypotheses_spent": 1,
        "reserved_at": "2026-08-23T02:00:00+00:00",
        "trial_config": trial,
        "hypothesis_identity": hypothesis_hash(trial),
        "packet_public_path": "/glassbox/trial-packets/new_family_v1.json",
        "paper_public_path": "/research/new-family-v1",
        "governance_epoch": _governance_fixture(repo),
        "evidence": evidence,
    }
    path = repo / "reservation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _record(log: ExperimentLog, config: dict[str, object], reservation: Path | None = None) -> None:
    log.record(
        config,
        sharpe_ann=0.1,
        sharpe_per_period=0.01,
        n_obs=100,
        skew=0.0,
        kurtosis=3.0,
        now_ms=1,
        reservation_path=reservation,
    )


def test_canonical_ledger_blocks_new_hypothesis_without_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiments, "_REPO_ROOT", tmp_path)
    log = ExperimentLog(tmp_path / "var" / "experiments.jsonl")
    with pytest.raises(RuntimeError, match="forward trial reservation is required"):
        _record(log, {"alpha_names": ["new_edge"], "start": 1, "end": 2})
    assert not log.path.exists()


def test_valid_reservation_allows_one_identity_and_window_remeasurements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiments, "_REPO_ROOT", tmp_path)
    log = ExperimentLog(tmp_path / "var_sharadar" / "experiments.jsonl")
    trial = {"alpha_names": ["new_edge"], "start": 1, "end": 2}
    _record(log, trial, _reservation(tmp_path, trial))
    _record(log, {**trial, "start": 3, "end": 4})
    assert log.n_trials() == 2
    assert log.n_hypotheses() == 1


def test_changed_reservation_evidence_blocks_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiments, "_REPO_ROOT", tmp_path)
    log = ExperimentLog(tmp_path / "artifacts" / "candidate" / "experiments.jsonl")
    trial = {"alpha_names": ["new_edge"], "start": 1, "end": 2}
    reservation = _reservation(tmp_path, trial)
    (tmp_path / "runner.txt").write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="runner evidence hash mismatch"):
        _record(log, trial, reservation)
    assert not log.path.exists()


def test_incomplete_historical_packets_block_new_identity_before_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiments, "_REPO_ROOT", tmp_path)
    log = ExperimentLog(tmp_path / "var" / "experiments.jsonl")
    trial = {"alpha_names": ["new_edge"], "start": 1, "end": 2}
    reservation = _reservation(tmp_path, trial)
    path = tmp_path / "artifacts" / "research" / "trial_packet_manifest.json"
    manifest = json.loads(path.read_text())
    manifest.pop("content_hash")
    manifest["summary"].update(
        {
            "complete_trial_packets": 2,
            "incomplete_trial_packets": 1,
            "coverage_status": "INCOMPLETE_BACKFILL_REQUIRED",
        }
    )
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["content_hash"] = "sha256:" + hashlib.sha256(canonical).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="historical trial-packet coverage is incomplete"):
        _record(log, trial, reservation)
    assert not log.path.exists()


def test_union_allows_existing_hypothesis_to_enter_a_new_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiments, "_REPO_ROOT", tmp_path)
    trial = {"alpha_names": ["existing_edge"], "start": 1, "end": 2}
    base = ExperimentLog(tmp_path / "var" / "experiments.jsonl")
    _record(base, trial, _reservation(tmp_path, trial))

    active_path = tmp_path / "var_sharadar" / "experiments.jsonl"
    union = ExperimentUnion(active_path, (base.path, active_path))
    _record(union, {**trial, "start": 3, "end": 4})
    assert ExperimentLog(active_path).n_trials() == 1
    assert union.n_hypotheses() == 1
