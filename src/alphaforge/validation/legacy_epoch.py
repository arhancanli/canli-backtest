"""Fail-closed access to the retired legacy research epoch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from alphaforge.validation.experiments import ExperimentUnion

LEGACY_CLOSURE = Path("artifacts/research/legacy_research_epoch_closure.json")
EXPECTED_STATUS = "LEGACY_EPOCH_RETIRED_FAIL_CLOSED"
EXPECTED_IDENTITIES = 228


def _content_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "content_hash"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def load_legacy_identity_keys(repo: Path) -> frozenset[str]:
    """Validate the immutable closure and return its exact retired identity set."""
    path = repo / LEGACY_CLOSURE
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("content_hash") != _content_hash(payload):
        raise ValueError("legacy research epoch closure content hash mismatch")
    if payload.get("status") != EXPECTED_STATUS:
        raise ValueError("legacy research epoch is not retired fail closed")
    rows = payload.get("identities")
    if not isinstance(rows, list) or len(rows) != EXPECTED_IDENTITIES:
        raise ValueError("legacy research epoch must contain exactly 228 identities")
    keys = frozenset(row.get("hypothesis_key") for row in rows if isinstance(row, dict))
    if None in keys or len(keys) != EXPECTED_IDENTITIES:
        raise ValueError("legacy research epoch identities are missing or duplicated")
    return frozenset(str(key) for key in keys)


def legacy_selection_context(repo: Path) -> tuple[int, float, list[str]]:
    """Compute selection statistics only over the closure-bound legacy identities."""
    keys = load_legacy_identity_keys(repo)
    union = ExperimentUnion.discover(repo / "var" / "experiments.jsonl", repo)
    count, variance = union.hypothesis_selection_context(keys)
    if count != EXPECTED_IDENTITIES:
        raise ValueError("legacy selection context did not resolve exactly 228 identities")
    paths = [str(path.relative_to(repo)) for path in union.paths]
    return count, variance, paths


def validate_legacy_packet_bound_file(repo: Path, relative: str) -> str:
    """Validate a file hash through the closure-bound identity-packet tree."""
    load_legacy_identity_keys(repo)
    closure_path = repo / LEGACY_CLOSURE
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    binding = closure.get("source_bindings", {}).get("identity_packet_index", {})
    index_path = repo / str(binding.get("path", ""))
    if not index_path.is_file() or hashlib.sha256(
        index_path.read_bytes()
    ).hexdigest() != binding.get("sha256"):
        raise ValueError("legacy identity-packet index file hash mismatch")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("content_hash") != _content_hash(index) or index.get(
        "content_hash"
    ) != binding.get("content_hash"):
        raise ValueError("legacy identity-packet index content hash mismatch")

    def references(value: Any) -> list[str]:
        if isinstance(value, dict):
            found = []
            if value.get("path") == relative and isinstance(value.get("sha256"), str):
                found.append(value["sha256"])
            for child in value.values():
                found.extend(references(child))
            return found
        if isinstance(value, list):
            return [digest for child in value for digest in references(child)]
        return []

    expected_hashes: set[str] = set()
    for row in index.get("packets", []):
        identity = row.get("hypothesis_key")
        packet_path = index_path.parent / f"{identity}.json"
        if hashlib.sha256(packet_path.read_bytes()).hexdigest() != row.get("packet_file_sha256"):
            raise ValueError(f"{identity}: legacy identity-packet file hash mismatch")
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        if packet.get("content_hash") != _content_hash(packet):
            raise ValueError(f"{identity}: legacy identity-packet content hash mismatch")
        expected_hashes.update(references(packet))
    if not expected_hashes:
        replay_path = repo / "artifacts/audit/sleeve_publication_replay_verification.json"
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        if replay.get("content_hash") != _content_hash(replay):
            raise ValueError("legacy publication replay receipt content hash mismatch")
        replay_hash = replay.get("source_bindings", {}).get("result_objects", {}).get(relative)
        if isinstance(replay_hash, str):
            expected_hashes.add(replay_hash)
    if len(expected_hashes) != 1:
        raise ValueError(f"legacy evidence does not bind one unambiguous hash for {relative}")
    expected = next(iter(expected_hashes))
    path = repo / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"legacy packet-bound file hash mismatch: {relative}")
    return expected
