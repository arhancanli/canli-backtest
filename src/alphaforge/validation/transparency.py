"""Validation primitives for the disclosed track-record transparency chain."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

GENESIS: Final[str] = "0" * 64
PAYLOAD_SCHEMA: Final[str] = "canli.alphac-track-record-daily-digest.v1"
TRANSPARENCY_SCHEMA: Final[str] = "glassbox.transparency_log/2"


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def daily_track_record_payload(state: dict[str, Any]) -> dict[str, Any]:
    """Return the exact daily-resolution payload committed by the public chain."""

    def daily(curve: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        by_day: dict[str, float] = {}
        for point in curve or []:
            by_day[str(point["date"])] = round(float(point["equity"]), 2)
        return [{"date": day, "equity": by_day[day]} for day in sorted(by_day)]

    algorithms = [
        {
            "key": algorithm["key"],
            "standalone_sharpe": algorithm.get("standalone_sharpe"),
            "go_live": algorithm.get("go_live"),
            "live": daily(algorithm.get("live_curve")),
        }
        for algorithm in state.get("algorithms", [])
    ]
    return {
        "go_live_date": state.get("go_live_date"),
        "rebaseline": state.get("rebaseline"),
        "strategic_tilt": (state.get("book") or {}).get("strategic_tilt"),
        "book_sleeves": [
            {"key": sleeve.get("key"), "weight": sleeve.get("weight")}
            for sleeve in (state.get("book") or {}).get("sleeves", [])
        ],
        "algorithms": algorithms,
        "book_live": daily(state.get("live_curve")),
    }


def validate_transparency_document(
    document: dict[str, Any], *, expected_state: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate links, signatures, disclosed payloads and optional current-state identity."""
    if document.get("schema") != TRANSPARENCY_SCHEMA:
        raise ValueError("transparency schema mismatch")
    entries = document.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("transparency chain is empty")
    try:
        public_key = Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(str(document["public_key_ed25519_hex"]))
        )
    except (KeyError, ValueError) as error:
        raise ValueError("invalid transparency public key") from error

    previous = GENESIS
    first_disclosed: int | None = None
    disclosed = 0
    for index, entry in enumerate(entries):
        if entry.get("seq") != index:
            raise ValueError(f"non-contiguous transparency sequence at index {index}")
        if entry.get("prev_chain_hash") != previous:
            raise ValueError(f"transparency link mismatch at seq {index}")
        signed_fields = (
            f"{entry['prev_chain_hash']}|{entry['payload_sha256']}|"
            f"{entry['date']}|{entry['seq']}"
        )
        expected_chain_hash = sha256_hex(signed_fields.encode())
        if entry.get("chain_hash") != expected_chain_hash:
            raise ValueError(f"transparency chain hash mismatch at seq {index}")
        try:
            public_key.verify(
                bytes.fromhex(str(entry["signature"])),
                bytes.fromhex(expected_chain_hash),
            )
        except (InvalidSignature, ValueError, KeyError) as error:
            raise ValueError(f"transparency signature mismatch at seq {index}") from error
        if "payload" in entry:
            if entry.get("payload_schema") != PAYLOAD_SCHEMA:
                raise ValueError(f"transparency payload schema mismatch at seq {index}")
            if sha256_hex(canonical_json(entry["payload"])) != entry["payload_sha256"]:
                raise ValueError(f"transparency payload hash mismatch at seq {index}")
            if first_disclosed is None:
                first_disclosed = index
            disclosed += 1
        elif first_disclosed is not None:
            raise ValueError(f"opaque transparency entry after disclosure at seq {index}")
        previous = expected_chain_hash

    disclosure = document.get("payload_disclosure")
    if not isinstance(disclosure, dict) or first_disclosed is None:
        raise ValueError("transparency payload-disclosure boundary missing")
    expected_contract = {
        "payload_schema": PAYLOAD_SCHEMA,
        "first_disclosed_seq": first_disclosed,
        "disclosed_entries": disclosed,
        "opaque_historical_entries": len(entries) - disclosed,
    }
    for field, expected in expected_contract.items():
        if disclosure.get(field) != expected:
            raise ValueError(f"transparency disclosure field mismatch: {field}")

    head = entries[-1]
    if expected_state is not None:
        expected_payload = daily_track_record_payload(expected_state)
        if head.get("payload") != expected_payload:
            raise ValueError("transparency head payload does not equal evaluated paper state")
        if head.get("payload_sha256") != sha256_hex(canonical_json(expected_payload)):
            raise ValueError("transparency head hash does not bind evaluated paper state")

    return {
        "entries": len(entries),
        "head_seq": head["seq"],
        "head_chain_hash": head["chain_hash"],
        "first_disclosed_seq": first_disclosed,
        "disclosed_entries": disclosed,
        "opaque_historical_entries": len(entries) - disclosed,
        "head_payload_matches_state": expected_state is not None,
    }
