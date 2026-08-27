"""Atomic private input snapshots for measured walk-forward results.

A result is not reproducible merely because its output curve has a checksum.  This
module freezes the *inputs actually crossing the execution boundary* before a run:

* the exact derived signal frame;
* overlapping point-in-time universe intervals;
* every SCD2 instrument version for the declared ids;
* the raw execution-data partitions for the run window;
* resolved settings and run parameters; and
* the complete local Python source tree plus environment lockfiles.

Raw lake partitions are hard-linked when possible.  The lake writer replaces whole
partition files atomically, so the link retains the old inode if the live lake is
later rebuilt; cross-filesystem snapshots fall back to a byte copy.  The snapshot is
assembled in a sibling temporary directory and promoted with one ``os.replace``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import pandas as pd
import pyarrow as pa

from alphaforge.core.time import Timeframe
from alphaforge.core.types import AssetClass
from alphaforge.data.schemas import Dataset, ohlcv_dataset

if TYPE_CHECKING:
    from alphaforge.core.instruments import Instrument, InstrumentStore
    from alphaforge.core.time import Ms
    from alphaforge.data.store.lake import LakePaths
    from alphaforge.data.universe.store import UniverseStore

__all__ = ["seal_walkforward_input_snapshot", "validate_input_snapshot"]

SCHEMA: Final = "canli.alphac-walkforward-input-snapshot.v1"
MANIFEST_NAME: Final = "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _content_hash(document: dict[str, Any]) -> str:
    body = {key: value for key, value in document.items() if key != "content_hash"}
    return f"sha256:{hashlib.sha256(_canonical(body)).hexdigest()}"


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _instrument_row(
    instrument: Instrument, *, valid_from_ms: int, valid_to_ms: int | None
) -> dict[str, Any]:
    row = _json_value(instrument)
    if not isinstance(row, dict):  # pragma: no cover - Instrument is a dataclass
        raise TypeError("instrument did not serialize to an object")
    return {**row, "valid_from_ms": valid_from_ms, "valid_to_ms": valid_to_ms}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_files(repo_root: Path) -> list[Path]:
    source_root = repo_root / "src" / "alphaforge"
    files = list(source_root.rglob("*.py"))
    files.extend(path for path in (repo_root / "configs").glob("*.yaml") if path.is_file())
    files.extend(
        path for path in (repo_root / "pyproject.toml", repo_root / "uv.lock") if path.is_file()
    )
    if not files:
        raise RuntimeError(f"no source files found under repository root: {repo_root}")
    return sorted(set(files))


def _copy_source_tree(repo_root: Path, destination: Path) -> dict[str, Any]:
    leaves = []
    for source in _source_files(repo_root):
        relative = source.relative_to(repo_root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        leaves.append(
            {
                "path": str(relative),
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
            }
        )
    return {
        "files": len(leaves),
        "bytes": sum(row["bytes"] for row in leaves),
        "root_sha256": hashlib.sha256(_canonical(leaves)).hexdigest(),
    }


def _execution_datasets(asset_class: AssetClass, timeframe: Timeframe) -> tuple[Dataset, ...]:
    datasets = [ohlcv_dataset(timeframe)]
    if asset_class is AssetClass.CRYPTO_PERP:
        datasets.append(Dataset.FUNDING)
    if asset_class is AssetClass.EQUITY:
        datasets.append(Dataset.CORPORATE_ACTIONS)
    return tuple(datasets)


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
        return "copy"
    return "hardlink"


def _freeze_raw_partitions(
    *,
    lake_paths: LakePaths,
    destination: Path,
    instrument_ids: list[str],
    start: Ms,
    end: Ms,
    asset_class: AssetClass,
    timeframe: Timeframe,
) -> dict[str, Any]:
    start_year = pd.Timestamp(start, unit="ms", tz="UTC").year
    end_year = pd.Timestamp(end - 1, unit="ms", tz="UTC").year
    rows = []
    modes: dict[str, int] = {}
    lake_root = lake_paths.root.resolve()
    for dataset in _execution_datasets(asset_class, timeframe):
        for instrument_id in instrument_ids:
            paths = lake_paths.partition_paths(
                dataset,
                instrument_id,
                year_min=start_year,
                year_max=end_year,
            )
            for source in paths:
                source = source.resolve()
                relative = source.relative_to(lake_root)
                target = destination / relative
                mode = _link_or_copy(source, target)
                modes[mode] = modes.get(mode, 0) + 1
                rows.append(
                    {
                        "dataset": dataset.value,
                        "instrument_id": instrument_id,
                        "snapshot_path": str(Path("raw_partitions") / relative),
                        "bytes": target.stat().st_size,
                        "sha256": _sha256(target),
                        "freeze_mode": mode,
                    }
                )
    rows.sort(key=lambda row: row["snapshot_path"])
    return {
        "datasets": [item.value for item in _execution_datasets(asset_class, timeframe)],
        "year_min": start_year,
        "year_max": end_year,
        "files": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "freeze_modes": modes,
        "root_sha256": hashlib.sha256(_canonical(rows)).hexdigest(),
        "leaves": rows,
    }


def _universe_snapshot(
    universe: UniverseStore,
    instrument_ids: list[str],
    start: Ms,
    end: Ms,
) -> pa.Table:
    table = universe.read_intervals()
    if table.num_rows == 0:
        return table
    ids = table.column("instrument_id").to_pylist()
    froms = table.column("effective_from").cast(pa.int64()).to_pylist()
    tos = table.column("effective_to").cast(pa.int64()).to_pylist()
    selected = set(instrument_ids)
    mask = pa.array(
        [
            str(iid) in selected
            and int(effective_from) < end
            and (effective_to is None or int(effective_to) > start)
            for iid, effective_from, effective_to in zip(ids, froms, tos, strict=True)
        ],
        type=pa.bool_(),
    )
    return table.filter(mask)


def _file_inventory(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if path.name == MANIFEST_NAME:
            continue
        rows.append(
            {
                "path": str(path.relative_to(directory)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def seal_walkforward_input_snapshot(
    destination: Path,
    *,
    signal_frame: pd.DataFrame,
    instrument_ids: list[str],
    universe: UniverseStore,
    instruments: InstrumentStore,
    lake_paths: LakePaths,
    start: Ms,
    end: Ms,
    timeframe: Timeframe,
    asset_class: AssetClass,
    declared_run: dict[str, Any],
    resolved_settings: dict[str, Any],
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Create an immutable private snapshot and return its result binding.

    ``destination`` must not exist.  The caller should invoke this after deriving
    ``signal_frame`` but before executing the first walk-forward leg.
    """
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite input snapshot: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        signal_path = temporary / "derived_signal_frame.parquet"
        signal_frame.to_parquet(signal_path, index=True)

        universe_path = temporary / "universe_intervals.parquet"
        intervals = _universe_snapshot(universe, instrument_ids, start, end)
        pd.DataFrame(intervals.to_pydict()).to_parquet(universe_path, index=False)

        metadata_rows: list[dict[str, Any]] = []
        for instrument_id in instrument_ids:
            history = instruments.history(instrument_id)
            if not history:
                raise RuntimeError(f"missing instrument metadata history: {instrument_id}")
            metadata_rows.extend(
                _instrument_row(item, valid_from_ms=valid_from, valid_to_ms=valid_to)
                for valid_from, valid_to, item in history
            )
        metadata_rows.sort(key=lambda row: (row["instrument_id"], row["valid_from_ms"]))
        _write_json(temporary / "instrument_metadata.json", metadata_rows)
        _write_json(
            temporary / "declared_run.json",
            {
                "declared_run": _json_value(declared_run),
                "resolved_settings": _json_value(resolved_settings),
            },
        )

        raw = _freeze_raw_partitions(
            lake_paths=lake_paths,
            destination=temporary / "raw_partitions",
            instrument_ids=instrument_ids,
            start=start,
            end=end,
            asset_class=asset_class,
            timeframe=timeframe,
        )
        source = _copy_source_tree(root, temporary / "source_environment")
        files = _file_inventory(temporary)
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "author": "Arhan Canli",
            "status": "SEALED_PRE_RUN_PRIVATE_INPUT_SNAPSHOT",
            "stage": "after derived-signal computation; before first execution leg",
            "scope": {
                "start_inclusive": int(start),
                "end_exclusive": int(end),
                "timeframe": timeframe.value,
                "asset_class": asset_class.value,
                "instrument_count": len(instrument_ids),
                "instrument_ids_sha256": hashlib.sha256(_canonical(instrument_ids)).hexdigest(),
                "signal_rows": len(signal_frame),
                "signal_columns": [str(column) for column in signal_frame.columns],
                "universe_interval_rows": intervals.num_rows,
                "instrument_metadata_rows": len(metadata_rows),
            },
            "raw_execution_partitions": raw,
            "source_environment": source,
            "files": files,
            "file_count": len(files),
            "snapshot_bytes": sum(row["bytes"] for row in files),
            "root_sha256": hashlib.sha256(_canonical(files)).hexdigest(),
            "data_rights": {
                "public_release_allowed": False,
                "policy": (
                    "Private reproducibility evidence. Publish hashes and aggregate metadata "
                    "only unless every underlying source grants redistribution rights."
                ),
            },
            "claim_boundary": (
                "This snapshot freezes the derived decisions and execution inputs for exact local "
                "replay. It is pre-run evidence and does not claim that an output completed, "
                "passed a research gate, or may be publicly redistributed."
            ),
        }
        manifest["content_hash"] = _content_hash(manifest)
        _write_json(temporary / MANIFEST_NAME, manifest)
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "path": str(Path(destination.name) / MANIFEST_NAME),
        "schema": manifest["schema"],
        "status": manifest["status"],
        "content_hash": manifest["content_hash"],
        "root_sha256": manifest["root_sha256"],
        "file_count": manifest["file_count"],
        "public_release_allowed": False,
    }


def validate_input_snapshot(destination: Path) -> dict[str, Any]:
    """Fail closed unless every file and both manifest hashes still reconcile."""
    destination = destination.resolve()
    manifest_path = destination / MANIFEST_NAME
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    if manifest.get("schema") != SCHEMA or manifest.get("content_hash") != _content_hash(manifest):
        raise RuntimeError(f"invalid input snapshot manifest: {manifest_path}")
    actual = _file_inventory(destination)
    if actual != manifest["files"]:
        raise RuntimeError(f"input snapshot file inventory drift: {destination}")
    root = hashlib.sha256(_canonical(actual)).hexdigest()
    if root != manifest["root_sha256"]:
        raise RuntimeError(f"input snapshot root hash drift: {destination}")
    return manifest
