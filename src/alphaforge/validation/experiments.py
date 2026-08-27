"""Experiment tracking for the Deflated Sharpe Ratio's trial count (alphaDesign.md section 7.4).

The Deflated Sharpe Ratio inflates the benchmark Sharpe ``SR*`` by the expected
maximum Sharpe across ``N`` independent trials (alphaDesign.md section 7.4)::

    SR* = sqrt(V[SR]) * ( (1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N*e)) )

``N`` must be *measured*, not guessed: it is the number of strategy
configurations actually tried, and ``V[SR]`` is the sample variance of their
per-period Sharpe estimates. The design (section 7.4) mandates this be logged
automatically -- "every CPCV run appends its config hash to ``experiments.log``,
so N is measured, not guessed". This module is that log.

:class:`ExperimentLog` is an append-only JSONL ledger. Every walk-forward (or
CPCV) configuration that is ever evaluated records one line: a deterministic
hash of the *config that defines the trial* (alphas / rebalance / band /
allocator / window), the wall-clock timestamp **passed in by the caller** (the
clock is never read here -- determinism, leakageCritique discipline), and the
realized per-period and annualized Sharpe plus the daily-return moments.

The hash makes the log **idempotent**: re-evaluating an identical configuration
(a re-run, a retry, the same grid swept twice) does not double-count ``N`` -- a
config that is already on the ledger is skipped. This is the whole point: an
honest ``N`` counts *distinct* trials, so re-running a backtest 100 times cannot
silently inflate the deflation and make a weak edge look unbeatable.

JSON record schema (one object per line, sorted keys, ASCII)::

    config_hash         str    16-hex-char truncated SHA-256 of the canonical config
    config              obj    the canonical config that was hashed (audit trail)
    now_ms              int    caller-supplied wall-clock epoch ms (never read here)
    sharpe_ann          float  annualized Sharpe (human-facing diagnostic)
    sharpe_per_period   float  per-period Sharpe -- THE quantity DSR variance uses
    n_obs               int    number of (daily) return observations behind the Sharpe
    skew                float  sample skewness of the daily returns (gamma_3, DSR moment)
    kurtosis            float  sample kurtosis of the daily returns (gamma_4, non-excess)
    sharpe_per_path     list   OPTIONAL per-CPCV-path Sharpes (CombinatorialPurgedCV
                               ``n_backtest_paths`` of them) -- the HONESTER V[SR] source

CPCV per-path V[SR] (alphaDesign.md section 7.4, the de Prado research protocol)
--------------------------------------------------------------------------------
The legacy ``V[SR]`` is the between-CONFIG variance of the single ``sharpe_per_period``
per trial (the spread of the configs you searched). The design's stated intent is the
WITHIN-PATH variance: CPCV resamples one config's own history into
``phi = k*C(N,k)/N`` complete OOS backtest paths (cpcv.py ``n_backtest_paths``), and
``V[SR]`` is the dispersion of those per-path Sharpes. That spread is genuinely larger
than the between-config spread of mean estimates, so feeding it to the deflation
benchmark ``SR*`` makes DSR strictly MORE conservative -- the honester deflation.

``sharpe_per_path`` is the optional carrier for those per-path Sharpes. It is
``None`` (absent from the JSON) on every legacy record, so the ledger stays
backwards-compatible and :meth:`ExperimentLog.trial_sharpe_variance` falls back to the
between-config variance until path Sharpes are present (see that method).

Determinism: pure functions of inputs + the JSONL file; no clock, no RNG. The
file is created (with parents) on first :meth:`record`; reads of a missing file
return an empty log.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from alphaforge.core.time import Ms

__all__ = [
    "DEFAULT_LOG_PATH",
    "DEFAULT_SR_TRIALS_VARIANCE",
    "ExperimentLog",
    "ExperimentRecord",
    "ExperimentUnion",
    "config_hash",
    "hypothesis_hash",
]

DEFAULT_LOG_PATH: Final[Path] = Path("var/experiments.jsonl")
"""Default ledger location (``settings.paths.var_dir / "experiments.jsonl"`` in wiring)."""

DEFAULT_SR_TRIALS_VARIANCE: Final[float] = 1.0
"""Fallback ``V[SR]`` when fewer than 2 distinct trials exist.

With < 2 trials the sample variance of per-period Sharpes is undefined, but the
DSR benchmark ``SR*`` still needs a cross-trial SR variance. We default to 1.0:
the de Prado convention treating per-period Sharpe estimates as drawn from a
unit-variance population when no empirical spread is available yet. This is a
*conservative documented placeholder*, not a measurement -- a one-trial DSR is
necessarily provisional and the verdict surfaces ``n_trials`` so a reader knows
the deflation is not yet data-driven.
"""

_HASH_LEN: Final[int] = 16
_HYPOTHESIS_WINDOW_KEYS: Final[frozenset[str]] = frozenset({"start", "end"})
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]


def _is_canonical_return_ledger(path: Path) -> bool:
    """True for selection ledgers owned by this repository, false for test/scratch logs."""
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(_REPO_ROOT)
    except ValueError:
        return False
    return (
        resolved.name == "experiments.jsonl"
        and bool(relative.parts)
        and (relative.parts[0].startswith("var") or relative.parts[0] == "artifacts")
    )


def config_hash(config: Mapping[str, Any]) -> str:
    """Deterministic ``_HASH_LEN``-hex-char SHA-256 of a canonicalized config.

    The config is canonicalized with ``json.dumps(sort_keys=True,
    separators=(",", ":"))`` so key order, whitespace, and equivalent
    numeric/string spellings that round-trip through JSON do not change the
    hash. Two configs hash equal iff their canonical JSON is byte-identical;
    this is the identity used for idempotency in :meth:`ExperimentLog.record`.

    Raises ``TypeError`` if ``config`` is not JSON-serializable (a non-trivial
    object in the config is a caller bug -- trials must be described by plain
    data so the ledger is portable and reproducible).
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    return digest[:_HASH_LEN]


def hypothesis_hash(config: Mapping[str, object]) -> str:
    """Hash the economic hypothesis while excluding only its evaluation window.

    ``start`` and ``end`` describe when an existing idea was remeasured. Every other
    field remains identity-bearing, so parameter, universe, allocator, feature, and
    preregistration changes continue to spend distinct hypotheses.
    """
    payload = {k: v for k, v in config.items() if k not in _HYPOTHESIS_WINDOW_KEYS}
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:_HASH_LEN]


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentRecord:
    """One logged trial -- a single walk-forward/CPCV configuration's outcome.

    ``sharpe_per_period`` is THE field the DSR variance consumes; ``sharpe_ann``
    is the human-facing annualized figure. All Sharpe/moment fields may be
    non-finite (e.g. a degenerate single-observation leg) and round-trip through
    JSON as ``null`` <-> ``nan`` so the ledger never silently drops a trial.
    """

    config_hash: str
    config: dict[str, Any]
    now_ms: Ms
    sharpe_ann: float
    sharpe_per_period: float
    n_obs: int
    skew: float
    kurtosis: float
    sharpe_per_path: list[float] | None = None

    def to_json_obj(self) -> dict[str, Any]:
        """JSON-safe dict (non-finite floats -> ``None``); keys sorted on dump.

        ``sharpe_per_path`` is emitted ONLY when present, so a legacy record
        (path Sharpes never supplied) serializes byte-identically to the
        pre-CPCV schema -- the ledger stays backwards-compatible (D7-style:
        the new key appears only when it carries data). Non-finite per-path
        Sharpes map to ``null`` element-wise, mirroring the scalar fields.
        """
        obj: dict[str, Any] = {
            "config_hash": self.config_hash,
            "config": self.config,
            "now_ms": int(self.now_ms),
            "sharpe_ann": _json_float(self.sharpe_ann),
            "sharpe_per_period": _json_float(self.sharpe_per_period),
            "n_obs": int(self.n_obs),
            "skew": _json_float(self.skew),
            "kurtosis": _json_float(self.kurtosis),
        }
        if self.sharpe_per_path is not None:
            obj["sharpe_per_path"] = [_json_float(s) for s in self.sharpe_per_path]
        return obj

    @classmethod
    def from_json_obj(cls, obj: Mapping[str, Any]) -> ExperimentRecord:
        """Inverse of :meth:`to_json_obj` (``None`` -> ``nan``).

        A record missing ``sharpe_per_path`` (every legacy line) reads back with
        ``sharpe_per_path=None``; element ``null`` maps to ``nan``.
        """
        raw_paths = obj.get("sharpe_per_path")
        sharpe_per_path = None if raw_paths is None else [_from_json_float(s) for s in raw_paths]
        return cls(
            config_hash=str(obj["config_hash"]),
            config=dict(obj["config"]),
            now_ms=int(obj["now_ms"]),
            sharpe_ann=_from_json_float(obj["sharpe_ann"]),
            sharpe_per_period=_from_json_float(obj["sharpe_per_period"]),
            n_obs=int(obj["n_obs"]),
            skew=_from_json_float(obj["skew"]),
            kurtosis=_from_json_float(obj["kurtosis"]),
            sharpe_per_path=sharpe_per_path,
        )


def _json_float(value: float) -> float | None:
    """Map non-finite floats to ``None`` (JSON has no NaN/Inf in strict mode)."""
    return None if not math.isfinite(value) else float(value)


def _from_json_float(value: Any) -> float:
    """Map JSON ``null`` back to ``nan``; pass finite floats through."""
    return math.nan if value is None else float(value)


class ExperimentLog:
    """Append-only, idempotent JSONL ledger of evaluated trials (module docstring).

    Args:
        path: JSONL ledger file. Defaults to :data:`DEFAULT_LOG_PATH`
            (``var/experiments.jsonl``). Created with parents on first
            :meth:`record`; a missing file reads as an empty log.

    Thread/process-safety is intentionally NOT provided: the research workflow
    is single-writer (one walk-forward at a time appends its trial). The ledger
    is plain JSONL so it can be inspected, diffed, and hand-audited.
    """

    __slots__ = ("_path",)

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else DEFAULT_LOG_PATH

    @property
    def path(self) -> Path:
        """The ledger file this log reads/writes."""
        return self._path

    def all(self) -> list[ExperimentRecord]:
        """Every recorded trial in file (append) order; ``[]`` if the file is absent.

        Blank lines are skipped (tolerant of trailing newlines). A malformed
        line raises ``ValueError`` with the line number -- a corrupt ledger is a
        loud failure, never a silently-truncated ``N``.
        """
        if not self._path.exists():
            return []
        records: list[ExperimentRecord] = []
        text = self._path.read_text(encoding="utf-8")
        for lineno, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(ExperimentRecord.from_json_obj(obj))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"corrupt experiment ledger {self._path} at line {lineno}: {exc}"
                ) from exc
        return records

    def _hashes(self) -> set[str]:
        """The set of config hashes already on the ledger (idempotency check)."""
        return {rec.config_hash for rec in self.all()}

    def preflight_registration(
        self,
        config: Mapping[str, Any],
        *,
        reservation_path: Path | str | None = None,
        _known_hypothesis_elsewhere: bool = False,
    ) -> None:
        """Fail before return compute when a canonical new identity is not reserved."""
        existing_records = self.all()
        if config_hash(config) in {record.config_hash for record in existing_records}:
            return
        hypothesis = self._hypothesis_key(config)
        known_hypotheses = {self._hypothesis_key(record.config) for record in existing_records}
        if (
            hypothesis in known_hypotheses
            or _known_hypothesis_elsewhere
            or not _is_canonical_return_ledger(self._path)
        ):
            return
        if reservation_path is None:
            raise RuntimeError(
                "new return hypothesis blocked: a forward trial reservation is required"
            )
        reservation_file = Path(reservation_path)
        if not reservation_file.is_absolute():
            reservation_file = _REPO_ROOT / reservation_file
        if not reservation_file.is_file():
            raise RuntimeError(f"forward trial reservation is missing: {reservation_file}")
        from alphaforge.validation.trial_reservation import validate_reservation

        payload = json.loads(reservation_file.read_text(encoding="utf-8"))
        validate_reservation(payload, trial_config=dict(config), repo=_REPO_ROOT)

    def record(
        self,
        config: Mapping[str, Any],
        *,
        sharpe_ann: float,
        sharpe_per_period: float,
        n_obs: int,
        skew: float,
        kurtosis: float,
        now_ms: Ms,
        path_sharpes: Sequence[float] | None = None,
        reservation_path: Path | str | None = None,
        _known_hypothesis_elsewhere: bool = False,
    ) -> ExperimentRecord:
        """Append one trial; idempotent on the config hash.

        Hashes ``config`` (see :func:`config_hash`). If that hash is already on
        the ledger the call is a no-op for the file -- the *existing* record is
        returned unchanged, so re-running an identical configuration never
        inflates ``N`` (module docstring). Otherwise a new line is appended and
        the freshly-built record returned.

        ``path_sharpes`` is the OPTIONAL list of per-CPCV-path Sharpes for this
        trial (cpcv.py ``n_backtest_paths`` of them). When supplied it is stored
        on the record's ``sharpe_per_path`` field and feeds the within-path
        ``V[SR]`` in :meth:`trial_sharpe_variance`; when omitted (the legacy /
        single-walk-forward path) the record is byte-identical to today's schema
        and the variance falls back to the between-config form.

        ``now_ms`` is the caller's wall-clock timestamp; the clock is never read
        here (determinism). All float arguments may be non-finite.
        """
        h = config_hash(config)
        existing_records = self.all()
        for existing in existing_records:
            if existing.config_hash == h:
                return existing
        self.preflight_registration(
            config,
            reservation_path=reservation_path,
            _known_hypothesis_elsewhere=_known_hypothesis_elsewhere,
        )
        record = ExperimentRecord(
            config_hash=h,
            config=dict(config),
            now_ms=int(now_ms),
            sharpe_ann=float(sharpe_ann),
            sharpe_per_period=float(sharpe_per_period),
            n_obs=int(n_obs),
            skew=float(skew),
            kurtosis=float(kurtosis),
            sharpe_per_path=None if path_sharpes is None else [float(s) for s in path_sharpes],
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_json_obj(), sort_keys=True, ensure_ascii=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return record

    def n_trials(self) -> int:
        """Number of DISTINCT configurations on the ledger.

        Distinct by construction (idempotent :meth:`record`), but counted from
        the deduplicated hash set so a hand-edited ledger with duplicate hashes
        still yields an honest count.

        NOTE: this is the AUDIT count (every row ever evaluated). For the DSR ``N``
        use :meth:`n_hypotheses` -- see that docstring for why they differ.
        """
        return len(self._hashes())

    def _hypothesis_key(self, config: Mapping[str, object]) -> str:
        return hypothesis_hash(config)

    def n_hypotheses(self) -> int:
        """Number of distinct HYPOTHESES tried -- the honest DSR ``N``.

        WHY THIS IS NOT :meth:`n_trials` (found 2026-08-04, and it was drifting daily)
        -----------------------------------------------------------------------------
        The DSR deflates a Sharpe against the number of trials in the SELECTION SEARCH:
        how many distinct ideas were tried before this one was picked. The live system,
        however, re-evaluates its already-selected config every single day as the rolling
        window advances, and each of those lands on the ledger with a new config hash
        because ``start``/``end`` moved by 86,400,000 ms.

        Counting those as new trials is wrong in a way that compounds: ``N`` grows by ~365
        a year on calendar time alone, so the deflated Sharpe of a GOOD strategy decays
        toward zero simply because it stayed deployed. On this ledger it was already 117
        rows for 93 real hypotheses -- 24 rows of pure re-measurement.

        The exemption is deliberately narrow, and :meth:`window_only_reevaluations`
        enforces it: a row is only forgiven if it differs from an earlier row in NOTHING
        BUT the evaluation window. Change a parameter, an instrument, an allocator -- any
        field that is part of the hypothesis -- and it counts as a new trial. That is what
        keeps this a correction rather than a loophole: you cannot hide a parameter sweep
        behind it, because a sweep by definition changes something other than the dates.
        """
        return len({self._hypothesis_key(r.config) for r in self.all()})

    def hypothesis_records(self) -> list[ExperimentRecord]:
        """First immutable measurement for each hypothesis identity, in time order.

        Rolling-window records remain in :meth:`all` for audit and operations, but selection
        statistics must not drift merely because an already-selected strategy stayed live. The
        first record is the observation available when that identity entered the search.
        """
        first: dict[str, ExperimentRecord] = {}
        for record in sorted(self.all(), key=lambda item: (item.now_ms, item.config_hash)):
            first.setdefault(self._hypothesis_key(record.config), record)
        return list(first.values())

    def window_only_reevaluations(self) -> int:
        """Rows that are re-measurements of an existing hypothesis, not new trials.

        ``n_trials() - n_hypotheses()``. Published alongside ``N`` so the exemption is
        always visible and auditable: a reader can see exactly how many rows were forgiven
        and satisfy themselves that the count of real ideas was never quietly reduced.
        """
        return self.n_trials() - self.n_hypotheses()

    def trial_sharpe_variance(self) -> float:
        """Sample variance (ddof=1) of recorded per-period Sharpes -- DSR ``V[SR]``.

        Two regimes (the CPCV per-path V[SR] honester-deflation upgrade):

        * **Within-path (preferred).** If ANY record carries ``sharpe_per_path``
          (CPCV was run for at least one trial), the variance is taken over the
          FLATTENED pool of every record's per-path Sharpes -- the dispersion of
          a config's own resampled OOS backtest paths, which is the spread the
          design's ``SR*`` is meant to deflate against (module docstring). This
          pool is genuinely wider than the between-config spread, so DSR comes
          out strictly more conservative.
        * **Between-config (legacy fallback).** If NO record has path Sharpes,
          the variance is the existing ddof=1 spread of the scalar
          ``sharpe_per_period`` across distinct trials -- byte-identical to the
          pre-CPCV behaviour, so a legacy ledger (like today's 77 records) is
          unaffected.

        In BOTH regimes only **finite** Sharpes contribute (a degenerate trial /
        path logged ``nan`` cannot define a spread), and fewer than 2 finite
        values returns the documented conservative
        :data:`DEFAULT_SR_TRIALS_VARIANCE` placeholder. The variance is over
        *per-period* Sharpes because DSR's ``SR*`` and ``SR`` are both per-period
        (section 7.4).
        """
        return _sharpe_variance(self.all())

    def hypothesis_sharpe_variance(self) -> float:
        """DSR ``V[SR]`` over first-recorded hypothesis identities only.

        This is the variance companion to :meth:`n_hypotheses`. Using all rows here while using
        hypothesis identities for ``N`` lets rolling operational remeasurements change ``SR*``;
        that inconsistency is exactly what this method closes.
        """
        return _sharpe_variance(self.hypothesis_records())


class ExperimentUnion:
    """Append to one profile while reading every durable selection-evidence ledger.

    The active profile remains the sole write destination. Exact config hashes already present in
    any member are idempotent across the union. Audit rows and hypothesis identities are deduped
    deterministically, with the earliest immutable record representing each selection identity.
    Discovery includes top-level research profiles and durable experiment ledgers filed beneath
    ``artifacts/``; explicitly withdrawn archive paths never re-enter the selection denominator.
    """

    __slots__ = ("_active", "_members", "_root")

    def __init__(
        self,
        active_path: Path | str,
        member_paths: Sequence[Path | str],
        *,
        root: Path | str | None = None,
    ) -> None:
        self._active = ExperimentLog(active_path)
        self._root = None if root is None else Path(root).resolve()
        paths = {Path(path).resolve() for path in member_paths}
        paths.add(self._active.path.resolve())
        self._members = tuple(ExperimentLog(path) for path in sorted(paths))

    @classmethod
    def discover(cls, active_path: Path | str, root: Path | str) -> ExperimentUnion:
        """Discover durable experiment ledgers under a verified repository root.

        Top-level ``var*`` profile ledgers are the normal write locations. Some legacy campaigns
        filed their immutable ledgers with their result packets under ``artifacts/`` instead. Both
        locations are selection evidence and therefore count. Paths containing an archive marker
        are intentionally withdrawn evidence and remain excluded.
        """
        base = Path(root).resolve()
        if not (base / "configs" / "base.yaml").is_file():
            raise ValueError(f"experiment-union root has no configs/base.yaml: {base}")
        candidates = (
            *base.glob("var*/experiments.jsonl"),
            *base.glob("artifacts/**/experiments.jsonl"),
        )
        members = sorted(
            path
            for path in candidates
            if not any("archive" in part.casefold() for part in path.relative_to(base).parts)
        )
        return cls(active_path, members, root=base)

    @property
    def path(self) -> Path:
        """The sole append destination (compatibility with :class:`ExperimentLog`)."""
        return self._active.path

    @property
    def paths(self) -> tuple[Path, ...]:
        """Every union member in deterministic order."""
        return tuple(member.path for member in self._members)

    def all(self) -> list[ExperimentRecord]:
        """Config-hash-unique audit records across all profiles, earliest record wins."""
        first: dict[str, ExperimentRecord] = {}
        records = (record for member in self._members for record in member.all())
        for record in sorted(records, key=lambda item: (item.now_ms, item.config_hash)):
            first.setdefault(record.config_hash, record)
        return list(first.values())

    def record(
        self,
        config: Mapping[str, Any],
        *,
        sharpe_ann: float,
        sharpe_per_period: float,
        n_obs: int,
        skew: float,
        kurtosis: float,
        now_ms: Ms,
        path_sharpes: Sequence[float] | None = None,
        reservation_path: Path | str | None = None,
    ) -> ExperimentRecord:
        """Append locally unless the exact configuration already exists anywhere."""
        h = config_hash(config)
        records = self.all()
        for existing in records:
            if existing.config_hash == h:
                return existing
        hypothesis = self._active._hypothesis_key(config)
        known_hypotheses = {self._active._hypothesis_key(record.config) for record in records}
        if hypothesis not in known_hypotheses:
            pause_status = self._research_pause_status()
            if pause_status is not None:
                raise RuntimeError(
                    f"new return-hypothesis registration blocked by {pause_status}; "
                    "see config/trial_accounting.json"
                )
        return self._active.record(
            config,
            sharpe_ann=sharpe_ann,
            sharpe_per_period=sharpe_per_period,
            n_obs=n_obs,
            skew=skew,
            kurtosis=kurtosis,
            now_ms=now_ms,
            path_sharpes=path_sharpes,
            reservation_path=reservation_path,
            _known_hypothesis_elsewhere=hypothesis in known_hypotheses,
        )

    def preflight_registration(
        self,
        config: Mapping[str, Any],
        *,
        reservation_path: Path | str | None = None,
    ) -> None:
        """Apply pause and reservation gates before any return compute is spent."""
        records = self.all()
        if config_hash(config) in {record.config_hash for record in records}:
            return
        hypothesis = self._active._hypothesis_key(config)
        known_hypotheses = {self._active._hypothesis_key(record.config) for record in records}
        if hypothesis not in known_hypotheses:
            pause_status = self._research_pause_status()
            if pause_status is not None:
                raise RuntimeError(
                    f"new return-hypothesis registration blocked by {pause_status}; "
                    "see config/trial_accounting.json"
                )
        self._active.preflight_registration(
            config,
            reservation_path=reservation_path,
            _known_hypothesis_elsewhere=hypothesis in known_hypotheses,
        )

    def preflight_hypotheses(self, configs: Sequence[Mapping[str, Any]]) -> None:
        """Fail before computation when paused policy forbids any requested identity.

        Historical scripts can otherwise spend minutes rebuilding panels before discovering at
        append time that a return hypothesis is not authorized. This read-only preflight uses the
        exact same identity rule as :meth:`record`. An active campaign permits unknown identities;
        a paused campaign permits only exact hypotheses already represented in the union, including
        start/end-only operational remeasurements.
        """
        pause_status = self._research_pause_status()
        if pause_status is None:
            return
        known = {self._active._hypothesis_key(record.config) for record in self.all()}
        unknown = [
            config for config in configs if self._active._hypothesis_key(config) not in known
        ]
        if unknown:
            hashes = ", ".join(config_hash(config) for config in unknown[:5])
            suffix = "" if len(unknown) <= 5 else f", +{len(unknown) - 5} more"
            raise RuntimeError(
                f"{len(unknown)} unregistered return hypothesis/hypotheses blocked by "
                f"{pause_status} before computation (config hashes: {hashes}{suffix}); "
                "see config/trial_accounting.json"
            )

    def _research_pause_status(self) -> str | None:
        """Return the machine-enforced pause status, if this union has a policy root."""
        if self._root is None:
            return None
        policy_path = self._root / "config" / "trial_accounting.json"
        if not policy_path.is_file():
            return None
        try:
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"cannot read trial-accounting policy {policy_path}: {exc}") from exc
        status = str(policy.get("research_status", "ACTIVE"))
        return status if status.startswith("PAUSED_") else None

    def n_trials(self) -> int:
        """Config-hash-unique immutable execution records across the union."""
        return len(self.all())

    def hypothesis_records(self) -> list[ExperimentRecord]:
        """Earliest record for each union-wide hypothesis identity."""
        first: dict[str, ExperimentRecord] = {}
        for record in self.all():
            first.setdefault(self._active._hypothesis_key(record.config), record)
        return list(first.values())

    def n_hypotheses(self) -> int:
        """Union-wide DSR selection ``N``."""
        return len(self.hypothesis_records())

    def window_only_reevaluations(self) -> int:
        """Audit records excluded from union selection ``N``."""
        return self.n_trials() - self.n_hypotheses()

    def trial_sharpe_variance(self) -> float:
        """Raw-row variance retained for audit/backward compatibility."""
        return _sharpe_variance(self.all())

    def hypothesis_sharpe_variance(self) -> float:
        """Union-wide DSR variance aligned to :meth:`n_hypotheses`."""
        return _sharpe_variance(self.hypothesis_records())

    def hypothesis_selection_context(
        self,
        hypothesis_keys: Collection[str] | None = None,
    ) -> tuple[int, float]:
        """Return ``(N, V[SR])`` for the full union or an explicit frozen identity set."""
        records = self.hypothesis_records()
        if hypothesis_keys is None:
            return len(records), _sharpe_variance(records)
        requested = set(hypothesis_keys)
        by_key = {self._active._hypothesis_key(record.config): record for record in records}
        missing = requested - set(by_key)
        if missing:
            preview = ", ".join(sorted(missing)[:5])
            raise ValueError(f"selection context is missing frozen identities: {preview}")
        selected = [by_key[key] for key in sorted(requested)]
        return len(selected), _sharpe_variance(selected)


def _sharpe_variance(records: Sequence[ExperimentRecord]) -> float:
    """Variance implementation shared by raw-row and identity-level accounting."""
    if any(rec.sharpe_per_path is not None for rec in records):
        sharpes = [
            s
            for rec in records
            if rec.sharpe_per_path is not None
            for s in rec.sharpe_per_path
            if math.isfinite(s)
        ]
    else:
        sharpes = [rec.sharpe_per_period for rec in records if math.isfinite(rec.sharpe_per_period)]
    n = len(sharpes)
    if n < 2:
        return DEFAULT_SR_TRIALS_VARIANCE
    mean = math.fsum(sharpes) / n
    ss = math.fsum((x - mean) ** 2 for x in sharpes)
    return ss / (n - 1)
