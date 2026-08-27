"""CPCV per-path V[SR] — the honester within-path Sharpe variance (alphaDesign.md §7.4).

The Deflated Sharpe Ratio's ``V[SR]`` is meant to be the dispersion of a config's own
CombinatorialPurgedCV backtest PATHS (cpcv.py ``n_backtest_paths``), not the spread of
mean Sharpe estimates ACROSS configs. This module pins the upgraded
:meth:`~alphaforge.validation.experiments.ExperimentLog.trial_sharpe_variance`:

* when ANY ledger record carries ``sharpe_per_path``, the variance is the ddof=1
  variance over the FLATTENED pool of every record's per-path Sharpes (== ``numpy.var``
  with ``ddof=1`` on that pool);
* when NO record has path Sharpes, it FALLS BACK to the legacy between-config variance
  of the scalar ``sharpe_per_period`` — so today's 77-record path-less ledger is
  unaffected;
* the new ``sharpe_per_path`` field round-trips through the JSONL schema and is ABSENT
  from a legacy record's JSON (backwards-compatibility).

Everything is offline, deterministic, ``tmp_path`` ledgers only.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from alphaforge.validation.experiments import (
    DEFAULT_SR_TRIALS_VARIANCE,
    ExperimentLog,
    ExperimentRecord,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def _record_paths(
    log: ExperimentLog,
    config: dict[str, object],
    paths: Sequence[float],
    now_ms: int,
) -> None:
    """Record a trial carrying per-CPCV-path Sharpes (the mean stands in for the scalar)."""
    log.record(
        config,
        sharpe_ann=float(np.mean(paths)) * math.sqrt(365.0),
        sharpe_per_period=float(np.mean(paths)),
        n_obs=200,
        skew=-0.3,
        kurtosis=4.0,
        now_ms=now_ms,
        path_sharpes=list(paths),
    )


class TestWithinPathVariance:
    def test_variance_equals_numpy_var_ddof1_over_flattened_paths(self, tmp_path: Path) -> None:
        log = ExperimentLog(tmp_path / "e.jsonl")
        # Two trials, nine CPCV paths each (the N=10,k=2 default phi).
        a = [0.02, 0.05, -0.01, 0.03, 0.00, 0.04, -0.02, 0.06, 0.01]
        b = [0.10, 0.08, 0.12, 0.07, 0.09, 0.11, 0.06, 0.13, 0.05]
        _record_paths(log, {"cfg": "a"}, a, 1)
        _record_paths(log, {"cfg": "b"}, b, 2)
        flattened = np.asarray([*a, *b], dtype=float)
        expected = float(np.var(flattened, ddof=1))
        assert log.trial_sharpe_variance() == pytest.approx(expected, abs=1e-12)

    def test_single_trial_path_variance_is_within_path(self, tmp_path: Path) -> None:
        # One config but many CPCV paths -> a within-path spread IS defined (>= 2 finite
        # path Sharpes), so it does NOT fall through to the single-trial default.
        log = ExperimentLog(tmp_path / "e.jsonl")
        paths = [0.01, 0.04, -0.02, 0.05, 0.00]
        _record_paths(log, {"cfg": "solo"}, paths, 1)
        assert log.trial_sharpe_variance() == pytest.approx(
            float(np.var(np.asarray(paths), ddof=1)), abs=1e-12
        )

    def test_non_finite_path_sharpes_are_excluded(self, tmp_path: Path) -> None:
        log = ExperimentLog(tmp_path / "e.jsonl")
        paths = [0.02, math.nan, 0.06, math.inf, 0.04]
        _record_paths(log, {"cfg": "a"}, paths, 1)
        finite = np.asarray([0.02, 0.06, 0.04], dtype=float)
        assert log.trial_sharpe_variance() == pytest.approx(
            float(np.var(finite, ddof=1)), abs=1e-12
        )

    def test_fewer_than_two_finite_paths_returns_default(self, tmp_path: Path) -> None:
        log = ExperimentLog(tmp_path / "e.jsonl")
        _record_paths(log, {"cfg": "a"}, [0.03, math.nan, math.inf], 1)
        assert log.trial_sharpe_variance() == DEFAULT_SR_TRIALS_VARIANCE


class TestLegacyFallback:
    def test_no_path_records_use_between_config_variance(self, tmp_path: Path) -> None:
        # No record carries path Sharpes -> the legacy between-config spread, unchanged.
        log = ExperimentLog(tmp_path / "e.jsonl")
        for cfg, sr in (({"c": 1}, 0.1), ({"c": 2}, 0.3), ({"c": 3}, 0.5)):
            log.record(
                cfg,
                sharpe_ann=sr * math.sqrt(365.0),
                sharpe_per_period=sr,
                n_obs=200,
                skew=-0.3,
                kurtosis=4.0,
                now_ms=1,
            )
        # mean 0.3, ddof=1 var over [0.1, 0.3, 0.5] = 0.04 (the pre-CPCV behaviour).
        assert log.trial_sharpe_variance() == pytest.approx(0.04, abs=1e-12)

    def test_any_path_record_switches_to_within_path_pool(self, tmp_path: Path) -> None:
        # A mixed ledger: as soon as ONE record has path Sharpes the variance is taken
        # over the pooled path Sharpes of the path-carrying records only.
        log = ExperimentLog(tmp_path / "e.jsonl")
        # A legacy (path-less) record — contributes its scalar to NOTHING once paths exist.
        log.record(
            {"c": "legacy"},
            sharpe_ann=99.0,
            sharpe_per_period=9.9,
            n_obs=200,
            skew=0.0,
            kurtosis=3.0,
            now_ms=1,
        )
        paths = [0.02, 0.05, -0.01, 0.03, 0.04]
        _record_paths(log, {"c": "cpcv"}, paths, 2)
        # The huge legacy scalar (9.9) is excluded; only the CPCV path pool counts.
        assert log.trial_sharpe_variance() == pytest.approx(
            float(np.var(np.asarray(paths), ddof=1)), abs=1e-12
        )


class TestSchemaRoundTrip:
    def test_path_sharpes_round_trip_through_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "e.jsonl"
        log = ExperimentLog(path)
        paths = [0.02, math.nan, 0.06]
        _record_paths(log, {"cfg": "a"}, paths, 7)
        raw = json.loads(path.read_text(encoding="utf-8").strip())
        # nan element serializes as JSON null, the finite ones pass through.
        assert raw["sharpe_per_path"] == [0.02, None, 0.06]
        back = log.all()[0]
        assert back.sharpe_per_path is not None
        assert back.sharpe_per_path[0] == pytest.approx(0.02)
        assert math.isnan(back.sharpe_per_path[1])
        assert back.sharpe_per_path[2] == pytest.approx(0.06)

    def test_legacy_record_omits_path_field_and_reads_back_none(self, tmp_path: Path) -> None:
        path = tmp_path / "e.jsonl"
        log = ExperimentLog(path)
        log.record(
            {"cfg": "legacy"},
            sharpe_ann=1.0,
            sharpe_per_period=0.05,
            n_obs=200,
            skew=0.0,
            kurtosis=3.0,
            now_ms=1,
        )
        raw = json.loads(path.read_text(encoding="utf-8").strip())
        assert "sharpe_per_path" not in raw, "a path-less trial must not emit the new key (D7)"
        assert log.all()[0].sharpe_per_path is None

    def test_record_obj_inverse_with_paths(self) -> None:
        rec = ExperimentRecord(
            config_hash="abc",
            config={"a": 1},
            now_ms=7,
            sharpe_ann=1.0,
            sharpe_per_period=0.05,
            n_obs=10,
            skew=0.1,
            kurtosis=3.0,
            sharpe_per_path=[0.01, 0.02, 0.03],
        )
        back = ExperimentRecord.from_json_obj(rec.to_json_obj())
        assert back == rec
