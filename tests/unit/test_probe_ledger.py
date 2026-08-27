"""Tests for screen-stage trial registration.

The idempotence test is the load-bearing one. Probes get re-run all the time — to regenerate an
artifact, to reproduce a published number, to persist a curve that was thrown away. If a re-run
inflated N, then every act of *verifying our own work* would raise the deflation bar for the whole
book, and the honest response would be to verify less. That is exactly backwards, so the ledger
must key on the hypothesis and not on the execution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaforge.validation.experiments import ExperimentLog, ExperimentUnion
from alphaforge.validation.probe_ledger import record_probe_trial, selection_context

NOW = 1_780_000_000_000


def _returns(n: int = 400, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(0.0003, 0.007, n), index=pd.date_range("2023-01-01", periods=n, freq="D")
    )


def test_records_one_trial(tmp_path: Path) -> None:
    led = tmp_path / "experiments.jsonl"
    rec = record_probe_trial("demo", {"series": "EMPLOY", "sign": 1}, _returns(), now_ms=NOW,
                             ledger_path=led)
    assert ExperimentLog(led).n_trials() == 1
    assert rec.config_hash


def test_rerunning_the_same_probe_does_not_inflate_n(tmp_path: Path) -> None:
    """Verifying our own work must never raise the bar for the book."""
    led = tmp_path / "experiments.jsonl"
    cfg = {"series": "EMPLOY", "sign": 1}
    a = record_probe_trial("demo", cfg, _returns(), now_ms=NOW, ledger_path=led)
    b = record_probe_trial("demo", cfg, _returns(), now_ms=NOW + 86_400_000, ledger_path=led)
    assert a.config_hash == b.config_hash
    assert ExperimentLog(led).n_trials() == 1, "a re-run inflated N — the budget is now a lie"


def test_a_genuinely_different_hypothesis_is_a_new_trial(tmp_path: Path) -> None:
    led = tmp_path / "experiments.jsonl"
    record_probe_trial("demo", {"series": "EMPLOY", "sign": 1}, _returns(), now_ms=NOW,
                       ledger_path=led)
    record_probe_trial("demo", {"series": "EMPLOY", "sign": -1}, _returns(), now_ms=NOW,
                       ledger_path=led)
    assert ExperimentLog(led).n_trials() == 2, "flipping the declared sign is a different idea"


def test_two_probes_sharing_parameter_names_do_not_collide(tmp_path: Path) -> None:
    """The probe name is part of the hypothesis, so `sign=1` in two probes is two hypotheses."""
    led = tmp_path / "experiments.jsonl"
    record_probe_trial("probe_a", {"sign": 1}, _returns(), now_ms=NOW, ledger_path=led)
    record_probe_trial("probe_b", {"sign": 1}, _returns(), now_ms=NOW, ledger_path=led)
    assert ExperimentLog(led).n_trials() == 2


def test_sharpe_is_annualised_consistently(tmp_path: Path) -> None:
    led = tmp_path / "experiments.jsonl"
    r = _returns()
    rec = record_probe_trial("demo", {"k": 1}, r, now_ms=NOW, ledger_path=led)
    want = float(r.mean() / r.std(ddof=0) * np.sqrt(252))
    assert rec.sharpe_ann == pytest.approx(want, rel=1e-12)
    assert rec.n_obs == len(r)


def test_refuses_a_degenerate_series(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite returns"):
        record_probe_trial("demo", {"k": 1}, pd.Series([np.nan, np.nan]), now_ms=NOW,
                           ledger_path=tmp_path / "e.jsonl")


def test_selection_context_reads_union_identities_and_identity_variance(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "base.yaml").write_text("{}\n", encoding="utf-8")
    active = tmp_path / "var" / "experiments.jsonl"
    other = tmp_path / "var_macro" / "experiments.jsonl"
    record_probe_trial("active", {"k": 1}, _returns(seed=1), now_ms=NOW, ledger_path=active)
    record_probe_trial("macro", {"k": 1}, _returns(seed=2), now_ms=NOW + 1, ledger_path=other)

    n_hypotheses, variance = selection_context(active_path=active, root=tmp_path)
    assert n_hypotheses == 2
    assert variance > 0.0


def test_accepts_explicit_union_and_rejects_ambiguous_destination(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "base.yaml").write_text("{}\n", encoding="utf-8")
    active = tmp_path / "var" / "experiments.jsonl"
    union = ExperimentUnion.discover(active, tmp_path)

    record_probe_trial("demo", {"k": 1}, _returns(), now_ms=NOW, experiment_log=union)
    assert union.n_hypotheses() == 1
    with pytest.raises(ValueError, match="not both"):
        record_probe_trial(
            "demo",
            {"k": 1},
            _returns(),
            now_ms=NOW,
            ledger_path=active,
            experiment_log=union,
        )
