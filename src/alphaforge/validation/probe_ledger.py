"""Register a SCREEN-STAGE probe's trial on the honest ledger.

THE HOLE THIS CLOSES. Multiple-testing deflation is this project's central honesty mechanism:
every hypothesis tested raises the bar that every sleeve in the book must clear, so the deflation
is only as truthful as the trial count behind it. That count has now been found short twice in
two days, both times in the flattering direction:

    2026-08-06  the count read ONE ledger while research ran under four var* profile dirs.
                N 101 -> 127 (undercounted 26%). Fixed by counting the union.
    2026-08-07  screen-stage probes (scripts/probe_*.py) write to artifacts/probe/ and never
                touch a ledger at all. At least TEN spent trials were invisible, including the
                five PIT macro series whose own pre-registration promised in writing that they
                "enter var/experiments.jsonl". N 128 -> >=138.

The reasoning is the same both times, and it is worth stating once so the next variant is
recognised on sight: **which directory a trial landed in, and which stage of the funnel it ran
at, are filing conventions. Multiple-testing correction does not care about filing conventions.**
A screen-stage probe is a real hypothesis, about the same markets, spent by the same researcher,
looking for sleeves for the same book. If it can dodge the budget by not being a walk-forward,
the budget is decorative.

IDEMPOTENCE IS THE POINT. :meth:`ExperimentLog.record` no-ops on a config hash it already holds,
so re-running a probe to persist an artifact (as `crypto_lowvol_720` and `cpi_surprise_size` both
were) does NOT inflate N. Only a genuinely new configuration does. That is what makes it safe to
call this unconditionally from every probe.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from alphaforge.validation.experiments import ExperimentLog, ExperimentUnion

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    import pandas as pd

    from alphaforge.core.time import Ms
    from alphaforge.validation.experiments import ExperimentRecord

__all__ = ["record_probe_trial", "selection_context"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def selection_context(
    *,
    active_path: Path | str | None = None,
    root: Path | str | None = None,
) -> tuple[int, float]:
    """Return identity-aligned union ``(N, V[SR])`` after a probe is recorded."""
    base = Path(root).resolve() if root is not None else _repo_root()
    active = Path(active_path) if active_path is not None else base / "var" / "experiments.jsonl"
    union = ExperimentUnion.discover(active, base)
    return union.n_hypotheses(), union.hypothesis_sharpe_variance()


def record_probe_trial(
    probe: str,
    config: Mapping[str, Any],
    returns: pd.Series,
    *,
    now_ms: Ms,
    periods_per_year: int = 252,
    prereg: str | None = None,
    ledger_path: Path | str | None = None,
    experiment_log: ExperimentLog | ExperimentUnion | None = None,
) -> ExperimentRecord:
    """Append this probe's hypothesis to the honest ledger; idempotent on its config.

    ``probe`` is the script's stem (e.g. ``"macro_vintage_family"``) and ``config`` the parameters
    that DEFINE the hypothesis — whatever a reasonable person would agree makes this a different
    idea rather than the same idea re-run. Both are folded into the hashed config, so two probes
    that happen to share parameter names cannot collide, and ``prereg`` (a path or commit) is
    recorded alongside so a trial can be traced to the document that declared it.

    ``returns`` are SIMPLE per-period returns, matching what probes compute and what
    :func:`alphaforge.analytics.curve_store.write_curve` persists.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        raise ValueError(f"{probe}: need >= 2 finite returns to record a trial, got {r.size}")
    sd = float(r.std(ddof=0))
    per_period = float(r.mean() / sd) if sd > 0 else 0.0
    centred = r - r.mean()
    m2 = float((centred**2).mean())
    skew = float((centred**3).mean() / m2**1.5) if m2 > 0 else 0.0
    kurt = float((centred**4).mean() / m2**2) if m2 > 0 else 0.0

    full_config: dict[str, Any] = {"probe": probe, **dict(config)}
    if prereg is not None:
        full_config["prereg"] = prereg

    if ledger_path is not None and experiment_log is not None:
        raise ValueError("pass ledger_path or experiment_log, not both")
    ledger: ExperimentLog | ExperimentUnion
    if experiment_log is not None:
        ledger = experiment_log
    elif ledger_path is None:
        base = _repo_root()
        ledger = ExperimentUnion.discover(base / "var" / "experiments.jsonl", base)
    else:
        ledger = ExperimentLog(ledger_path)
    return ledger.record(
        full_config,
        sharpe_ann=per_period * float(np.sqrt(periods_per_year)),
        sharpe_per_period=per_period,
        n_obs=int(r.size),
        skew=skew,
        kurtosis=kurt,
        now_ms=now_ms,
    )
