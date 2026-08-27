"""Measured cost of the anti-overfitting machinery.

Run it yourself -- the numbers in the README came from this file and nothing else:

    uv run python benchmarks/validation_throughput.py

Two things are timed:

* **CSCV / PBO.** The combinatorics are the expensive part: splitting T observations
  into S blocks and choosing half of them is C(S, S/2) = 12,870 ways at S=16. The
  implementation caps this at 5,000 uniformly-sampled seeded combinations by default
  rather than enumerating all of them, so the figure printed below is 5,000 -- the
  cap, not the full space. This measures whether the honest test is affordable
  enough that there is no excuse to skip it.
* **DSR.** Per-call cost of deflating a Sharpe, which decides whether you can afford
  to run it on every candidate rather than only on the winner.

The PBO case is built adversarially on purpose: the configs are pure noise, so the
in-sample winner is always luck. A correct implementation must return a PBO near 0.5.
A benchmark that only measured speed could pass while computing nonsense, so the
measured PBO is printed beside the timing.
"""

from __future__ import annotations

import platform
import statistics
import time

import numpy as np
import pandas as pd

from alphaforge.validation.dsr import dsr_from_returns
from alphaforge.validation.pbo import pbo_cscv

T, N, S = 2_000, 100, 16
rng = np.random.default_rng(20260827)

# Pure noise: no config has any edge, so whichever wins in-sample won by luck.
perf = pd.DataFrame(rng.normal(0.0, 0.01, size=(T, N)))

runs = []
for _ in range(3):
    t0 = time.perf_counter()
    result = pbo_cscv(perf, n_splits=S)
    runs.append(time.perf_counter() - t0)
best = min(runs)
n_splits_evaluated = result.n_combinations
print(
    f"PBO / CSCV   T={T:,} x N={N} configs, S={S} -> {n_splits_evaluated:,} splits "
    f"in {best:5.2f}s   (median {statistics.median(runs):.2f}s of 3)"
)
print(f"             measured PBO on pure noise = {result.pbo:.3f}   (0.5 is correct here)")

daily = pd.Series(rng.normal(0.0006, 0.011, size=1_260))  # ~5 years of daily returns

# sr_trials_variance is the variance of Sharpe ACROSS the trial family, in the same
# PER-PERIOD units as the series. Getting its scale wrong makes the whole exercise
# meaningless: at V=0.04 (a per-period Sharpe sd of 0.2, twenty times this series'
# own per-period Sharpe) the deflated ratio collapses at N=2 and the trial count
# stops mattering at all. A per-period sd of 0.02 across a parameter sweep is a
# defensible scale, and it is what lets N be the thing under study.
TRIAL_SD = 0.02
TRIAL_VARIANCE = TRIAL_SD**2

runs = []
for _ in range(200):
    t0 = time.perf_counter()
    report = dsr_from_returns(daily, n_trials=200, sr_trials_variance=TRIAL_VARIANCE)
    runs.append(time.perf_counter() - t0)
print(
    f"DSR          1,260 daily returns, n_trials=200 -> {statistics.median(runs) * 1e6:,.0f} us "
    f"per call (median of 200)"
)
print(
    f"             annualised SR {report.sr_ann:+.3f}, PSR against zero "
    f"{report.psr:.4f} (that is the number without deflation)"
)
# The point of the instrument, shown rather than asserted: one unchanged return
# series, and the only thing that moves is how many things you admit to trying.
print(f"             same series, trial sd {TRIAL_SD}, deflated against N trials:")
for n_trials in (2, 10, 50, 200, 1000):
    row = dsr_from_returns(daily, n_trials=n_trials, sr_trials_variance=TRIAL_VARIANCE)
    print(f"               N={n_trials:<5} SR*={row.expected_max_sr:.5f}  DSR={row.dsr:.4f}")
print(f"machine      {platform.machine()} - python {platform.python_version()}")
