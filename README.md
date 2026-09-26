# canli-backtest

**A backtester built so that the obvious ways to fool yourself raise an exception.**

[![ci](https://github.com/arhancanli/canli-backtest/actions/workflows/ci.yml/badge.svg)](https://github.com/arhancanli/canli-backtest/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)
[![parity: byte-identical to engine](https://img.shields.io/badge/parity-byte--identical-brightgreen.svg)](tools/check_parity.py)

A strategy decides at the close of bar `t` and fills at the open of bar `t+1`. Not sometimes —
structurally. The fill model is handed only the `t+1` bar, verifies that bar opened after the
decision, and raises `LookaheadError` if it did not. The engine re-checks every fill it gets back.
There is no code path where a decision touches its own bar's price.

This is the simulation and validation layer of [ALPHAC](https://github.com/arhancanli/alphac),
the research engine behind [canlicapital.com](https://canlicapital.com). Built and maintained by
[Arhan Canli](https://github.com/arhancanli).

---

## The two ways a backtest lies

### 1. It saw something it could not have seen

Look-ahead is usually not a dramatic bug. It is an off-by-one on a timestamp, a feature computed
over a window that includes the present, a flag derived from tomorrow's reversion. It never throws.
It just makes the curve go up.

The defence here is that the *shape of the API* forbids it:

- The strategy receives history through `StrategyContext`, whose `bars` accessor is a
  point-in-time read with `as_of` = the decision close. The engine slices; the strategy cannot ask
  for more. There is no handle on the future to grab.
- Fills are computed by a model that receives exactly one bar — the next one — and asserts
  `next_bar.ts_open >= order.decision_ts`.
- Funding cashflows are replayed from the **stored funding-events table**, each event carrying its
  own `ts_funding`. There is no funding clock in the engine, so 8h/4h/1h schedules are honoured
  automatically instead of being assumed.
- Splits and dividends are replayed from the point-in-time corporate-actions table before ex-date
  fills. Splits transform held *and queued* quantities; dividends accrue against the signed
  position entering the ex date.
- Every price and fee comes from one shared `TransactionCostModel`. The engine never computes a
  cost itself, so there is exactly one place where cost assumptions can be wrong.

```
P_fill = open_{t+1} · (1 + s · (half_spread + impact + latency)),   s = side.sign
fee    = fee_frac(taker) · qty · P_fill
```

The fee is charged on *executed* notional and travels on the `Fill` as a separate cash line, so
commission and price slippage stay separately attributable instead of blurring into one number.

### 2. It was the best of two hundred tries and only one got published

This is the harder lie, and the more common one. Run enough variants and something will look
brilliant by luck alone. The fix is not a better backtest — it is an honest denominator.

**Deflated Sharpe.** The Probabilistic Sharpe Ratio of an observed per-period Sharpe `SR` against
benchmark `SR*` over `T` observations, with sample skewness `g₃` and non-excess kurtosis `g₄`:

```
                    ⎛        (SR − SR*) · √(T − 1)          ⎞
PSR(SR*) =  Φ ⎜ ─────────────────────────────────── ⎟
                    ⎝  √( 1 − g₃·SR + ((g₄ − 1)/4)·SR² )     ⎠
```

The Deflated Sharpe Ratio is `PSR` evaluated at a benchmark `SR*` that is not zero but the
expected maximum Sharpe of `N` independent trials — that is, the level luck alone would reach
given how many things you tried. Every Sharpe in this module is **per-period**; the `√(T−1)` term
already carries the sample scaling, which is why `dsr_from_returns` takes the raw daily series and
derives `SR`, `g₃`, `g₄` and `T` internally. Handing it an annualized Sharpe with a per-period `n`
is the classic way to get a confident wrong answer, so the API does not let you.

**Probability of Backtest Overfitting**, by Combinatorially Symmetric Cross-Validation
(Bailey, Borwein, López de Prado, Zhu 2017). Given a `T × N` matrix of per-period returns:

1. Split the `T` rows into `S` contiguous equal blocks (`S` even, default 16); drop the remainder
   so every block is exactly `T // S` rows.
2. For each of the `C(S, S/2)` ways to pick half the blocks as in-sample, form the IS matrix from
   those and the OOS matrix from the rest.
3. Take the IS winner `n* = argmax_n SR_IS(n)`.
4. Find `n*`'s rank among all `N` configs **out of sample**, ranked ascending, and form the
   logit of its relative rank.

PBO is the fraction of splits where the in-sample winner lands in the bottom half out of sample.
If your best config is best because of noise, this number goes to 0.5 and stays there.

**The union trial ledger.** DSR is only honest if `N` counts every hypothesis you ever tested,
not the ones you remembered. The ledger here deduplicates identities across every durable research
ledger in the project, so which directory a trial happened to land in cannot change the correction.
Multiple-testing does not care about your filing conventions.

---

## Architecture

```
   strategy                                   ┌──────────────────────────────┐
      │  decide at close(t)                   │  StrategyContext             │
      │◄─────────────────────────────────────►│  bars → PIT read, as_of =    │
      │                                       │  decision close. No handle   │
      │  orders                               │  on the future exists.       │
      ▼                                       └──────────────────────────────┘
   ┌──────────────────────────────────────────────────────────────────────┐
   │  backtest/engine.py — EventDrivenBacktester                          │
   │                                                                       │
   │   for each bar close t_k on the union grid of all instruments:        │
   │     1. replay corporate actions with ex_date ≤ t_k  (PIT table)       │
   │     2. accrue funding from the stored events table                    │
   │     3. mark to market, apply financing and borrow                     │
   │     4. call strategy.on_bar(context)  ────────► queued orders         │
   │     5. hand queued orders to the fill model with ONLY bar t_k+1       │
   │     6. re-verify every returned fill  ─────────► LookaheadError       │
   └───────────────────────────────────┬──────────────────────────────────┘
                                       │  Fill(qty, price, fee_quote, ts)
                    ┌──────────────────▼───────────────────┐
                    │  backtest/ledger.py                   │
                    │  cash · positions · separate fee line  │
                    └──────────────────┬───────────────────┘
                                       │  equity curve (epoch-ms indexed)
   ┌───────────────────────────────────▼──────────────────────────────────┐
   │  validation/                                                          │
   │    dsr.py         PSR / DSR — is this Sharpe real given N tries?      │
   │    pbo.py         CSCV — does the IS winner survive OOS?              │
   │    splits.py      walk-forward and purged/embargoed splits            │
   │    experiments.py union trial ledger — the honest N                   │
   │    prereg.py      frozen hypothesis before the run, not after         │
   │    publish_gate.py fail-closed: no artifact, no publication           │
   └───────────────────────────────────────────────────────────────────────┘
```

Execution realism (`execution/`) covers borrow availability and cost, financing accrual, and
corporate-action replay. Market/venue state comes from
[canli-pit-lake](https://github.com/arhancanli/canli-pit-lake), which this package depends on for
every point-in-time read.

---

## Quick start

```bash
pip install canli-backtest
```

Two numbers from one series, which is the whole argument:

```python
import numpy as np, pandas as pd
from alphaforge.validation.dsr import dsr_from_returns

# 1,260 draws from a distribution with no edge in it. Five years of nothing.
noise = pd.Series(np.random.default_rng(20260828).normal(0.0004, 0.01, 1260))
report = dsr_from_returns(noise, n_trials=200, sr_trials_variance=0.04)

report.sr_ann   # 0.361  -- an annualised Sharpe you would put in a deck
report.psr      # 0.749  -- "75% odds it is real", if you tried this once
report.dsr      # 2.0e-80 -- the same series, told you tried 200 things first
```

Nothing about the data changes between those three lines. The only new
information in the last one is how many configurations were tried before this
one was chosen, and it is the difference between a result and an artefact.
`n_trials=1` is rejected rather than accepted quietly: a search of one is not a
search, and the correction is undefined there.

### Working on it, or checking it

```bash
git clone https://github.com/arhancanli/canli-backtest.git
cd canli-backtest
uv venv --python 3.12 && uv pip install -e . && uv pip install --group dev
uv run pytest
uv run python tools/check_parity.py   # prove this is the engine's code, byte for byte
```

Judging a Sharpe honestly:

```python
import pandas as pd
from alphaforge.validation.dsr import dsr_from_returns

daily = pd.Series(...)   # per-period (UTC-daily) simple returns

report = dsr_from_returns(
    daily,
    n_trials=200,             # every configuration tried before this one was chosen
    sr_trials_variance=0.04,  # variance of Sharpe ACROSS those trials
)
```

`n_trials` is the whole point: passing 1 when you tried 200 is the lie the instrument exists to
catch. `sr_trials_variance` is the spread of Sharpe across that trial family — a wide sweep sets a
higher bar than a narrow one, because a wide sweep gives luck more room. Note the function takes
the raw daily series, not a Sharpe: it derives the per-period `SR`, `g₃`, `g₄` and `T` itself, so
the per-period convention cannot be violated by a caller.

Measuring whether the winner is really a winner:

```python
from alphaforge.validation.pbo import pbo_cscv

# perf_matrix: T observations x N config variants, on a common time grid
result = pbo_cscv(perf_matrix, n_splits=16)
# result.pbo -> 0.5 means the in-sample ranking carries no out-of-sample information
```

---

## What is inside

| package | what it owns |
|---|---|
| `alphaforge.backtest` | the event-driven engine, fill models, ledger, result types |
| `alphaforge.validation` | DSR/PSR, PBO via CSCV, walk-forward splits, the union trial ledger, pre-registration, the fail-closed publish gate, transparency chain |
| `alphaforge.costs` | the single transaction-cost model every price and fee comes from |
| `alphaforge.execution` | borrow, financing accrual, corporate-action replay |
| `alphaforge.features.library` | the volatility primitives the engine needs |
| `alphaforge.analytics` | per-period and annualized metrics, tearsheets |

Fully typed (`mypy --strict`). Point-in-time reads come from
[`canli-pit-lake`](https://github.com/arhancanli/canli-pit-lake).

---

## Benchmarks

Measured, not estimated. Every number comes from
[`benchmarks/validation_throughput.py`](benchmarks/validation_throughput.py):

```bash
uv run python benchmarks/validation_throughput.py
```

| operation | cost |
|---|---|
| PBO via CSCV, 2,000 observations × 100 configs, S=16, 5,000 sampled splits | **1.6 s** |
| DSR, 1,260 daily returns | **~300 µs** per call |

Apple silicon (arm64), Python 3.12.13, single process. `C(16, 8)` is 12,870 splits; the
implementation samples 5,000 of them with a fixed seed rather than enumerating all — the number
above is that cap, not the full space.

The point of these numbers is not that they are fast. It is that they are cheap enough that
running the honest test on **every** candidate, not just the one you want to publish, costs
nothing worth saving. 289 µs is not a reason to skip deflation.

### The benchmark is also a correctness check

A speed benchmark can be fast and wrong. So both cases assert behaviour:

- The PBO case is built adversarially: 100 configs of **pure noise**, so the in-sample winner
  always won by luck. A correct CSCV must land near 0.5 — the measured value is **0.569**. If the
  implementation ever started finding signal in noise, that number would move before the timing did.
- The DSR case holds one return series fixed and moves only the trial count. Annualized Sharpe
  **+1.139**, and a Probabilistic Sharpe against zero of **0.9828**: the kind of number that gets
  published. Deflated against a trial family whose per-period Sharpes scatter with sd 0.02:

  | trials N | expected max Sharpe | DSR |
  |---:|---:|---:|
  | 2 | 0.01040 | **0.9596** |
  | 10 | 0.03149 | 0.8408 |
  | 50 | 0.04553 | 0.6914 |
  | 200 | 0.05531 | 0.5608 |
  | 1000 | 0.06510 | 0.4230 |

  Nothing about the strategy changed between those rows. Against the project's 0.95 deployment
  gate, that Sharpe is admissible if you tried two things and inadmissible if you tried ten.

  **A correction, since this README claimed otherwise on the day it was published.** The first
  version used `sr_trials_variance=0.04`, and reported that the same Sharpe deflated to 0.000 at
  200 trials. The arithmetic was right and the attribution was wrong: 0.04 is a per-period Sharpe
  variance of sd 0.2, roughly twenty times this series' own per-period Sharpe, and at that scale
  the ratio collapses at N=2 and the trial count stops mattering at all. It was the variance
  argument doing the work, not the trial count, while the sentence credited the trial count. The
  benchmark now uses a defensible scale and prints the sweep instead of one figure.

### What is verified

| check | result |
|---|---|
| tests | **279 passed** |
| types | `mypy --strict`, **0 issues** across 42 source files |
| lint | `ruff`, **clean** |
| parity with the engine | **118 files byte-identical** |

---

## Provenance

This repository is a **mechanical extraction**, not a fork.

`extraction_manifest.json` records the SHA-256 of every file at extraction time.
`tools/check_parity.py` re-reads each one from the engine repository at the pinned commit and
fails if a single byte differs. It runs in CI on every push, and is mutation-tested three ways:
a drifted file, a tampered manifest entry, and a deleted file each turn it red.

The module set is the transitive import closure of `alphaforge.backtest`, `alphaforge.validation`
and `alphaforge.costs`, minus everything already published in `canli-pit-lake` — computed, not
hand-listed. The test set is every engine test whose imports that closure satisfies.
The extraction ships only files the engine actually **publishes** — its `git ls-files` set, not
its working tree. That rule exists because the first version of this repo did not have it and
shipped 70 files the engine deliberately gitignores; the parity check against GitHub caught them,
a local check never would have.

Tests that read the private research corpus cannot run here; they are **removed rather than
skipped**, and each one is listed with its reason in
[`excluded_tests.json`](excluded_tests.json), so the suite has no silently-passing holes.

---

## Related

- **[alphac](https://github.com/arhancanli/alphac)** — the full research engine this comes from
- **[canli-pit-lake](https://github.com/arhancanli/canli-pit-lake)** — the point-in-time data
  layer this reads
- **[canlicapital.com](https://canlicapital.com)** — the live paper record these produce

## License and citation

MIT. Copyright © 2026 Arhan Canli. Machine-readable citation metadata is in
[`CITATION.cff`](CITATION.cff).

Created and maintained by **[Arhan Canli](https://github.com/arhancanli)**. Ownership, design
decisions, and published claims are mine.
