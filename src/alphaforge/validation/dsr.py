"""Probabilistic and Deflated Sharpe Ratio (alphaDesign.md section 7.4).

Bailey and Lopez de Prado's anti-overfit Sharpe instruments. They JUDGE whether
an observed Sharpe is real once you account for (a) the finite, non-Gaussian
return sample that produced it and (b) the number of strategy configurations that
were tried before this one was selected. Neither function is gated on a positive
edge; they are measurement tools.

Periodicity convention (load-bearing)
--------------------------------------
Every Sharpe quantity in this module is a PER-PERIOD Sharpe, with no annualization
factor of its own: the ``sqrt(n_obs - 1)`` term in the PSR already converts the
per-period Sharpe into a standard-error-scaled statistic over ``n_obs`` periods.
Our analytics harness (:mod:`alphaforge.analytics.metrics`) instead reports an
ANNUALIZED Sharpe computed from UTC-DAILY returns. To bridge the two,
:func:`dsr_from_returns` takes the daily return series directly and derives the
per-period Sharpe, sample skewness, non-excess kurtosis, and ``n`` internally, so
callers never hand a mismatched (annualized SR, per-period n) pair to the maths.

Exact formulas (alphaDesign.md section 7.4)
-------------------------------------------
Probabilistic Sharpe Ratio of observed per-period Sharpe ``SR`` against a
benchmark ``SR*`` over ``T`` observations with sample skewness ``g3`` and sample
(non-excess) kurtosis ``g4`` (``g4 = 3`` for a Gaussian)::

    PSR(SR*) = Phi( ( (SR - SR*) * sqrt(T - 1) )
                    / sqrt( 1 - g3*SR + ((g4 - 1)/4)*SR^2 ) )

Deflated Sharpe Ratio sets ``SR*`` to the expected MAXIMUM Sharpe among ``N``
effectively-independent trials with cross-trial Sharpe variance ``V[SR]``::

    SR* = sqrt(V[SR]) * ( (1 - gamma)*PPF(1 - 1/N) + gamma*PPF(1 - 1/(N*e)) )
    gamma = 0.5772156649  (Euler-Mascheroni),  e = exp(1)

``DSR = PSR(SR*)``. Phi is the standard-normal CDF, PPF its inverse (quantile);
both via :mod:`scipy.stats.norm`. Deployment gate (applied by the caller, not
here): DSR >= 0.95, equivalently PSR against the deflated benchmark.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import norm
from scipy.stats import skew as _skew

from alphaforge.analytics.metrics import DAYS_PER_YEAR

__all__ = [
    "EULER_MASCHERONI",
    "DSRReport",
    "deflated_sharpe_ratio",
    "dsr_from_returns",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
]

EULER_MASCHERONI: float = 0.5772156649
"""Euler-Mascheroni constant gamma, as fixed by alphaDesign.md section 7.4."""

_E: float = math.e


def probabilistic_sharpe_ratio(
    sr: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    sr_benchmark: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio PSR(``sr_benchmark``) (alphaDesign.md section 7.4).

    All Sharpe arguments are PER-PERIOD (same periodicity as the returns that
    produced ``n_obs``); ``kurtosis`` is NON-EXCESS (3 for a Gaussian)::

        PSR = Phi( ( (sr - sr_benchmark) * sqrt(n_obs - 1) )
                   / sqrt( 1 - skew*sr + ((kurtosis - 1)/4)*sr^2 ) )

    The numerator measures the per-period Sharpe edge over the benchmark scaled by
    the sample length; the denominator is the standard deviation of the Sharpe
    estimator under non-normality (Mertens / Bailey-Lopez de Prado). For a
    Gaussian (skew=0, kurtosis=3) the denominator collapses to
    ``sqrt(1 + sr^2/2)`` and, for small ``sr``, to ``1`` so that
    ``PSR -> Phi(sr * sqrt(n_obs - 1))``.

    Returns a probability in ``(0, 1)``. Raises ``ValueError`` for ``n_obs < 2``
    or for a non-positive variance term in the denominator (a degenerate /
    impossible moment combination, e.g. ``1 - skew*sr + ((kurtosis-1)/4)*sr^2 <= 0``).
    """
    if n_obs < 2:
        raise ValueError(f"n_obs must be >= 2; got {n_obs}")
    variance_term = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr * sr
    if variance_term <= 0.0:
        raise ValueError(
            "non-positive PSR denominator variance term "
            f"(1 - skew*sr + ((kurtosis-1)/4)*sr^2 = {variance_term}); "
            "skew/kurtosis are inconsistent with this Sharpe"
        )
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1.0) / math.sqrt(variance_term)
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_trials_variance: float) -> float:
    """Expected maximum Sharpe under the null from ``n_trials`` trials.

    The benchmark used by :func:`deflated_sharpe_ratio`. With cross-trial Sharpe
    variance ``V[SR] = sr_trials_variance`` and ``gamma`` the Euler-Mascheroni
    constant (alphaDesign.md section 7.4)::

        SR* = sqrt(V[SR]) * ( (1 - gamma)*PPF(1 - 1/N) + gamma*PPF(1 - 1/(N*e)) )

    This is the Bailey-Lopez de Prado closed form for the expected maximum of
    ``N`` i.i.d. standard-normal Sharpe estimates scaled by their dispersion; it
    grows (weakly) with ``N`` because the more independent trials you run, the
    larger the best Sharpe you expect to see by luck alone.

    Requires ``n_trials >= 2`` (with a single trial there is no selection bias and
    the expected maximum is undefined under this asymptotic form) and
    ``sr_trials_variance >= 0``. Returns 0.0 exactly when the variance is 0
    (all trials identical -> no dispersion to exploit).
    """
    if n_trials < 2:
        raise ValueError(f"n_trials must be >= 2; got {n_trials}")
    if sr_trials_variance < 0.0:
        raise ValueError(f"sr_trials_variance must be >= 0; got {sr_trials_variance}")
    if sr_trials_variance == 0.0:
        return 0.0
    n = float(n_trials)
    quantile = (1.0 - EULER_MASCHERONI) * float(norm.ppf(1.0 - 1.0 / n)) + EULER_MASCHERONI * float(
        norm.ppf(1.0 - 1.0 / (n * _E))
    )
    return math.sqrt(sr_trials_variance) * quantile


def deflated_sharpe_ratio(
    sr: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    n_trials: int,
    sr_trials_variance: float,
) -> float:
    """Deflated Sharpe Ratio = PSR against the expected-max-Sharpe benchmark.

    ``DSR = PSR(SR*)`` where ``SR* = expected_max_sharpe(n_trials,
    sr_trials_variance)`` (alphaDesign.md section 7.4). It is the probability that
    the observed per-period Sharpe ``sr`` exceeds what the BEST of ``n_trials``
    independent random trials would produce under the null, given the sample
    length ``n_obs`` and the return moments (``kurtosis`` non-excess).

    A high-Sharpe-but-heavily-searched configuration deflates toward (or below)
    0.5: the more trials, the higher ``SR*``, the lower the DSR for a fixed ``sr``.
    Deployment gate (caller-applied): DSR >= 0.95.

    Raises ``ValueError`` via the underlying helpers for invalid ``n_obs``,
    ``n_trials``, ``sr_trials_variance``, or a degenerate PSR denominator.
    """
    sr_benchmark = expected_max_sharpe(n_trials, sr_trials_variance)
    return probabilistic_sharpe_ratio(
        sr=sr,
        n_obs=n_obs,
        skew=skew,
        kurtosis=kurtosis,
        sr_benchmark=sr_benchmark,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class DSRReport:
    """Result of :func:`dsr_from_returns` (alphaDesign.md section 7.4).

    - ``psr``: PSR against a zero per-period benchmark (probability the true
      Sharpe is positive given the sample).
    - ``dsr``: Deflated Sharpe Ratio (PSR against ``expected_max_sr``).
    - ``sr_ann``: annualized Sharpe (per-period SR x sqrt(periods_per_year)) for
      human reporting only; never fed back into the PSR maths.
    - ``sr_per_period``: the per-period Sharpe actually used in the formulas.
    - ``skew`` / ``kurtosis``: sample skewness and NON-excess kurtosis of the
      per-period returns (kurtosis = 3 for a Gaussian).
    - ``n_obs``: number of per-period return observations.
    - ``expected_max_sr``: the deflation benchmark ``SR*`` (per period).
    """

    psr: float
    dsr: float
    sr_ann: float
    sr_per_period: float
    skew: float
    kurtosis: float
    n_obs: int
    expected_max_sr: float


def _per_period_moments(
    returns: npt.NDArray[np.float64],
) -> tuple[float, float, float, int]:
    """Per-period Sharpe, sample skewness, non-excess kurtosis, and n.

    Sharpe uses ddof=1 (sample) std to match
    :func:`alphaforge.analytics.metrics.sharpe`; skew/kurtosis are the biased
    (population) moment estimators (``bias=True``), the standard Bailey-Lopez de
    Prado convention, with kurtosis reported non-excess (``fisher=False``).

    A constant series (all returns identical) has zero variance and undefined
    Sharpe/skew/kurtosis; it is rejected up front via the exact peak-to-peak test
    (``np.std`` of identical floats leaks a tiny non-zero residual from the mean
    subtraction, so an ``== 0`` std test alone would not catch it).
    """
    n = int(returns.size)
    if n < 2:
        raise ValueError(f"need >= 2 return observations; got {n}")
    if float(np.ptp(returns)) == 0.0:
        raise ValueError("return series has zero variance; Sharpe is undefined")
    std = float(np.std(returns, ddof=1))
    if std == 0.0:  # pragma: no cover - ptp guard above already caught it
        raise ValueError("return series has zero variance; Sharpe is undefined")
    sr = float(np.mean(returns)) / std
    g3 = float(_skew(returns, bias=True))
    g4 = float(_kurtosis(returns, fisher=False, bias=True))
    return sr, g3, g4, n


def dsr_from_returns(
    daily_returns: pd.Series,
    n_trials: int,
    sr_trials_variance: float,
    periods_per_year: float = DAYS_PER_YEAR,
) -> DSRReport:
    """PSR + DSR from a DAILY return series, deriving every moment internally.

    ``daily_returns`` is the per-period (UTC-daily) simple return series, e.g. the
    output of :func:`alphaforge.analytics.metrics.daily_returns`. This is the safe
    entry point: it computes the PER-PERIOD Sharpe, sample skewness, non-excess
    kurtosis, and ``n`` from the series itself, so the per-period convention of
    PSR/DSR is honoured automatically. ``sr_ann`` is reported separately
    (``sr_per_period * sqrt(periods_per_year)``, default ``periods_per_year`` =
    365, the 24/7 crypto day count) purely for human consumption.

    ``n_trials`` is the number of strategy configurations actually tried (measured
    by the experiment tracker, not guessed) and ``sr_trials_variance`` is the
    sample variance of their per-period Sharpe estimates (alphaDesign.md section
    7.4). Raises ``ValueError`` for fewer than 2 returns, a zero-variance series,
    or invalid trial arguments.
    """
    values = np.asarray(daily_returns.to_numpy(), dtype=np.float64)
    values = values[np.isfinite(values)]
    sr, g3, g4, n = _per_period_moments(values)
    expected_max_sr = expected_max_sharpe(n_trials, sr_trials_variance)
    psr = probabilistic_sharpe_ratio(sr=sr, n_obs=n, skew=g3, kurtosis=g4, sr_benchmark=0.0)
    dsr = probabilistic_sharpe_ratio(
        sr=sr, n_obs=n, skew=g3, kurtosis=g4, sr_benchmark=expected_max_sr
    )
    return DSRReport(
        psr=psr,
        dsr=dsr,
        sr_ann=sr * math.sqrt(periods_per_year),
        sr_per_period=sr,
        skew=g3,
        kurtosis=g4,
        n_obs=n,
        expected_max_sr=expected_max_sr,
    )
