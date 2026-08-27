"""P&L concentration screening — the guard that catches an edge made of bad prints.

WHY THIS EXISTS. On 2026-08-10 a zero-trial audit surfaced ``eq_ilrev``, a completed 15-year
walk-forward that looked like the best free sleeve this book had seen: net Sharpe 0.69, turnover
only 3.0x/yr, Newey-West t on the mean +2.46, and correlation to all four live sleeves under 0.05
(AlphaMax +0.042, AlphaTrend -0.039, AlphaVintage +0.029, AlphaForge -0.000). Regressed on SIZE and
SPY it had R^2 of 0.0002, so it was not the forbidden size/low-vol trade either. Every screen this
repo runs said deploy it.

It was made of five bad prints. Skew +16.58, kurtosis +589.5, and the top 10 days out of 3,873
(0.26% of the sample) carried 83.6% of the total log P&L. Opening the legs showed every one of them
was an UNAPPLIED REVERSE SPLIT read straight off an unadjusted close: ADXS 0.04 -> 5.20 (+12,900%,
a 38-sigma day contributing +22.8% of NAV on a 0.18% weight), NBR 0.205 -> 11.67, FCEL 0.25 -> 6.68,
MAXN 0.071 -> 6.33. The corporate-action records for all of these EXIST and are correct in
data/lake/corporate_actions; the backtest engine applies them correctly and the deployed sleeves are
provably continuous across every split boundary. The defect is confined to research paths that
compute returns directly from ``data/lake_sharadar/ohlcv_1d.close``, which is UNADJUSTED and carries
no ``closeadj`` column even though the raw SEP.zip has one.

The failure is self-selecting, which is what makes it dangerous: a mean-reversion factor
mechanically buys whatever just collapsed, and a stock that just did a 1-for-125 reverse split looks
exactly like something that just collapsed. The factor seeks out precisely the names where the data
is broken. No amount of Sharpe, t-stat, turnover or correlation screening catches that — only
looking at WHERE THE MONEY CAME FROM does.

So this module screens the shape of the P&L rather than its summary statistics. It is deliberately
mechanism-agnostic: it does not know about splits. Any edge concentrated in a handful of days is
suspect whether the cause is a bad print, a single lucky trade, or a genuine but uninvestable jump
event — and in all three cases it is not a sleeve.

CALIBRATION, measured on this book's own deployed sleeves 2026-08-10 (top-5-day share of total
log P&L): AlphaMax -34.9% (its largest days are LOSSES), AlphaForge +28.8%, AlphaTrend +15.0%.
Against eq_ilrev's top-10 share of +83.6%. The defaults below sit well outside every healthy sleeve
and well inside the pathological one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_KURTOSIS_MAX",
    "DEFAULT_TOP_SHARE_MAX",
    "ConcentrationReport",
    "concentration_report",
]

# Fraction of total P&L the top-k days may carry before the curve is called concentrated.
# 0.60 sits above every healthy deployed sleeve measured here (max +28.8% on a top-5 basis) and far
# below eq_ilrev (+83.6% on top-10). It is a screen, not a law: a FAIL is a demand for evidence
# about where the money came from, not an automatic kill.
DEFAULT_TOP_SHARE_MAX = 0.60
# Excess kurtosis. eq_ilrev printed +589.5; a fat-tailed but real daily strategy rarely exceeds ~20.
DEFAULT_KURTOSIS_MAX = 50.0


@dataclass(frozen=True)
class ConcentrationReport:
    """What the P&L is made of. ``passed`` is False when the edge rests on too few days."""

    n_obs: int
    total_log_pnl: float
    top_k: int
    top_k_share: float
    """Signed share of total log P&L carried by the ``top_k`` largest-|return| days.

    SIGNED deliberately. A NEGATIVE share means the biggest days are LOSSES, which is the healthy
    shape (AlphaMax measures -34.9%) — an edge that survives its worst days is the opposite of a
    concentration problem. Only a large POSITIVE share indicts the curve.
    """
    skew: float
    excess_kurtosis: float
    max_abs_day: float
    max_day_sigma: float
    passed: bool
    reasons: tuple[str, ...]

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"[{verdict}] n={self.n_obs} top{self.top_k}_share={self.top_k_share:+.1%} "
            f"skew={self.skew:+.2f} kurt={self.excess_kurtosis:+.1f} "
            f"max_day={self.max_abs_day:+.4f} ({self.max_day_sigma:.1f} sd)"
            + ("" if self.passed else " — " + "; ".join(self.reasons))
        )


def concentration_report(
    log_returns: np.ndarray,
    *,
    top_k: int = 10,
    top_share_max: float = DEFAULT_TOP_SHARE_MAX,
    kurtosis_max: float = DEFAULT_KURTOSIS_MAX,
) -> ConcentrationReport:
    """Screen a daily log-return series for P&L that rests on too few observations.

    ``log_returns`` must be DAILY LOG returns, so that summing them is the total log P&L and the
    share arithmetic is exact. Passing simple returns would make the shares approximate and the
    tails understated — precisely the direction that would hide the defect this screens for.

    Returns a report rather than raising: the caller decides. A FAIL means "open the legs and show
    where the money came from before proposing this", which is exactly the step that killed
    eq_ilrev after every conventional screen had passed it.
    """
    r = np.asarray(log_returns, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < 2:
        return ConcentrationReport(
            n_obs=n, total_log_pnl=0.0, top_k=top_k, top_k_share=float("nan"),
            skew=float("nan"), excess_kurtosis=float("nan"), max_abs_day=float("nan"),
            max_day_sigma=float("nan"), passed=False,
            reasons=("fewer than 2 finite observations — nothing to screen",),
        )

    total = float(r.sum())
    k = min(top_k, n)
    order = np.argsort(np.abs(r))[::-1]
    top_sum = float(r[order[:k]].sum())
    sd = float(r.std(ddof=0))
    mu = float(r.mean())

    # WHEN THE SHARE IS MEANINGLESS. A ratio to a near-zero denominator is noise wearing a decimal
    # point, and the test must be SCALE-RELATIVE, not absolute. An earlier version of this file
    # used `abs(total) > 1e-9` and duly reported artifacts/walkforward/equity_live_fwd at a
    # top-10 share of -266.7% — and PASSED it, because the number was negative and negative shares
    # are the healthy case. Its total log P&L was +0.0009 against a daily sd of 0.0093: the whole
    # run earned less than one ordinary day's move, so no share computed from it means anything.
    # The honest denominator test is therefore "is the total bigger than a single day of noise".
    total_is_meaningful = abs(total) > sd
    share = top_sum / total if total_is_meaningful else float("nan")
    if sd > 0:
        z = (r - mu) / sd
        skew = float(np.mean(z**3))
        kurt = float(np.mean(z**4) - 3.0)
        max_sigma = float(np.max(np.abs(z)))
    else:
        skew = kurt = max_sigma = 0.0

    reasons: list[str] = []
    if np.isfinite(share) and share > top_share_max:
        reasons.append(
            f"top {k} of {n} days ({100 * k / n:.2f}% of the sample) carry {share:.1%} of total "
            f"P&L (limit {top_share_max:.0%}) — open those days before believing this"
        )
    if np.isfinite(kurt) and kurt > kurtosis_max:
        reasons.append(f"excess kurtosis {kurt:.1f} > {kurtosis_max:.0f}")
    if not total_is_meaningful:
        reasons.append(
            f"total log P&L {total:+.5f} is smaller than one daily sd ({sd:.5f}) — the curve has "
            "not earned enough for any concentration share to be meaningful; screen a longer run"
        )

    return ConcentrationReport(
        n_obs=n, total_log_pnl=total, top_k=k, top_k_share=share, skew=skew,
        excess_kurtosis=kurt, max_abs_day=float(r[order[0]]), max_day_sigma=max_sigma,
        passed=not reasons, reasons=tuple(reasons),
    )
