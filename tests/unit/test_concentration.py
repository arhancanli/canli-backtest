"""The screen that would have caught eq_ilrev before it was proposed.

eq_ilrev passed EVERY conventional screen this repo runs: net Sharpe 0.69, turnover 3.0x/yr,
Newey-West t +2.46, correlation under 0.05 to all four live sleeves, and R^2 0.0002 against
SIZE+SPY so it was not the forbidden size/low-vol trade. It was still made of five unapplied
reverse splits, and the only thing that revealed it was asking where the money came from: the top
10 days out of 3,873 carried 83.6% of total P&L.

These tests pin that the screen separates that shape from the shape of the sleeves actually
deployed here, whose measured top-5-day shares are AlphaMax -34.9%, AlphaForge +28.8%,
AlphaTrend +15.0%.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphaforge.validation.concentration import concentration_report


def _rng(seed: int = 7) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_healthy_diffuse_curve_passes() -> None:
    """A real edge earns a little on most days. Nothing to explain."""
    r = _rng().normal(0.0004, 0.006, 2000)
    rep = concentration_report(r)
    assert rep.passed, rep.summary()
    assert rep.top_k_share < 0.60


def test_eq_ilrev_shape_is_caught() -> None:
    """THE TEST THIS FILE EXISTS FOR — an edge that is five bad prints.

    Built to eq_ilrev's measured shape: a long series that loses slightly on almost every day, plus
    a handful of enormous positive spikes that carry the entire result.
    """
    r = _rng().normal(-0.0002, 0.004, 3873)
    spikes = (0.234, 0.180, 0.150, 0.120, 0.100, 0.090, 0.080, 0.070, 0.060, 0.055)
    for i, spike in enumerate(spikes):
        r[500 + i * 300] = spike
    rep = concentration_report(r)
    assert not rep.passed, f"failed to catch the eq_ilrev shape: {rep.summary()}"
    assert rep.top_k_share > 0.60
    assert any("carry" in x for x in rep.reasons)


def test_biggest_days_being_losses_is_healthy_not_concentrated() -> None:
    """A NEGATIVE share must PASS.

    AlphaMax's five largest days are losses (-34.9% of total). An edge that survives its worst days
    is the opposite of a concentration problem, and a screen that flagged it would be worse than no
    screen — it would train everyone to ignore the warning.
    """
    r = _rng().normal(0.0006, 0.005, 1500)
    for i in (100, 400, 700, 1000, 1300):
        r[i] = -0.03
    rep = concentration_report(r, top_k=5)
    assert rep.top_k_share < 0, "the largest days here are losses, so the share must be negative"
    assert rep.passed, rep.summary()


def test_near_zero_total_reports_nan_rather_than_a_spectacular_number() -> None:
    """equity_live_fwd printed a -266.7% share purely because its total was +0.0009.

    A number like that is noise wearing a decimal point. The denominator test must be
    SCALE-RELATIVE: an absolute 1e-9 floor let this through and PASSED it, because the bogus share
    came out negative and negative shares are the healthy case.
    """
    r = _rng().normal(0.0, 0.006, 300)
    r = r - r.mean()  # force the total to ~0
    rep = concentration_report(r)
    assert np.isnan(rep.top_k_share)
    assert not rep.passed
    assert any("daily sd" in x for x in rep.reasons)


def test_total_smaller_than_one_day_of_noise_is_not_screenable() -> None:
    """The exact equity_live_fwd shape: a real, non-zero total that is still too small to divide by.

    total = +0.0009 against a daily sd of 0.0093. This must FAIL as unscreenable rather than pass
    on a nonsense negative share.
    """
    r = _rng().normal(0.0, 0.0093, 176)
    r = r - r.mean() + (0.0009 / 176)  # total ~= +0.0009, sd ~= 0.0093
    rep = concentration_report(r)
    assert abs(float(np.sum(r))) < float(np.std(r))
    assert np.isnan(rep.top_k_share), "a total below one daily sd cannot yield a meaningful share"
    assert not rep.passed


def test_extreme_kurtosis_is_flagged_independently() -> None:
    """eq_ilrev printed excess kurtosis +589.5. That alone warrants opening the legs."""
    r = _rng().normal(0.0005, 0.004, 4000)
    r[1234] = 1.2  # one 300-sigma print
    rep = concentration_report(r)
    assert rep.excess_kurtosis > 50.0
    assert not rep.passed


def test_degenerate_input_fails_closed() -> None:
    """Too little data must not silently PASS — an unscreenable curve is not a screened one."""
    for bad in (np.array([]), np.array([0.01]), np.array([np.nan, np.inf])):
        rep = concentration_report(bad)
        assert not rep.passed


def test_share_is_exact_on_log_returns() -> None:
    """Summing LOG returns is the total log P&L, so the share arithmetic must be exact."""
    r = np.array([0.5, 0.1, -0.05, 0.02, 0.03])
    rep = concentration_report(r, top_k=1)
    assert rep.total_log_pnl == pytest.approx(0.60)
    assert rep.top_k_share == pytest.approx(0.5 / 0.60)
