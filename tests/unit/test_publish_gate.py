"""The gate that would have stopped 400,207 reaching the public site.

Every other guard in this repo protects an INPUT — point-in-time reads, split sanity, cost models,
calendar joins. Nothing protected the OUTPUT, so a database merge defect had a clear run from a
subtle bug to a published 300% one-day gain, and a human noticing a chart three days later was the
only thing that caught it.

These tests pin both halves of that: the gate must catch the exact curve that shipped, and it must
NOT block anything the book has ever legitimately published. The second half matters as much as the
first — a gate that fires on real numbers gets switched off, and then it protects nothing.
"""

from __future__ import annotations

from alphaforge.validation.publish_gate import (
    MAX_STEP,
    check_curve,
    check_published_state,
)


def _c(*pairs: tuple[str, float]) -> list[dict]:
    return [{"date": d, "equity": e} for d, e in pairs]


# --------------------------------------------------------------------- the bug that shipped
def test_the_exact_published_phantom_is_blocked() -> None:
    """THE TEST THIS FILE EXISTS FOR — the real ALPHAC curve served for three days."""
    v = check_curve("book.live_curve", _c(
        ("2026-08-07", 100000.0),
        ("2026-08-08", 400207.73),
        ("2026-08-10", 400207.73),
    ))
    assert v, "the 300% gain that reached canlicapital.com must not be publishable"
    assert any(x.kind == "impossible step" for x in v)
    assert "+300" in str(v[0]) or "300" in str(v[0])


def test_the_sleeve_level_893_percent_is_blocked() -> None:
    """The AlphaTrend step that produced it: a superseded $100k account spliced into a $1M one."""
    v = check_curve("AlphaTrend.live_curve", _c(
        ("2026-08-07", 10068.15),
        ("2026-08-08", 100000.0),
    ))
    assert any(x.kind == "impossible step" for x in v)


def test_duplicate_dates_are_blocked() -> None:
    """The other half of the defect: two marks on one date, so a 'return' spans one afternoon."""
    v = check_curve("AlphaMax.live_curve", _c(
        ("2026-08-07", 100000.0),
        ("2026-08-07", 93116.76),
        ("2026-08-08", 100000.0),
    ))
    assert any(x.kind == "duplicate date" for x in v)


# ------------------------------------------------------- it must not block legitimate numbers
def test_every_curve_the_book_actually_publishes_passes() -> None:
    """Calibration check against the measured worst legitimate step (+8.40%).

    A gate that blocks real publishing is worse than no gate: it gets disabled, and then the next
    phantom ships unopposed.
    """
    for pct in (0.0840, -0.0840, 0.15, -0.15, 0.2499):
        v = check_curve("x", _c(("2026-01-01", 100.0), ("2026-01-08", 100.0 * (1 + pct))))
        assert not v, f"a {pct:+.2%} step must publish — worst ever seen is +8.40% ({v})"


def test_a_genuinely_terrible_day_still_publishes() -> None:
    """A -20% drawdown is a disaster, not an impossibility. The record must be able to say so."""
    assert not check_curve("x", _c(("2026-01-01", 100.0), ("2026-01-02", 80.0)))


def test_the_bound_is_where_we_calibrated_it() -> None:
    """Pin the threshold: silently loosening it would silently re-open the hole."""
    assert MAX_STEP == 0.25
    assert not check_curve("x", _c(("2026-01-01", 100.0), ("2026-01-02", 124.0)))
    assert check_curve("x", _c(("2026-01-01", 100.0), ("2026-01-02", 126.0)))


# ----------------------------------------------------------------------- structural nonsense
def test_impossible_values_are_blocked() -> None:
    for bad in (0.0, -5.0, float("nan"), float("inf")):
        assert check_curve("x", _c(("2026-01-01", 100.0), ("2026-01-02", bad))), f"{bad!r} passed"


def test_out_of_order_dates_are_blocked() -> None:
    v = check_curve("x", _c(("2026-01-05", 100.0), ("2026-01-02", 101.0)))
    assert any(x.kind == "out of order" for x in v)


def test_malformed_points_are_blocked_not_skipped() -> None:
    v = check_curve("x", [{"date": "2026-01-01"}, {"equity": 100.0}])
    assert len(v) == 2 and all(x.kind == "malformed point" for x in v)


# ------------------------------------------------------------------- whole-artifact screening
def test_state_screening_reaches_every_curve() -> None:
    state = {
        "live_curve": _c(("2026-08-07", 100000.0), ("2026-08-08", 100100.0)),
        "algorithms": [
            {"name": "AlphaMax",
             "live_curve": _c(("2026-08-07", 100000.0), ("2026-08-08", 400000.0))},
            {"name": "AlphaTrend",
             "research_curve": _c(("2020-01-01", 100.0), ("2020-02-01", 104.0))},
        ],
    }
    rep = check_published_state(state)
    assert not rep.ok
    assert rep.curves_checked == 3
    assert any("AlphaMax" in v.where for v in rep.violations)
    assert "NOT PUBLISHING" in rep.summary()


def test_a_clean_artifact_passes_and_says_what_it_checked() -> None:
    state = {
        "live_curve": _c(("2026-08-07", 100000.0), ("2026-08-08", 100100.0)),
        "algorithms": [
            {"name": "AlphaMax", "live_curve": _c(("2026-08-07", 1.0), ("2026-08-08", 1.01))}
        ],
    }
    rep = check_published_state(state)
    assert rep.ok
    assert rep.curves_checked == 2 and rep.points_checked == 4
    assert "PASS" in rep.summary()


def test_empty_state_is_not_a_pass_by_accident() -> None:
    """Nothing to check must report zero curves, so 'PASS' can never mean 'found nothing'."""
    rep = check_published_state({})
    assert rep.ok and rep.curves_checked == 0
    assert "0 curves" in rep.summary()
