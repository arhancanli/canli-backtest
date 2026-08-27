"""A pre-registration must be enforceable, not merely written.

On 2026-08-07 three runs contradicted their own declarations and nothing noticed:

    eq_net_issuance  ran against a lake with no `shares_basic`   -> silent null, trial burned
    eq_accruals      ran against a lake with no `op_cash_flow`   -> silent null, trial burned
    AlphaLedger      ran against a lake 11% microcaps by ADV     -> crash, ~4h wasted, twice

Every declaration was right. The runs were wrong. The gap was that a markdown table is not a
control. These tests pin the control: every pre-registration carries a machine-readable block, and
a run that disagrees with one dies before it spends compute.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alphaforge.validation.prereg import (
    PreRegError,
    assert_matches,
    load_prereg,
    prereg_docs,
)

_REPO = Path(__file__).resolve().parents[2]
_SLEEVE4 = _REPO / "docs/design/PREREG_SLEEVE4_INVESTMENT.md"


def test_every_preregistration_is_machine_readable() -> None:
    docs = prereg_docs(_REPO)
    assert docs, "no PREREG_*.md found"
    for p in docs:
        decl = load_prereg(p)
        assert decl.get("lake_dir"), f"{p.name} declares no lake_dir — the field that broke 3 runs"
        assert decl.get("profile"), f"{p.name} declares no profile"


def test_the_declaration_matches_what_the_document_says_in_prose() -> None:
    """The block must agree with the table above it, or it is decoration.

    PREREG_SLEEVE4_INVESTMENT.md line 76 reads "History (validation): Sharadar SF1, on disk".
    """
    decl = load_prereg(_SLEEVE4)
    assert decl["lake_dir"] == "data/lake_sharadar"
    assert decl["profile"] == "sharadar"
    assert "Sharadar" in _SLEEVE4.read_text(encoding="utf-8")


def test_the_wrong_lake_is_rejected() -> None:
    """THE TEST THIS FILE EXISTS FOR — reproduce the exact failure that cost a day."""
    with pytest.raises(PreRegError, match="lake_dir"):
        assert_matches(_SLEEVE4, lake_dir="/Users/x/alphaforge/data/lake", profile="sharadar")


def test_the_wrong_profile_is_rejected() -> None:
    with pytest.raises(PreRegError, match="profile"):
        assert_matches(_SLEEVE4, profile="equity")


def test_the_wrong_factor_is_rejected() -> None:
    with pytest.raises(PreRegError, match="alpha_names"):
        assert_matches(_SLEEVE4, alpha_names=["eq_net_issuance"])


def test_the_declared_configuration_passes() -> None:
    decl = assert_matches(
        _SLEEVE4,
        lake_dir=str(_REPO / "data/lake_sharadar"),   # absolute resolved path must match
        profile="sharadar",
        alpha_names=["eq_asset_growth"],
        allocator="rank",
    )
    assert decl["universe_sha256"].startswith("2fd82d30")


def test_unchecked_fields_are_ignored_not_guessed() -> None:
    """A caller that cannot resolve a field passes None; the guard must not invent a comparison."""
    assert_matches(_SLEEVE4, profile="sharadar")  # lake/alphas unspecified -> no error


def test_a_missing_block_is_an_error_not_a_pass() -> None:
    """Silence must never read as compliance — that is the whole failure mode."""
    tmp = _REPO / "docs" / "design" / "_tmp_no_block.md"
    tmp.write_text("# a prereg with prose only\n\nUniverse: the frozen cohort.\n", encoding="utf-8")
    try:
        with pytest.raises(PreRegError, match="no ```prereg block"):
            load_prereg(tmp)
    finally:
        tmp.unlink()
