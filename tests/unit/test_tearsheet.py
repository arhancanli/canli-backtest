"""Unit tests for alphaforge.analytics.tearsheet (execDesign.md §10.3).

Covers: Agg backend forced at import; PNG + TXT written to tmp_path with the
PNG above a sanity size floor; the text block containing every PerfSummary
field; deterministic byte-stable re-rendering (no timestamps in PNG metadata,
fixed figsize/dpi — documented contract); graceful rendering when optional
inputs (fills/funding/positions) are absent or history is short.

Offline and tmp_path-only by construction (synthetic seed-free fixtures).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from alphaforge.analytics import PerfSummary, build_tearsheet, render_text, summarize

HOUR = 3_600_000
T0 = 1_704_067_200_000  # 2024-01-01T00:00:00Z


@dataclass(frozen=True, slots=True, kw_only=True)
class FakeResult:
    """Minimal ResultLike satisfier for tests."""

    equity: pd.Series
    fills: pd.DataFrame | None = None
    funding: pd.Series | None = None
    positions: pd.DataFrame | None = None


def hourly_equity(n_days: int) -> pd.Series:
    """Deterministic (seed-free) strictly-positive hourly curve spanning n_days."""
    n = n_days * 24
    rets = 0.0001 + 0.002 * np.sin(np.arange(n) * 0.61) - 0.0003 * np.cos(np.arange(n) * 0.13)
    vals = 100_000.0 * np.cumprod(1.0 + rets)
    idx = pd.Index([T0 + (i + 1) * HOUR for i in range(n)], name="ts")
    return pd.Series(vals, index=idx, dtype=float)


def full_result(n_days: int = 45) -> FakeResult:
    equity = hourly_equity(n_days)
    ts = [int(t) for t in equity.index[::24]]
    fills = pd.DataFrame({"ts": ts, "notional": [1_000.0] * len(ts), "fee": [0.4] * len(ts)})
    funding = pd.Series([0.5, -0.25], index=pd.Index(ts[:2], name="ts"))
    positions = pd.DataFrame({"ts": ts, "qty": [0.5] * len(ts), "mark": [40_000.0] * len(ts)})
    return FakeResult(equity=equity, fills=fills, funding=funding, positions=positions)


class TestBackend:
    def test_agg_forced_at_import(self) -> None:
        # tearsheet is imported via the package __init__ at test import time.
        assert matplotlib.get_backend().lower() == "agg"


class TestBuildTearsheet:
    def test_writes_png_and_txt(self, tmp_path: Path) -> None:
        out = build_tearsheet(full_result(), tmp_path / "ts")
        assert out == tmp_path / "ts" / "tearsheet.png"
        assert out.is_file()
        assert out.stat().st_size > 10_000  # a real multi-panel render, not a stub
        txt = out.with_suffix(".txt")
        assert txt.is_file()
        assert txt.read_text(encoding="utf-8").strip()

    def test_txt_contains_every_perf_summary_field(self, tmp_path: Path) -> None:
        out = build_tearsheet(full_result(), tmp_path)
        text = out.with_suffix(".txt").read_text(encoding="utf-8")
        for field in dataclasses.fields(PerfSummary):
            assert field.name in text, f"missing field {field.name} in tearsheet.txt"

    def test_rerender_is_byte_stable(self, tmp_path: Path) -> None:
        # Contract (documented in tearsheet.py): identical inputs + same
        # matplotlib version -> identical PNG bytes (Agg, fixed figsize/dpi,
        # pinned metadata with no timestamps).
        result = full_result()
        png_a = build_tearsheet(result, tmp_path / "a")
        png_b = build_tearsheet(result, tmp_path / "b")
        assert png_a.read_bytes() == png_b.read_bytes()
        assert png_a.with_suffix(".txt").read_bytes() == png_b.with_suffix(".txt").read_bytes()

    def test_renders_without_optional_inputs(self, tmp_path: Path) -> None:
        out = build_tearsheet(FakeResult(equity=hourly_equity(40)), tmp_path)
        assert out.is_file()
        text = out.with_suffix(".txt").read_text(encoding="utf-8")
        assert "turnover_ann" in text
        assert "nan" in text  # missing optionals surface as nan, never 0

    def test_renders_short_history(self, tmp_path: Path) -> None:
        # < 2 UTC days: daily panels fall back to placeholders, no crash.
        out = build_tearsheet(FakeResult(equity=hourly_equity(1)), tmp_path)
        assert out.is_file()
        assert out.stat().st_size > 10_000

    def test_creates_out_dir(self, tmp_path: Path) -> None:
        out = build_tearsheet(full_result(3), tmp_path / "deep" / "nested")
        assert out.is_file()


class TestRenderText:
    def test_aligned_and_complete(self) -> None:
        s = summarize(hourly_equity(5))
        text = render_text(s)
        lines = text.strip().splitlines()
        field_names = [f.name for f in dataclasses.fields(PerfSummary)]
        assert len(lines) == len(field_names) + 2  # header + rule + one line per field
        width = max(len(n) for n in field_names)
        for line, name in zip(lines[2:], field_names, strict=True):
            assert line.startswith(f"{name:<{width}}  ")

    def test_timestamp_fields_render_iso(self) -> None:
        s = summarize(hourly_equity(2))
        text = render_text(s)
        assert f"{T0 + HOUR}  (2024-01-01 01:00:00Z)" in text

    def test_float_display_rounding_six_decimals(self) -> None:
        s = summarize(hourly_equity(5))
        text = render_text(s)
        line = next(ln for ln in text.splitlines() if ln.startswith("total_return"))
        rendered = line.split()[-1]
        assert rendered == f"{s.total_return:.6f}"  # display rounding only

    def test_deterministic(self) -> None:
        s = summarize(hourly_equity(4))
        assert render_text(s) == render_text(s)
