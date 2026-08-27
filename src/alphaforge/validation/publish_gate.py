"""Refuse to PUBLISH a number that cannot be true.

WHY THIS EXISTS. On 2026-08-08 canlicapital.com began serving a flagship live curve that read
100,000.00 then 400,207.73 — a 300% gain in one day, on a market-neutral book that runs gross at or
below 1.0x. It stood for three days. The underlying cause was a database merge that spliced two
broker accounts into one curve, and that specific bug is now fixed and pinned by tests.

This module exists because fixing that bug is not the same as making the CLASS of bug unpublishable.
The publisher regenerates and ships numbers hourly, unattended, and until now nothing between the
computation and the public site ever asked whether the result was plausible. Every check in this
repo guards an INPUT — point-in-time reads, split sanity, cost models, calendar joins. Nothing
guarded the OUTPUT. A wrong number therefore had a clear run from a subtle merge defect all the way
to a public page, and it took a human noticing a chart to catch it.

WHAT THIS IS NOT. It is not a correctness proof and cannot be one: a curve can be wrong in ways no
bound detects, and a gate that implied otherwise would be worse than none because it would invite
trust it has not earned. It catches arithmetic impossibility, nothing subtler.

CALIBRATION, measured on the published artifact 2026-08-11 rather than assumed:
    worst legitimate step in ANY published curve : +8.40%  (AlphaTrend research, downsampled)
    the phantom that shipped                     : +300.21%
`MAX_STEP` sits at 25% — 3x above anything real, 12x below the bug. The published curves are
DOWNSAMPLED, so consecutive points span several days; these are step-to-step moves, not daily
returns, and calling them daily would understate what is normal and cause false blocks.

FAILS CLOSED. On a violation the caller must not publish. A stale site is a recoverable problem; a
confidently wrong number on a page whose entire product is a verifiable record is not.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MAX_STEP",
    "GateReport",
    "GateViolation",
    "check_curve",
    "check_published_state",
]

#: Largest step-to-step move any published curve may contain. See CALIBRATION above.
MAX_STEP = 0.25


@dataclass(frozen=True)
class GateViolation:
    where: str
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.where}] {self.kind}: {self.detail}"


@dataclass
class GateReport:
    violations: list[GateViolation] = field(default_factory=list)
    curves_checked: int = 0
    points_checked: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        if self.ok:
            return (
                f"publish gate PASS — {self.curves_checked} curves, "
                f"{self.points_checked} points, no impossible values"
            )
        lines = [f"publish gate FAIL — {len(self.violations)} violation(s); NOT PUBLISHING"]
        lines += [f"    {v}" for v in self.violations]
        return "\n".join(lines)


def check_curve(
    name: str, curve: Sequence[Mapping[str, Any]], *, max_step: float = MAX_STEP
) -> list[GateViolation]:
    """Screen one published curve for values that cannot be true.

    Four checks, each earned by an observed defect rather than invented:
      * DUPLICATE DATE — two points on one date meant a "return" was computed between two marks of
        the same afternoon. This is half of how the 400,207 arose.
      * OUT OF ORDER — a curve that goes backwards in time cannot be compounded meaningfully.
      * BAD VALUE — non-finite or non-positive equity is never a real portfolio value, and a zero
        silently becomes a division by zero downstream.
      * IMPOSSIBLE STEP — the other half of the phantom, and the one a reader actually sees.
    """
    out: list[GateViolation] = []
    seen: dict[str, int] = {}
    prev_date: str | None = None
    prev_eq: float | None = None

    for i, pt in enumerate(curve):
        d, eq = pt.get("date"), pt.get("equity")
        if d is None or eq is None:
            out.append(GateViolation(name, "malformed point", f"index {i}: {pt!r}"))
            continue
        if d in seen:
            out.append(GateViolation(
                name, "duplicate date",
                f"{d} appears at index {seen[d]} and {i} — a step between two marks of the same "
                f"day is not a return",
            ))
        seen[d] = i
        if prev_date is not None and d < prev_date:
            out.append(GateViolation(name, "out of order", f"{d} follows {prev_date}"))
        try:
            eqf = float(eq)
        except (TypeError, ValueError):
            out.append(GateViolation(name, "bad value", f"{d}: equity {eq!r} is not a number"))
            prev_date = d
            continue
        if eqf != eqf or eqf in (float("inf"), float("-inf")) or eqf <= 0:
            out.append(GateViolation(name, "bad value", f"{d}: equity {eqf!r}"))
        elif prev_eq is not None and prev_eq > 0:
            step = eqf / prev_eq - 1.0
            if abs(step) > max_step:
                out.append(GateViolation(
                    name, "impossible step",
                    f"{prev_date} -> {d}: {prev_eq:,.2f} -> {eqf:,.2f} = {step:+.2%}, beyond the "
                    f"{max_step:.0%} bound (worst legitimate step ever published: +8.40%)",
                ))
            prev_eq = eqf
        else:
            prev_eq = eqf
        prev_date = d
    return out


def _iter_curves(
    state: Mapping[str, Any],
) -> Iterable[tuple[str, Sequence[Mapping[str, Any]]]]:
    for key in ("live_curve", "research_curve"):
        c = state.get(key)
        if isinstance(c, list) and c:
            yield f"book.{key}", c
    for algo in state.get("algorithms") or []:
        if not isinstance(algo, Mapping):
            continue
        nm = algo.get("name") or algo.get("key") or "?"
        for key in ("live_curve", "research_curve"):
            c = algo.get(key)
            if isinstance(c, list) and c:
                yield f"{nm}.{key}", c


def check_published_state(state: Mapping[str, Any], *, max_step: float = MAX_STEP) -> GateReport:
    """Screen every curve in the published artifact. The caller MUST NOT publish on failure."""
    rep = GateReport()
    for name, curve in _iter_curves(state):
        rep.curves_checked += 1
        rep.points_checked += len(curve)
        rep.violations.extend(check_curve(name, curve, max_step=max_step))
    return rep
