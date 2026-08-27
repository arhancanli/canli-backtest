"""Make a pre-registration MACHINE-CHECKABLE, so a run cannot silently ignore it.

THE BUG CLASS THIS CLOSES. On 2026-08-07, three separate runs failed for one reason: a
pre-registration named a data source in prose, and nothing in the code ever read it. The profile
is chosen by whoever launches the run; the declared source lives in a markdown table nobody parses.

    eq_net_issuance   ran against data/lake (no `shares_basic`)  -> silent null, trial burned
    eq_accruals       ran against data/lake (no `op_cash_flow`)  -> silent null, trial burned
    AlphaLedger       ran against data/lake (11% of names under $10k ADV)
                      -> CostModelMisuse crash, ~4h of compute, twice

The declarations were correct every time. PREREG_FUNDAMENTAL_SINGLES.md says "Sharadar SF1 lake";
PREREG_SLEEVE4_INVESTMENT.md says "History (validation): Sharadar SF1, on disk". Both runs used
`data/lake` anyway, because a document is not a constraint until something enforces it.

The crash was the LUCKY case. The two silent nulls each consumed a hypothesis against the honest N
and produced no measurement, which is strictly worse: it raises the deflation bar for every sleeve
in the book while buying nothing, and it looked like a completed experiment for three days.

HOW IT WORKS. A pre-registration carries a fenced ```prereg block of key: value lines. A runner
calls :func:`assert_matches` before spending compute, and the run dies immediately if the resolved
settings disagree with what the document declared. Prose stays prose; the block is the contract.

    ```prereg
    profile: sharadar
    lake_dir: data/lake_sharadar
    alpha_names: eq_asset_growth
    allocator: rank
    ```

Deliberately NOT a schema for everything. It covers only the fields whose mismatch has actually
caused a failure — the data source, the profile, the factor set, the allocator. A guard that tries
to encode every parameter becomes a second source of truth that drifts from the first, which is
the same disease one layer up.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

__all__ = ["PreRegError", "assert_matches", "load_prereg", "prereg_docs"]

_BLOCK = re.compile(r"```prereg\s*\n(.*?)```", re.DOTALL)
#: Fields compared against resolved settings. Anything else in the block is recorded, not enforced.
_ENFORCED = ("profile", "lake_dir", "alpha_names", "allocator")


class PreRegError(RuntimeError):
    """A run disagrees with the pre-registration it claims to execute."""


def prereg_docs(root: Path) -> list[Path]:
    return sorted((root / "docs" / "design").glob("PREREG_*.md"))


def load_prereg(path: Path | str) -> dict[str, Any]:
    """Parse the ```prereg block. Raises if absent — an unparseable declaration is not a control."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    m = _BLOCK.search(text)
    if not m:
        raise PreRegError(
            f"{p} has no ```prereg block. A pre-registration that names its inputs only in prose "
            "cannot be enforced, which is exactly how three runs used the wrong lake on 2026-08-07."
        )
    out: dict[str, Any] = {}
    for line in m.group(1).splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        out[k] = [t.strip() for t in v.split(",")] if "," in v else v
    return out


def assert_matches(
    prereg_path: Path | str,
    *,
    lake_dir: Path | str | None = None,
    profile: str | None = None,
    alpha_names: list[str] | None = None,
    allocator: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail LOUDLY unless the run matches its declaration. Returns the parsed block.

    Call this BEFORE spending compute. Every argument left as ``None`` is skipped, so a caller
    checks what it actually resolved rather than being forced to restate the whole document.
    """
    decl = load_prereg(prereg_path)
    actual: dict[str, Any] = {}
    if profile is not None:
        actual["profile"] = profile
    if lake_dir is not None:
        # Compare by trailing path so an absolute resolved dir matches a repo-relative declaration.
        actual["lake_dir"] = str(lake_dir)
    if alpha_names is not None:
        actual["alpha_names"] = list(alpha_names)
    if allocator is not None:
        actual["allocator"] = allocator
    if extra:
        actual.update(dict(extra))

    problems: list[str] = []
    for key in (*_ENFORCED, *(extra or {})):
        if key not in decl or key not in actual:
            continue
        want, got = decl[key], actual[key]
        if key == "lake_dir":
            ok = str(got).rstrip("/").endswith(str(want).rstrip("/"))
        elif key == "alpha_names":
            ok = sorted(want if isinstance(want, list) else [want]) == sorted(got)
        else:
            ok = str(want) == str(got)
        if not ok:
            problems.append(f"    {key}: declared {want!r}  but run resolved {got!r}")

    if problems:
        raise PreRegError(
            f"RUN CONTRADICTS ITS PRE-REGISTRATION ({Path(prereg_path).name}):\n"
            + "\n".join(problems)
            + "\n  Refusing to spend compute. Either run the declared configuration, or write a "
            "NEW pre-registration — silently running something else is how a trial gets burned on "
            "a lake that lacks the columns the factor reads."
        )
    return decl
