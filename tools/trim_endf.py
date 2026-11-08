# SPDX-License-Identifier: MIT
"""Cut an ENDF-6 file down to a chosen set of sections.

Full evaluations run to tens of megabytes, most of it covariance data, which is
more than a test fixture needs to be useful. This keeps the sections named on
the command line and drops the rest, preserving the record structure (SEND
after each section, FEND after each file, MEND and TEND) so the result is a
valid ENDF tape that both readers accept.

Sections are kept whole. Truncating one would leave its own counts (NP, NE, NR)
describing records that are no longer there, which is a different file format,
not a smaller fixture.

    python tools/trim_endf.py in.endf out.endf 1:451 2:151 3:* 6:16 33:2

``MF:MT`` keeps one section, ``MF:*`` keeps every section of that file.
Listing the sections present, with their line counts:

    python tools/trim_endf.py --list in.endf
"""

from __future__ import annotations

import sys
from pathlib import Path


def control(line: str) -> tuple[int, int, int]:
    """The (MAT, MF, MT) in the last 14 columns."""

    def field(a: int, b: int) -> int:
        s = line[a:b].strip()
        return int(s) if s else 0

    return field(66, 70), field(70, 72), field(72, 75)


def sections(lines: list[str]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for line in lines:
        try:
            _, mf, mt = control(line)
        except ValueError:
            continue
        if mf > 0 and mt > 0:
            counts[mf, mt] = counts.get((mf, mt), 0) + 1
    return counts


def parse_keep(specs: list[str]) -> tuple[set[tuple[int, int]], set[int]]:
    pairs: set[tuple[int, int]] = set()
    whole: set[int] = set()
    for spec in specs:
        mf_s, _, mt_s = spec.partition(":")
        if mt_s == "*":
            whole.add(int(mf_s))
        else:
            pairs.add((int(mf_s), int(mt_s)))
    return pairs, whole


def trim(lines: list[str], pairs: set[tuple[int, int]], whole: set[int]) -> list[str]:
    def keep(mf: int, mt: int) -> bool:
        return mf in whole or (mf, mt) in pairs

    out: list[str] = []
    if lines:
        out.append(lines[0])  # TPID

    emitted_in_file = False
    last_kept = False
    for line in lines[1:]:
        try:
            mat, mf, mt = control(line)
        except ValueError:
            continue

        if mat == -1:  # TEND
            out.append(line)
        elif mat == 0:  # MEND
            out.append(line)
        elif mf == 0:  # FEND
            if emitted_in_file:
                out.append(line)
                emitted_in_file = False
        elif mt == 0:  # SEND
            if last_kept:
                out.append(line)
                last_kept = False
        else:
            last_kept = keep(mf, mt)
            if last_kept:
                out.append(line)
                emitted_in_file = True
    return out


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--list":
        lines = Path(args[1]).read_text().splitlines(keepends=True)
        for (mf, mt), n in sorted(sections(lines).items()):
            print(f"MF={mf:<3} MT={mt:<4} {n:>8} lines")
        return

    if len(args) < 3:
        sys.exit(__doc__)

    source, target, *specs = args
    lines = Path(source).read_text().splitlines(keepends=True)
    pairs, whole = parse_keep(specs)
    out = trim(lines, pairs, whole)
    Path(target).write_text("".join(out))

    kept = sections(out)
    print(
        f"{source} -> {target}: {len(lines)} lines -> {len(out)}, {len(kept)} sections",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
