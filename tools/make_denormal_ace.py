# SPDX-License-Identifier: MIT
"""Write a tiny ACE table carrying the float form NJOY writes for a denormal.

An exponent below about 1e-100 needs three digits, which overflows the field
NJOY has to write it in, so it drops the ``e``: ``6.10562372605-318`` rather
than ``6.10562372605e-318``. Both readers have to put it back.

This surfaced converting TENDL-2025, where Db262, Db263, Db264, Sg272 and
Sg273 each failed on two malformed tokens out of 772,031; see issue #20. No
ACE file small enough to keep as a fixture contains one, so the table is built
here, and the golden then holds both readers to the same values.

The values around it are ordinary, so a reader that mangles the whole array
rather than the one token is caught too.

    python tools/make_denormal_ace.py tests/synthetic-denormal.ace
"""

from __future__ import annotations

import sys
from pathlib import Path

#: The XSS values, written verbatim rather than formatted, so the malformed
#: tokens survive into the file exactly as NJOY would write them.
XSS_TEXT = [
    " 1.00000000000E+00",
    " 6.10562372605-318",
    "-6.10562372605-318",
    " 1.23456700000-120",
    " 2.00000000000E+00",
    " 9.99999999999-323",
    " 0.00000000000E+00",
    " 3.00000000000E+00",
]


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    target = Path(sys.argv[1])

    n_xss = len(XSS_TEXT)

    nxs = [0] * 16
    nxs[0] = n_xss  # NXS(1): length of XSS
    nxs[1] = 1001  # NXS(2): ZA

    jxs = [0] * 32

    def fixed(values, width: int, per_line: int, fmt) -> list[str]:
        out = []
        for i in range(0, len(values), per_line):
            out.append("".join(fmt(v).rjust(width) for v in values[i : i + per_line]))
        return out

    lines = [
        f"{'1001.00c':>10}{'0.999167':>12}{'2.5300E-08':>12}   01/01/26",
        f"{'synthetic denormal fixture, tools/make_denormal_ace.py':<70}mat0100",
    ]
    for _ in range(4):
        lines.append("".join(f"{0:>7}{0.0:>11.6f}" for _ in range(4)))
    lines += fixed(nxs, 9, 8, lambda v: str(int(v)))
    lines += fixed(jxs, 9, 8, lambda v: str(int(v)))
    # Four values to a line, as the format writes them.
    for i in range(0, len(XSS_TEXT), 4):
        lines.append("".join(v.rjust(20) for v in XSS_TEXT[i : i + 4]))

    target.write_text("\n".join(lines) + "\n")
    print(f"{target}: {n_xss} XSS values, 4 of them denormal", file=sys.stderr)


if __name__ == "__main__":
    main()
