# SPDX-License-Identifier: MIT
"""Write a synthetic ACE table exercising every secondary distribution law.

Li6 is a real table and covers what a light nuclide's evaluation happens to
use: laws 3, 33 and 44. The remaining laws (a discrete photon, a continuous
tabulation, the Maxwell, evaporation and Watt spectra, a correlated
angle-energy distribution and N-body phase space) need a table of their own,
and no real one small enough to keep as a fixture holds them all. JXS stores
absolute offsets into XSS, so a large table cannot be trimmed down to the
interesting part either.

So the block is built here. The values are invented; the *layout* is the
format's, which is what the two readers are being held to. Each law is given
its own reaction so the DLW locators, the applicability records and the law
dispatch are all exercised, and the shapes within a law are varied
deliberately: the continuous tabulation carries one purely continuous
distribution, one purely discrete and one mixture, and the correlated law
carries both isotropic and tabulated cosines.

    python tools/make_laws_ace.py tests/synthetic-laws.ace
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Every law this table carries, in the order the reactions are written.
LAWS = [2, 4, 7, 9, 11, 61, 66]


class Block:
    """A run of XSS values addressed by one-based offsets from its start."""

    def __init__(self):
        self.values: list[float] = []

    def add(self, values) -> int:
        """Append `values`; return the one-based offset of the first."""
        start = len(self.values) + 1
        self.values.extend(float(v) for v in values)
        return start

    def __len__(self) -> int:
        return len(self.values)


def tab1(x, y) -> list[float]:
    """A TAB1 as `Tabulated1D.from_ace` reads it, with a single region."""
    return [0.0, float(len(x)), *x, *y]


def outgoing(n_discrete: int, intt: int, eout, pdf, cdf, *extra) -> list[float]:
    """One outgoing energy distribution, as `_ace_outgoing_energy` reads it.

    `extra` supplies the columns beyond the first three, one list per column:
    two for Kalbach-Mann, one for the correlated law.
    """
    header = 10 * n_discrete + intt
    rows = [eout, pdf, cdf, *extra]
    values = [float(header), float(len(eout))]
    for row in rows:
        values.extend(float(v) for v in row)
    return values


def law_2(dlw: Block) -> int:
    """A discrete photon: a primary flag and an energy in MeV."""
    return dlw.add([1.0, 2.5])


def law_4(dlw: Block) -> int:
    """A continuous tabulation at three incident energies.

    The three differ on purpose: the first is a plain continuum, the second is
    nothing but discrete lines, and the third mixes the two: the three
    branches `_ace_outgoing_energy` can take.
    """
    continuum = dlw.add(
        outgoing(
            0,
            2,
            [0.1, 0.5, 1.0],
            [0.4, 0.8, 0.2],
            [0.0, 0.55, 1.0],
        )
    )
    discrete = dlw.add(
        outgoing(
            2,
            2,
            [0.25, 0.75],
            [0.3, 0.7],
            [0.3, 1.0],
        )
    )
    mixture = dlw.add(
        outgoing(
            1,
            1,
            [0.2, 0.6, 1.2],
            [0.25, 0.5, 0.25],
            [0.25, 0.6, 1.0],
        )
    )
    # A single lin-lin region over the whole incident grid.
    return dlw.add([0.0, 3.0, 1.0e-6, 1.0, 20.0, continuum, discrete, mixture])


def law_7(dlw: Block) -> int:
    """A Maxwell fission spectrum: theta against energy, then U."""
    return dlw.add([*tab1([1.0e-6, 20.0], [0.5, 1.4]), 0.3])


def law_9(dlw: Block) -> int:
    """An evaporation spectrum, laid out exactly as law 7."""
    return dlw.add([*tab1([1.0e-6, 10.0, 20.0], [0.4, 0.9, 1.1]), 0.25])


def law_11(dlw: Block) -> int:
    """A Watt spectrum: the a and b parameters, then U."""
    return dlw.add(
        [
            *tab1([1.0e-6, 20.0], [0.9, 1.1]),
            *tab1([1.0e-6, 20.0], [3.0, 3.4]),
            0.4,
        ]
    )


def law_61(dlw: Block) -> int:
    """A correlated angle-energy distribution at two incident energies.

    The cosine locators cover both cases: zero for isotropic, and an offset to
    a tabulated cosine distribution.
    """
    cosine = dlw.add(
        [
            2.0,  # lin-lin
            3.0,  # three cosines
            -1.0,
            0.0,
            1.0,
            0.25,
            0.5,
            0.25,
            0.0,
            0.5,
            1.0,
        ]
    )
    first = dlw.add(
        outgoing(
            0,
            2,
            [0.1, 0.9],
            [0.6, 0.4],
            [0.0, 1.0],
            [0.0, cosine],
        )
    )
    second = dlw.add(
        outgoing(
            0,
            1,
            [0.2, 0.8, 1.6],
            [0.5, 0.3, 0.2],
            [0.0, 0.5, 1.0],
            [cosine, 0.0, cosine],
        )
    )
    # Two regions, so the branch that reads breakpoints is taken here.
    return dlw.add([2.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0e-6, 20.0, first, second])


def law_66(dlw: Block) -> int:
    """N-body phase space: the particle count and their total mass."""
    return dlw.add([4.0, 3.98])


LAW_DATA = {
    2: law_2,
    4: law_4,
    7: law_7,
    9: law_9,
    11: law_11,
    61: law_61,
    66: law_66,
}


def build_dlw() -> tuple[list[float], list[int]]:
    """The DLW block, and the LDLW locators that point into it.

    Each reaction gets an entry of three control words and an applicability
    record; the law's own data is appended afterwards and the entry's third
    word patched to point at it.
    """
    dlw = Block()
    locators = []
    entries = []
    for _ in LAWS:
        # LNW, LAW and IDAT are filled in below; the applicability follows.
        start = dlw.add([0.0, 0.0, 0.0, *tab1([1.0e-6, 20.0], [1.0, 1.0])])
        locators.append(start)
        entries.append(start)

    for law, start in zip(LAWS, entries):
        idat = LAW_DATA[law](dlw)
        dlw.values[start] = float(law)  # word 2 of the entry
        dlw.values[start + 1] = float(idat)  # word 3

    return dlw.values, locators


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    target = Path(sys.argv[1])

    dlw_values, locators = build_dlw()

    # XSS is one-based, so index 0 is padding. LDLW comes first, then DLW.
    xss = [0.0]
    ldlw_start = len(xss)
    xss.extend(float(v) for v in locators)
    dlw_start = len(xss)
    xss.extend(dlw_values)
    n_xss = len(xss) - 1

    nxs = [0] * 16
    nxs[0] = n_xss  # NXS(1): length of XSS
    nxs[1] = 1001  # NXS(2): ZA
    nxs[3] = len(LAWS)  # NXS(4): reactions besides elastic
    nxs[4] = len(LAWS)  # NXS(5): those with secondary neutrons

    jxs = [0] * 32
    jxs[9] = ldlw_start  # JXS(10): LDLW
    jxs[10] = dlw_start  # JXS(11): DLW

    def fixed(values, width: int, per_line: int, fmt) -> list[str]:
        out = []
        for i in range(0, len(values), per_line):
            out.append("".join(fmt(v).rjust(width) for v in values[i : i + per_line]))
        return out

    lines = [
        f"{'1001.00c':>10}{'0.999167':>12}{'2.5300E-08':>12}   01/01/26",
        f"{'synthetic law fixture, generated by tools/make_laws_ace.py':<70}mat0100",
    ]
    for _ in range(4):
        lines.append("".join(f"{0:>7}{0.0:>11.6f}" for _ in range(4)))
    lines += fixed(nxs, 9, 8, lambda v: str(int(v)))
    lines += fixed(jxs, 9, 8, lambda v: str(int(v)))
    # XSS omits the padding element; the reader puts it back.
    lines += fixed(xss[1:], 20, 4, lambda v: f"{v:.11E}")

    target.write_text("\n".join(lines) + "\n")
    print(
        f"{target}: {n_xss} XSS values, {len(LAWS)} laws, DLW at {dlw_start}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
