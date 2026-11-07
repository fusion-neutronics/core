# SPDX-License-Identifier: MIT
"""Write synthetic thermal scattering ACE tables (ACER ``iopt=2``).

A real thermal table at one temperature runs to 20-40 MB of text, and JXS
stores absolute offsets into XSS, so it cannot be trimmed to a fixture. The
tables are built here instead. The values are invented; the *layout* is the
thermal ACE format NJOY's ACER writes (NJOY2016 manual, LA-UR-17-20093), which
is what ``endf::ThermalAce`` is held to. The reader was checked against real
NJOY output separately; this keeps every layout it reads under test in CI.

Three tables, one per elastic representation, between them covering both
inelastic forms:

- ``cont.10t``: continuous outgoing energies (NXS(7) = 2) and incoherent
  elastic (NXS(5) = 3). One outgoing energy has its cosines out of order and
  its cdf does not start at zero, the two things the reader fixes up.
- ``disc.10t``: discrete, skewed outgoing energies (NXS(7) = 1) and coherent
  elastic (NXS(5) = 4).
- ``mixd.10t``: continuous outgoing energies and mixed elastic (NXS(5) = 5),
  whose incoherent part sits at JXS(7) and JXS(9).

    python tools/make_thermal_ace.py crates/endf/fixtures/synthetic-thermal.ace
"""

from __future__ import annotations

import sys
from pathlib import Path

N_MU = 4  # equiprobable cosines per outgoing energy, continuous form
E_IN = [1.0e-11, 1.0e-9, 4.0e-6]  # incident energies, MeV


def continuous_itxe(base: int) -> tuple[list[float], int]:
    """ITXE in the continuous form, starting at XSS index `base`.

    Returns the values and NXS(3) (cosines + 1).
    """
    rows_per_e = [2, 3, 2]
    locators_len = 2 * len(E_IN)
    values: list[float] = []
    data: list[float] = []
    locc: list[float] = []
    at = base + locators_len
    for i, n in enumerate(rows_per_e):
        # The first value of the block is read at locc + 1.
        locc.append(float(at + len(data) - 1))
        for j in range(n):
            e_out = (j + 1) * 1.0e-9 * (i + 1)
            pdf = 1.0e5 * (j + 1)
            # The first incident energy's cdf starts above zero, so the reader
            # prepends E' = 0.
            cdf = (j + 1) / n if i == 0 else j / (n - 1)
            mus = [-0.75 + 0.5 * k + 0.01 * i for k in range(N_MU)]
            if i == 1 and j == 1:
                mus = [mus[2], mus[0], mus[3], mus[1]]  # out of order
            data += [e_out, pdf, cdf, *mus]
    values += locc
    values += [float(n) for n in rows_per_e]
    values += data
    return values, N_MU + 1


def discrete_itxe() -> tuple[list[float], int, int]:
    """ITXE in the discrete form. Returns the values, NXS(3) and NXS(4)."""
    n_out, nil = 3, 2  # three outgoing energies, nil + 1 = 3 cosines each
    values: list[float] = []
    for i in range(len(E_IN)):
        for j in range(n_out):
            values += [(j + 1) * 2.0e-9 * (i + 1), -0.5, 0.0 + 0.1 * j, 0.5]
    return values, nil, n_out


def table(name: str, inelastic: str, elastic: str) -> str:
    xss: list[float] = [0.0]  # one-based: index 0 is padding
    nxs = [0] * 16
    jxs = [0] * 32

    def add(values) -> int:
        start = len(xss)
        xss.extend(float(v) for v in values)
        return start

    jxs[0] = add([len(E_IN), *E_IN, *[2.0 + i for i in range(len(E_IN))]])  # ITIE
    if inelastic == "continuous":
        values, nil = continuous_itxe(len(xss))
        jxs[2] = add(values)
        nxs[2], nxs[6] = nil, 2
    else:
        values, nil, nieb = discrete_itxe()
        jxs[2] = add(values)
        nxs[2], nxs[3], nxs[6] = nil, nieb, 1

    e_el = [1.0e-9, 5.0e-9, 2.0e-8]
    coherent = [len(e_el), *e_el, *[1.0e-6 * (k + 1) for k in range(len(e_el))]]
    ncl = 3  # ncl + 1 = 4 cosines per incoherent elastic energy
    incoherent = [len(e_el), *e_el, *[4.0 - k for k in range(len(e_el))]]
    inc_mu = [m for _ in e_el for m in (-0.6, -0.2, 0.2, 0.6)]
    if elastic == "incoherent":
        jxs[3] = add(incoherent)
        jxs[5] = add(inc_mu)
        nxs[4], nxs[5] = 3, ncl
    elif elastic == "coherent":
        jxs[3] = add(coherent)
        nxs[4], nxs[5] = 4, -1
    else:
        jxs[3] = add(coherent)
        jxs[6] = add(incoherent)
        jxs[8] = add(inc_mu)
        nxs[4], nxs[5], nxs[7] = 5, -1, ncl

    nxs[0] = len(xss) - 1
    nxs[1] = 1

    def fixed(values, width: int, per_line: int, fmt) -> list[str]:
        return [
            "".join(fmt(v).rjust(width) for v in values[i : i + per_line])
            for i in range(0, len(values), per_line)
        ]

    lines = [
        f"{name:>10}{'0.999167':>12}{'2.5300E-08':>12}   01/01/26",
        f"{'synthetic thermal fixture, tools/make_thermal_ace.py':<70}mat0100",
    ]
    pairs = [(1001, 0.0), (8016, 0.0)] + [(0, 0.0)] * 14
    for row in range(4):
        lines.append("".join(f"{z:>7}{a:>11.6f}" for z, a in pairs[4 * row : 4 * row + 4]))
    lines += fixed(nxs, 9, 8, lambda v: str(int(v)))
    lines += fixed(jxs, 9, 8, lambda v: str(int(v)))
    lines += fixed(xss[1:], 20, 4, lambda v: f"{v:.11E}")
    return "\n".join(lines) + "\n"


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    target = Path(sys.argv[1])
    target.write_text(
        table("cont.10t", "continuous", "incoherent")
        + table("disc.10t", "discrete", "coherent")
        + table("mixd.10t", "continuous", "mixed")
    )
    print(f"{target}: three thermal tables", file=sys.stderr)


if __name__ == "__main__":
    main()
