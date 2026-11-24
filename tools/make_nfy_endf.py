# SPDX-License-Identifier: MIT
"""Write a small synthetic ENDF evaluation carrying fission product yields.

MF=8 MT=454 and MT=459 are the only files with no fixture: a real neutron-
induced fission yield evaluation lists a thousand or so products at each of
several incident energies, which is megabytes, and it cannot be trimmed the way
the other ENDF fixtures were because the yields *are* the file: cutting
products would only test a shorter list of the same thing.

So a small one is built here instead. The values are invented, but the layout
is the format's, and that is what both readers are held to: a HEAD giving LE+1,
then one LIST per incident energy whose L1 field is overloaded (LE+1 on the
first, the interpolation scheme on the rest) and four values per product.

The independent (MT=454) and cumulative (MT=459) yields differ, so a reader
that returns one for the other is caught. The products include an isomer
(FPS=1) so the naming path is exercised, and the second energy has a different
product count from the first so a reader that reuses NFP is caught too.

    python tools/make_nfy_endf.py tests/synthetic-nfy.endf
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from endf_writer import Section, descriptive, write_material  # noqa: E402

#: MAT number of the synthetic evaluation. 9999 is unassigned.
MAT = 9999

#: Z*1000 + A of the fissioning nuclide, and its mass in neutron masses.
ZA = 92235
AWR = 233.0248

#: Incident energies in eV, thermal and fast.
ENERGIES = [0.0253, 500000.0]

#: Interpolation scheme for the energies after the first. 2 is linear-linear.
INTERPOLATION = 2

#: (ZAFP, FPS, yield, uncertainty) per energy, for MT=454 and MT=459. The
#: cumulative yields are larger than the independent ones, as they are in a
#: real evaluation, and the second energy carries one more product.
INDEPENDENT = [
    [
        (40095.0, 0.0, 0.0201, 0.0008),  # Zr95
        (54135.0, 1.0, 0.0134, 0.0006),  # Xe135_m1
        (55137.0, 0.0, 0.0619, 0.0021),  # Cs137
    ],
    [
        (40095.0, 0.0, 0.0188, 0.0009),
        (54135.0, 1.0, 0.0142, 0.0007),
        (55137.0, 0.0, 0.0575, 0.0024),
        (58144.0, 0.0, 0.0043, 0.0002),  # Ce144, fast only
    ],
]

CUMULATIVE = [
    [
        (40095.0, 0.0, 0.0605, 0.0018),
        (54135.0, 1.0, 0.0139, 0.0006),
        (55137.0, 0.0, 0.0632, 0.0022),
    ],
    [
        (40095.0, 0.0, 0.0588, 0.0020),
        (54135.0, 1.0, 0.0147, 0.0008),
        (55137.0, 0.0, 0.0601, 0.0025),
        (58144.0, 0.0, 0.0049, 0.0003),
    ],
]


def yields_section(mt: int, sets: list[list[tuple[float, float, float, float]]]):
    """MF=8 MT=454 or MT=459: the yields at each incident energy."""
    s = Section(MAT, 8, mt)
    le_plus_one = len(sets)
    s.cont(ZA, AWR, le_plus_one, 0, 0, 0)
    for i, (energy, products) in enumerate(zip(ENERGIES, sets)):
        # L1 is LE+1 on the first energy and the interpolation scheme after,
        # which is the overload both readers have to reproduce.
        l1 = le_plus_one if i == 0 else INTERPOLATION
        flat = [v for product in products for v in product]
        s.list_record(energy, 0.0, l1, 0, len(products), flat)
    return s.finish()


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    target = Path(sys.argv[1])

    sections = [
        descriptive(
            MAT,
            ZA,
            AWR,
            # NSUB=11, neutron-induced fission product yields.
            11,
            [(1, 451, 9), (8, 454, 5), (8, 459, 5)],
            " 92-U -235",
            [
                "----SYNTHETIC         MATERIAL 9999",
                "-----NEUTRON-INDUCED FISSION PRODUCT YIELDS",
                "------ENDF-6 FORMAT",
            ],
            author="tools/make_nfy_endf.py",
            lfi=1,
        ),
        yields_section(454, INDEPENDENT),
        yields_section(459, CUMULATIVE),
    ]
    write_material(target, MAT, sections)

    products = sum(len(s) for s in INDEPENDENT) + sum(len(s) for s in CUMULATIVE)
    print(f"{target}: {len(ENERGIES)} energies, {products} yields", file=sys.stderr)


if __name__ == "__main__":
    main()
