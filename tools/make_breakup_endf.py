# SPDX-License-Identifier: MIT
"""Write a small synthetic Li7-like neutron evaluation with break-up levels.

ENDF gives a break-up after inelastic scattering, such as Li7(n,n't)alpha, as
discrete levels MT=51-91 carrying an LR flag naming the reaction, with no
section of that reaction's own. The Li7 fixture in the repository is MF=33
only, so this builds the MF=3 shape instead: an LR=0 level, two LR=33 levels
(the (n,nt) break-up), an LR=39 level (internal conversion, which is not a
reaction) and MT=102. The values are invented.

    python tools/make_breakup_endf.py crates/endf/fixtures/synthetic-breakup.endf
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from endf_writer import Section, descriptive, write_material

MAT = 9999
ZA = 3007
AWR = 6.955732

#: (MT, QM, QI, LR, threshold) in eV. QM of a break-up level is the Q of the
#: break-up, QI the level's.
SECTIONS = [
    (2, 0.0, 0.0, 0, 1.0e-5),
    (51, -477612.0, -477612.0, 0, 5.5e5),
    (52, -2467000.0, -4630000.0, 33, 5.3e6),
    (53, -2467000.0, -6680000.0, 33, 7.6e6),
    (54, -7459000.0, -7459000.0, 39, 8.5e6),
    (102, 2032800.0, 2032800.0, 0, 1.0e-5),
]


def mf3(mt: int, qm: float, qi: float, lr: int, threshold: float):
    s = Section(MAT, 3, mt)
    s.cont(ZA, AWR, 0, 0, 0, 0)
    s.tab1(qm, qi, 0, lr, [threshold, 1.4e7, 2.0e7], [0.0, 0.01 * (mt - 50), 0.005 * (mt - 50)])
    return s.finish()


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sections = [mf3(*row) for row in SECTIONS]
    directory = [(1, 451, 9)] + [(3, row[0], len(s) - 1) for row, s in zip(SECTIONS, sections)]
    header = descriptive(
        MAT,
        ZA,
        AWR,
        # NSUB=10, incident neutron data.
        10,
        directory,
        "  3-Li-  7",
        [
            "----SYNTHETIC         MATERIAL 9999",
            "-----INCIDENT NEUTRON DATA",
            "------ENDF-6 FORMAT",
        ],
        author="tools/make_breakup_endf.py",
    )
    write_material(Path(sys.argv[1]), MAT, [header, *sections])


if __name__ == "__main__":
    main()
