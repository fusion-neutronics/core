# SPDX-License-Identifier: MIT
"""Write a synthetic evaluation carrying the shapes no fixture reaches.

Four parsers were structurally complete and entirely unverified, because no
evaluation small enough to keep in the repository writes them:

* **MF=2 LRF=2**, multi-level Breit-Wigner resonance parameters. Fe56 and U235
  give Reich-Moore; the Breit-Wigner branch had never been read.
* **MF=5 LF=12**, the Madland-Nix fission spectrum. LF=1, 5, 7 and 9 are
  covered by Li6, Am244 and U235.
* **MF=6 LANG=2 and LAW=6**, Kalbach-Mann angular parameters and the n-body
  phase space. Li6 gives LAW=2 and LAW=4, Fe56 gives LAW=1 with LANG=1.
* **MF=13**, photon production written as a cross section rather than as a
  multiplicity on MF=12.

The numbers are invented; the layout is the format's, and that is what both
readers are held to. What each shape is built to catch:

* The Breit-Wigner section has two L values with different resonance counts, so
  a reader that reuses NRS is caught, and QX and LRX are non-zero in one of
  them, since those two fields are easy to drop.
* LF=12 carries EFL and EFH in the C1/C2 fields of the TAB1 that introduces the
  subsection, not in the record that follows, which is the trap in that law.
* LAW=1 with LANG=2 has NA=1, so the row stride is 3 rather than the 2 a
  Legendre representation of the same law would give.
* LAW=6 is a bare CONT record whose N2 is NPSX; nothing else in MF=6 is
  laid out that way.
* MF=13 has NK=2, so the total production cross section record is present; with
  NK=1 it is omitted, and that branch is the one a reader gets wrong.

    python tools/make_shapes_endf.py tests/synthetic-shapes.endf
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from endf_writer import Section, descriptive, write_material  # noqa: E402

#: MAT number of the synthetic evaluation. 9998 is unassigned.
MAT = 9998

#: A light-ish fictional target, so the reaction Q values are plausible.
ZA = 26056
AWR = 55.45414

#: The incident energy grid the cross sections are given on, in eV.
GRID = [1.0e-5, 1.0e3, 1.0e6, 2.0e7]


def mf1_nu(mt: int, values: list[float]) -> list[str]:
    """MF=1 MT=452 or MT=456: neutrons per fission, tabulated (LNU=2).

    A fission spectrum on MF=5 is only reachable through a fission reaction,
    and the reader builds that reaction's product from the yield here.
    """
    s = Section(MAT, 1, mt)
    s.cont(ZA, AWR, 0, 2, 0, 0)
    s.tab1(0.0, 0.0, 0, 0, [1.0e-5, 2.0e7], values)
    return s.finish()


def mf2_breit_wigner() -> list[str]:
    """MF=2 MT=151 with LRU=1, LRF=2: multi-level Breit-Wigner."""
    s = Section(MAT, 2, 151)
    s.cont(ZA, AWR, 0, 0, 1, 0)  # NIS=1
    s.cont(ZA, 1.0, 0, 0, 1, 0)  # ZAI, ABN, LFW=0, NER=1
    # EL, EH, LRU=1 resolved, LRF=2 multi-level BW, NRO=0, NAPS=0.
    s.cont(1.0e-5, 1.0e5, 1, 2, 0, 0)
    # SPI, AP, 0, 0, NLS=2, 0.
    s.cont(0.0, 0.54, 0, 0, 2, 0)

    # l = 0: three resonances, no competitive width. One row per resonance,
    # as the record lays them out.
    # fmt: off
    resonances_s = [
        # ER,   AJ,  GT,   GN,   GG,   GF
        1.1e3,  0.5, 1.34, 1.30, 0.04, 0.0,
        2.7e3,  0.5, 0.91, 0.87, 0.04, 0.0,
        7.6e3,  0.5, 2.15, 2.11, 0.04, 0.0,
    ]
    # fmt: on
    s.list_record(AWR, 0.0, 0, 0, len(resonances_s) // 6, resonances_s)

    # l = 1: two resonances, and a competitive reaction: QX and LRX are the
    # two fields a reader is most likely to drop.
    # fmt: off
    resonances_p = [
        3.4e4,  1.5, 5.02, 4.90, 0.12, 0.0,
        5.9e4,  0.5, 3.41, 3.30, 0.11, 0.0,
    ]
    # fmt: on
    s.list_record(AWR, 8.62e5, 1, 51, len(resonances_p) // 6, resonances_p)
    return s.finish()


def mf3(mt: int, qm: float, qi: float, values: list[float]) -> list[str]:
    """An MF=3 cross section, so the reactions the other files describe exist.

    A HEAD record and then the TAB1: the Q values are on the TAB1, not the
    HEAD, which is where MF=3 differs from the files that open with one record.
    """
    s = Section(MAT, 3, mt)
    s.cont(ZA, AWR, 0, 0, 0, 0)
    s.tab1(qm, qi, 0, 0, GRID, values)
    return s.finish()


def mf5_madland_nix() -> list[str]:
    """MF=5 MT=18 with LF=12: the Madland-Nix fission spectrum."""
    s = Section(MAT, 5, 18)
    s.cont(ZA, AWR, 0, 0, 1, 0)  # NK=1
    # EFL and EFH ride in C1 and C2 of the record that introduces the
    # subsection, which is what makes this law awkward; L2 is LF.
    s.tab1(1.029e6, 5.467e5, 0, 12, [1.0e-5, 2.0e7], [1.0, 1.0])
    # T_M against incident energy.
    s.tab1(0.0, 0.0, 0, 0, [1.0e-5, 1.0e6, 2.0e7], [1.092e6, 1.108e6, 1.301e6])
    return s.finish()


def mf6_kalbach_and_phase_space() -> list[str]:
    """MF=6 MT=16 with two products: LAW=1 LANG=2, then LAW=6."""
    s = Section(MAT, 6, 16)
    # ZA, AWR, JP=0, LCT=2 centre of mass, NK=2.
    s.cont(ZA, AWR, 0, 2, 2, 0)

    # Product 1: a neutron, LAW=1 with Kalbach-Mann angular parameters.
    s.tab1(1.0, 1.0, 0, 1, [1.2e7, 2.0e7], [2.0, 2.0])  # ZAP=1, LAW=1
    # LANG=2 Kalbach-Mann, LEP=2 lin-lin in secondary energy, NE=2.
    s.tab2(0.0, 0.0, 2, 2, 2)
    # At each incident energy: ND=0 discrete lines, NA=1 angular parameter, so
    # each outgoing energy carries E', f and a, a stride of three.
    # fmt: off
    s.list_record(0.0, 1.2e7, 0, 1, 3, [
        # E',   f,      a
        0.0,    0.0,    0.0,
        5.0e5,  1.4e-6, 0.21,
        1.0e6,  0.0,    0.33,
    ])
    s.list_record(0.0, 2.0e7, 0, 1, 3, [
        0.0,    0.0,    0.0,
        2.0e6,  3.9e-7, 0.48,
        8.0e6,  0.0,    0.94,
    ])
    # fmt: on

    # Product 2: the recoil, LAW=6, whose whole body is one CONT record.
    s.tab1(float(ZA - 1000), AWR - 1.0, 0, 6, [1.2e7, 2.0e7], [1.0, 1.0])
    # APSX, 0, 0, 0, 0, NPSX: the total mass and the number of particles.
    s.cont(AWR + 1.0, 0.0, 0, 0, 0, 3)
    return s.finish()


def mf13_photon_production() -> list[str]:
    """MF=13 MT=102: photon production as a cross section.

    NK=2, so the total production record is written before the two photons.
    """
    s = Section(MAT, 13, 102)
    s.cont(ZA, AWR, 0, 0, 2, 0)
    # The total, present only because NK > 1.
    s.tab1(0.0, 0.0, 0, 0, GRID, [4.4e-1, 1.4e-2, 3.1e-4, 8.0e-6])
    # A discrete line: EG is its energy, LP=0, LF=2 discrete.
    s.tab1(8.462e5, 0.0, 0, 2, GRID, [3.1e-1, 9.8e-3, 2.2e-4, 5.6e-6])
    # A continuum: EG=0, LF=1 tabulated, with the spectrum on MF=15.
    s.tab1(0.0, 0.0, 0, 1, GRID, [1.3e-1, 4.2e-3, 9.0e-5, 2.4e-6])
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
            10,  # NSUB=10, incident neutron
            [
                (1, 451, 9),
                (1, 452, 4),
                (1, 456, 4),
                (2, 151, 8),
                (3, 1, 4),
                (3, 16, 4),
                (3, 18, 4),
                (3, 102, 4),
                (5, 18, 6),
                (6, 16, 12),
                (13, 102, 10),
            ],
            " 26-Fe- 56",
            [
                "----SYNTHETIC         MATERIAL 9998",
                "-----THE SHAPES NO REAL FIXTURE REACHES",
                "------ENDF-6 FORMAT",
            ],
            author="tools/make_shapes_endf.py",
            # LRP=1, resonance parameters are given; LFI=1, the material fissions.
            lrp=1,
            lfi=1,
        ),
        mf1_nu(452, [2.51, 2.98]),
        mf1_nu(456, [2.44, 2.79]),
        mf2_breit_wigner(),
        mf3(1, 0.0, 0.0, [2.1e1, 1.4e1, 3.9e0, 2.5e0]),
        mf3(16, 0.0, -1.16e7, [0.0, 0.0, 0.0, 4.6e-1]),
        mf3(18, 1.86e8, 1.86e8, [1.2e-3, 4.0e-5, 9.0e-7, 1.1e0]),
        mf3(102, 7.646e6, 7.646e6, [2.6e0, 8.3e-2, 1.9e-3, 4.8e-5]),
        mf5_madland_nix(),
        mf6_kalbach_and_phase_space(),
        mf13_photon_production(),
    ]
    write_material(target, MAT, sections)
    print(
        f"{target}: MF2 LRF=2, MF5 LF=12, MF6 LANG=2 and LAW=6, MF13",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
