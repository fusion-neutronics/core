# SPDX-License-Identifier: MIT
"""Write a synthetic evaluation with unresolved resonance Cases A and B.

These are the two shapes issue #15 made unreachable. Both readers dispatched on
LRF where the format uses LRU, so a range with LRU=2 and LRF=1 matched neither
branch: its records were left on the stream, and the *next* range was then read
from the middle of it. Case C (LRU=2, LRF=2) worked only because the two flags
happen to coincide there.

No real evaluation small enough to keep here has Case A or Case B, so the
fixture is built:

* **Isotope 1, LFW=0**: a Case A range (all parameters energy-independent),
  followed by a resolved Breit-Wigner range. The second range is the point: if
  the first is skipped without consuming its records, the second is parsed from
  the wrong offset and comes back with LRU=0 and its EL/EH holding the first
  range's SPI and AP.
* **Isotope 2, LFW=1**: a Case B range, where only the fission widths are
  energy-dependent, so the parameters arrive one LIST per J value with the
  energy grid in a separate record.

Two L values in Case A with different J counts, so a reader that reuses NJS is
caught as well.

    python tools/make_urr_endf.py tests/synthetic-urr-cases.endf
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from endf_writer import Section, descriptive, write_material  # noqa: E402

#: MAT number of the synthetic evaluation. 9997 is unassigned.
MAT = 9997

ZA = 94239.0
AWR = 236.9986

#: The energies the Case B fission widths are tabulated on, in eV.
CASE_B_ENERGIES = [1.0e3, 1.0e4, 3.0e4]


def mf2() -> list[str]:
    s = Section(MAT, 2, 151)
    s.cont(ZA, AWR, 0, 0, 2, 0)  # NIS=2

    # ---- Isotope 1: LFW=0, so Case A, then a resolved range after it -------
    s.cont(ZA, 0.7, 0, 0, 2, 0)  # ZAI, ABN, LFW=0, NER=2

    # Range 1: LRU=2 unresolved, LRF=1 -> Case A.
    s.cont(1.0e3, 3.0e4, 2, 1, 0, 0)
    s.cont(0.5, 0.94, 0, 0, 2, 0)  # SPI, AP, LSSF=0, NLS=2
    # Six values per J -- D, AJ, AMUN, GNO, GG and a spare the format reserves.
    # l=0, two J values.
    # fmt: off
    s.list_record(AWR, 0.0, 0, 0, 2, [
        8.9, 0.5, 1.0, 0.001, 0.040, 0.0,
        4.4, 1.5, 1.0, 0.002, 0.040, 0.0,
    ])
    # l=1, three J values, so a reader that reuses NJS from l=0 is caught.
    s.list_record(AWR, 0.0, 1, 0, 3, [
        3.1, 0.5, 1.0, 0.003, 0.041, 0.0,
        2.7, 1.5, 2.0, 0.004, 0.041, 0.0,
        1.9, 2.5, 1.0, 0.005, 0.041, 0.0,
    ])
    # fmt: on

    # Range 2: resolved multi-level Breit-Wigner. Reading this correctly is
    # only possible if the range above consumed exactly its own records.
    s.cont(3.0e4, 1.0e5, 1, 2, 0, 0)
    s.cont(0.5, 0.94, 0, 0, 1, 0)
    # fmt: off
    s.list_record(AWR, 0.0, 0, 0, 1, [
        5.5e4, 0.5, 1.10, 1.00, 0.10, 0.0,
    ])
    # fmt: on

    # ---- Isotope 2: LFW=1, so Case B ---------------------------------------
    s.cont(ZA + 1.0, 0.3, 0, 1, 1, 0)  # ZAI, ABN, LFW=1, NER=1
    s.cont(1.0e3, 3.0e4, 2, 1, 0, 0)  # LRU=2, LRF=1

    # With LFW=1 and LRF=1 the spin and radius arrive on a LIST whose values
    # are the energy grid, rather than on the CONT that the other cases use.
    # SPI, AP, LSSF=0, 0, NE, NLS=1
    s.list_record(0.5, 0.94, 0, 0, 1, CASE_B_ENERGIES)

    # One CONT per L, then one LIST per J.
    s.cont(AWR, 0.0, 0, 0, 2, 0)  # AWRI, 0, L=0, 0, NJS=2, 0
    # MUF in L2; values are D, AJ, AMUN, GN0, GG, 0, then GF per energy.
    # fmt: off
    s.list_record(0.0, 0.0, 0, 1, 0, [
        8.9, 0.5, 1.0, 0.001, 0.040, 0.0,
        0.010, 0.012, 0.015,
    ])
    s.list_record(0.0, 0.0, 0, 2, 0, [
        4.4, 1.5, 1.0, 0.002, 0.040, 0.0,
        0.020, 0.022, 0.025,
    ])
    # fmt: on
    return s.finish()


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    target = Path(sys.argv[1])

    write_material(
        target,
        MAT,
        [
            descriptive(
                MAT,
                ZA,
                AWR,
                10,
                [(1, 451, 9), (2, 151, 16)],
                " 94-Pu-239",
                [
                    "----SYNTHETIC         MATERIAL 9997",
                    "-----UNRESOLVED RESONANCE CASES A AND B",
                    "------ENDF-6 FORMAT",
                ],
                author="tools/make_urr_endf.py",
                lrp=1,
            ),
            mf2(),
        ],
    )
    print(
        f"{target}: unresolved Case A and Case B, plus a resolved range after A",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
