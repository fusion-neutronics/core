#!/usr/bin/env python3
"""Regenerate the bundled natural abundance table from TICE 2013.

Reads ``crates/yamc-nuclide/src/data/tice_2013_table1.csv``, a transcription
of Table 1 of Meija et al., "Isotopic compositions of the elements 2013 (IUPAC
Technical Report)", Pure Appl. Chem. 88(3), 293-306 (2016),
doi:10.1515/pac-2015-0503 (© IUPAC, De Gruyter 2016), and writes the
whitespace-delimited table that ``yamc-nuclide`` embeds with ``include_str!``.

Usage:
    python scripts/gen_natural_abundance.py

Which value an element expands with
-----------------------------------
The second column is the abundance every lookup uses. It is column 9, the
representative abundance CIAAW recommends for material of unspecified natural
origin, wherever column 9 is a value. For the 12 elements whose column 9 is an
interval (H, Li, B, C, N, O, Mg, Si, S, Cl, Br, Tl) there is no value to take,
so it is the column 6 best measurement. Mononuclidic elements are 1.

The remaining columns carry columns 4, 5, 6 and 9 of the row, so the
uncertainty ships with the value it belongs to. ``-`` marks a field the
table leaves empty; readers take it as "not stated", never as zero.

Output columns: nuclide, abundance, representative_value,
representative_uncertainty, representative_low, representative_high,
interval_low, interval_high, best_value, best_uncertainty, best_coverage,
best_calibration, annotations. See the CSV header for what each one is.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "crates/yamc-nuclide/src/data"
SOURCE = DATA / "tice_2013_table1.csv"
OUT = DATA / "natural_abundance.txt"

# TICE lists tantalum-180 by mass number; what occurs in nature is the
# long-lived isomer, which is the name yamc's nuclear data uses.
NAMES = {("Ta", "180"): "Ta180_m1"}

FIELDS = (
    "representative_value",
    "representative_uncertainty",
    "representative_low",
    "representative_high",
    "interval_low",
    "interval_high",
    "best_value",
    "best_uncertainty",
    "best_coverage",
    "best_calibration",
    "annotations",
)


def abundance(row: dict[str, str]) -> str:
    """The value lookups use, as the literal the table prints."""
    if row["representative_value"]:
        return row["representative_value"]
    if not (row["representative_low"] and row["representative_high"]):
        raise SystemExit(
            f"{row['element']}{row['a']}: column 9 is neither a value nor an interval"
        )
    return row["best_value"]


def main() -> int:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    rows = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    out = []
    for row in rows:
        name = NAMES.get((row["element"], row["a"]), f"{row['element']}{row['a']}")
        cells = [name, abundance(row)]
        for field in FIELDS:
            value = row[field]
            if " " in value:
                raise SystemExit(f"{name}: {field} {value!r} contains a space")
            cells.append(value or "-")
        out.append(" ".join(cells))
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} isotopes to {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
