#!/usr/bin/env python3
"""Regenerate the bundled PNNL material collection from the source report.

Parses the PNNL Compendium of Material Composition Data for Radiation
Transport Modeling (PNNL-15870, Rev. 2) and writes the pipe-delimited table
that ``yamc-materials`` embeds with ``include_str!``.

Usage:
    python scripts/gen_pnnl_collection.py                 # downloads the PDF
    python scripts/gen_pnnl_collection.py --pdf PNNL.pdf  # uses a local copy

Requires ``pdftotext`` (poppler-utils) on PATH.

How a composition is chosen
---------------------------
The report prints, for every material, a WEIGHT FRACTION column and an ATOM
FRACTION column, each in an "Isotopic" and an "Elemental" flavour. What gets
written here is the ELEMENTAL atom fractions verbatim, so every number is
quotable straight back to the report and no weight-to-atom conversion is done
by us.

The isotopic table is consulted only to decide, element by element, whether
that element is isotopically natural. Where it is, the ELEMENT SYMBOL is
written and yamc expands it with yamc's own abundance table. Where it is not
(enriched lithium and boron, uranium, plutonium, heavy water, He-3) the
individual NUCLIDES are written instead. See ``is_natural`` for the
tolerances and why they are what they are.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

PDF_URL = (
    "https://www.pnnl.gov/main/publications/external/technical_reports/"
    "PNNL-15870Rev2.pdf"
)
REPORT = "PNNL-15870, Rev. 2"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "crates/yamc-materials/src/collections/pnnl_15870_rev2.txt"
ABUNDANCE = REPO / "crates/yamc-nuclide/src/data/natural_abundance.txt"

# Entry 173 "Iron Boride (Fe2B)" is self-contradictory in the source report:
# its stated formula (Fe2B) and molecular weight (122.5035 = 2*55.845 +
# 10.811) are those of Fe2B, but every composition number in the row --
# atom fractions, weight fractions and atom densities alike -- describes
# FeB2 (B at 2/3, Fe at 1/3). There is no reading of the entry that is both
# faithful and correct, so it is omitted rather than shipped wrong.
# Entry 174 "Iron Boride (FeB)" is unaffected and is included.
OMITTED = {173: "formula/molecular weight (Fe2B) contradict the composition (FeB2)"}

# --- how "isotopically natural" is decided -------------------------------
# Isotopes are compared by their share WITHIN the element against three
# tolerances, the largest winning:
#
#   PRINT_QUANTUM/total  the report prints atom fractions to six decimal
#                        places, so an element present at total atom fraction
#                        `total` has its shares quantized to 1e-6/total. For a
#                        1e-4 trace element that alone is ~1% on every share.
#   SHARE_REL * expected PNNL and yamc use different vintages of the abundance
#                        tables (~1.5% apart on selenium, ~35% on the
#                        deuterium trace).
#   SHARE_ABS            a floor for isotopes with tiny natural shares.
#
# Rounding and table-vintage differences all land inside these. Genuine
# enrichment does not: depleted uranium is the tightest real case, with U235
# at 0.2% against a natural 0.72%, and it clears the tolerance five times
# over. Every element flagged by this test is one of Li, B, H, He, U, Pu, Am.
PRINT_QUANTUM = 1e-6
SHARE_ABS = 1e-3
SHARE_REL = 0.05

ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr"
).split()
Z_TO_SYM = {i + 1: s for i, s in enumerate(ELEMENTS)}

NUC = re.compile(r"^([A-Z][a-z]?)(\d+)(?:_m\d+|m)?$")
HEADER = re.compile(r"^(\d{1,3})\.\s+(\S.*?)\s*$")
DENSITY = re.compile(r"Density\s*\(g/cm3\)\s*=\s*([-\d.Ee+]+)")
ATOMDEN = re.compile(r"Total Atom Weight\s*\(atoms/b-cm\)\s*=\s*([-\d.Ee+]+)")
FORMULA = re.compile(r"Formula\s*=\s*(.*?)\s{2,}Molecular Weight")
ROW = re.compile(
    r"^\s*(\S+)\s+(\d{4,6})\s+(-?[\d.Ee+-]+)\s+(\d{4,6})\s+(-?[\d.Ee+-]+)"
    r"(?:\s+(\d{4,6})\s+(-?[\d.Ee+-]+))?\s*$"
)


def load_abundance() -> dict[str, dict[int, float]]:
    """yamc's natural abundances, grouped element -> {mass number: fraction}."""
    by_element: dict[str, dict[int, float]] = {}
    names: dict[tuple[str, int], str] = {}
    for line in ABUNDANCE.read_text(encoding="utf-8").splitlines():
        parts = line.split("#")[0].strip().split()
        if len(parts) < 2:
            continue
        m = NUC.match(parts[0])
        if not m:
            continue
        sym, a = m.group(1), int(m.group(2))
        by_element.setdefault(sym, {})[a] = float(parts[1])
        names[(sym, a)] = parts[0]
    return by_element, names


def extract_text(pdf: Path) -> str:
    """`pdftotext -layout` with the repeated page furniture stripped, so an
    entry that straddles a page break still parses as one block."""
    text = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    text = re.sub(r"^\s*200-DMAMC-128170\s*$", "", text, flags=re.M)
    text = re.sub(r"^\s*PNNL-15870, Rev\. 2\s*$", "", text, flags=re.M)
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.M)
    return text.replace("\f", "\n")


def parse_entries(text: str) -> list[dict]:
    lines = text.split("\n")
    starts = []
    for i, ln in enumerate(lines):
        m = HEADER.match(ln)
        if m and "...." not in ln and "Formula =" in "\n".join(lines[i + 1 : i + 4]):
            starts.append((i, int(m.group(1)), m.group(2)))

    entries = []
    for idx, (line_no, number, name) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        body = lines[line_no + 1 : end]
        blob = "\n".join(body)

        section = None
        blocks = {"Isotopic": {}, "Elemental": {}}
        for ln in body:
            s = ln.strip()
            if s in ("Isotopic", "Elemental"):
                section = s
                continue
            if s.startswith("Comments and References"):
                section = None
                continue
            if section:
                m = ROW.match(ln)
                if m:
                    zw, za, af = m.group(2), m.group(4), m.group(5)
                    assert zw == za, f"{name}: ZAID mismatch {zw} != {za}"
                    blocks[section][zw] = float(af)
                elif s:
                    section = None

        d = DENSITY.search(blob)
        f = FORMULA.search(blob)
        entries.append(
            {
                "number": number,
                "name": name,
                "density": float(d.group(1)) if d else None,
                "formula": (f.group(1).strip() if f else "") or "",
                "isotopic": blocks["Isotopic"],
                "elemental": blocks["Elemental"],
            }
        )
    return entries


def is_natural(sym: str, isos: dict[int, float], natural) -> bool:
    nat = natural.get(sym)
    if nat is None:
        return False  # Pu, Am, ... have no natural form
    if set(isos) == {0}:
        return True  # the report itself kept this element natural
    total = sum(isos.values())
    if total <= 0:
        return True  # zero-fraction trace; nothing to distinguish
    quantum = PRINT_QUANTUM / total
    for a in set(isos) | set(nat):
        got = isos.get(a, 0.0) / total
        expected = nat.get(a, 0.0)
        if abs(got - expected) > max(SHARE_ABS, SHARE_REL * expected, quantum):
            return False
    return True


def build_composition(entry, natural, yamc_names):
    iso_by_z: dict[int, dict[int, float]] = {}
    for zaid, frac in entry["isotopic"].items():
        z, a = int(zaid) // 1000, int(zaid) % 1000
        iso_by_z.setdefault(z, {})[a] = frac

    composition: dict[str, float] = {}
    order: list[str] = []
    done: set[str] = set()
    for zaid, frac in entry["elemental"].items():
        z = int(zaid) // 1000
        sym = Z_TO_SYM[z]
        isos = iso_by_z.get(z, {})
        if is_natural(sym, isos, natural):
            # The elemental table sometimes itemises one element across
            # several isotope rows (e.g. "Uranium, Natural"); those rows
            # accumulate into a single element key.
            if frac > 0:
                if sym not in composition:
                    order.append(sym)
                composition[sym] = composition.get(sym, 0.0) + frac
        elif sym not in done:
            done.add(sym)
            for a, f in sorted(isos.items()):
                if f <= 0:
                    continue
                name = yamc_names.get((sym, a), f"{sym}{a}")
                if name not in composition:
                    order.append(name)
                composition[name] = f
    return [(k, composition[k]) for k in order]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", type=Path, help="local copy of the Rev. 2 PDF")
    ap.add_argument("--json", type=Path, help="also dump the parsed entries here")
    args = ap.parse_args()

    pdf = args.pdf
    if pdf is None:
        pdf = Path("PNNL-15870Rev2.pdf")
        if not pdf.exists():
            print(f"downloading {PDF_URL}", file=sys.stderr)
            req = urllib.request.Request(PDF_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as r, pdf.open("wb") as fh:
                fh.write(r.read())

    natural, yamc_names = load_abundance()
    entries = parse_entries(extract_text(pdf))
    print(f"parsed {len(entries)} entries from {pdf}", file=sys.stderr)
    if len(entries) != 411:
        raise SystemExit(f"expected 411 entries in {REPORT}, got {len(entries)}")

    rows, skipped = [], []
    for e in entries:
        if e["number"] in OMITTED:
            skipped.append(e)
            continue
        if e["density"] is None:
            raise SystemExit(f"{e['name']}: no density")
        comp = build_composition(e, natural, yamc_names)
        if not comp:
            raise SystemExit(f"{e['name']}: empty composition")
        total = sum(v for _, v in comp)
        if abs(total - 1.0) > 5e-3:
            raise SystemExit(f"{e['name']}: atom fractions sum to {total:.6f}")
        assert "|" not in e["name"], e["name"]
        pairs = " ".join(f"{k} {v:.6g}" for k, v in comp)
        rows.append(f"{e['number']}|{e['name']}|{e['density']:.6g}|{e['formula']}|{pairs}")

    header = [
        "# Material compositions from the PNNL Compendium of Material Composition",
        f"# Data for Radiation Transport Modeling ({REPORT}), R.J. McConn Jr et al.,",
        "# Pacific Northwest National Laboratory, 2021.",
        f"# {PDF_URL}",
        "#",
        "# A US Department of Energy technical report; the compositions are",
        "# published reference data. yamc bundles them for lookup convenience and",
        "# claims no authorship of them.",
        "#",
        "# GENERATED FILE -- do not edit by hand.",
        "# Regenerate with: python scripts/gen_pnnl_collection.py",
        "#",
        "# Columns, pipe-delimited:",
        "#   number | name | density g/cm3 | formula | composition",
        "# Composition is space-delimited '<key> <atom fraction>' pairs, taken",
        "# verbatim from the report's Elemental atom-fraction column. Element",
        "# symbols are expanded by yamc's own natural abundances; explicit",
        "# nuclides appear only where the report's material is not natural.",
        "#",
    ]
    for e in skipped:
        header.append(f"# Omitted: entry {e['number']} {e['name']!r} -- {OMITTED[e['number']]}")
    header.append("#")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(header + rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} materials to {OUT}", file=sys.stderr)
    for e in skipped:
        print(f"omitted entry {e['number']}: {e['name']}", file=sys.stderr)

    if args.json:
        args.json.write_text(json.dumps(entries, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
