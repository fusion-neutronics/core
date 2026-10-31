"""Fetch and check the photon coefficient tabulations the contact dose folds against.

Two files under `crates/yamc-nuclide/src/data/photon_attenuation/`, with
different provenance stories:

`mass_energy_absorption_air_nist126.txt` is **fetched first-hand**. NIST SRD 126
publishes mu_en/rho for dry air near sea level as a static HTML table, so this
script downloads it, parses it and writes the file. Anyone can rerun it.

`mass_attenuation_xcom.txt` is **vendored, and checked here**. XCOM (NIST SRD 8)
serves mu/rho through a query form rather than as a download, so the file was
transcribed from the tabulation OpenMC ships (`openmc/data/dose/
mass_attenuation.h5`, MIT licensed), which is that database on its standard grid
from 1 keV to 20 MeV for Z = 1 to 100. What this script can do is hold it
against an independent NIST tabulation: SRD 126's per-element tables cover the
same coefficient for Z = 1 to 92 on a coarser grid, first-hand and static. So
`--check` compares the two everywhere they share an energy and reports the
worst disagreement, which is how the vendored file earns its place. Z = 93 to
100 have no SRD 126 tables and go unchecked.

    python tools/fetch_dose_data.py            # fetch air, write it, then check mu/rho
    python tools/fetch_dose_data.py --check    # compare both, write nothing

Absorption edges are skipped by the comparison: both tabulations record an edge
as two rows at one energy, and which row a reader lands on is a property of the
reader, not of the data.

Standard library only, so it runs anywhere without a data-science stack.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Where the tabulations live.
DATA = ROOT / "crates" / "yamc-nuclide" / "src" / "data" / "photon_attenuation"
AIR_FILE = DATA / "mass_energy_absorption_air_nist126.txt"
ATTENUATION_FILE = DATA / "mass_attenuation_xcom.txt"

#: NIST Standard Reference Database 126, X-Ray Mass Attenuation Coefficients.
NIST_AIR_URL = "https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/air.html"
NIST_ELEMENT_URL = "https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z{z:02d}.html"

#: SRD 126 tabulates the elements up to uranium; XCOM goes further.
NIST_ELEMENT_RANGE = range(1, 93)

#: The disagreement the vendored mu/rho is allowed against SRD 126. The two
#: are close rather than identical: they round to different digit counts, and
#: SRD 126 was published in 1995 against a later XCOM. Measured over the 3311
#: shared points, all but a handful agree to under 1%, and the worst is 2.49% at
#: Z=87 (francium) at 1 keV -- the lowest energy of an element with no stable
#: isotope, where the two evaluations have the least to go on. A 3% ceiling
#: passes that and still fails loudly if the file is ever replaced by something
#: that is not this database.
TOLERANCE = 0.03

#: Points past this are listed individually, so a rerun says where the two
#: tabulations part company rather than only how far.
NOTABLE = 0.01

AIR_HEADER = """Mass energy-absorption coefficients mu_en/rho for air, in units of cm^2/g.

Air (dry, near sea level). NIST Standard Reference Database 126: X-Ray Mass
Attenuation Coefficients, Table 4 (doi: 10.18434/T4D01F), fetched from
https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/air.html by
tools/fetch_dose_data.py. The K edge of argon is two rows at the same energy:
the first holds the value below the edge, the second the value above it.

Energy (eV)     mu_en/rho (cm^2/g)"""


#: physics.nist.gov answers 403 to the default urllib user agent.
USER_AGENT = "yamc-data-fetch (tools/fetch_dose_data.py)"


def download(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("latin-1")


def parse_nist_table(html: str) -> list[tuple[float, float, float]]:
    """Rows of `(energy MeV, mu/rho, mu_en/rho)` from a NIST SRD 126 page.

    The pages print their table twice, once per column layout; the copies are
    identical, so the first run of ascending energies is the whole table.
    """
    text = re.sub(r"<[^>]+>", " ", html)
    numbers = [float(token) for token in re.findall(r"\d\.\d+E[+-]\d+", text)]
    if len(numbers) % 3 != 0:
        raise ValueError(f"expected triples of numbers, parsed {len(numbers)}")
    rows = [tuple(numbers[i : i + 3]) for i in range(0, len(numbers), 3)]
    for i in range(1, len(rows)):
        if rows[i][0] < rows[i - 1][0]:
            return rows[:i]
    return rows


def read_attenuation(path: Path) -> dict[int, list[tuple[float, float]]]:
    """`{Z: [(energy eV, mu/rho cm^2/g), ...]}` from the packed table.

    One element per line: `Z <atomic number> <n> <n energies> <n coefficients>`.
    """
    tables: dict[int, list[tuple[float, float]]] = {}
    for line in path.read_text().splitlines():
        tokens = line.split()
        if not tokens or tokens[0] != "Z":
            continue
        z, count = int(tokens[1]), int(tokens[2])
        body = tokens[3:]
        if len(body) != 2 * count:
            raise ValueError(f"Z={z} declares {count} points but carries {len(body)} numbers")
        energies = [float(token) for token in body[:count]]
        values = [float(token) for token in body[count:]]
        tables[z] = list(zip(energies, values))
    return tables


def format_air(rows: list[tuple[float, float, float]]) -> str:
    lines = [f"{energy * 1e6:.10g}".rjust(11) + f"  {mu_en:.10g}" for energy, _mu, mu_en in rows]
    return AIR_HEADER + "\n" + "\n".join(lines) + "\n"


def fetch_air(check_only: bool) -> bool:
    """Fetch air's mu_en/rho and write it, or compare it against the file."""
    rows = parse_nist_table(download(NIST_AIR_URL))
    rendered = format_air(rows)
    if check_only:
        current = AIR_FILE.read_text()
        if current == rendered:
            print(f"air: {len(rows)} points, identical to {AIR_FILE.name}")
            return True
        print(f"air: FETCHED TABLE DIFFERS from {AIR_FILE.name}", file=sys.stderr)
        return False
    AIR_FILE.write_text(rendered)
    print(f"air: wrote {len(rows)} points to {AIR_FILE.name}")
    return True


def without_edges(rows: list[tuple[float, float]]) -> dict[float, float]:
    """Drop the energies a table records twice -- the absorption edges."""
    seen: dict[float, int] = {}
    for energy, _value in rows:
        seen[energy] = seen.get(energy, 0) + 1
    return {energy: value for energy, value in rows if seen[energy] == 1}


def check_attenuation() -> bool:
    """Hold the vendored XCOM mu/rho against NIST SRD 126, element by element."""
    vendored = read_attenuation(ATTENUATION_FILE)

    worst = (0.0, None)
    notable = []
    compared = 0
    for z in NIST_ELEMENT_RANGE:
        published = parse_nist_table(download(NIST_ELEMENT_URL.format(z=z)))
        ours = without_edges(vendored[z])
        theirs = without_edges([(energy * 1e6, mu) for energy, mu, _mu_en in published])
        for energy, mu in theirs.items():
            if energy not in ours:
                continue
            compared += 1
            difference = abs(ours[energy] - mu) / mu
            if difference > worst[0]:
                worst = (difference, (z, energy, ours[energy], mu))
            if difference > NOTABLE:
                notable.append((difference, z, energy))
        print(f"Z={z:3d} checked", end="\r", flush=True)

    difference, where = worst
    print(f"mu/rho: compared {compared} points over Z = 1 to 92" + " " * 20)
    if where is not None:
        z, energy, ours_value, theirs_value = where
        print(
            f"mu/rho: worst disagreement {difference * 100:.3f}% at Z={z}, "
            f"{energy:.6g} eV ({ours_value:.6g} vs {theirs_value:.6g})"
        )
    print(f"mu/rho: {len(notable)} points disagree by more than {NOTABLE * 100:.0f}%")
    for point_difference, z, energy in sorted(notable, reverse=True):
        print(f"          Z={z:3d}  {energy:>10.6g} eV  {point_difference * 100:.3f}%")
    if difference > TOLERANCE:
        print(f"mu/rho: EXCEEDS the {TOLERANCE * 100:.1f}% tolerance", file=sys.stderr)
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare against the files on disk and write nothing",
    )
    parser.add_argument(
        "--skip-attenuation",
        action="store_true",
        help="skip the mu/rho comparison, which downloads 92 pages",
    )
    arguments = parser.parse_args()

    ok = fetch_air(arguments.check)
    if not arguments.skip_attenuation:
        ok = check_attenuation() and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
