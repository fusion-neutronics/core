"""Build the Compton profile tabulation from its primary source.

The profiles are Biggs, Mendelsohn and Mann (1975). Geant4 distributes them in
the G4EMLOW dataset, which is the closest thing to a machine-readable original
that exists, so this fetches that dataset and parses it rather than converting
somebody else's already-processed copy. That gives the file first-hand
provenance and a path anyone can rerun.

Three files inside the dataset carry it:

    doppler/p-biggs.dat        the 31 projected momentum values, shared by all
    doppler/profile-<z>.dat    J(pz), one row per subshell
    doppler/shell-doppler.dat  per element, one "<electrons> <binding/MeV>"
                               line per subshell, terminated by a -1 line

Writes `packages/yamc-core/python/yamc/data/compton_profiles_biggs1975.txt` in
the same format `tools/convert_photon_data.py` produces, so the two are
interchangeable and either can check the other.

Binding energies are converted to eV on the way out, because both readers want
them that way and doing it once removes a chance to disagree. Floats are written
with `repr`, the shortest string that round-trips, so the parse is exact rather
than nearly exact.

    python tools/fetch_photon_data.py                      # download and write
    python tools/fetch_photon_data.py --check              # compare, write nothing
    python tools/fetch_photon_data.py --tarball G4EMLOW.6.48.tar.gz

Standard library only, so it runs anywhere without a data-science stack.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Where the tabulation is written.
DEFAULT_OUTPUT = ROOT / "packages" / "yamc-core" / "python" / "yamc" / "data" / "compton_profiles_biggs1975.txt"

#: The G4EMLOW release the vendored data was built from. Pinned rather than
#: floating: a later release would silently change every profile, and the point
#: of this script is to say exactly which numbers are in the file.
DEFAULT_VERSION = "6.48"

#: geant4.cern.ch redirects here. Resolved once and hard-coded so a redirect
#: chain that stops working is a clear failure rather than a mystery.
BASE_URL = "https://cern.ch/geant4-data/datasets/"

#: The archive this script was written against. Verified rather than recorded:
#: building a data file from an archive nobody checked is how a silent change
#: gets in.
KNOWN_SHA256 = {
    "6.48": "9815be88cbbcc4e8855b20244d586552a8b1819b8bf4e538c342b27c17dff1c7",
    "9.0": "413fbfee2a5bfbe3ca70e5947bdb3c8dc95dc8e90eb6636be96588363820c31f",
}

#: The data covers Z = 1 to 100.
MAX_Z = 100

#: The profiles are tabulated against 31 momentum values.
N_PZ = 31

#: eV per MeV, matching `endf::data::EV_PER_MEV`.
EV_PER_MEV = 1.0e6


def floats(values) -> str:
    return " ".join(repr(float(v)) for v in values)


def display(path: Path) -> str:
    """`path` relative to the repository when it is inside it, else as given."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def download(url: str, dest: Path) -> None:
    print(f"downloading {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as response, dest.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)


def read_members(tarball: Path, version: str) -> dict[str, str]:
    """The `doppler/` files, by base name, decoded as text.

    Read straight out of the archive rather than extracting it: the dataset is
    25 MB and only a few hundred kB of it is wanted.
    """
    prefix = f"G4EMLOW{version}/doppler/"
    wanted = {"p-biggs.dat", "shell-doppler.dat"}
    wanted |= {f"profile-{z}.dat" for z in range(1, MAX_Z + 1)}

    found: dict[str, str] = {}
    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar:
            if not member.name.startswith(prefix):
                continue
            name = member.name[len(prefix) :]
            if name not in wanted:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            found[name] = handle.read().decode()

    missing = sorted(wanted - set(found))
    if missing:
        raise SystemExit(
            f"{tarball}: {len(missing)} expected file(s) absent under {prefix}, "
            f"first few: {missing[:5]}"
        )
    return found


def shells(text: str) -> list[list[tuple[float, float]]]:
    """Split `shell-doppler.dat` into one list of (electrons, binding) per element.

    The file is a flat sequence of element blocks, each closed by a `-1` line,
    with a final `-2` line ending the data. Nothing in it names the element, so
    the blocks are in atomic number order by construction and the count is
    checked below.
    """
    elements: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        marker = float(fields[0])
        if marker == -2.0:
            break
        if marker == -1.0:
            elements.append(current)
            current = []
            continue
        if len(fields) < 2:
            raise SystemExit(f"shell-doppler.dat: malformed line {line!r}")
        current.append((marker, float(fields[1])))
    if current:
        raise SystemExit("shell-doppler.dat: the last element block is unterminated")
    return elements


def header(version: str, digest: str) -> list[str]:
    """Name the source in the file, so it does not need this script to explain it."""
    return [
        "# Compton profiles J(pz) per subshell, with subshell occupancies and",
        "# binding energies in eV. Z = 1 to 100 on a 31-point momentum grid.",
        "#",
        "# Biggs, Mendelsohn and Mann, At. Data Nucl. Data Tables 16 (1975) 201,",
        "# as distributed in the Geant4 G4EMLOW data set under doppler/.",
        "#",
        f"# Built by tools/fetch_photon_data.py from G4EMLOW {version}",
        f"# sha256 {digest}",
    ]


def build(members: dict[str, str], version: str, digest: str) -> str:
    pz = [float(t) for t in members["p-biggs.dat"].split()]
    if len(pz) != N_PZ:
        raise SystemExit(f"p-biggs.dat: expected {N_PZ} momentum values, got {len(pz)}")

    per_element = shells(members["shell-doppler.dat"])
    if len(per_element) != MAX_Z:
        raise SystemExit(
            f"shell-doppler.dat: expected {MAX_Z} element blocks, got {len(per_element)}"
        )

    out = header(version, digest)
    out += ["COMPTON", f"pz {len(pz)} {floats(pz)}"]
    for z in range(1, MAX_Z + 1):
        subshells = per_element[z - 1]
        nss = len(subshells)

        j = [float(t) for t in members[f"profile-{z}.dat"].split()]
        if len(j) != nss * N_PZ:
            raise SystemExit(
                f"profile-{z}.dat: {len(j)} values is not {nss} subshells "
                f"by {N_PZ} momenta; shell-doppler.dat and the profiles disagree"
            )

        num_electrons = [n for n, _ in subshells]
        # Converted here so neither reader has to remember to.
        binding = [b * EV_PER_MEV for _, b in subshells]
        out.append(f"Z {z} {nss} {floats(num_electrons)} {floats(binding)} {floats(j)}")

    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--version", default=DEFAULT_VERSION, help=f"G4EMLOW release (default {DEFAULT_VERSION})"
    )
    parser.add_argument(
        "--tarball",
        type=Path,
        help="use this local G4EMLOW archive instead of downloading one",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare with what is on disk and write nothing; exits non-zero on a difference",
    )
    args = parser.parse_args()

    tarball = args.tarball
    downloaded = None
    if tarball is None:
        name = f"G4EMLOW.{args.version}.tar.gz"
        downloaded = Path(name)
        if not downloaded.is_file():
            download(BASE_URL + name, downloaded)
        tarball = downloaded

    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    expected = KNOWN_SHA256.get(args.version)
    if expected is None:
        print(
            f"{tarball}: sha256 {digest} (no pin recorded for G4EMLOW "
            f"{args.version}; add one to KNOWN_SHA256 if this becomes the default)",
            file=sys.stderr,
        )
    elif digest != expected:
        raise SystemExit(
            f"{tarball} is not the archive this script was written against.\n"
            f"  expected {expected}\n  got      {digest}\n"
            "Refusing to build a data file from an unverified archive."
        )
    else:
        print(f"{tarball}: sha256 {digest}, matches the pin", file=sys.stderr)

    text = build(read_members(tarball, args.version), args.version, digest)

    if args.check:
        if not args.output.is_file():
            raise SystemExit(f"{args.output}: missing, nothing to check against")
        current = args.output.read_text()
        if current == text:
            print(f"{display(args.output)} matches G4EMLOW {args.version}", file=sys.stderr)
            return
        raise SystemExit(
            f"{display(args.output)} differs from what G4EMLOW "
            f"{args.version} produces"
        )

    args.output.write_text(text)
    print(
        f"{display(args.output)}: {args.output.stat().st_size / 1e3:.0f} kB "
        f"from G4EMLOW {args.version}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
