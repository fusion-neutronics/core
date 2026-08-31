# SPDX-License-Identifier: MIT
"""Convert the auxiliary photon data out of HDF5 into a plain text form.

`compton_profiles.h5` and `density_effect.h5` are the only reason this package
needs `h5py`, and HDF5 is the only reason the Rust crate could not read them --
it has no dependencies at all, and an HDF5 reader would mean a C library. The
data itself is a few thousand floats; the container is the problem, not the
contents.

So both are rewritten as one whitespace-separated text file. It is left
uncompressed, unlike the fixtures: the Rust crate reads this one at *runtime*,
and xz would mean a decompressor in a crate that deliberately has no
dependencies. Git packs it well enough. The format is deliberately dull,
because two readers have to agree on it:

    COMPTON
    pz <count> <values...>
    Z <z> <nss> <num_electrons...> <binding_energy...> <J row-major...>
    ...
    DENSITY
    Z <z> <nss> <I> <num_electrons...> <ionization_energy...>
    ...

`binding_energy` is written in eV, already multiplied by EV_PER_MEV, since both
readers want it that way and doing it once removes a chance to disagree. `J` is
row-major, `nss` rows of `len(pz)`.

Floats are written with `repr`, the shortest string that round-trips, so the
conversion is exact rather than nearly exact.

    python tools/convert_photon_data.py

Writes one file per tabulation into `packages/yamc-core/python/yamc/data`, each
named for what it holds and where it came from, because the two HDF5 files hold
unrelated measurements and only shared a container:

    compton_profiles_biggs1975.txt
    density_effect_sternheimer1982.txt

The HDF5 originals are not in this repository. Point `--from` at a checkout of
shimwell/endf-python, whose `src/endf/datafiles` holds them for the Python
reader. See `packages/yamc-core/python/yamc/data/README.md`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Where the converted tabulations are written.
DATA = ROOT / "packages" / "yamc-core" / "python" / "yamc" / "data"

#: The data covers Z = 1 to 100.
MAX_Z = 100

#: eV per MeV, matching `endf.data.EV_PER_MEV`.
EV_PER_MEV = 1.0e6


def floats(values) -> str:
    return " ".join(repr(float(v)) for v in values)


def display(path: Path) -> str:
    """`path` relative to the repository when it is inside it, else as given."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


#: Named in the file itself, so it does not need this script to explain it.
DENSITY_HEADER = [
    "# Mean excitation energy I, subshell occupancies and subshell ionization",
    "# energies in eV. Z = 1 to 100.",
    "#",
    "# NIST ESTAR mean excitation energies with Sternheimer, Berger and Seltzer",
    "# (1982). Vendored rather than fetched: ESTAR publishes through a web form,",
    "# so there is no archive to download and verify. See the README.md beside this file.",
    "#",
    "# Built by tools/convert_photon_data.py from density_effect.h5.",
]


def cross_check(target: Path, lines: list[str]) -> None:
    """Compare the HDF5-derived Compton data with the shipped file.

    Header lines are excluded: the two routes name different sources, which is
    the whole point. Everything below the header must agree exactly.
    """
    if not target.is_file():
        raise SystemExit(f"{display(target)}: missing, nothing to check against")
    shipped = [
        line for line in target.read_text().splitlines() if not line.startswith("#")
    ]
    derived = [line for line in lines if not line.startswith("#")]
    if shipped == derived:
        print(
            f"{display(target)}: agrees with the HDF5 line for line "
            f"({len(derived)} lines)",
            file=sys.stderr,
        )
        return
    first = next(
        (i for i, (a, b) in enumerate(zip(shipped, derived)) if a != b), min(len(shipped), len(derived))
    )
    raise SystemExit(
        f"{display(target)} disagrees with the HDF5 at data line {first}: "
        "the Geant4 route and the vendored copy are no longer the same data"
    )


def write(target: Path, lines: list[str]) -> None:
    target.write_text("\n".join(lines) + "\n")
    print(
        f"{display(target)}: {target.stat().st_size / 1e3:.0f} kB",
        file=sys.stderr,
    )


def main() -> None:
    import argparse

    import h5py

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from",
        dest="source",
        type=Path,
        required=True,
        metavar="DIR",
        help="directory holding compton_profiles.h5 and density_effect.h5, "
        "i.e. src/endf/datafiles in a shimwell/endf-python checkout",
    )
    args = parser.parse_args()

    out = []

    with h5py.File(args.source / "compton_profiles.h5", "r") as f:
        pz = f["pz"][()]
        out.append("COMPTON")
        out.append(f"pz {len(pz)} {floats(pz)}")
        for z in range(1, MAX_Z + 1):
            group = f[f"{z:03}"]
            num_electrons = group["num_electrons"][()]
            # Converted here so neither reader has to remember to.
            binding = group["binding_energy"][()] * EV_PER_MEV
            j = group["J"][()]
            nss = len(num_electrons)
            assert j.shape == (nss, len(pz)), f"Z={z}: unexpected J shape {j.shape}"
            out.append(
                f"Z {z} {nss} {floats(num_electrons)} {floats(binding)} "
                f"{floats(j.reshape(-1))}"
            )

    # The Compton half is not written. tools/fetch_photon_data.py builds the
    # shipped file from the Geant4 distribution, which is first-hand; this
    # route exists to check it, and the two agreeing is what proves the HDF5
    # held that dataset and nothing else.
    cross_check(DATA / "compton_profiles_biggs1975.txt", out)
    out = []

    with h5py.File(args.source / "density_effect.h5", "r") as f:
        out.append("DENSITY")
        for z in range(1, MAX_Z + 1):
            group = f[f"{z:03}"]
            num_electrons = group["num_electrons"][()]
            ionization = group["ionization_energy"][()]
            nss = len(num_electrons)
            assert len(ionization) == nss, f"Z={z}: ragged density effect data"
            out.append(
                f"Z {z} {nss} {float(group.attrs['I'])!r} "
                f"{floats(num_electrons)} {floats(ionization)}"
            )

    write(DATA / "density_effect_sternheimer1982.txt", DENSITY_HEADER + out)


if __name__ == "__main__":
    main()
