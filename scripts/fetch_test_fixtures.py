#!/usr/bin/env python3
"""Fetch the nuclear-data fixtures the test suite reads.

The fixtures under ``crates/yamc/tests/*.arrow`` used to be committed binaries
(~530 MB, re-committed on every library migration). They are published data, not
source, so this script downloads them into the same on-disk cache production
uses (``~/.cache/yamc``, or ``$YAMC_CACHE_DIR``) and symlinks them into place.

Usage::

    python scripts/fetch_test_fixtures.py            # fetch what is missing
    python scripts/fetch_test_fixtures.py --check    # report, download nothing
    python scripts/fetch_test_fixtures.py --force    # re-download everything

Already-cached sections are left alone, so re-running is cheap. See issue #126.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys
import urllib.error
import urllib.request

ORIGIN = "https://yamc-data.xsplot.com"
LIBRARY = "endf-b8.1"
USER_AGENT = "yamc-fetch-test-fixtures/1.0"

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "crates" / "yamc" / "tests"

# (filename, required) per option-D section directory, mirroring
# NEUTRON_SECTIONS / PHOTON_SECTIONS in crates/yamc-nuclide/src/storage/url_cache.rs.
#
# An optional section the origin 404s gets the same zero-byte `.absent` marker
# the runtime cache writes (issue #389). It is not just about re-asking: the
# loader reads the marker as "this nuclide has no such section", and without it
# a full-scope read of a fixture cannot distinguish that from a half-downloaded
# directory and fails.
ABSENT_SUFFIX = ".absent"  # crates/yamc-nuclide/src/storage/url_cache.rs

NEUTRON_SECTIONS = [
    ("version.json", True),
    ("nuclide.arrow", True),
    ("reactions.arrow", True),
    ("products.arrow", False),
    ("distributions.arrow", False),
    ("fast_xs.arrow", False),
    ("urr.arrow", False),
    ("total_nu.arrow", False),
    ("fission_photon.arrow", False),
]
PHOTON_SECTIONS = [
    ("version.json", True),
    ("element.arrow", True),
    ("subshells.arrow", False),
    ("compton.arrow", False),
    ("bremsstrahlung.arrow", False),
]
# Transmutation subsections, and the section files published in each.
CHAIN_SECTIONS = {
    "decay": [
        ("nuclides.arrow", True),
        ("decay_modes.arrow", False),
        ("sources.arrow", False),
        ("provenance.json", False),
    ],
    "reactions": [("reactions.arrow", True), ("provenance.json", False)],
    "fission_yields": [
        ("fission_yields.arrow", True),
        ("aliases.arrow", False),
        ("provenance.json", False),
    ],
}

NUCLIDES = [
    "Al27", "B10", "Be9", "C12", "Co58", "Cr52", "Fe54", "Fe56", "Fe57",
    "Fe58", "H2", "Li6", "Li7", "O16", "Pb208",
    # The breeder-blanket benchmark (crates/yani-transmute/tests/tritium_breeder.rs)
    # runs Li4SiO4 + Be9 + steel + water, which needs silicon and light
    # hydrogen on top of the set above. 7.4 MB for the four.
    "H1", "Si28", "Si29", "Si30",
    # The only ENDF/B-VIII.1 nuclide with non-redundant partial fission
    # channels (MT 19/20/21/38, each with its own prompt spectrum, while its
    # MT 18 is redundant and carries no neutron product). It covers the
    # per-channel fission chi of issue #425 and the partial-fission-MT draw of
    # issue #418, and it is the set's only fissionable.
    "U240",
    # Carries the MT 91 continuum-inelastic correlated angle-energy law that
    # crates/yamc/tests/correlated_flat_reference_parity.rs samples 4M times
    # through both the reference sampler and the flattened path the CPU
    # transport actually calls. Without it here that test self-skips, which it
    # has done in CI since it was written (issue #371 is the regression it
    # exists to catch, and it went unguarded). At about 27 MB it is the largest
    # entry in the set, and it buys the only coverage the correlated sampler
    # has.
    "W184",
    # The only nuclide any test resolves through `yamc_test_cache::nuclide`
    # that was not in this list, so
    # `f19_correlated_sampler_matches_legacy_product`
    # (crates/yamc-gpu/src/neutron/xs/distributions.rs) has taken its "no F19
    # cache" skip on every run since it was written. It is the sole regression
    # guard for the MT 16 multi-applicability collapse that issue #155 was
    # filed about, so until now that fix has been unpinned. About 5 MB, and it
    # samples MT 16/22/28/91 two million times through both the legacy product
    # and the flattened path.
    "F19",
]
ELEMENTS = ["Be", "Fe", "Li"]
CHAIN_FIXTURE = "transmutation-endf-b8.1-sfr"


def cache_root() -> pathlib.Path:
    env = os.environ.get("YAMC_CACHE_DIR")
    if env:
        return pathlib.Path(env)
    return pathlib.Path.home() / ".cache" / "yamc"


def fetch(url: str, dest: pathlib.Path, required: bool, force: bool) -> str:
    """Fetch one section. Returns 'cached', 'downloaded', or 'absent'."""
    marker = dest.with_name(dest.name + ABSENT_SUFFIX)
    if (dest.exists() or marker.exists()) and not force:
        return "cached"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and not required:
            # Record the 404 the way the runtime cache does. Without this the
            # loader cannot tell "this nuclide has no total_nu" from "this
            # fixture is half-downloaded", and a full-scope read fails with a
            # bare NotFound (Fe58, which has no total_nu, did exactly that).
            dest.parent.mkdir(parents=True, exist_ok=True)
            marker.write_bytes(b"")
            return "absent"
        raise SystemExit(f"failed to fetch {url}: {exc}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    # A section that used to be absent and is now published must lose its
    # marker, or the loader keeps believing the 404.
    marker.unlink(missing_ok=True)
    return "downloaded"


def fetch_sections(base_url: str, dest_dir: pathlib.Path, sections, force: bool) -> int:
    downloaded = 0
    for filename, required in sections:
        state = fetch(f"{base_url}/{filename}", dest_dir / filename, required, force)
        downloaded += state == "downloaded"
    return downloaded


def fetch_chain(dest_dir: pathlib.Path, force: bool) -> int:
    """Assemble the transmutation chain fixture from its published subsections.

    The published layout is one directory per subsection; the local fixture adds
    the manifest.json that names them, which the chain loader reads.
    """
    downloaded = 0
    for subsection, sections in CHAIN_SECTIONS.items():
        downloaded += fetch_sections(
            f"{ORIGIN}/{LIBRARY}/transmutation/{subsection}.arrow",
            dest_dir / subsection,
            sections,
            force,
        )
    manifest = dest_dir / "manifest.json"
    if not manifest.exists() or force:
        manifest.write_text(
            json.dumps(
                {
                    "format_version": 2,
                    "library": LIBRARY,
                    "subsections": {
                        name: {"path": name} for name in CHAIN_SECTIONS
                    },
                },
                indent=2,
            )
            + "\n"
        )
        downloaded += 1
    return downloaded


def link(fixture: pathlib.Path, target: pathlib.Path) -> None:
    """Point crates/yamc/tests/<name>.arrow at its cache directory.

    A symlink where the platform allows one (nothing is duplicated on disk),
    a copy on Windows, where creating a symlink needs developer mode or
    elevation and raises otherwise.
    """
    if fixture.is_symlink():
        if fixture.resolve() == target.resolve():
            return
        fixture.unlink()
    elif fixture.is_dir():
        # A previous run copied it (Windows), or it is a leftover committed
        # fixture. Either way the cache is the source of truth now.
        shutil.rmtree(fixture)
    elif fixture.exists():
        fixture.unlink()
    try:
        fixture.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        shutil.copytree(target, fixture)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report only")
    parser.add_argument("--force", action="store_true", help="re-download everything")
    args = parser.parse_args()

    cache = cache_root()
    plan = [(n, f"{ORIGIN}/{LIBRARY}/neutron/{n}.arrow", NEUTRON_SECTIONS) for n in NUCLIDES]
    plan += [(e, f"{ORIGIN}/{LIBRARY}/photon/{e}.arrow", PHOTON_SECTIONS) for e in ELEMENTS]

    if args.check:
        missing = [
            name for name, _, _ in plan
            if not (cache / f"{LIBRARY}-{name}.arrow").is_dir()
        ]
        if not (cache / f"{LIBRARY}-{CHAIN_FIXTURE}.arrow").is_dir():
            missing.append(CHAIN_FIXTURE)
        print(f"{len(plan) + 1 - len(missing)}/{len(plan) + 1} fixtures cached in {cache}")
        if missing:
            print("missing: " + ", ".join(missing))
        return 1 if missing else 0

    downloaded = 0
    for name, base_url, sections in plan:
        dest = cache / f"{LIBRARY}-{name}.arrow"
        downloaded += fetch_sections(base_url, dest, sections, args.force)
        link(FIXTURE_DIR / f"{name}.arrow", dest)

    chain_dest = cache / f"{LIBRARY}-{CHAIN_FIXTURE}.arrow"
    downloaded += fetch_chain(chain_dest, args.force)
    link(FIXTURE_DIR / f"{CHAIN_FIXTURE}.arrow", chain_dest)

    print(
        f"{len(plan) + 1} fixtures ready in {cache} "
        f"({downloaded} section files downloaded), linked into {FIXTURE_DIR}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
