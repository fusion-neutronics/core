#!/usr/bin/env python3
"""Fetch the nuclear-data fixtures the test suite reads.

The fixtures under ``crates/yamc/tests/*.arrow`` used to be committed binaries
(~530 MB, re-committed on every library migration). They are published data, not
source, so this script downloads them into the same cache directory production
uses (the platform cache directory: ``~/.cache/yamc`` on Linux,
``~/Library/Caches/yamc`` on macOS, ``%LOCALAPPDATA%\\yamc`` on Windows) and
symlinks them into place. The fixtures keep their own flat layout there
(``endf-b8.1-<Name>.arrow``), apart from the per-release folders the runtime
downloader writes (``endf-b8.1/<release>/...``).

They come from the release layout the runtime reads: ``<keyword>/latest.json``
names the current release, its ``manifest.json`` is checked against the
pointer's size and sha256, and every file is downloaded from
``<keyword>/<release>/<path>`` and checked against its manifest entry.

Usage::

    python scripts/fetch_test_fixtures.py                  # fetch what is missing or stale
    python scripts/fetch_test_fixtures.py --check          # report, download nothing
    python scripts/fetch_test_fixtures.py --force          # re-download everything
    python scripts/fetch_test_fixtures.py --print-release  # print the current release id

A cached file whose sha256 matches the manifest is left alone, so re-running
is cheap: the pointer is one request, the manifest is read back from the cache
while the pointer still names it, and the rest is hashing local files. A file
whose sha256 differs is downloaded again, and an optional section the release
no longer lists is removed and marked absent, so a new release reaches the
fixtures rather than leaving a mix of two releases behind a symlink a test is
reading.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import pathlib
import shutil
import sys
import time
import urllib.error
import urllib.request

ORIGIN = "https://yamc-data.xsplot.com"
LIBRARY = "endf-b8.1"
USER_AGENT = "yamc-fetch-test-fixtures/2.0"
# The release format this build reads (SUPPORTED_FORMAT_VERSIONS in
# crates/yamc-nuclide/src/storage/release.rs).
FORMAT_VERSION = 2

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "crates" / "yamc" / "tests"

# (filename, required) per option-D section directory, mirroring
# NEUTRON_SECTIONS / PHOTON_SECTIONS in crates/yamc-nuclide/src/storage/url_cache.rs.
#
# An optional section the release manifest does not list gets the same
# zero-byte `.absent` marker the runtime cache writes. The loader reads the
# marker as "this nuclide has no such section", and without it a full-scope
# read of a fixture cannot distinguish that from a half-downloaded directory
# and fails.
ABSENT_SUFFIX = ".absent"  # crates/yamc-nuclide/src/storage/url_cache.rs

NEUTRON_SECTIONS = [
    ("version.json", True),
    ("nuclide.arrow", True),
    ("energy.arrow", True),
    ("reactions.arrow", True),
    ("products.arrow", True),
    ("distributions.arrow", True),
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
# Transmutation subsections, and the section files fetched from each. The
# release also publishes `branching`, which is left out: the tests that need
# branching rows read the ones committed under tests/fixtures, and the chain
# loader reads the subsection whenever manifest.json names it, so adding it
# would put an isomeric-branching overlay under every test that loads this
# chain.
CHAIN_SECTIONS = {
    "decay": [
        ("provenance.json", True),
        ("nuclides.arrow", True),
        ("decay_modes.arrow", False),
        ("sources.arrow", False),
    ],
    "reactions": [("provenance.json", True), ("reactions.arrow", True)],
    "fission_yields": [
        ("provenance.json", True),
        ("fission_yields.arrow", True),
        ("aliases.arrow", False),
        ("evaluated_yields.arrow", False),
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
    # per-channel fission chi and the partial-fission-MT draw.
    "U240",
    # Carries the MT 91 continuum-inelastic correlated angle-energy law that
    # crates/yamc/tests/correlated_flat_reference_parity.rs samples 4M times
    # through both the reference sampler and the flattened path the CPU
    # transport actually calls, catching any divergence between the two.
    # Without it here that test self-skips. At about 27 MB it is the largest
    # entry in the set, and it buys the only coverage the correlated sampler
    # has.
    "W184",
    # Needed by `f19_correlated_sampler_matches_legacy_product`
    # (crates/yamc-gpu/src/neutron/xs/distributions.rs), which otherwise takes
    # its "no F19 cache" skip. It is the sole regression guard for the MT 16
    # multi-applicability collapse. About 5 MB, and it samples MT 16/22/28/91
    # two million times through both the legacy product and the flattened path.
    "F19",
    # Th232 is one of only three ENDF/B-VIII.1 fissionables whose prompt
    # fission spectrum is a `CorrelatedAngleEnergy` (ENDF File 6 LAW 1), and
    # `crates/yamc/tests/gpu_th232_correlated_chi.rs` is the guard for the fix
    # that taught the CPU's `prompt_chi_dist` to flatten that encoding. Without
    # it here the test's non-GPU half self-skips.
    #
    # It is also the second fissionable in the set, which is what
    # `crates/yamc/tests/mixed_fissile_yield_fixture.rs` needs: the per-nuclide
    # nu_bar / beta rows would otherwise need a U235 / U238 pair, and those two
    # are 370 MB. Th232 with the U240 already here gives
    # the same extractor guard for nothing further.
    #
    # 54 MB, the second largest entry. The other actinides stay out: U235
    # (191 MB), U238 (179 MB) and Pu239 (78 MB) are each larger than the rest
    # of this list put together, and nothing needs them that Th232 + U240 does
    # not now cover.
    "Th232",
    # ENDF/B-VIII.1 lists only the isomer for In115 (n,gamma) and (n,2n) and
    # for Mo92 (n,p), the ground state being the remainder, and
    # crates/yani-transmute/tests/isomer_only_branching.rs checks the split
    # against the library itself, with the branching rows committed beside it.
    # About 17 MB and 3 MB.
    "In115", "Mo92",
    # The cross sections crates/yani-transmute/tests/resonance_parameter_sampling.rs
    # rebuilds W186's resolved range against: its MF=2 and MF=32 are committed
    # (crates/endf/fixtures), and the test compares the capture rate's sigma
    # from sampled parameters with the first-order rows'. About 20 MB.
    "W186",
]
ELEMENTS = ["Be", "Fe", "Li"]
CHAIN_FIXTURE = "transmutation-endf-b8.1-sfr"


def cache_root() -> pathlib.Path:
    """The platform cache directory with ``yamc`` appended, the rule
    ``cache_root`` in crates/yamc-nuclide/src/storage/url_cache.rs applies
    (through ``etcetera``'s native strategy). Kept in step by hand because this
    script runs before yamc is built."""
    if sys.platform == "darwin":
        return pathlib.Path.home() / "Library" / "Caches" / "yamc"
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        base = pathlib.Path(local) if local else pathlib.Path.home() / "AppData" / "Local"
        return base / "yamc"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = pathlib.Path(xdg) if xdg and os.path.isabs(xdg) else pathlib.Path.home() / ".cache"
    return base / "yamc"


# How many times a section download is attempted before the script gives up,
# and the base delay between attempts (doubled each time).
#
# A 404 is an answer and is never retried; this is for the transport dropping
# underneath a transfer, which a runner does hit: a windows-latest job took
# `ConnectionResetError: [WinError 10054] An existing connection was forcibly
# closed by the remote host` partway through a section and failed the whole
# job. The sections are tens of megabytes each and the cache is cold on every
# key change, so one reset in a few hundred requests is enough to be a regular
# red build.
FETCH_ATTEMPTS = 4
FETCH_BACKOFF_SECONDS = 2.0


def read_url(request: urllib.request.Request) -> bytes:
    """The body at ``request``, retrying a failed transport.

    A reset, timeout or truncated body is the connection failing rather than
    the origin answering, so it is retried with backoff. An HTTP status is the
    origin's answer and is raised to the caller as the ``HTTPError`` it is.
    """
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return response.read()
        # `HTTPError` is a `URLError`, so it is let through before the
        # transport failures are caught.
        except urllib.error.HTTPError:
            raise
        # `http.client.HTTPException` is the body going wrong after the headers
        # arrived: `IncompleteRead` when the connection drops mid-transfer, which
        # is the shape a reset takes once `read()` has started. It is not a
        # `URLError`, so without it here the retry only covered a reset during
        # the connect.
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            ConnectionError,
        ) as exc:
            if attempt == FETCH_ATTEMPTS:
                raise SystemExit(
                    f"failed to fetch {request.full_url} after {FETCH_ATTEMPTS} attempts: {exc}"
                ) from exc
            delay = FETCH_BACKOFF_SECONDS * 2 ** (attempt - 1)
            print(
                f"{request.full_url}: {exc}; retrying in {delay:.0f}s "
                f"({attempt}/{FETCH_ATTEMPTS})"
            )
            time.sleep(delay)
    raise AssertionError("unreachable: the last attempt returns or exits")


def get(url: str) -> bytes:
    """The body at ``url``. Any HTTP status but a 2xx is the origin answering,
    so it ends the run rather than being retried."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        return read_url(request)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"failed to fetch {url}: {exc}")


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: pathlib.Path) -> str | None:
    """The sha256 of the file at ``path``, or None when there is no file."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(what: str, payload: bytes, entry: dict) -> None:
    """Refuse ``payload`` unless its size and sha256 are the ones ``entry``
    (a manifest entry, or the pointer's record of the manifest) gives."""
    if len(payload) != entry["bytes"] or sha256_hex(payload) != entry["sha256"]:
        raise SystemExit(
            f"{what}: got {len(payload)} bytes with sha256 {sha256_hex(payload)}, "
            f"expected {entry['bytes']} bytes with sha256 {entry['sha256']}"
        )


def write_atomically(dest: pathlib.Path, payload: bytes) -> None:
    """Write ``payload`` beside ``dest`` and rename it into place, so a test
    reading through the symlink never sees half a file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    tmp.write_bytes(payload)
    os.replace(tmp, dest)


def fetch_pointer() -> dict:
    """The library's ``latest.json``, which names the current release.

    The CDN caches the pointer, so the request carries a query string it has
    not seen and is answered by the origin, as
    ``origin_serves_a_readable_release_for_every_keyword`` does.
    """
    bust = int(time.time())
    url = f"{ORIGIN}/{LIBRARY}/latest.json?release-check={bust}"
    try:
        pointer = json.loads(get(url))
    except ValueError as exc:
        raise SystemExit(f"{url} is not a valid release pointer: {exc}")
    if pointer.get("keyword") != LIBRARY:
        raise SystemExit(f"{url} names keyword {pointer.get('keyword')!r}, not {LIBRARY!r}")
    if pointer.get("format_version") != FORMAT_VERSION:
        raise SystemExit(
            f"{url} names release {pointer.get('release')} at format_version "
            f"{pointer.get('format_version')}, and this build reads {FORMAT_VERSION}"
        )
    return pointer


def release_files(pointer: dict, cache: pathlib.Path, check: bool) -> dict:
    """The files of the release ``pointer`` names, by path.

    The manifest is checked against the pointer's size and sha256 and kept
    beside the fixtures, so a warm run reads it back instead of fetching it
    again while the pointer still names it.
    """
    record = {"bytes": pointer["manifest_bytes"], "sha256": pointer["manifest_sha256"]}
    cached = cache / f"{LIBRARY}-fixtures-manifest.json"
    payload = cached.read_bytes() if cached.is_file() else b""
    if len(payload) != record["bytes"] or sha256_hex(payload) != record["sha256"]:
        url = f"{ORIGIN}/{LIBRARY}/{pointer['manifest']}"
        payload = get(url)
        verify(url, payload, record)
        if not check:
            write_atomically(cached, payload)
    manifest = json.loads(payload)
    expected = {
        "keyword": LIBRARY,
        "release": pointer["release"],
        "format_version": FORMAT_VERSION,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise SystemExit(
                f"manifest {pointer['manifest']} has {field} {manifest.get(field)!r}, "
                f"expected {value!r}"
            )
    return {entry["path"]: entry for entry in manifest["files"]}


def sync_sections(
    prefix: str,
    dest_dir: pathlib.Path,
    sections,
    files: dict,
    release: str,
    force: bool,
    check: bool,
) -> list[str]:
    """Bring ``dest_dir`` in line with the release's ``<prefix>/<file>``
    entries, and return the paths that needed it.

    A file whose sha256 matches its manifest entry is left alone; any other is
    downloaded and verified. An optional section the manifest does not list is
    removed and marked absent, a required one ends the run. With ``check``
    nothing is changed and the list is only a report.
    """
    changed = []
    for filename, required in sections:
        path = f"{prefix}/{filename}"
        dest = dest_dir / filename
        marker = dest.with_name(dest.name + ABSENT_SUFFIX)
        entry = files.get(path)
        if entry is None:
            if required:
                raise SystemExit(
                    f"release {release} of {LIBRARY} does not list {path}, "
                    "which the fixtures need"
                )
            if dest.exists() or not marker.exists():
                changed.append(path)
                if not check:
                    # Recorded the way the runtime cache records a section the
                    # release does not publish. Without the marker the loader
                    # cannot tell "this nuclide has no total_nu" from "this
                    # fixture is half-downloaded", and a full-scope read fails
                    # with a bare NotFound (Fe58, which has no total_nu, did
                    # exactly that).
                    dest.unlink(missing_ok=True)
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    marker.write_bytes(b"")
            continue
        if not force and not marker.exists() and sha256_file(dest) == entry["sha256"]:
            continue
        changed.append(path)
        if check:
            continue
        url = f"{ORIGIN}/{LIBRARY}/{release}/{path}"
        payload = get(url)
        verify(url, payload, entry)
        write_atomically(dest, payload)
        # A section that used to be absent and is now published must lose its
        # marker, or the loader keeps believing it is not there.
        marker.unlink(missing_ok=True)
    return changed


def chain_manifest() -> str:
    """The manifest.json the chain loader reads, naming the subsections. The
    published layout has one directory per subsection and no such file."""
    return json.dumps(
        {
            "format_version": FORMAT_VERSION,
            "library": LIBRARY,
            "subsections": {name: {"path": name} for name in CHAIN_SECTIONS},
        },
        indent=2,
    ) + "\n"


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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="report only")
    parser.add_argument("--force", action="store_true", help="re-download everything")
    parser.add_argument(
        "--print-release",
        action="store_true",
        help="print the release latest.json names and exit (the CI cache key)",
    )
    args = parser.parse_args()

    pointer = fetch_pointer()
    release = pointer["release"]
    if args.print_release:
        print(release)
        return 0

    cache = cache_root()
    files = release_files(pointer, cache, args.check)
    # (name, cache directory, [(release path prefix, subdirectory, sections)])
    # per fixture. The chain fixture is one directory per subsection.
    plan = [
        (n, cache / f"{LIBRARY}-{n}.arrow", [(f"neutron/{n}.arrow", "", NEUTRON_SECTIONS)])
        for n in NUCLIDES
    ]
    plan += [
        (e, cache / f"{LIBRARY}-{e}.arrow", [(f"photon/{e}.arrow", "", PHOTON_SECTIONS)])
        for e in ELEMENTS
    ]
    plan.append((
        CHAIN_FIXTURE,
        cache / f"{LIBRARY}-{CHAIN_FIXTURE}.arrow",
        [(f"transmutation/{sub}.arrow", sub, s) for sub, s in CHAIN_SECTIONS.items()],
    ))

    missing, stale, changed = [], [], 0
    for name, dest, parts in plan:
        was_there = dest.is_dir()
        paths = []
        for prefix, subdir, sections in parts:
            paths += sync_sections(
                prefix, dest / subdir, sections, files, release, args.force, args.check
            )
        if name == CHAIN_FIXTURE:
            manifest = dest / "manifest.json"
            if not manifest.is_file() or manifest.read_text() != chain_manifest():
                paths.append(str(manifest))
                if not args.check:
                    manifest.write_text(chain_manifest())
        if not was_there:
            missing.append(name)
        elif paths:
            stale.append(name)
        changed += len(paths)
        if not args.check:
            link(FIXTURE_DIR / f"{name}.arrow", dest)

    if args.check:
        print(f"{len(plan) - len(missing)}/{len(plan)} fixtures cached in {cache}")
        if missing:
            print("missing: " + ", ".join(missing))
        if stale:
            print(f"differing from release {release}: " + ", ".join(stale))
        return 1 if missing or stale else 0

    print(
        f"{len(plan)} fixtures of {LIBRARY} release {release} ready in {cache} "
        f"({changed} files downloaded or marked absent), linked into {FIXTURE_DIR}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
