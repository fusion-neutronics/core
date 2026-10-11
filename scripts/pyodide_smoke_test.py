"""Smoke test of the yamc-core Pyodide wheel, run inside Pyodide.

Usage (from a `pyodide venv` with the wheel installed)::

    python scripts/pyodide_smoke_test.py <fixture-cache-dir>

`<fixture-cache-dir>` is the cache `scripts/fetch_test_fixtures.py` fills
(`~/.cache/yamc`). Pyodide under Node has no sockets and no synchronous
XMLHttpRequest, so the data fetcher installed here answers each download from
those files instead of the network. It serves them as a release of
`endf-b8.1` the way the origin does (`latest.json`, then the release's
`manifest.json`, then each file), so release resolution, verification against
the manifest, the cache and the transport run the same code they run in the
browser. Only the transfer is local.
"""

import hashlib
import json
import os
import sys

ORIGIN = "https://yamc-data.xsplot.com/"
KEYWORD = "endf-b8.1"
RELEASE = "2026-01-01"
# The nuclides the model below is made of, the only ones the release carries.
NUCLIDES = ("Li6",)


def publish(fixtures):
    """The release the fetcher serves: origin-relative path to bytes.

    Args:
        fixtures: The directory of fixture sections, laid out as
            `<library>-<name>.arrow/<section>`.

    Returns:
        Every object of the release, keyed by its path under the origin.
    """
    objects = {}
    files = []
    for name in NUCLIDES:
        folder = os.path.join(fixtures, f"{KEYWORD}-{name}.arrow")
        for section in sorted(os.listdir(folder)):
            # `.absent` markers record a 404 upstream; the manifest just
            # leaves those sections out.
            if section.endswith(".absent"):
                continue
            with open(os.path.join(folder, section), "rb") as f:
                body = f.read()
            path = f"neutron/{name}.arrow/{section}"
            objects[f"{KEYWORD}/{RELEASE}/{path}"] = body
            files.append(
                {"path": path, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
            )
    manifest = json.dumps(
        {"keyword": KEYWORD, "release": RELEASE, "format_version": 2, "files": files}
    ).encode()
    objects[f"{KEYWORD}/{RELEASE}/manifest.json"] = manifest
    objects[f"{KEYWORD}/latest.json"] = json.dumps(
        {
            "release": RELEASE,
            "format_version": 2,
            "manifest": f"{RELEASE}/manifest.json",
            "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "manifest_bytes": len(manifest),
        }
    ).encode()
    return objects


def main(fixtures):
    """Build a small model, run transport on fixture data, and check the result.

    Args:
        fixtures: The directory of fixture sections, laid out as
            `<library>-<name>.arrow/<section>`.
    """
    # The cache is in Pyodide's own filesystem, empty at start, so every
    # file the run needs goes through the fetcher.
    import yamc
    from yamc import _core

    objects = publish(fixtures)
    requests = []

    def fetch(url, range_header):
        """Answer one GET from the published release, as the origin would."""
        requests.append((url, range_header))
        if range_header is not None:
            # A keyword load downloads every file whole, to verify it.
            return 416, b""
        body = objects.get(url.removeprefix(ORIGIN)) if url.startswith(ORIGIN) else None
        if body is None:
            return 404, b""
        return 200, body

    _core._set_data_fetcher(fetch)

    yamc.cross_section_data = "endf-b8.1"
    lithium = yamc.Material(
        composition={"Li6": 1.0}, density=0.46, temperature=294, name="Li6"
    )
    sphere = yamc.Sphere(radius=10.0, boundary="vacuum")
    cell = yamc.Cell(region=sphere.below, material=lithium, name="li6 sphere")
    geometry = yamc.Geometry(cells=[cell])
    source = yamc.NeutronSource(energy=14.06e6, position=(0.0, 0.0, 0.0))
    # MT 105 is (n,t): one tritium atom per reaction.
    tritium = yamc.Tally(cells=cell, scores=[105], name="tritium production")
    model = yamc.Model(geometry=geometry, tallies=[tritium], source=source)

    results = model.simulate_transport(total_particles=2_000, seed=1)
    mean = results[tritium].mean[0]
    print(f"tritium production: {mean:.4e} per source neutron")
    print(f"{len(requests)} requests")

    if not requests:
        sys.exit("the data fetcher was never called")
    if requests[0][0] != f"{ORIGIN}{KEYWORD}/latest.json":
        sys.exit(f"the first request was {requests[0][0]}, not the release pointer")
    release = results.runs[0]["data_releases"][KEYWORD]["release"]
    if release != RELEASE:
        sys.exit(f"the run recorded release {release}, not {RELEASE}")
    if not 0.0 < mean < 1.0:
        sys.exit(f"tritium production {mean} is outside (0, 1)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
