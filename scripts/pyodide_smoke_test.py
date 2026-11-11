"""Smoke test of the yamc-core Pyodide wheel, run inside Pyodide.

Usage (from a `pyodide venv` with the wheel installed)::

    python scripts/pyodide_smoke_test.py <fixture-cache-dir>

`<fixture-cache-dir>` is the cache `scripts/fetch_test_fixtures.py` fills
(`~/.cache/yamc`). Pyodide under Node has no sockets and no synchronous
XMLHttpRequest, so the data fetcher installed here answers each download from
those files instead of the network. It serves them as a release of
`endf-b8.1` the way the origin does (`latest.json`, then the release's
`manifest.json`, then each file), so release resolution, the batched
prefetch, verification against the manifest, the cache and the transport run
the same code they run in the browser. Only the transfer is local. The pointer
is asked for with a cache-busting query string, which the origin answers by
path as the real one does, and nothing else carries one.

The fetcher takes a batch of requests per call, as the browser's does: the
prefetch sends a nuclide's whole files together in one call, then the byte
ranges its index names in another.

The model is at one temperature, so `reactions.arrow` and `energy.arrow` are
fetched as byte ranges, each checked against the sha256 its `version.json`
index entry carries. The fixtures may predate those hashes, so the release
served here adds them to each `version.json` (what the converter's
`index_reactions` pass does to a published tree) before the manifest is built,
and the fetcher answers `Range` requests as the origin does: one range as the
bytes alone, several as `multipart/byteranges`.

`publish`, `origin` and `check_release_requests` are shared with
`pyodide_smoke_test_yani.py`, which serves the yani-core wheel the same release
with the transmutation chain added.
"""

import hashlib
import json
import os
import sys

ORIGIN = "https://yamc-data.xsplot.com/"
KEYWORD = "endf-b8.1"
RELEASE = "2026-01-01"
# How every request for the pointer starts: a cache-busting query follows, so
# the CDN's edge cannot answer it with an old copy (`pointer_url` in
# yamc-nuclide's release.rs).
POINTER = f"{ORIGIN}{KEYWORD}/latest.json?release-check="
# The nuclides the model below is made of, the only ones the release carries.
NUCLIDES = ("Li6",)


def add_hashes(version, sections):
    """Give every byte-range index entry the sha256 of its bytes.

    Args:
        version: The parsed `version.json`, changed in place.
        sections: Section file name to its bytes.
    """
    for key, section, nested in (
        ("reaction_ranges", "reactions.arrow", True),
        ("energy_ranges", "energy.arrow", False),
    ):
        index = version.get(key)
        if not index or section not in sections:
            continue
        body = sections[section]
        groups = index["mts"].values() if nested else [index["temperatures"]]
        entries = [index["schema"]] + [e for group in groups for e in group.values()]
        for entry in entries:
            offset, length = entry[0], entry[1]
            digest = hashlib.sha256(body[offset : offset + length]).hexdigest()
            entry[2:] = [digest]


def answer_range(body, header):
    """The 206 body for a `Range` header, as the origin would send it.

    Args:
        body: The whole object.
        header: The `Range` header value, `bytes=0-99,200-299`.

    Returns:
        The bytes of a single range, or a `multipart/byteranges` body for
        several.
    """
    spans = []
    for part in header.removeprefix("bytes=").split(","):
        first, last = (int(x) for x in part.split("-"))
        spans.append((first, last))
    if len(spans) == 1:
        first, last = spans[0]
        return body[first : last + 1]
    boundary = b"yamc-smoke-boundary"
    out = b""
    for first, last in spans:
        out += (
            b"\r\n--" + boundary + b"\r\nContent-Type: application/octet-stream\r\n"
            + f"Content-Range: bytes {first}-{last}/{len(body)}\r\n\r\n".encode()
            + body[first : last + 1]
        )
    return out + b"\r\n--" + boundary + b"--\r\n"


def publish(fixtures, nuclides=NUCLIDES, chain=None):
    """The release the fetcher serves: origin-relative path to bytes.

    Args:
        fixtures: The directory of fixture sections, laid out as
            `<library>-<name>.arrow/<section>`.
        nuclides: The neutron nuclides the release carries.
        chain: The transmutation chain fixture to publish as the release's
            `transmutation/` subsections, a directory under `fixtures` laid
            out as `<subsection>/<section>`, or None for no chain.

    Returns:
        Every object of the release, keyed by its path under the origin.
    """
    objects = {}
    files = []

    def add(path, body):
        objects[f"{KEYWORD}/{RELEASE}/{path}"] = body
        files.append({"path": path, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})

    for name in nuclides:
        folder = os.path.join(fixtures, f"{KEYWORD}-{name}.arrow")
        sections = {}
        for section in sorted(os.listdir(folder)):
            # `.absent` markers record a 404 upstream; the manifest just
            # leaves those sections out.
            if section.endswith(".absent"):
                continue
            with open(os.path.join(folder, section), "rb") as f:
                sections[section] = f.read()
        # Hashes go into version.json before the manifest pins it.
        version = json.loads(sections["version.json"])
        add_hashes(version, sections)
        sections["version.json"] = json.dumps(version, indent=2).encode()
        for section, body in sections.items():
            add(f"neutron/{name}.arrow/{section}", body)
    if chain is not None:
        root = os.path.join(fixtures, chain)
        for subsection in sorted(os.listdir(root)):
            folder = os.path.join(root, subsection)
            if not os.path.isdir(folder):
                continue
            for section in sorted(os.listdir(folder)):
                with open(os.path.join(folder, section), "rb") as f:
                    add(f"transmutation/{subsection}.arrow/{section}", f.read())
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


def origin(objects):
    """A data fetcher answering from `objects` as the origin would.

    Args:
        objects: The release, as `publish` returns it.

    Returns:
        `(fetch, requests, batches)`: the fetcher to install, the list of
        `(url, range)` it has been asked for, and the size of each batch.
    """
    requests = []
    batches = []

    def fetch_one(url, range_header):
        """Answer one GET from the published release, as the origin would.

        By path: the query string a request for the pointer carries is for
        the CDN's cache, and does not change the object.
        """
        requests.append((url, range_header))
        path = url.removeprefix(ORIGIN).split("?", 1)[0]
        body = objects.get(path) if url.startswith(ORIGIN) else None
        if body is None:
            return 404, b""
        if range_header is not None:
            return 206, answer_range(body, range_header)
        return 200, body

    def fetch(batch):
        """Answer a batch of GETs, one `(status, body)` per `(url, range)`."""
        batches.append(len(batch))
        return [fetch_one(url, range_header) for url, range_header in batch]

    return fetch, requests, batches


def check_release_requests(requests):
    """Exit unless the run asked for the pointer first, with a cache-busting
    query, and for everything else plainly.

    Args:
        requests: The `(url, range)` pairs the fetcher was asked for, in order.
    """
    if not requests:
        sys.exit("the data fetcher was never called")
    first = requests[0][0]
    bust = first.removeprefix(POINTER)
    if bust == first or len(bust) != 16 or any(c not in "0123456789abcdef" for c in bust):
        sys.exit(
            f"the first request was {first}, not the release pointer with a "
            "cache-busting query"
        )
    queried = [url for url, _ in requests[1:] if "?" in url]
    if queried:
        sys.exit(f"only the pointer carries a query string, but so did {queried}")


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

    fetch, requests, batches = origin(publish(fixtures))
    _core._set_data_fetcher(fetch)

    yamc.neutron_cross_section_data = "endf-b8.1"
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
    print(
        f"{len(requests)} requests in {len(batches)} calls, "
        f"{sum(r is not None for _, r in requests)} ranged"
    )

    check_release_requests(requests)
    release = results.runs[0]["data_releases"][KEYWORD]["release"]
    if release != RELEASE:
        sys.exit(f"the run recorded release {release}, not {RELEASE}")
    if max(batches) < 2:
        sys.exit("no call carried more than one request, so the batched prefetch did not run")
    if not 0.0 < mean < 1.0:
        sys.exit(f"tritium production {mean} is outside (0, 1)")
    # The model is at 294 K, so the two indexed sections are fetched as the
    # byte ranges of that temperature, verified against the hashes, and never
    # whole.
    for section in ("reactions.arrow", "energy.arrow"):
        asked = [r for u, r in requests if u.endswith(f"Li6.arrow/{section}")]
        if not asked or any(r is None for r in asked):
            sys.exit(f"{section} was not fetched as byte ranges: {asked}")
        print(f"{section}: {len(asked)} ranged request(s), {asked[0]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
