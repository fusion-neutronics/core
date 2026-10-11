"""Smoke test of the yamc-core Pyodide wheel, run inside Pyodide.

Usage (from a `pyodide venv` with the wheel installed)::

    python scripts/pyodide_smoke_test.py <fixture-cache-dir>

`<fixture-cache-dir>` is the cache `scripts/fetch_test_fixtures.py` fills
(`~/.cache/yamc`). Pyodide under Node has no sockets and no synchronous
XMLHttpRequest, so the data fetcher installed here answers each download from
those files instead of the network. It honours the `Range` header the way the
CDN does, including the `multipart/byteranges` answer to a header naming
several ranges, so the ranged temperature download, the cache and the transport
run the same code they run in the browser. Only the transfer is local.
"""

import os
import sys
import tempfile

ORIGIN = "https://yamc-data.xsplot.com/"
BOUNDARY = "yamc-smoke-test-boundary"


def main(fixtures):
    """Build a small model, run transport on fixture data, and check the result.

    Args:
        fixtures: The directory of fixture sections, laid out as
            `<library>-<name>.arrow/<section>`.
    """
    # A fresh cache in Pyodide's own filesystem, so every section the run
    # needs goes through the fetcher rather than being found on disk.
    os.environ["YAMC_CACHE_DIR"] = tempfile.mkdtemp()

    import yamc
    from yamc import _core

    requests = []

    def fetch(url, range_header):
        """Answer one GET from the fixture files, as the CDN would."""
        requests.append((url, range_header))
        if not url.startswith(ORIGIN):
            return 404, b""
        # `endf-b8.1/neutron/Li6.arrow/reactions.arrow` is cached by the
        # fixture script as `endf-b8.1-Li6.arrow/reactions.arrow`.
        library, _kind, name, section = url[len(ORIGIN) :].split("/")
        path = os.path.join(fixtures, f"{library}-{name}", section)
        if not os.path.isfile(path):
            return 404, b""
        with open(path, "rb") as f:
            body = f.read()
        if range_header is None:
            return 200, body
        spans = []
        for spec in range_header.removeprefix("bytes=").split(","):
            first, last = (int(n) for n in spec.split("-"))
            spans.append((first, last))
        if len(spans) == 1:
            first, last = spans[0]
            return 206, body[first : last + 1]
        parts = []
        for first, last in spans:
            parts.append(
                f"--{BOUNDARY}\r\n"
                f"Content-Type: application/octet-stream\r\n"
                f"Content-Range: bytes {first}-{last}/{len(body)}\r\n\r\n".encode()
                + body[first : last + 1]
                + b"\r\n"
            )
        return 206, b"".join(parts) + f"--{BOUNDARY}--\r\n".encode()

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
    print(f"{len(requests)} requests, {sum(r is not None for _, r in requests)} ranged")

    if not requests:
        sys.exit("the data fetcher was never called")
    if not any(r is not None and "," in r for _, r in requests):
        sys.exit("no request named several ranges, so the ranged download did not run")
    if not 0.0 < mean < 1.0:
        sys.exit(f"tritium production {mean} is outside (0, 1)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
