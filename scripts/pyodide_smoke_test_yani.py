"""Smoke test of the yani-core Pyodide wheel, run inside Pyodide.

Usage (from a `pyodide venv` with the wheel installed)::

    python scripts/pyodide_smoke_test_yani.py <fixture-cache-dir>

The yani counterpart of `pyodide_smoke_test.py`, and served the same way: the
fixtures `scripts/fetch_test_fixtures.py` fills (`~/.cache/yamc`) are published
to the data fetcher as a release of `endf-b8.1` (`latest.json`, the release's
`manifest.json`, then each file), here with the transmutation chain fixture as
the release's `transmutation/` subsections. An iron material is irradiated for
a year and cooled, and its activity and decay heat are read back, so release
resolution, the chain download, the batched prefetch of the closure's cross
sections, the byte-range reads verified against the manifest and the solve run
the code they run in a browser. Only the transfer is local.

The release carries only the fixture nuclides. Most of iron's closure is not
among them, which the solver tolerates for daughters (they decay, but do not
react), so the numbers here are a smoke test of the wheel rather than a
reference inventory. Fe56(n,p)Mn56 is in it, and Mn56 is what leads the
activity at shutdown.
"""

import sys

from pyodide_smoke_test import (
    KEYWORD,
    ORIGIN,
    RELEASE,
    check_release_requests,
    origin,
    publish,
)

# What the material is made of and what its closure reaches among the
# fixtures. The rest of the closure is absent from this release.
NUCLIDES = ("Fe54", "Fe56", "Fe57", "Fe58", "Co58", "Cr52")
CHAIN = f"{KEYWORD}-transmutation-endf-b8.1-sfr.arrow"


def main(fixtures):
    """Irradiate iron on fixture data, and check the activity and decay heat.

    Args:
        fixtures: The directory of fixture sections, laid out as
            `<library>-<name>.arrow/<section>`, with the chain fixture beside
            them.
    """
    # The cache is in Pyodide's own filesystem, empty at start, so every
    # file the run needs goes through the fetcher.
    import yani
    from yani import _core

    fetch, requests, batches = origin(publish(fixtures, NUCLIDES, CHAIN))
    _core._set_data_fetcher(fetch)

    yani.neutron_cross_section_data = KEYWORD
    yani.transmutation_decay_data = KEYWORD
    yani.transmutation_reactions = KEYWORD
    yani.transmutation_fission_yields = KEYWORD
    # The chain fixture has no branching subsection.
    yani.transmutation_branch_ratios = None

    iron = yani.Material(
        composition={"Fe56": 1.0}, density=7.87, volume=1.0, temperature=294, name="iron"
    )
    # A 14 MeV peak over a slowed-down tail. Edges in eV.
    edges = [1e-5, 0.625, 1e3, 1e5, 1e6, 1e7, 13.5e6, 14.5e6]
    shares = [0.05, 0.10, 0.15, 0.20, 0.15, 0.05, 0.30]
    spectrum = yani.NeutronSource(energy=yani.sources.Histogram(edges, shares))
    schedule = yani.PulseSchedule(
        [
            yani.Pulse(rate=1e14, duration=(1, "a"), source=spectrum),
            yani.Cooldown(duration=(1, "h")),
            yani.Cooldown(duration=(1, "a")),
        ]
    )
    results = iron.transmute(schedule=schedule)
    steps = results.step_materials(iron.id)

    print("step   activity (Bq)   decay heat (W)   largest")
    for i, material in enumerate(steps):
        by_nuclide = material.activity(by_nuclide=True)
        top = max(by_nuclide, key=by_nuclide.get) if by_nuclide else "-"
        print(f"{i:4d}   {material.activity():13.4e}   {material.decay_heat():14.4e}   {top}")
    print(
        f"{len(requests)} requests in {len(batches)} calls, "
        f"{sum(r is not None for _, r in requests)} ranged"
    )

    check_release_requests(requests)
    asked = {url.removeprefix(f"{ORIGIN}{KEYWORD}/{RELEASE}/") for url, _ in requests}
    for subsection in ("decay", "reactions"):
        if not any(a.startswith(f"transmutation/{subsection}.arrow/") for a in asked):
            sys.exit(f"the {subsection} subsection was not fetched from the release")
    if max(batches) < 2:
        sys.exit("no call carried more than one request, so the batched prefetch did not run")
    # The material is at 294 K, so Fe56's reactions are read as byte ranges
    # of that temperature, verified against the hashes, and never whole.
    ranged = [r for u, r in requests if u.endswith("Fe56.arrow/reactions.arrow")]
    if not ranged or any(r is None for r in ranged):
        sys.exit(f"Fe56's reactions.arrow was not fetched as byte ranges: {ranged}")

    # One material per step, at the end of it: the pulse, then the two
    # cooldowns.
    shutdown = steps[0]
    activity = shutdown.activity(by_nuclide=True)
    if not shutdown.activity() > 0.0 or not shutdown.decay_heat() > 0.0:
        sys.exit("no activity or decay heat at shutdown")
    if max(activity, key=activity.get) != "Mn56":
        sys.exit(f"Mn56 does not lead the activity at shutdown: {activity}")
    if not steps[2].activity() < steps[1].activity() < shutdown.activity():
        sys.exit("the activity does not fall as the iron cools")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
