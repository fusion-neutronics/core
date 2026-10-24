"""A run on a library keyword records, for each library it read, the
published release and the converters that wrote it: ``converter_versions``,
as the release's manifest lists them.

End-to-end network tests against the published data. They skip when it is
unavailable (offline with an empty cache), and compare each record with the
manifest the run cached, so they hold for whichever release is current; the
2026-10-12.1 values are pinned as well.
"""

import json
from pathlib import Path

import pytest

import yamc

# What the 2026-10-12.1 manifest of every keyword yamc resolves lists.
CONVERTERS_2026_10_12_1 = ["yamc-convert 0.1.0", "yani-convert 0.1.0"]
KEYS = {"release", "manifest_sha256", "format_version", "offline", "converter_versions"}


def _check(keyword, record):
    """`record` names the release the run read and the converters its manifest lists."""
    assert set(record) == KEYS
    manifest = Path(yamc.cache_dir()) / keyword / record["release"] / "manifest.json"
    listed = json.loads(manifest.read_text()).get("converter_versions", [])
    assert record["converter_versions"] == listed
    if record["release"] == "2026-10-12.1":
        assert record["converter_versions"] == CONVERTERS_2026_10_12_1


def _run(run):
    """`run()`, skipping when the data cannot be had."""
    try:
        return run()
    except Exception as e:  # pragma: no cover - offline with an empty cache
        # A verification or format error means the origin answered with data
        # this build refuses: a publishing mistake to fail on, not the missing
        # network the skip is for.
        if "mismatch" in str(e) or "format_version" in str(e):
            raise
        pytest.skip(f"endf-b8.1 data unavailable (offline?): {e}")


def test_a_transport_run_records_the_converters_of_its_release():
    yamc.cross_section_data = "endf-b8.1"
    lithium = yamc.Material(composition={"Li6": 1.0}, density=0.46, temperature=294)
    sphere = yamc.Sphere(radius=10.0, boundary="vacuum")
    cell = yamc.Cell(region=sphere.below, material=lithium)
    model = yamc.Model(
        geometry=yamc.Geometry([cell]),
        source=yamc.NeutronSource(energy=14.06e6),
    )
    results = _run(lambda: model.simulate_transport(total_particles=100, seed=1))
    _check("endf-b8.1", results.runs[0]["data_releases"]["endf-b8.1"])


def test_a_transmutation_records_the_converters_of_its_releases():
    yamc.cross_section_data = "endf-b8.1"
    yamc.transmutation_decay_data = "endf-b8.1"
    yamc.transmutation_reactions = "endf-b8.1"
    yamc.transmutation_fission_yields = "endf-b8.1"
    hydrogen = yamc.Material(
        composition={"H1": 1.0}, density=0.07, volume=1.0, temperature=294
    )
    spectrum = yamc.NeutronSource(
        energy=yamc.sources.Histogram([1e-5, 0.625, 1e5, 2e7], [1.0, 1.0, 1.0])
    )
    schedule = yamc.PulseSchedule(
        [yamc.Pulse(rate=1e14, duration=86400.0, source=spectrum)]
    )
    results = _run(lambda: hydrogen.transmute(schedule=schedule))
    assert "endf-b8.1" in results.data_releases
    for keyword, record in results.data_releases.items():
        _check(keyword, record)
