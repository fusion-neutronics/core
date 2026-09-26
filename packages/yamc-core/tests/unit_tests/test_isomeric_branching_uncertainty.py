"""The isomeric branching uncertainty source at the Python boundary.

The numbers are pinned in Rust
(``crates/yani-transmute/tests/isomeric_branching_uncertainty.rs``), where a
``branching_covariance.arrow`` is written with the real converter. What is
pinned here is what a Python caller meets: the source is offered, the report
carries its keys with the types a caller reads them as, it says what it held
at nominal, and asking for it moves no mean.
"""

import pytest
import yamc

NUC_DATA = "tests"
DAY = 86400.0

ENERGY_GROUPS = [1e-5, 0.625, 1e5, 2e7]
MULTIGROUP_FLUX = [1e12, 5e12, 1e14]
RATE = sum(MULTIGROUP_FLUX)


@pytest.fixture(autouse=True)
def _set_cross_sections():
    yamc.cross_section_data = NUC_DATA
    yield
    yamc.cross_section_data = None


def _iron():
    return yamc.Material(
        composition={"Fe56": 1.0},
        density=7.87,
        name="iron",
        volume=1.0,
        temperature=294,
    )


def _schedule():
    spectrum = yamc.NeutronSource(
        energy=yamc.sources.Histogram(ENERGY_GROUPS, MULTIGROUP_FLUX)
    )
    return yamc.PulseSchedule([
        yamc.Pulse(rate=RATE, duration=DAY, source=spectrum),
        yamc.Cooldown(duration=DAY),
    ])


def _info(sources):
    iron = _iron()
    results = iron.transmute(
        schedule=_schedule(),
        data_uncertainty=yamc.DataUncertainty(seed=1, samples=8, sources=sources),
    )
    return results.get_data_uncertainty_info(iron.id or 0)


def test_isomeric_branching_is_an_available_source():
    assert "isomeric_branching" in yamc.DataUncertainty.available_sources()


def test_leaving_it_out_is_reported():
    info = _info(["cross_sections"])
    assert "isomeric branching (MF=9/MF=10)" in info["not_perturbed"]
    assert "isomeric_branching" not in info["sources"]


def test_the_report_carries_its_keys():
    info = _info(["isomeric_branching"])
    assert info["sources"] == ["isomeric_branching"]
    for key in (
        "isomeric_channels_perturbed",
        "no_isomeric_branching_uncertainty",
        "isomeric_partials_without_covariance",
    ):
        assert isinstance(info[key], list), key
    for key in ("isomeric_rate_fraction_covered", "isomeric_blocks_skipped"):
        assert isinstance(info[key], dict), key
    for key in ("isomeric_matrices_clipped", "isomeric_partials_sampled"):
        assert isinstance(info[key], int), key
    # Nothing publishes an MF=40 x MF=33 correlation, and the report says the
    # two were sampled as independent rather than leaving it to be assumed.
    assert (
        "isomeric-branching x cross-section correlation (none published)"
        in info["not_perturbed"]
    )
    assert "isomeric branching (MF=9/MF=10)" not in info["not_perturbed"]


def test_the_means_are_unchanged_by_asking_for_it():
    iron = _iron()
    plain = iron.transmute(schedule=_schedule())
    with_it = _iron().transmute(
        schedule=_schedule(),
        data_uncertainty=yamc.DataUncertainty(
            seed=3, samples=8, sources=["isomeric_branching"]
        ),
    )
    mid = iron.id or 0
    a = plain.get_material_nuclides(mid, 1)
    b = with_it.get_material_nuclides(mid, 1)
    assert a.keys() == b.keys()
    for name in a:
        assert a[name] == b[name], f"{name} moved when the source was switched on"
