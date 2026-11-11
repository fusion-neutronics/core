"""``Processing``: continuous removal, transfer between materials and feed.

The solve itself is checked against closed forms in Rust
(``yani_transmute::processing``). These pin the Python surface on both
entry points: a destination outside the run is carried and reported, a rate of
zero changes nothing, and a circulating liquid fills a pool whose contact dose
reads like any other material's.
"""

import os

import pytest

import yamc

TESTS_DIR = os.path.join("crates", "yamc", "tests")
CHAIN = os.path.join(TESTS_DIR, "transmutation-endf-b8.1-sfr.arrow")
DAY = 86400.0
FAST = [1e-5, 0.625, 1e5, 2e7]


@pytest.fixture(autouse=True)
def _data():
    yamc.neutron_cross_section_data = "tests"
    yamc.transmutation_decay_data = CHAIN
    yamc.transmutation_reactions = CHAIN
    yamc.transmutation_fission_yields = CHAIN


def _iron():
    return yamc.Material(
        composition={"Fe56": 1.0}, density=7.87, volume=1.0, temperature=294, id=1
    )


def _water(material_id, volume, transmutable=False):
    return yamc.Material(
        composition={"H1": 2.0, "O16": 1.0},
        density=1.0,
        volume=volume,
        temperature=294,
        transmutable=transmutable,
        id=material_id,
    )


def _schedule():
    spectrum = yamc.NeutronSource(energy=yamc.sources.Histogram(FAST, [1e12, 5e12, 1e14]))
    return yamc.PulseSchedule([
        yamc.Pulse(rate=1e14, duration=DAY, source=spectrum),
        yamc.Cooldown(duration=DAY),
    ])


def _density(results, material_id, nuclide, step):
    return results.get_material_nuclides(material_id, step).get(nuclide, 0.0)


def test_extracted_manganese_lands_in_a_tank_outside_the_run():
    tank = _water(9, 4.0)
    processing = yamc.Processing()
    processing.add_transfer(
        _iron(), "Mn", rate=(1, "h"), rate_units="half-time", destination=tank, steps=[0]
    )
    assert processing.material_ids == [1, 9]
    plain = yamc.transmute([_iron()], _schedule())
    processed = yamc.transmute([_iron()], _schedule(), processing=processing)

    assert sorted(processed.material_ids) == [1, 9]
    assert _density(processed, 1, "Mn56", 1) < 0.5 * _density(plain, 1, "Mn56", 1)
    assert _density(processed, 9, "Mn56", 1) > 0.0
    assert processed.get_source_rates(9) == [0.0, 0.0]
    assert processed.get_material(9, 1).contact_dose() > 0.0


def test_a_rate_of_zero_changes_nothing():
    processing = yamc.Processing()
    processing.add_transfer(_iron(), "Mn", rate=0.0, destination=_water(9, 4.0))
    plain = yamc.transmute([_iron()], _schedule())
    processed = yamc.transmute([_iron()], _schedule(), processing=processing)
    for nuclide, density in plain.get_material_nuclides(1, 2).items():
        assert _density(processed, 1, nuclide, 2) == pytest.approx(density, rel=1e-9)


def test_a_circulating_liquid_fills_a_pool_with_n16():
    """Water through a sphere of 14 MeV neutrons and back from a pool."""
    core = _water(1, 4188.79, transmutable=True)
    core.read_nuclear_data({
        "H1": os.path.join(TESTS_DIR, "H1.arrow"),
        "O16": os.path.join(TESTS_DIR, "O16.arrow"),
    })
    pool = _water(2, 1.0e5)
    flow = 500.0  # cm3/s
    processing = yamc.Processing()
    processing.add_transfer(core, rate=flow / core.volume, destination=pool)
    processing.add_transfer(pool, rate=flow / pool.volume, destination=core)

    sphere = yamc.Sphere(radius=10.0, boundary="vacuum")
    cell = yamc.Cell(region=sphere.below, material=core)
    source = yamc.NeutronSource(
        position=(0, 0, 0), energy=yamc.sources.Discrete([14.1e6], [1.0])
    )
    model = yamc.Model(geometry=yamc.Geometry([cell]), source=source, verbose=[])
    schedule = yamc.PulseSchedule([
        yamc.Pulse(rate=1e18, duration=60.0, source=source),
        yamc.Cooldown(duration=60.0),
    ])
    results = model.simulate_transmutation(
        method="independent",
        schedule=schedule,
        total_particles=2000,
        processing=processing,
    )

    assert sorted(results.material_ids) == [1, 2]
    # The volumes stay full: what leaves each comes back from the other.
    h1 = _density(results, 2, "H1", 0)
    assert _density(results, 2, "H1", 2) == pytest.approx(h1, rel=1e-6)
    assert _density(results, 1, "H1", 2) == pytest.approx(h1, rel=1e-6)
    assert _density(results, 2, "N16", 1) > 0.0
    assert results.get_material(2, 1).contact_dose() > 0.0


def test_what_cannot_be_solved_is_refused():
    processing = yamc.Processing()
    with pytest.raises(ValueError, match="not a removal rate"):
        processing.add_transfer(_iron(), "Mn", rate=1.0, rate_units="g/s")
    with pytest.raises(ValueError, match="needs proportional_to"):
        processing.add_feed(_iron(), {"Fe56": 1.0}, rate=1.0, rate_units="1/s")
    no_volume = yamc.Material(composition={"Fe56": 1.0}, density=7.87, id=1)
    processing.add_feed(no_volume, {"Fe": 1.0}, rate=1.0, rate_units="g/d")
    with pytest.raises(ValueError, match="needs a volume"):
        yamc.transmute([no_volume], _schedule(), processing=processing)
