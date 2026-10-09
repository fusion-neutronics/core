"""Hydrogen and helium gas production in appm, at the Python boundary.

The arithmetic, the refusal of a chain without the gas nuclides and the
uncertainty fold are pinned in Rust (``crates/yani-transmute/src/gas.rs``), and
the real-data check against the hand calculation for natural iron in
``crates/yani-transmute/tests/gas_production.rs``. What is pinned here is the
shape a Python caller meets, and that it is the issue's few lines of
arithmetic on the inventory the same results hold.
"""

import pytest
import yamc

NUC_DATA = "tests"
DAY = 86400.0

ENERGY_GROUPS = [1e-5, 0.625, 1e5, 2e7]
MULTIGROUP_FLUX = [1e12, 5e12, 1e14]
RATE = sum(MULTIGROUP_FLUX)

KEYS = ["H", "H1", "H2", "H3", "He", "He3", "He4"]
GAS = ["H1", "H2", "H3", "He3", "He4"]


@pytest.fixture(autouse=True)
def _set_cross_sections():
    yamc.cross_section_data = NUC_DATA
    yield
    yamc.cross_section_data = None


def _schedule():
    spectrum = yamc.NeutronSource(
        energy=yamc.sources.Histogram(ENERGY_GROUPS, MULTIGROUP_FLUX)
    )
    return yamc.PulseSchedule([
        yamc.Pulse(rate=RATE, duration=30 * DAY, source=spectrum),
        yamc.Cooldown(duration=DAY),
        yamc.Pulse(rate=RATE, duration=30 * DAY, source=spectrum),
        yamc.Cooldown(duration=30 * DAY),
    ])


def _material(composition, density=0.08):
    """A material at a total atom density [atoms/barn-cm]."""
    return yamc.Material(
        composition=composition,
        density=density,
        units="atom/barn-cm",
        volume=1.0,
        temperature=294,
    )


def _by_hand(results, mid):
    """The issue's lines: change in each gas density over the initial atoms."""
    a0 = results.get_material(mid, 0).get_atoms_per_barn_cm()
    total = sum(a0.values())
    return [
        {
            n: (m.get_atoms_per_barn_cm().get(n, 0.0) - a0.get(n, 0.0)) / total * 1e6
            for n in GAS
        }
        for m in [results.get_material(mid, 0), *results.step_materials(mid)]
    ]


def test_every_time_point_matches_the_inventory():
    iron = _material({"Fe56": 1.0})
    results = iron.transmute(schedule=_schedule())
    mid = iron.id or 0
    gas = results.get_gas_production(mid)

    assert sorted(gas) == KEYS
    for key in KEYS:
        assert len(gas[key]) == len(results.times), key
        assert gas[key][0] == 0.0, "nothing is produced before the first step"

    for step, expected in enumerate(_by_hand(results, mid)):
        for n in GAS:
            assert gas[n][step] == pytest.approx(expected[n], rel=1e-12, abs=1e-12)
        assert gas["H"][step] == pytest.approx(
            expected["H1"] + expected["H2"] + expected["H3"], rel=1e-12, abs=1e-12
        )
        assert gas["He"][step] == pytest.approx(
            expected["He3"] + expected["He4"], rel=1e-12, abs=1e-12
        )

    # Fe56 (n,p) and (n,a) at 14 MeV: gas grows over each irradiation.
    assert gas["H1"][1] > 0.0
    assert gas["He4"][1] > 0.0
    assert gas["He4"][3] > gas["He4"][1]


def test_starting_gas_is_subtracted_unless_the_total_is_asked_for():
    # The same oxygen atom density in both.
    oxide = _material({"O16": 1.0}, density=0.03)
    water = _material({"O16": 1.0, "H1": 2.0}, density=0.09)
    oxide_results = oxide.transmute(schedule=_schedule())
    water_results = water.transmute(schedule=_schedule())
    oxide_gas = oxide_results.get_gas_production(oxide.id or 0)
    water_gas = water_results.get_gas_production(water.id or 0)

    # The same oxygen makes the same helium. appm is per initial atom, and the
    # water has three for every one the oxide has, so per oxygen atom they
    # agree.
    assert oxide_gas["He4"][-1] > 0.0
    assert water_gas["He4"][-1] * 3.0 == pytest.approx(oxide_gas["He4"][-1], rel=1e-9)

    # The water's own hydrogen is not production: the H1 left is the oxygen's
    # (n,p), less what capture on the water's H1 turned into H2, nowhere near
    # the 2/3 of a million appm the water started with.
    assert 0.0 < water_gas["H1"][-1] * 3.0 <= oxide_gas["H1"][-1]
    total = water_results.get_gas_production(water.id or 0, produced=False)
    starting = 2.0 / 3.0 * 1e6
    assert total["H1"][0] == pytest.approx(starting, rel=1e-12)
    assert total["H1"][-1] == pytest.approx(starting + water_gas["H1"][-1], rel=1e-12)
    assert total["He4"] == water_gas["He4"]


def test_an_unknown_material_is_none():
    iron = _material({"Fe56": 1.0})
    results = iron.transmute(schedule=_schedule())
    assert results.get_gas_production((iron.id or 0) + 1000) is None


def test_uncertainty_is_none_unless_asked_for():
    iron = _material({"Fe56": 1.0})
    results = iron.transmute(schedule=_schedule())
    assert results.get_gas_production_uncertainty(iron.id or 0, 1) is None


def test_uncertainty_comes_with_the_replicas():
    iron = _material({"Fe56": 1.0})
    results = iron.transmute(
        schedule=_schedule(),
        data_uncertainty=yamc.DataUncertainty(seed=1, samples=8),
    )
    mid = iron.id or 0
    gas = results.get_gas_production(mid)
    for step in range(len(results.times)):
        band = results.get_gas_production_uncertainty(mid, step)
        assert sorted(band) == KEYS
        for key in KEYS:
            assert isinstance(band[key], yamc.Estimate)
            assert band[key].nominal == gas[key][step]
            assert band[key].replicas == 8
            assert band[key].mean is not None
            assert band[key].std_dev is not None
            assert band[key].std_dev_standard_error is not None
    start = results.get_gas_production_uncertainty(mid, 0)
    assert start["He4"].std_dev == 0.0, "the starting inventory has no spread"
