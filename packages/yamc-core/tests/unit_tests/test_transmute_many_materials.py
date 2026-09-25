"""``transmute(materials, schedules)``: several materials in one call (issue #146).

The plural call must be a faster route to the answers ``Material.transmute``
gives, never different answers. So each test compares a material solved in a
group against the same material solved alone, and checks the rules the result
type forces: one timeline, one id per material.
"""

import pytest
import yamc

NUC_DATA = "tests"
DAY = 86400.0

ENERGY_GROUPS = [1e-5, 0.625, 1e5, 2e7]


@pytest.fixture(autouse=True)
def _set_cross_sections():
    yamc.cross_section_data = NUC_DATA
    yield
    yamc.cross_section_data = None


def _iron(material_id, density=7.87):
    return yamc.Material(
        composition={"Fe56": 1.0},
        density=density,
        name=f"iron {material_id}",
        volume=1.0,
        temperature=294,
        id=material_id,
    )


def _schedule(rate, flux):
    spectrum = yamc.NeutronSource(energy=yamc.sources.Histogram(ENERGY_GROUPS, flux))
    return yamc.PulseSchedule([
        yamc.Pulse(rate=rate, duration=DAY, source=spectrum),
        yamc.Cooldown(duration=DAY),
    ])


HARD = [1e12, 5e12, 1e14]
SOFT = [1e14, 5e12, 1e12]


def _same(a, b, material_id):
    for step in range(3):
        assert a.get_material_nuclides(material_id, step) == b.get_material_nuclides(
            material_id, step
        ), f"material {material_id} differs at step {step}"


def test_the_package_exports_it():
    # Only this wheel: yani is not installed alongside yamc, and it carries
    # its own copy of the bindings, asserted by the yani surface test.
    assert callable(yamc.transmute)
    assert "transmute" in yamc.__all__


def test_each_material_matches_its_own_transmute():
    alone = [
        _iron(1).transmute(_schedule(1e14, HARD)),
        _iron(2).transmute(_schedule(3e13, HARD)),
        _iron(3).transmute(_schedule(2e14, SOFT)),
    ]
    together = yamc.transmute(
        [_iron(1), _iron(2), _iron(3)],
        [_schedule(1e14, HARD), _schedule(3e13, HARD), _schedule(2e14, SOFT)],
    )
    assert sorted(together.material_ids) == [1, 2, 3]
    for material_id, single in zip((1, 2, 3), alone):
        _same(together, single, material_id)
        assert together.get_source_rates(material_id) == single.get_source_rates(
            material_id
        )
    assert together.get_source_rates(2) == [3e13, 0.0]


def test_a_shared_spectrum_on_one_steel_is_collapsed_once():
    # One NeutronSource object shared by two schedules is one spectrum, and
    # the two materials are the same steel, so their collapse is shared even
    # though their flux magnitudes differ.
    spectrum = yamc.NeutronSource(energy=yamc.sources.Histogram(ENERGY_GROUPS, HARD))

    def sched(rate):
        return yamc.PulseSchedule([
            yamc.Pulse(rate=rate, duration=DAY, source=spectrum),
            yamc.Cooldown(duration=DAY),
        ])

    results = yamc.transmute([_iron(1), _iron(2)], [sched(1e14), sched(1e13)])
    assert results.collapse_reuse == {"performed": 1, "requested": 2}

    different = yamc.transmute([_iron(1), _iron(2, density=1.0)], [sched(1e14), sched(1e13)])
    assert different.collapse_reuse == {"performed": 2, "requested": 2}


def test_one_schedule_serves_every_material():
    results = yamc.transmute([_iron(1), _iron(2)], _schedule(1e14, HARD))
    _same(results, _iron(2).transmute(_schedule(1e14, HARD)), 2)
    assert results.get_self_shielding_info(1) is not None
    assert results.get_data_uncertainty_info(1) is None


def test_a_repeated_id_is_refused():
    with pytest.raises(ValueError, match="both have id 4"):
        yamc.transmute([_iron(4), _iron(4)], _schedule(1e14, HARD))


def test_the_same_material_twice_is_refused():
    iron = _iron(1)
    with pytest.raises(TypeError, match="more than once"):
        yamc.transmute([iron, iron], _schedule(1e14, HARD))


def test_the_schedule_count_must_match():
    with pytest.raises(ValueError, match="2 schedules for 3 materials"):
        yamc.transmute(
            [_iron(1), _iron(2), _iron(3)],
            [_schedule(1e14, HARD), _schedule(1e14, HARD)],
        )


def test_timelines_must_agree():
    spectrum = yamc.NeutronSource(energy=yamc.sources.Histogram(ENERGY_GROUPS, HARD))
    longer = yamc.PulseSchedule([
        yamc.Pulse(rate=1e14, duration=DAY, source=spectrum),
        yamc.Cooldown(duration=2 * DAY),
    ])
    with pytest.raises(ValueError, match="step 1 lasts"):
        yamc.transmute([_iron(1), _iron(2)], [_schedule(1e14, HARD), longer])


def test_an_error_names_the_material():
    # A composition the chain cannot drive fails as Material.transmute would,
    # and the message says which material it was.
    inert = yamc.Material(
        composition={"He4": 1.0}, density=1e-3, volume=1.0, temperature=294, id=9
    )
    with pytest.raises(ValueError, match="material 9"):
        yamc.transmute([_iron(1), inert], _schedule(1e14, HARD))
