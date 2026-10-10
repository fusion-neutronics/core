"""Displacement damage from ``transmute(..., displacement_damage=True)``.

Damage energy per atom and NRT dpa, per element and for the material, with the
displacement threshold energies reported beside them. The arithmetic and the
fold are checked in Rust (``crates/yani-transmute/tests/displacement_damage.rs``);
these check the Python surface: what is returned, when, and what is refused.
"""

import pytest
import yamc

NUC_DATA = "tests"
DAY = 86400.0

ENERGY_GROUPS = [1e-5, 0.625, 1e5, 2e7]
FLUX = [1e12, 5e12, 1e14]


@pytest.fixture(autouse=True)
def _set_cross_sections():
    yamc.cross_section_data = NUC_DATA
    yield
    yamc.cross_section_data = None


def _iron(material_id=1, composition=None):
    return yamc.Material(
        composition=composition or {"Fe56": 1.0},
        density=7.87,
        volume=1.0,
        temperature=294,
        id=material_id,
    )


def _schedule():
    spectrum = yamc.NeutronSource(energy=yamc.sources.Histogram(ENERGY_GROUPS, FLUX))
    return yamc.PulseSchedule([
        yamc.Pulse(rate=sum(FLUX), duration=DAY, source=spectrum),
        yamc.Cooldown(duration=DAY),
    ])


def test_nothing_is_reported_unless_asked_for():
    results = _iron().transmute(_schedule())
    assert results.get_dpa(1) is None
    assert results.get_damage_energy(1) is None
    assert results.get_displacement_damage_info(1) is None


def test_dpa_and_damage_energy_per_state():
    plain = _iron().transmute(_schedule())
    results = _iron().transmute(_schedule(), displacement_damage=True)

    dpa = results.get_dpa(1)
    energy = results.get_damage_energy(1)
    assert len(dpa) == len(results.times) == 3
    assert dpa[0] == 0.0 and energy[0] == 0.0
    assert dpa[1] > 0.0 and energy[1] > 0.0
    # The cooldown adds nothing.
    assert dpa[2] == dpa[1] and energy[2] == energy[1]
    # NRT at the ASTM E521 value for iron.
    assert dpa[1] == pytest.approx(0.8 * results.get_damage_energy(1, element="Fe")[1] / 80.0, rel=1e-6)
    assert results.get_dpa(1, element="Fe")[1] == pytest.approx(dpa[1], rel=1e-6)

    info = results.get_displacement_damage_info(1)
    assert info["model"] == "NRT"
    assert info["efficiency"] == 0.8
    assert info["displacement_energies"]["Fe"] == {"energy": 40.0, "source": "ASTM E521"}

    # Asking for damage leaves the inventories as they were.
    for step in range(3):
        assert results.get_material_nuclides(1, step) == plain.get_material_nuclides(1, step)


def test_an_unknown_element_in_the_breakdown_is_refused():
    results = _iron().transmute(_schedule(), displacement_damage=True)
    with pytest.raises(ValueError, match="no dpa for element"):
        results.get_dpa(1, element="W")


def test_an_override_is_used_and_reported_as_the_users():
    default = _iron().transmute(_schedule(), displacement_damage=True)
    user = _iron().transmute(
        _schedule(), displacement_damage=True, displacement_energies={"Fe": 50.0}
    )
    assert user.get_displacement_damage_info(1)["displacement_energies"]["Fe"] == {
        "energy": 50.0,
        "source": "user",
    }
    assert user.get_dpa(1)[1] == pytest.approx(default.get_dpa(1)[1] * 40.0 / 50.0, rel=1e-12)


def test_an_element_without_a_displacement_energy_is_refused():
    lithium = _iron(composition={"Li7": 1.0})
    with pytest.raises(ValueError, match="displacement_energies"):
        lithium.transmute(_schedule(), displacement_damage=True)


@pytest.mark.parametrize(
    "energies, message",
    [
        ({"Iron": 40.0}, "not an element symbol"),
        ({"Fe": 0.0}, "positive"),
        ({"Fe": float("nan")}, "positive"),
    ],
)
def test_bad_overrides_are_refused(energies, message):
    with pytest.raises(ValueError, match=message):
        _iron().transmute(_schedule(), displacement_damage=True, displacement_energies=energies)


def test_overrides_without_the_switch_are_refused():
    with pytest.raises(ValueError, match="displacement_damage=True"):
        _iron().transmute(_schedule(), displacement_energies={"Fe": 40.0})


def test_several_materials_in_one_call():
    alone = _iron(1).transmute(_schedule(), displacement_damage=True)
    together = yamc.transmute(
        [_iron(1), _iron(2)], _schedule(), displacement_damage=True
    )
    assert together.get_dpa(1) == alone.get_dpa(1)
    assert together.get_dpa(2) == alone.get_dpa(1)
