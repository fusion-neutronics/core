"""Every calculation that needs neutron cross sections refuses to run without
a library, and says how to configure one.

There is no default neutron library (the photon data and the transmutation
chain have one; the neutron library is the calculation's central choice), so
with ``yamc.cross_section_data`` unset these used to either panic with a
message about file paths, which Python sees as a ``PanicException`` that
``except Exception`` does not catch, or, for ``Material.transmute``, run and
return zero activity.
"""

import pytest
import yamc

DAY = 86400.0


@pytest.fixture(autouse=True)
def _no_library():
    yamc.cross_section_data = None
    yield
    yamc.cross_section_data = None


def _iron():
    return yamc.Material(
        composition={"Fe56": 1.0},
        density=7.87,
        temperature=294,
        volume=1.0,
        id=1,
        transmutable=True,
    )


def _spectrum():
    return yamc.NeutronSource(
        energy=yamc.sources.Histogram([1e-5, 1e5, 2e7], [1.0, 1.0])
    )


def _model(material):
    sphere = yamc.Sphere(radius=10.0, boundary="vacuum")
    source = yamc.NeutronSource(
        position=(0.0, 0.0, 0.0), energy=yamc.sources.Discrete([14.06e6], [1.0])
    )
    return yamc.Model(
        geometry=yamc.Geometry([yamc.Cell(region=sphere.below, material=material)]),
        source=source,
        verbose=[],
    ), source


def _assert_names_the_setting(error):
    message = str(error.value)
    assert "no neutron cross-section library is configured" in message
    assert "cross_section_data" in message
    assert "Fe56" in message


def test_material_transmute_refuses_rather_than_returning_zero_activity():
    schedule = yamc.PulseSchedule(
        [yamc.Pulse(rate=1e14, duration=DAY, source=_spectrum())]
    )
    with pytest.raises(ValueError) as error:
        _iron().transmute(schedule)
    _assert_names_the_setting(error)


def test_transmute_of_several_materials_refuses_too():
    schedule = yamc.PulseSchedule(
        [yamc.Pulse(rate=1e14, duration=DAY, source=_spectrum())]
    )
    with pytest.raises(ValueError) as error:
        yamc.transmute([_iron()], schedule)
    _assert_names_the_setting(error)


def test_a_decay_only_schedule_needs_no_cross_sections():
    iron = yamc.Material(
        composition={"Fe55": 1.0}, density=7.87, temperature=294, volume=1.0
    )
    results = iron.transmute(yamc.PulseSchedule([yamc.Cooldown(duration=DAY)]))
    assert results.get_final_material(iron.id).activity() > 0.0


def test_simulate_transport_refuses():
    model, _ = _model(_iron())
    with pytest.raises(Exception) as error:
        model.simulate_transport(total_particles=10)
    _assert_names_the_setting(error)


def test_simulate_transmutation_refuses():
    model, source = _model(_iron())
    schedule = yamc.PulseSchedule([yamc.Pulse(rate=1e14, duration=DAY, source=source)])
    with pytest.raises(Exception) as error:
        model.simulate_transmutation(
            method="independent", schedule=schedule, total_particles=10
        )
    _assert_names_the_setting(error)


@pytest.mark.parametrize(
    "query",
    [
        lambda m: m.macroscopic_cross_section(1),
        lambda m: m.mean_free_path_neutron(1e6),
        lambda m: m.calculate_microscopic_xs_neutron(),
        lambda m: m.unified_energy_grid_neutron(),
        lambda m: m.sample_distance_to_collision(1e6, 1),
        lambda m: m.sample_interacting_nuclide(1e6),
        lambda m: m.reaction_mts,
    ],
    ids=[
        "macroscopic_cross_section",
        "mean_free_path_neutron",
        "calculate_microscopic_xs_neutron",
        "unified_energy_grid_neutron",
        "sample_distance_to_collision",
        "sample_interacting_nuclide",
        "reaction_mts",
    ],
)
def test_cross_section_queries_raise_a_value_error(query):
    with pytest.raises(ValueError) as error:
        query(_iron())
    _assert_names_the_setting(error)
