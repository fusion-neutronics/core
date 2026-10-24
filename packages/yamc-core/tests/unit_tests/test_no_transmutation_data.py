"""There is no default transmutation library, as there is no default neutron
cross-section library. A calculation that reads a transmutation source that is
unset refuses before any work, naming each such setting and how to set it.

These used to read endf-b8.1 for an unset decay, reactions or fission-yields
setting, so a run's chain came from a library nobody chose.
"""

from pathlib import Path

import pytest
import yamc

DAY = 86400.0
_CHAIN = str(
    Path(__file__).resolve().parents[4] / "tests" / "transmutation-endf-b8.1-sfr.arrow"
)
_SOURCES = (
    "transmutation_decay_data",
    "transmutation_reactions",
    "transmutation_fission_yields",
)


@pytest.fixture(autouse=True)
def _nothing_configured():
    """Unset every source, cross sections included, so a refusal that came
    after any data was read would name the cross sections instead."""
    yamc.cross_section_data = None
    for setting in _SOURCES:
        setattr(yamc, setting, None)
    yield
    yamc.cross_section_data = None


def _assert_names(error, *unset):
    """The refusal names exactly the `unset` settings, each with how to set it."""
    message = str(error.value)
    assert message.startswith("no transmutation data is configured"), message
    assert "There is no default library" in message, message
    assert "local folder" in message, message
    for setting in _SOURCES:
        line = f'yamc.{setting} = "endf-b8.1"'
        assert (line in message) == (setting in unset), f"{setting}: {message}"


def _iron():
    return yamc.Material(
        composition={"Fe56": 1.0},
        density=7.87,
        temperature=294,
        volume=1.0,
        transmutable=True,
    )


def _irradiation():
    spectrum = yamc.NeutronSource(
        energy=yamc.sources.Histogram([1e-5, 1e5, 2e7], [1.0, 1.0])
    )
    return yamc.PulseSchedule([yamc.Pulse(rate=1e14, duration=DAY, source=spectrum)])


def _model(**options):
    material = _iron()
    sphere = yamc.Sphere(radius=10.0, boundary="vacuum")
    source = yamc.NeutronSource(
        position=(0.0, 0.0, 0.0), energy=yamc.sources.Discrete([14.06e6], [1.0])
    )
    cell = yamc.Cell(region=sphere.below, material=material)
    model = yamc.Model(
        geometry=yamc.Geometry([cell]), source=source, verbose=[], **options
    )
    return model, source


@pytest.mark.parametrize("setting", _SOURCES)
def test_an_unset_source_reads_back_unset(setting):
    assert getattr(yamc, setting) is None


def test_an_irradiation_names_every_unset_source():
    with pytest.raises(ValueError) as error:
        _iron().transmute(_irradiation())
    _assert_names(error, *_SOURCES)
    assert "also take False" in str(error.value)


def test_only_the_unset_source_is_named():
    yamc.cross_section_data = "tests"
    yamc.transmutation_decay_data = _CHAIN
    yamc.transmutation_reactions = _CHAIN
    with pytest.raises(ValueError) as error:
        _iron().transmute(_irradiation())
    _assert_names(error, "transmutation_fission_yields")
    assert "transmutation_fission_yields is not set" in str(error.value)


def test_false_is_a_choice():
    """Fission yields turned off are not loaded, and iron needs none."""
    yamc.cross_section_data = "tests"
    yamc.transmutation_decay_data = _CHAIN
    yamc.transmutation_reactions = _CHAIN
    yamc.transmutation_fission_yields = False
    iron = _iron()
    results = iron.transmute(_irradiation())
    assert results.get_final_material(iron.id).activity() > 0.0


def test_a_cooldown_reads_only_the_decay_data():
    yamc.transmutation_decay_data = _CHAIN
    iron = yamc.Material(
        composition={"Fe55": 1.0}, density=7.87, temperature=294, volume=1.0
    )
    results = iron.transmute(yamc.PulseSchedule([yamc.Cooldown(duration=DAY)]))
    assert results.get_final_material(iron.id).activity() > 0.0


def test_a_decay_quantity_names_the_decay_data_alone():
    cobalt = yamc.Material(composition={"Co60": 1.0}, density=8.9, volume=1.0)
    for quantity in (cobalt.activity, cobalt.decay_heat, cobalt.contact_dose):
        with pytest.raises(ValueError) as error:
            quantity()
        _assert_names(error, "transmutation_decay_data")
    yamc.transmutation_decay_data = _CHAIN
    assert cobalt.activity() > 0.0


def test_transmute_of_several_materials_refuses_too():
    with pytest.raises(ValueError) as error:
        yamc.transmute([_iron()], _irradiation())
    _assert_names(error, *_SOURCES)


def test_simulate_transmutation_refuses_before_transport():
    model, source = _model()
    schedule = yamc.PulseSchedule([yamc.Pulse(rate=1e14, duration=DAY, source=source)])
    with pytest.raises(ValueError) as error:
        model.simulate_transmutation(
            method="independent", schedule=schedule, total_particles=10
        )
    _assert_names(error, *_SOURCES)


def test_radionuclides_read_the_decay_data_and_the_reactions():
    model, _ = _model()
    with pytest.raises(ValueError) as error:
        model.radionuclides()
    _assert_names(error, "transmutation_decay_data", "transmutation_reactions")


def test_decay_photon_transport_refuses_before_any_data_loads():
    model, _ = _model(transport_secondary_photons=True, use_decay_photons=True)
    with pytest.raises(ValueError) as error:
        model.simulate_transport(total_particles=10)
    message = str(error.value)
    assert message.startswith("use_decay_photons: no transmutation data"), message
    assert "transmutation_decay_data and transmutation_reactions are not set" in message
