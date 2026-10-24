"""yani has no default transmutation library, as it has no default
cross-section library: a calculation that reads an unset transmutation source
refuses before any work, naming each one. Needs no nuclear data.
"""

import pytest

yani = pytest.importorskip("yani", reason="standalone yani wheel not installed")

_SOURCES = (
    "transmutation_decay_data",
    "transmutation_reactions",
    "transmutation_fission_yields",
)


@pytest.fixture(autouse=True)
def _nothing_configured():
    yani.cross_section_data = None
    for setting in _SOURCES:
        setattr(yani, setting, None)


def _steel():
    return yani.Material({"Fe56": 0.7, "Cr52": 0.2, "Ni58": 0.1}, density=7.9, volume=1.0)


def test_an_irradiation_names_every_unset_source():
    spectrum = yani.NeutronSource(energy=yani.sources.Histogram([1e-5, 1e5, 2e7], [1.0, 1.0]))
    schedule = yani.PulseSchedule([yani.Pulse(rate=1e14, duration=(1, "d"), source=spectrum)])
    with pytest.raises(ValueError) as error:
        _steel().transmute(schedule)
    message = str(error.value)
    assert message.startswith("no transmutation data is configured"), message
    for setting in _SOURCES:
        assert f'{setting} = "endf-b8.1"' in message, message


def test_a_decay_quantity_names_the_decay_data_alone():
    with pytest.raises(ValueError) as error:
        _steel().activity()
    message = str(error.value)
    assert "transmutation_decay_data is not set" in message, message
    assert "transmutation_reactions" not in message, message


@pytest.mark.parametrize("setting", ("transmutation_reactions", "transmutation_fission_yields"))
def test_off_and_unset_read_back_apart(setting):
    assert getattr(yani, setting) is None
    setattr(yani, setting, False)
    assert getattr(yani, setting) is False
