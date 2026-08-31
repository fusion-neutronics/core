"""The decay photon line spectrum of an activated inventory (issue #381 C3).

Irradiated natural iron is the clean check: its short-lived activity is
dominated by Mn56, whose 846.8 keV gamma is emitted on essentially every decay,
so both the line energy and its rate are predictable from the activity alone.
"""

from pathlib import Path

import pytest

import yamc

_CHAIN = str(
    Path(__file__).resolve().parents[4] / "tests" / "transmutation-endf-b8.1-sfr.arrow"
)
MN56_GAMMA_EV = 846754.0


@pytest.fixture
def activated_iron():
    yamc.cross_section_data = "endf-b8.1"
    material = yamc.Material(
        composition={"Fe56": 1.0},
        density=7.874,
        temperature=294,
        volume=1000.0,
    )
    spectrum = yamc.NeutronSource(
        energy=yamc.sources.Histogram([1e-5, 1e5, 1e6, 1.5e7], [1e12, 1e13, 1e14])
    )
    schedule = yamc.PulseSchedule(
        [
            yamc.Pulse(rate=1.11e14, duration=(30, "d"), source=spectrum),
            yamc.Cooldown(duration=(1, "d")),
        ]
    )
    results = material.transmute(schedule=schedule)
    return results.get_final_material(material.id or 0)


def test_spectrum_is_ascending_positive_lines(activated_iron):
    energies, rates = activated_iron.decay_photon_spectrum()
    assert len(energies) == len(rates)
    assert energies, "an activated material must emit something"
    assert energies == sorted(energies), "lines must come out ascending in energy"
    assert len(set(energies)) == len(energies), "coincident lines must be summed"
    assert all(rate > 0.0 for rate in rates)


def test_mn56_dominates_irradiated_iron(activated_iron):
    """Mn56 (Fe56 (n,p), 2.58 h) emits its 846.8 keV gamma on 98.85% of decays,
    so that line's rate must be that share of Mn56's own activity.

    This is the units check. The chain stores intensities per atom per second,
    not per decay, so a fold that reached for `activity()` instead of the atom
    count would land a factor of the decay constant out and this would catch
    it: Mn56's is 7.5e-5.
    """
    energies, rates = activated_iron.decay_photon_spectrum()
    lines = dict(zip(energies, rates))
    del rates
    assert MN56_GAMMA_EV in lines, f"no 846.8 keV line in {sorted(lines)[:5]}..."
    mn56_bq = activated_iron.activity(by_nuclide=True)["Mn56"]
    assert lines[MN56_GAMMA_EV] == pytest.approx(0.9885 * mn56_bq, rel=0.02)


def test_spectrum_scales_with_the_inventory(activated_iron):
    """Rates are photons per second, not probabilities, so ten times the volume
    is ten times the emission."""
    energies, rates = activated_iron.decay_photon_spectrum()
    bigger = yamc.Material.from_atom_densities(dict(activated_iron.nuclides))
    bigger.volume = activated_iron.volume * 10.0
    bigger.temperature = 294
    big_energies, big_rates = bigger.decay_photon_spectrum()
    assert big_energies == energies
    for one, ten in zip(rates, big_rates):
        assert ten == pytest.approx(10.0 * one, rel=1e-12)


def test_volume_is_required():
    material = yamc.Material(composition={"Co60": 1.0}, density=8.9, temperature=294)
    with pytest.raises(ValueError, match="volume"):
        material.decay_photon_spectrum()
