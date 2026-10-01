"""Decay photon continua from Python, on a real JEFF-4.0 actinide.

JEFF-4.0 Cf252 gives its gamma spectrum as three weak lines beside a
linear-linear continuum, the spontaneous-fission photons, at FC = 0.249957
photons per decay: nearly all of the nuclide's photon emission. The evaluation
is converted here with the wheel's own converter, so the test also covers the
interpolation column from ENDF to the reader.
"""

import lzma
import math
from pathlib import Path

import pytest

import yamc

FIXTURE = (
    Path(__file__).resolve().parents[4]
    / "crates"
    / "endf"
    / "fixtures"
    / "dec-098_Cf_252.jeff40.endf.xz"
)
CF252_HALF_LIFE = 83469800.0
# FC times the integral of the tape's RP, which is normalised to 1.0000099.
CONTINUUM_PER_DECAY = 0.249957 * 1.0000099


@pytest.fixture
def cf252(tmp_path):
    """A one-nuclide chain converted from the fixture, and a material of it."""
    decay = tmp_path / "dec-098_Cf_252.endf"
    decay.write_bytes(lzma.decompress(FIXTURE.read_bytes()))
    out = tmp_path / "transmutation-jeff-4.0.arrow"
    yamc.convert_transmutation(
        decay_files=[str(decay)],
        fpy_files=[],
        neutron_files=[],
        output_path=str(out),
        library="jeff-4.0",
        data_version="test",
        subsections=["decay"],
    )
    yamc.transmutation_decay_data = str(out)
    yamc.transmutation_reactions = str(out)
    yamc.transmutation_fission_yields = str(out)
    # Cf252 in iron, so the contact dose has a host whose attenuation is known.
    material = yamc.Material.from_atom_densities({"Fe56": 0.084912, "Cf252": 1.0e-9})
    material.volume = 2.0
    return material


def test_the_continuum_comes_back_apart_from_the_lines(cf252):
    continua = cf252.decay_photon_continua()
    assert [c.nuclide for c in continua] == ["Cf252"]
    continuum = continua[0]
    assert continuum.interpolation == "linear-linear"
    assert len(continuum.energies) == len(continuum.rates) == 15

    atoms = 1.0e-9 * 1.0e24 * 2.0
    decays = atoms * math.log(2) / CF252_HALF_LIFE
    assert continuum.emission_rate == pytest.approx(decays * CONTINUUM_PER_DECAY, rel=1e-6)

    # The lines are the weak gammas and the x-rays, far below the continuum.
    energies, rates = cf252.decay_photon_spectrum()
    assert energies, "the lines are still there"
    assert sum(rates) < 0.5 * continuum.emission_rate


def test_the_continuum_scales_per_unit(cf252):
    """``per`` scales the densities exactly as it scales the line rates."""
    whole = cf252.decay_photon_continua()[0]
    per_cm3 = cf252.decay_photon_continua(per="cm3")[0]
    assert per_cm3.emission_rate == pytest.approx(whole.emission_rate / 2.0, rel=1e-12)


def test_the_contact_dose_integrates_the_continuum(cf252):
    """Cf252's dose is almost all its continuum.

    Measured when the continuum was first integrated: 0.0614 Gy/h, where
    reading its per-eV values as lines gave 2.46e-5 Gy/h, the x-ray and gamma
    lines' share. The Rust tests pin the fold itself against a closed form.
    """
    dose = cf252.contact_dose(by_nuclide=True)
    assert dose["Cf252"] == pytest.approx(0.0614226, rel=1e-5)


def test_photon_sources_tag_the_continuum_with_its_law(cf252):
    chain = yamc.TransmutationChain(yamc.transmutation_decay_data)
    rows = chain.photon_sources["Cf252"]
    # The gamma lines, the gamma continuum and the x-ray lines: each spectrum
    # keeps its own rows, since each has its own normalisation.
    assert sorted(row[0] for row in rows) == ["discrete", "discrete", "tabular"]
    tabular = next(row for row in rows if row[0] == "tabular")
    assert tabular[3] == "linear-linear"
