"""`per=` on the three quantities that count atoms (issue #567).

The solve holds atom DENSITIES, so the intensive answer is the one it natively
has and `volume` is only the multiplier that makes it extensive. These check
that the intensive forms are reachable without inventing a volume, and -- the
part that matters more -- that the default is untouched, so the unit is a
function of what the caller wrote and never of what happens to be set on the
material.
"""

from pathlib import Path

import pytest
import yamc

_TESTS = Path(__file__).resolve().parents[4] / "tests"
DENSITY_G_CM3 = 7.874
VOLUME_CM3 = 1000.0


def _irradiated(volume):
    yamc.cross_section_data = "endf-b8.1"
    material = yamc.Material(
        composition={"Fe56": 1.0},
        density=DENSITY_G_CM3,
        temperature=294,
        volume=volume,
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


@pytest.fixture
def with_volume():
    return _irradiated(VOLUME_CM3)


@pytest.fixture
def without_volume():
    return _irradiated(None)


@pytest.mark.parametrize("quantity", ["activity", "decay_heat"])
def test_the_default_still_needs_a_volume(without_volume, quantity):
    """The unchanged path: no `per`, no volume, same refusal as before.

    This is the guarantee that makes `per=` safe to add. A caller who never
    passes it cannot tell it exists.
    """
    with pytest.raises(ValueError, match="requires material.volume"):
        getattr(without_volume, quantity)()


def test_the_spectrum_default_still_needs_a_volume(without_volume):
    with pytest.raises(ValueError, match="requires material.volume"):
        without_volume.decay_photon_spectrum()


@pytest.mark.parametrize("quantity", ["activity", "decay_heat"])
def test_per_cm3_needs_no_volume(without_volume, quantity):
    """The point of the issue: the intensive form is reachable with no volume."""
    value = getattr(without_volume, quantity)(per="cm3")
    assert value > 0.0


def test_spectrum_per_cm3_needs_no_volume(without_volume):
    energies, rates = without_volume.decay_photon_spectrum(per="cm3")
    assert energies and all(rate > 0.0 for rate in rates)


@pytest.mark.parametrize("quantity", ["activity", "decay_heat"])
def test_per_cm3_is_the_total_divided_by_the_volume(with_volume, quantity):
    """`volume` is a pure multiplier, so the two forms must agree exactly."""
    total = getattr(with_volume, quantity)()
    per_cm3 = getattr(with_volume, quantity)(per="cm3")
    assert per_cm3 == pytest.approx(total / VOLUME_CM3, rel=1e-12)


@pytest.mark.parametrize("quantity", ["activity", "decay_heat"])
def test_per_g_divides_by_the_mass_density(with_volume, quantity):
    """Bq/g is Bq/cm3 over g/cm3, and needs only `density`, already mandatory.

    This is the call the FNS decay-heat benchmark wants: it currently invents a
    volume from mass and density, takes a total, then divides the mass back out.

    Divided by the material's OWN mass density, not by the 7.874 it was built
    with. Activation moves mass between nuclides of different atomic mass, so
    the irradiated inventory here comes out at 7.873999643 g/cm3, 4.5e-8 below
    the input. Small, but far above the 1e-9 this asserts to, and using the
    current inventory is the behaviour that is correct rather than merely
    convenient.
    """
    density = with_volume.get_mass_density()
    assert density != pytest.approx(DENSITY_G_CM3, rel=1e-12), (
        "the point of this test is that the two differ; if activation stopped "
        "moving the density, assert against the input instead"
    )
    per_cm3 = getattr(with_volume, quantity)(per="cm3")
    per_g = getattr(with_volume, quantity)(per="g")
    assert per_g == pytest.approx(per_cm3 / density, rel=1e-9)


@pytest.mark.parametrize("quantity", ["activity", "decay_heat"])
def test_by_nuclide_carries_the_same_unit(with_volume, quantity):
    """`per` and `by_nuclide` compose: the dict's values scale the same way."""
    totals = getattr(with_volume, quantity)(by_nuclide=True)
    per_g = getattr(with_volume, quantity)(by_nuclide=True, per="g")
    assert set(totals) == set(per_g)
    scale = VOLUME_CM3 * with_volume.get_mass_density()
    for nuclide, value in totals.items():
        assert per_g[nuclide] == pytest.approx(value / scale, rel=1e-9)


@pytest.mark.parametrize(
    "quantity", ["activity", "decay_heat", "decay_photon_spectrum"]
)
def test_an_unknown_per_is_refused_by_name(with_volume, quantity):
    """Not silently treated as the default, which would be a wrong unit."""
    with pytest.raises(ValueError, match="unknown per="):
        getattr(with_volume, quantity)(per="kg")


def test_contact_dose_takes_no_per(with_volume):
    """It is a half-space estimate: a bigger lump of the same material reads the
    same, so it never needed a volume and must not grow a `per`."""
    with pytest.raises(TypeError):
        with_volume.contact_dose(per="g")
