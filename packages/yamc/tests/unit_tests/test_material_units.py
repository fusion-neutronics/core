"""`units=` on the material quantities: activity, decay heat, the decay photon
lines and continua, and the contact dose.

The solve holds atom DENSITIES, so the per-volume answer is the one it natively
has and `volume` is only the multiplier that makes it a total. These check that
per-volume and per-mass units are reachable without inventing a volume, that
the default is the total in Bq or W, and that a unit of the wrong kind is
refused rather than read as something else.
"""

import pytest
import yamc

DENSITY_G_CM3 = 7.874
VOLUME_CM3 = 1000.0


def _irradiated(volume):
    yamc.neutron_cross_section_data = "endf-b8.1"
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
    return results.get_final_material(material.id)


@pytest.fixture
def with_volume():
    return _irradiated(VOLUME_CM3)


@pytest.fixture
def without_volume():
    return _irradiated(None)


NATIVE = {"activity": "Bq", "decay_heat": "W"}


@pytest.mark.parametrize("quantity", ["activity", "decay_heat"])
def test_the_default_is_the_total(with_volume, quantity):
    total = getattr(with_volume, quantity)()
    assert total == getattr(with_volume, quantity)(units=NATIVE[quantity])
    assert total > 0.0


@pytest.mark.parametrize("quantity", ["activity", "decay_heat"])
def test_a_total_still_needs_a_volume(without_volume, quantity):
    with pytest.raises(ValueError, match="requires material.volume"):
        getattr(without_volume, quantity)()


def test_the_spectrum_total_still_needs_a_volume(without_volume):
    with pytest.raises(ValueError, match="requires material.volume"):
        without_volume.decay_photon_spectrum()


@pytest.mark.parametrize("quantity", ["activity", "decay_heat"])
def test_per_volume_needs_no_volume(without_volume, quantity):
    assert getattr(without_volume, quantity)(units=f"{NATIVE[quantity]}/cm3") > 0.0


def test_spectrum_per_volume_needs_no_volume(without_volume):
    energies, rates = without_volume.decay_photon_spectrum(units="1/(s cm3)")
    assert energies and all(rate > 0.0 for rate in rates)


@pytest.mark.parametrize("quantity", ["activity", "decay_heat"])
def test_per_cm3_is_the_total_divided_by_the_volume(with_volume, quantity):
    total = getattr(with_volume, quantity)()
    per_cm3 = getattr(with_volume, quantity)(units=f"{NATIVE[quantity]}/cm3")
    assert per_cm3 == pytest.approx(total / VOLUME_CM3, rel=1e-12)


@pytest.mark.parametrize("quantity", ["activity", "decay_heat"])
def test_per_m3_is_a_million_times_per_cm3(with_volume, quantity):
    unit = NATIVE[quantity]
    per_cm3 = getattr(with_volume, quantity)(units=f"{unit}/cm3")
    per_m3 = getattr(with_volume, quantity)(units=f"{unit}/m3")
    assert per_m3 == pytest.approx(1e6 * per_cm3, rel=1e-12)


@pytest.mark.parametrize("quantity", ["activity", "decay_heat"])
def test_per_g_divides_by_the_mass_density(with_volume, quantity):
    """Per mass divides by the material's OWN mass density, not the 7.874 it
    was built with. Activation moves mass between nuclides of different atomic
    mass, so the irradiated inventory here comes out at 7.873999643 g/cm3,
    4.5e-8 below the input: small, but far above the 1e-9 this asserts to."""
    density = with_volume.get_mass_density()
    assert density != pytest.approx(DENSITY_G_CM3, rel=1e-12), (
        "the point of this test is that the two differ; if activation stopped "
        "moving the density, assert against the input instead"
    )
    unit = NATIVE[quantity]
    per_cm3 = getattr(with_volume, quantity)(units=f"{unit}/cm3")
    per_g = getattr(with_volume, quantity)(units=f"{unit}/g")
    assert per_g == pytest.approx(per_cm3 / density, rel=1e-9)
    per_kg = getattr(with_volume, quantity)(units=f"{unit}/kg")
    assert per_kg == pytest.approx(1e3 * per_g, rel=1e-12)


def test_per_mass_needs_no_volume(without_volume):
    assert without_volume.activity(units="Bq/kg") > 0.0


@pytest.mark.parametrize(
    "quantity, units, scale",
    [
        ("activity", "TBq", 1e-12),
        ("activity", "Ci", 1 / 3.7e10),
        ("activity", "kBq/g", 1e-3 / (VOLUME_CM3 * DENSITY_G_CM3)),
        ("decay_heat", "kW", 1e-3),
        ("decay_heat", "MeV/s", 1 / 1.602176634e-13),
        ("decay_heat", "MW/m3", 1e-6 * 1e6 / VOLUME_CM3),
    ],
)
def test_prefixed_and_other_units(with_volume, quantity, units, scale):
    total = getattr(with_volume, quantity)()
    value = getattr(with_volume, quantity)(units=units)
    # Per mass uses the inventory's own density, 4.5e-8 off the input's.
    assert value == pytest.approx(total * scale, rel=1e-7)


@pytest.mark.parametrize("quantity", ["activity", "decay_heat"])
def test_by_nuclide_carries_the_same_unit(with_volume, quantity):
    totals = getattr(with_volume, quantity)(by_nuclide=True)
    per_g = getattr(with_volume, quantity)(by_nuclide=True, units=f"{NATIVE[quantity]}/g")
    assert set(totals) == set(per_g)
    scale = VOLUME_CM3 * with_volume.get_mass_density()
    for nuclide, value in totals.items():
        assert per_g[nuclide] == pytest.approx(value / scale, rel=1e-9)


@pytest.mark.parametrize(
    "quantity, units",
    [("activity", "W"), ("decay_heat", "Bq/kg"), ("activity", "Bq/cm2"), ("activity", "lb")],
)
def test_a_unit_of_the_wrong_kind_is_refused(with_volume, quantity, units):
    """Not silently treated as the default, which would be a wrong number."""
    with pytest.raises(ValueError, match=quantity):
        getattr(with_volume, quantity)(units=units)


def test_the_contact_dose_takes_dose_rate_units(with_volume):
    """A contact dose is already intensive: a bigger lump reads the same, so it
    takes no per-volume or per-mass unit, only a rate of its own kind."""
    gy_h = with_volume.contact_dose()
    assert with_volume.contact_dose(units="uGy/h") == pytest.approx(1e6 * gy_h, rel=1e-12)
    assert with_volume.contact_dose(units="Gy/s") == pytest.approx(gy_h / 3600, rel=1e-12)
    sv_h = with_volume.contact_dose(dose_quantity="effective")
    assert with_volume.contact_dose(dose_quantity="effective", units="mSv/h") == pytest.approx(
        1e3 * sv_h, rel=1e-12
    )
    by_nuclide = with_volume.contact_dose(by_nuclide=True, units="mGy/h")
    assert sum(by_nuclide.values()) == pytest.approx(1e3 * gy_h, rel=1e-9)


@pytest.mark.parametrize(
    "dose_quantity, units", [("absorbed-air", "Sv/h"), ("effective", "Gy/h"), ("absorbed-air", "Gy/kg")]
)
def test_the_contact_dose_refuses_another_kind(with_volume, dose_quantity, units):
    with pytest.raises(ValueError, match="contact_dose"):
        with_volume.contact_dose(dose_quantity=dose_quantity, units=units)
