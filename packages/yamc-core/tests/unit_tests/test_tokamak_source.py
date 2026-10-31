"""The parametric tokamak plasma source and the plasma profile helpers.

The Rust core has its own tests for the physics (Bosch-Hale reactivities,
profile shapes, emission weighting); these cover the Python surface: argument
handling, the shape of what comes back, and that the sources drop straight
into a Model.
"""
import math

import pytest

import yamc

# A EU-DEMO-like plasma, the parameter set the openmc-plasma-source examples
# use, so a comparison between the two codes starts from the same numbers.
DEMO = dict(
    major_radius=906.0,
    minor_radius=292.258,
    elongation=1.557,
    triangularity=0.270,
    mode="H",
    ion_density_centre=1.09e20,
    ion_density_peaking_factor=1,
    ion_density_pedestal=1.09e20,
    ion_density_separatrix=3e19,
    ion_temperature_centre=45.9e3,
    ion_temperature_peaking_factor=8.06,
    ion_temperature_beta=6.0,
    ion_temperature_pedestal=6.09e3,
    ion_temperature_separatrix=0.1e3,
    pedestal_radius=0.8 * 292.258,
    shafranov_factor=0.44789,
)

# A coarse mesh keeps these tests quick; the default (100, 100) builds ~15000
# sources.
COARSE = dict(DEMO, mesh_resolution=(20, 20), grid_density=100)


def test_returns_neutron_sources_with_normalised_strengths():
    sources = yamc.sources.tokamak_source(**COARSE)

    assert len(sources) > 100
    assert all(isinstance(source, yamc.NeutronSource) for source in sources)
    assert sum(source.strength for source in sources) == pytest.approx(1.0)


def test_sources_are_rings_with_ballabio_spectra():
    sources = yamc.sources.tokamak_source(**COARSE)

    for source in sources:
        assert isinstance(source.position, yamc.sources.CylindricalRing)
        assert isinstance(source.energy, yamc.sources.Normal)
        assert isinstance(source.direction, yamc.sources.Isotropic)

    means = [source.energy.mean for source in sources]
    # 50:50 D-T fuel emits both D-T (~14.1 MeV) and D-D (~2.5 MeV) neutrons.
    assert any(mean > 13e6 for mean in means)
    assert any(2e6 < mean < 3e6 for mean in means)


def test_samples_land_inside_the_plasma_bounding_box():
    sources = yamc.sources.tokamak_source(**COARSE)

    shift = abs(DEMO["shafranov_factor"])
    r_max = DEMO["major_radius"] + DEMO["minor_radius"] + shift
    r_min = DEMO["major_radius"] - DEMO["minor_radius"] - shift
    z_max = DEMO["elongation"] * DEMO["minor_radius"]
    for source in sources[::20]:
        positions, energies = source.sample_n(5)
        for (x, y, z), energy in zip(positions, energies):
            assert r_min - 1e-9 <= math.hypot(x, y) <= r_max + 1e-9
            assert abs(z) <= z_max + 1e-9
            assert energy > 0.0


def test_a_toroidal_sector_only_emits_inside_itself():
    sources = yamc.sources.tokamak_source(
        **COARSE, start_angle=0.0, rotation_angle=math.pi / 2
    )

    for source in sources[::20]:
        positions, _ = source.sample_n(5)
        for x, y, _ in positions:
            assert -1e-9 <= math.atan2(y, x) <= math.pi / 2 + 1e-9


def test_deuterium_only_fuel_gives_dd_neutrons_alone():
    sources = yamc.sources.tokamak_source(**COARSE, fuel={"D": 1.0})

    assert all(source.energy.mean < 3e6 for source in sources)


def test_mesh_resolution_sets_the_number_of_sources():
    coarse = yamc.sources.tokamak_source(**dict(COARSE, mesh_resolution=(10, 10)))
    fine = yamc.sources.tokamak_source(**dict(COARSE, mesh_resolution=(20, 20)))

    assert len(fine) > len(coarse)


def test_sources_drive_a_model():
    """The source list goes straight into a Model, no adapter needed."""
    sources = yamc.sources.tokamak_source(**dict(COARSE, mesh_resolution=(5, 5)))
    sphere = yamc.Sphere(radius=2000.0, boundary="vacuum")
    cell = yamc.Cell(region=sphere.below)
    model = yamc.Model(yamc.Geometry([cell]), source=sources)

    assert len(model.source) == len(sources)


@pytest.mark.parametrize(
    "override, message",
    [
        (dict(minor_radius=2000.0), "minor_radius must be less than major_radius"),
        (dict(mode="X"), "mode must be one of"),
        (dict(rotation_angle=0.0), "rotation_angle must be a non-zero value"),
        (dict(triangularity=2.0), "triangularity must be between -1 and 1"),
        (dict(fuel={"D": 0.3, "T": 0.3}), "fuel fractions must sum to 1"),
        (dict(fuel={"He3": 1.0}), 'fuel species must be "D" or "T"'),
        (dict(fuel={"T": 1.0}), "T-T fusion is not supported"),
        (
            dict(ion_temperature_centre=0.0, ion_temperature_pedestal=0.0,
                 ion_temperature_separatrix=0.0),
            "total neutron source density is zero",
        ),
    ],
)
def test_invalid_plasma_parameters_are_rejected(override, message):
    with pytest.raises(ValueError, match=message):
        yamc.sources.tokamak_source(**dict(COARSE, **override))


def test_ion_density_profile():
    profile = dict(
        mode="H",
        ion_density_centre=1.09e20,
        ion_density_peaking_factor=1,
        ion_density_pedestal=1.09e20,
        minor_radius=292.258,
        pedestal_radius=0.8 * 292.258,
        ion_density_separatrix=3e19,
    )

    # A float in, a float out.
    assert yamc.sources.tokamak_ion_density(**profile, r=0.0) == pytest.approx(1.09e20)
    # A sequence in, a list out, in the same order.
    edges = yamc.sources.tokamak_ion_density(**profile, r=[0.0, 292.258])
    assert isinstance(edges, list)
    assert edges == pytest.approx([1.09e20, 3e19])


def test_ion_temperature_profile():
    profile = dict(
        mode="H",
        pedestal_radius=0.8 * 292.258,
        ion_temperature_pedestal=6.09e3,
        ion_temperature_centre=45.9e3,
        ion_temperature_beta=6.0,
        ion_temperature_peaking_factor=8.06,
        ion_temperature_separatrix=0.1e3,
        minor_radius=292.258,
    )

    assert yamc.sources.tokamak_ion_temperature(**profile, r=0.0) == pytest.approx(45.9e3)
    assert yamc.sources.tokamak_ion_temperature(**profile, r=292.258) == pytest.approx(0.1e3)
    # The temperature falls monotonically from the axis to the separatrix.
    radii = [i * 292.258 / 10 for i in range(11)]
    profile_values = yamc.sources.tokamak_ion_temperature(**profile, r=radii)
    assert profile_values == sorted(profile_values, reverse=True)


def test_profiles_reject_radii_outside_the_plasma():
    with pytest.raises(ValueError, match="minor radius position"):
        yamc.sources.tokamak_ion_density(
            mode="L",
            ion_density_centre=1e20,
            ion_density_peaking_factor=1,
            ion_density_pedestal=1e20,
            minor_radius=100.0,
            pedestal_radius=80.0,
            ion_density_separatrix=1e19,
            r=101.0,
        )


def test_convert_a_alpha_to_r_z():
    shape = dict(
        shafranov_factor=0.44789,
        minor_radius=292.258,
        major_radius=906.0,
        triangularity=0.270,
        elongation=1.557,
    )

    r, z = yamc.sources.tokamak_convert_a_alpha_to_r_z(a=292.258, alpha=0.0, **shape)
    assert r == pytest.approx(906.0 + 292.258)
    assert z == pytest.approx(0.0)

    # A scalar broadcasts against a sequence, and the result keeps its shape.
    radii, heights = yamc.sources.tokamak_convert_a_alpha_to_r_z(
        a=292.258, alpha=[0.0, math.pi], **shape
    )
    assert radii == pytest.approx([906.0 + 292.258, 906.0 - 292.258])
    assert heights == pytest.approx([0.0, 0.0])


def test_neutron_source_density():
    # D-T at 20 keV: 1e40 reacting pairs/m^6 times <sigma v> = 4.33e-22 m^3/s.
    density = yamc.sources.tokamak_neutron_source_density(
        ion_density=1e40, ion_temperature=20e3, reaction="DT"
    )
    assert density == pytest.approx(4.33e18, rel=0.01)

    # D-D is far weaker at the same temperature.
    dd = yamc.sources.tokamak_neutron_source_density(
        ion_density=1e40, ion_temperature=20e3, reaction="DD"
    )
    assert 0.0 < dd < 0.01 * density

    values = yamc.sources.tokamak_neutron_source_density(
        ion_density=1e40, ion_temperature=[10e3, 20e3]
    )
    assert isinstance(values, list) and values[0] < values[1]


def test_neutron_source_density_rejects_unsupported_reaction():
    with pytest.raises(ValueError, match='reaction must be "DD" or "DT"'):
        yamc.sources.tokamak_neutron_source_density(
            ion_density=1e40, ion_temperature=20e3, reaction="TT"
        )
