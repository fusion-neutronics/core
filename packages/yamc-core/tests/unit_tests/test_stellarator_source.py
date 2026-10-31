"""The stellarator plasma source and its Fourier-Zernike shape.

The Rust core tests the physics (basis evaluation against DESC's conventions,
volume, sampled moments against quadrature); these cover the Python surface.
"""
import math

import pytest

import yamc

R0, A, E, NFP = 600.0, 150.0, 50.0, 5

# Rotating ellipse R = R0 + a rho cos(t) + e rho cos(t - N z),
# Z = a rho sin(t) - e rho sin(t - N z), in DESC's product basis.
ELLIPSE = dict(
    num_field_periods=NFP,
    r_modes=[(0, 0, 0), (1, 1, 0), (1, 1, 1), (1, -1, -1)],
    r_lmn=[R0, A, E, E],
    z_modes=[(1, -1, 0), (1, -1, 1), (1, 1, -1)],
    z_lmn=[A, -E, E],
)

PROFILES = dict(
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
    pedestal_rho=0.8,
)


@pytest.fixture(scope="module")
def shape():
    return yamc.sources.StellaratorShape(**ELLIPSE)


@pytest.fixture(scope="module")
def sources(shape):
    return yamc.sources.stellarator_source(shape=shape, **PROFILES)


def test_shape_round_trips_its_coefficients(shape):
    assert shape.num_field_periods == NFP
    assert dict(zip(shape.r_modes, shape.r_lmn)) == dict(
        zip(ELLIPSE["r_modes"], ELLIPSE["r_lmn"])
    )
    assert dict(zip(shape.z_modes, shape.z_lmn)) == dict(
        zip(ELLIPSE["z_modes"], ELLIPSE["z_lmn"])
    )


def test_shape_volume_and_evaluation(shape):
    assert shape.volume() == pytest.approx(2 * math.pi**2 * R0 * (A + E) * (A - E))

    r, z = shape.evaluate(1.0, 0.0, 0.0)
    assert (r, z) == pytest.approx((R0 + A + E, 0.0))

    thetas = [2 * math.pi * i / 8 for i in range(8)]
    r, z = shape.evaluate(0.5, thetas, 0.3)
    assert len(r) == len(z) == 8
    for theta, ri, zi in zip(thetas, r, z):
        angle = theta - NFP * 0.3
        assert ri == pytest.approx(R0 + 0.5 * (A * math.cos(theta) + E * math.cos(angle)))
        assert zi == pytest.approx(0.5 * (A * math.sin(theta) - E * math.sin(angle)))


def test_sources_are_flux_shells_with_ballabio_spectra(sources, shape):
    assert len(sources) == 100  # 50 shells for each of D-T and D-D
    assert sum(source.strength for source in sources) == pytest.approx(1.0)
    for source in sources:
        assert isinstance(source, yamc.NeutronSource)
        assert isinstance(source.position, yamc.sources.FluxShell)
        assert isinstance(source.energy, yamc.sources.Normal)
        assert source.position.shape.num_field_periods == NFP
    inner, outer = sources[0].position.rho
    assert (inner, outer) == pytest.approx((0.0, 0.02))


def test_samples_lie_inside_the_plasma(sources):
    for source in sources[::10]:
        positions, _ = source.sample_n(200)
        for x, y, z in positions:
            r = math.hypot(x, y)
            assert R0 - A - E - 1e-9 <= r <= R0 + A + E + 1e-9
            assert abs(z) <= A + E + 1e-9


def test_deuterium_only_fuel_gives_dd_neutrons_alone(shape):
    sources = yamc.sources.stellarator_source(shape=shape, **PROFILES, fuel={"D": 1.0})
    assert len(sources) == 50
    assert all(2e6 < source.energy.mean < 3e6 for source in sources)


def test_sources_drive_a_model(sources):
    sphere = yamc.Sphere(radius=2000.0, boundary="vacuum")
    model = yamc.Model(yamc.Geometry([yamc.Cell(region=sphere.below)]), source=sources)
    assert len(model.source) == len(sources)


@pytest.mark.parametrize(
    "override, message",
    [
        (dict(num_field_periods=0), "num_field_periods"),
        (dict(r_lmn=[R0, A, E]), "same length"),
        (dict(r_modes=[(0, 0, 0), (2, 1, 0), (1, 1, 1), (1, -1, -1)]), "even"),
        (dict(r_lmn=[100.0, A, E, E]), "R ="),
        (dict(r_lmn=[R0, A, E, math.nan]), "finite"),
    ],
)
def test_invalid_shapes_are_rejected(override, message):
    with pytest.raises(ValueError, match=message):
        yamc.sources.StellaratorShape(**dict(ELLIPSE, **override))


def test_self_intersecting_shape_is_rejected():
    with pytest.raises(ValueError, match="changes sign"):
        yamc.sources.StellaratorShape(
            num_field_periods=1,
            r_modes=[(0, 0, 0), (1, 1, 0), (2, 2, 0)],
            r_lmn=[R0, A, 1.5 * A],
            z_modes=[(1, -1, 0)],
            z_lmn=[A],
        )


@pytest.mark.parametrize(
    "override, message",
    [
        (dict(pedestal_rho=1.0), "pedestal_rho must be between 0 and 1"),
        (dict(mode="X"), "mode must be one of"),
        (dict(fuel={"T": 1.0}), "T-T fusion is not supported"),
        (dict(ion_density_centre=math.inf), "finite"),
    ],
)
def test_invalid_plasma_parameters_are_rejected(shape, override, message):
    with pytest.raises(ValueError, match=message):
        yamc.sources.stellarator_source(shape=shape, **dict(PROFILES, **override))
