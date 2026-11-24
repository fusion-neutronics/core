"""The stellarator plasma source and its Fourier-Zernike shape.

The Rust core tests the physics (basis evaluation against DESC's conventions,
volume, sampled moments against quadrature, the VMEC fit); these cover the
Python surface.
"""
import math
import random

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

# The same ellipse as a VMEC wout file holds it, in cm, on the surfaces
# s_j = j / (NS - 1) of rho = sqrt(s).
NS = 21
VMEC_RHO = [math.sqrt(j / (NS - 1)) for j in range(NS)]
VMEC = dict(
    num_field_periods=NFP,
    xm=[0.0, 1.0, 1.0],
    xn=[0.0, 0.0, float(NFP)],
    rmnc=[[R0, A * rho, E * rho] for rho in VMEC_RHO],
    zmns=[[0.0, A * rho, -E * rho] for rho in VMEC_RHO],
)


def vmec_point(vmec, j, theta, zeta):
    """VMEC's (R, Z) on surface j, summed term by term."""
    r = z = 0.0
    for k, (m, n) in enumerate(zip(vmec["xm"], vmec["xn"])):
        angle = m * theta - n * zeta
        r += vmec["rmnc"][j][k] * math.cos(angle)
        z += vmec["zmns"][j][k] * math.sin(angle)
        if "rmns" in vmec:
            r += vmec["rmns"][j][k] * math.sin(angle)
            z += vmec["zmnc"][j][k] * math.cos(angle)
    return r, z


def inside_ellipse(x, y, z):
    """Whether a point is inside the rotating ellipse: its cut at phi has
    semi-axes a + e and a - e, the major one turned by N phi / 2."""
    turn = NFP * math.atan2(y, x) / 2
    dr, dz = math.hypot(x, y) - R0, z
    u = math.cos(turn) * dr + math.sin(turn) * dz
    v = -math.sin(turn) * dr + math.cos(turn) * dz
    return math.hypot(u / (A + E), v / (A - E)) <= 1 + 1e-9

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


@pytest.mark.parametrize(
    "rho, theta, zeta, message",
    [
        (1.5, 0.0, 0.0, r"rho must be between 0 and 1 \(got 1.5\)"),
        (1.0, math.nan, 0.0, r"theta must be a finite number \(got NaN\)"),
        (0.5, [0.0, 1.0], math.inf, r"zeta must be a finite number \(got inf\)"),
    ],
)
def test_shape_evaluate_rejects_bad_coordinates(shape, rho, theta, zeta, message):
    with pytest.raises(ValueError, match=message):
        shape.evaluate(rho, theta, zeta)


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


@pytest.fixture(scope="module")
def vmec_shape():
    return yamc.sources.StellaratorShape.from_vmec(**VMEC)


def test_from_vmec_fits_the_vmec_surfaces(vmec_shape):
    # One Zernike polynomial per term is exact, so the fit is DESC's series.
    assert vmec_shape.num_field_periods == NFP
    for modes, lmn, key in (("r_modes", "r_lmn", "r"), ("z_modes", "z_lmn", "z")):
        fitted = dict(zip(getattr(vmec_shape, modes), getattr(vmec_shape, lmn)))
        expected = dict(zip(ELLIPSE[f"{key}_modes"], ELLIPSE[f"{key}_lmn"]))
        assert fitted.keys() == expected.keys()
        for mode, value in expected.items():
            assert fitted[mode] == pytest.approx(value, abs=1e-9 * R0)
    assert vmec_shape.fit_error < 1e-9 * A
    assert vmec_shape.volume() == pytest.approx(2 * math.pi**2 * R0 * (A + E) * (A - E))
    assert "fit_error=" in repr(vmec_shape)


def test_from_vmec_keeps_vmec_angles(vmec_shape):
    # The shape at (rho, theta, zeta) is VMEC's surface s = rho**2 at the same
    # angles, so theta runs counter-clockwise in (R, Z) as in a wout file.
    rng = random.Random(1)
    for j in (1, 8, NS - 1):
        thetas = [rng.uniform(0, 2 * math.pi) for _ in range(10)]
        zetas = [rng.uniform(0, 2 * math.pi) for _ in range(10)]
        radii, heights = vmec_shape.evaluate(VMEC_RHO[j], thetas, zetas)
        for theta, zeta, r, z in zip(thetas, zetas, radii, heights):
            assert (r, z) == pytest.approx(vmec_point(VMEC, j, theta, zeta), abs=1e-9 * R0)
    _, z_up = vmec_shape.evaluate(1.0, math.pi / 2, 0.0)
    assert z_up > 0


def test_from_vmec_takes_wout_arrays_as_they_are(vmec_shape):
    np = pytest.importorskip("numpy")
    # A wout file stores xm and xn as floats, and the coefficients as one row
    # per surface.
    shape = yamc.sources.StellaratorShape.from_vmec(
        num_field_periods=NFP,
        xm=np.array(VMEC["xm"]),
        xn=np.array(VMEC["xn"]),
        rmnc=np.array(VMEC["rmnc"]),
        zmns=np.array(VMEC["zmns"]),
    )
    assert shape.r_modes == vmec_shape.r_modes
    assert shape.r_lmn == pytest.approx(vmec_shape.r_lmn)


def test_from_vmec_fits_an_asymmetric_equilibrium():
    # lasym: R gains 10 rho sin(t) and the plasma is lifted 20 cm.
    vmec = dict(
        VMEC,
        rmns=[[0.0, 10.0 * rho, 0.0] for rho in VMEC_RHO],
        zmnc=[[20.0, 0.0, 0.0] for _ in VMEC_RHO],
    )
    shape = yamc.sources.StellaratorShape.from_vmec(**vmec)
    assert shape.fit_error < 1e-9 * A
    rng = random.Random(2)
    for j in (3, NS - 1):
        theta, zeta = rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi)
        r, z = shape.evaluate(VMEC_RHO[j], theta, zeta)
        assert (r, z) == pytest.approx(vmec_point(vmec, j, theta, zeta), abs=1e-9 * R0)


def test_from_vmec_refuses_surfaces_not_smooth_in_rho():
    # The minor radius jumps by 20% half way out.
    step = [1.2 if rho > 0.5 else 1.0 for rho in VMEC_RHO]
    vmec = dict(
        VMEC,
        rmnc=[[R0, s * A * rho, E * rho] for s, rho in zip(step, VMEC_RHO)],
        zmns=[[0.0, s * A * rho, -E * rho] for s, rho in zip(step, VMEC_RHO)],
    )
    with pytest.raises(
        ValueError,
        match=r"misses the VMEC surfaces by up to [0-9.]+ cm, on surface j = \d+ of 21 "
        r"\(rho = [0-9.]+\).*more than the [0-9.]+ cm allowed, 1% of the minor radius",
    ):
        yamc.sources.StellaratorShape.from_vmec(**vmec)


@pytest.mark.parametrize(
    "override, message",
    [
        (dict(xn=[0.0, 0.0, 3.0]), "multiples of num_field_periods"),
        (dict(xm=[0.0, 1.5, 1.0]), "whole numbers"),
        (dict(xn=[0.0, 0.0]), "same length"),
        (dict(zmns=VMEC["zmns"][:5]), "surfaces and rmnc 21"),
        (dict(rmnc=[row + [0.0] for row in VMEC["rmnc"]]), "one per mode"),
        (dict(rmnc=[[R0, math.nan, 0.0]] * NS), "finite"),
    ],
)
def test_from_vmec_rejects_bad_arrays(override, message):
    with pytest.raises(ValueError, match=message):
        yamc.sources.StellaratorShape.from_vmec(**dict(VMEC, **override))


def test_fit_error_is_none_for_fourier_zernike_coefficients(shape):
    assert shape.fit_error is None
    assert "fit_error" not in repr(shape)


def test_vmec_shape_samples_inside_the_plasma(vmec_shape):
    sources = yamc.sources.stellarator_source(shape=vmec_shape, **PROFILES)
    assert sum(source.strength for source in sources) == pytest.approx(1.0)
    for source in sources[::10]:
        positions, _ = source.sample_n(200)
        assert all(inside_ellipse(*position) for position in positions)
