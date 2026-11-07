"""FluxSurface and OffsetSurface: stellarator and tokamak surfaces as ordinary CSG surfaces."""

import math
import random

import warnings

import pytest

import yamc


def _shape():
    """The rotating ellipse of examples/python/stellarator_plasma_source.py:
    at zeta = 0 the boundary is R = 600 + 200 cos(theta), Z = 100 sin(theta)."""
    a, e = 150.0, 50.0
    return yamc.sources.StellaratorShape(
        num_field_periods=5,
        r_modes=[(0, 0, 0), (1, 1, 0), (1, 1, 1), (1, -1, -1)],
        r_lmn=[600.0, a, e, e],
        z_modes=[(1, -1, 0), (1, -1, 1), (1, 1, -1)],
        z_lmn=[a, -e, e],
    )


def _tokamak():
    """A D-shaped plasma: R0 = 600 cm, a = 150 cm, with an outward Shafranov
    shift, which is zero on the boundary."""
    return yamc.sources.TokamakShape(
        major_radius=600.0,
        minor_radius=150.0,
        elongation=1.7,
        triangularity=0.45,
        shafranov_factor=20.0,
    )


def test_a_flux_surface_is_an_ordinary_surface():
    plasma = yamc.FluxSurface(_shape(), surface_id=1, name="plasma")
    assert isinstance(plasma, yamc.Surface)
    assert repr(plasma) == "Surface(id=1, type=FluxSurface, boundary=None)"
    assert plasma.distance == 0.0
    # Outboard midplane at zeta = 0 is R = 800: evaluate is a signed distance.
    assert plasma.evaluate((801.0, 0.0, 0.0)) == pytest.approx(1.0, abs=1e-9)
    assert plasma.evaluate((799.0, 0.0, 0.0)) == pytest.approx(-1.0, abs=1e-9)
    assert plasma.below.contains((700.0, 0.0, 0.0))
    assert (plasma.above & yamc.Sphere(radius=1500.0).below).contains((900.0, 0.0, 0.0))


def test_offset_layers_are_true_distances_that_add_up():
    plasma = yamc.FluxSurface(_shape())
    first_wall = yamc.OffsetSurface(plasma, 15.0, surface_id=2, boundary="vacuum")
    back = yamc.OffsetSurface(first_wall, 2.0, name="first_wall_back")
    assert (first_wall.distance, back.distance) == (15.0, 17.0)
    assert repr(back) == "Surface(id=None, type=OffsetSurface(distance=17.0), boundary=None)"
    assert repr(first_wall).endswith("boundary='vacuum')")
    # 15 cm straight out from the outboard midplane at zeta = 0, where the
    # boundary is at its largest R for every zeta, so the cut's closest
    # point is the closest point in 3-D.
    assert first_wall.evaluate((815.0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-9)
    assert back.evaluate((815.0, 0.0, 0.0)) == pytest.approx(-2.0, abs=1e-9)


def test_layers_beyond_the_crease_free_distance_warn():
    # A circular torus, R = 600 + 150 cos(theta), Z = 150 sin(theta): its
    # layers are smooth out to 450 cm, where the inboard side reaches the axis.
    torus = yamc.sources.StellaratorShape(
        num_field_periods=1,
        r_modes=[(0, 0, 0), (1, 1, 0)],
        r_lmn=[600.0, 150.0],
        z_modes=[(1, -1, 0)],
        z_lmn=[150.0],
    )
    plasma = yamc.FluxSurface(torus)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yamc.OffsetSurface(plasma, 440.0)
        yamc.OffsetSurface(yamc.OffsetSurface(plasma, 300.0), 140.0)
    with pytest.warns(UserWarning, match=r"offset of 460\.00 cm .*\(about 450\.00 cm"):
        layer = yamc.OffsetSurface(plasma, 460.0)
    assert layer.distance == 460.0
    # Offsets of offsets add, so 300.1 + 150.1 cm is past it too.
    with pytest.warns(UserWarning, match=r"offset of 450\.20 cm .* exactly 450\.20 cm"):
        yamc.OffsetSurface(yamc.OffsetSurface(plasma, 300.1), 150.1)


def test_bad_inputs_fail_with_clear_messages():
    with pytest.raises(ValueError, match=r"rho must be in \(0, 1\]"):
        yamc.FluxSurface(_shape(), rho=1.5)
    with pytest.raises(ValueError, match="only flux surfaces"):
        yamc.OffsetSurface(yamc.Sphere(radius=10.0), 1.0)
    with pytest.raises(ValueError, match=r"FluxSurface\(shape, rho < 1\)"):
        yamc.OffsetSurface(yamc.FluxSurface(_shape()), -80.0)
    with pytest.raises(
        TypeError, match=r"shape must be a yamc.sources.StellaratorShape or TokamakShape \(got str\)"
    ):
        yamc.FluxSurface("not a shape")


def test_the_plasma_cell_holds_the_plasma_volume():
    shape = _shape()
    inside = yamc.FluxSurface(shape).below
    result = inside.calculate_volume(samples=200_000, seed=7)
    assert abs(result.volume - shape.volume()) < 4.0 * result.std_dev


def test_a_tokamak_flux_surface_is_an_ordinary_surface():
    shape = _tokamak()
    plasma = yamc.FluxSurface(shape, surface_id=1, name="plasma")
    assert isinstance(plasma, yamc.Surface)
    assert repr(plasma) == "Surface(id=1, type=FluxSurface, boundary=None)"
    # The outboard midplane of the boundary is R0 + a = 750 cm, the same at
    # every toroidal angle.
    assert plasma.evaluate((751.0, 0.0, 0.0)) == pytest.approx(1.0, abs=1e-9)
    assert plasma.evaluate((0.0, -749.0, 0.0)) == pytest.approx(-1.0, abs=1e-9)
    # The shape's points at rho = 0.5 are on its rho = 0.5 surface, which is
    # inside the boundary.
    core = yamc.FluxSurface(shape, rho=0.5)
    for theta in (0.0, 1.0, 2.5, 4.0):
        r, z = shape.evaluate(0.5, theta)
        phi = 0.7 * theta
        point = (r * math.cos(phi), r * math.sin(phi), z)
        assert core.evaluate(point) == pytest.approx(0.0, abs=1e-9)
        assert plasma.below.contains(point)


def test_offsets_of_a_circular_tokamak_are_tori():
    circle = yamc.sources.TokamakShape(
        major_radius=600.0,
        minor_radius=150.0,
        elongation=1.0,
        triangularity=0.0,
        shafranov_factor=0.0,
    )
    first_wall = yamc.OffsetSurface(yamc.FluxSurface(circle), 15.0)
    rng = random.Random(3)
    for _ in range(50):
        point = (rng.uniform(-900, 900), rng.uniform(-900, 900), rng.uniform(-300, 300))
        torus = math.hypot(math.hypot(point[0], point[1]) - 600.0, point[2]) - 165.0
        assert first_wall.evaluate(point) == pytest.approx(torus, abs=1e-9)


def test_bad_tokamak_inputs_fail_with_clear_messages():
    shape = _tokamak()
    for rho, shown in ((0.0, "0"), (1.5, "1.5")):
        with pytest.raises(
            ValueError,
            match=rf"rho must be in \(0, 1\] \(got {shown}\); a surface outside the plasma is an "
            r"OffsetSurface of the rho = 1 flux surface",
        ):
            yamc.FluxSurface(shape, rho=rho)
    with pytest.raises(ValueError, match=r"FluxSurface\(shape, rho < 1\)"):
        yamc.OffsetSurface(yamc.FluxSurface(shape), -120.0)


def test_the_tokamak_plasma_cell_holds_the_plasma_volume():
    shape = _tokamak()
    inside = yamc.FluxSurface(shape).below
    result = inside.calculate_volume(samples=200_000, seed=7)
    assert abs(result.volume - shape.volume()) < 4.0 * result.std_dev


def test_the_tokamak_source_lies_inside_the_plasma_surface():
    shape = _tokamak()
    plasma = yamc.FluxSurface(shape).below
    sources = yamc.sources.tokamak_source(
        shape=shape,
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
        pedestal_radius=0.8 * shape.minor_radius,
    )
    # Birth positions in proportion to each source's strength, as a model
    # draws them. Each source emits across a whole (R, Z) voxel, so the
    # voxels on the edge reach past the boundary, but together they carry
    # under 1e-8 of the emission and none is drawn from here.
    positions = [
        tuple(p) for source in sources for p in source.sample_n(round(source.strength * 20_000))[0]
    ]
    assert len(positions) > 19_000
    assert all(plasma.contains(p) for p in positions)


@pytest.mark.parametrize("shape", [_shape(), _tokamak()], ids=["stellarator", "tokamak"])
def test_a_layered_model_saves_and_loads(tmp_path, shape):
    plasma = yamc.FluxSurface(shape, surface_id=1)
    wall = yamc.OffsetSurface(plasma, 15.0, surface_id=2)
    world = yamc.Sphere(radius=1200.0, surface_id=3, boundary="vacuum")
    cells = [
        yamc.Cell(region=plasma.below, name="plasma"),
        yamc.Cell(region=plasma.above & wall.below, name="scrape_off_layer"),
        yamc.Cell(region=wall.above & world.below, name="outside"),
    ]
    model = yamc.Model(
        geometry=yamc.Geometry(cells),
        source=yamc.NeutronSource(energy=yamc.sources.Discrete([14.06e6], [1.0])),
    )
    path = tmp_path / "plasma.json"
    model.save(str(path))
    # To the bit: a tokamak's mode coefficients are full 17-digit floats.
    assert yamc.Model.load(str(path)).to_json() == model.to_json()
