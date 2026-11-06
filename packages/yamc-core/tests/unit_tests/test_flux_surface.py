"""FluxSurface and OffsetSurface: stellarator surfaces as ordinary CSG surfaces."""

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
    with pytest.raises(TypeError):
        yamc.FluxSurface("not a shape")


def test_the_plasma_cell_holds_the_plasma_volume():
    shape = _shape()
    inside = yamc.FluxSurface(shape).below
    result = inside.calculate_volume(samples=200_000, seed=7)
    assert abs(result.volume - shape.volume()) < 4.0 * result.std_dev


def test_a_layered_model_saves_and_loads(tmp_path):
    plasma = yamc.FluxSurface(_shape(), surface_id=1)
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
    path = tmp_path / "stellarator.json"
    model.save(str(path))
    assert yamc.Model.load(str(path)).to_json() == model.to_json()
