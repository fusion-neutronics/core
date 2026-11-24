"""Model.set_tet_density_factors: per-tet densities on a mesh geometry match the
same densities set on the materials (history for history under Woodcock), with
the factors given to each material's tets through MeshGeometry.tet_volume_ids."""

import pytest

import yamc

MESH = "crates/yamt/tests/data/two_region_tets.arrow"
RHO = 40.0


def _iron(density):
    iron = yamc.Material(composition={"Fe56": 1.0}, density=density, temperature=294)
    iron.read_nuclear_data({"Fe56": "tests/Fe56.arrow"})
    return iron


def _model(fuel_density, moderator_density):
    fuel, moderator = _iron(fuel_density), _iron(moderator_density)
    mesh = yamc.MeshGeometry.from_arrow(MESH, {"fuel": fuel, "moderator": moderator})
    tally = yamc.Tally(scores=["heating"], materials=[fuel, moderator])
    model = yamc.Model(
        geometry=mesh,
        source=yamc.NeutronSource(position=(0.25, 0.5, 0.5), energy=14.06e6),
        tallies=[tally],
        tracking_mode="woodcock",
        verbose=[],
    )
    return model, mesh, tally


def _heating(model, tally):
    return list(model.simulate_transport(total_particles=20_000, seed=3)[tally].mean)


def _factors_by_volume(mesh, fuel, moderator):
    by_name = {"fuel": fuel, "moderator": moderator}
    names = mesh.volume_names
    return [by_name[names[v]] for v in mesh.tet_volume_ids]


def test_tet_volume_ids_follow_the_tets():
    _, mesh, _ = _model(RHO, RHO)
    ids = mesh.tet_volume_ids
    assert len(ids) == len(mesh.tetrahedra)
    vertices = mesh.vertices
    for tet, v in zip(mesh.tetrahedra, ids):
        x = sum(vertices[i][0] for i in tet) / 4
        assert mesh.volume_names[v] == ("fuel" if x < 0.5 else "moderator")


def test_factors_match_material_densities():
    model, mesh, tally = _model(RHO, RHO)
    unscaled = _heating(model, tally)
    model.set_tet_density_factors(_factors_by_volume(mesh, 0.5, 1.5))
    scaled = _heating(model, tally)
    reference = _heating(*_model(0.5 * RHO, 1.5 * RHO)[::2])
    assert scaled == pytest.approx(reference, rel=1e-12)
    assert scaled != pytest.approx(unscaled, rel=0.05)
    model.set_tet_density_factors(None)
    assert _heating(model, tally) == pytest.approx(unscaled, rel=1e-12)


def test_bad_factors_raise():
    model, mesh, _ = _model(RHO, RHO)
    n = len(mesh.tetrahedra)
    with pytest.raises(ValueError, match="one density factor per tet"):
        model.set_tet_density_factors([1.0] * (n - 1))
    with pytest.raises(ValueError, match="non-negative"):
        model.set_tet_density_factors([-1.0] * n)


def test_surface_tracking_is_refused():
    model, mesh, _ = _model(RHO, RHO)
    model.tracking_mode = "surface"
    model.set_tet_density_factors([1.0] * len(mesh.tetrahedra))
    with pytest.raises(Exception, match="hybrid"):
        model.simulate_transport(total_particles=10)


def test_csg_geometry_is_refused():
    sphere = yamc.Sphere(radius=10.0, boundary="vacuum")
    model = yamc.Model(
        geometry=yamc.Geometry([yamc.Cell(region=sphere.below, material=_iron(RHO))]),
        source=yamc.NeutronSource(energy=14.06e6),
        verbose=[],
    )
    with pytest.raises(ValueError, match="mesh geometry"):
        model.set_tet_density_factors([1.0])


def test_factors_take_any_iterable_of_numbers():
    model, mesh, _ = _model(RHO, RHO)
    n = len(mesh.tetrahedra)
    model.set_tet_density_factors(1.0 for _ in range(n))
    model.set_tet_density_factors(range(1, n + 1))
    np = pytest.importorskip("numpy")
    model.set_tet_density_factors(np.ones(n))
    with pytest.raises(TypeError):
        model.set_tet_density_factors(["one"] * n)


def test_geometry_factors_are_where_the_model_starts():
    model, mesh, tally = _model(RHO, RHO)
    factors = _factors_by_volume(mesh, 0.5, 1.5)
    model.set_tet_density_factors(factors)
    expected = _heating(model, tally)

    fuel, moderator = _iron(RHO), _iron(RHO)
    geometry = yamc.MeshGeometry.from_arrow(MESH, {"fuel": fuel, "moderator": moderator})
    geometry.tet_density_factors = factors
    assert geometry.tet_density_factors == pytest.approx(factors)
    tally = yamc.Tally(scores=["heating"], materials=[fuel, moderator])
    built = yamc.Model(
        geometry=geometry,
        source=yamc.NeutronSource(position=(0.25, 0.5, 0.5), energy=14.06e6),
        tallies=[tally],
        tracking_mode="woodcock",
        verbose=[],
    )
    assert _heating(built, tally) == pytest.approx(expected, rel=1e-12)
    # the model's copy changes; the geometry keeps its field
    built.set_tet_density_factors(None)
    assert geometry.tet_density_factors == pytest.approx(factors)


def test_geometry_factors_are_checked():
    _, mesh, _ = _model(RHO, RHO)
    n = len(mesh.tetrahedra)
    assert mesh.tet_density_factors is None
    with pytest.raises(ValueError, match="one density factor per tet"):
        mesh.tet_density_factors = [1.0] * (n - 1)
    with pytest.raises(ValueError, match="non-negative"):
        mesh.tet_density_factors = [-1.0] * n
    mesh.tet_density_factors = (1.0 for _ in range(n))
    assert mesh.tet_density_factors == [1.0] * n
    mesh.tet_density_factors = None
    assert mesh.tet_density_factors is None
