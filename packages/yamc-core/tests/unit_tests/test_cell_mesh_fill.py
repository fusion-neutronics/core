"""Hybrid geometry: a CSG Cell filled by a MeshGeometry (issue #232).

CI-safe construction and geometry-query tests using the cadquery-free
two-region fixture (a unit cube split at x=0.5 into "fuel" (x in [0, 0.5])
and "moderator" volumes). Transport agreement lives in
``packages/yamc-core/tests/integration_tests/test_cell_mesh_fill_transport.py``.
"""

import numpy as np
import pytest
import yamc

TWO_REGION_ARROW = "crates/yamt/tests/data/two_region.arrow"


def _dummy_material(name):
    """Minimal Material (no nuclear data): geometry-only tests."""
    return yamc.Material(
        composition={"H": 1.0}, density=1.0, units="g/cc", name=name,
    )


def _mesh(fuel=None, moderator=None, **kwargs):
    return yamc.MeshGeometry(
        TWO_REGION_ARROW,
        {
            "fuel": fuel or _dummy_material("fuel"),
            "moderator": moderator or _dummy_material("moderator"),
        },
        **kwargs,
    )


def _filled_cell(mesh=None, **kwargs):
    """Sphere chamber (r=3 around the unit cube) filled by the mesh."""
    sphere = yamc.Sphere(x0=0.5, y0=0.5, z0=0.5, radius=3.0, boundary="vacuum")
    return yamc.Cell(
        region=sphere.below,
        material=_dummy_material("complement"),
        name="chamber",
        fill=mesh or _mesh(),
        **kwargs,
    )


def test_filled_geometry_flattens_mesh_volumes_into_cells():
    cell = _filled_cell()
    geometry = yamc.Geometry([cell])

    cells = geometry.cells
    assert len(cells) == 3, "host + two embedded mesh volumes"
    ids = [c.id for c in cells]
    assert len(set(ids)) == 3, f"cell ids must be unique: {ids}"
    names = [c.name for c in cells]
    assert names[0] == "chamber"
    assert any("fuel" in n for n in names[1:])
    assert any("moderator" in n for n in names[1:])
    assert all(n.startswith("chamber/") for n in names[1:])
    # Embedded volumes carry the mesh's analytic volume (0.5 cm^3 each).
    for c in cells[1:]:
        assert c.volume == pytest.approx(0.5, rel=0.1)


def test_point_location_resolves_the_fill():
    geometry = yamc.Geometry([_filled_cell()])

    assert geometry.find_cell(0.25, 0.5, 0.5).material.name == "fuel"
    assert geometry.find_cell(0.75, 0.5, 0.5).material.name == "moderator"
    # Gap between the cube and the sphere: the cell's own material.
    assert geometry.find_cell(0.5, 0.5, 2.0).material.name == "complement"
    # Outside the region entirely.
    assert geometry.find_cell(0.5, 0.5, 5.0) is None


def test_translation_moves_the_body():
    cell = _filled_cell(translation=(0.0, 0.0, 1.2))
    geometry = yamc.Geometry([cell])
    assert geometry.find_cell(0.25, 0.5, 1.7).material.name == "fuel"
    assert geometry.find_cell(0.25, 0.5, 0.5).material.name == "complement"


def test_rotation_turns_the_body():
    # 90 degrees about z maps mesh (x, y) to world (-y, x): the fuel half
    # (x_mesh < 0.5) becomes the y_world < 0.5 half at x_world in [-1, 0].
    cell = _filled_cell(rotation=(0.0, 0.0, 90.0))
    geometry = yamc.Geometry([cell])
    assert geometry.find_cell(-0.5, 0.25, 0.5).material.name == "fuel"
    assert geometry.find_cell(-0.5, 0.75, 0.5).material.name == "moderator"
    assert geometry.find_cell(0.5, 0.5, 0.5).material.name == "complement"


def test_fill_accessors():
    cell = _filled_cell(translation=(1.0, 2.0, 3.0), rotation=(0.0, 0.0, 45.0))
    assert cell.fill is not None
    assert cell.fill.num_volumes == 2
    assert cell.translation == (1.0, 2.0, 3.0)
    assert cell.rotation == (0.0, 0.0, 45.0)
    assert cell.allow_clipping is False
    plain = yamc.Cell(region=yamc.Sphere(radius=1.0).below)
    assert plain.fill is None
    assert plain.translation is None
    assert plain.rotation is None


def test_sample_slice_shows_the_body():
    geometry = yamc.Geometry([_filled_cell()])
    fuel_id = geometry.find_cell(0.25, 0.5, 0.5).material.id
    comp_id = geometry.find_cell(0.5, 0.5, 2.0).material.id

    slice_data = geometry.sample_slice(
        origin=(0.5, 0.5, 0.5), width=(4.0, 4.0), resolution=(41, 41), basis="xy"
    )
    material_ids = np.asarray(slice_data.material_ids)
    centre = material_ids[20, 18]  # x ~ 0.3, y ~ 0.5: inside the fuel half
    gap = material_ids[20, 3]  # x ~ -1.2: inside the sphere, outside the cube
    assert centre == fuel_id
    assert gap == comp_id


def test_protrusion_is_a_construction_error():
    small = yamc.Sphere(x0=0.5, y0=0.5, z0=0.5, radius=0.4, boundary="vacuum")
    cell = yamc.Cell(
        region=small.below, material=_dummy_material("complement"), fill=_mesh()
    )
    with pytest.raises(ValueError, match="protrudes"):
        yamc.Geometry([cell])


def test_allow_clipping_permits_protrusion():
    small = yamc.Sphere(x0=0.5, y0=0.5, z0=0.5, radius=0.4, boundary="vacuum")
    cell = yamc.Cell(
        region=small.below,
        material=_dummy_material("complement"),
        fill=_mesh(),
        allow_clipping=True,
    )
    geometry = yamc.Geometry([cell])
    assert geometry.find_cell(0.45, 0.5, 0.5).material.name == "fuel"
    # Beyond the sphere the body is clipped away.
    assert geometry.find_cell(0.95, 0.5, 0.5) is None


def test_transform_without_fill_is_rejected():
    sphere = yamc.Sphere(radius=3.0)
    with pytest.raises(ValueError, match="only apply to a cell"):
        yamc.Cell(region=sphere.below, translation=(1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="only apply to a cell"):
        yamc.Cell(region=sphere.below, rotation=(0.0, 0.0, 90.0))
    with pytest.raises(ValueError, match="only apply to a cell"):
        yamc.Cell(region=sphere.below, allow_clipping=True)


def test_fill_must_be_a_mesh_geometry():
    sphere = yamc.Sphere(radius=3.0)
    with pytest.raises(TypeError, match="MeshGeometry"):
        yamc.Cell(region=sphere.below, fill=_dummy_material("oops"))


def test_graveyard_fill_is_rejected():
    mesh = _mesh(graveyard_offset=1.0)
    cell = _filled_cell(mesh=mesh)
    with pytest.raises(ValueError, match="graveyard"):
        yamc.Geometry([cell])


def test_implicit_complement_material_fill_is_rejected():
    mesh = yamc.MeshGeometry(
        TWO_REGION_ARROW,
        {"fuel": _dummy_material("fuel"), "moderator": _dummy_material("moderator")},
        implicit_complement_material=_dummy_material("air"),
    )
    cell = _filled_cell(mesh=mesh)
    with pytest.raises(ValueError, match="implicit_complement_material"):
        yamc.Geometry([cell])


def test_rebuilding_geometry_from_filled_cells_is_rejected():
    """Cell views of a filled geometry do not carry the fill; rebuilding
    from them would silently drop the mesh bodies."""
    geometry = yamc.Geometry([_filled_cell()])
    with pytest.raises(ValueError, match="cannot seed"):
        yamc.Geometry(geometry.cells)


def test_two_fills_with_default_ids_are_rejected():
    """Each MeshGeometry auto-assigns material ids independently, so two
    fills built from unrelated materials collide; the constructor must
    fail fast instead of silently conflating them."""
    sphere_a = yamc.Sphere(x0=0.5, y0=0.5, z0=0.5, radius=3.0, boundary="vacuum")
    sphere_b = yamc.Sphere(x0=10.5, y0=0.5, z0=0.5, radius=3.0, boundary="vacuum")
    cell_a = yamc.Cell(
        region=sphere_a.below, material=_dummy_material("gap_a"), fill=_mesh()
    )
    cell_b = yamc.Cell(
        region=sphere_b.below,
        material=_dummy_material("gap_b"),
        fill=yamc.MeshGeometry(
            TWO_REGION_ARROW,
            {"fuel": _dummy_material("water"), "moderator": _dummy_material("oil")},
        ),
        translation=(10.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="distinct ids"):
        yamc.Geometry([cell_a, cell_b])


def test_two_fills_sharing_materials_are_allowed():
    fuel = _dummy_material("fuel")
    moderator = _dummy_material("moderator")
    sphere_a = yamc.Sphere(x0=0.5, y0=0.5, z0=0.5, radius=3.0, boundary="vacuum")
    sphere_b = yamc.Sphere(x0=10.5, y0=0.5, z0=0.5, radius=3.0, boundary="vacuum")
    mesh = {"fuel": fuel, "moderator": moderator}
    cell_a = yamc.Cell(
        region=sphere_a.below,
        material=_dummy_material("gap"),
        fill=yamc.MeshGeometry(TWO_REGION_ARROW, mesh),
    )
    cell_b = yamc.Cell(
        region=sphere_b.below,
        fill=yamc.MeshGeometry(TWO_REGION_ARROW, mesh),
        translation=(10.0, 0.0, 0.0),
    )
    geometry = yamc.Geometry([cell_a, cell_b])
    assert geometry.find_cell(0.25, 0.5, 0.5).material.name == "fuel"
    assert geometry.find_cell(10.25, 0.5, 0.5).material.name == "fuel"


def test_filled_cell_calculate_volume_excludes_the_body():
    """The host material occupies region minus body: the cell-level
    stochastic volume must not count points inside the fill."""
    cell = _filled_cell()
    result = cell.calculate_volume(samples=200_000, seed=3)
    sphere_volume = 4.0 / 3.0 * 3.14159265358979 * 27.0
    assert result.volume == pytest.approx(sphere_volume - 1.0, rel=0.02)
    plain = yamc.Cell(
        region=yamc.Sphere(x0=0.5, y0=0.5, z0=0.5, radius=3.0).below
    )
    plain_result = plain.calculate_volume(samples=200_000, seed=3)
    assert plain_result.volume == pytest.approx(sphere_volume, rel=0.02)


def test_woodcock_tracking_rejects_mesh_fills():
    geometry = yamc.Geometry([_filled_cell()])
    source = yamc.NeutronSource(
        position=[0.5, 0.5, 2.0], energy=yamc.sources.Discrete([14.06e6], [1.0])
    )
    for mode in ("woodcock", "hybrid"):
        model = yamc.Model(geometry=geometry, source=source, tracking_mode=mode)
        with pytest.raises(ValueError, match="mesh-filled"):
            model.simulate_transport(total_particles=100, seed=1)


def test_model_saves_fingerprint_but_does_not_load():
    geometry = yamc.Geometry([_filled_cell()])
    source = yamc.NeutronSource(
        position=[0.5, 0.5, 2.0], energy=yamc.sources.Discrete([14.06e6], [1.0])
    )
    model = yamc.Model(geometry=geometry, source=source)
    text = model.to_json()
    assert "fills" in text
    with pytest.raises(ValueError, match="cannot be deserialized"):
        yamc.Model.load(text)
