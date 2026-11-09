"""A source spread over the elements of a mesh, by per-element strength."""

import math
import os
import re

import numpy as np
import pytest
import yamc

MeshDistribution = yamc.sources.MeshDistribution


def _positions(space, n=50_000):
    positions, _ = yamc.NeutronSource(position=space).sample_n(n)
    return np.array(positions)


def test_rectangular_strengths_follow_tally_bin_order():
    """The voxel given all the strength is where a mesh tally on the same mesh
    scores most, so strengths and tally bins share an order."""
    mesh = yamc.RegularRectangularMesh([-3, -2, -1], [3, 2, 1], [3, 2, 1])
    for hot in range(6):
        strengths = [0.0] * 6
        strengths[hot] = 1.0
        sphere = yamc.Sphere(radius=10, boundary="vacuum")
        tally = yamc.Tally(scores=["flux"], mesh=mesh)
        model = yamc.Model(
            geometry=yamc.Geometry([yamc.Cell(region=sphere.below)]),
            source=yamc.NeutronSource(position=MeshDistribution(mesh, strengths)),
            tallies=[tally],
        )
        mean = model.simulate_transport(total_particles=2_000, seed=1)[tally].mean
        assert int(np.argmax(mean)) == hot


def test_rectangular_samples_stay_in_their_voxels():
    mesh = yamc.RegularRectangularMesh([0, 0, 0], [2, 1, 1], [2, 1, 1])
    p = _positions(MeshDistribution(mesh, [3.0, 1.0]))
    assert p.min() >= 0 and p[:, 0].max() <= 2 and p[:, 1:].max() <= 1
    assert abs((p[:, 0] < 1).mean() - 0.75) < 0.01


def test_cylindrical_default_is_uniform_in_volume():
    mesh = yamc.RegularCylindricalMesh(
        r_bounds=(0, 2), z_bounds=(0, 1), shape=(2, 1, 1), origin=[1, 2, 3]
    )
    space = MeshDistribution(mesh)
    p = _positions(space)
    r = np.hypot(p[:, 0] - 1, p[:, 1] - 2)
    assert r.max() <= 2 and p[:, 2].min() >= 3 and p[:, 2].max() <= 4
    # The outer ring holds three quarters of the volume.
    assert abs((r > 1).mean() - 0.75) < 0.01
    assert math.isclose(space.total_strength, 4 * math.pi)
    assert space.num_elements == 2


def test_cylindrical_strengths_pick_sectors():
    mesh = yamc.RegularCylindricalMesh(
        r_bounds=(1, 2), z_bounds=(0, 1), shape=(1, 4, 1)
    )
    p = _positions(MeshDistribution(mesh, [0, 0, 1, 0]))
    phi = np.arctan2(p[:, 1], p[:, 0]) % (2 * math.pi)
    assert phi.min() >= math.pi and phi.max() <= 1.5 * math.pi


def test_tet_mesh(tmp_path):
    cq = pytest.importorskip("cadquery")
    from yamc.cad import CadToYamc

    c2y = CadToYamc()
    assembly = cq.Assembly()
    assembly.add(cq.Workplane().box(4, 4, 4), name="box")
    c2y.add_cadquery_object(assembly, material_tags=["box"])
    c2y.mesh(tet_volumes=["box"], target_edge_length=2.0)
    path = os.path.join(tmp_path, "box.arrow")
    c2y.to_arrow(path)
    geometry = yamc.MeshGeometry(path, {}, graveyard_offset=1.0)

    space = MeshDistribution(geometry)
    assert space.num_elements > 1
    assert math.isclose(space.total_strength, 64, rel_tol=1e-6)
    p = _positions(space)
    assert np.abs(p).max() <= 2 + 1e-9

    strengths = [0.0] * space.num_elements
    strengths[-1] = 2.0
    one = MeshDistribution(geometry, strengths)
    assert one.total_strength == 2.0
    assert "1 of" in repr(one)


@pytest.mark.parametrize(
    "strengths, message",
    [
        ([1.0], "has 1 values but the mesh has 2"),
        ([1.0, -1.0], "strengths[1] is -1"),
        ([math.nan, 1.0], "strengths[0] is NaN"),
        ([0.0, 0.0], "never emits"),
    ],
)
def test_bad_strengths_are_refused(strengths, message):
    mesh = yamc.RegularRectangularMesh([0, 0, 0], [2, 1, 1], [2, 1, 1])
    with pytest.raises(ValueError, match=re.escape(message)):
        MeshDistribution(mesh, strengths)


def test_not_a_mesh_is_refused():
    with pytest.raises(TypeError, match="mesh must be"):
        MeshDistribution((0, 0, 0))


def test_source_hands_back_the_distribution():
    mesh = yamc.RegularRectangularMesh([0, 0, 0], [1, 1, 1], [1, 1, 1])
    source = yamc.PhotonSource(position=MeshDistribution(mesh), energy=1e6)
    assert isinstance(source.position, MeshDistribution)
    assert "MeshDistribution(1 of 1 elements emitting)" in repr(source)


def _iron_ball():
    """An iron ball of radius 1 in void, inside a two-voxel mesh whose second
    voxel holds only void."""
    iron = yamc.Material(composition={"Fe56": 1.0}, density=7.874, temperature=294)
    iron.read_nuclear_data({"Fe56": "tests/Fe56.arrow"})
    ball, world = yamc.Sphere(radius=1.0), yamc.Sphere(radius=10.0, boundary="vacuum")
    inner = yamc.Cell(region=ball.below, material=iron)
    geometry = yamc.Geometry([inner, yamc.Cell(region=ball.above & world.below)])
    mesh = yamc.RegularRectangularMesh([-2, -2, -2], [6, 2, 2], [2, 1, 1])
    return geometry, inner, iron, mesh


def _births(geometry, space, n=400):
    model = yamc.Model(geometry=geometry, source=yamc.NeutronSource(position=space))
    results = model.simulate_transport(total_particles=n, seed=3, capture_tracks=n)
    births = [
        (r["x"], r["y"], r["z"])
        for r in results.tracks.to_dataframe_records()
        if r["event_type"] == "source_birth"
    ]
    assert len(births) == n
    return np.linalg.norm(births, axis=1)


@pytest.mark.parametrize("by", ["cells", "materials"])
def test_births_limited_to_cells_or_materials_in_the_voxel(by):
    geometry, inner, iron, mesh = _iron_ball()
    domain = {"cells": inner, "materials": [iron]}[by]
    space = MeshDistribution(mesh, [1.0, 0.0], **{by: domain})
    assert _births(geometry, space).max() < 1.0
    # Unlimited, most of the voxel is void.
    assert _births(geometry, MeshDistribution(mesh, [1.0, 0.0])).max() > 1.0


def test_an_element_holding_none_of_its_cells_is_refused():
    geometry, inner, _, mesh = _iron_ball()
    with pytest.raises(ValueError, match="mesh element 1 has a strength but no point in its cells"):
        _births(geometry, MeshDistribution(mesh, [1.0, 1.0], cells=inner))


def test_a_cell_not_in_the_geometry_is_refused():
    geometry, _, _, mesh = _iron_ball()
    stray = yamc.Cell(id=99, region=yamc.Sphere(radius=0.5).below)
    with pytest.raises(ValueError, match="cell id 99, which no cell"):
        _births(geometry, MeshDistribution(mesh, [1.0, 0.0], cells=stray))


def test_cells_and_materials_together_are_refused():
    _, inner, iron, mesh = _iron_ball()
    with pytest.raises(ValueError, match="not both"):
        MeshDistribution(mesh, cells=inner, materials=iron)
