"""UnstructuredMesh: tally-only meshes apart from the transport geometry, and
a geometry's own tets as a tally mesh."""

import itertools

import numpy as np
import pytest

import yamc

NX, NY, NZ, H = 4, 2, 2, 0.5  # a 2 x 1 x 1 box of 96 tets, x < 1 is "a"


def _box(offset=(0.0, 0.0, 0.0)):
    index, vertices = {}, []
    for i, j, k in itertools.product(range(NX + 1), range(NY + 1), range(NZ + 1)):
        index[i, j, k] = len(vertices)
        vertices.append([i * H + offset[0], j * H + offset[1], k * H + offset[2]])
    kuhn = [(0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)]
    tets, regions = [], []
    for i, j, k in itertools.product(range(NX), range(NY), range(NZ)):
        corner = [index[i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1)] for c in range(8)]
        for t in kuhn:
            tets.append([corner[c] for c in t])
            regions.append("a" if i < NX // 2 else "b")
    return vertices, tets, regions


def _iron():
    fe = yamc.Material(composition={"Fe56": 1.0}, density=7.8, temperature=294)
    fe.read_nuclear_data({"Fe56": "tests/Fe56.arrow"})
    return fe


def _csg_box(iron):
    """The same 2 x 1 x 1 box as one CSG cell, in a vacuum sphere."""
    x0, x2 = yamc.Plane(axis="x", offset=0.0), yamc.Plane(axis="x", offset=2.0)
    y0, y1 = yamc.Plane(axis="y", offset=0.0), yamc.Plane(axis="y", offset=1.0)
    z0, z1 = yamc.Plane(axis="z", offset=0.0), yamc.Plane(axis="z", offset=1.0)
    box = x0.above & x2.below & y0.above & y1.below & z0.above & z1.below
    world = yamc.Sphere(radius=10.0, boundary="vacuum")
    cell = yamc.Cell(region=box, material=iron)
    return yamc.Geometry([cell, yamc.Cell(region=world.below & ~box)]), cell


def _run(geometry, tallies, n=20_000):
    model = yamc.Model(
        geometry=geometry,
        source=yamc.NeutronSource(position=(0.5, 0.5, 0.5), energy=2e6),
        tallies=tallies,
        verbose=[],
    )
    return model.simulate_transport(total_particles=n, seed=1)


def test_properties():
    vertices, tets, regions = _box()
    mesh = yamc.UnstructuredMesh(vertices, tets, regions)
    assert len(mesh) == mesh.num_cells == 96
    assert mesh.regions == ["a", "b"]
    assert mesh.cell_regions == regions
    assert mesh.cells == tets
    assert sum(mesh.cell_volumes) == pytest.approx(2.0, rel=1e-12)
    assert yamc.UnstructuredMesh(vertices, tets).regions == ["mesh"]
    with pytest.raises(ValueError, match="manifold"):
        yamc.UnstructuredMesh(vertices, tets + [tets[0]])


def test_an_overlay_sums_to_the_cell_flux():
    iron = _iron()
    geometry, cell = _csg_box(iron)
    vertices, tets, regions = _box()
    mesh = yamc.UnstructuredMesh(vertices, tets, regions)
    per_tet = yamc.Tally(scores=["flux"], mesh=mesh)
    in_a = yamc.Tally(scores=["flux"], mesh=(mesh, "a"))
    whole = yamc.Tally(scores=["flux"], cells=[cell])
    results = _run(geometry, [per_tet, in_a, whole])
    flux = np.array(results[per_tet].mean)
    assert len(flux) == 96
    # the tets tile the cell, so their track lengths add up to the cell's
    assert flux.sum() == pytest.approx(results[whole].mean[0], rel=1e-9)
    a = np.array(results[in_a].mean)
    mask = np.array(regions) == "a"
    assert (a[~mask] == 0).all()
    assert a[mask] == pytest.approx(flux[mask], rel=1e-12)
    with pytest.raises(ValueError, match="no region named"):
        yamc.Tally(scores=["flux"], mesh=(mesh, "c"))


def test_an_overlay_need_not_match_the_geometry():
    """Half of the tally mesh sticks out of the box into the void."""
    iron = _iron()
    geometry, cell = _csg_box(iron)
    vertices, tets, _ = _box(offset=(1.0, 0.0, 0.0))
    mesh = yamc.UnstructuredMesh(vertices, tets)
    tally = yamc.Tally(scores=["flux"], mesh=mesh)
    results = _run(geometry, [tally])
    flux = np.array(results[tally].mean)
    centres = np.array(vertices)[np.array(tets)].mean(axis=1)
    assert flux[centres[:, 0] < 2.0].sum() > 0
    assert flux[centres[:, 0] > 2.0].sum() > 0  # void tracks score too


def test_to_vtkhdf(tmp_path):
    h5py = pytest.importorskip("h5py")
    vertices, tets, regions = _box()
    mesh = yamc.UnstructuredMesh(vertices, tets, regions)
    mesh.to_vtkhdf(tmp_path / "mesh.vtkhdf")
    with h5py.File(tmp_path / "mesh.vtkhdf") as f:
        ids = f["VTKHDF/CellData/region_id"][()]
    assert list(ids) == [0 if r == "a" else 1 for r in regions]


def test_a_geometry_tallies_its_own_tets():
    iron = _iron()
    vertices, tets, regions = _box()
    geometry = yamc.MeshGeometry.from_tets(
        vertices, tets, regions, {"a": iron, "b": iron}, graveyard_offset=1.0
    )
    every = yamc.Tally(scores=["flux"], mesh=geometry)
    own = yamc.UnstructuredMesh(geometry.vertices, geometry.tetrahedra, geometry.tet_volume_names)
    via_mesh = yamc.Tally(scores=["flux"], mesh=own)
    in_a = yamc.Tally(scores=["flux"], mesh=(geometry, "a"))
    in_b = yamc.Tally(scores=["flux"], mesh=(geometry, "b"))
    results = _run(geometry, [every, via_mesh, in_a, in_b])
    flux = np.array(results[every].mean)
    assert len(flux) == len(tets)
    assert flux == pytest.approx(np.array(results[via_mesh].mean), rel=1e-12)
    assert flux == pytest.approx(
        np.array(results[in_a].mean) + np.array(results[in_b].mean), rel=1e-12
    )


def test_a_surface_only_geometry_has_no_unstructured_mesh(tmp_path):
    from yamc._core import mesh_to_arrow

    # one closed box, surfaces only
    v = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (0, 4, 7, 3), (1, 2, 6, 5)]
    tris = [t for a, b, c, d in quads for t in ([a, b, c], [a, c, d])]
    path = str(tmp_path / "box.arrow")
    mesh_to_arrow(
        path,
        [x for p in v for x in p],
        [i for t in tris for i in t],
        [1] * 12,
        [1] * 12,
        [],
        [],
        [],
        '{"1": {"name": "mat:a", "dim": 3}}',
        "[[0, null]]",
    )
    geometry = yamc.MeshGeometry.from_arrow(path, {"a": _iron()}, graveyard_offset=1.0)
    with pytest.raises(ValueError, match="no tets"):
        yamc.Tally(scores=["flux"], mesh=geometry)
