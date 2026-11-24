"""MeshGeometry.from_tets: another code's tet mesh used as the geometry, its
surfaces built from the faces where the material changes."""

import itertools

import numpy as np
import pytest

import yamc

# A 2 x 1 x 1 box of unit cubes, each cut into six tets around its diagonal
# (a cut that is conformal between neighbouring cubes); x < 1 is "a".
NX, NY, NZ = 4, 2, 2  # cells along each axis, cell size 0.5
H = 0.5


def _box():
    index = {}
    vertices = []
    for i, j, k in itertools.product(range(NX + 1), range(NY + 1), range(NZ + 1)):
        index[i, j, k] = len(vertices)
        vertices.append([i * H, j * H, k * H])
    kuhn = [(0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)]
    tets, names = [], []
    for i, j, k in itertools.product(range(NX), range(NY), range(NZ)):
        corner = [index[i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1)] for c in range(8)]
        for t in kuhn:
            tets.append([corner[c] for c in t])
            names.append("a" if i < NX // 2 else "b")
    return vertices, tets, names


def _materials():
    fe = yamc.Material(composition={"Fe56": 1.0}, density=7.8, temperature=294)
    fe.read_nuclear_data({"Fe56": "tests/Fe56.arrow"})
    li = yamc.Material(composition={"Li6": 1.0}, density=0.5, temperature=294)
    li.read_nuclear_data({"Li6": "tests/Li6.arrow"})
    return {"a": fe, "b": li}


def test_volumes_and_tets():
    vertices, tets, names = _box()
    mats = _materials()
    geometry = yamc.MeshGeometry.from_tets(vertices, tets, names, mats, graveyard_offset=1.0)
    assert geometry.volume_names == ["a", "b"]
    assert geometry.volume_measures == pytest.approx([1.0, 1.0], rel=1e-12)
    assert geometry.tet_volume_names == names
    assert len(geometry.tetrahedra) == len(tets)
    assert sum(geometry.tet_volumes) == pytest.approx(2.0, rel=1e-12)
    # the outer surface, 10 cm2 in squares of 0.25 cm2, and the interface at
    # x = 1, 4 squares, two triangles each; then the graveyard box's 12
    assert geometry.num_triangles == 2 * (40 + 4) + 12
    assert geometry.exterior_surface_ids() and not geometry.discontinuous


def test_negative_tets_are_rewound():
    vertices, tets, names = _box()
    tets = [[t[1], t[0], t[2], t[3]] if i % 3 == 0 else t for i, t in enumerate(tets)]
    geometry = yamc.MeshGeometry.from_tets(vertices, tets, names, _materials())
    assert all(v > 0 for v in geometry.tet_volumes)
    assert geometry.volume_measures == pytest.approx([1.0, 1.0], rel=1e-12)


def test_transport_matches_csg():
    vertices, tets, names = _box()
    mats = _materials()
    geometry = yamc.MeshGeometry.from_tets(vertices, tets, names, mats, graveyard_offset=1.0)
    source = yamc.NeutronSource(position=(0.5, 0.5, 0.5), energy=2e6)
    mesh_tally = yamc.Tally(scores=["flux"], materials=[mats["a"], mats["b"]])
    tet_tally = yamc.Tally(scores=["flux"], mesh=(geometry, "a"))
    mesh_results = yamc.Model(
        geometry=geometry, source=source, tallies=[mesh_tally, tet_tally], verbose=[]
    ).simulate_transport(total_particles=20_000, seed=1)
    # a bin for every tet of the mesh, scoring in a's tets only
    per_tet = np.array(mesh_results[tet_tally].mean)
    assert len(per_tet) == len(tets)
    assert (per_tet[np.array(names) == "a"] > 0).all()
    assert (per_tet[np.array(names) == "b"] == 0).all()
    assert sum(mesh_results[tet_tally].mean) == pytest.approx(
        mesh_results[mesh_tally].mean[0], rel=1e-9
    )

    x0, x1, x2 = (yamc.Plane(axis="x", offset=x) for x in (0.0, 1.0, 2.0))
    y0, y1 = (yamc.Plane(axis="y", offset=y) for y in (0.0, 1.0))
    z0, z1 = (yamc.Plane(axis="z", offset=z) for z in (0.0, 1.0))
    side = y0.above & y1.below & z0.above & z1.below
    world = yamc.Sphere(radius=10.0, boundary="vacuum")
    cells = [
        yamc.Cell(region=x0.above & x1.below & side, material=mats["a"]),
        yamc.Cell(region=x1.above & x2.below & side, material=mats["b"]),
    ]
    box = x0.above & x2.below & side
    cells.append(yamc.Cell(region=world.below & ~box))
    csg_tally = yamc.Tally(scores=["flux"], materials=[mats["a"], mats["b"]])
    csg_results = yamc.Model(
        geometry=yamc.Geometry(cells), source=source, tallies=[csg_tally], verbose=[]
    ).simulate_transport(total_particles=20_000, seed=1)
    for got, want, sd in zip(
        mesh_results[mesh_tally].mean,
        csg_results[csg_tally].mean,
        csg_results[csg_tally].standard_deviation,
    ):
        assert got == pytest.approx(want, abs=5 * sd + 1e-12)


def test_round_trip():
    vertices, tets, names = _box()
    mats = _materials()
    geometry = yamc.MeshGeometry.from_tets(vertices, tets, names, mats)
    again = yamc.MeshGeometry.from_arrow(geometry.to_arrow(), mats)
    assert again.tetrahedra == geometry.tetrahedra
    assert again.tet_volume_names == names
    assert again.volume_measures == geometry.volume_measures


def test_bad_meshes_raise():
    vertices, tets, names = _box()
    mats = _materials()
    with pytest.raises(ValueError, match="manifold"):
        yamc.MeshGeometry.from_tets(vertices, tets + [tets[0]], names + ["a"], mats)
    with pytest.raises(ValueError, match="no volume"):
        yamc.MeshGeometry.from_tets(vertices, [[0, 1, 2, 3]], ["a"], mats)
    with pytest.raises(ValueError, match="one per tet"):
        yamc.MeshGeometry.from_tets(vertices, tets, names[:-1], mats)
    with pytest.raises(ValueError, match="vertex"):
        yamc.MeshGeometry.from_tets(vertices, [[0, 1, 2, 10_000]], ["a"], mats)


def test_a_coupled_field_goes_straight_on():
    vertices, tets, names = _box()
    mats = _materials()
    geometry = yamc.MeshGeometry.from_tets(vertices, tets, names, mats, graveyard_offset=1.0)
    centres = np.array(vertices)[np.array(tets)].mean(axis=1)
    factors = 1.0 - 0.2 * centres[:, 2]  # as a thermal code would give, per tet
    model = yamc.Model(
        geometry=geometry,
        source=yamc.NeutronSource(position=(0.5, 0.5, 0.5), energy=2e6),
        tallies=[yamc.Tally(scores=["flux"], mesh=(geometry, "a"))],
        tracking_mode="hybrid",
        verbose=[],
    )
    model.set_tet_density_factors(factors)
    model.simulate_transport(total_particles=2_000, seed=1)
