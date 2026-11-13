"""A mesh solid cut into parts by internal faces (issues #517 and #533).

An internal face is a surface with the same volume on both sides (``[v, v]``
in ``surface_volumes``), as a mesh from another writer can have. The 30 cm
cube here is one volume in 27 cells. From its centre cell every test ray of
point location met an internal face first, so every particle born there was
lost, and a CSG cell refused the cube as a fill, as not watertight. Now
neither happens, and the flux is that of the same cube without the internal
faces.
"""
import os
import tempfile

import pytest
import yamc
from yamc._core import mesh_to_arrow

LI6 = "tests/Li6.arrow"
PARTICLES = 200
CENTRE = (15.0, 15.0, 15.0)


def _cube(cut):
    """Write [0, 30]^3 as one volume on a lattice of 10 cm cells: its outside
    is surface 1, wound outward, and with ``cut`` each lattice plane inside it
    is an internal face of its own. Returns the file's path."""
    n = 3

    def vertex(i):
        return (i[2] * (n + 1) + i[1]) * (n + 1) + i[0]

    vertices = [
        [10.0 * i, 10.0 * j, 10.0 * k]
        for k in range(n + 1)
        for j in range(n + 1)
        for i in range(n + 1)
    ]
    triangles, surface_ids, surface_volumes = [], [], ["[0, null]"]
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        for plane in range(n + 1):
            if plane in (0, n):
                surface = 1
            elif cut:
                surface_volumes.append("[0, 0]")
                surface = len(surface_volumes)
            else:
                continue
            for jb in range(n):
                for jc in range(n):
                    quad = []
                    for db, dc in [(0, 0), (1, 0), (1, 1), (0, 1)]:
                        i = [0, 0, 0]
                        i[a], i[b], i[c] = plane, jb + db, jc + dc
                        quad.append(vertex(i))
                    if plane == 0:  # listed facing +a
                        quad.reverse()
                    triangles += [quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]]
                    surface_ids += [surface, surface]
    path = os.path.join(tempfile.mkdtemp(prefix="yamc_internal_faces_"), "cube.arrow")
    mesh_to_arrow(
        path,
        [x for v in vertices for x in v],
        triangles,
        surface_ids,
        [1] * len(surface_ids),
        physical_groups_json='{"1": {"name": "mat:a", "dim": 3}}',
        surface_volumes_json="[" + ", ".join(surface_volumes) + "]",
        imprinted=True,
    )
    return path


def _material():
    material = yamc.Material(composition={"Li6": 1.0}, density=1e-9, name="a", temperature=294)
    material.read_nuclear_data({"Li6": LI6})
    return material


def _flux(geometry, material):
    """Lost histories and the flux in the cube, from a source in its centre."""
    tally = yamc.Tally(scores=["flux"], materials=material, name="flux")
    source = yamc.NeutronSource(position=CENTRE, energy=yamc.sources.Discrete([14.06e6], [1.0]))
    model = yamc.Model(geometry=geometry, tallies=[tally], source=source,
                       max_lost_particles=10 * PARTICLES)
    results = model.simulate_transport(total_particles=PARTICLES, seed=1)
    return len(model.lost_particles), results[tally].mean[0]


def test_particles_born_in_a_cell_enclosed_by_internal_faces_are_not_lost():
    material = _material()
    cut, control = (
        _flux(yamc.MeshGeometry(_cube(cut), {"a": material}), material)
        for cut in (True, False)
    )
    assert control[0] == 0
    assert control[1] > 0.0
    # The same histories, but threads sum the flux in any order.
    assert cut[0] == 0
    assert cut[1] == pytest.approx(control[1], rel=1e-12)


def test_a_cell_can_be_filled_by_a_mesh_with_internal_faces():
    material = _material()

    def filled(cut):
        sphere = yamc.Sphere(x0=15.0, y0=15.0, z0=15.0, radius=40.0, boundary="vacuum")
        mesh = yamc.MeshGeometry(_cube(cut), {"a": material})
        return yamc.Geometry([yamc.Cell(region=sphere.below, name="chamber", fill=mesh)])

    cut, control = (_flux(filled(cut), material) for cut in (True, False))
    assert control[0] == 0
    assert control[1] > 0.0
    # The same histories, but threads sum the flux in any order.
    assert cut[0] == 0
    assert cut[1] == pytest.approx(control[1], rel=1e-12)


def test_the_internal_faces_belong_to_the_volume_but_not_its_measure():
    mesh = yamc.MeshGeometry(_cube(True), {})
    assert mesh.volume_triangle_indices(0) == list(range(mesh.num_triangles))
    assert mesh.volume_measures == yamc.MeshGeometry(_cube(False), {}).volume_measures == [27000.0]
