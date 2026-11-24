"""A point source sitting exactly on a surface.

A particle born on a surface belongs to the cell it is heading into, as in
OpenMC. Its position alone cannot say which: a CSG point on a surface is in
neither half-space, so the particle was lost at birth, and a mesh point on a
facet was put on one side of it whatever the particle's direction, so the
particles heading the other way carried the wrong material until their next
crossing, with no particle lost.
"""

import os

import pytest
import yamc


def _materials():
    out = {}
    for tag, (nuclide, density) in {"light": ("Be9", 1.85), "heavy": ("Pb208", 11.35)}.items():
        material = yamc.Material(
            composition={nuclide: 1.0}, density=density, name=tag, temperature=294
        )
        material.read_nuclear_data({nuclide: f"tests/{nuclide}.arrow"})
        out[tag] = material
    return out


def _fluxes(geometry, materials, x):
    tallies = [yamc.Tally(scores=["flux"], materials=m, name=t) for t, m in materials.items()]
    model = yamc.Model(
        geometry=geometry,
        tallies=tallies,
        source=yamc.NeutronSource(
            energy=yamc.sources.Discrete([14.06e6], [1.0]), position=(x, 0.3, 0.1)
        ),
    )
    results = model.simulate_transport(total_particles=20_000, seed=7)
    return {
        t.name: (results[t.name].mean[0], results[t.name].standard_deviation[0])
        for t in tallies
    }


def _assert_agree(on, beside):
    """Within 4 sigma: a source a hair off the surface draws one more
    distance to collision for the particles heading across it, so the
    histories are not in lockstep."""
    for tag, (a, sa) in on.items():
        b, sb = beside[tag]
        assert abs(a - b) < 4.0 * (sa * sa + sb * sb) ** 0.5, (tag, a, b)


def _csg(materials):
    sphere = yamc.Sphere(radius=10, boundary="vacuum")
    plane = yamc.Plane("x", 0.0)
    return yamc.Geometry(
        [
            yamc.Cell(name="light", region=sphere.below & plane.below, material=materials["light"]),
            yamc.Cell(name="heavy", region=sphere.below & plane.above, material=materials["heavy"]),
        ]
    )


def test_a_source_on_a_csg_surface_starts_where_each_particle_heads():
    # A source on the plane runs (it lost every particle) and agrees with
    # sources a hair either side of it.
    def fluxes(x):
        materials = _materials()
        return _fluxes(_csg(materials), materials, x)

    on = fluxes(0.0)
    for x in (1e-7, -1e-7):
        _assert_agree(on, fluxes(x))


def test_a_source_on_a_shared_mesh_face_starts_where_each_particle_heads(tmp_path):
    cq = pytest.importorskip("cadquery")
    from yamc.cad.mesher import _CadMesher

    assembly = cq.Assembly()
    assembly.add(cq.Workplane().box(10, 10, 10).translate((-5, 0, 0)), name="light")
    assembly.add(cq.Workplane().box(10, 10, 10).translate((5, 0, 0)), name="heavy")
    c2y = _CadMesher()
    c2y.add_cadquery_object(assembly, material_tags=["light", "heavy"])
    c2y.mesh()
    path = os.path.join(tmp_path, "boxes.arrow")
    c2y.to_arrow(path)

    def fluxes(x):
        materials = _materials()
        geometry = yamc.MeshGeometry.from_arrow(path, materials, graveyard_offset=1.0)
        return _fluxes(geometry, materials, x)

    on = fluxes(0.0)
    for x in (1e-7, -1e-7):
        _assert_agree(on, fluxes(x))


# Far from the origin the spacing of doubles is coarse: at z = 2200 cm a
# direction within 1e-3 of the plane moves z by less than half of it in a
# 1e-10 cm step, so a fixed step put those particles back on the face, lost
# on CSG and in the cell behind them on a mesh. Void cells, so every history
# is one flight from the face to the outside.
_FACE = 2200.0


def _lost_from_the_face(geometry):
    model = yamc.Model(
        geometry=geometry,
        source=yamc.NeutronSource(
            energy=yamc.sources.Discrete([14.06e6], [1.0]), position=(500.0, 500.0, _FACE)
        ),
        max_lost_particles=10**6,
    )
    model.simulate_transport(total_particles=100_000, seed=1)
    return len(model.lost_particles)


def test_a_source_on_a_far_csg_plane_loses_no_grazing_particle():
    walls = [
        yamc.Plane("x", 0.0, boundary="vacuum").above,
        yamc.Plane("x", 1000.0, boundary="vacuum").below,
        yamc.Plane("y", 0.0, boundary="vacuum").above,
        yamc.Plane("y", 1000.0, boundary="vacuum").below,
    ]
    side = walls[0] & walls[1] & walls[2] & walls[3]
    face = yamc.Plane("z", _FACE)
    geometry = yamc.Geometry(
        [
            yamc.Cell(
                name="lower",
                region=side & yamc.Plane("z", 0.0, boundary="vacuum").above & face.below,
            ),
            yamc.Cell(
                name="upper",
                region=side & face.above & yamc.Plane("z", 2 * _FACE, boundary="vacuum").below,
            ),
        ]
    )
    assert _lost_from_the_face(geometry) == 0


def test_a_source_on_a_far_shared_mesh_face_loses_no_grazing_particle(tmp_path):
    # Two 1000 x 1000 x 2200 cm blocks stacked along z, sharing z = _FACE.
    vertices, triangles, surfaces = [], [], []
    for z in (0.0, _FACE, 2 * _FACE):
        vertices += [[0, 0, z], [1000, 0, z], [1000, 1000, z], [0, 1000, z]]

    def quad(a, b, c, d, surface):  # wound so the normal is (b - a) x (c - a)
        triangles.extend([[a, b, c], [a, c, d]])
        surfaces.extend([surface, surface])

    quad(0, 3, 2, 1, 1)
    for i in range(4):
        j = (i + 1) % 4
        quad(i, j, j + 4, i + 4, 1)
    quad(4, 5, 6, 7, 2)
    for i in range(4):
        j = (i + 1) % 4
        quad(i + 4, j + 4, j + 8, i + 8, 3)
    quad(8, 9, 10, 11, 3)
    path = os.path.join(tmp_path, "blocks.arrow")
    yamc.mesh_to_arrow(
        path,
        [x for p in vertices for x in p],
        [i for f in triangles for i in f],
        surfaces,
        [{1: 1, 2: 1, 3: 2}[s] for s in surfaces],
        physical_groups_json=(
            '{"1": {"name": "mat:lower", "dim": 3}, "2": {"name": "mat:upper", "dim": 3}}'
        ),
        surface_volumes_json="[[0, null], [0, 1], [1, null]]",
        imprinted=True,
    )
    assert _lost_from_the_face(yamc.MeshGeometry.from_arrow(path, {})) == 0
