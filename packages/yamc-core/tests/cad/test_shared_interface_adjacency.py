"""Tets must link across an imprinted interface between two touching solids.

``build_tet_adjacency`` pairs tet faces by their sorted vertex INDICES, so two
solids meeting at a shared face only link if the interface vertices carry the
same indices in both tet blocks. When each solid's block was appended whole,
the interface got two coincident-but-distinct copies of every vertex and the
tets either side never paired: all 256 interface faces of two touching cuboids
were claimed by one tet each, so the exported mesh had a hole along the whole
seam -- 100% of it, not a marginal fraction.

No answer moves because of it, and these tests do not pretend otherwise.
Transport tracks on the surface representation, and the tet mesh is a tally
overlay walked one volume at a time: ``element_walk::walk_into`` stops at a tet
belonging to another volume whether or not the two are linked, so nothing
crosses the seam either way. A two-cuboid model matches its CSG twin
identically (z = 0.10) before and after. What is pinned here is that the mesh
being written out is conformal -- a face interior to the model that only one
tet claims is a hole, whatever today's transport makes of it, and the
per-volume guard that currently hides it is not something the exported file
carries to whoever reads it next.

The gap left by the existing tests is why this needs its own file:
``test_tet_mesh.py::test_tet_mesh_selective`` meshes two touching boxes but
only counts how many solids were tetrahedralised, and ``test_two_cubes_gap.py``
puts a 10 cm gap between its cubes, so no test had two solids actually sharing
a face.
"""

import pytest

cq = pytest.importorskip("cadquery")
from yamc.cad import CadToYamc  # noqa: E402

# Face i of a tet is the one opposite vertex i, matching
# yamt::mesh::topology::TET_FACE_VERTICES -- the table build_tet_adjacency
# itself walks.
TET_FACE_VERTICES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


def _two_touching_cuboids(width1=10.0, width2=4.0):
    """Two cuboids sharing the full face at ``y = width1 / 2``.

    The same geometry as ``model_benchmark_zoo.TwoTouchingCuboids``, built
    inline so the test needs no extra dependency. Both solids are planar, so
    the interface is meshed with no faceting error to muddy the count.
    """
    assembly = cq.Assembly(name="TwoTouchingCuboids")
    assembly.add(cq.Workplane().box(width1, width1, width1))
    assembly.add(
        cq.Workplane()
        .transformed(offset=(0, 0.5 * width1 + 0.5 * width2, 0))
        .box(width1, width2, width1)
    )
    return assembly


def _global_tets(c2y):
    """Tet connectivity in the global vertex numbering the Arrow export writes.

    Applies each block's offset the way ``flatten_tet_blocks`` does, rather
    than assuming it is zero, so the reading is right whatever the blocks carry.
    """
    return [
        [v + offset for v in tet]
        for _solid_id, offset, block in c2y._tet_blocks()
        for tet in block
    ]


def _face_use_counts(tets):
    """How many tets claim each face, keyed by its sorted vertex triple."""
    counts = {}
    for tet in tets:
        for a, b, c in TET_FACE_VERTICES:
            key = tuple(sorted((tet[a], tet[b], tet[c])))
            counts[key] = counts.get(key, 0) + 1
    return counts


def test_tets_link_across_a_shared_interface():
    """Every tet face on the shared plane is claimed by two tets, not one."""
    width1, width2 = 10.0, 4.0
    interface_y = 0.5 * width1

    c2y = CadToYamc()
    c2y.add_cadquery_object(
        _two_touching_cuboids(width1, width2), material_tags=["a", "b"]
    )
    c2y.mesh_surfaces()
    tet_data = c2y.mesh_volumes()
    assert len(tet_data) == 2, "both solids should be tetrahedralised"

    vertices = c2y._surface_mesh.vertices
    counts = _face_use_counts(_global_tets(c2y))

    on_interface = [
        face
        for face in counts
        if all(abs(vertices[v][1] - interface_y) < 1e-6 for v in face)
    ]
    assert on_interface, "expected tet faces lying on the shared plane"

    unshared = [face for face in on_interface if counts[face] != 2]
    assert not unshared, (
        f"{len(unshared)} of {len(on_interface)} faces on the shared interface are "
        "claimed by one tet only: the two solids' tets are not linked there, so "
        "the exported mesh has a hole along the seam"
    )


def test_interface_vertices_are_not_duplicated():
    """The shared plane carries one vertex per coordinate across both blocks.

    The direct cause, asserted separately from its consequence so a regression
    says which half broke. Only tet-block vertices are counted: the surface
    mesh has its own vertices on this plane and they are a separate entity,
    correctly not merged into the tet numbering.
    """
    width1 = 10.0
    interface_y = 0.5 * width1

    c2y = CadToYamc()
    c2y.add_cadquery_object(_two_touching_cuboids(width1), material_tags=["a", "b"])
    c2y.mesh_surfaces()
    c2y.mesh_volumes()

    vertices = c2y._surface_mesh.vertices
    referenced = {v for tet in _global_tets(c2y) for v in tet}
    on_interface = [
        v for v in referenced if abs(vertices[v][1] - interface_y) < 1e-6
    ]
    distinct = {
        (round(vertices[v][0], 9), round(vertices[v][2], 9)) for v in on_interface
    }

    assert len(on_interface) == len(distinct), (
        f"{len(on_interface)} tet vertices on the interface occupy only "
        f"{len(distinct)} distinct positions: the two blocks each hold their own "
        "copy, so faces built from them cannot pair by index"
    )


def test_separated_solids_keep_their_interfaces_open():
    """A gap between solids is not welded shut by the merge.

    The counterpart to the test above: the merge keys on the exact coordinate,
    so it must not pull together two solids that merely come close. Both cubes
    keep every outward face to themselves.
    """
    side, gap = 10.0, 0.5
    assembly = cq.Assembly()
    assembly.add(
        cq.Workplane().box(side, side, side).translate((-(gap + side) / 2, 0, 0)),
        name="left",
    )
    assembly.add(
        cq.Workplane().box(side, side, side).translate(((gap + side) / 2, 0, 0)),
        name="right",
    )

    c2y = CadToYamc()
    c2y.add_cadquery_object(assembly, material_tags=["a", "b"])
    c2y.mesh_surfaces()
    c2y.mesh_volumes()

    vertices = c2y._surface_mesh.vertices
    counts = _face_use_counts(_global_tets(c2y))

    for face, use in counts.items():
        assert use <= 2, "a face claimed by more than two tets is a broken mesh"

    # The facing walls sit at x = -gap/2 and x = +gap/2. Nothing should bridge them.
    for x in (-gap / 2, gap / 2):
        facing = [
            face
            for face in counts
            if all(abs(vertices[v][0] - x) < 1e-6 for v in face)
        ]
        shared = [face for face in facing if counts[face] == 2]
        assert not shared, (
            f"{len(shared)} faces on the wall at x={x} are shared between the two "
            "cubes: a gap has been welded shut"
        )
