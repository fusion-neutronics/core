"""Mesh checks for solids that touch or overlap without sharing a surface.

Tracking on a mesh crosses from one solid into the next only through a
surface the two share. Two touching cubes that each carry their own closed
surface meet along two surfaces, so a particle leaving one is read into the
implicit complement and never enters the other; where the two facetings
cross, the particle can be lost. Loading such a mesh warns. One whose file
says it was imprinted is tracked discontinuously, which places the particle
in the solid it enters.

``solid_overlaps`` is the mesh-level overlap check ``CadToYamc.mesh`` runs
before export; it is tested here directly, without cadquery.
"""
import warnings

import pytest
import yamc
from yamc._core import mesh_to_arrow, solid_overlaps


def _box(x0, x1, y0, y1, z0, z1):
    """(vertices, triangles) for an axis-aligned box, outward winding."""
    v = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (0, 4, 7, 3), (1, 2, 6, 5)]
    t = []
    for a, b, c, d in quads:
        t += [[a, b, c], [a, c, d]]
    return v, t


GROUPS = '{"1": {"name": "mat:a", "dim": 3}, "2": {"name": "mat:b", "dim": 3}}'


def _write_two_cubes(path, x_b, imprinted=None):
    """Two 10 cm cubes, the second starting at x = *x_b*, each with its own
    closed surface; *imprinted* is what the file says of them."""
    va, ta = _box(0, 10, 0, 10, 0, 10)
    vb, tb = _box(x_b, x_b + 10, 0, 10, 0, 10)
    vertices = [c for v in va + vb for c in v]
    triangles = [i for t in ta for i in t] + [i + 8 for t in tb for i in t]
    n = len(ta)
    mesh_to_arrow(
        path,
        vertices,
        triangles,
        [1] * n + [2] * n,
        [1] * n + [2] * n,
        physical_groups_json=GROUPS,
        surface_volumes_json="[[0, null], [1, null]]",
        imprinted=imprinted,
    )


def _write_shared_interface(path):
    """The same two touching cubes, imprinted: the face at x = 10 is one
    surface with cube a in front of it and cube b behind it."""
    va, ta = _box(0, 10, 0, 10, 0, 10)
    vb, tb = _box(10, 20, 0, 10, 0, 10)
    vertices = [c for v in va + vb for c in v]
    # Faces come in pairs of triangles, the +x face last and the -x face
    # second to last.
    a_own, interface = ta[:10], ta[10:]
    b_own = tb[:8] + tb[10:]
    triangles = (
        [i for t in a_own for i in t]
        + [i + 8 for t in b_own for i in t]
        + [i for t in interface for i in t]
    )
    surfaces = [1] * 10 + [2] * 10 + [3] * 2
    groups = [1] * 10 + [2] * 10 + [1] * 2
    mesh_to_arrow(
        path,
        vertices,
        triangles,
        surfaces,
        groups,
        physical_groups_json=GROUPS,
        surface_volumes_json="[[0, null], [1, null], [0, 1]]",
    )


def test_touching_solids_without_a_shared_surface_warn(tmp_path):
    path = str(tmp_path / "touching.arrow")
    _write_two_cubes(path, 10.0)
    with pytest.warns(UserWarning, match="touch without sharing a surface"):
        yamc.MeshGeometry(path, {}, graveyard_offset=1.0)


def test_slightly_overlapping_solids_without_a_shared_surface_warn(tmp_path):
    """Two facetings of one CAD surface can cross by a sliver."""
    path = str(tmp_path / "sliver.arrow")
    _write_two_cubes(path, 9.99)
    with pytest.warns(UserWarning, match="touch without sharing a surface"):
        yamc.MeshGeometry(path, {}, graveyard_offset=1.0)


def _track_length_in_b(path, contact_tolerance=None):
    """Lost histories and the track length in cube b per history, for a
    beam along +x from the middle of cube a, through near-void Li6."""
    a, b = (
        yamc.Material(composition={"Li6": 1.0}, density=1e-9, name=n, temperature=294)
        for n in "ab"
    )
    for m in (a, b):
        m.read_nuclear_data({"Li6": "tests/Li6.arrow"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geometry = yamc.MeshGeometry(
            path, {"a": a, "b": b}, graveyard_offset=1.0, contact_tolerance=contact_tolerance
        )
    tally = yamc.Tally(scores=["flux"], materials=b, name="b")
    source = yamc.NeutronSource(
        position=(5.0, 5.0, 5.0),
        energy=14.06e6,
        direction=yamc.sources.Monodirectional([1.0, 0.0, 0.0]),
    )
    model = yamc.Model(geometry=geometry, tallies=[tally], source=source,
                       max_lost_particles=1000)
    results = model.simulate_transport(total_particles=100, seed=1)
    return len(model.lost_particles), results[tally].mean[0]


@pytest.mark.parametrize("x_b", [10.0, 9.99])
def test_a_particle_enters_a_solid_the_imprint_did_not_join(tmp_path, x_b):
    """Touching, or overlapping by a sliver as two facetings of one CAD
    surface can: the beam runs on through cube b from where it leaves a."""
    path = str(tmp_path / "cubes.arrow")
    _write_two_cubes(path, x_b, imprinted=True)
    with pytest.warns(UserWarning, match="tracked discontinuously"):
        geometry = yamc.MeshGeometry(path, {}, graveyard_offset=1.0)
    assert geometry.discontinuous
    lost, in_b = _track_length_in_b(path)
    assert lost == 0
    assert in_b == pytest.approx(x_b, abs=1e-6)


def test_a_file_that_does_not_say_how_it_was_made_is_only_warned_about(tmp_path):
    """It could hold a real overlap, which discontinuous tracking would hide."""
    path = str(tmp_path / "cubes.arrow")
    _write_two_cubes(path, 9.99)
    with pytest.warns(UserWarning, match="does not say it was made that way"):
        geometry = yamc.MeshGeometry(path, {}, graveyard_offset=1.0)
    assert not geometry.discontinuous
    assert geometry.contact_tolerance is None


def test_a_contact_tolerance_given_is_kept(tmp_path):
    path = str(tmp_path / "cubes.arrow")
    _write_two_cubes(path, 9.99, imprinted=True)
    geometry = yamc.MeshGeometry(path, {}, graveyard_offset=1.0, contact_tolerance=0.5)
    assert geometry.contact_tolerance == 0.5
    assert _track_length_in_b(path, contact_tolerance=0.5) == (0, pytest.approx(9.99, abs=1e-6))


@pytest.mark.parametrize("writer", ["apart", "shared"])
def test_solids_apart_or_sharing_their_interface_do_not_warn(tmp_path, writer):
    path = str(tmp_path / f"{writer}.arrow")
    if writer == "apart":
        _write_two_cubes(path, 30.0)
    else:
        _write_shared_interface(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        geometry = yamc.MeshGeometry(path, {}, graveyard_offset=1.0)
    assert geometry.num_volumes == 2


def _two_boxes(x_b):
    va, ta = _box(0, 10, 0, 10, 0, 10)
    vb, tb = _box(x_b, x_b + 10, 0, 10, 0, 10)
    return va + vb, [(1, ta), (2, [[i + 8 for i in t] for t in tb])]


def test_solid_overlaps_finds_an_overlap_and_its_depth():
    vertices, solids = _two_boxes(7.0)
    found = solid_overlaps(vertices, solids, 0.1)
    assert len(found) == 1
    a, b, depth, exposure = found[0]
    assert (a, b) == (1, 2)
    assert depth == pytest.approx(3.0, rel=0.05)
    assert exposure > 0.0


def test_solid_overlaps_finds_a_solid_inside_another():
    va, ta = _box(0, 30, 0, 30, 0, 30)
    vb, tb = _box(10, 20, 10, 20, 10, 20)
    found = solid_overlaps(va + vb, [(1, ta), (2, [[i + 8 for i in t] for t in tb])], 0.1)
    assert [(a, b) for a, b, _, _ in found] == [(1, 2)]


@pytest.mark.parametrize("x_b", [10.0, 10.05, 30.0])
def test_solid_overlaps_ignores_contact_and_separation(x_b):
    vertices, solids = _two_boxes(x_b)
    assert solid_overlaps(vertices, solids, 0.1) == []


def test_solid_overlaps_counts_contact_within_the_tolerance():
    """Interfering by 0.05 is contact at a 0.1 tolerance, overlap at 0.01."""
    vertices, solids = _two_boxes(9.95)
    assert solid_overlaps(vertices, solids, 0.1) == []
    assert len(solid_overlaps(vertices, solids, 0.01)) == 1
