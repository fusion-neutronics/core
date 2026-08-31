"""Unit tests for the Rust CAD export finalize kernels (issue #246).

cad_mesh_labels and cad_mesh_to_arrow replaced the per-element Python loops in
CadToYamc.to_vtkhdf/to_arrow. These tests pin the Rust kernels against direct
Python reference implementations of the replaced loops, on data that exercises
shared faces, unowned faces and solids without a material tag. No cadquery or
h5py needed.
"""

import json

import pytest

from yamc._core import cad_mesh_labels, cad_mesh_to_arrow, mesh_to_arrow


# Solid 1 owns faces 10 and 11; solid 2 owns 11 (shared) and 12; solid 3 owns
# 13 but has no material tag; face 99 is owned by nobody.
SOLID_FACES = {1: [10, 11], 2: [11, 12], 3: [13]}
MATERIAL_TAGS = ["steel", "water"]
TRIANGLE_FACE_IDS = [10, 11, 12, 13, 99]
TRIANGLE_SURFACE_IDS = [1, 2, 3, 4, 5]
FACE_TO_SURFACE_ID = {10: 1, 11: 2, 12: 3, 13: 4, 99: 5}
FACE_SOLID_REVERSED = {
    (1, 10): False,
    (1, 11): False,
    (2, 11): True,
    (2, 12): False,
    (3, 13): True,
}
VERTICES = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [2.0, 0.0, 0.0],
    [0.0, 2.0, 0.0],
]
TRIANGLES = [
    [0, 1, 2],
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
]
# (solid_id, vertex_offset, tets): local tet indices offset into VERTICES.
TET_BLOCKS = [
    (1, 0, [[0, 1, 2, 3]]),
    (2, 6, [[0, 1, 2, 3], [1, 2, 3, 0]]),
]


def reference_labels():
    """The exact loops cad_mesh_labels replaced in CadToYamc.to_vtkhdf."""
    volume_ids = [-1] * len(TRIANGLES)
    for i, fid in enumerate(TRIANGLE_FACE_IDS):
        for solid_id, face_ids in SOLID_FACES.items():
            if fid in face_ids:
                volume_ids[i] = solid_id
                break
    material_ids = [-1] * len(TRIANGLES)
    for i, fid in enumerate(TRIANGLE_FACE_IDS):
        for solid_id, face_ids in SOLID_FACES.items():
            if fid in face_ids:
                material_ids[i] = solid_id - 1
                break
    all_tets = []
    tet_volume_ids = []
    for solid_id, offset, tet_t in TET_BLOCKS:
        for t in tet_t:
            all_tets.append([t[0] + offset, t[1] + offset, t[2] + offset, t[3] + offset])
            tet_volume_ids.append(solid_id)
    return volume_ids, material_ids, all_tets, tet_volume_ids


def reference_arrow_arrays():
    """The exact loops cad_mesh_to_arrow replaced in CadToYamc.to_arrow."""
    physical_groups = {}
    material_pg_ids = {}
    for i, tag in enumerate(MATERIAL_TAGS):
        physical_groups[str(i + 1)] = {"name": f"mat:{tag}", "dim": 3}
        material_pg_ids[i + 1] = i + 1

    tri_physical_groups = []
    for fid in TRIANGLE_FACE_IDS:
        pg = -1
        for solid_id, face_ids in SOLID_FACES.items():
            if fid in face_ids:
                pg = material_pg_ids.get(solid_id, -1)
                break
        tri_physical_groups.append(pg)

    flat_tets = []
    tet_volume_ids = []
    tet_physical_groups = []
    for solid_id, offset, tet_t in TET_BLOCKS:
        for t in tet_t:
            flat_tets.extend([v + offset for v in t])
            tet_volume_ids.append(solid_id)
            tet_physical_groups.append(material_pg_ids.get(solid_id, -1))

    num_surfaces = max(FACE_TO_SURFACE_ID.values())
    surface_volumes = [[None, None] for _ in range(num_surfaces)]
    for fid, sid in FACE_TO_SURFACE_ID.items():
        for solid_id, face_ids in SOLID_FACES.items():
            if fid not in face_ids:
                continue
            slot = 1 if FACE_SOLID_REVERSED.get((solid_id, fid), False) else 0
            surface_volumes[sid - 1][slot] = solid_id - 1

    return (
        tri_physical_groups,
        flat_tets,
        tet_volume_ids,
        tet_physical_groups,
        physical_groups,
        surface_volumes,
    )


def test_cad_mesh_labels_matches_reference():
    got = cad_mesh_labels(
        TRIANGLE_FACE_IDS, list(SOLID_FACES.items()), TET_BLOCKS
    )
    volume_ids, material_ids, all_tets, tet_volume_ids = reference_labels()
    assert list(got[0]) == volume_ids
    assert list(got[1]) == material_ids
    assert [list(t) for t in got[2]] == all_tets
    assert list(got[3]) == tet_volume_ids


def test_cad_mesh_labels_no_tets():
    volume_ids, material_ids, tets, tet_volume_ids = cad_mesh_labels(
        TRIANGLE_FACE_IDS, list(SOLID_FACES.items())
    )
    assert list(volume_ids) == [1, 1, 2, 3, -1]
    assert list(material_ids) == [0, 0, 1, 2, -1]
    assert tets == []
    assert tet_volume_ids == []


def test_cad_mesh_to_arrow_matches_lowlevel_writer(tmp_path):
    """cad_mesh_to_arrow output must equal mesh_to_arrow fed the reference
    arrays computed by the replaced Python loops."""
    pa = pytest.importorskip("pyarrow")

    (
        tri_physical_groups,
        flat_tets,
        tet_volume_ids,
        tet_physical_groups,
        physical_groups,
        surface_volumes,
    ) = reference_arrow_arrays()

    ref_path = tmp_path / "reference.arrow"
    mesh_to_arrow(
        path=str(ref_path),
        vertices=[c for v in VERTICES for c in v],
        triangles=[i for t in TRIANGLES for i in t],
        triangle_surface_ids=TRIANGLE_SURFACE_IDS,
        triangle_physical_groups=tri_physical_groups,
        tetrahedra=flat_tets,
        tet_volume_ids=tet_volume_ids,
        tet_physical_groups=tet_physical_groups,
        physical_groups_json=json.dumps(physical_groups),
        surface_volumes_json=json.dumps(surface_volumes),
    )

    new_path = tmp_path / "finalized.arrow"
    cad_mesh_to_arrow(
        path=str(new_path),
        vertices=VERTICES,
        triangles=TRIANGLES,
        triangle_surface_ids=TRIANGLE_SURFACE_IDS,
        triangle_face_ids=TRIANGLE_FACE_IDS,
        solid_faces=list(SOLID_FACES.items()),
        material_tags=MATERIAL_TAGS,
        face_to_surface_id=list(FACE_TO_SURFACE_ID.items()),
        face_solid_reversed=FACE_SOLID_REVERSED,
        tet_blocks=TET_BLOCKS,
    )

    with pa.ipc.open_file(str(ref_path)) as r:
        ref_batches = [r.get_batch(i) for i in range(r.num_record_batches)]
        ref_meta = r.schema.metadata
    with pa.ipc.open_file(str(new_path)) as r:
        new_batches = [r.get_batch(i) for i in range(r.num_record_batches)]
        new_meta = r.schema.metadata

    assert len(new_batches) == len(ref_batches)
    for ref_b, new_b in zip(ref_batches, new_batches):
        assert new_b.equals(ref_b)

    # JSON metadata must agree as parsed objects (key order and whitespace may
    # differ between json.dumps and serde_json).
    assert set(new_meta) == set(ref_meta)
    for key in ref_meta:
        if key in (b"yamc.physical_groups", b"yamc.surface_volumes"):
            assert json.loads(new_meta[key]) == json.loads(ref_meta[key])
        else:
            assert new_meta[key] == ref_meta[key]
