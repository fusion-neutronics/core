"""MeshGeometry.to_vtkhdf and MeshGeometry.exterior_surface_ids, which match
_CadMesher's, and boundary tags given to MeshGeometry.from_cadquery."""

import pytest

import yamc

cq = pytest.importorskip("cadquery")
from yamc.cad.mesher import _CadMesher  # noqa: E402


def _assembly():
    sphere = cq.Workplane("XY").sphere(3.0)
    box = cq.Workplane("XY").box(10, 10, 10).cut(sphere)
    assembly = cq.Assembly()
    assembly.add(sphere, name="sphere")
    assembly.add(box, name="box")
    return assembly


MESH = dict(tet_volumes=["sphere", "box"], target_edge_length=1.5, tolerance=0.2)


def _iron():
    return yamc.Material(composition={"Fe56": 1.0}, density=7.8)


def _both(**kwargs):
    """The same mesh from _CadMesher and from MeshGeometry.from_cadquery."""
    mesher = _CadMesher()
    mesher.add_cadquery_object(_assembly(), material_tags="assembly_names")
    mesher.mesh(**MESH)
    iron = _iron()
    geometry = yamc.MeshGeometry.from_cadquery(
        _assembly(), {"sphere": iron, "box": iron}, **MESH, **kwargs
    )
    return mesher, geometry


def test_exterior_surface_ids_match_cad_to_yamc():
    mesher, geometry = _both()
    assert geometry.exterior_surface_ids() == mesher.exterior_surface_ids()
    assert geometry.exterior_surface_ids()


def test_the_graveyard_is_not_exterior():
    mesher, geometry = _both(graveyard_offset=5.0)
    assert geometry.exterior_surface_ids() == mesher.exterior_surface_ids()


def test_a_mesh_made_without_imprinting_has_no_exterior():
    iron = _iron()
    geometry = yamc.MeshGeometry.from_cadquery(
        _assembly(), {"sphere": iron, "box": iron}, imprint=False, tolerance=0.2
    )
    with pytest.raises(ValueError, match="discontinuously"):
        geometry.exterior_surface_ids()


def test_to_vtkhdf_matches_cad_to_yamc(tmp_path):
    h5py = pytest.importorskip("h5py")
    mesher, geometry = _both()
    mesher.to_vtkhdf(tmp_path / "mesher.vtkhdf")
    geometry.to_vtkhdf(tmp_path / "geometry.vtkhdf")
    with h5py.File(tmp_path / "mesher.vtkhdf") as a, h5py.File(tmp_path / "geometry.vtkhdf") as b:
        for name in ("Points", "Connectivity", "Offsets", "Types"):
            assert (a["VTKHDF"][name][()] == b["VTKHDF"][name][()]).all(), name
        assert (
            a["VTKHDF/CellData/volume_id"][()] == b["VTKHDF/CellData/volume_id"][()]
        ).all()
        n_cells = len(geometry.triangles) + len(geometry.tetrahedra)
        assert len(b["VTKHDF/CellData/material_id"][()]) == n_cells


def test_exterior_vacuum_boundary_tag():
    _, untagged = _both()
    _, tagged = _both(boundary_tags={"vacuum": "exterior"})
    assert b"boundary:vacuum" in tagged.to_arrow()
    assert b"boundary:vacuum" not in untagged.to_arrow()
    assert tagged.tetrahedra == untagged.tetrahedra
