"""Tests for VTKHDF visualization export."""

import os

import pytest

cq = pytest.importorskip("cadquery")

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from yamc.cad import CadToYamc  # noqa: E402


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
def test_vtkhdf_surface_only(tmp_path):
    """Export surface mesh to VTKHDF and verify HDF5 structure."""
    box = cq.Workplane("XY").box(1, 1, 1)
    assy = cq.Assembly()
    assy.add(box, name="box")

    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["steel"])
    c2y.mesh_surfaces()

    path = str(tmp_path / "box.vtkhdf")
    c2y.to_vtkhdf(path)

    assert os.path.exists(path)

    with h5py.File(path, "r") as f:
        root = f["VTKHDF"]
        assert list(root.attrs["Version"]) == [2, 1]
        assert root.attrs["Type"] == b"UnstructuredGrid"
        assert root["NumberOfCells"][0] == 12  # 6 faces * 2 tris
        assert root["Points"].shape[1] == 3
        # All cell types should be VTK_TRIANGLE (5)
        types = root["Types"][:]
        assert all(t == 5 for t in types)
        # Cell data should exist
        assert "volume_id" in root["CellData"]


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
def test_vtkhdf_mixed_mesh(tmp_path):
    """Export mixed tri+tet mesh to VTKHDF."""
    box = cq.Workplane("XY").box(1, 1, 1)
    assy = cq.Assembly()
    assy.add(box, name="box")

    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["fuel"])
    c2y.mesh_surfaces()
    c2y.mesh_volumes()

    path = str(tmp_path / "mixed.vtkhdf")
    c2y.to_vtkhdf(path)

    assert os.path.exists(path)

    with h5py.File(path, "r") as f:
        root = f["VTKHDF"]
        types = root["Types"][:]
        # Should have both triangles (5) and tetrahedra (10)
        assert 5 in types
        assert 10 in types
