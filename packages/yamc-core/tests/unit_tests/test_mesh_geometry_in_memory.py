"""Meshes that go to transport without a file (MeshGeometry.from_cadquery,
MeshGeometry.to_arrow / from_arrow with bytes), and the per-tet properties a
coupled code needs: tet_volume_names and tet_volumes."""

import pathlib

import numpy as np
import pytest

import yamc

MESH = "crates/yamt/tests/data/two_region_tets.arrow"


def _iron():
    return yamc.Material(composition={"Fe56": 1.0}, density=7.8, temperature=294)


def _geometry(source):
    iron = _iron()
    return yamc.MeshGeometry.from_arrow(source, {"fuel": iron, "moderator": iron})


def test_there_is_no_constructor():
    with pytest.raises(TypeError):
        yamc.MeshGeometry(MESH, {})


def test_bytes_load_as_the_file_does():
    from_file = _geometry(MESH)
    from_bytes = _geometry(pathlib.Path(MESH).read_bytes())
    assert from_bytes.volume_names == from_file.volume_names
    assert from_bytes.vertices == from_file.vertices
    assert from_bytes.tetrahedra == from_file.tetrahedra
    assert from_bytes.volume_measures == from_file.volume_measures


def test_to_arrow_round_trips(tmp_path):
    mesh = _geometry(MESH)
    data = mesh.to_arrow()
    assert isinstance(data, bytes)
    assert mesh.to_arrow(tmp_path / "copy.arrow") is None
    assert (tmp_path / "copy.arrow").read_bytes() == data
    iron = _iron()
    for source in (data, tmp_path / "copy.arrow"):
        again = yamc.MeshGeometry.from_arrow(source, {"fuel": iron, "moderator": iron})
        assert again.tetrahedra == mesh.tetrahedra
        assert again.vertices == mesh.vertices
        assert again.volume_names == mesh.volume_names
        assert again.volume_measures == mesh.volume_measures
        assert again.to_arrow() == data


def test_a_path_object_loads():
    assert _geometry(pathlib.Path(MESH)).tetrahedra == _geometry(MESH).tetrahedra


def test_bad_bytes_raise():
    with pytest.raises(RuntimeError, match="Failed to load Arrow mesh"):
        _geometry(b"not a mesh")


def test_tet_volume_names_follow_the_ids():
    mesh = _geometry(MESH)
    names = mesh.volume_names
    assert mesh.tet_volume_names == [
        names[v] if v is not None else None for v in mesh.tet_volume_ids
    ]
    vertices = np.array(mesh.vertices)
    centres = vertices[np.array(mesh.tetrahedra)].mean(axis=1)
    in_fuel = np.array(mesh.tet_volume_names) == "fuel"
    assert in_fuel.any() and not in_fuel.all()
    assert (in_fuel == (centres[:, 0] < 0.5)).all()


def test_tet_volumes_sum_to_each_volume():
    mesh = _geometry(MESH)
    volumes = np.array(mesh.tet_volumes)
    assert len(volumes) == len(mesh.tetrahedra)
    assert (volumes > 0).all()
    names = np.array(mesh.tet_volume_names)
    for v, name in enumerate(mesh.volume_names):
        assert volumes[names == name].sum() == pytest.approx(mesh.volume_measures[v], rel=1e-12)


def test_cad_mesh_goes_to_transport_without_a_file(tmp_path):
    cq = pytest.importorskip("cadquery")
    from yamc.cad.mesher import _CadMesher

    assembly = cq.Assembly()
    assembly.add(cq.Workplane("XY").box(2, 2, 2), name="block")
    iron = _iron()
    iron.read_nuclear_data({"Fe56": "tests/Fe56.arrow"})

    mesh = yamc.MeshGeometry.from_cadquery(
        assembly,
        materials={"block": iron},
        tet_volumes=["block"],
        target_edge_length=0.5,
        graveyard_offset=5.0,
        tolerance=0.1,
    )
    assert sum(mesh.tet_volumes) == pytest.approx(8.0, rel=1e-9)
    assert set(mesh.tet_volume_names) == {"block"}

    # the same mesh _CadMesher writes to a file
    mesher = _CadMesher()
    mesher.add_cadquery_object(assembly, material_tags=["block"])
    mesher.mesh(tet_volumes=["block"], target_edge_length=0.5, tolerance=0.1)
    mesher.to_arrow(tmp_path / "block.arrow")
    assert mesh.to_arrow() == (tmp_path / "block.arrow").read_bytes()

    results = yamc.Model(
        geometry=mesh,
        source=yamc.NeutronSource(position=(0, 0, 0), energy=1e6),
        tallies=[],
        verbose=[],
    ).simulate_transport(total_particles=10, seed=1)
    assert results is not None


def test_from_cadquery_needs_an_edge_length_for_tets():
    cq = pytest.importorskip("cadquery")
    assembly = cq.Assembly()
    assembly.add(cq.Workplane("XY").box(2, 2, 2), name="block")
    with pytest.raises(ValueError, match="target_edge_length"):
        yamc.MeshGeometry.from_cadquery(assembly, {"block": _iron()}, tet_volumes=["block"])
