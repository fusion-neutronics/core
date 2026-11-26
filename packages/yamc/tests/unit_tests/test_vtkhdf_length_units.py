"""``length_units`` on the VTK-HDF writers: positions are scaled from yamc's cm,
data arrays are not."""

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import yamc  # noqa: E402
from yamc.vtkhdf import (  # noqa: E402
    mesh_tally_to_vtkhdf,
    polylines_to_vtkhdf,
    source_to_vtkhdf,
)

SCALES = [("cm", 1.0), ("m", 0.01), ("mm", 10.0)]


class _MockTally:
    """The tally properties mesh_tally_to_vtkhdf reads, on a 2x3x4 mesh."""

    def __init__(self):
        self.mesh = yamc.RegularRectangularMesh(
            lower_left=[-1.0, 2.0, 5.0], upper_right=[3.0, 8.0, 13.0], shape=[2, 3, 4]
        )
        self.scores = ["flux"]
        self.mean = np.arange(1.0, 25.0).tolist()
        self.standard_deviation = (np.arange(1.0, 25.0) * 0.1).tolist()
        self.has_unstructured_mesh = False
        self.n_mesh_bins = 24
        self.n_energy_bins = 1
        self.n_parent_bins = 1
        self.parent_nuclides = None
        self.energy_bins = None


def _sphere_model():
    """A model, and its one cell and region."""
    sphere = yamc.Sphere(radius=5.0, boundary="vacuum")
    mat = yamc.Material(composition={"H1": 1.0}, density=0.001, temperature=294)
    cell = yamc.Cell(region=sphere.below, material=mat)
    source = yamc.NeutronSource(position=(1, 2, 3), energy=14.06e6)
    return yamc.Model(geometry=yamc.Geometry([cell]), source=source), cell, sphere.below


@pytest.mark.parametrize("units, scale", SCALES)
def test_mesh_tally_coordinates_are_scaled_and_data_is_not(tmp_path, units, scale):
    tally = _MockTally()
    mesh_tally_to_vtkhdf(str(tmp_path / "cm.vtkhdf"), tally)
    mesh_tally_to_vtkhdf(str(tmp_path / "x.vtkhdf"), tally, length_units=units)
    with h5py.File(tmp_path / "cm.vtkhdf") as a, h5py.File(tmp_path / "x.vtkhdf") as b:
        ra, rb = a["VTKHDF"], b["VTKHDF"]
        np.testing.assert_allclose(rb.attrs["Origin"], [-1.0 * scale, 2.0 * scale, 5.0 * scale])
        np.testing.assert_allclose(rb.attrs["Spacing"], [2.0 * scale, 2.0 * scale, 2.0 * scale])
        np.testing.assert_array_equal(ra.attrs["WholeExtent"], rb.attrs["WholeExtent"])
        for name in ra["CellData"]:
            np.testing.assert_array_equal(ra["CellData"][name][()], rb["CellData"][name][()])


def test_tally_to_vtkhdf_forwards_length_units(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "yamc.vtkhdf.mesh_tally_to_vtkhdf", lambda f, t, **kw: seen.update(kw)
    )
    yamc.Tally(scores=["flux"]).to_vtkhdf("unused.vtkhdf", length_units="m")
    assert seen == {"length_units": "m"}


@pytest.mark.parametrize("writer", ["geometry", "cell", "region"])
def test_voxel_writers_scale_origin_and_spacing(tmp_path, writer):
    model, cell, region = _sphere_model()
    obj = {"geometry": model.geometry, "cell": cell, "region": region}[writer]
    obj.to_vtkhdf(str(tmp_path / "cm.vtkhdf"), resolution=125)
    obj.to_vtkhdf(str(tmp_path / "m.vtkhdf"), resolution=125, length_units="m")
    with h5py.File(tmp_path / "cm.vtkhdf") as a, h5py.File(tmp_path / "m.vtkhdf") as b:
        ra, rb = a["VTKHDF"], b["VTKHDF"]
        np.testing.assert_allclose(rb.attrs["Origin"], ra.attrs["Origin"] * 0.01)
        np.testing.assert_allclose(rb.attrs["Spacing"], ra.attrs["Spacing"] * 0.01)
        for name in ra["CellData"]:
            np.testing.assert_array_equal(ra["CellData"][name][()], rb["CellData"][name][()])


def test_source_points_are_scaled(tmp_path):
    source = yamc.NeutronSource(position=(1, 2, 3), energy=14.06e6)
    source.to_vtkhdf(str(tmp_path / "s.vtkhdf"), samples=5, length_units="mm")
    with h5py.File(tmp_path / "s.vtkhdf") as f:
        np.testing.assert_allclose(f["VTKHDF/Points"][()], [[10, 20, 30]] * 5)


def test_model_blocks_share_the_unit(tmp_path):
    model, _, _ = _sphere_model()
    model.to_vtkhdf(str(tmp_path / "cm.vtkhdf"), resolution=125, samples=5)
    model.to_vtkhdf(str(tmp_path / "m.vtkhdf"), resolution=125, samples=5, length_units="m")
    with h5py.File(tmp_path / "cm.vtkhdf") as a, h5py.File(tmp_path / "m.vtkhdf") as b:
        np.testing.assert_allclose(
            b["VTKHDF/geometry"].attrs["Spacing"], a["VTKHDF/geometry"].attrs["Spacing"] * 0.01
        )
        np.testing.assert_allclose(b["VTKHDF/source/Points"][()], [[0.01, 0.02, 0.03]] * 5)


@pytest.mark.parametrize("include", ["geometry", "source"])
def test_model_single_block_takes_the_unit(tmp_path, include):
    _sphere_model()[0].to_vtkhdf(
        str(tmp_path / "m.vtkhdf"), include=include, resolution=125, samples=5, length_units="m"
    )
    with h5py.File(tmp_path / "m.vtkhdf") as f:
        root = f["VTKHDF"]
        if include == "source":
            np.testing.assert_allclose(root["Points"][()], [[0.01, 0.02, 0.03]] * 5)
        else:
            assert root.attrs["Spacing"][0] < 1.0  # 10 cm sphere over 5 voxels, in m


def test_polyline_points_are_scaled(tmp_path):
    polylines_to_vtkhdf(
        str(tmp_path / "p.vtkhdf"), [[0, 0, 0], [100, 0, 0]], [0, 2], length_units="m"
    )
    with h5py.File(tmp_path / "p.vtkhdf") as f:
        np.testing.assert_allclose(f["VTKHDF/Points"][()], [[0, 0, 0], [1, 0, 0]])


def test_unstructured_mesh_points_are_scaled(tmp_path):
    if not hasattr(yamc, "UnstructuredMesh"):
        pytest.skip("built without the mesh feature")
    vertices = [[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]]
    mesh = yamc.UnstructuredMesh(vertices, [[0, 1, 2, 3]])
    mesh.to_vtkhdf(tmp_path / "u.vtkhdf", length_units="m")
    with h5py.File(tmp_path / "u.vtkhdf") as f:
        np.testing.assert_allclose(f["VTKHDF/Points"][()], np.array(vertices) * 0.01)


def test_an_unknown_unit_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="length_units must be one of"):
        mesh_tally_to_vtkhdf(str(tmp_path / "x.vtkhdf"), _MockTally(), length_units="inch")


def _units_attr(group):
    value = group.attrs["length_units"]
    return value.decode() if isinstance(value, bytes) else value


@pytest.mark.parametrize("units", ["cm", "m", "mm", "km"])
def test_every_file_records_its_length_unit(tmp_path, units):
    """A reader can tell cm from m without being told."""
    model, cell, region = _sphere_model()
    writers = {
        "tally": lambda p: mesh_tally_to_vtkhdf(p, _MockTally(), length_units=units),
        "geometry": lambda p: model.geometry.to_vtkhdf(p, resolution=27, length_units=units),
        "cell": lambda p: cell.to_vtkhdf(p, resolution=27, length_units=units),
        "region": lambda p: region.to_vtkhdf(p, resolution=27, length_units=units),
        "source": lambda p: source_to_vtkhdf(p, model.source, samples=3, length_units=units),
        "polylines": lambda p: polylines_to_vtkhdf(p, [[0, 0, 0], [1, 0, 0]], [0, 2], length_units=units),
    }
    for name, write in writers.items():
        path = str(tmp_path / f"{name}.vtkhdf")
        write(path)
        with h5py.File(path) as f:
            assert _units_attr(f["VTKHDF"]) == units, name

    path = str(tmp_path / "model.vtkhdf")
    model.to_vtkhdf(path, resolution=27, samples=3, length_units=units)
    with h5py.File(path) as f:
        assert _units_attr(f["VTKHDF"]) == units
        for block in ("geometry", "source"):
            assert _units_attr(f["VTKHDF"][block]) == units, block


def test_an_unstructured_mesh_records_its_length_unit(tmp_path):
    if not hasattr(yamc, "UnstructuredMesh"):
        pytest.skip("built without the mesh feature")
    mesh = yamc.UnstructuredMesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 1, 2, 3]])
    mesh.to_vtkhdf(tmp_path / "u.vtkhdf", length_units="km")
    with h5py.File(tmp_path / "u.vtkhdf") as f:
        assert _units_attr(f["VTKHDF"]) == "km"
        np.testing.assert_allclose(f["VTKHDF/Points"][()][1], [1e-5, 0, 0])
