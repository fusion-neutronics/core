"""Tests for yamc.vtkhdf mesh tally VTK-HDF export."""

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import yamc  # noqa: E402
from yamc.vtkhdf import mesh_tally_to_vtkhdf, source_to_vtkhdf  # noqa: E402


def _keywords_available():
    try:
        m = yamc.Material(
            composition={"Li6": 1.0},
            density=1.0,
            temperature=294,
        )
        m.read_nuclear_data("endf-b8.1")
        return True
    except Exception:
        return False


requires_keywords = pytest.mark.skipif(
    not _keywords_available(),
    reason="keyword download requires download feature"
)


# ---------------------------------------------------------------------------
# Helper: a lightweight stand-in for a scored Tally.
#
# mesh_tally_to_vtkhdf reads tally properties: .mesh, .has_unstructured_mesh,
# .n_parent_bins, .n_energy_bins, .n_mesh_bins, .parent_nuclides, .energy_bins,
# .scores, .mean, .standard_deviation -- so we construct a tiny object with those attrs.
# ---------------------------------------------------------------------------


class _MockTally:
    """Minimal tally-like object with known data for testing."""

    def __init__(self, scores, mesh, mean, std_dev, energy_bins=None):
        self.scores = scores
        self._mesh = mesh
        self.mean = mean
        self.standard_deviation = std_dev
        self._energy_bins = energy_bins

    @property
    def mesh(self):
        return self._mesh

    @property
    def has_unstructured_mesh(self):
        return False

    @property
    def n_mesh_bins(self):
        if self._mesh is None:
            return 1
        dim = self._mesh.shape
        return dim[0] * dim[1] * dim[2]

    @property
    def n_energy_bins(self):
        if self._energy_bins is None:
            return 1
        return len(self._energy_bins) - 1

    @property
    def n_parent_bins(self):
        return 1

    @property
    def parent_nuclides(self):
        return None

    @property
    def energy_bins(self):
        return self._energy_bins


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mesh_2x3x4():
    """A 2x3x4 RegularRectangularMesh."""
    mesh = yamc.RegularRectangularMesh(
        lower_left=[0.0, 0.0, 0.0],
        upper_right=[2.0, 3.0, 4.0],
        shape=[2, 3, 4])
    return mesh


@pytest.fixture
def simple_tally(mesh_2x3x4):
    """Single-score tally with no energy/parent filters (24 mesh bins)."""
    mesh = mesh_2x3x4
    n = 2 * 3 * 4  # 24
    mean = np.arange(1.0, n + 1.0)
    std_dev = mean * 0.1
    return _MockTally(
        scores=["flux"],
        mesh=mesh,
        mean=mean.tolist(),
        std_dev=std_dev.tolist())


@pytest.fixture
def energy_tally(mesh_2x3x4):
    """Single-score tally with 3 energy bins (3 x 24 = 72 bins)."""
    mesh = mesh_2x3x4
    energy_bins = [0.0, 1e6, 10e6, 20e6]
    n_mesh = 2 * 3 * 4  # 24
    n_energy = 3
    n = n_energy * n_mesh  # 72

    mean = np.arange(1.0, n + 1.0)
    std_dev = mean * 0.05
    return _MockTally(
        scores=["flux"],
        mesh=mesh,
        mean=mean.tolist(),
        std_dev=std_dev.tolist(),
        energy_bins=energy_bins)


@pytest.fixture
def multi_score_tally(mesh_2x3x4):
    """Two-score tally with no energy/parent filters."""
    mesh = mesh_2x3x4
    n_mesh = 2 * 3 * 4  # 24
    n = 2 * n_mesh  # 48

    mean = np.arange(1.0, n + 1.0)
    std_dev = mean * 0.1
    return _MockTally(
        scores=["flux", "heating"],
        mesh=mesh,
        mean=mean.tolist(),
        std_dev=std_dev.tolist())


@pytest.fixture
def nonunit_mesh_tally():
    """Tally on a 2x2x2 mesh where each voxel has volume 8 cm^3."""
    mesh = yamc.RegularRectangularMesh(
        lower_left=[0.0, 0.0, 0.0],
        upper_right=[4.0, 4.0, 4.0],
        shape=[2, 2, 2])
    n = 2 * 2 * 2  # 8
    mean = np.arange(1.0, n + 1.0)
    std_dev = mean * 0.1
    return _MockTally(
        scores=["flux"],
        mesh=mesh,
        mean=mean.tolist(),
        std_dev=std_dev.tolist())


# ---------------------------------------------------------------------------
# Tests -- simple (single score, mesh only)
# ---------------------------------------------------------------------------


def test_tally_to_vtkhdf_forwards_to_mesh_writer(monkeypatch):
    """tally.to_vtkhdf forwards to yamc.vtkhdf.mesh_tally_to_vtkhdf."""
    calls = []

    def fake_writer(filename, tally, **kwargs):
        calls.append((filename, tally, kwargs))

    monkeypatch.setattr("yamc.vtkhdf.mesh_tally_to_vtkhdf", fake_writer)

    tally = yamc.Tally(scores=["flux"])
    tally.to_vtkhdf("flux.vtkhdf", sum_energy=False, scaling_factor=2.0)

    assert calls == [
        (
            "flux.vtkhdf",
            tally,
            {"sum_energy": False, "scaling_factor": 2.0},
        )
    ]


def test_tally_to_vtkhdf_per_nuclide_rejected(tmp_path, mesh_2x3x4):
    """Per-nuclide (microscopic) tallies fail fast with a clear message."""
    # A reaction-rate score: a per-nuclide axis needs a score with a cross
    # section (issue #305). What is under test is the export rejection.
    tally = yamc.Tally(
        scores=["(n,gamma)"], mesh=mesh_2x3x4, nuclides=["Li6", "Li7"]
    )
    with pytest.raises(ValueError, match="nuclides.*not supported"):
        tally.to_vtkhdf(str(tmp_path / "rejected.vtkhdf"))


def test_tally_to_vtkhdf_unrun_tally(tmp_path, mesh_2x3x4):
    """tally.to_vtkhdf works end-to-end on a real Tally (zeros pre-run)."""
    tally = yamc.Tally(scores=["flux"], mesh=mesh_2x3x4)
    path = tmp_path / "unrun.vtkhdf"
    tally.to_vtkhdf(str(path))

    with h5py.File(str(path), "r") as f:
        root = f["VTKHDF"]
        assert root.attrs["Type"] == b"ImageData"
        cd = root["CellData"]
        assert cd["flux_mean"].shape == (4, 3, 2)
        np.testing.assert_array_equal(cd["flux_mean"][...], 0.0)
        np.testing.assert_array_equal(cd["flux_relative_error"][...], 0.0)


@requires_keywords
def test_tally_to_vtkhdf_simulated(tmp_path):
    """tally.to_vtkhdf exports simulated results matching results[tally]."""
    yamc.set_cross_section_data_entry('fendl-3.2d')

    sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=10.0, boundary='vacuum')
    material = yamc.Material(
        composition={"H1": 1.0},
        density=0.001,
        temperature=294,
    )
    cell = yamc.Cell(name="sphere", region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1]),
        position=(0, 0, 0)
    )
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-5.0, -5.0, -5.0],
        upper_right=[5.0, 5.0, 5.0],
        shape=[2, 3, 4])
    tally = yamc.Tally(scores=['flux'], mesh=mesh)

    model = yamc.Model(geometry=geometry, tallies=[tally], source=source)
    results = model.simulate_transport(total_particles=1000, seed=42)

    # Tally exposes the same accumulated statistics as results[tally]
    assert tally.mean == results[tally].mean
    assert tally.standard_deviation == results[tally].standard_deviation

    path = tmp_path / "simulated.vtkhdf"
    tally.to_vtkhdf(str(path))

    voxel_volume = 5.0 * (10.0 / 3.0) * 2.5  # dx * dy * dz
    expected = np.array(results[tally].mean).reshape(4, 3, 2) / voxel_volume
    with h5py.File(str(path), "r") as f:
        cd = f["VTKHDF"]["CellData"]
        np.testing.assert_allclose(cd["flux_mean"][...], expected)
    assert expected.any()  # simulation actually scored something


def test_imagedata_structure(tmp_path, simple_tally):
    """VTK-HDF ImageData has correct header and CellData datasets."""
    path = tmp_path / "test.vtkhdf"
    mesh_tally_to_vtkhdf(str(path), simple_tally)

    with h5py.File(str(path), "r") as f:
        root = f["VTKHDF"]

        # Header
        assert list(root.attrs["Version"]) == [2, 1]
        assert root.attrs["Type"] == b"ImageData"
        np.testing.assert_array_equal(root.attrs["Origin"], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(root.attrs["Spacing"], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(
            root.attrs["WholeExtent"], [0, 2, 0, 3, 0, 4]
        )

        # CellData
        cd = root["CellData"]
        assert "flux_mean" in cd
        assert "flux_standard_deviation" in cd
        assert "flux_relative_error" in cd

        # Shape should be (nz=4, ny=3, nx=2)
        assert cd["flux_mean"].shape == (4, 3, 2)
        assert cd["flux_standard_deviation"].shape == (4, 3, 2)
        assert cd["flux_relative_error"].shape == (4, 3, 2)


def test_imagedata_values(tmp_path, simple_tally):
    """Mean and std_dev are volume-normalised by default."""
    path = tmp_path / "test.vtkhdf"
    mesh_tally_to_vtkhdf(str(path), simple_tally)

    mean_flat = np.array(simple_tally.mean)
    std_flat = np.array(simple_tally.standard_deviation)
    # 2×3×4 mesh over [0,2]×[0,3]×[0,4] → dx=1, dy=1, dz=1 → vol=1
    vol = 1.0 * 1.0 * 1.0

    with h5py.File(str(path), "r") as f:
        cd = f["VTKHDF/CellData"]
        np.testing.assert_allclose(
            cd["flux_mean"][()], (mean_flat / vol).reshape(4, 3, 2)
        )
        np.testing.assert_allclose(
            cd["flux_standard_deviation"][()], (std_flat / vol).reshape(4, 3, 2)
        )
        expected_rel = np.where(
            mean_flat != 0, std_flat / mean_flat, 0.0
        ).reshape(4, 3, 2)
        np.testing.assert_allclose(cd["flux_relative_error"][()], expected_rel)


def test_imagedata_values_no_volume_normalization(tmp_path, simple_tally):
    """With volume_normalization=False, raw values are written."""
    path = tmp_path / "test.vtkhdf"
    mesh_tally_to_vtkhdf(str(path), simple_tally, volume_normalization=False)

    mean_flat = np.array(simple_tally.mean)
    std_flat = np.array(simple_tally.standard_deviation)

    with h5py.File(str(path), "r") as f:
        cd = f["VTKHDF/CellData"]
        np.testing.assert_allclose(
            cd["flux_mean"][()], mean_flat.reshape(4, 3, 2)
        )
        np.testing.assert_allclose(
            cd["flux_standard_deviation"][()], std_flat.reshape(4, 3, 2)
        )
        expected_rel = np.where(
            mean_flat != 0, std_flat / mean_flat, 0.0
        ).reshape(4, 3, 2)
        np.testing.assert_allclose(cd["flux_relative_error"][()], expected_rel)


def test_volume_normalization_nonunit(tmp_path, nonunit_mesh_tally):
    """Volume normalization divides by voxel volume (dx*dy*dz = 8)."""
    path = tmp_path / "test.vtkhdf"
    mesh_tally_to_vtkhdf(str(path), nonunit_mesh_tally)

    mean_flat = np.array(nonunit_mesh_tally.mean)
    std_flat = np.array(nonunit_mesh_tally.standard_deviation)
    vol = 2.0 * 2.0 * 2.0  # dx=dy=dz=2

    with h5py.File(str(path), "r") as f:
        cd = f["VTKHDF/CellData"]
        np.testing.assert_allclose(
            cd["flux_mean"][()], (mean_flat / vol).reshape(2, 2, 2)
        )
        np.testing.assert_allclose(
            cd["flux_standard_deviation"][()], (std_flat / vol).reshape(2, 2, 2)
        )
        # rel_error unchanged by volume normalization
        expected_rel = np.where(
            mean_flat != 0, std_flat / mean_flat, 0.0
        ).reshape(2, 2, 2)
        np.testing.assert_allclose(cd["flux_relative_error"][()], expected_rel)


def test_scaling_factor(tmp_path, simple_tally):
    """scaling_factor multiplies mean and std_dev but not rel_error."""
    path = tmp_path / "test.vtkhdf"
    factor = 1e10
    mesh_tally_to_vtkhdf(
        str(path), simple_tally, volume_normalization=False,
        scaling_factor=factor)

    mean_flat = np.array(simple_tally.mean)
    std_flat = np.array(simple_tally.standard_deviation)

    with h5py.File(str(path), "r") as f:
        cd = f["VTKHDF/CellData"]
        np.testing.assert_allclose(
            cd["flux_mean"][()], (mean_flat * factor).reshape(4, 3, 2)
        )
        np.testing.assert_allclose(
            cd["flux_standard_deviation"][()], (std_flat * factor).reshape(4, 3, 2)
        )
        # rel_error unchanged by scaling
        expected_rel = np.where(
            mean_flat != 0, std_flat / mean_flat, 0.0
        ).reshape(4, 3, 2)
        np.testing.assert_allclose(cd["flux_relative_error"][()], expected_rel)


def test_values_subset(tmp_path, simple_tally):
    """Only requested value types are written."""
    path = tmp_path / "test.vtkhdf"
    mesh_tally_to_vtkhdf(
        str(path), simple_tally, datasets=("mean",)
    )

    with h5py.File(str(path), "r") as f:
        cd = f["VTKHDF/CellData"]
        assert "flux_mean" in cd
        assert "flux_standard_deviation" not in cd
        assert "flux_relative_error" not in cd


# ---------------------------------------------------------------------------
# Tests -- energy bins
# ---------------------------------------------------------------------------


def test_energy_summed(tmp_path, energy_tally):
    """With sum_energy=True (default), energy bins are summed."""
    path = tmp_path / "test.vtkhdf"
    mesh_tally_to_vtkhdf(str(path), energy_tally)

    mean_4d = np.array(energy_tally.mean).reshape(1, 1, 3, 24)
    expected = mean_4d.sum(axis=2).reshape(4, 3, 2)

    with h5py.File(str(path), "r") as f:
        cd = f["VTKHDF/CellData"]
        assert "flux_mean" in cd
        assert "flux_E0_mean" not in cd
        np.testing.assert_allclose(cd["flux_mean"][()], expected)


def test_energy_per_bin(tmp_path, energy_tally):
    """With sum_energy=False, per-energy arrays are written."""
    path = tmp_path / "test.vtkhdf"
    mesh_tally_to_vtkhdf(str(path), energy_tally, sum_energy=False)

    mean_4d = np.array(energy_tally.mean).reshape(1, 1, 3, 24)

    with h5py.File(str(path), "r") as f:
        cd = f["VTKHDF/CellData"]
        assert "flux_mean" not in cd

        for i in range(3):
            ds_name = f"flux_E{i}_mean"
            assert ds_name in cd
            expected = mean_4d[0, 0, i, :].reshape(4, 3, 2)
            np.testing.assert_allclose(cd[ds_name][()], expected)

            # Energy boundary attributes
            ds = cd[ds_name]
            assert "energy_low_eV" in ds.attrs
            assert "energy_high_eV" in ds.attrs


def test_energy_boundaries_correct(tmp_path, energy_tally):
    """Per-energy datasets have correct energy boundary attributes."""
    path = tmp_path / "test.vtkhdf"
    mesh_tally_to_vtkhdf(str(path), energy_tally, sum_energy=False)

    bins = [0.0, 1e6, 10e6, 20e6]
    with h5py.File(str(path), "r") as f:
        cd = f["VTKHDF/CellData"]
        for i in range(3):
            ds = cd[f"flux_E{i}_mean"]
            assert ds.attrs["energy_low_eV"] == pytest.approx(bins[i])
            assert ds.attrs["energy_high_eV"] == pytest.approx(bins[i + 1])


# ---------------------------------------------------------------------------
# Tests -- multiple scores
# ---------------------------------------------------------------------------


def test_multi_score(tmp_path, multi_score_tally):
    """Multiple scores produce separate datasets."""
    path = tmp_path / "test.vtkhdf"
    mesh_tally_to_vtkhdf(str(path), multi_score_tally)

    with h5py.File(str(path), "r") as f:
        cd = f["VTKHDF/CellData"]
        assert "flux_mean" in cd
        assert "heating_mean" in cd
        assert cd["flux_mean"].shape == (4, 3, 2)
        assert cd["heating_mean"].shape == (4, 3, 2)


def test_score_selection_by_name(tmp_path, multi_score_tally):
    """Selecting a subset of scores by name works."""
    path = tmp_path / "test.vtkhdf"
    mesh_tally_to_vtkhdf(
        str(path), multi_score_tally, scores=["heating"]
    )

    with h5py.File(str(path), "r") as f:
        cd = f["VTKHDF/CellData"]
        assert "flux_mean" not in cd
        assert "heating_mean" in cd


def test_score_selection_by_index(tmp_path, multi_score_tally):
    """Selecting a subset of scores by index works."""
    path = tmp_path / "test.vtkhdf"
    mesh_tally_to_vtkhdf(str(path), multi_score_tally, scores=[0])

    with h5py.File(str(path), "r") as f:
        cd = f["VTKHDF/CellData"]
        assert "flux_mean" in cd
        assert "heating_mean" not in cd


# ---------------------------------------------------------------------------
# Tests -- error handling
# ---------------------------------------------------------------------------


def test_no_mesh_filter_raises(tmp_path):
    """Raises ValueError when tally has no mesh."""
    tally = _MockTally(
        scores=["flux"],
        mesh=None,
        mean=[1.0],
        std_dev=[0.1])
    with pytest.raises(ValueError, match="mesh"):
        mesh_tally_to_vtkhdf(str(tmp_path / "fail.vtkhdf"), tally)


def test_bad_score_name_raises(tmp_path, simple_tally):
    """Raises ValueError for unknown score name."""
    with pytest.raises(ValueError, match="not found"):
        mesh_tally_to_vtkhdf(
            str(tmp_path / "fail.vtkhdf"),
            simple_tally,
            scores=["nonexistent"])


def test_bad_score_index_raises(tmp_path, simple_tally):
    """Raises ValueError for out-of-range score index."""
    with pytest.raises(ValueError, match="out of range"):
        mesh_tally_to_vtkhdf(
            str(tmp_path / "fail.vtkhdf"), simple_tally, scores=[99]
        )


def test_bad_value_type_raises(tmp_path, simple_tally):
    """Raises ValueError for unknown value type."""
    with pytest.raises(ValueError, match="Unknown value type"):
        mesh_tally_to_vtkhdf(
            str(tmp_path / "fail.vtkhdf"),
            simple_tally,
            datasets=("mean", "bogus"))


# ---------------------------------------------------------------------------
# Tests -- geometry / cell / region writers
# ---------------------------------------------------------------------------


@pytest.fixture
def sphere_geometry():
    """Simple sphere geometry for testing to_vtkhdf methods."""
    sphere = yamc.Sphere(
        x0=0.0, y0=0.0, z0=0.0, radius=5.0, boundary="vacuum"
    )
    region = sphere.below
    mat = yamc.Material(
        composition={"H1": 1.0},
        density=0.001,
        name="hydrogen",
        temperature=294,
    )
    cell = yamc.Cell(name="sphere", region=region, material=mat)
    geom = yamc.Geometry([cell])
    return geom, cell, region


def test_geometry_to_vtkhdf(tmp_path, sphere_geometry):
    """geometry.to_vtkhdf produces valid ImageData with cell/material IDs."""
    geom, _, _ = sphere_geometry
    path = str(tmp_path / "geom.vtkhdf")
    geom.to_vtkhdf(path, resolution=125)

    with h5py.File(path, "r") as f:
        root = f["VTKHDF"]
        assert root.attrs["Type"] == b"ImageData"
        cd = root["CellData"]
        assert "material_id" in cd
        assert "cell_id" in cd
        # Verify some voxels are inside (material_id != -1)
        assert np.any(cd["material_id"][()] != -1)


def test_cell_bounding_box(sphere_geometry):
    """Cell.bounding_box returns finite bounds from region."""
    _, cell, _ = sphere_geometry
    bb = cell.bounding_box()
    np.testing.assert_allclose(bb.lower_left, [-5.0, -5.0, -5.0])
    np.testing.assert_allclose(bb.upper_right, [5.0, 5.0, 5.0])


def test_cell_to_vtkhdf(tmp_path, sphere_geometry):
    """cell.to_vtkhdf produces valid ImageData."""
    _, cell, _ = sphere_geometry
    path = str(tmp_path / "cell.vtkhdf")
    cell.to_vtkhdf(path, resolution=125)

    with h5py.File(path, "r") as f:
        root = f["VTKHDF"]
        assert root.attrs["Type"] == b"ImageData"
        cd = root["CellData"]
        assert "material_id" in cd
        assert "cell_id" in cd
        assert np.any(cd["cell_id"][()] == 1)


def test_region_to_vtkhdf(tmp_path, sphere_geometry):
    """region.to_vtkhdf produces valid ImageData with inside/outside."""
    _, _, region = sphere_geometry
    path = str(tmp_path / "region.vtkhdf")
    region.to_vtkhdf(path, resolution=125)

    with h5py.File(path, "r") as f:
        root = f["VTKHDF"]
        assert root.attrs["Type"] == b"ImageData"
        cd = root["CellData"]
        assert "region" in cd
        data = cd["region"][()]
        # Center voxel should be inside (1), corners outside (0)
        assert np.any(data == 1)
        assert np.any(data == 0)


# ---------------------------------------------------------------------------
# Tests -- voxelize (Rust core of the geometry writers) matches a find_cell loop
# ---------------------------------------------------------------------------


def test_geometry_voxelize_matches_find_cell():
    """Geometry.voxelize equals the reference per-voxel find_cell loop.

    Uses an offset sphere and an asymmetric (nx != ny != nz) grid so a wrong
    axis order or index formula would show up.
    """
    from yamc.vtkhdf import _check_finite_bbox

    sph = yamc.Sphere(x0=1.0, y0=-2.0, z0=0.5, radius=3.0, boundary="vacuum")
    mat = yamc.Material(composition={"H1": 1.0}, density=0.001, name="h", temperature=294)
    cell = yamc.Cell(name="s", region=sph.below, material=mat)
    geom = yamc.Geometry([cell])

    bb = geom.bounding_box()
    ll, ur = _check_finite_bbox(bb, "Geometry")
    nx, ny, nz = 5, 7, 9
    dx = (ur[0] - ll[0]) / nx
    dy = (ur[1] - ll[1]) / ny
    dz = (ur[2] - ll[2]) / nz

    vox = geom.voxelize(ll, (dx, dy, dz), (nx, ny, nz))
    cell_ids = np.asarray(vox.cell_ids, dtype="int32").reshape(nz, ny, nx)
    mat_ids = np.asarray(vox.material_ids, dtype="int32").reshape(nz, ny, nx)

    ref_cell = np.full((nz, ny, nx), -1, dtype="int32")
    ref_mat = np.full((nz, ny, nx), -1, dtype="int32")
    for iz in range(nz):
        z = ll[2] + (iz + 0.5) * dz
        for iy in range(ny):
            y = ll[1] + (iy + 0.5) * dy
            for ix in range(nx):
                x = ll[0] + (ix + 0.5) * dx
                c = geom.find_cell(x, y, z)
                if c is not None:
                    ref_cell[iz, iy, ix] = c.id if c.id is not None else -1
                    m = c.material
                    if m is not None:
                        ref_mat[iz, iy, ix] = m.id if m.id is not None else -1

    np.testing.assert_array_equal(cell_ids, ref_cell)
    np.testing.assert_array_equal(mat_ids, ref_mat)
    assert (mat_ids != -1).any()  # sphere actually intersects the grid
    assert vox.shape == (nx, ny, nz)


def test_region_voxelize_mask(sphere_geometry):
    """Region.voxelize returns a 1/0 inside mask matching region.contains."""
    _, _, region = sphere_geometry
    ll = (-5.0, -5.0, -5.0)
    nx, ny, nz = 4, 5, 6
    dx, dy, dz = 10.0 / nx, 10.0 / ny, 10.0 / nz
    mask = np.asarray(
        region.voxelize(ll, (dx, dy, dz), (nx, ny, nz)), dtype="int32"
    ).reshape(nz, ny, nx)

    for iz in range(nz):
        z = ll[2] + (iz + 0.5) * dz
        for iy in range(ny):
            y = ll[1] + (iy + 0.5) * dy
            for ix in range(nx):
                x = ll[0] + (ix + 0.5) * dx
                expected = 1 if region.contains((x, y, z)) else 0
                assert mask[iz, iy, ix] == expected


def test_source_sample_n_matches_point_source():
    """NeutronSource.sample_n returns per-sample positions and energies."""
    src = yamc.NeutronSource(position=(1.0, 2.0, 3.0), energy=14.06e6)
    positions, energies = src.sample_n(17)
    assert len(positions) == 17
    assert len(energies) == 17
    np.testing.assert_allclose(np.asarray(positions), [[1.0, 2.0, 3.0]] * 17)
    np.testing.assert_allclose(np.asarray(energies), [14.06e6] * 17)


# ---------------------------------------------------------------------------
# Tests -- source writer
# ---------------------------------------------------------------------------


def test_source_single(tmp_path):
    """Single source writes correct point cloud."""
    src = yamc.NeutronSource(
        position=(1, 2, 3),
        energy=14.06e6)
    path = str(tmp_path / "source.vtkhdf")
    source_to_vtkhdf(path, src, samples=50)

    with h5py.File(path, "r") as f:
        root = f["VTKHDF"]
        assert root.attrs["Type"] == b"UnstructuredGrid"
        assert root["NumberOfPoints"][()][0] == 50
        assert root["NumberOfCells"][()][0] == 50
        assert root["Points"].shape == (50, 3)
        # All points should be at (1, 2, 3) since Point is deterministic
        np.testing.assert_allclose(root["Points"][()], [[1, 2, 3]] * 50)
        # PointData
        pd = root["PointData"]
        assert "energy" in pd
        assert "source_id" in pd
        assert np.all(pd["source_id"][()] == 0)


def test_source_multiple(tmp_path):
    """Multiple sources get distinct source_id values."""
    sources = [
        yamc.NeutronSource(position=(0, 0, 0), energy=14.06e6),
        yamc.NeutronSource(position=(10, 0, 0), energy=2.5e6),
        yamc.NeutronSource(position=(0, 10, 0), energy=1e6),
    ]
    path = str(tmp_path / "sources.vtkhdf")
    source_to_vtkhdf(path, sources, samples=20)

    with h5py.File(path, "r") as f:
        root = f["VTKHDF"]
        n = 3 * 20
        assert root["NumberOfPoints"][()][0] == n
        assert root["Points"].shape == (n, 3)
        sid = root["PointData"]["source_id"][()]
        assert np.sum(sid == 0) == 20
        assert np.sum(sid == 1) == 20
        assert np.sum(sid == 2) == 20
        # First 20 points at origin, next 20 at (10,0,0), last 20 at (0,10,0)
        np.testing.assert_allclose(root["Points"][:20], [[0, 0, 0]] * 20)
        np.testing.assert_allclose(root["Points"][20:40], [[10, 0, 0]] * 20)
        np.testing.assert_allclose(root["Points"][40:60], [[0, 10, 0]] * 20)


# ---------------------------------------------------------------------------
# Tests -- model writer
# ---------------------------------------------------------------------------


@pytest.fixture
def sphere_model(sphere_geometry):
    """Model with sphere geometry and a deterministic point source."""
    geom, _, _ = sphere_geometry
    src = yamc.NeutronSource(position=(1, 2, 3), energy=14.06e6)
    return yamc.Model(geometry=geom, source=src)


def test_model_to_vtkhdf_combined(tmp_path, sphere_model):
    """Default model.to_vtkhdf writes a two-block PartitionedDataSetCollection."""
    path = str(tmp_path / "model.vtkhdf")
    sphere_model.to_vtkhdf(path, resolution=125, samples=25)

    with h5py.File(path, "r") as f:
        root = f["VTKHDF"]
        assert root.attrs["Type"] == b"PartitionedDataSetCollection"

        geo = root["geometry"]
        assert geo.attrs["Type"] == b"ImageData"
        assert geo.attrs["Index"] == 0
        assert np.any(geo["CellData"]["material_id"][()] != -1)

        src = root["source"]
        assert src.attrs["Type"] == b"UnstructuredGrid"
        assert src.attrs["Index"] == 1
        assert src["NumberOfPoints"][()][0] == 25
        np.testing.assert_allclose(src["Points"][()], [[1, 2, 3]] * 25)

        # Assembly softlinks resolve to the block groups
        assembly = root["Assembly"]
        assert assembly["geometry"].attrs["Type"] == b"ImageData"
        assert assembly["source"].attrs["Type"] == b"UnstructuredGrid"


def test_model_to_vtkhdf_geometry_only(tmp_path, sphere_model):
    """include=("geometry",) writes a plain ImageData file."""
    path = str(tmp_path / "geom_only.vtkhdf")
    sphere_model.to_vtkhdf(path, include=("geometry",), resolution=125)

    with h5py.File(path, "r") as f:
        root = f["VTKHDF"]
        assert root.attrs["Type"] == b"ImageData"
        assert np.any(root["CellData"]["material_id"][()] != -1)


def test_model_to_vtkhdf_source_only(tmp_path, sphere_model):
    """include=("source",) writes a plain UnstructuredGrid point cloud."""
    path = str(tmp_path / "src_only.vtkhdf")
    sphere_model.to_vtkhdf(path, include=("source",), samples=25)

    with h5py.File(path, "r") as f:
        root = f["VTKHDF"]
        assert root.attrs["Type"] == b"UnstructuredGrid"
        np.testing.assert_allclose(root["Points"][()], [[1, 2, 3]] * 25)


def test_model_to_vtkhdf_bad_include(tmp_path, sphere_model):
    """Unknown include entries are rejected."""
    with pytest.raises(ValueError, match="Unknown include"):
        sphere_model.to_vtkhdf(
            str(tmp_path / "bad.vtkhdf"), include=("geometry", "tallies"))


def test_model_to_vtkhdf_empty_include(tmp_path, sphere_model):
    """Empty include is rejected."""
    with pytest.raises(ValueError, match="include must contain"):
        sphere_model.to_vtkhdf(str(tmp_path / "empty.vtkhdf"), include=())


def test_model_to_vtkhdf_default_source(tmp_path, sphere_geometry):
    """A model built without an explicit source writes its default source."""
    geom, _, _ = sphere_geometry
    model = yamc.Model(geometry=geom)
    path = str(tmp_path / "default_src.vtkhdf")
    model.to_vtkhdf(path, resolution=125, samples=10)

    with h5py.File(path, "r") as f:
        src = f["VTKHDF"]["source"]
        # Default source: 14.06 MeV point source at the origin
        np.testing.assert_allclose(src["Points"][()], [[0, 0, 0]] * 10)
        np.testing.assert_allclose(
            src["PointData"]["energy"][()], [14.06e6] * 10)
