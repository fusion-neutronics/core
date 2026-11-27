"""``TallyResult.to_vtkhdf``: a mesh tally written in the units the result was
converted to, with each dataset's unit recorded."""

import numpy as np
import pytest

import yamc

h5py = pytest.importorskip("h5py")

J_PER_EV = 1.602176634e-19
STRENGTH = 1e20  # source neutrons per second


@pytest.fixture(scope="module")
def run():
    """Heating, flux and tritium production on a 2x2x2 mesh in an iron sphere."""
    iron = yamc.Material(composition={"Fe56": 1.0}, density=7.874, temperature=294)
    iron.read_nuclear_data({"Fe56": "tests/Fe56.arrow"})
    sphere = yamc.Sphere(radius=20.0, boundary="vacuum")
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-10.0] * 3, upper_right=[10.0] * 3, shape=[2, 2, 2]
    )
    tally = yamc.Tally(scores=["heating", "flux", "H3-production"], mesh=mesh)
    model = yamc.Model(
        geometry=yamc.Geometry([yamc.Cell(region=sphere.below, material=iron)]),
        source=yamc.NeutronSource(position=(0, 0, 0), energy=14.0e6),
        tallies=[tally],
        verbose=[],
    )
    results = model.simulate_transport(total_particles=200, seed=1)
    return tally, results.with_source_strength(STRENGTH)[tally]


def _read(path):
    with h5py.File(path) as f:
        root = f["VTKHDF"]
        cell_data = {
            name: (ds[()].ravel(), dict(ds.attrs)) for name, ds in root["CellData"].items()
        }
        return dict(root.attrs), cell_data


def test_values_and_units_are_the_results(tmp_path, run):
    _, result = run
    units = ["W/m3", "1/(m2 s)", "reactions/(m3 s)"]
    converted = result.to(units)
    converted.to_vtkhdf(str(tmp_path / "r.vtkhdf"), length_units="m")
    attrs, cell_data = _read(tmp_path / "r.vtkhdf")

    np.testing.assert_allclose(attrs["Spacing"], [0.1, 0.1, 0.1])
    mean = np.asarray(converted.mean).reshape(3, 8)
    std = np.asarray(converted.standard_deviation).reshape(3, 8)
    for i, (score, unit) in enumerate(zip(["heating", "flux", "H3-production"], units)):
        values, ds_attrs = cell_data[f"{score}_mean"]
        np.testing.assert_allclose(values, mean[i], rtol=1e-12)
        assert ds_attrs["units"] == unit
        values, ds_attrs = cell_data[f"{score}_standard_deviation"]
        np.testing.assert_allclose(values, std[i], rtol=1e-12)
        assert ds_attrs["units"] == unit
        assert cell_data[f"{score}_relative_error"][1]["units"] == "1"


def test_matches_the_hand_scaled_tally_export(tmp_path, run):
    """W/cm3 from the result equals the per-source-particle export scaled by
    hand, which is what users had to do before."""
    tally, result = run
    result.to(["W/cm3", "1/(cm2 s)", "1/(cm3 s)"]).to_vtkhdf(
        str(tmp_path / "r.vtkhdf"), scores=["heating"]
    )
    tally.to_vtkhdf(
        str(tmp_path / "t.vtkhdf"), scores=["heating"], scaling_factor=STRENGTH * J_PER_EV
    )
    _, by_result = _read(tmp_path / "r.vtkhdf")
    _, by_hand = _read(tmp_path / "t.vtkhdf")
    np.testing.assert_allclose(
        by_result["heating_mean"][0], by_hand["heating_mean"][0], rtol=1e-12
    )
    assert "units" not in by_hand["heating_mean"][1]


def test_an_unconverted_result_says_its_units(tmp_path, run):
    _, result = run
    result.to_vtkhdf(str(tmp_path / "r.vtkhdf"))
    _, cell_data = _read(tmp_path / "r.vtkhdf")
    assert cell_data["heating_mean"][1]["units"] == "eV/s"
    np.testing.assert_allclose(
        cell_data["heating_mean"][0], np.asarray(result.mean).reshape(3, 8)[0], rtol=1e-12
    )
