"""Tests for TallyResult.to_xarray / to_dataset."""

import pytest
import yamc


# ---------------------------------------------------------------------------
# Coordinate-building helper (no simulation needed)
# ---------------------------------------------------------------------------
def test_tally_axis_coords_labels_known_dims():
    from yamc import _tally_axis_coords

    # A reaction-rate score, not flux: a per-nuclide axis folds the nuclide's
    # cross section in, which flux has none of (issue #305).
    tally = yamc.Tally(
        scores=["(n,gamma)"],
        nuclides=["Li6", "Li7", "total"],
        energy_bins=[0.0, 1.0, 1e6, 2e7],
        name="x",
    )
    coords = _tally_axis_coords(tally, ["score", "nuclide", "energy"], [1, 3, 3])
    assert coords["score"] == ["(n,gamma)"]
    assert coords["nuclide"] == ["Li6", "Li7", "total"]
    # energy: 4 edges -> 3 bins, labelled by lower edge
    assert coords["energy"] == [0.0, 1.0, 1e6]


def test_tally_axis_coords_skips_length_mismatch():
    from yamc import _tally_axis_coords

    tally = yamc.Tally(scores=["flux"], name="x")
    # claim 5 energy bins but the tally has none -> no energy coord attached
    coords = _tally_axis_coords(tally, ["score", "energy"], [1, 5])
    assert "energy" not in coords
    assert coords["score"] == ["flux"]


# ---------------------------------------------------------------------------
# End-to-end to_xarray / to_dataset (needs nuclear data + xarray)
# ---------------------------------------------------------------------------
def _keywords_available():
    try:
        m = yamc.Material(composition={"Li6": 1.0}, density=1.0)
        m.read_nuclear_data("endf-b8.1")
        return True
    except Exception:
        return False


requires_keywords = pytest.mark.skipif(
    not _keywords_available(), reason="keyword download requires download feature"
)


def _make_model(tally, total_particles=2000):
    sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=10.0, boundary="vacuum")
    material = yamc.Material(composition={"Li6": 0.5, "Li7": 0.5}, density=0.5)
    cell = yamc.Cell(name="sphere", region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1]), position=(0, 0, 0)
    )
    model = yamc.Model(
        geometry=geometry,
        tallies=[tally],
        source=source,
    )
    return model, {"total_particles": total_particles, "seed": 42}


@requires_keywords
def test_to_xarray_energy_binned():
    xr = pytest.importorskip("xarray")
    yamc.set_cross_section_data_entry("fendl-3.2d")
    tally = yamc.Tally(scores=["flux"], energy_bins=[0.0, 1e3, 1e6, 2e7], name="flux")
    model, run_kwargs = _make_model(tally)
    results = model.simulate_transport(**run_kwargs)

    da = results[tally].to_xarray()
    assert isinstance(da, xr.DataArray)
    assert "energy" in da.dims
    # 3 energy bins from 4 edges, labelled by lower edge
    assert list(da.coords["energy"].values) == [0.0, 1e3, 1e6]
    assert da.name == "flux"


@requires_keywords
def test_to_dataset_has_mean_std_relerr():
    pytest.importorskip("xarray")
    yamc.set_cross_section_data_entry("fendl-3.2d")
    tally = yamc.Tally(scores=["flux"], energy_bins=[0.0, 1e3, 1e6, 2e7], name="flux")
    model, run_kwargs = _make_model(tally)
    results = model.simulate_transport(**run_kwargs)

    ds = results[tally].to_dataset()
    assert set(ds.data_vars) == {"mean", "std_dev", "relative_error"}
    assert "energy" in ds.dims
