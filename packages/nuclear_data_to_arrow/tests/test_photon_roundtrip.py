"""Test photon data roundtrip: ENDF -> endf.IncidentPhoton -> .arrow/ -> readback."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nuclear_data_to_arrow import (
    export_photon_to_arrow, read_photon_from_arrow, verify_photon,
)


class TestPhotonRoundtrip:
    """Test exporting and re-reading photon data preserves all values."""

    def test_basic_element(self, photon_data):
        """Test roundtrip with a basic element."""
        data = photon_data

        with tempfile.TemporaryDirectory() as tmpdir:
            arrow_path = Path(tmpdir) / f"{data.name}.arrow"
            export_photon_to_arrow(data, arrow_path)

            # Check files exist
            assert (arrow_path / "version.json").exists()
            assert (arrow_path / "element.arrow").exists()

            # Read back
            arrow_data = read_photon_from_arrow(arrow_path)

            # Verify version.json
            assert "version" in arrow_data
            assert arrow_data["version"]["format_version"] == 1

            # Verify metadata
            elem = arrow_data["element"]
            assert elem["name"] == data.name
            assert elem["Z"] == data.atomic_number

            # Full verification
            assert verify_photon(data, arrow_path)

    def test_cross_sections(self, photon_data):
        """Test that photon cross sections match."""
        data = photon_data

        with tempfile.TemporaryDirectory() as tmpdir:
            arrow_path = Path(tmpdir) / f"{data.name}.arrow"
            export_photon_to_arrow(data, arrow_path)
            arrow_data = read_photon_from_arrow(arrow_path)

            elem = arrow_data["element"]

            # Build union grid
            union_grid = np.array([])
            for rx in data:
                union_grid = np.union1d(union_grid, rx.xs.x)

            # The raw grid was retired (#500); ln_energy is the one that ships.
            np.testing.assert_allclose(
                np.exp(np.array(elem["ln_energy"])), union_grid, rtol=1e-14)

    def test_log_space_data(self, photon_data):
        """Test that log-space energy and XS are present and correct."""
        data = photon_data

        with tempfile.TemporaryDirectory() as tmpdir:
            arrow_path = Path(tmpdir) / f"{data.name}.arrow"
            export_photon_to_arrow(data, arrow_path)
            arrow_data = read_photon_from_arrow(arrow_path)

            elem = arrow_data["element"]

            # ln_energy is the shipped form of the grid: the raw `energy`
            # column and the three ln_*_xs columns were retired (#500), the
            # first because nothing read it and the others because the reader
            # recomputes the log from the raw cross sections for consistent
            # zero clamping. Check it against the source grid rather than
            # against a sibling column that no longer exists.
            assert "ln_energy" in elem
            assert "energy" not in elem
            union_grid = np.array([])
            for rx in data:
                union_grid = np.union1d(union_grid, rx.xs.x)
            np.testing.assert_allclose(
                np.array(elem["ln_energy"]), np.log(union_grid), rtol=1e-14)
            for retired in ("ln_coherent_xs", "ln_incoherent_xs", "ln_photoelectric_xs"):
                assert retired not in elem

    def test_subshells(self, photon_data):
        """Test that subshell data is preserved with log-space XS."""
        data = photon_data

        with tempfile.TemporaryDirectory() as tmpdir:
            arrow_path = Path(tmpdir) / f"{data.name}.arrow"
            export_photon_to_arrow(data, arrow_path)
            arrow_data = read_photon_from_arrow(arrow_path)

            if arrow_data["subshells"]:
                for sub in arrow_data["subshells"]:
                    assert sub["designator"] is not None
                    assert len(sub["xs"]) > 0
                    # Verify ln_xs is present
                    assert "ln_xs" in sub
                    if sub["ln_xs"] is not None and len(sub["ln_xs"]) > 0:
                        xs = np.array(sub["xs"])
                        ln_xs = np.array(sub["ln_xs"])
                        # Log of XS values (with safe handling of zeros)
                        mask = xs > 0
                        if np.any(mask):
                            np.testing.assert_allclose(
                                ln_xs[mask], np.log(xs[mask]),
                                rtol=1e-14)

    def test_compton_profiles(self, photon_data):
        """Test that Compton profile data is preserved with CDFs."""
        data = photon_data

        if not data.compton_profiles:
            pytest.skip("No Compton profile data")

        with tempfile.TemporaryDirectory() as tmpdir:
            arrow_path = Path(tmpdir) / f"{data.name}.arrow"
            export_photon_to_arrow(data, arrow_path)
            arrow_data = read_photon_from_arrow(arrow_path)

            assert "compton" in arrow_data
            cmp = arrow_data["compton"]
            profile = data.compton_profiles

            np.testing.assert_array_equal(
                np.array(cmp["num_electrons"]),
                np.array(profile['num_electrons']))
            np.testing.assert_array_equal(
                np.array(cmp["binding_energy"]),
                np.array(profile['binding_energy']))
            np.testing.assert_array_equal(
                np.array(cmp["pz"]),
                np.array(profile['J'][0].x))

            # Verify CDFs exist and are valid
            assert "J_cdf_data" in cmp
            if cmp["J_cdf_data"] is not None and len(cmp["J_cdf_data"]) > 0:
                cdf = np.array(cmp["J_cdf_data"]).reshape(cmp["J_cdf_shape"])
                J = np.array(cmp["J_data"]).reshape(cmp["J_shape"])
                pz = np.array(cmp["pz"])
                for s in range(cdf.shape[0]):
                    # The CDF is the cumulative trapezoidal integral of J, left
                    # un-normalized (OpenMC convention: ends near 0.5), so its
                    # final value equals the full trapezoidal integral of J.
                    np.testing.assert_allclose(
                        cdf[s, -1], np.trapezoid(J[s], pz), rtol=1e-10)
                    # Should be monotonically non-decreasing
                    assert np.all(cdf[s, 1:] >= cdf[s, :-1] - 1e-15)

    def test_compton_subshell_map(self, photon_data):
        """The Compton -> atomic-relaxation subshell map is well formed."""
        data = photon_data

        if not data.compton_profiles:
            pytest.skip("No Compton profile data")

        with tempfile.TemporaryDirectory() as tmpdir:
            arrow_path = Path(tmpdir) / f"{data.name}.arrow"
            export_photon_to_arrow(data, arrow_path)
            arrow_data = read_photon_from_arrow(arrow_path)

            cmp = arrow_data["compton"]
            subshells = arrow_data["subshells"]
            n_compton = len(cmp["num_electrons"])

            offsets = list(cmp["subshell_map_offsets"])
            indices = list(cmp["subshell_map_indices"])
            weights = list(cmp["subshell_map_weights"])

            # CSR structure
            assert len(offsets) == n_compton + 1
            assert offsets[0] == 0
            assert offsets[-1] == len(indices) == len(weights)
            assert all(offsets[i] <= offsets[i + 1] for i in range(n_compton))
            # Injectivity: each subshell is claimed by at most one Compton shell
            assert len(indices) == len(set(indices))

            sub_occ = [float(s["num_electrons"]) for s in subshells]
            for c in range(n_compton):
                lo, hi = offsets[c], offsets[c + 1]
                if hi == lo:
                    continue  # outer/valence shell with no clean counterpart
                grp_idx = indices[lo:hi]
                grp_w = weights[lo:hi]
                assert all(0 <= j < len(subshells) for j in grp_idx)
                # Weights are occupancy fractions summing to 1 within the group
                np.testing.assert_allclose(sum(grp_w), 1.0, atol=1e-9)
                c_occ = float(cmp["num_electrons"][c])
                np.testing.assert_allclose(
                    sum(sub_occ[j] for j in grp_idx), c_occ,
                    rtol=1e-3, atol=1e-6)
                for j, w in zip(grp_idx, grp_w):
                    np.testing.assert_allclose(w, sub_occ[j] / c_occ, rtol=1e-9)

    def test_bremsstrahlung(self, photon_data):
        """Test that bremsstrahlung data is preserved."""
        data = photon_data

        if not data.bremsstrahlung:
            pytest.skip("No bremsstrahlung data")

        with tempfile.TemporaryDirectory() as tmpdir:
            arrow_path = Path(tmpdir) / f"{data.name}.arrow"
            export_photon_to_arrow(data, arrow_path)
            arrow_data = read_photon_from_arrow(arrow_path)

            assert "bremsstrahlung" in arrow_data
            abrem = arrow_data["bremsstrahlung"]
            brem = data.bremsstrahlung

            assert abrem["I"] == float(brem['I'])
            np.testing.assert_array_equal(
                np.array(abrem["electron_energy"]),
                np.array(brem['electron_energy']))

    def test_version_json(self, photon_data):
        """Test that version.json contains expected fields."""
        data = photon_data

        with tempfile.TemporaryDirectory() as tmpdir:
            arrow_path = Path(tmpdir) / f"{data.name}.arrow"
            export_photon_to_arrow(data, arrow_path, library="test-lib")

            version = json.loads((arrow_path / "version.json").read_text())
            assert version["format_version"] == 1
            assert version["library"] == "test-lib"
            assert "converter_version" in version
            assert "created_utc" in version
