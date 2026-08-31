"""Test neutron data roundtrip: ACE -> endf.IncidentNeutron -> .arrow/ -> readback."""

import json
import tempfile
from pathlib import Path

import numpy as np

from nuclear_data_to_arrow import (
    export_neutron_to_arrow, read_neutron_from_arrow, verify_neutron,
)


class TestNeutronRoundtrip:
    """Test exporting and re-reading neutron data preserves all values."""

    def test_simple_nuclide(self, neutron_data):
        """Test roundtrip with a simple nuclide (e.g., Li6, H1)."""
        data = neutron_data

        with tempfile.TemporaryDirectory() as tmpdir:
            arrow_path = Path(tmpdir) / f"{data.name}.arrow"
            export_neutron_to_arrow(data, arrow_path)

            # Check files exist
            assert (arrow_path / "version.json").exists()
            assert (arrow_path / "nuclide.arrow").exists()
            assert (arrow_path / "reactions.arrow").exists()
            assert (arrow_path / "fast_xs.arrow").exists()

            # Read back
            arrow_data = read_neutron_from_arrow(arrow_path)

            # Verify version.json
            assert "version" in arrow_data
            assert arrow_data["version"]["format_version"] == 1

            # Verify metadata
            nuc = arrow_data["nuclide"]
            assert nuc["name"] == data.name
            assert nuc["Z"] == data.atomic_number
            assert nuc["A"] == data.mass_number
            assert nuc["atomic_weight_ratio"] == data.atomic_weight_ratio

            # Verify temperatures
            assert nuc["temperatures"] == list(data.temperatures)

            # Verify energy grids
            for i, t in enumerate(nuc["energy_temperatures"]):
                np.testing.assert_array_equal(
                    np.array(nuc["energy_values"][i]),
                    np.array(data.energy[t])
                )

            # Verify reactions exist (including synthesized)
            assert len(arrow_data["reactions"]) > 0

            # Verify synthesized MTs present
            arrow_mts = {r["mt"] for r in arrow_data["reactions"]}
            assert 1 in arrow_mts, "MT 1 (total) should be present"
            assert 101 in arrow_mts, "MT 101 (disappearance) should be present"

            # Full verification
            assert verify_neutron(data, arrow_path)

    def test_fissile_nuclide(self, fissile_data):
        """Test roundtrip with a fissile nuclide (U235, URR, total_nu)."""
        data = fissile_data

        with tempfile.TemporaryDirectory() as tmpdir:
            arrow_path = Path(tmpdir) / f"{data.name}.arrow"
            export_neutron_to_arrow(data, arrow_path)

            # Check files exist
            assert (arrow_path / "version.json").exists()
            assert (arrow_path / "nuclide.arrow").exists()
            assert (arrow_path / "reactions.arrow").exists()
            assert (arrow_path / "fast_xs.arrow").exists()

            # Check URR if present
            if data.urr:
                assert (arrow_path / "urr.arrow").exists()
                arrow_data = read_neutron_from_arrow(arrow_path)
                assert "urr" in arrow_data
                assert len(arrow_data["urr"]) == len(data.urr)
                for i, (temp, urr) in enumerate(data.urr.items()):
                    au = arrow_data["urr"][i]
                    assert au["temperature"] == temp
                    np.testing.assert_array_equal(
                        np.array(au["energy"]),
                        np.array(urr.energy)
                    )

            # Check total_nu if present
            has_derived = any(
                len(rx.derived_products) > 0
                for rx in data.reactions.values()
            )
            if has_derived:
                assert (arrow_path / "total_nu.arrow").exists()

            # Full verification
            assert verify_neutron(data, arrow_path)

    def test_cross_section_values(self, neutron_data):
        """Test that original cross section values are preserved."""
        data = neutron_data

        with tempfile.TemporaryDirectory() as tmpdir:
            arrow_path = Path(tmpdir) / f"{data.name}.arrow"
            export_neutron_to_arrow(data, arrow_path)
            arrow_data = read_neutron_from_arrow(arrow_path)

            arrow_rx_by_mt = {r["mt"]: r for r in arrow_data["reactions"]}

            for rx_orig in data.reactions.values():
                mt = rx_orig.MT
                assert mt in arrow_rx_by_mt, f"MT {mt} missing"
                rx_arrow = arrow_rx_by_mt[mt]

                for t_idx, T in enumerate(rx_arrow["xs_temperatures"]):
                    if rx_orig.xs[T] is not None:
                        expected = np.array(rx_orig.xs[T].y)
                        actual = np.array(rx_arrow["xs_values"][t_idx])
                        np.testing.assert_array_equal(
                            expected, actual,
                            err_msg=f"XS mismatch for MT={mt} T={T}")

    def test_product_yields(self, neutron_data):
        """Test that product yield data is preserved."""
        data = neutron_data

        with tempfile.TemporaryDirectory() as tmpdir:
            arrow_path = Path(tmpdir) / f"{data.name}.arrow"
            export_neutron_to_arrow(data, arrow_path)
            arrow_data = read_neutron_from_arrow(arrow_path)

            prod_idx = 0
            for rx in data.reactions.values():
                for p in rx.products:
                    if prod_idx >= len(arrow_data["products"]):
                        break
                    ap = arrow_data["products"][prod_idx]
                    assert ap["particle"] == str(p.name)
                    assert ap["emission_mode"] == str(p.emission_mode)
                    assert ap["decay_rate"] == float(p.decay_rate)
                    prod_idx += 1

    def test_fast_xs_data(self, neutron_data):
        """Test that FastXSGrid data is present and consistent."""
        data = neutron_data

        with tempfile.TemporaryDirectory() as tmpdir:
            arrow_path = Path(tmpdir) / f"{data.name}.arrow"
            export_neutron_to_arrow(data, arrow_path)
            arrow_data = read_neutron_from_arrow(arrow_path)

            assert "fast_xs" in arrow_data
            assert len(arrow_data["fast_xs"]) == len(data.temperatures)

            for fxs in arrow_data["fast_xs"]:
                # Check log_grid_index size
                assert len(fxs["log_grid_index"]) == 8001

                # Check log_grid_index is non-decreasing
                lgi = np.array(fxs["log_grid_index"])
                assert np.all(lgi[1:] >= lgi[:-1])

                # Check xs consistency
                xs = np.array(fxs["xs"]).reshape(fxs["xs_shape"])
                total = xs[:, 0]
                absorption = xs[:, 1]
                scattering = xs[:, 2]
                fission = xs[:, 3]
                np.testing.assert_allclose(
                    total, absorption + scattering + fission,
                    rtol=1e-10, atol=1e-30)

    def test_version_json(self, neutron_data):
        """Test that version.json contains expected fields."""
        data = neutron_data

        with tempfile.TemporaryDirectory() as tmpdir:
            arrow_path = Path(tmpdir) / f"{data.name}.arrow"
            export_neutron_to_arrow(data, arrow_path, library="test-lib")

            version = json.loads((arrow_path / "version.json").read_text())
            assert version["format_version"] == 1
            assert version["library"] == "test-lib"
            assert "converter_version" in version
            assert "created_utc" in version

    def test_library_parameter(self, neutron_data):
        """Test that library parameter is written to version.json."""
        data = neutron_data

        with tempfile.TemporaryDirectory() as tmpdir:
            arrow_path = Path(tmpdir) / f"{data.name}.arrow"
            export_neutron_to_arrow(data, arrow_path, library="fendl-3.2c")
            arrow_data = read_neutron_from_arrow(arrow_path)
            assert arrow_data["version"]["library"] == "fendl-3.2c"
