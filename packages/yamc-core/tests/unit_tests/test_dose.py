"""Tests for yamc dose coefficient functions and EnergyFunctionFilter."""

import pytest

import yamc


class TestDoseCoefficients:
    """Tests for yamc.data.dose_coefficients function."""

    def test_basic_neutron_dose_coefficients(self):
        """Test getting neutron dose coefficients."""
        _dc = yamc.data.dose_coefficients("neutron", "AP")
        energy, coeffs = _dc.energy, _dc.coefficients

        assert len(energy) > 0
        assert len(energy) == len(coeffs)
        assert all(e > 0 for e in energy)
        assert all(c > 0 for c in coeffs)

    def test_all_geometries(self):
        """Test all irradiation geometries."""
        geometries = ["AP", "PA", "LLAT", "RLAT", "ROT", "ISO"]

        for geom in geometries:
            _dc = yamc.data.dose_coefficients("neutron", geom)
            energy, coeffs = _dc.energy, _dc.coefficients
            assert len(energy) > 0
            assert len(coeffs) == len(energy)

    def test_both_data_sources(self):
        """Test ICRP-74 and ICRP-116 data sources."""
        for source in ["icrp74", "icrp116"]:
            energy = yamc.data.dose_coefficients("neutron", "AP", source).energy
            assert len(energy) > 0

    def test_energy_in_ev(self):
        """Test that energy is returned in eV (not MeV)."""
        energy = yamc.data.dose_coefficients("neutron", "AP").energy

        # Lowest energy in ICRP-116 is 1e-9 MeV = 1e-3 eV
        # So energy[0] should be small but > 0
        assert energy[0] < 1.0  # Less than 1 eV
        assert energy[0] > 0

        # Highest energy should be in MeV range (converted to eV)
        assert energy[-1] > 1e6  # Greater than 1 MeV in eV

    def test_photon_dose_coefficients(self):
        """ICRP-116 photon coefficients: 55 points from 10 keV to 10 GeV."""
        _dc = yamc.data.dose_coefficients("photon", "AP")
        energy, coeffs = _dc.energy, _dc.coefficients

        assert len(energy) == 55
        assert energy[0] == pytest.approx(1e4)
        assert energy[-1] == pytest.approx(1e10)
        # ICRP-116 Table A.1, AP geometry.
        assert coeffs[0] == pytest.approx(0.0685)

    def test_photon_and_neutron_coefficients_differ(self):
        """The two particles read different tables, not the same one twice."""
        photon = yamc.data.dose_coefficients("photon", "AP").coefficients
        neutron = yamc.data.dose_coefficients("neutron", "AP").coefficients
        assert len(photon) != len(neutron)

    def test_invalid_particle(self):
        """Test error handling for invalid particle."""
        with pytest.raises(ValueError, match="not supported"):
            yamc.data.dose_coefficients("proton", "AP")

    def test_invalid_geometry(self):
        """Test error handling for invalid geometry."""
        with pytest.raises(ValueError, match="Invalid geometry"):
            yamc.data.dose_coefficients("neutron", "INVALID")

    def test_invalid_data_source(self):
        """Test error handling for invalid data source."""
        with pytest.raises(ValueError, match="Invalid data_source"):
            yamc.data.dose_coefficients("neutron", "AP", "invalid")

    def test_case_insensitivity(self):
        """Test that geometry and data_source are case-insensitive."""
        # These should all work
        yamc.data.dose_coefficients("neutron", "ap")
        yamc.data.dose_coefficients("neutron", "AP")
        yamc.data.dose_coefficients("neutron", "Ap")
        yamc.data.dose_coefficients("neutron", "AP", "ICRP116")
        yamc.data.dose_coefficients("neutron", "AP", "icrp116")


class TestDoseTallyIntegration:
    """Integration tests for dose tallies."""

    def test_tally_with_energy_function_filter(self):
        """Test creating a tally with energy_function kwarg."""
        # Create geometry (auto-assigns cell IDs)
        sphere = yamc.Sphere(x0=0, y0=0, z0=0, radius=10, boundary="vacuum")
        cell = yamc.Cell(region=sphere.below)
        yamc.Geometry([cell])

        # Create dose coefficients
        _dc = yamc.data.dose_coefficients("neutron", "AP")
        energy, coeffs = _dc.energy, _dc.coefficients

        # Create tally with cells and energy_function kwargs
        tally = yamc.Tally(
            scores=["flux"],
            name="neutron_dose",
            cells=cell,
            energy_function=(energy, coeffs))

        # Verify setup
        assert tally.cells == [cell.id]
        assert tally.energy_function is not None

    def test_multiple_filter_types(self):
        """Test tally with mesh and energy function."""
        # Create geometry
        sphere = yamc.Sphere(x0=0, y0=0, z0=0, radius=50, boundary="vacuum")
        yamc.Cell(region=sphere.below)

        # Create mesh
        mesh = yamc.RegularRectangularMesh(
            lower_left=[-50, -50, -50], upper_right=[50, 50, 50], shape=[5, 5, 5]
        )

        # Create dose coefficients
        _dc = yamc.data.dose_coefficients("neutron", "AP")
        energy, coeffs = _dc.energy, _dc.coefficients

        # Create tally with mesh and energy_function kwargs
        tally = yamc.Tally(
            scores=["flux"],
            mesh=mesh,
            energy_function=(energy, coeffs))

        assert tally.mesh is not None
        assert tally.energy_function is not None


class TestDoseCoefficientsValues:
    """Tests to verify actual dose coefficient values."""

    def test_icrp116_has_68_points(self):
        """ICRP-116 should have 68 neutron energy points."""
        _dc = yamc.data.dose_coefficients("neutron", "AP", "icrp116")
        energy, coeffs = _dc.energy, _dc.coefficients
        assert len(energy) == 68
        assert len(coeffs) == 68

    def test_icrp74_has_47_points(self):
        """ICRP-74 should have 47 neutron energy points."""
        _dc = yamc.data.dose_coefficients("neutron", "AP", "icrp74")
        energy, coeffs = _dc.energy, _dc.coefficients
        assert len(energy) == 47
        assert len(coeffs) == 47

    def test_energy_monotonically_increasing(self):
        """Energy grid should be monotonically increasing."""
        energy = yamc.data.dose_coefficients("neutron", "AP", "icrp116").energy
        for i in range(1, len(energy)):
            assert energy[i] > energy[i - 1]

    def test_ap_vs_pa_different(self):
        """AP and PA geometries should have different coefficients."""
        ap_coeffs = yamc.data.dose_coefficients("neutron", "AP", "icrp116").coefficients
        pa_coeffs = yamc.data.dose_coefficients("neutron", "PA", "icrp116").coefficients

        # At least some values should be different
        different = any(
            abs(ap - pa) > 1e-10 for ap, pa in zip(ap_coeffs, pa_coeffs, strict=False)
        )
        assert different, "AP and PA should have different coefficients"


class TestPhotonCoefficients:
    """Tests for the NIST photon attenuation and energy-absorption tables."""

    def test_mass_attenuation_by_symbol_and_by_atomic_number(self):
        """'Fe' and 26 name the same element, so they read the same table."""
        by_symbol = yamc.data.mass_attenuation_coefficient("Fe")
        by_number = yamc.data.mass_attenuation_coefficient(26)

        assert by_symbol.energy == by_number.energy
        assert by_symbol.coefficients == by_number.coefficients
        assert by_symbol.units == "cm2/g"

    def test_mass_attenuation_matches_the_published_table(self):
        """NIST XCOM: iron is 0.05995 cm2/g at 1 MeV, where Compton dominates."""
        iron = yamc.data.mass_attenuation_coefficient("Fe")

        assert iron.interpolate(1.0e6) == pytest.approx(0.05995)
        assert min(iron.energy) == pytest.approx(1.0e3)
        assert max(iron.energy) == pytest.approx(2.0e7)

    def test_mass_attenuation_jumps_at_an_absorption_edge(self):
        """Iron's K edge at 7112 eV lifts mu/rho by about 7.7x."""
        iron = yamc.data.mass_attenuation_coefficient("Fe")

        assert iron.interpolate(7113.0) / iron.interpolate(7111.0) > 7.0

    def test_mass_attenuation_covers_hydrogen_to_fermium(self):
        for z in (1, 26, 92, 100):
            assert len(yamc.data.mass_attenuation_coefficient(z).energy) > 0

        with pytest.raises(ValueError, match="Z = 1 to 100"):
            yamc.data.mass_attenuation_coefficient(101)

    def test_mass_attenuation_rejects_nonsense(self):
        with pytest.raises(ValueError, match="not a recognized element symbol"):
            yamc.data.mass_attenuation_coefficient("Xx")

        with pytest.raises(TypeError, match="element symbol"):
            yamc.data.mass_attenuation_coefficient(1.5)

    def test_air_energy_absorption_matches_the_published_table(self):
        """NIST SRD 126 Table 4: air is 0.02789 cm2/g at 1 MeV."""
        air = yamc.data.mass_energy_absorption_coefficient("air")

        assert air.interpolate(1.0e6) == pytest.approx(0.02789)
        assert air.units == "cm2/g"

    def test_air_is_the_only_tabulated_material(self):
        with pytest.raises(ValueError, match="Only 'air' is tabulated"):
            yamc.data.mass_energy_absorption_coefficient("water")

    def test_energy_absorption_is_below_attenuation(self):
        """Absorption counts deposited energy, attenuation counts photons
        removed, so the absorption coefficient is the smaller of the two."""
        air = yamc.data.mass_energy_absorption_coefficient("air")
        oxygen = yamc.data.mass_attenuation_coefficient("O")

        assert air.interpolate(1.0e6) < oxygen.interpolate(1.0e6)

    def test_as_energy_function_round_trips_into_a_tally(self):
        air = yamc.data.mass_energy_absorption_coefficient("air")
        energy, coefficients, units = air.as_energy_function()

        assert len(energy) == len(coefficients)
        assert units == "cm2/g"
        tally = yamc.Tally(scores=["flux"], energy_function=(energy, coefficients, units))
        assert tally is not None

    def test_interpolating_outside_the_grid_clamps(self):
        air = yamc.data.mass_energy_absorption_coefficient("air")

        assert air.interpolate(1.0) == air.interpolate(min(air.energy))
        assert air.interpolate(1.0e12) == air.interpolate(max(air.energy))
