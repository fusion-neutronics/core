import yamc
import pytest
from yamc import Material

def test_macroscopic_xs_neutron():

    yamc.cross_section_data = ({"Li6": "tests/Li6.arrow", "Li7": "tests/Li7.arrow"})
    # Create a material with Li6 and Li7
    material = Material(
        composition={"Li6": 0.5, "Li7": 0.5},
        density=1.0,
        temperature=294,
    )

    # Get the unified energy grid - this will cache it
    grid = material.unified_energy_grid_neutron()

    # Calculate microscopic cross sections for all MT numbers using the cached grid
    micro_xs = material.calculate_microscopic_xs_neutron()
    print(micro_xs.keys())
    # Verify that we have cross sections for both nuclides
    assert "Li6" in micro_xs, "No microscopic cross sections for Li6"
    assert "Li7" in micro_xs, "No microscopic cross sections for Li7"

    # Calculate macroscopic cross section for MT=2
    macro_xs_mt2, energy = material.macroscopic_cross_section(2)

    # Verify the length of the macroscopic cross section array
    assert len(macro_xs_mt2) == len(grid), "Macro XS length doesn't match grid length"
    assert all(xs >= 0 for xs in macro_xs_mt2), "Negative cross section values found"

def test_macroscopic_xs_with_atoms_per_barn_cm():
    # Create a material with Li isotopes that have defined atomic masses
    yamc.cross_section_data = ({"Li6": "tests/Li6.arrow", "Li7": "tests/Li7.arrow"})
    material = Material(
        composition={"Li6": 0.5, "Li7": 0.5},
        density=1.0,
        temperature=294,
    )

    # Calculate macroscopic cross section (this loads nuclide data)
    macro_xs_mt1, energy = material.macroscopic_cross_section(1)

    # Get the atoms per cc (nuclide data must be loaded first for AWR)
    atoms_per_cc = material.get_atoms_per_barn_cm()

    # Verify that macroscopic cross sections were calculated
    assert len(macro_xs_mt1) > 0, "No macroscopic cross sections were calculated"

    # Test the relationship between density and macroscopic XS
    # If we double the density, atoms per cc should double, and so should macroscopic XS
    material2 = Material(
        composition={"Li6": 0.5, "Li7": 0.5},
        density=2.0,
        temperature=294,
    )
    material2.read_nuclear_data({"Li6": "tests/Li6.arrow", "Li7": "tests/Li7.arrow"})
    atoms_per_cc_doubled = material2.get_atoms_per_barn_cm()
    macro_xs_doubled, _ = material2.macroscopic_cross_section(1)

    # Check that atoms per cc doubled (with tolerance for AWR-derived atomic mass differences)
    for nuclide in atoms_per_cc.keys():
        assert atoms_per_cc_doubled[nuclide] == pytest.approx(2 * atoms_per_cc[nuclide], rel=1e-3)

    # Check that macroscopic XS doubled
    for i in range(len(macro_xs_mt1)):
        if abs(macro_xs_mt1[i]) > 1e-10:
            assert macro_xs_doubled[i] == pytest.approx(2 * macro_xs_mt1[i], rel=1e-6)

def test_macroscopic_xs_calculation_formula():
    # Create a test material with nuclides that have defined atomic masses
    yamc.cross_section_data = ({"Li6": "tests/Li6.arrow", "Li7": "tests/Li7.arrow"})
    material = Material(
        composition={"Li6": 1.0},  # Using a single nuclide for simplicity
        density=1.0,
        temperature=294,
    )

    # Get microscopic cross sections
    micro_xs = material.calculate_microscopic_xs_neutron()

    # Get atoms per cc
    atoms_per_cc = material.get_atoms_per_barn_cm()

    # Calculate macroscopic cross section for MT=2
    macro_xs_mt2, _ = material.macroscopic_cross_section(2)

    # Check for MT=2 (elastic scattering)
    assert "Li6" in micro_xs, "Li6 not present in micro_xs"
    assert 2 in micro_xs["Li6"], "MT=2 not present in micro_xs['Li6']"
    for i in range(min(10, len(macro_xs_mt2))):
        expected = atoms_per_cc["Li6"] * micro_xs["Li6"][2][i]
        assert macro_xs_mt2[i] == pytest.approx(expected, rel=1e-6), \
            f"Macroscopic XS calculation incorrect at index {i}"
