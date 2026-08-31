import math

import pytest
import yamc as mmc


def test_tally_cells_kwarg():
    """Test that Tally can be created with the cells keyword argument."""
    sphere = mmc.Sphere(surface_id=1, x0=0.0, y0=0.0, z0=0.0, radius=2.0, boundary='vacuum')
    region = sphere.below

    material = mmc.Material(
        composition={"Li6": 1.0},
        density=1.0, units="g/cc")

    cell = mmc.Cell(region, 42, "test_cell", material)

    tally = mmc.Tally(cells=cell)

    assert tally.cells is not None
    assert tally.cells == [42]


def test_tally_cells_kwarg_with_material_id():
    """Test Tally cells kwarg with a material that has an ID."""
    sphere = mmc.Sphere(surface_id=1, x0=0.0, y0=0.0, z0=0.0, radius=2.0, boundary='vacuum')
    region = sphere.below

    material = mmc.Material(
        composition={"Li6": 1.0},
        density=1.0, units="g/cc",
        name="test_material",
        id=123)

    cell = mmc.Cell(region, 42, "test_cell_with_material_id", material)

    tally = mmc.Tally(cells=cell)
    assert tally.cells == [42]


def test_tally_cells_kwarg_invalid_input():
    """Test Tally cells kwarg behavior with invalid inputs."""
    with pytest.raises((TypeError, AttributeError)):
        mmc.Tally(cells="not_a_cell")


def test_tally_materials_kwarg():
    """Test that Tally can be created with the materials keyword argument."""
    material = mmc.Material(
        composition={"Li6": 1.0},
        density=1.0, units="g/cc",
        name="test_material",
        id=123)

    tally = mmc.Tally(materials=material)

    assert tally.materials is not None
    assert tally.materials == [123]


def test_tally_materials_kwarg_no_id_fails():
    """Test Tally materials kwarg fails with a material that has no ID."""
    material = mmc.Material(
        composition={"Li6": 1.0},
        density=1.0, units="g/cc",
        name="test_material")

    with pytest.raises(ValueError, match="no ID"):
        mmc.Tally(materials=material)


def test_tally_materials_kwarg_readback():
    """Test Tally materials kwarg round-trips correctly."""
    material_with_id = mmc.Material(
        composition={"Li6": 1.0},
        density=1.0, units="g/cc",
        name="test_material",
        id=123)
    tally = mmc.Tally(materials=material_with_id)

    assert tally.materials == [123]


def test_tally_with_materials_kwarg_and_scores():
    """Test that Tally can use materials kwarg with scores."""
    material = mmc.Material(
        composition={"Li6": 1.0},
        density=1.0, units="g/cc",
        name="test_material",
        id=123)

    tally = mmc.Tally(materials=material, scores=[101])

    assert tally.materials == [123]


def test_tally_cells_and_materials_mutually_exclusive():
    """Test that cells and materials kwargs cannot be used together."""
    sphere = mmc.Sphere(surface_id=1, x0=0.0, y0=0.0, z0=0.0, radius=2.0, boundary='vacuum')
    region = sphere.below
    material = mmc.Material(
        composition={"Li6": 1.0},
        density=1.0, units="g/cc",
        id=1)
    cell = mmc.Cell(region, 42, "test_cell", material)

    with pytest.raises(ValueError, match="mutually exclusive"):
        mmc.Tally(cells=cell, materials=material)


def test_tally_energy_bins_kwarg():
    """Test that Tally can be created with energy_bins keyword argument."""
    bins = [0.0, 1e6, 10e6, 20e6]
    tally = mmc.Tally(energy_bins=bins)

    assert tally.energy_bins is not None
    assert len(tally.energy_bins) == 4  # 4 boundaries = 3 bins


def test_tally_energy_bins_from_list():
    """Test that Tally energy_bins can be set from a list."""
    bins = [0.0, 1e3, 100e3, 1e6, 10e6, 20e6]
    tally = mmc.Tally(energy_bins=bins)

    assert tally.energy_bins is not None
    assert len(tally.energy_bins) == 6  # 6 boundaries = 5 bins


def test_tally_energy_bins_logspace():
    """Test Tally energy_bins with logarithmically spaced bins (like in flux.py example)."""
    lo, hi, n = math.log10(0.1), math.log10(20e6), 50
    bins = [10 ** (lo + i * (hi - lo) / (n - 1)) for i in range(n)]
    tally = mmc.Tally(energy_bins=bins)

    retrieved_bins = tally.energy_bins
    assert len(retrieved_bins) == 50
    assert retrieved_bins[0] == pytest.approx(0.1, rel=1e-10)
    assert retrieved_bins[-1] == pytest.approx(20e6, rel=1e-3)

    for i in range(1, len(retrieved_bins)):
        assert retrieved_bins[i] > retrieved_bins[i-1], \
            f"Bins should be strictly increasing: bins[{i}]={retrieved_bins[i]} should be > bins[{i-1}]={retrieved_bins[i-1]}"


def test_tally_energy_bins_validation():
    """Test that Tally energy_bins validates input properly."""
    with pytest.raises(ValueError, match="at least 2"):
        mmc.Tally(energy_bins=[1e6])

    with pytest.raises(ValueError, match="strictly ascending order"):
        mmc.Tally(energy_bins=[1e6, 10e6, 5e6])

    with pytest.raises(ValueError, match="strictly ascending order"):
        mmc.Tally(energy_bins=[1e6, 10e6, 10e6, 20e6])


def test_tally_energy_bins_readback():
    """Test Tally energy_bins round-trips correctly."""
    bins = [0.0, 1e6, 10e6, 20e6]
    tally = mmc.Tally(energy_bins=bins)

    retrieved = tally.energy_bins
    assert retrieved is not None
    assert len(retrieved) == len(bins)
    for expected, actual in zip(bins, retrieved):
        assert actual == pytest.approx(expected)


def test_tally_with_energy_bins_and_cells():
    """Test that Tally can use energy_bins with cells kwarg."""
    sphere = mmc.Sphere(surface_id=1, x0=0.0, y0=0.0, z0=0.0, radius=2.0, boundary='vacuum')
    region = sphere.below
    cell = mmc.Cell(region, 42, "test_cell")

    bins = [0.0, 1e3, 100e3, 1e6, 10e6, 20e6]
    tally = mmc.Tally(cells=cell, energy_bins=bins, scores=['flux'])

    assert tally.cells == [42]
    assert tally.energy_bins is not None
    assert len(tally.energy_bins) == 6


def test_tally_energy_groups_kwarg():
    """Test creating Tally with energy_group_structure='VITAMIN-J-175'."""
    tally = mmc.Tally(energy_group_structure='VITAMIN-J-175')

    bins = tally.energy_bins
    assert bins is not None
    assert len(bins) == 176  # 175 bins = 176 boundaries
    assert bins[0] == 1e-5
    assert bins[-1] == pytest.approx(1.964e7)

    for i in range(1, len(bins)):
        assert bins[i] > bins[i-1], \
            f"Bins should be strictly increasing: bins[{i}]={bins[i]} should be > bins[{i-1}]={bins[i-1]}"


def test_tally_energy_groups_invalid():
    """Test that invalid energy_group_structure raises ValueError."""
    with pytest.raises(ValueError, match="Unknown group structure"):
        mmc.Tally(energy_group_structure='INVALID-STRUCTURE')

    try:
        mmc.Tally(energy_group_structure='NONEXISTENT')
    except ValueError as e:
        error_msg = str(e)
        assert 'NONEXISTENT' in error_msg
        assert 'VITAMIN-J-175' in error_msg


def test_tally_energy_groups_case_sensitive():
    """Test that energy_group_structure names are case-sensitive."""
    with pytest.raises(ValueError):
        mmc.Tally(energy_group_structure='vitamin-j-175')

    with pytest.raises(ValueError):
        mmc.Tally(energy_group_structure='VITAMIN-j-175')


def test_tally_energy_groups_with_cells():
    """Test using energy_group_structure in a tally with a cells kwarg."""
    sphere = mmc.Sphere(surface_id=1, x0=0.0, y0=0.0, z0=0.0, radius=2.0, boundary='vacuum')
    region = sphere.below
    cell = mmc.Cell(region, 42, "test_cell")

    tally = mmc.Tally(cells=cell, energy_group_structure='VITAMIN-J-175', scores=['flux'])

    assert tally.cells == [42]
    bins = tally.energy_bins
    assert bins is not None
    assert len(bins) == 176


def test_tally_energy_bins_and_groups_mutually_exclusive():
    """Test that energy_bins and energy_group_structure cannot be used together."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        mmc.Tally(energy_bins=[0.0, 1e6, 20e6], energy_group_structure='VITAMIN-J-175')


def test_tally_energy_bins_lethargy_bin_width():
    """Test lethargy_bin_width computation from energy_bins on a Tally.

    lethargy_bin_width = log10(E_high / E_low) for each bin.
    """
    bins = [1.0, 10.0, 100.0, 1000.0]
    tally = mmc.Tally(energy_bins=bins)
    retrieved = tally.energy_bins
    assert retrieved is not None

    widths = [math.log10(retrieved[i+1] / retrieved[i]) for i in range(len(retrieved) - 1)]
    assert len(widths) == 3
    for w in widths:
        assert abs(w - 1.0) < 1e-12

    tally175 = mmc.Tally(energy_group_structure='VITAMIN-J-175')
    bins175 = tally175.energy_bins
    widths175 = [math.log10(bins175[i+1] / bins175[i]) for i in range(len(bins175) - 1)]
    assert len(widths175) == 175
    assert all(w > 0 for w in widths175)


def test_tally_particle_kwarg():
    """Test that Tally can be created with the particle keyword argument."""
    tally = mmc.Tally(particle="photon")
    assert tally.particle == "photon"

    tally2 = mmc.Tally(particle="neutron")
    assert tally2.particle == "neutron"


def test_tally_parent_nuclides_kwarg():
    """Test that Tally can be created with the parent_nuclides keyword argument."""
    nuclides = ["Co60", "Mn56", "Fe59"]
    tally = mmc.Tally(parent_nuclides=nuclides)
    assert tally.parent_nuclides is not None
    assert tally.parent_nuclides == nuclides


def test_tally_no_filters_returns_none():
    """Test that Tally properties return None when no filters are set."""
    tally = mmc.Tally()
    assert tally.cells is None
    assert tally.materials is None
    assert tally.mesh is None
    assert tally.energy_bins is None
    assert tally.energy_function is None
    assert tally.particle is None
    assert tally.parent_nuclides is None
