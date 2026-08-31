"""
Tests for RegularRectangularMesh and mesh tally functionality (kwarg API)
"""

import pytest
import yamc


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


# RegularRectangularMesh tests

def test_mesh_creation():
    """Test basic mesh creation"""
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-10.0, -10.0, -10.0],
        upper_right=[10.0, 10.0, 10.0],
        shape=[5, 5, 5]
    )
    assert mesh.num_bins == 125  # 5*5*5
    assert mesh.lower_left == [-10.0, -10.0, -10.0]
    assert mesh.upper_right == [10.0, 10.0, 10.0]
    assert mesh.shape == [5, 5, 5]


def test_mesh_width_calculation():
    """Test that mesh widths are calculated correctly"""
    mesh = yamc.RegularRectangularMesh(
        lower_left=[0.0, 0.0, 0.0],
        upper_right=[10.0, 20.0, 30.0],
        shape=[2, 4, 6]
    )
    assert mesh.width == [5.0, 5.0, 5.0]  # 10/2, 20/4, 30/6


def test_mesh_non_uniform_dimensions():
    """Test mesh with different dimensions in each direction"""
    mesh = yamc.RegularRectangularMesh(
        lower_left=[0.0, 0.0, 0.0],
        upper_right=[10.0, 10.0, 10.0],
        shape=[2, 3, 5]
    )
    assert mesh.num_bins == 30  # 2*3*5


def test_mesh_single_voxel():
    """Test mesh with single voxel"""
    mesh = yamc.RegularRectangularMesh(
        lower_left=[0.0, 0.0, 0.0],
        upper_right=[1.0, 1.0, 1.0],
        shape=[1, 1, 1]
    )
    assert mesh.num_bins == 1


# Mesh tally via kwarg tests

def test_mesh_tally_creation():
    """Test creating a tally with mesh kwarg"""
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-5.0, -5.0, -5.0],
        upper_right=[5.0, 5.0, 5.0],
        shape=[10, 10, 10]
    )
    tally = yamc.Tally(scores=["flux"], mesh=mesh)
    assert tally.n_mesh_bins == 1000  # 10*10*10


def test_mesh_tally_preserves_mesh_properties():
    """Test that tally preserves mesh properties"""
    mesh = yamc.RegularRectangularMesh(
        lower_left=[1.0, 2.0, 3.0],
        upper_right=[4.0, 5.0, 6.0],
        shape=[3, 3, 3]
    )
    tally = yamc.Tally(scores=["flux"], mesh=mesh)

    # Access the mesh property
    retrieved_mesh = tally.mesh
    assert retrieved_mesh.lower_left == [1.0, 2.0, 3.0]
    assert retrieved_mesh.upper_right == [4.0, 5.0, 6.0]
    assert retrieved_mesh.shape == [3, 3, 3]


# Mesh tally shape tests

@requires_keywords
def test_mesh_tally_bin_count_matches_mesh_dimensions():
    """Test that tally.mean length matches mesh.num_bins"""
    yamc.set_cross_section_data_entry('fendl-3.2d')

    # Create simple geometry
    sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=10.0, boundary='vacuum')
    region = sphere.below

    material = yamc.Material(
        composition={"H1": 1.0},
        density=0.001,
        temperature=294,
    )

    cell = yamc.Cell(name="sphere", region=region, material=material)
    geometry = yamc.Geometry([cell])

    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1]),
        position=(0, 0, 0)
    )

    # Create mesh with specific dimensions
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-5.0, -5.0, -5.0],
        upper_right=[5.0, 5.0, 5.0],
        shape=[3, 4, 5]  # 60 voxels
    )

    tally = yamc.Tally(scores=['flux'], mesh=mesh)

    model = yamc.Model(geometry=geometry, tallies=[tally], source=source)
    results = model.simulate_transport(total_particles=100, seed=42)

    # Check that tally results have correct shape
    expected_bins = 3 * 4 * 5  # 60
    result = results[tally]
    assert len(result.mean) == expected_bins
    assert len(result.standard_deviation) == expected_bins
    assert len(result.relative_error) == expected_bins


@requires_keywords
def test_mesh_tally_with_energy_filter_shape():
    """Test that mesh+energy_bins produces correct shape"""
    yamc.set_cross_section_data_entry('fendl-3.2d')

    # Create simple geometry
    sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=10.0, boundary='vacuum')
    region = sphere.below

    material = yamc.Material(
        composition={"H1": 1.0},
        density=0.001,
        temperature=294,
    )

    cell = yamc.Cell(name="sphere", region=region, material=material)
    geometry = yamc.Geometry([cell])

    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1]),
        position=(0, 0, 0)
    )

    # Create mesh
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-5.0, -5.0, -5.0],
        upper_right=[5.0, 5.0, 5.0],
        shape=[2, 2, 2]  # 8 voxels
    )

    tally = yamc.Tally(
        scores=['flux'],
        mesh=mesh,
        energy_bins=[0.0, 1e6, 10e6, 20e6],  # 3 energy bins
    )

    model = yamc.Model(geometry=geometry, tallies=[tally], source=source)
    results = model.simulate_transport(total_particles=100, seed=42)

    # Check that tally results have correct shape: energy_bins * mesh_bins
    expected_bins = 3 * 8  # 24
    result = results[tally]
    assert len(result.mean) == expected_bins
    assert len(result.standard_deviation) == expected_bins


@requires_keywords
def test_mesh_tally_multiple_scores_shape():
    """Test that mesh tally with multiple scores produces correct shape"""
    yamc.set_cross_section_data_entry('fendl-3.2d')

    # Create simple geometry
    sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=10.0, boundary='vacuum')
    region = sphere.below

    material = yamc.Material(
        composition={"H1": 1.0},
        density=0.001,
        temperature=294,
    )

    cell = yamc.Cell(name="sphere", region=region, material=material)
    geometry = yamc.Geometry([cell])

    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1]),
        position=(0, 0, 0)
    )

    # Create mesh
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-5.0, -5.0, -5.0],
        upper_right=[5.0, 5.0, 5.0],
        shape=[2, 2, 2]  # 8 voxels
    )

    tally = yamc.Tally(scores=['flux', 'heating'], mesh=mesh)  # 2 scores

    model = yamc.Model(geometry=geometry, tallies=[tally], source=source)
    results = model.simulate_transport(total_particles=100, seed=42)

    # Check that tally results have correct shape: num_scores * mesh_bins
    expected_bins = 2 * 8  # 16
    assert len(results[tally].mean) == expected_bins


# Mesh tally value tests

@requires_keywords
def test_mesh_tally_no_garbage_values():
    """Test that mesh tally doesn't produce garbage values"""
    yamc.set_cross_section_data_entry('fendl-3.2d')

    # Create simple geometry
    sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=10.0, boundary='vacuum')
    region = sphere.below

    material = yamc.Material(
        composition={"H1": 1.0},
        density=0.001,
        temperature=294,
    )

    cell = yamc.Cell(name="sphere", region=region, material=material)
    geometry = yamc.Geometry([cell])

    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1]),
        position=(0, 0, 0)
    )

    # Create mesh
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-5.0, -5.0, -5.0],
        upper_right=[5.0, 5.0, 5.0],
        shape=[2, 2, 2]  # 8 voxels
    )

    tally = yamc.Tally(scores=['flux'], mesh=mesh)

    model = yamc.Model(geometry=geometry, tallies=[tally], source=source)
    results = model.simulate_transport(total_particles=100, seed=42)

    # Check that all values are reasonable (not garbage)
    for i, mean_val in enumerate(results[tally].mean):
        # Flux values should be positive and not absurdly large
        # For this simple problem, flux should be O(1) or less
        assert mean_val >= 0.0, f"Bin {i} has negative flux: {mean_val}"
        assert mean_val < 1e10, f"Bin {i} has garbage value: {mean_val}"


@requires_keywords
def test_mesh_tally_all_bins_scored():
    """Test that particles score to multiple mesh bins"""
    yamc.set_cross_section_data_entry('fendl-3.2d')

    # Create simple geometry
    sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=10.0, boundary='vacuum')
    region = sphere.below

    material = yamc.Material(
        composition={"H1": 1.0},
        density=0.001,
        temperature=294,
    )

    cell = yamc.Cell(name="sphere", region=region, material=material)
    geometry = yamc.Geometry([cell])

    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1]),
        position=(0, 0, 0)
    )

    # Create mesh
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-8.0, -8.0, -8.0],
        upper_right=[8.0, 8.0, 8.0],
        shape=[2, 2, 2]  # 8 voxels
    )

    tally = yamc.Tally(scores=['flux'], mesh=mesh)

    # Run more particles to ensure we hit multiple voxels
    model = yamc.Model(geometry=geometry, tallies=[tally], source=source)
    results = model.simulate_transport(total_particles=1000, seed=42)

    # Check that multiple bins have non-zero scores
    # With 1000 particles from center, we should score to most/all voxels
    non_zero_bins = sum(1 for val in results[tally].mean if val > 0)
    assert non_zero_bins >= 4, f"Only {non_zero_bins} bins have non-zero scores, expected most bins to be scored"


@requires_keywords
def test_mesh_tally_center_voxel_highest_flux():
    """Test that center voxel has higher flux than outer voxels for point source"""
    yamc.set_cross_section_data_entry('fendl-3.2d')

    # Create simple geometry
    sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=20.0, boundary='vacuum')
    region = sphere.below

    material = yamc.Material(
        composition={"H1": 1.0},
        density=0.001,
        temperature=294,
    )

    cell = yamc.Cell(name="sphere", region=region, material=material)
    geometry = yamc.Geometry([cell])

    # Point source at origin
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1]),
        position=(0, 0, 0)
    )

    # Create 3x3x3 mesh centered at origin
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-15.0, -15.0, -15.0],
        upper_right=[15.0, 15.0, 15.0],
        shape=[3, 3, 3]  # 27 voxels
    )

    tally = yamc.Tally(scores=['flux'], mesh=mesh)

    model = yamc.Model(geometry=geometry, tallies=[tally], source=source)
    results = model.simulate_transport(total_particles=5000, seed=42)

    # Center voxel index: for 3x3x3, center is at (1,1,1)
    # Z-major ordering: bin = iz*ny*nx + iy*nx + ix = 1*3*3 + 1*3 + 1 = 13
    center_bin = 13
    mean = results[tally].mean
    center_flux = mean[center_bin]

    # Check that center has reasonably high flux
    max_flux = max(mean)

    # Center should be among the highest flux regions (at least 50% of max)
    assert center_flux > 0.5 * max_flux, \
        f"Center flux ({center_flux}) is too low compared to max ({max_flux})"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
