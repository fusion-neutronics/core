"""
Tests for MPI support detection and functionality.

These tests verify:
1. has_mpi() correctly reports MPI availability
2. mpi_rank() and mpi_size() return sensible values
3. Warning messages appear when running under mpirun without MPI support
"""

import pytest
import yamc
import os
import subprocess
import sys


def test_has_mpi_is_boolean():
    """Test that has_mpi() returns a boolean value."""
    result = yamc.parallel.has_mpi()
    assert isinstance(result, bool), "has_mpi() should return a boolean"


def test_mpi_rank_returns_int():
    """Test that mpi_rank() returns an integer."""
    result = yamc.parallel.mpi_rank()
    assert isinstance(result, int), "mpi_rank() should return an integer"
    assert result >= 0, "mpi_rank() should return a non-negative integer"


def test_mpi_size_returns_int():
    """Test that mpi_size() returns an integer."""
    result = yamc.parallel.mpi_size()
    assert isinstance(result, int), "mpi_size() should return an integer"
    assert result >= 1, "mpi_size() should return at least 1"


def test_mpi_consistency():
    """Test that MPI rank and size are consistent."""
    rank = yamc.parallel.mpi_rank()
    size = yamc.parallel.mpi_size()

    # Rank should always be less than size
    assert rank < size, f"Rank {rank} should be less than size {size}"

    # In single-process mode (no MPI or MPI disabled)
    if size == 1:
        assert rank == 0, "In single-process mode, rank should be 0"


def test_mpi_disabled_defaults():
    """Test that without MPI, we get sensible defaults."""
    if not yamc.parallel.has_mpi():
        assert yamc.parallel.mpi_rank() == 0, "Without MPI, rank should be 0"
        assert yamc.parallel.mpi_size() == 1, "Without MPI, size should be 1"


def test_mpi_enabled_detection():
    """Test MPI detection when running under mpirun."""
    # Check if we're running under MPI based on environment variables
    is_mpi_env = any(var in os.environ for var in
                     ['OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS'])

    if is_mpi_env and yamc.parallel.has_mpi():
        # If we're in MPI environment and MPI is enabled, size should be > 1
        size = yamc.parallel.mpi_size()
        rank = yamc.parallel.mpi_rank()

        # At least verify the values are in valid ranges
        assert rank >= 0, "MPI rank should be non-negative"
        assert size >= 1, "MPI size should be at least 1"
        assert rank < size, "MPI rank should be less than size"


@pytest.mark.skipif(yamc.parallel.has_mpi(), reason="Test only for non-MPI builds")
def test_mpi_warning_without_support():
    """Test that warning message appears when running under mpirun without MPI support."""
    # Check if mpirun is available
    try:
        subprocess.run(["mpirun", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("mpirun not available on this system")

    # Run a simple script under mpirun
    script = "import yamc; yamc.parallel.mpi_rank(); yamc.parallel.mpi_size()"
    result = subprocess.run(
        ["mpirun", "-np", "2", sys.executable, "-c", script],
        capture_output=True,
        text=True
    )

    # Check that warning message appears in stderr
    assert "WARNING" in result.stderr, "Expected warning message in stderr"
    assert "not built with MPI support" in result.stderr, "Expected specific warning about MPI not built"
    assert "maturin develop" in result.stderr, "Expected build instructions in warning"


@pytest.mark.skipif(not yamc.parallel.has_mpi(), reason="Test only for MPI builds")
def test_mpi_functions_with_support():
    """Test MPI functions when MPI support is compiled in."""
    assert yamc.parallel.has_mpi() is True, "MPI should be enabled"

    # These functions should work without error
    rank = yamc.parallel.mpi_rank()
    size = yamc.parallel.mpi_size()

    assert isinstance(rank, int), "mpi_rank() should return int"
    assert isinstance(size, int), "mpi_size() should return int"
    assert rank >= 0, "rank should be non-negative"
    assert size >= 1, "size should be at least 1"
    assert rank < size, "rank should be less than size"


def test_model_runs_without_mpi():
    """Test that Model.simulate_transport() works regardless of MPI support."""
    import os

    # Check if test nuclear data is available
    test_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'crates', 'yamc', 'tests')
    li6_path = os.path.join(test_dir, "Li6.arrow")
    if not os.path.exists(li6_path):
        # Try symlink path
        li6_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tests', 'Li6.arrow')
        if not os.path.exists(li6_path):
            pytest.skip("Test nuclear data not available")

    # Create a minimal model
    material = yamc.Material(
        composition={"Li6": 1.0},
        density=1.0,
        temperature=294)
    material.read_nuclear_data({"Li6": li6_path})

    # Simple sphere geometry
    sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=5.0, boundary='vacuum')
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])

    # Simple source
    source = yamc.NeutronSource(
        position=(0.0, 0.0, 0.0),
        energy=yamc.sources.Discrete([1.0e6], [1.0])
    )

    model = yamc.Model(geometry=geometry, source=source)

    # Should run without error regardless of MPI support
    results = model.simulate_transport(total_particles=100)
    assert results.particles_per_second > 0, "Model should have run successfully"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
