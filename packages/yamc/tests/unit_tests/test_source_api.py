"""
Comprehensive pytest tests for yamc Python API

Tests the NeutronSource and stats module functionality.
"""

import pytest
import yamc


class TestSource:
    """Test cases for NeutronSource class."""

    def test_parameterless_constructor(self):
        """Test that NeutronSource() works without arguments with correct defaults."""
        source = yamc.NeutronSource()

        # Check default values
        assert isinstance(source.position, list)
        assert source.position == [0.0, 0.0, 0.0]
        assert isinstance(source.energy, yamc.sources.Discrete)
        assert source.energy.energies == [14.06e6]
        assert source.energy.probabilities == [1.0]

        # Sample should work
        particle = source.sample()
        assert particle.position == [0.0, 0.0, 0.0]
        assert particle.energy == 14.06e6
        assert particle.alive is True

        # Direction should be normalized (isotropic by default)
        direction = particle.direction
        magnitude = sum(d * d for d in direction) ** 0.5
        assert abs(magnitude - 1.0) < 1e-10

    def test_space_property(self):
        """Test space property getter and setter."""
        source = yamc.NeutronSource()

        # Test getter
        assert isinstance(source.position, list)
        assert source.position == [0.0, 0.0, 0.0]

        # Test setter
        source.position = [1.0, 2.0, 3.0]
        assert isinstance(source.position, list)
        assert source.position == [1.0, 2.0, 3.0]

        # Test particle sampling reflects new space
        particle = source.sample()
        assert particle.position == [1.0, 2.0, 3.0]

    def test_energy_property(self):
        """Test energy property getter and setter."""
        source = yamc.NeutronSource()

        # Test getter (default)
        assert isinstance(source.energy, yamc.sources.Discrete)
        assert source.energy.energies == [14.06e6]

        # Test setter
        source.energy = yamc.sources.Discrete([2.5e6], [1.0])
        assert source.energy.energies == [2.5e6]
        assert source.energy.probabilities == [1.0]

        # Test particle sampling reflects new energy
        particle = source.sample()
        assert particle.energy == 2.5e6

    def test_set_angle_isotropic(self):
        """Test setting angle to isotropic using property."""
        source = yamc.NeutronSource()
        
        # Should be isotropic by default, but set it explicitly using property
        source.direction = yamc.sources.Isotropic()
        
        # Sample multiple particles and check they have different directions
        directions = []
        for _ in range(10):
            particle = source.sample()
            direction = particle.direction
            
            # Each direction should be normalized
            magnitude = sum(d * d for d in direction) ** 0.5
            assert abs(magnitude - 1.0) < 1e-10
            
            directions.append(tuple(direction))
        
        # With isotropic sampling, very unlikely to get duplicates
        unique_directions = set(directions)
        assert len(unique_directions) > 1

    def test_set_angle_monodirectional(self):
        """Test setting angle to monodirectional using property."""
        source = yamc.NeutronSource()
        
        # Set monodirectional using property
        reference_direction = [0.0, 0.0, 1.0]
        source.direction = yamc.sources.Monodirectional(reference_direction)
        
        # All sampled particles should have the same direction
        for _ in range(10):
            particle = source.sample()
            assert particle.direction == reference_direction

    def test_angle_switching(self):
        """Test switching between isotropic and monodirectional angles."""
        source = yamc.NeutronSource()
        
        # Start with monodirectional
        source.direction = yamc.sources.Monodirectional([1.0, 0.0, 0.0])
        particle1 = source.sample()
        assert particle1.direction == [1.0, 0.0, 0.0]
        
        # Switch to isotropic
        source.direction = yamc.sources.Isotropic()
        particle2 = source.sample()
        particle3 = source.sample()
        
        # Directions should be normalized but likely different
        for p in [particle2, particle3]:
            magnitude = sum(d * d for d in p.direction) ** 0.5
            assert abs(magnitude - 1.0) < 1e-10
        
        # Very unlikely to be the same
        assert particle2.direction != particle3.direction

    def test_repr(self):
        """Test string representation."""
        source = yamc.NeutronSource()
        repr_str = repr(source)

        # Should contain space and energy info
        assert "NeutronSource" in repr_str
        assert "position=(0, 0, 0)" in repr_str
        assert "Discrete" in repr_str
        assert "14060000" in repr_str


class TestIsotropic:
    """Test cases for Isotropic distribution."""

    def test_construction(self):
        """Test Isotropic construction."""
        iso = yamc.sources.Isotropic()
        assert iso is not None

    def test_sample(self):
        """Test Isotropic sampling."""
        iso = yamc.sources.Isotropic()
        
        # Sample multiple directions
        directions = []
        for _ in range(100):
            direction = iso.sample()
            
            # Should be 3D
            assert len(direction) == 3
            
            # Should be normalized
            magnitude = sum(d * d for d in direction) ** 0.5
            assert abs(magnitude - 1.0) < 1e-10
            
            directions.append(tuple(direction))
        
        # Should have many unique directions (isotropic)
        unique_directions = set(directions)
        assert len(unique_directions) > 50  # Conservative check for randomness

    def test_repr(self):
        """Test Isotropic string representation."""
        iso = yamc.sources.Isotropic()
        repr_str = repr(iso)
        assert "Isotropic" in repr_str


class TestMonodirectional:
    """Test cases for Monodirectional distribution."""

    def test_construction(self):
        """Test Monodirectional construction."""
        mono = yamc.sources.Monodirectional([1.0, 0.0, 0.0])
        assert mono is not None

    def test_sample_consistency(self):
        """Test that Monodirectional always returns the same direction."""
        reference = [0.0, 1.0, 0.0]
        mono = yamc.sources.Monodirectional(reference)
        
        # All samples should be identical
        for _ in range(10):
            direction = mono.sample()
            assert direction == reference

    def test_different_directions(self):
        """Test Monodirectional with different reference directions."""
        test_directions = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], 
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.5773502691896257, 0.5773502691896257, 0.5773502691896257]  # Normalized (1,1,1)
        ]
        
        for ref_dir in test_directions:
            mono = yamc.sources.Monodirectional(ref_dir)
            sample = mono.sample()
            assert sample == ref_dir

    def test_repr(self):
        """Test Monodirectional string representation."""
        mono = yamc.sources.Monodirectional([1.0, 0.0, 0.0])
        repr_str = repr(mono)
        assert "Monodirectional" in repr_str
        assert "[1.0, 0.0, 0.0]" in repr_str


class TestIntegration:
    """Integration tests for the complete API."""

    def test_source_workflow(self):
        """Test a complete source workflow."""
        # Create source with defaults
        source = yamc.NeutronSource()
        
        # Modify properties
        source.position = [0.0, 0.0, -10.0]
        source.energy = yamc.sources.Discrete([14.1e6], [1.0])
        
        # Set different angular distributions
        source.direction = yamc.sources.Monodirectional([0.0, 0.0, 1.0])
        particle1 = source.sample()
        assert particle1.position == [0.0, 0.0, -10.0]
        assert particle1.energy == 14.1e6
        assert particle1.direction == [0.0, 0.0, 1.0]
        
        # Switch to isotropic
        source.direction = yamc.sources.Isotropic()
        particle2 = source.sample()
        assert particle2.position == [0.0, 0.0, -10.0]
        assert particle2.energy == 14.1e6
        # Direction should be different (very likely)
        assert particle2.direction != [0.0, 0.0, 1.0]

    def test_direct_distribution_usage(self):
        """Test using distributions directly outside of source."""
        # Create distributions
        iso = yamc.sources.Isotropic()
        mono = yamc.sources.Monodirectional([1.0, 0.0, 0.0])
        
        # Sample from each
        iso_samples = [iso.sample() for _ in range(5)]
        mono_samples = [mono.sample() for _ in range(5)]
        
        # Isotropic should vary
        unique_iso = set(tuple(s) for s in iso_samples)
        assert len(unique_iso) > 1
        
        # Monodirectional should be constant
        unique_mono = set(tuple(s) for s in mono_samples)
        assert len(unique_mono) == 1
        assert mono_samples[0] == [1.0, 0.0, 0.0]

    def test_flexible_constructor(self):
        """Test the new flexible constructor with keyword arguments."""
        # Test all combinations of keyword arguments

        # Only space
        source1 = yamc.NeutronSource(position=[1.0, 2.0, 3.0])
        assert isinstance(source1.position, list)
        assert source1.position == [1.0, 2.0, 3.0]
        assert source1.energy.energies == [14.06e6]  # default

        # Only energy
        source2 = yamc.NeutronSource(energy=yamc.sources.Discrete([2e6], [1.0]))
        assert isinstance(source2.position, list)
        assert source2.position == [0.0, 0.0, 0.0]  # default
        assert source2.energy.energies == [2e6]

        # Only angle
        angle_dist = yamc.sources.Monodirectional([0.0, 0.0, 1.0])
        source3 = yamc.NeutronSource(direction=angle_dist)
        particle = source3.sample()
        assert isinstance(source3.position, list)
        assert source3.position == [0.0, 0.0, 0.0]  # default
        assert source3.energy.energies == [14.06e6]  # default
        assert particle.direction == [0.0, 0.0, 1.0]

        # Space and energy
        source4 = yamc.NeutronSource(position=[5.0, 0.0, -2.0], energy=yamc.sources.Discrete([3e6], [1.0]))
        assert isinstance(source4.position, list)
        assert source4.position == [5.0, 0.0, -2.0]
        assert source4.energy.energies == [3e6]

        # Energy and angle
        source5 = yamc.NeutronSource(
            energy=yamc.sources.Discrete([1e6], [1.0]),
            direction=yamc.sources.Monodirectional([1.0, 0.0, 0.0])
        )
        assert source5.energy.energies == [1e6]
        particle5 = source5.sample()
        assert particle5.direction == [1.0, 0.0, 0.0]

        # All three arguments (original failing case)
        source6 = yamc.NeutronSource(
            position=[0.0, 0.0, 0.0],
            direction=yamc.sources.Monodirectional([0.0, 0.0, 1.0]),
            energy=yamc.sources.Discrete([1e6], [1.0])
        )
        assert isinstance(source6.position, list)
        assert source6.position == [0.0, 0.0, 0.0]
        assert source6.energy.energies == [1e6]
        particle6 = source6.sample()
        assert particle6.direction == [0.0, 0.0, 1.0]

    def test_backwards_compatibility_failures(self):
        """Test that old positional constructor patterns fail appropriately."""
        # Old constructor with positional arguments should fail
        with pytest.raises(TypeError):
            yamc.NeutronSource([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1e6)


if __name__ == "__main__":
    pytest.main([__file__])