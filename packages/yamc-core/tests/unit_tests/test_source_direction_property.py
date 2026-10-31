#!/usr/bin/env python3
"""
Test getter/setter functionality specifically for NeutronSource.direction property.
"""
import yamc


def test_direction_getter_setter_consistency():
    """Test that direction getter/setter work consistently."""
    source = yamc.NeutronSource()

    # Default should be Isotropic
    direction = source.direction
    assert isinstance(direction, yamc.sources.Isotropic)
    assert str(direction) == "Isotropic()"

    # Set to Monodirectional and verify getter returns correct object
    mono = yamc.sources.Monodirectional([1.0, 0.0, 0.0])
    source.direction = mono

    retrieved_direction = source.direction
    assert isinstance(retrieved_direction, yamc.sources.Monodirectional)
    assert retrieved_direction.direction == [1.0, 0.0, 0.0]
    assert str(retrieved_direction) == "Monodirectional(direction=[1.0, 0.0, 0.0])"

    # Set to Isotropic and verify
    iso = yamc.sources.Isotropic()
    source.direction = iso

    retrieved_direction = source.direction
    assert isinstance(retrieved_direction, yamc.sources.Isotropic)
    assert str(retrieved_direction) == "Isotropic()"


def test_direction_property_consistency():
    """Test that direction property works consistently across multiple sources."""
    source1 = yamc.NeutronSource()
    source2 = yamc.NeutronSource()

    # Set same direction on both using property
    mono = yamc.sources.Monodirectional([0.0, 1.0, 0.0])
    source1.direction = mono
    source2.direction = mono

    # Both should produce same samples
    p1 = source1.sample()
    p2 = source2.sample()
    assert p1.direction == p2.direction

    # And getters should return equivalent objects
    direction1 = source1.direction
    direction2 = source2.direction
    assert isinstance(direction1, yamc.sources.Monodirectional)
    assert isinstance(direction2, yamc.sources.Monodirectional)
    assert direction1.direction == direction2.direction


def test_direction_getter_returns_independent_objects():
    """Test that direction getter returns independent objects (not shared references)."""
    source = yamc.NeutronSource()
    source.direction = yamc.sources.Monodirectional([1.0, 0.0, 0.0])

    # Get direction object twice
    direction1 = source.direction
    direction2 = source.direction

    # Should be equivalent but independent objects
    assert direction1.direction == direction2.direction
    assert direction1 is not direction2  # Different objects

    # Modifying one shouldn't affect the other or the source
    direction1.direction = [0.0, 1.0, 0.0]

    # Source and direction2 should be unchanged
    assert source.direction.direction == [1.0, 0.0, 0.0]
    assert direction2.direction == [1.0, 0.0, 0.0]


def test_all_properties_have_getters_setters():
    """Test that all properties consistently have both getters and setters."""
    source = yamc.NeutronSource()

    # Test position property
    assert hasattr(source, 'position')  # getter
    _original_position = source.position  # noqa: F841
    source.position = [1.0, 2.0, 3.0]  # setter
    assert isinstance(source.position, list)
    assert source.position == [1.0, 2.0, 3.0]

    # Test energy property
    assert hasattr(source, 'energy')  # getter
    _original_energy = source.energy  # noqa: F841
    source.energy = yamc.sources.Discrete([2e6], [1.0])  # setter
    assert source.energy.energies == [2e6]
    assert source.energy.probabilities == [1.0]

    # Test direction property
    assert hasattr(source, 'direction')  # getter
    _original_direction = source.direction  # noqa: F841
    source.direction = yamc.sources.Monodirectional([0.0, 0.0, 1.0])  # setter
    retrieved_direction = source.direction
    assert isinstance(retrieved_direction, yamc.sources.Monodirectional)
    assert retrieved_direction.direction == [0.0, 0.0, 1.0]
