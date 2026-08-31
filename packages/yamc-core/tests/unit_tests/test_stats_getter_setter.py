#!/usr/bin/env python3
"""
Test getter/setter functionality for stats distributions.
"""
import yamc


def test_monodirectional_getter_setter():
    """Test that Monodirectional has both getter and setter for direction."""
    # Create with initial direction
    mono = yamc.sources.Monodirectional([1.0, 0.0, 0.0])
    assert mono.direction == [1.0, 0.0, 0.0]
    
    # Test setter
    mono.direction = [0.0, 1.0, 0.0]
    assert mono.direction == [0.0, 1.0, 0.0]
    
    # Test another direction
    mono.direction = [0.0, 0.0, 1.0]
    assert mono.direction == [0.0, 0.0, 1.0]
    
    # Test that samples return the set direction
    sample1 = mono.sample()
    sample2 = mono.sample()
    assert sample1 == sample2 == [0.0, 0.0, 1.0]


def test_isotropic_has_no_settable_properties():
    """Test that Isotropic doesn't have settable properties (as expected)."""
    iso = yamc.sources.Isotropic()
    
    # Should not have direction property
    assert not hasattr(iso, 'direction')
    
    # Should still sample properly
    sample1 = iso.sample()
    sample2 = iso.sample()
    assert len(sample1) == 3
    assert len(sample2) == 3
    # Samples should be normalized
    import math
    mag1 = math.sqrt(sum(x*x for x in sample1))
    mag2 = math.sqrt(sum(x*x for x in sample2))
    assert abs(mag1 - 1.0) < 1e-10
    assert abs(mag2 - 1.0) < 1e-10


def test_setter_consistency_with_samples():
    """Test that setting direction affects sampling consistently."""
    mono = yamc.sources.Monodirectional([1.0, 0.0, 0.0])
    
    # Initial samples should match direction
    for _ in range(5):
        sample = mono.sample()
        assert sample == mono.direction
    
    # Change direction and test again
    mono.direction = [-1.0, 0.0, 0.0]
    for _ in range(5):
        sample = mono.sample()
        assert sample == [-1.0, 0.0, 0.0]
        assert sample == mono.direction


if __name__ == "__main__":
    test_monodirectional_getter_setter()
    test_isotropic_has_no_settable_properties()
    test_setter_consistency_with_samples()
    print("All getter/setter tests passed!")