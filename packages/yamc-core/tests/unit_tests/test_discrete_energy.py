import pytest
import yamc


def test_discrete_construction():
    """Test creating Discrete energy distribution."""
    dist = yamc.sources.Discrete([14.06e6, 25e6], [0.9, 0.1])
    assert dist is not None
    assert dist.energies == [14.06e6, 25e6]
    assert dist.probabilities == [0.9, 0.1]


def test_discrete_validation_empty():
    """Test Discrete validation with empty energies."""
    with pytest.raises(ValueError, match="cannot be empty"):
        yamc.sources.Discrete([], [])


def test_discrete_validation_mismatched():
    """Test Discrete validation with mismatched lengths."""
    with pytest.raises(ValueError, match="must have same length"):
        yamc.sources.Discrete([1.0], [0.5, 0.5])


def test_discrete_validation_negative():
    """Test Discrete validation with negative probabilities."""
    with pytest.raises(ValueError, match="cannot be negative"):
        yamc.sources.Discrete([1.0], [-0.5])


def test_discrete_validation_all_zero():
    """Test Discrete validation with all zero probabilities."""
    with pytest.raises(ValueError, match="At least one probability"):
        yamc.sources.Discrete([1.0, 2.0], [0.0, 0.0])


def test_discrete_sampling():
    """Test that Discrete samples from correct distribution."""
    dist = yamc.sources.Discrete([1e6, 2e6, 3e6], [0.5, 0.3, 0.2])

    # Sample many times
    samples = [dist.sample() for _ in range(10000)]

    # Count occurrences
    counts = {1e6: 0, 2e6: 0, 3e6: 0}
    for s in samples:
        counts[s] += 1

    # Check distribution (within 2% tolerance)
    assert abs(counts[1e6] / 10000 - 0.5) < 0.02
    assert abs(counts[2e6] / 10000 - 0.3) < 0.02
    assert abs(counts[3e6] / 10000 - 0.2) < 0.02


def test_source_with_discrete_energy():
    """Test NeutronSource with Discrete energy."""
    energy_dist = yamc.sources.Discrete([14.06e6, 25e6], [0.9, 0.1])

    source = yamc.NeutronSource(
        position=[0.0, 0.0, 0.0],
        energy=energy_dist
    )

    # Sample particles
    energies = [source.sample().energy for _ in range(1000)]

    # All energies should be one of the two values
    unique_energies = set(energies)
    assert len(unique_energies) == 2
    assert 14.06e6 in unique_energies
    assert 25e6 in unique_energies

    # Check approximate distribution
    count_1406 = sum(1 for e in energies if abs(e - 14.06e6) < 1e3)
    ratio = count_1406 / len(energies)
    assert abs(ratio - 0.9) < 0.05


def test_source_energy_property_getter():
    """Test getting energy distribution from source."""
    source = yamc.NeutronSource()

    # Get energy back
    retrieved = source.energy
    assert isinstance(retrieved, yamc.sources.Discrete)
    # Default should be 14.06 MeV
    assert retrieved.energies == [14.06e6]
    assert retrieved.probabilities == [1.0]


def test_source_energy_property_setter():
    """Test setting energy on source."""
    source = yamc.NeutronSource()

    # Set discrete energy
    energy_dist = yamc.sources.Discrete([1e6, 2e6], [0.5, 0.5])
    source.energy = energy_dist

    # Get energy back
    retrieved = source.energy
    assert isinstance(retrieved, yamc.sources.Discrete)
    assert retrieved.energies == [1e6, 2e6]
    assert retrieved.probabilities == [0.5, 0.5]


def test_comparison_format():
    """Test that yamc format works correctly."""
    energy_dist = yamc.sources.Discrete([14.06e6], [1.0])
    source = yamc.NeutronSource(
        position=[0.0, 0.0, 0.0],
        energy=energy_dist
    )

    particle = source.sample()
    assert particle.energy == 14.06e6


def test_normalization():
    """Test that probabilities are automatically normalized."""
    # Probabilities don't sum to 1
    dist = yamc.sources.Discrete([1e6, 2e6], [2.0, 8.0])

    samples = [dist.sample() for _ in range(10000)]
    count_1 = sum(1 for s in samples if abs(s - 1e6) < 1e3)
    ratio = count_1 / len(samples)

    # Should still be 20% (2/10)
    assert abs(ratio - 0.2) < 0.02


def test_dt_dd_fusion_source():
    """Test D-T (14.06 MeV) and D-D (2.5 MeV) fusion source."""
    # 90% D-T, 10% D-D
    energy_dist = yamc.sources.Discrete([14.06e6, 2.5e6], [0.9, 0.1])
    source = yamc.NeutronSource(energy=energy_dist)

    energies = [source.sample().energy for _ in range(5000)]

    count_dt = sum(1 for e in energies if abs(e - 14.06e6) < 1e3)
    count_dd = sum(1 for e in energies if abs(e - 2.5e6) < 1e3)

    ratio_dt = count_dt / len(energies)
    ratio_dd = count_dd / len(energies)

    assert abs(ratio_dt - 0.9) < 0.03
    assert abs(ratio_dd - 0.1) < 0.03


def test_float_energy_conversion():
    """Test that floats are automatically converted to Discrete distributions."""
    source = yamc.NeutronSource()

    # Set energy as a float - should be automatically converted
    source.energy = 14.06e6

    # Should be converted to a Discrete distribution internally
    assert isinstance(source.energy, yamc.sources.Discrete)
    assert source.energy.energies == [14.06e6]
    assert source.energy.probabilities == [1.0]

    # Sampling should work
    particle = source.sample()
    assert particle.energy == 14.06e6


def test_discrete_repr():
    """Test Discrete __repr__."""
    dist = yamc.sources.Discrete([1e6, 2e6], [0.5, 0.5])
    repr_str = repr(dist)
    assert "Discrete" in repr_str
    assert "energies" in repr_str
    assert "probabilities" in repr_str


def test_source_repr_with_discrete():
    """Test NeutronSource __repr__ with discrete energy."""
    energy_dist = yamc.sources.Discrete([14.06e6, 25e6], [0.9, 0.1])
    source = yamc.NeutronSource(energy=energy_dist)

    repr_str = repr(source)
    assert "NeutronSource" in repr_str
    assert "Discrete" in repr_str


def test_single_energy_distribution():
    """Test discrete distribution with single energy."""
    dist = yamc.sources.Discrete([5e6], [1.0])

    # All samples should be 5 MeV
    for _ in range(100):
        assert dist.sample() == 5e6


def test_source_constructor_with_discrete():
    """Test creating source with discrete energy in constructor."""
    energy_dist = yamc.sources.Discrete([10e6, 15e6], [0.3, 0.7])

    source = yamc.NeutronSource(
        position=[1.0, 2.0, 3.0],
        direction=yamc.sources.Monodirectional([0.0, 0.0, 1.0]),
        energy=energy_dist
    )

    assert isinstance(source.position, list)
    assert source.position == [1.0, 2.0, 3.0]
    assert source.energy.energies == [10e6, 15e6]
    assert source.energy.probabilities == [0.3, 0.7]


def test_multiple_discrete_energies():
    """Test discrete distribution with many energies."""
    energies = [1e6, 2e6, 3e6, 4e6, 5e6]
    probabilities = [0.1, 0.2, 0.3, 0.25, 0.15]

    dist = yamc.sources.Discrete(energies, probabilities)

    samples = [dist.sample() for _ in range(20000)]
    counts = {e: 0 for e in energies}
    for s in samples:
        counts[s] += 1

    # Check all probabilities (2% tolerance)
    for energy, prob in zip(energies, probabilities):
        ratio = counts[energy] / len(samples)
        assert abs(ratio - prob) < 0.02, f"Energy {energy}: expected {prob}, got {ratio}"


def test_float_in_constructor():
    """Test that floats work in NeutronSource constructor."""
    source = yamc.NeutronSource(
        position=[1.0, 2.0, 3.0],
        energy=5e6
    )

    assert isinstance(source.position, list)
    assert source.position == [1.0, 2.0, 3.0]
    assert isinstance(source.energy, yamc.sources.Discrete)
    assert source.energy.energies == [5e6]
    assert source.energy.probabilities == [1.0]

    # Verify particle sampling
    particle = source.sample()
    assert particle.energy == 5e6


def test_mixed_discrete_and_float():
    """Test that we can use both Discrete and float interchangeably."""
    source = yamc.NeutronSource()

    # Start with Discrete
    source.energy = yamc.sources.Discrete([1e6, 2e6], [0.5, 0.5])
    assert source.energy.energies == [1e6, 2e6]

    # Switch to float
    source.energy = 10e6
    assert source.energy.energies == [10e6]
    assert source.energy.probabilities == [1.0]

    # Back to Discrete
    source.energy = yamc.sources.Discrete([3e6], [1.0])
    assert source.energy.energies == [3e6]
