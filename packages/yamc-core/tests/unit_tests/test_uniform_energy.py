import pytest
import yamc


def test_uniform_construction():
    """Test creating Uniform energy distribution."""
    dist = yamc.sources.Uniform(1e6, 20e6)
    assert dist is not None
    assert dist.low == 1e6
    assert dist.high == 20e6


def test_uniform_validation_invalid_bounds():
    """Test Uniform validation with invalid bounds."""
    # a >= b should fail
    with pytest.raises(ValueError, match="Lower bound must be less than upper bound"):
        yamc.sources.Uniform(10e6, 5e6)

    # a == b should also fail
    with pytest.raises(ValueError, match="Lower bound must be less than upper bound"):
        yamc.sources.Uniform(5e6, 5e6)


def test_uniform_sampling_range():
    """Test that Uniform samples are within bounds."""
    dist = yamc.sources.Uniform(1e6, 20e6)

    # Sample many times and verify all samples are within bounds
    for _ in range(1000):
        energy = dist.sample()
        assert 1e6 <= energy <= 20e6, f"Sample {energy} outside bounds [1e6, 20e6]"


def test_uniform_sampling_distribution():
    """Test that Uniform samples are uniformly distributed."""
    dist = yamc.sources.Uniform(0.0, 10.0)

    # Sample many times
    samples = [dist.sample() for _ in range(10000)]

    # Check mean (should be ~5.0 for uniform [0, 10])
    #
    # Widened from 0.1, which was 3.46 sigma and so failed about one run in
    # 2000. The standard error of the mean here is (b - a) / sqrt(12 n) =
    # 10 / sqrt(120000) = 0.0289, so 0.2 is 6.9 sigma. The bin check below is
    # already 6.7 sigma per bin and is left alone.
    mean = sum(samples) / len(samples)
    assert abs(mean - 5.0) < 0.2, f"Expected mean ~5.0, got {mean}"

    # Check that samples are well distributed across range
    bins = [0] * 10
    for sample in samples:
        bin_idx = min(int(sample), 9)
        bins[bin_idx] += 1

    # Each bin should have roughly 10% of samples (within 2%)
    for i, count in enumerate(bins):
        ratio = count / len(samples)
        assert abs(ratio - 0.1) < 0.02, f"Bin {i} has ratio {ratio}, expected ~0.1"


def test_source_with_uniform_energy():
    """Test NeutronSource with Uniform energy."""
    energy_dist = yamc.sources.Uniform(1e6, 20e6)

    source = yamc.NeutronSource(
        position=[0.0, 0.0, 0.0],
        energy=energy_dist
    )

    # sample_n is the compiled batch form, so this draws 200x the old count in
    # less wall time than the 1000-iteration Python loop it replaces.
    #
    # The count is load bearing, not arbitrary. sample() takes no seed, so the
    # mean below is a random variable and the tolerance has to be read in sigma.
    # For Uniform(a, b) the standard error of the mean is (b - a) / sqrt(12 n),
    # which at the old n = 1000 is 173 keV against a 525 keV tolerance: 3.03
    # sigma, so it failed about one run in 370. It duly did, on windows-latest.
    # At n = 200000 the standard error is 12.3 keV and the same tolerance is 43
    # sigma. Do not reduce n without widening the tolerance to match.
    _positions, energies = source.sample_n(200_000)

    # All energies should be within bounds
    assert min(energies) >= 1e6, f"Energy {min(energies)} below the lower bound"
    assert max(energies) <= 20e6, f"Energy {max(energies)} above the upper bound"

    # Check approximate mean
    mean_energy = sum(energies) / len(energies)
    expected_mean = (1e6 + 20e6) / 2  # 10.5 MeV
    assert abs(mean_energy - expected_mean) / expected_mean < 0.05  # 43 sigma


def test_source_energy_property_getter():
    """Test getting Uniform energy distribution from source."""
    energy_dist = yamc.sources.Uniform(5e6, 15e6)
    source = yamc.NeutronSource(energy=energy_dist)

    # Get energy back
    retrieved = source.energy
    assert isinstance(retrieved, yamc.sources.Uniform)
    assert retrieved.low == 5e6
    assert retrieved.high == 15e6


def test_source_energy_property_setter():
    """Test setting Uniform energy on source."""
    source = yamc.NeutronSource()

    # Set uniform energy
    energy_dist = yamc.sources.Uniform(2e6, 10e6)
    source.energy = energy_dist

    # Get energy back
    retrieved = source.energy
    assert isinstance(retrieved, yamc.sources.Uniform)
    assert retrieved.low == 2e6
    assert retrieved.high == 10e6


def test_comparison_format():
    """Test that yamc format works correctly."""
    energy_dist = yamc.sources.Uniform(1e6, 20e6)
    source = yamc.NeutronSource(
        position=[0.0, 0.0, 0.0],
        energy=energy_dist
    )

    particle = source.sample()
    assert 1e6 <= particle.energy <= 20e6


def test_uniform_repr():
    """Test Uniform __repr__."""
    dist = yamc.sources.Uniform(1e6, 20e6)
    repr_str = repr(dist)
    assert "Uniform" in repr_str
    assert "low=" in repr_str
    assert "high=" in repr_str


def test_source_repr_with_uniform():
    """Test NeutronSource __repr__ with uniform energy."""
    energy_dist = yamc.sources.Uniform(1e6, 20e6)
    source = yamc.NeutronSource(energy=energy_dist)

    repr_str = repr(source)
    assert "NeutronSource" in repr_str
    assert "Uniform" in repr_str


def test_source_constructor_with_uniform():
    """Test creating source with uniform energy in constructor."""
    energy_dist = yamc.sources.Uniform(5e6, 15e6)

    source = yamc.NeutronSource(
        position=[1.0, 2.0, 3.0],
        direction=yamc.sources.Monodirectional([0.0, 0.0, 1.0]),
        energy=energy_dist
    )

    assert isinstance(source.position, list)
    assert source.position == [1.0, 2.0, 3.0]
    assert isinstance(source.energy, yamc.sources.Uniform)
    assert source.energy.low == 5e6
    assert source.energy.high == 15e6


def test_mixed_discrete_and_uniform():
    """Test that we can use both Discrete and Uniform interchangeably."""
    source = yamc.NeutronSource()

    # Start with Discrete
    source.energy = yamc.sources.Discrete([14.06e6], [1.0])
    assert isinstance(source.energy, yamc.sources.Discrete)

    # Switch to Uniform
    source.energy = yamc.sources.Uniform(1e6, 20e6)
    assert isinstance(source.energy, yamc.sources.Uniform)
    assert source.energy.low == 1e6
    assert source.energy.high == 20e6

    # Back to Discrete
    source.energy = yamc.sources.Discrete([5e6], [1.0])
    assert isinstance(source.energy, yamc.sources.Discrete)
    assert source.energy.energies == [5e6]


def test_example_usage():
    """Test the usage example: yamc.sources.Uniform(1e6, 20e6)."""
    dist = yamc.sources.Uniform(1e6, 20e6)

    # Verify properties
    assert dist.low == 1e6
    assert dist.high == 20e6

    # Verify sampling works
    samples = [dist.sample() for _ in range(100)]
    assert all(1e6 <= s <= 20e6 for s in samples)


def test_uniform_with_different_ranges():
    """Test Uniform distribution with various energy ranges."""
    # Low energy range
    dist_low = yamc.sources.Uniform(0.1e6, 1e6)
    samples_low = [dist_low.sample() for _ in range(100)]
    assert all(0.1e6 <= s <= 1e6 for s in samples_low)

    # High energy range
    dist_high = yamc.sources.Uniform(10e6, 30e6)
    samples_high = [dist_high.sample() for _ in range(100)]
    assert all(10e6 <= s <= 30e6 for s in samples_high)

    # Wide range
    dist_wide = yamc.sources.Uniform(0.01e6, 100e6)
    samples_wide = [dist_wide.sample() for _ in range(100)]
    assert all(0.01e6 <= s <= 100e6 for s in samples_wide)


def test_uniform_statistical_properties():
    """Test statistical properties of Uniform distribution."""
    # Test with a simple range [0, 100] for easier calculation
    dist = yamc.sources.Uniform(0.0, 100.0)

    n_samples = 50000
    samples = [dist.sample() for _ in range(n_samples)]

    # Mean should be 50
    #
    # Widened from 0.5, which was 3.87 sigma, about one run in 9000. Rarer than
    # the two above but the same defect, and rare flakes are worse than frequent
    # ones because they get re-run rather than fixed. The standard error here is
    # (b - a) / sqrt(12 n) = 100 / sqrt(600000) = 0.129, so 1.0 is 7.7 sigma.
    # The variance check below is already 12.5 sigma and is left alone.
    mean = sum(samples) / len(samples)
    assert abs(mean - 50.0) < 1.0, f"Expected mean ~50.0, got {mean}"

    # Variance for uniform [a, b] is (b-a)^2 / 12
    # For [0, 100], variance = 10000 / 12 = 833.33
    variance = sum((s - mean) ** 2 for s in samples) / len(samples)
    expected_variance = (100.0 ** 2) / 12
    assert abs(variance - expected_variance) / expected_variance < 0.05  # Within 5%
