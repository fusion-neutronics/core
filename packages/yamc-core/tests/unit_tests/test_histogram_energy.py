"""Tests for the Histogram (piecewise-constant) energy distribution."""

import pytest
import yamc


def test_histogram_construction():
    """Histogram exposes its boundaries and (unnormalized) probabilities."""
    dist = yamc.sources.Histogram([0.0, 1e6, 20e6], [0.3, 0.7])
    assert dist.boundaries == [0.0, 1e6, 20e6]
    assert dist.probabilities == [0.3, 0.7]


def test_histogram_validation_boundary_count():
    """boundaries must be exactly one longer than probabilities."""
    with pytest.raises(ValueError, match="one more boundary"):
        yamc.sources.Histogram([0.0, 1e6], [0.3, 0.7])
    with pytest.raises(ValueError, match="one more boundary"):
        yamc.sources.Histogram([0.0, 1e6, 20e6], [1.0])


def test_histogram_validation_non_ascending():
    """boundaries must be strictly ascending."""
    with pytest.raises(ValueError, match="strictly ascending"):
        yamc.sources.Histogram([0.0, 20e6, 1e6], [0.5, 0.5])
    with pytest.raises(ValueError, match="strictly ascending"):
        yamc.sources.Histogram([0.0, 1e6, 1e6], [0.5, 0.5])


def test_histogram_validation_negative_and_zero():
    """Negative or all-zero probabilities are rejected."""
    with pytest.raises(ValueError, match="cannot be negative"):
        yamc.sources.Histogram([0.0, 1e6, 2e6], [-0.5, 1.0])
    with pytest.raises(ValueError, match="non-zero"):
        yamc.sources.Histogram([0.0, 1e6, 2e6], [0.0, 0.0])


def test_histogram_sampling_in_range():
    """Every sample falls within the overall energy range."""
    dist = yamc.sources.Histogram([0.0, 1e6, 20e6], [0.3, 0.7])
    for _ in range(2000):
        e = dist.sample()
        assert 0.0 <= e <= 20e6, f"sample {e} out of range"


def test_histogram_sampling_mass_per_bin():
    """probabilities are per-bin MASS: ~30% land in the narrow first bin."""
    dist = yamc.sources.Histogram([0.0, 1e6, 20e6], [0.3, 0.7])
    n = 50000
    frac_low = sum(1 for _ in range(n) if dist.sample() < 1e6) / n
    assert frac_low == pytest.approx(0.3, abs=0.02)


def test_histogram_auto_normalizes():
    """Only relative weights matter: [3, 7] behaves like [0.3, 0.7]."""
    dist = yamc.sources.Histogram([0.0, 1e6, 20e6], [3.0, 7.0])
    n = 50000
    frac_low = sum(1 for _ in range(n) if dist.sample() < 1e6) / n
    assert frac_low == pytest.approx(0.3, abs=0.02)


def test_histogram_from_named_group_structure():
    """boundaries may be a built-in group structure name instead of a list."""
    # CCFE-709: 709 groups -> 710 resolved boundaries, 1e-5 eV to 1 GeV.
    dist = yamc.sources.Histogram("CCFE-709", [1.0] * 709)
    assert len(dist.boundaries) == 710
    assert dist.boundaries[0] == 1e-5
    assert dist.boundaries[-1] == 1e9

    # VITAMIN-J-175: 175 groups -> 176 boundaries.
    v = yamc.sources.Histogram("VITAMIN-J-175", [1.0] * 175)
    assert len(v.boundaries) == 176


def test_histogram_named_structure_group_count_mismatch():
    """A name resolves the edges, so probabilities must match the group count."""
    # CCFE-709 has 709 groups; 708 probabilities is one short.
    with pytest.raises(ValueError, match="one more boundary"):
        yamc.sources.Histogram("CCFE-709", [1.0] * 708)


def test_histogram_unknown_named_structure():
    """An unknown group structure name is rejected, listing the known ones."""
    with pytest.raises(ValueError, match="Unknown group structure"):
        yamc.sources.Histogram("NOPE-1", [1.0])


def test_histogram_as_neutron_source_energy():
    """A NeutronSource accepts a Histogram energy distribution and round-trips it."""
    hist = yamc.sources.Histogram([0.0, 1e6, 20e6], [0.3, 0.7])
    src = yamc.NeutronSource(position=(0, 0, 0), energy=hist)
    energy = src.energy
    assert energy.boundaries == [0.0, 1e6, 20e6]
    assert energy.probabilities == [0.3, 0.7]
