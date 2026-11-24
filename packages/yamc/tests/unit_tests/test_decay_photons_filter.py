"""Tests for ParentNuclideFilter → parent_nuclides kwarg on Tally."""

import yamc as mmc


def test_parent_nuclide_tally_creation():
    """Test creating a tally with parent_nuclides kwarg."""
    tally = mmc.Tally(
        scores=["flux"],
        parent_nuclides=["Co60", "Mn56", "Fe59"],
        particle="photon",
    )
    assert tally.parent_nuclides == ["Co60", "Mn56", "Fe59"]
    assert tally.n_parent_bins == 3


def test_parent_nuclide_tally_single():
    """Test tally with a single parent nuclide."""
    tally = mmc.Tally(
        scores=["flux"],
        parent_nuclides=["Co60"],
        particle="photon",
    )
    assert tally.parent_nuclides == ["Co60"]
    assert tally.n_parent_bins == 1


def test_parent_nuclide_empty():
    """An empty parent_nuclides list is allowed and is distinct from None: it is
    a decay-photon tally whose material has no gamma-emitting activation
    products, so it scores zero rather than raising. None means no parent filter
    at all and keeps the default single bin."""
    tally = mmc.Tally(
        scores=["flux"],
        parent_nuclides=[],
        particle="photon",
    )
    assert tally.parent_nuclides == []
    assert tally.n_parent_bins == 0

    no_filter = mmc.Tally(scores=["flux"], particle="photon")
    assert no_filter.parent_nuclides is None
    assert no_filter.n_parent_bins == 1


def test_parent_nuclide_in_tally():
    """Test that parent_nuclides and particle can be set on a tally."""
    tally = mmc.Tally(
        scores=["flux"],
        parent_nuclides=["Co60", "Mn56"],
        particle="photon",
    )

    # Verify properties were set
    assert tally.parent_nuclides == ["Co60", "Mn56"]
    assert tally.particle == "photon"
    assert tally.n_parent_bins == 2
