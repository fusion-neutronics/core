"""Decay branching ratio uncertainty, at the Python boundary.

The numerical rules (a two-mode parent's sigma realised on both modes,
anticorrelated, with the pair's total kept, and every other multi-mode parent
held at nominal and named by why) are pinned in Rust, in
crates/yani-transmute/tests/decay_branching_uncertainty.rs. Here: the source is
offered and on by default, and the report carries its keys either way.
"""

import pytest
import yamc

NUC_DATA = "tests"
DAY = 86400.0

ENERGY_GROUPS = [1e-5, 0.625, 1e5, 2e7]
MULTIGROUP_FLUX = [1e12, 5e12, 1e14]
RATE = sum(MULTIGROUP_FLUX)

HELD_AT_NOMINAL = (
    "no_decay_branching_uncertainty",
    "decay_branchings_three_or_more_modes",
    "decay_branchings_unequal_sigmas",
    "decay_branchings_too_wide",
)


@pytest.fixture(autouse=True)
def _set_cross_sections():
    yamc.cross_section_data = NUC_DATA
    yield
    yamc.cross_section_data = None


def _iron():
    return yamc.Material(
        composition={"Fe56": 1.0},
        density=7.87,
        name="iron",
        volume=1.0,
        temperature=294,
    )


def _schedule():
    spectrum = yamc.NeutronSource(
        energy=yamc.sources.Histogram(ENERGY_GROUPS, MULTIGROUP_FLUX)
    )
    return yamc.PulseSchedule([
        yamc.Pulse(rate=RATE, duration=DAY, source=spectrum),
        yamc.Cooldown(duration=DAY),
    ])


def _info(sources):
    iron = _iron()
    results = iron.transmute(
        schedule=_schedule(),
        data_uncertainty=yamc.DataUncertainty(seed=1, samples=8, sources=sources),
    )
    return results.get_data_uncertainty_info(iron.id or 0)


def test_decay_branching_is_an_available_source_and_on_by_default():
    assert "decay_branching" in yamc.DataUncertainty.available_sources()
    assert "decay_branching" in yamc.DataUncertainty(seed=1).sources


def test_leaving_decay_branching_out_is_reported():
    info = _info(["cross_sections"])
    assert "decay branching ratio" in info["not_perturbed"]
    assert info["decay_branchings_perturbed"] == []
    for key in HELD_AT_NOMINAL:
        assert info[key] == [], key
    assert info["decay_branchings_sampled"] == 0
    assert info["decay_branchings_floored"] == 0


def test_asking_for_decay_branching_reports_disjoint_categories():
    """Every category key is present and no parent is in two of them.

    Which parent lands in which category is pinned on a synthetic chain in
    the Rust test, since the local fixture chain need not fill every one.
    """
    info = _info(["decay_branching"])
    assert "decay branching ratio" not in info["not_perturbed"]
    assert info["sources"] == ["decay_branching"]
    assert isinstance(info["has_gaps"], bool)
    for key in ("decay_branchings_perturbed", *HELD_AT_NOMINAL):
        assert all(isinstance(name, str) for name in info[key]), key
    # Every parent held at nominal is a gap the summary flag must show.
    if any(info[key] for key in HELD_AT_NOMINAL):
        assert info["has_gaps"]
    # A parent is in at most one category.
    seen = set(info["decay_branchings_perturbed"])
    for key in HELD_AT_NOMINAL:
        assert not seen & set(info[key]), key
        seen |= set(info[key])
    assert info["decay_branchings_sampled"] == 8 * len(
        info["decay_branchings_perturbed"]
    )
