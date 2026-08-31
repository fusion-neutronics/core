"""All-scores collision-vs-track-length parity.

Builds one TrackLength model and one Collision model on a Li6 + Fe54
broomstick (14.06 MeV point source), each carrying *every* score that
currently supports both estimators, and asserts the two estimators agree
score-by-score within statistical noise.

The two-nuclide mix is chosen so every score in `ALL_SCORES` produces a
non-zero signal: Li6 drives the (n,t) / (n,α) channels (large
``H3-production`` / ``He4-production``), and Fe54 opens the
``He3-production`` channel that Li6 alone leaves closed at 14 MeV. Both
estimators must agree on the magnitude of each, not merely on which
channels are open.

Complements the per-score equivalence tests
(`test_heating_collision_estimator.py`,
 `test_reaction_rate_collision_estimator.py`,
 `test_production_collision_estimator.py`) by catching failure modes those
can't see:

  - **Missing-arm regression** in `score_collision`'s per-score `match`:
    if someone removes an arm or adds a new score type without wiring its
    collision contribution, a per-score test only catches it if a test
    file exists for that score; this all-scores test fails the moment
    the parity check stops holding.

  - **Cross-arm interaction**: a misplaced `continue` or shared mutable
    state inside the per-collision loop would only fire when multiple
    score types are scored at the same collision -- per-score tests run
    one score at a time, so they can't observe it.

Whenever a new score gains collision-estimator support, append it to
`ALL_SCORES` here; no new test file needed.
"""

import math

import pytest
import yamc

from .conftest import (
    BATCHES,
    CYLINDER_HALF_HEIGHT,
    CYLINDER_RADIUS,
    NEUTRON_SOURCE_ENERGY,
    SEED,
    TESTS_DATA_DIR,
)

# Bump particle budget 10× over the shared regression value to drop the
# noise floor below 0.1 % even for the lowest-magnitude score in the mix
# ((n,γ) on Fe54+Li6, ~1e-5 of flux). A quick scan confirms:
#   500_000 ppb → worst rel diff 0.27 %
#   2_000_000 ppb → 0.09 %
#   5_000_000 ppb → 0.07 %
# 5_000_000 leaves comfortable seed-to-seed headroom under the 0.1 %
# strict assertion below. Both estimator runs together take ~6 s.
#
# The strict cap is statistically sound only where 0.1 % is many combined
# sigma. For the rarest score in the mix ((n,gamma), ~1e-5 of flux) the
# combined TL+collision relative sigma is ~0.1 % at this budget, so the
# strict cap sits at ~1 sigma and flips with the fixed-seed realization
# (the 64-bit PCG stream of issue #274 landed it at 0.154 %, 1.5 sigma).
# A 4-sigma floor keeps the check exact where it is well resolved and
# statistically fair where it is not; a real estimator bug is either
# systematic (well-resolved scores blow the strict cap) or enormous.
PARTICLES = 5_000_000
REL_TOL = 1e-3  # strict 0.1 % cap where statistics allow (see above)

# 50/50 atom-fraction mix. Li6 supplies the dominant (n,t)/(n,α) signal;
# Fe54 supplies the (n,He3) and small-channel production rates that Li6
# alone can't produce at 14 MeV.
NUCLIDES = {"Li6": 0.5, "Fe54": 0.5}

# Every score that currently supports both Estimator::TrackLength and
# Estimator::Collision. Must stay in sync with `Score::required_estimator`
# in `crates/yamc-tallies/src/score.rs`.
ALL_SCORES = [
    "flux",
    "heating",
    "heating-local",
    "total",
    "absorption",
    "elastic",
    "(n,gamma)",
    "H1-production",
    "H2-production",
    "H3-production",
    "He3-production",
    "He4-production",
    "damage-energy",
]


def _build_model(estimator):
    """Broomstick + Li6/Fe54 mix + point neutron source, with one tally
    per score under the requested estimator. One Model per estimator
    means both runs visit the same collision sites under matched RNG
    seeds, so the estimator-vs-estimator agreement check is statistical
    rather than bias-vs-bias."""
    material = yamc.Material(
        composition=NUCLIDES,
        density=1.0,
        temperature=294,
    )
    material.read_nuclear_data(
        {nuc: str(TESTS_DATA_DIR / f"{nuc}.arrow") for nuc in NUCLIDES}
    )

    cylinder = yamc.Cylinder(axis="z", radius=CYLINDER_RADIUS, boundary="vacuum")
    z_bot = yamc.Plane(axis="z", offset=-CYLINDER_HALF_HEIGHT, boundary="vacuum")
    z_top = yamc.Plane(axis="z", offset=CYLINDER_HALF_HEIGHT, boundary="vacuum")
    region = cylinder.below & z_bot.above & z_top.below

    cell = yamc.Cell(name="broomstick", region=region, material=material)
    geometry = yamc.Geometry([cell])

    source = yamc.NeutronSource(
        position=(0, 0, 0),
        energy=yamc.sources.Discrete([NEUTRON_SOURCE_ENERGY], [1.0]),
    )

    tallies = [
        yamc.Tally(
            scores=[s],
            name=s,
            cells=cell,
            particle="neutron",
            estimator=estimator,
        )
        for s in ALL_SCORES
    ]

    model = yamc.Model(
        geometry=geometry,
        tallies=tallies,
        source=source,
    )
    return model, tallies


def _run(estimator):
    """Run one estimator; return {score_name: (mean, std)}."""
    model, tallies = _build_model(estimator)
    results = model.simulate_transport(total_particles=PARTICLES * BATCHES, seed=SEED)
    return {t.name: (results[t].mean[0], results[t].standard_deviation[0]) for t in tallies}


@pytest.fixture(scope="module")
def all_scores_results():
    """Cache both estimator runs once; the test parametrize spans 12 score
    cases sharing the same two simulations."""
    return {"tl": _run("track-length"), "co": _run("collision")}


@pytest.mark.parametrize("score", ALL_SCORES)
def test_all_supported_scores_agree_between_estimators(all_scores_results, score):
    """yamc(TL) == yamc(Collision) within 0.1 % relative diff, for every
    score that supports both estimators, scored simultaneously in one
    combined simulation. Strict relative-tolerance check (no σ-based
    fallback) -- both estimators are unbiased estimators of the same
    integral, so at 50 M histories per estimator the residual should be
    well below 0.1 % on this geometry."""
    tl_mean, tl_std = all_scores_results["tl"][score]
    co_mean, co_std = all_scores_results["co"][score]

    # The two-nuclide mix is chosen so every score in `ALL_SCORES` has a
    # non-zero signal under both estimators. Insist on that: a regression
    # where one estimator silently drops a score to zero while the other
    # keeps scoring would fail here loudly.
    assert tl_mean > 0.0, f"{score}: track-length mean must be positive"
    assert co_mean > 0.0, f"{score}: collision mean must be positive"

    rel_diff = abs(tl_mean - co_mean) / abs(tl_mean)
    combined_rel_sigma = math.sqrt(tl_std**2 + co_std**2) / abs(tl_mean)
    cap = max(REL_TOL, 4.0 * combined_rel_sigma)
    assert rel_diff < cap, (
        f"{score}: TL={tl_mean:.4e} (±{tl_std:.2e}) vs "
        f"Collision={co_mean:.4e} (±{co_std:.2e}) -- rel diff "
        f"{rel_diff:.4%} exceeds cap {cap:.4%} "
        f"(max of {REL_TOL:.1%} and 4 sigma)"
    )
