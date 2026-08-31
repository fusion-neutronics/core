"""Photon-side all-scores collision-vs-track-length parity.

Mirror of ``test_all_scores_collision_parity.py`` for the photon side:
one TrackLength model and one Collision model on an Fe sphere with a
high-energy photon source (5 MeV -- pair-production threshold is
~1.022 MeV, so all four PhotonXS components have non-trivial signal).
Each model carries *every* score that fires for photons under both
estimators.

Catches the same classes of regression as the neutron parity test
(missing-arm, cross-arm interaction) but for the photon dispatch -- in
particular, this guards against the latent "photon flux under
``Estimator::Collision`` silently returns 0" bug that existed before
PhotonXS support landed (the pre-collision dispatch fell back on the
neutron MT-1 lookup, which returns 0 for photons).

Whenever a new photon-side score gains collision-estimator support,
append it to ``ALL_PHOTON_SCORES`` here.
"""

import pytest
import yamc

from .conftest import (
    BATCHES,
    CYLINDER_HALF_HEIGHT,
    CYLINDER_RADIUS,
    SEED,
    TESTS_DATA_DIR,
)

# Bump particle budget so the strict 0.1 % cap below has comfortable
# margin even on the lower-XS photon components. Photons are roughly
# as cheap per-history as neutrons here, so this is fine performance-wise.
PARTICLES = 5_000_000
REL_TOL = 1e-3  # strict 0.1 % cap, no σ-based fallback

# Heating crosses a convention boundary between the estimators since
# #356: track-length scores collision KERMA along tracks (electron and
# positron kinetic energy counted at the interaction site, INCLUDING
# the part later radiated as TTB bremsstrahlung), while the collision
# estimator scores the analog absorbed energy (the TTB photons are
# banked and deposit wherever they are absorbed). In this THIN
# broomstick most bremsstrahlung escapes, so the two differ by the
# radiative fraction -- the textbook kerma-vs-dose distinction
# (measured 4.7% at 5 MeV; on a thick absorbing sphere they agree to
# ~1% because the radiated photons re-deposit locally).
HEATING_REL_TOL = 0.08

NUCLIDE = "Fe56"
ELEMENT = "Fe"
SOURCE_ENERGY = 5.0e6  # 5 MeV -- opens pair-production cleanly

# Every score that fires for a photon source under both estimators.
# Must stay in sync with `Score::required_estimator` in
# `crates/yamc-tallies/src/score.rs` (only photon-applicable scores).
ALL_PHOTON_SCORES = [
    "flux",
    "heating",
    "heating-local",
    "coherent-scatter",
    "incoherent-scatter",
    "photoelectric",
    "pair-production",
]


def _build_model(estimator):
    """Fe broomstick + point photon source at SOURCE_ENERGY, with one
    tally per photon score under the requested estimator. All tallies
    carry a ``particle="photon"`` filter so neutron contributions can't
    accidentally enter (defensive -- the source is photon-only)."""
    neutron_path = str(TESTS_DATA_DIR / f"{NUCLIDE}.arrow")
    element_path = str(TESTS_DATA_DIR / f"{ELEMENT}.arrow")

    material = yamc.Material(
        composition={NUCLIDE: 1.0},
        density=7.874,
        temperature=294,
    )
    material.read_nuclear_data(
        {NUCLIDE: neutron_path},
        photon_data={ELEMENT: element_path},
    )

    cylinder = yamc.Cylinder(axis="z", radius=CYLINDER_RADIUS, boundary="vacuum")
    z_bot = yamc.Plane(axis="z", offset=-CYLINDER_HALF_HEIGHT, boundary="vacuum")
    z_top = yamc.Plane(axis="z", offset=CYLINDER_HALF_HEIGHT, boundary="vacuum")
    region = cylinder.below & z_bot.above & z_top.below

    cell = yamc.Cell(name="broomstick", region=region, material=material)
    geometry = yamc.Geometry([cell])

    source = yamc.PhotonSource(
        position=(0, 0, 0),
        energy=yamc.sources.Discrete([SOURCE_ENERGY], [1.0]),
    )

    tallies = [
        yamc.Tally(
            scores=[s],
            name=s,
            cells=cell,
            particle="photon",
            estimator=estimator,
        )
        for s in ALL_PHOTON_SCORES
    ]

    model = yamc.Model(
        geometry=geometry,
        tallies=tallies,
        source=source,
    )
    return model, tallies


def _run(estimator):
    model, tallies = _build_model(estimator)
    results = model.simulate_transport(total_particles=PARTICLES * BATCHES, seed=SEED)
    return {t.name: (results[t].mean[0], results[t].standard_deviation[0]) for t in tallies}


@pytest.fixture(scope="module")
def all_photon_scores_results():
    """Cache both estimator runs once; the test parametrize spans the
    photon scores sharing the same two simulations."""
    return {"tl": _run("track-length"), "co": _run("collision")}


@pytest.mark.parametrize("score", ALL_PHOTON_SCORES)
def test_all_supported_photon_scores_agree_between_estimators(all_photon_scores_results, score):
    """yamc(TL) == yamc(Collision) within 0.1 % relative diff, for every
    photon-applicable score under both estimators, scored simultaneously
    in one combined photon simulation."""
    tl_mean, tl_std = all_photon_scores_results["tl"][score]
    co_mean, co_std = all_photon_scores_results["co"][score]

    assert tl_mean > 0.0, f"{score}: track-length mean must be positive"
    assert co_mean > 0.0, f"{score}: collision mean must be positive"

    tol = HEATING_REL_TOL if score in ("heating", "heating-local") else REL_TOL
    rel_diff = abs(tl_mean - co_mean) / abs(tl_mean)
    assert rel_diff < tol, (
        f"{score}: TL={tl_mean:.4e} (±{tl_std:.2e}) vs "
        f"Collision={co_mean:.4e} (±{co_std:.2e}) -- rel diff "
        f"{rel_diff:.4%} exceeds {tol:.1%} cap"
    )
