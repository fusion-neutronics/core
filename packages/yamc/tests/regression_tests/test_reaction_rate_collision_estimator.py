"""Regression test: neutron reaction-rate tally under both estimators.

Verifies on a Li6 broomstick (14.06 MeV point source) that for each
representative reaction-rate score:

  1. yamc(TrackLength) == yamc(Collision) within statistics
     (both estimators converge to the same physical reactions/source-neutron).
  2. yamc(Collision) matches the reference data the existing
     regression suite already validates the TrackLength path against.

Covers `total`, `absorption`, `elastic` -- three different XS lookup
paths (smooth lookup, named-MT constructor, MT-1 URR shortcut). Li6
is small/fast and present in the regression reference set.
"""

import pytest
import yamc

from .conftest import (
    BATCHES,
    CYLINDER_HALF_HEIGHT,
    CYLINDER_RADIUS,
    NEUTRON_SOURCE_ENERGY,
    PARTICLES,
    SEED,
    TESTS_DATA_DIR,
    assert_scalar_agreement,
    load_reference,
    skip_if_no_reference,
)

NUCLIDE = "Li6"
RR_SCORES = ["total", "absorption", "elastic"]


def _build_model(estimator):
    """Broomstick + Li6 + point neutron source; one tally per RR score
    in `RR_SCORES`, all under the requested estimator."""
    neutron_path = str(TESTS_DATA_DIR / f"{NUCLIDE}.arrow")

    material = yamc.Material(
        composition={NUCLIDE: 1.0},
        density=1.0,
        temperature=294,
    )
    material.read_nuclear_data({NUCLIDE: neutron_path})

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
            scores=[score],
            name=score,
            cells=cell,
            particle="neutron",
            estimator=estimator,
        )
        for score in RR_SCORES
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
def yamc_results():
    """Cache both estimator runs; module-scoped to avoid double work."""
    return {"tl": _run("track-length"), "co": _run("collision")}


@pytest.mark.parametrize("score", RR_SCORES)
def test_collision_estimator_matches_track_length(yamc_results, score):
    """yamc(TL) == yamc(Collision) for each reaction-rate score."""
    tl_mean, tl_std = yamc_results["tl"][score]
    co_mean, co_std = yamc_results["co"][score]

    assert tl_mean > 0.0, f"{score}: track-length mean must be positive"
    assert co_mean > 0.0, f"{score}: collision mean must be positive"

    assert_scalar_agreement(
        tl_mean, co_mean, tl_std, co_std,
        name=f"{NUCLIDE}/{score}(TL-vs-Collision)",
        rel_tol=0.05,
    )


@pytest.mark.parametrize("score", RR_SCORES)
def test_collision_estimator_matches_reference(yamc_results, score):
    """yamc Collision matches the reference data the existing
    TrackLength regression test validates."""
    ref = load_reference("neutron", NUCLIDE)
    skip_if_no_reference(ref, "neutron", NUCLIDE)

    if score not in ref["scalar_results"]:
        pytest.skip(f"reference data has no '{score}' entry")

    co_mean, co_std = yamc_results["co"][score]
    ref_mean = ref["scalar_results"][score]
    ref_std = ref["scalar_stds"][score]
    flux_ref = ref["scalar_results"].get("flux", 0.0)

    assert_scalar_agreement(
        ref_mean, co_mean, ref_std, co_std,
        name=f"{NUCLIDE}/{score}(Collision-vs-reference)",
        flux_ref=flux_ref,
    )


@pytest.mark.parametrize("score", RR_SCORES)
def test_track_length_estimator_matches_reference(yamc_results, score):
    """Sanity check: yamc TL still matches reference. Duplicates
    a slice of `test_neutron_scalar_scores` but isolates RR scores so
    a regression in either estimator is easy to bisect."""
    ref = load_reference("neutron", NUCLIDE)
    skip_if_no_reference(ref, "neutron", NUCLIDE)

    if score not in ref["scalar_results"]:
        pytest.skip(f"reference data has no '{score}' entry")

    tl_mean, tl_std = yamc_results["tl"][score]
    ref_mean = ref["scalar_results"][score]
    ref_std = ref["scalar_stds"][score]
    flux_ref = ref["scalar_results"].get("flux", 0.0)

    assert_scalar_agreement(
        ref_mean, tl_mean, ref_std, tl_std,
        name=f"{NUCLIDE}/{score}(TL-vs-reference)",
        flux_ref=flux_ref,
    )
