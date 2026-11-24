"""Regression test: neutron heating tally under both estimators.

Verifies two equivalences on a Li6 broomstick (14.06 MeV point source):

  1. yamc heating(TrackLength) == yamc heating(Collision) within statistics.
     Both estimators target the same physical eV/source-neutron quantity.
  2. yamc heating(Collision) matches the reference data the existing
     regression suite already validates the TrackLength path against.

Li6 is chosen because it's a small, fast-running nuclide (single isotope,
high (n,t) heating signal at 14 MeV).
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


def _run(estimator):
    """Broomstick + Li6 + point neutron source; one heating tally with
    the requested estimator. Same parameters as the rest of the
    regression suite so reference data applies."""
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

    tally = yamc.Tally(
        scores=["heating"],
        name="heating",
        cells=cell,
        particle="neutron",
        estimator=estimator,
    )

    model = yamc.Model(
        geometry=geometry,
        tallies=[tally],
        source=source,
    )
    results = model.simulate_transport(total_particles=PARTICLES * BATCHES, seed=SEED)
    r = results[tally]
    return r.mean[0], r.standard_deviation[0]


@pytest.fixture(scope="module")
def yamc_results():
    """Cache both estimator runs; module-scoped to avoid double work."""
    tl_mean, tl_std = _run("track-length")
    co_mean, co_std = _run("collision")
    return {
        "tl": (tl_mean, tl_std),
        "co": (co_mean, co_std),
    }


def test_collision_estimator_matches_track_length(yamc_results):
    """yamc TL == yamc Collision for neutron heating."""
    tl_mean, tl_std = yamc_results["tl"]
    co_mean, co_std = yamc_results["co"]

    assert tl_mean > 0.0, "track-length heating must be positive"
    assert co_mean > 0.0, "collision heating must be positive"

    assert_scalar_agreement(
        tl_mean, co_mean, tl_std, co_std,
        name=f"{NUCLIDE}/heating(TL-vs-Collision)",
        rel_tol=0.05,
    )


def test_collision_estimator_matches_reference(yamc_results):
    """yamc Collision heating matches the reference data
    (same JSON the existing TrackLength regression test validates)."""
    ref = load_reference("neutron", NUCLIDE)
    skip_if_no_reference(ref, "neutron", NUCLIDE)

    co_mean, co_std = yamc_results["co"]
    ref_mean = ref["scalar_results"]["heating"]
    ref_std = ref["scalar_stds"]["heating"]
    flux_ref = ref["scalar_results"].get("flux", 0.0)

    assert_scalar_agreement(
        ref_mean, co_mean, ref_std, co_std,
        name=f"{NUCLIDE}/heating(Collision-vs-reference)",
        flux_ref=flux_ref,
    )


def test_track_length_estimator_matches_reference(yamc_results):
    """Sanity check: yamc TL heating still matches reference.
    Duplicates a slice of `test_neutron_scalar_scores` but isolates
    heating so a regression in either estimator is easy to bisect."""
    ref = load_reference("neutron", NUCLIDE)
    skip_if_no_reference(ref, "neutron", NUCLIDE)

    tl_mean, tl_std = yamc_results["tl"]
    ref_mean = ref["scalar_results"]["heating"]
    ref_std = ref["scalar_stds"]["heating"]
    flux_ref = ref["scalar_results"].get("flux", 0.0)

    assert_scalar_agreement(
        ref_mean, tl_mean, ref_std, tl_std,
        name=f"{NUCLIDE}/heating(TL-vs-reference)",
        flux_ref=flux_ref,
    )
