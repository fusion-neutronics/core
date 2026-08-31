"""Regression tests: Type 1 -- Neutron-only transport vs reference data."""

import pytest
import yamc

from .conftest import (
    ALL_NEUTRON_NUCLIDES,
    BATCHES,
    CYLINDER_HALF_HEIGHT,
    CYLINDER_RADIUS,
    NEUTRON_GROUP_STRUCTURE,
    NEUTRON_SCALAR_SCORES,
    NEUTRON_SOURCE_ENERGY,
    PARTICLES,
    SEED,
    TESTS_DATA_DIR,
    assert_scalar_agreement,
    assert_spectrum_agreement,
    load_reference,
    skip_if_no_reference)

# Cache to avoid running the same simulation twice per nuclide
_cache = {}


def get_result(nuclide):
    if nuclide not in _cache:
        _cache[nuclide] = run_yamc_neutron(nuclide)
    return _cache[nuclide]


def run_yamc_neutron(nuclide):
    """Run yamc neutron-only broomstick simulation."""
    neutron_path = str(TESTS_DATA_DIR / f"{nuclide}.arrow")

    material = yamc.Material(
        composition={nuclide: 1.0},
        density=1.0,
        temperature=294)
    material.read_nuclear_data({nuclide: neutron_path})

    cylinder = yamc.Cylinder(axis="z", radius=CYLINDER_RADIUS, boundary="vacuum")
    z_bot = yamc.Plane(axis="z", offset=-CYLINDER_HALF_HEIGHT, boundary="vacuum")
    z_top = yamc.Plane(axis="z", offset=CYLINDER_HALF_HEIGHT, boundary="vacuum")
    region = cylinder.below & z_bot.above & z_top.below

    cell = yamc.Cell(name="broomstick", region=region, material=material)
    geometry = yamc.Geometry([cell])

    source = yamc.NeutronSource(
        position=(0, 0, 0),
        energy=yamc.sources.Discrete([NEUTRON_SOURCE_ENERGY], [1.0]))
    tallies = []

    for score in NEUTRON_SCALAR_SCORES:
        t = yamc.Tally(
            scores=[score],
            name=score,
            cells=cell,
            particle="neutron")
        tallies.append(t)

    t_spectrum = yamc.Tally(
        scores=["flux"],
        name="neutron_spectrum",
        cells=cell,
        particle="neutron",
        energy_group_structure=NEUTRON_GROUP_STRUCTURE)
    tallies.append(t_spectrum)

    model = yamc.Model(geometry=geometry, tallies=tallies, source=source)
    results = model.simulate_transport(total_particles=PARTICLES * BATCHES, seed=SEED)

    scalar_results = {}
    scalar_stds = {}
    for t in tallies:
        if t.name == "neutron_spectrum":
            continue
        r = results[t]
        scalar_results[t.name] = r.mean[0]
        scalar_stds[t.name] = r.standard_deviation[0]

    spectrum = results[t_spectrum]
    return {
        "scalar_results": scalar_results,
        "scalar_stds": scalar_stds,
        "spectrum_mean": list(spectrum.mean),
        "spectrum_std": list(spectrum.standard_deviation),
    }


@pytest.mark.parametrize("nuclide", ALL_NEUTRON_NUCLIDES)
def test_neutron_flux(nuclide):
    """Test neutron flux matches reference (primary observable)."""
    ref = load_reference("neutron", nuclide)
    skip_if_no_reference(ref, "neutron", nuclide)

    result = get_result(nuclide)

    assert_scalar_agreement(
        ref["scalar_results"]["flux"], result["scalar_results"]["flux"],
        ref["scalar_stds"]["flux"], result["scalar_stds"]["flux"],
        name=f"{nuclide}/flux", rel_tol=0.05)


@pytest.mark.parametrize("nuclide", ALL_NEUTRON_NUCLIDES)
def test_neutron_scalar_scores(nuclide):
    """Test neutron scalar tallies broadly agree with reference.

    Uses generous tolerances (50%) since there are known systematic
    differences on secondary production and absorption channels.
    Fails only if more than 25% of scores are off by > 50%.
    """
    ref = load_reference("neutron", nuclide)
    skip_if_no_reference(ref, "neutron", nuclide)

    result = get_result(nuclide)
    flux_ref = ref["scalar_results"].get("flux", 0.0)

    failures = []
    n_compared = 0
    for score in NEUTRON_SCALAR_SCORES:
        if score == "flux":
            continue  # tested separately with tight tolerance
        ref_val = ref["scalar_results"].get(score, 0.0)
        yamc_val = result["scalar_results"].get(score, 0.0)

        # Skip near-zero scores (< 0.1% of flux)
        if flux_ref > 0:
            magnitude = max(abs(ref_val), abs(yamc_val))
            if magnitude < 1e-3 * flux_ref:
                continue

        n_compared += 1
        denom = max(abs(ref_val), abs(yamc_val))
        if denom > 0:
            rel_diff = abs(yamc_val - ref_val) / denom
            if rel_diff > 0.05:
                failures.append(f"{score}: yamc={yamc_val:.4e} ref={ref_val:.4e} ({rel_diff:.0%})")

    if n_compared > 0:
        assert len(failures) == 0, (
            f"{nuclide}: {len(failures)}/{n_compared} scores differ by >5%:\n"
            + "\n".join(failures)
        )


@pytest.mark.parametrize("nuclide", ALL_NEUTRON_NUCLIDES)
def test_neutron_spectrum(nuclide):
    """Test neutron flux spectrum matches reference."""
    ref = load_reference("neutron", nuclide)
    skip_if_no_reference(ref, "neutron", nuclide)

    result = get_result(nuclide)

    assert_spectrum_agreement(
        ref["spectrum_mean"], result["spectrum_mean"],
        ref["spectrum_std"], result["spectrum_std"])
