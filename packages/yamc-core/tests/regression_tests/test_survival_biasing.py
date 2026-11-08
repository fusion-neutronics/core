"""Regression tests: survival biasing (implicit capture + Russian roulette).

Survival biasing estimates the same physical means as analog transport, so
a survival-biased run must reproduce the committed analog reference
within the same statistical tolerances the analog regression tests use; no
new reference data is needed. Also covers the new Model API surface
(defaults, validation, setters).
"""

import pytest
import yamc

from .conftest import (
    BATCHES,
    CYLINDER_HALF_HEIGHT,
    CYLINDER_RADIUS,
    NEUTRON_GROUP_STRUCTURE,
    NEUTRON_SOURCE_ENERGY,
    PARTICLES,
    SEED,
    TESTS_DATA_DIR,
    assert_scalar_agreement,
    assert_spectrum_agreement,
    load_reference,
    skip_if_no_reference)

# One strong absorber (Li6) and one resonant scatterer (Fe56)
NUCLIDES = ["Li6", "Fe56"]

# Cache to avoid running the same simulation twice per nuclide
_cache = {}


def get_result(nuclide):
    if nuclide not in _cache:
        _cache[nuclide] = run_yamc_survival(nuclide)
    return _cache[nuclide]


def build_model(nuclide, **model_kwargs):
    """Broomstick model identical to test_neutron_transport's, so the
    committed analog reference applies unchanged."""
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

    flux = yamc.Tally(
        scores=["flux"],
        name="flux",
        cells=cell,
        particle="neutron")
    spectrum = yamc.Tally(
        scores=["flux"],
        name="neutron_spectrum",
        cells=cell,
        particle="neutron",
        energy_group_structure=NEUTRON_GROUP_STRUCTURE)

    model = yamc.Model(
        geometry=geometry,
        tallies=[flux, spectrum],
        source=source,
        **model_kwargs)
    return model, flux, spectrum


def run_yamc_survival(nuclide):
    model, flux, spectrum = build_model(
        nuclide, variance_reduction=[yamc.SurvivalBiasing()])
    results = model.simulate_transport(total_particles=PARTICLES * BATCHES, seed=SEED)
    r = results[flux]
    s = results[spectrum]
    return {
        "flux_mean": r.mean[0],
        "flux_std": r.standard_deviation[0],
        "spectrum_mean": list(s.mean),
        "spectrum_std": list(s.standard_deviation),
    }


@pytest.mark.parametrize("nuclide", NUCLIDES)
def test_survival_flux_matches_reference(nuclide):
    """Survival-biased flux must match the analog reference."""
    ref = load_reference("neutron", nuclide)
    skip_if_no_reference(ref, "neutron", nuclide)

    result = get_result(nuclide)
    assert_scalar_agreement(
        ref["scalar_results"]["flux"], result["flux_mean"],
        ref["scalar_stds"]["flux"], result["flux_std"],
        name=f"{nuclide}/flux(survival)", rel_tol=0.05)


@pytest.mark.parametrize("nuclide", NUCLIDES)
def test_survival_spectrum_matches_reference(nuclide):
    """Survival-biased spectrum must match the analog reference."""
    ref = load_reference("neutron", nuclide)
    skip_if_no_reference(ref, "neutron", nuclide)

    result = get_result(nuclide)
    assert_spectrum_agreement(
        ref["spectrum_mean"], result["spectrum_mean"],
        ref["spectrum_std"], result["spectrum_std"])


def test_survival_api_defaults_and_setters():
    # Analog by default: empty variance_reduction list.
    model, _, _ = build_model("Li6")
    assert model.variance_reduction == []

    # SurvivalBiasing defaults match the reference.
    sb = yamc.SurvivalBiasing()
    assert sb.weight_cutoff == 0.25
    assert sb.weight_survive == 1.0

    sb.weight_cutoff = 0.1
    sb.weight_survive = 0.5
    assert sb.weight_cutoff == 0.1
    assert sb.weight_survive == 0.5

    # Assign-whole-list semantics (the getter returns copies).
    model.variance_reduction = [sb]
    assert len(model.variance_reduction) == 1
    assert model.variance_reduction[0].weight_cutoff == 0.1
    assert "SurvivalBiasing" in repr(model.variance_reduction[0])

    with pytest.raises(ValueError):
        sb.weight_cutoff = 0.0
    with pytest.raises(ValueError):
        sb.weight_survive = -1.0


def test_survival_api_validation():
    # Invalid cutoff rejected at construction.
    with pytest.raises(ValueError):
        yamc.SurvivalBiasing(weight_cutoff=0.0)

    # A survivor weight below the cutoff would re-trigger roulette forever.
    with pytest.raises(ValueError):
        yamc.SurvivalBiasing(weight_cutoff=0.5, weight_survive=0.25)

    # Entries must be variance-reduction technique objects.
    with pytest.raises(TypeError):
        build_model("Li6", variance_reduction=["roulette"])

    # At most one SurvivalBiasing entry; duplicates are meaningless.
    with pytest.raises(ValueError):
        build_model(
            "Li6",
            variance_reduction=[
                yamc.SurvivalBiasing(),
                yamc.SurvivalBiasing()])
