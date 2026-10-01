"""``Model.data_uncertainty_coverage``: what nuclear-data uncertainty a model's
transport could carry, before any uncertainty run.

Iron is the case: ENDF/B-VIII.1 Fe56 states covariance for elastic, capture
and the total inelastic, so its inelastic levels are covered through MT 4, and
Fe57's fixture carries no covariance at all.
"""

import os

import pytest

import yamc

TESTS_DIR = os.path.join("crates", "yamc", "tests")


def _model(nuclides, tallies=()):
    material = yamc.Material(
        composition={n: 1.0 for n in nuclides},
        density=7.87,
        name="iron",
        temperature=294,
    )
    material.read_nuclear_data({n: os.path.join(TESTS_DIR, f"{n}.arrow") for n in nuclides})
    sphere = yamc.Sphere(radius=10.0, boundary="vacuum")
    cell = yamc.Cell(name="sphere", region=sphere.below, material=material)
    source = yamc.NeutronSource(position=(0, 0, 0), energy=yamc.sources.Discrete([14.1e6], [1.0]))
    return yamc.Model(
        geometry=yamc.Geometry([cell]), source=source, tallies=list(tallies), verbose=[]
    )


def _has_covariance(nuclide):
    return os.path.isfile(os.path.join(TESTS_DIR, f"{nuclide}.arrow", "covariance.arrow"))


def test_iron_reports_what_it_covers_and_how():
    if not _has_covariance("Fe56"):
        pytest.skip("Fe56 fixture carries no covariance")
    coverage = _model(["Fe56"]).data_uncertainty_coverage()
    fe56 = coverage["nuclides"]["Fe56"]
    perturbed = fe56["perturbed"]

    # Elastic and capture state their own covariance.
    assert perturbed[2]["via"] == 2
    assert perturbed[102]["via"] == 102
    # The largest per-cell sigma, which on a high-energy capture cell, where
    # capture is barely measured, can exceed 100%.
    assert perturbed[102]["max_relative_sigma"] > 0.0
    # A level of the total inelastic with none of its own takes MT 4's.
    levels = [mt for mt in perturbed if 51 <= mt <= 91]
    assert levels and all(perturbed[mt]["via"] in (mt, 4) for mt in levels)
    assert any(perturbed[mt]["via"] == 4 for mt in levels)
    # Redundant sums are never sampled themselves.
    assert 1 not in perturbed and 4 not in perturbed
    # Everything the nuclide samples is either perturbed or held, not both.
    assert not set(perturbed) & set(fe56["held_at_nominal"])
    assert fe56["cells"] > 0
    assert "repair" in fe56
    assert coverage["without_data"] == []
    assert any("MF=34" in item for item in coverage["not_perturbed"])


def test_a_nuclide_without_covariance_is_named_not_reported_exact():
    if not _has_covariance("Fe56") or _has_covariance("Fe57"):
        pytest.skip("needs Fe56 with covariance and Fe57 without")
    coverage = _model(["Fe56", "Fe57"]).data_uncertainty_coverage()
    assert coverage["without_data"] == ["Fe57"]
    assert "Fe57" not in coverage["nuclides"]
    assert "Fe56" in coverage["nuclides"]


def test_the_report_leaves_the_model_runnable():
    if not _has_covariance("Fe56"):
        pytest.skip("Fe56 fixture carries no covariance")
    tally = yamc.Tally(scores=["flux"], name="flux")
    model = _model(["Fe56"], tallies=[tally])
    model.data_uncertainty_coverage()
    results = model.simulate_transport(total_particles=200, seed=1)
    assert results[tally].mean[0] > 0.0
