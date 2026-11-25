"""Querying a material's cross sections before a run must not change the run.

`mean_free_path_neutron` builds the material's total on the fast lookup grid
and nothing else. The model took that as "already prepared for transport",
skipped the preparation, and transported with no collision partners: every
source neutron leaked uncollided. `macroscopic_cross_section` did the same with
the heating MTs missing, so heating came back zero.
"""

import numpy as np
import pytest
import yamc


def _run(query):
    yamc.neutron_cross_section_data = {"Li6": "tests/Li6.arrow", "Li7": "tests/Li7.arrow"}
    material = yamc.Material(composition={"Li": 1.0}, density=0.534)
    query(material)
    sphere = yamc.Sphere(radius=20.0, boundary="vacuum")
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    leakage = yamc.Tally(scores=["current-above"], surfaces=sphere, energy_bins=[1e-5, 1e6, 2e7])
    reactions = yamc.Tally(scores=["heating", "absorption"], cells=cell)
    model = yamc.Model(geometry=geometry, tallies=[leakage, reactions], source=yamc.NeutronSource())
    results = model.simulate_transport(total_particles=2_000)
    return np.array(results[leakage].mean), np.array(results[reactions].mean)


@pytest.mark.parametrize(
    "query",
    [
        lambda material: material.mean_free_path_neutron(14e6),
        lambda material: material.macroscopic_cross_section("(n,elastic)"),
    ],
    ids=["mean_free_path_neutron", "macroscopic_cross_section"],
)
def test_query_before_run_leaves_results_unchanged(query):
    expected_leakage, expected_reactions = _run(lambda material: None)
    leakage, reactions = _run(query)
    assert expected_leakage[0] > 0, "some neutrons must slow below 1 MeV in 20 cm of lithium"
    np.testing.assert_allclose(leakage, expected_leakage, rtol=1e-9)
    np.testing.assert_allclose(reactions, expected_reactions, rtol=1e-9)
