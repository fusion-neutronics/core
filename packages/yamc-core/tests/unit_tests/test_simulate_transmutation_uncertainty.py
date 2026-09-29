"""``data_uncertainty`` on ``Model.simulate_transmutation`` (issue #140, item 3).

The check that matters is against the thing a statistical uncertainty claims
to predict: run the same problem with independent seeds, and the spread of the
inventories across seeds must match the sigma each run reports for itself.
The rest pins the contract: bit-identical means, the coupled method refused,
the sources reported, the rate sigmas readable.
"""

import os

import numpy as np
import pytest

import yamc

TESTS_DIR = os.path.join("crates", "yamc", "tests")
CHAIN = os.path.join(TESTS_DIR, "transmutation-endf-b8.1-sfr.arrow")
MAT_ID = 1
HOUR = 3600.0


@pytest.fixture(autouse=True)
def _chain():
    yamc.transmutation_decay_data = CHAIN
    yamc.transmutation_reactions = CHAIN
    yamc.transmutation_fission_yields = CHAIN


def _run(*, seed=42, particles=4000, method="independent", data_uncertainty=None, threads=None):
    """A thermal-driven iron sphere, whose dominant product is Fe57."""
    material = yamc.Material(
        composition={"Fe56": 1.0},
        density=7.87,
        name="iron",
        transmutable=True,
        volume=4188.79,  # 4/3 pi 10^3
        temperature=294,
        id=MAT_ID,
    )
    material.read_nuclear_data({"Fe56": os.path.join(TESTS_DIR, "Fe56.arrow")})
    sphere = yamc.Sphere(radius=10.0, boundary="vacuum")
    cell = yamc.Cell(name="sphere", region=sphere.below, material=material)
    source = yamc.NeutronSource(
        position=(0, 0, 0), energy=yamc.sources.Discrete([0.0253], [1.0])
    )
    model = yamc.Model(geometry=yamc.Geometry([cell]), source=source, verbose=[])
    schedule = yamc.PulseSchedule([
        yamc.Pulse(rate=1.0e18, duration=HOUR, source=source),
        yamc.Cooldown(duration=HOUR),
    ])
    return model.simulate_transmutation(
        method=method,
        schedule=schedule,
        total_particles=particles,
        seed=seed,
        threads=threads,
        data_uncertainty=data_uncertainty,
    )


STATISTICAL = yamc.DataUncertainty(seed=5, samples=256, sources=["statistical"])


def test_the_statistical_sigma_predicts_the_seed_to_seed_spread():
    """Sixteen independent transports: the reported sigma must be the spread.

    With 16 runs the measured variance over the predicted one is a
    chi-squared on 15 degrees of freedom over 15, in [0.42, 1.83] 95% of the
    time. The seeds are fixed, so the bounds below are a little wider.
    """
    means, predicted = [], []
    for seed in range(1, 17):
        results = _run(seed=seed, data_uncertainty=STATISTICAL)
        means.append(results.get_nuclide_density(MAT_ID, "Fe57", 1))
        predicted.append(results.get_nuclide_uncertainty(MAT_ID, "Fe57", 1) ** 2)
    ratio = np.var(means, ddof=1) / np.mean(predicted)
    assert 0.35 <= ratio <= 2.2, f"measured over predicted variance {ratio:.3f}"


def test_the_means_are_untouched():
    # One thread: the tally sums across threads with atomic adds, so two
    # multi-threaded runs already differ in the last bit whatever is asked.
    plain = _run(threads=1)
    uncertain = _run(threads=1, data_uncertainty=STATISTICAL)
    for step in range(3):
        assert plain.get_material_nuclides(MAT_ID, step) == uncertain.get_material_nuclides(
            MAT_ID, step
        )


def test_the_report_names_the_sources_that_applied():
    results = _run(data_uncertainty=yamc.DataUncertainty(seed=1, samples=16))
    info = results.get_data_uncertainty_info(MAT_ID)
    # The default asks for everything; flux_spectrum has nothing to act on in
    # a transport run, whose flux error is the statistical one.
    assert "statistical" in info["sources"]
    assert "flux_spectrum" not in info["sources"]
    assert info["statistical_rates"] > 0
    assert info["statistical_sampled"] == 16 * info["statistical_rates"]
    # One transport feeds every replica, so how the flux would answer a
    # perturbed cross section is held, while the tallied values are drawn.
    assert "flux response to perturbed cross sections (one transport)" in info["not_perturbed"]
    assert "tallied-rate statistics" not in info["not_perturbed"]


def test_the_report_holds_no_flux_response_with_cross_sections_off():
    results = _run(data_uncertainty=STATISTICAL)
    info = results.get_data_uncertainty_info(MAT_ID)
    # Nothing perturbs a cross section, so there is no response of the flux to
    # hold; the unperturbed cross sections are named instead.
    assert "activation cross section (MF=33)" in info["not_perturbed"]
    assert "flux response to perturbed cross sections (one transport)" not in info["not_perturbed"]


def test_the_report_names_the_tallied_rates_held_with_statistical_off():
    sources = ["cross_sections"]
    results = _run(data_uncertainty=yamc.DataUncertainty(seed=1, samples=16, sources=sources))
    info = results.get_data_uncertainty_info(MAT_ID)
    assert "statistical" not in info["sources"]
    assert info["statistical_rates"] == 0
    assert "tallied-rate statistics" in info["not_perturbed"]


def test_each_rate_has_a_statistical_sigma():
    results = _run(data_uncertainty=STATISTICAL)
    entries = results.get_reaction_rate_uncertainty(MAT_ID, 0)
    capture = [e for e in entries if e[0] == "Fe56" and e[1] == "(n,gamma)" and e[2] is None]
    assert len(capture) == 1
    _, _, _, rate, sigma = capture[0]
    edge_rate = results.get_reaction_rates(MAT_ID, 0)["Fe56"]["(n,gamma)"][0][1]
    assert rate == pytest.approx(edge_rate, rel=1e-12)
    assert 0.0 < sigma < rate
    # The cooldown drives nothing.
    assert all(e[3] == 0.0 for e in results.get_reaction_rate_uncertainty(MAT_ID, 1))
    assert _run().get_reaction_rate_uncertainty(MAT_ID, 0) is None


def test_the_coupled_method_is_refused():
    with pytest.raises(RuntimeError, match="independent"):
        _run(method="coupled", data_uncertainty=STATISTICAL)
