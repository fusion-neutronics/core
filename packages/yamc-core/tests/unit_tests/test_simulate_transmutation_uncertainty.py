"""``data_uncertainty`` on ``Model.simulate_transmutation``.

The check that matters is against the thing a statistical uncertainty claims
to predict: run the same problem with independent seeds, and the spread of the
inventories across seeds must match the sigma each run reports for itself.
The rest pins the contract: bit-identical means, the coupled method rerun
whole per replica, the sources reported, the rate sigmas readable.
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


def _run(
    *, seed=42, particles=4000, method="independent", data_uncertainty=None, threads=None,
    material_id=MAT_ID,
):
    """A thermal-driven iron sphere, whose dominant product is Fe57."""
    material = yamc.Material(
        composition={"Fe56": 1.0},
        density=7.87,
        name="iron",
        transmutable=True,
        volume=4188.79,  # 4/3 pi 10^3
        temperature=294,
        id=material_id,
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


def test_the_statistical_draw_does_not_depend_on_the_material_id():
    """An id the user did not choose depends on what else the process built,
    so building the model again gives its material a new one. The same seeds
    must still give the same sigma, bit for bit, as under an explicit id."""
    request_ = yamc.DataUncertainty(seed=5, samples=32, sources=["statistical"])
    sigmas = []
    for material_id in (None, None, MAT_ID):
        results = _run(threads=1, data_uncertainty=request_, material_id=material_id)
        [used] = results.material_ids
        sigmas.append(results.get_nuclide_uncertainty(used, "Fe57", 1))
    assert sigmas[0] > 0.0
    assert sigmas[0] == sigmas[1] == sigmas[2], sigmas


@pytest.mark.parametrize(
    "request_",
    [STATISTICAL, yamc.DataUncertainty(seed=5, samples=16, sources=["cross_sections"])],
    ids=["statistical", "replica_weights"],
)
def test_the_means_are_untouched(request_):
    # One thread: the tally sums across threads with atomic adds, so two
    # multi-threaded runs already differ in the last bit whatever is asked.
    # The replica weights ride along the nominal histories without drawing a
    # random number, so the nominal tally is the one a plain run scores.
    plain = _run(threads=1)
    uncertain = _run(threads=1, data_uncertainty=request_)
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
    # The tally scores each MF=40 state per interval, so the isomeric split
    # is drawn on a transport run too, not held wholesale.
    assert "isomer_branching" in info["sources"]
    assert not any("every split held" in s for s in info["not_perturbed"])
    assert info["statistical_rates"] > 0
    assert info["statistical_sampled"] == 16 * info["statistical_rates"]
    # The transport carried a replica weight per replica, so each replica's
    # rates answer its own cross sections.
    assert info["flux_response"] == "replica_weights"
    assert info["transport_replicas"] == 16
    assert "tallied-rate statistics" not in info["not_perturbed"]
    # Each replica was solved again from the even half of the histories, and
    # the Monte Carlo noise that measures is out of every sigma.
    assert info["noise_removed"] is True
    for entry in info["noisy"]:
        assert entry["noise"] ** 2 > 0.25 * entry["std_dev"] ** 2


def test_the_monte_carlo_noise_is_removed_from_the_sigmas():
    results = _run(data_uncertainty=yamc.DataUncertainty(seed=1, samples=16))
    nuclides = results.get_material_nuclides(MAT_ID, 1)
    for nuclide in nuclides:
        sigma = results.get_nuclide_uncertainty(MAT_ID, nuclide, 1)
        spread, noise = results.get_nuclide_uncertainty_noise(MAT_ID, nuclide, 1)
        assert sigma**2 == pytest.approx(max(spread**2 - noise**2, 0.0), rel=1e-9, abs=1e-300)
    assert results.get_nuclide_uncertainty_noise(MAT_ID, next(iter(nuclides)), 0) == (0.0, 0.0)


def test_the_report_holds_no_flux_response_with_cross_sections_off():
    results = _run(data_uncertainty=STATISTICAL)
    info = results.get_data_uncertainty_info(MAT_ID)
    # Nothing perturbs a cross section, so there is no response of the flux to
    # carry; the unperturbed cross sections are named instead.
    assert "activation cross section (MF=33)" in info["not_perturbed"]
    assert info["flux_response"] == "unperturbed"
    assert info["transport_replicas"] is None


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


@pytest.mark.parametrize(
    "request_",
    [
        yamc.DataUncertainty(seed=1, samples=4),
        yamc.DataUncertainty(seed=1, samples=4, sources=["cross_sections"]),
        yamc.DataUncertainty(seed=1, samples=4, sources=["isomer_branching"]),
    ],
    ids=["every_source", "cross_sections_alone", "isomer_branching"],
)
def test_the_coupled_method_refuses_what_a_replica_cannot_carry(request_):
    with pytest.raises(RuntimeError, match="coupled"):
        _run(method="coupled", data_uncertainty=request_)


def test_the_coupled_method_reruns_the_whole_coupled_run():
    """Each replica is the coupled run again on its own seed: the nominal
    inventories are a plain coupled run's, bit for bit, and the spread is the
    replicas'."""
    request_ = yamc.DataUncertainty(seed=5, samples=8, sources=["statistical"])
    plain = _run(threads=1, method="coupled")
    uncertain = _run(threads=1, method="coupled", data_uncertainty=request_)
    for step in range(3):
        assert plain.get_material_nuclides(MAT_ID, step) == uncertain.get_material_nuclides(
            MAT_ID, step
        )
    info = uncertain.get_data_uncertainty_info(MAT_ID)
    assert info["flux_response"] == "coupled_replicas"
    assert info["samples"] == 8
    assert info["sources"] == ["statistical"]
    assert "tallied-rate statistics" not in info["not_perturbed"]
    assert uncertain.get_nuclide_uncertainty(MAT_ID, "Fe57", 1) > 0.0


def test_the_coupled_method_carries_cross_sections_beside_statistics():
    """With the cross sections each coupled replica transports on its own
    data; the nominal is still a plain coupled run's, bit for bit, and the
    report names what had no covariance to draw from."""
    request_ = yamc.DataUncertainty(
        seed=5, samples=4, sources=["statistical", "cross_sections"]
    )
    plain = _run(threads=1, particles=1000, method="coupled")
    uncertain = _run(
        threads=1, particles=1000, method="coupled", data_uncertainty=request_
    )
    for step in range(3):
        assert plain.get_material_nuclides(MAT_ID, step) == uncertain.get_material_nuclides(
            MAT_ID, step
        )
    info = uncertain.get_data_uncertainty_info(MAT_ID)
    assert info["flux_response"] == "coupled_replicas"
    assert info["sources"] == ["statistical", "cross_sections"]
    assert "activation cross section (MF=33)" not in info["not_perturbed"]
    assert info["cross_section_points_floored"] == 0
    covered = os.path.exists(os.path.join(TESTS_DIR, "Fe56.arrow", "covariance.arrow"))
    assert ("Fe56" in info["perturbed"]) == covered
    assert ("Fe56" in info["no_covariance_data"]) != covered


def test_the_cross_section_report_reads_the_tallied_spectrum_at_its_normalization():
    """The sigma report folds the evaluation against the tallied flux.

    The fold divides partial rates taken from the tally's spectrum by the
    tallied rates, so a spectrum on another normalization shows up as every
    channel listed in ``partials_above_rate``.
    """
    if not os.path.exists(os.path.join(TESTS_DIR, "Fe56.arrow", "covariance.arrow")):
        pytest.skip("the Fe56 fixture carries no covariance section")
    request = yamc.DataUncertainty(seed=1, samples=16, sources=["cross_sections"])
    info = _run(particles=2000, data_uncertainty=request).get_data_uncertainty_info(MAT_ID)
    assert info["partials_above_rate"] == {}
    assert info["flux_response"] == "replica_weights"
