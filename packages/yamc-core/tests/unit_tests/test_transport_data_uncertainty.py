"""Nuclear-data uncertainty on ``Model.simulate_transport``.

Each history carries one weight per cross-section replica, so the run is the
nominal run with per-replica sums beside it: the mean must be unchanged bit
for bit, and every tally gains a nuclear-data standard deviation and a
replica mean. ENDF/B-VIII.1 Fe56, whose covariance covers capture.
"""

import os

import pytest

import yamc

TESTS_DIR = os.path.join("crates", "yamc", "tests")


def _has_covariance():
    return os.path.isfile(os.path.join(TESTS_DIR, "Fe56.arrow", "covariance.arrow"))


def _model(tallies=None):
    material = yamc.Material(composition={"Fe56": 1.0}, density=7.87, name="iron", temperature=294)
    material.read_nuclear_data({"Fe56": os.path.join(TESTS_DIR, "Fe56.arrow")})
    sphere = yamc.Sphere(radius=0.5, boundary="vacuum")
    cell = yamc.Cell(name="sphere", region=sphere.below, material=material)
    source = yamc.NeutronSource(position=(0, 0, 0), energy=yamc.sources.Discrete([0.0253], [1.0]))
    if tallies is None:
        tallies = [yamc.Tally(scores=["flux"], name="flux"), yamc.Tally(scores=["(n,gamma)"], name="capture")]
    return yamc.Model(geometry=yamc.Geometry([cell]), source=source, tallies=tallies, verbose=[]), tallies


def test_the_nominal_mean_is_unchanged_and_the_uncertainty_is_reported():
    if not _has_covariance():
        pytest.skip("Fe56 fixture carries no covariance")
    model, (_, plain_capture) = _model()
    plain = model.simulate_transport(total_particles=3000, seed=2, threads=1)
    model, (_, capture) = _model()
    results = model.simulate_transport(
        total_particles=3000,
        seed=2,
        threads=1,
        data_uncertainty=yamc.DataUncertainty(seed=5, samples=16),
    )
    assert results[capture].mean == plain[plain_capture].mean
    assert plain[plain_capture].nuclear_data_standard_deviation is None

    sigma = results[capture].nuclear_data_standard_deviation[0]
    mean = results[capture].mean[0]
    # Fe56 thermal capture is known to a few percent; on a target this thin
    # the rate follows the cross section, so its relative sigma is that.
    assert 0.0 < sigma / mean < 0.2
    assert results[capture].replica_mean[0] > 0.0
    assert results[capture].replica_standard_error[0] > 0.0
    assert results[capture].nuclear_data_variance_negative == [False]


def test_local_heating_moves_with_the_capture_it_comes_from():
    # At thermal energy nearly all of iron's local heating is capture's
    # E + Q, so on a thin target its relative sigma is capture's, not the
    # flux's: the deposit of each reaction moves with its cross section.
    if not _has_covariance():
        pytest.skip("Fe56 fixture carries no covariance")
    model, (capture, heating, flux) = _model(
        [
            yamc.Tally(scores=["(n,gamma)"], name="capture"),
            yamc.Tally(scores=["heating-local"], name="heating"),
            yamc.Tally(scores=["flux"], name="flux"),
        ]
    )
    results = model.simulate_transport(
        total_particles=3000,
        seed=2,
        threads=1,
        data_uncertainty=yamc.DataUncertainty(seed=5, samples=16),
    )

    def relative(tally):
        return results[tally].nuclear_data_standard_deviation[0] / results[tally].mean[0]

    assert relative(heating) == pytest.approx(relative(capture), rel=0.02)
    assert relative(heating) > 10 * relative(flux)


def test_a_convergence_target_is_refused_on_a_fixed_replica_count():
    # Transport carries a fixed number of replica weights per history, so a
    # target it would never stop on is refused rather than ignored. Checked
    # before any data is read, so no covariance is needed.
    model, _ = _model()
    with pytest.raises(ValueError, match="remove convergence"):
        model.simulate_transport(
            total_particles=10, data_uncertainty=yamc.DataUncertainty(convergence=0.05)
        )


def test_unsupported_requests_are_refused_with_the_reason():
    if not _has_covariance():
        pytest.skip("Fe56 fixture carries no covariance")
    model, _ = _model()
    with pytest.raises(ValueError, match="cross sections only"):
        model.simulate_transport(
            total_particles=10, data_uncertainty=yamc.DataUncertainty(sources=["half_life"])
        )
    with pytest.raises(ValueError, match="attribution"):
        model.simulate_transport(
            total_particles=10, data_uncertainty=yamc.DataUncertainty(attribution=True)
        )
    every = ["cross_sections", "flux_spectrum", "half_life", "decay_branching", "statistical", "decay_energy", "decay_photon_lines", "fission_yield"]
    with pytest.raises(ValueError, match="cross sections only"):
        model.simulate_transport(
            total_particles=10, data_uncertainty=yamc.DataUncertainty(sources=every)
        )
