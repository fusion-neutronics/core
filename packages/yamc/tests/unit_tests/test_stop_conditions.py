"""Tests for the simulation stop conditions on ``simulate_transport`` /
``simulate_transmutation``.

A run ends at the first satisfied of {``total_particles`` exhausted,
``max_runtime`` elapsed, convergence targets met}. ``total_particles`` is
optional (``None`` = no particle cap); a run with no stop condition at all
goes on until every tally passes its statistical checks. Run from the repo root so the ``tests/*.arrow`` data paths
resolve.
"""

import pytest

import yamc


def _build_model(photon_data=None, tallies=None, **model_kwargs):
    inner = yamc.Sphere(x0=0, y0=0, z0=0, radius=1.0)
    outer = yamc.Sphere(x0=0, y0=0, z0=0, radius=200.0, boundary="vacuum")
    breeder = yamc.Material(
        composition={"Li6": 0.5, "Li7": 0.5}, density=2.0, temperature=294
    )
    breeder.read_nuclear_data(
        {"Li6": "tests/Li6.arrow", "Li7": "tests/Li7.arrow"}, photon_data=photon_data
    )
    void = yamc.Cell(name="void", region=inner.below)
    cell = yamc.Cell(name="breeder", region=inner.above & outer.below, material=breeder)
    geometry = yamc.Geometry([void, cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.fusion_neutron_spectrum(20000.0), position=(0, 0, 0)
    )
    tally = yamc.Tally(name="tbr", cells=cell, scores=["H3-production"])
    return yamc.Model(
        geometry=geometry,
        tallies=[tally] if tallies is None else tallies(cell),
        source=source,
        verbose=[],
        **model_kwargs,
    )


def test_no_stop_condition_runs_until_checks_pass():
    # A bare call stops once every scored bin passes the standard checks
    # (relative error and variance of the variance) at two checkpoints in a
    # row.
    model = _build_model()
    results = model.simulate_transport()
    assert results.stop_reason == "statistical_checks"
    r = results["tbr"]
    assert r.statistical_checks.passed
    assert r.relative_error[0] < 0.10
    assert r.aggregate_variance_of_variance < 0.10


@pytest.mark.parametrize("level", ["loose", "standard", "strict"])
def test_every_convergence_level_stops(level):
    model = _build_model()
    r = model.simulate_transport(convergence=level)["tbr"]
    assert r.relative_error[0] < 0.10


def test_every_bin_of_a_binned_tally_converges():
    # An energy-binned tally waits for every scored bin, not just its total.
    model = _build_model(
        tallies=lambda cell: [
            yamc.Tally(
                name="spectrum",
                cells=cell,
                scores=["flux"],
                energy_bins=[1e-5, 1e3, 1e5, 1e6, 2e7],
            )
        ]
    )
    r = model.simulate_transport()["spectrum"]
    scored = [e for m, e in zip(r.mean, r.relative_error) if m > 0]
    assert len(scored) > 1 and all(e < 0.10 for e in scored)


def test_unknown_convergence_level_lists_the_levels():
    model = _build_model()
    with pytest.raises(ValueError, match='"loose", "standard"'):
        model.simulate_transport(convergence="medium")


def test_convergence_with_a_stop_condition_raises():
    model = _build_model()
    with pytest.raises(ValueError, match="cannot be combined"):
        model.simulate_transport(total_particles=1000, convergence="loose")


def test_no_stop_condition_without_tallies_raises():
    model = _build_model(tallies=lambda cell: [])
    with pytest.raises(ValueError, match="no tallies"):
        model.simulate_transport()


def test_total_particles_zero_raises():
    # 0 is an error, not "unlimited" (that is None).
    model = _build_model()
    with pytest.raises(ValueError, match="must be positive"):
        model.simulate_transport(total_particles=0, seed=1)


def test_total_particles_only_runs_exactly_that_many():
    model = _build_model()
    results = model.simulate_transport(total_particles=3000, seed=1)
    assert results.stop_reason == "total_particles"
    assert results["tbr"].n_histories == 3000


def test_max_runtime_only_is_not_capped_at_1000():
    # Regression: a time-only run must NOT stop at a phantom 1000-particle
    # default. With total_particles=None it runs until the wall-time budget,
    # so a couple of seconds processes far more than 1000 histories. (The
    # uncapped chunk ramp checkpoints at 100 then 1100 histories, so even a
    # slow machine clears 1000 long before a 2 s budget elapses.)
    model = _build_model()
    results = model.simulate_transport(max_runtime=(2, "s"), seed=1)
    assert results.stop_reason == "max_runtime"
    r = results["tbr"]
    assert r.n_histories > 1000
    assert r.aggregate_mean > 0.0


def test_uncapped_run_stops_on_convergence():
    # total_particles=None with only a convergence target: the guard passes
    # (a condition is set) and the uncapped loop stops when the target is met.
    model = _build_model()
    model.convergence_targets = [
        yamc.ConvergenceTarget("relative_error", 0.10, tally="tbr")
    ]
    results = model.simulate_transport(seed=1)
    assert results.stop_reason == "convergence_targets"
    r = results["tbr"]
    assert r.n_histories > 0
    assert r.aggregate_relative_error <= 0.12


@pytest.mark.parametrize(
    "selector, expected",
    [({"tally": "tbrr"}, "tally='tbrr'"), ({"tally_id": 99}, "tally_id=99")],
)
def test_convergence_target_matching_no_tally_raises(selector, expected):
    # A target that names no tally in the model is never evaluated, so the run
    # used to treat it as met and stop at the first checkpoint. Fail fast.
    model = _build_model()
    model.convergence_targets = [
        yamc.ConvergenceTarget("relative_error", 0.01, **selector)
    ]
    with pytest.raises(ValueError, match=f"{expected} matches no tally"):
        model.simulate_transport(max_runtime=(1, "h"), seed=1)


@pytest.mark.skipif(
    not yamc.parallel.gpu_available(),
    reason="no GPU with f64 compute available, or yamc was built without the `gpu` feature",
)
def test_gpu_coupled_uncapped_run_stops_on_convergence():
    # The coupled (secondary photons on) GPU launch loop stops on the target
    # too. No cap, no budget: the target alone ends the run.
    model = _build_model(
        photon_data={"Li": "tests/Li.arrow"}, transport_secondary_photons=True
    )
    model.convergence_targets = [
        yamc.ConvergenceTarget("relative_error", 0.10, tally="tbr")
    ]
    r = model.simulate_transport(seed=1, compute="gpu")["tbr"]
    assert r.n_histories > 0
    assert 0.0 < r.aggregate_relative_error <= 0.10


@pytest.mark.skipif(
    not yamc.parallel.gpu_available(),
    reason="no GPU with f64 compute available, or yamc was built without the `gpu` feature",
)
def test_gpu_uncapped_run_stops_on_convergence():
    # The GPU twin of test_uncapped_run_stops_on_convergence: no cap, no
    # budget, the launch loop ends when the target is met and the aggregate
    # relative error it reports honours it.
    model = _build_model()
    model.convergence_targets = [
        yamc.ConvergenceTarget("relative_error", 0.10, tally="tbr")
    ]
    r = model.simulate_transport(seed=1, compute="gpu")["tbr"]
    assert r.n_histories > 0
    assert 0.0 < r.aggregate_relative_error <= 0.10


def test_simulate_transmutation_requires_a_stop_condition():
    # transmutation's per-step stop conditions are total_particles and
    # max_runtime; with neither set the run is rejected before any transport.
    model = _build_model()
    schedule = yamc.PulseSchedule([yamc.Cooldown(duration=86400.0)])
    with pytest.raises(ValueError, match="per-step stop condition"):
        model.simulate_transmutation(method="coupled", schedule=schedule)
