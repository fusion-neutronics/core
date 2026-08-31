"""Tests for the simulation stop conditions on ``simulate_transport`` /
``simulate_transmutation`` (issue #230).

A run ends at the first satisfied of {``total_particles`` exhausted,
``max_runtime`` elapsed, convergence targets met}. ``total_particles`` is
optional (``None`` = no particle cap); a run with no stop condition at all is
rejected up front. Run from the repo root so the ``tests/*.arrow`` data paths
resolve.
"""

import pytest

import yamc


def _build_model():
    inner = yamc.Sphere(x0=0, y0=0, z0=0, radius=1.0)
    outer = yamc.Sphere(x0=0, y0=0, z0=0, radius=200.0, boundary="vacuum")
    breeder = yamc.Material(
        composition={"Li6": 0.5, "Li7": 0.5}, density=2.0, temperature=294
    )
    breeder.read_nuclear_data({"Li6": "tests/Li6.arrow", "Li7": "tests/Li7.arrow"})
    void = yamc.Cell(name="void", region=inner.below)
    cell = yamc.Cell(name="breeder", region=inner.above & outer.below, material=breeder)
    geometry = yamc.Geometry([void, cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.fusion_neutron_spectrum(20000.0), position=(0, 0, 0)
    )
    tally = yamc.Tally(name="tbr", cells=cell, scores=["H3-production"])
    return yamc.Model(geometry=geometry, tallies=[tally], source=source, verbose=[])


def test_no_stop_condition_raises():
    # A bare call sets none of the three conditions, so the run would never
    # end: fail fast instead of looping forever (or the old phantom 1000 cap).
    model = _build_model()
    with pytest.raises(ValueError, match="at least one stop condition"):
        model.simulate_transport()


def test_total_particles_zero_raises():
    # 0 is an error, not "unlimited" (that is None).
    model = _build_model()
    with pytest.raises(ValueError, match="must be positive"):
        model.simulate_transport(total_particles=0, seed=1)


def test_total_particles_only_runs_exactly_that_many():
    model = _build_model()
    r = model.simulate_transport(total_particles=3000, seed=1)["tbr"]
    assert r.n_histories == 3000


def test_max_runtime_only_is_not_capped_at_1000():
    # Regression: a time-only run must NOT stop at a phantom 1000-particle
    # default. With total_particles=None it runs until the wall-time budget,
    # so a couple of seconds processes far more than 1000 histories. (The
    # uncapped chunk ramp checkpoints at 100 then 1100 histories, so even a
    # slow machine clears 1000 long before a 2 s budget elapses.)
    model = _build_model()
    r = model.simulate_transport(max_runtime=(2, "s"), seed=1)["tbr"]
    assert r.n_histories > 1000
    assert r.aggregate_mean > 0.0


def test_uncapped_run_stops_on_convergence():
    # total_particles=None with only a convergence target: the guard passes
    # (a condition is set) and the uncapped loop stops when the target is met.
    model = _build_model()
    model.convergence_targets = [
        yamc.ConvergenceTarget("relative_error", 0.10, tally="tbr")
    ]
    r = model.simulate_transport(seed=1)["tbr"]
    assert r.n_histories > 0
    assert r.aggregate_relative_error <= 0.12


def test_gpu_rejects_convergence_only_stop():
    # GPU stops on total_particles and/or max_runtime, not (yet) on convergence
    # targets (#241), so a convergence-only run on GPU is rejected up front
    # rather than launching forever.
    if not yamc.parallel.gpu_available():
        pytest.skip("no f64-capable GPU available")
    model = _build_model()
    model.convergence_targets = [
        yamc.ConvergenceTarget("relative_error", 0.10, tally="tbr")
    ]
    with pytest.raises(ValueError, match="cannot yet.*stop on convergence"):
        model.simulate_transport(compute="gpu")  # None total, no max_runtime


def test_simulate_transmutation_requires_a_stop_condition():
    # transmutation's per-step stop conditions are total_particles and
    # max_runtime; with neither set the run is rejected before any transport.
    model = _build_model()
    schedule = yamc.PulseSchedule([yamc.Cooldown(duration=86400.0)])
    with pytest.raises(ValueError, match="per-step stop condition"):
        model.simulate_transmutation(method="coupled", schedule=schedule)
