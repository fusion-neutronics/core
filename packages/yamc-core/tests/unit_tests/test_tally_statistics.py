"""Tests for the tally statistical-reliability outputs.

Covers the per-bin ``variance`` getter, the aggregate reliability scalars
(variance of the variance, skewness, kurtosis, tail slope), the
``statistical_checks`` verdict, the ``convergence_history`` series, the raw
``score_pdf``, and precision ``ConvergenceTarget`` early-stop. Run from the repo root so
the ``tests/*.arrow`` data paths resolve.
"""

import math

import pytest

import yamc


def _build_model(particles=5000):
    inner = yamc.Sphere(x0=0, y0=0, z0=0, radius=1.0)
    outer = yamc.Sphere(x0=0, y0=0, z0=0, radius=200.0, boundary="vacuum")
    breeder = yamc.Material(
        composition={"Li6": 0.5, "Li7": 0.5}, density=2.0, temperature=294
    )
    breeder.read_nuclear_data({"Li6": "tests/Li6.arrow", "Li7": "tests/Li7.arrow"})
    void = yamc.Cell(name="void", region=inner.below)
    cell = yamc.Cell(
        name="breeder", region=inner.above & outer.below, material=breeder
    )
    geometry = yamc.Geometry([void, cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.fusion_neutron_spectrum(20000.0), position=(0, 0, 0)
    )
    tally = yamc.Tally(name="tbr", cells=cell, scores=["H3-production"])
    model = yamc.Model(
        geometry=geometry,
        tallies=[tally],
        source=source,
        verbose=[],
    )
    return model, {"total_particles": particles, "seed": 1}


def _tbr_result(particles=5000):
    model, run_kwargs = _build_model(particles)
    return model.simulate_transport(**run_kwargs)["tbr"]


def test_variance_is_std_dev_squared():
    r = _tbr_result()
    assert len(r.variance) == len(r.standard_deviation)
    for v, s in zip(r.variance, r.standard_deviation):
        assert v == pytest.approx(s * s)


def test_aggregate_reliability_scalars_are_sane():
    r = _tbr_result()
    assert r.aggregate_mean > 0.0
    assert 0.0 <= r.aggregate_relative_error < 1.0
    assert r.aggregate_variance_of_variance >= 0.0
    assert math.isfinite(r.aggregate_skewness)
    assert math.isfinite(r.aggregate_kurtosis)
    assert r.aggregate_tail_slope >= 0.0


def test_statistical_checks_verdict():
    c = _tbr_result().statistical_checks
    assert isinstance(c.passed, bool)
    assert c.n_evaluated >= 1
    assert 0 <= c.n_passed <= c.n_evaluated
    for value in (
        c.relative_error_ok,
        c.variance_of_variance_ok,
        c.tail_slope_ok,
        c.mean_stable,
    ):
        assert value is None or isinstance(value, bool)
    assert isinstance(c.summary(), str)
    # Printing a result includes the verdict (the discoverability feature).
    assert "statistical checks" in repr(_tbr_result())


def test_convergence_history():
    hist = _tbr_result().convergence_history
    assert len(hist) >= 1
    last = hist[-1]
    assert last.n_histories > 0
    assert last.mean > 0.0
    assert last.relative_error >= 0.0
    assert last.figure_of_merit >= 0.0


def test_score_pdf_raw():
    pdf = _tbr_result().score_pdf
    assert len(pdf.counts) > 0
    assert len(pdf.bin_edges) == len(pdf.counts) + 1
    assert len(pdf.bin_centers) == len(pdf.counts)
    assert sum(pdf.counts) + pdf.zero > 0


def test_to_numpy_variance():
    np = pytest.importorskip("numpy")
    r = _tbr_result()
    arr = r.to_numpy("variance")
    assert arr.shape == tuple(r.shape)
    assert np.allclose(arr.ravel(), r.variance)


def test_trigger_stops_run_early():
    model, run_kwargs = _build_model(particles=50000)
    model.convergence_targets = [yamc.ConvergenceTarget("relative_error", 0.10, tally="tbr")]
    r = model.simulate_transport(**run_kwargs)["tbr"]
    # TBR converges quickly, so a 10% target is met well before all histories.
    assert r.n_histories < 50000
    assert r.aggregate_relative_error <= 0.12


def test_max_runtime_stops_run_early():
    # A tiny wall-time budget on a large run stops at the first chunk
    # checkpoint, yet the returned result is finalized and statistically
    # valid for the histories completed.
    model, run_kwargs = _build_model(particles=500000)
    r = model.simulate_transport(**run_kwargs, max_runtime=0.001)["tbr"]
    assert r.n_histories < 500000
    assert r.aggregate_mean > 0.0
    # A real (positive) FOM, not just the 0.0 fallback, confirms the early-stop
    # finalized usable statistics.
    assert r.convergence_history[-1].figure_of_merit > 0.0


def test_max_runtime_large_is_no_op():
    # A budget far larger than the run takes is never the binding constraint,
    # so all requested histories run. Also exercises the (value, unit) tuple.
    model, run_kwargs = _build_model(particles=5000)
    r = model.simulate_transport(**run_kwargs, max_runtime=(1, "h"))["tbr"]
    assert r.n_histories == 5000


# max_runtime is now supported on compute='gpu' (#230 task 2); its GPU
# behaviour (uncapped time-only run, generous-budget no-op) is covered by
# test_simulate_transport_gpu.py, which is gated on a real GPU being present.


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_max_runtime_rejects_non_finite_and_negative(bad):
    # A non-finite or negative budget is a user error, not a silent no-op.
    model, run_kwargs = _build_model(particles=5000)
    with pytest.raises(ValueError):
        model.simulate_transport(**run_kwargs, max_runtime=bad)
