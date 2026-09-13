"""A GPU run must say so when it cannot honour a Model setting (core#23).

Three settings the GPU dispatch reads differently from the CPU used to be
reported only under a verbosity flag the user can switch off, so a
``verbose=[]`` run did something other than what was asked and said nothing:

- ``tracking_mode``: the kernels always surface-track. The notice is now
  printed to stderr at every ``verbose`` setting.
- ``convergence_targets``: the neutron launch loops now stop on them
  (core#29). The photon launch loops still cannot, so a model that transports
  photons is refused up front, whatever else is set, rather than run silently
  to the cap as it once did.
- ``gpu_max_steps_per_particle``: a launch that truncated histories under-counts
  the flux. That was a gated warning; it is now an error, so the under-counted
  tallies are never returned.

The photon refusal happens before any GPU is touched, so that test runs
everywhere. The others need an f64 Vulkan adapter and skip without one.
"""

import os

import pytest

yamc = pytest.importorskip("yamc")

TESTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "crates", "yamc", "tests"
)

needs_gpu = pytest.mark.skipif(
    not yamc.parallel.gpu_available(),
    reason="no GPU with f64 compute available, or yamc was built without the `gpu` feature",
)


def _sphere(nuclide, radius, photon_element=None, **model_kwargs):
    material = yamc.Material(composition={nuclide: 1.0}, density=1.0, temperature=294)
    photon_data = (
        {photon_element: os.path.join(TESTS_DIR, f"{photon_element}.arrow")}
        if photon_element
        else None
    )
    material.read_nuclear_data(
        {nuclide: os.path.join(TESTS_DIR, f"{nuclide}.arrow")}, photon_data=photon_data
    )
    sphere = yamc.Sphere(radius=radius, boundary="vacuum")
    cell = yamc.Cell(name="s", region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    tally = yamc.Tally(scores=["flux"], name="t", cells=cell)
    source = yamc.NeutronSource(
        position=(0.0, 0.0, 0.0), energy=yamc.sources.Discrete([14.06e6], [1.0])
    )
    return yamc.Model(
        geometry=geometry, tallies=[tally], source=source, verbose=[], **model_kwargs
    )


@pytest.mark.parametrize(
    "run_kwargs",
    [
        pytest.param({"total_particles": 1000}, id="with-particle-cap"),
        pytest.param({"max_runtime": 1.0}, id="with-runtime-budget"),
        pytest.param({}, id="targets-only"),
    ],
)
def test_convergence_targets_are_refused_on_gpu_for_photon_models(run_kwargs):
    # The photon launch loops cannot evaluate a target yet, so a model that
    # transports photons is refused in the same words for every combination,
    # before any adapter is touched (this holds on hosts without a GPU too).
    # Before core#29 every GPU model was refused this way; the neutron loops
    # now stop on the targets (see the test below).
    # The material carries the Fe photon data so the model is a valid photon
    # model; the refusal is then the convergence one, not a missing-data error.
    model = _sphere("Fe56", 10.0, photon_element="Fe", transport_secondary_photons=True)
    model.convergence_targets = [yamc.ConvergenceTarget("relative_error", 0.05, tally="t")]
    with pytest.raises(ValueError, match="cannot stop on convergence targets") as exc:
        model.simulate_transport(seed=1, compute="gpu", **run_kwargs)
    # The message must name the way out.
    assert "Model.convergence_targets" in str(exc.value)
    assert "1 target(s)" in str(exc.value)
    assert "photon" in str(exc.value)


@needs_gpu
def test_convergence_targets_stop_a_neutron_run_on_gpu():
    # A neutron-only model with only a convergence target: no particle cap, no
    # time budget, the launch loop stops when the target is met (core#29). The
    # target is loose so the first launch chunk already satisfies it; the point
    # is that the run ends, that the aggregate relative error the GPU reports
    # honours the target, and that a cap set alongside is not what stopped it.
    model = _sphere("Fe56", 10.0)
    model.convergence_targets = [yamc.ConvergenceTarget("relative_error", 0.05, tally="t")]
    results = model.simulate_transport(seed=1, compute="gpu")
    r = results["t"]
    assert r.n_histories > 0
    assert 0.0 < r.aggregate_relative_error <= 0.05
    capped = _sphere("Fe56", 10.0)
    capped.convergence_targets = [yamc.ConvergenceTarget("relative_error", 0.05, tally="t")]
    rc = capped.simulate_transport(seed=1, compute="gpu", total_particles=10_000_000)["t"]
    assert rc.n_histories == r.n_histories


def test_convergence_targets_still_run_on_cpu():
    # The refusal is GPU-only: the CPU honours the targets, so the same model
    # runs there (and stops early, but that is the stop-condition suite's job).
    model = _sphere("Fe56", 10.0)
    model.convergence_targets = [yamc.ConvergenceTarget("relative_error", 0.5, tally="t")]
    results = model.simulate_transport(total_particles=500, seed=1)
    assert results["t"].n_histories <= 500


@needs_gpu
@pytest.mark.parametrize("mode", ["hybrid", "woodcock"])
def test_tracking_mode_notice_survives_verbose_off(mode, capfd):
    # verbose=[] used to silence the notice, so a Woodcock or Hybrid request
    # was dropped without a word. The flux is still unbiased, so the run
    # proceeds; the point is that the user is told at every verbosity.
    model = _sphere("Fe56", 10.0, tracking_mode=mode)
    model.simulate_transport(total_particles=200, seed=1, compute="gpu")
    err = capfd.readouterr().err
    assert "ignores tracking_mode" in err, err
    assert "surface-track" in err, err


@needs_gpu
def test_surface_tracking_on_gpu_prints_no_notice(capfd):
    # The default asks for exactly what the kernel does, so nothing to say.
    model = _sphere("Fe56", 10.0)
    model.simulate_transport(total_particles=200, seed=1, compute="gpu")
    assert "ignores tracking_mode" not in capfd.readouterr().err


@needs_gpu
def test_binding_step_cap_is_an_error_on_gpu():
    # 14 MeV neutrons in 35 cm of deuterium scatter many hundreds of times
    # before leaking, so a 1000-step cap truncates histories and the GPU flux
    # would come out ~10% low. That used to be a stderr warning gated on
    # verbose.summary, i.e. nothing at all on this verbose=[] model, with the
    # under-counted tallies returned as if valid. It is an error now.
    model = _sphere("H2", 35.0, gpu_max_steps_per_particle=1000)
    with pytest.raises(ValueError, match="gpu_max_steps_per_particle=1000") as exc:
        model.simulate_transport(total_particles=2000, seed=7, compute="gpu")
    assert "Raise gpu_max_steps_per_particle" in str(exc.value)


@needs_gpu
def test_default_step_cap_does_not_bind_on_gpu():
    # The same model with the default cap runs to completion: the error is
    # about a binding cap, not about deuterium.
    model = _sphere("H2", 35.0)
    results = model.simulate_transport(total_particles=2000, seed=7, compute="gpu")
    assert results["t"].n_histories == 2000
