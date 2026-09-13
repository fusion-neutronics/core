"""A GPU run must say so when it cannot honour a Model setting (core#23).

Three settings the GPU dispatch reads differently from the CPU used to be
reported only under a verbosity flag the user can switch off, so a
``verbose=[]`` run did something other than what was asked and said nothing:

- ``tracking_mode``: the kernels always surface-track. The notice is now
  printed to stderr at every ``verbose`` setting.
- ``convergence_targets``: the launch loop cannot stop on them. With a
  particle cap also set, the run used to go silently to the cap. It is now
  refused up front, whatever else is set.
- ``max_steps_per_particle``: a launch that truncated histories under-counts
  the flux. That was a gated warning; it is now an error, so the under-counted
  tallies are never returned.

The convergence refusal happens before any GPU is touched, so those tests run
everywhere. The other two need an f64 Vulkan adapter and skip without one.
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


def _sphere(nuclide, radius, **model_kwargs):
    material = yamc.Material(composition={nuclide: 1.0}, density=1.0, temperature=294)
    material.read_nuclear_data({nuclide: os.path.join(TESTS_DIR, f"{nuclide}.arrow")})
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
def test_convergence_targets_are_refused_on_gpu(run_kwargs):
    # Before, only the targets-only case was refused (as "needs total_particles
    # or max_runtime"); with a cap or budget present the targets were dropped
    # in silence and the run went to the cap. Now every combination is refused
    # in the same words, before any adapter is touched, so this holds on hosts
    # without a GPU too.
    model = _sphere("Fe56", 10.0)
    model.convergence_targets = [yamc.ConvergenceTarget("relative_error", 0.05, tally="t")]
    with pytest.raises(ValueError, match="cannot stop on convergence targets") as exc:
        model.simulate_transport(seed=1, compute="gpu", **run_kwargs)
    # The message must name the way out.
    assert "Model.convergence_targets" in str(exc.value)
    assert "1 target(s)" in str(exc.value)


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
    model = _sphere("H2", 35.0, max_steps_per_particle=1000)
    with pytest.raises(ValueError, match="max_steps_per_particle=1000") as exc:
        model.simulate_transport(total_particles=2000, seed=7, compute="gpu")
    assert "Raise max_steps_per_particle" in str(exc.value)


@needs_gpu
def test_default_step_cap_does_not_bind_on_gpu():
    # The same model with the default cap runs to completion: the error is
    # about a binding cap, not about deuterium.
    model = _sphere("H2", 35.0)
    results = model.simulate_transport(total_particles=2000, seed=7, compute="gpu")
    assert results["t"].n_histories == 2000
