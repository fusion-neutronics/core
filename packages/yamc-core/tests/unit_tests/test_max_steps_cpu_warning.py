"""`gpu_max_steps_per_particle` warns when it is set and then ignored (#302).

Only the GPU kernel applies the cap: it needs a bound in its loop condition
(driver watchdog, lockstep workgroups). CPU transport runs `while particle.alive`
and ends a history on absorption, leakage or the lost-particle diagnostics, so
the value does nothing there. That was documented but silent, and silently
ignoring a value the user typed is not the same as telling them.

The warning fires only when the cap was set explicitly, never for the default
that every model carries, which is what makes it worth having.
"""

import os
import warnings

import pytest

yamc = pytest.importorskip("yamc")

TESTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "crates", "yamc", "tests"
)
FE56 = os.path.join(TESTS_DIR, "Fe56.arrow")


def _model(**kwargs):
    material = yamc.Material(
        composition={"Fe56": 1.0}, density=7.874, temperature=294, name="iron"
    )
    material.read_nuclear_data({"Fe56": FE56})
    sphere = yamc.Sphere(radius=10.0, boundary="vacuum")
    cells = [yamc.Cell(name="core", region=sphere.below, material=material)]
    # The geometry assigns the cell ids, so it must exist before a CellFilter is
    # built from those cells (issue #305).
    geometry = yamc.Geometry(cells)
    tally = yamc.Tally(scores=["flux"], name="t", cells=cells)
    source = yamc.NeutronSource(
        position=(0.0, 0.0, 0.0), energy=yamc.sources.Discrete([14.06e6], [1.0])
    )
    return yamc.Model(
        geometry=geometry,
        tallies=[tally],
        source=source,
        verbose=[],
        **kwargs,
    )


def _max_steps_warnings(model, **run_kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.simulate_transport(total_particles=500, seed=1, **run_kwargs)
    return [str(w.message) for w in caught if "gpu_max_steps_per_particle" in str(w.message)]


def test_explicit_cap_warns_on_cpu():
    hits = _max_steps_warnings(_model(gpu_max_steps_per_particle=50_000))
    assert len(hits) == 1, f"expected one warning, got {hits}"
    assert "has no effect" in hits[0]
    assert "50000" in hits[0], "the warning should quote the value that was set"


def test_cap_set_through_the_setter_warns_on_cpu():
    model = _model()
    model.gpu_max_steps_per_particle = 4_242
    hits = _max_steps_warnings(model)
    assert len(hits) == 1, f"expected one warning, got {hits}"
    assert "4242" in hits[0]


def test_default_cap_is_silent():
    """Every model carries the default, so warning about it would be noise."""
    assert _max_steps_warnings(_model()) == []


def test_getter_still_reports_the_default():
    """Taking `None` as "not given" must not change the visible default."""
    assert _model().gpu_max_steps_per_particle == 100_000
    assert _model(gpu_max_steps_per_particle=7).gpu_max_steps_per_particle == 7


def test_gpu_run_is_silent_because_the_cap_applies_there():
    if not yamc.parallel.gpu_available():
        pytest.skip("no f64 GPU available")
    hits = _max_steps_warnings(
        _model(gpu_max_steps_per_particle=50_000), compute="gpu"
    )
    assert hits == [], f"the GPU honours the cap, so it must not warn: {hits}"


def test_transmutation_warns_because_it_is_cpu_only():
    """`simulate_transmutation` has no `compute=`, so its solves are CPU."""
    chain = os.path.join(TESTS_DIR, "transmutation-endf-b8.1-sfr.arrow")
    yamc.transmutation_decay_data = chain
    yamc.transmutation_reactions = chain
    yamc.transmutation_fission_yields = chain

    import math

    radius = 5.0
    material = yamc.Material(
        composition={"Fe56": 1.0},
        density=7.874,
        temperature=294,
        name="iron_t",
        id=301,
        transmutable=True,
        volume=4.0 / 3.0 * math.pi * radius**3,
    )
    material.read_nuclear_data({"Fe56": FE56})
    sphere = yamc.Sphere(radius=radius, boundary="vacuum")
    geometry = yamc.Geometry(
        [yamc.Cell(name="core", region=sphere.below, material=material)]
    )
    source = yamc.NeutronSource(
        position=(0.0, 0.0, 0.0), energy=yamc.sources.Discrete([14.06e6], [1.0])
    )
    model = yamc.Model(
        geometry=geometry, source=source, verbose=[], gpu_max_steps_per_particle=1_234
    )
    schedule = yamc.PulseSchedule(
        [
            yamc.Pulse(rate=1e14, duration=3600.0, source=source),
            yamc.Cooldown(duration=3600.0),
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.simulate_transmutation(
            method="independent", schedule=schedule, total_particles=500, seed=1
        )
    hits = [str(w.message) for w in caught if "gpu_max_steps_per_particle" in str(w.message)]
    assert len(hits) == 1, f"expected one warning, got {hits}"
    assert "1234" in hits[0]
