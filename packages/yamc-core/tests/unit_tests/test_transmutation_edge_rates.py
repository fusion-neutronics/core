"""Per-edge reaction rates carried onto TransmutationResults (issue #490).

The solve computes the rate of every production edge to build its burnup
matrix, and used to discard them, leaving a consumer able to enumerate the
routes into a product but not weight them. These tests check that what comes
back is the rate that actually drove the step, in the same shape from either
method.
"""

import os

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


def _run(method, *, rate=1.0e18, schedule=None, seed=42):
    """One thermal-driven iron sphere, whose dominant edge is Fe56(n,gamma)Fe57."""
    material = yamc.Material(
        composition={"Fe56": 1.0},
        density=7.87,
        name="iron",
        transmutable=True,
        volume=4188.79,  # 4/3 pi 10^3
        temperature=294,
    )
    material.read_nuclear_data({"Fe56": os.path.join(TESTS_DIR, "Fe56.arrow")})
    sphere = yamc.Sphere(radius=10.0, boundary="vacuum")
    cell = yamc.Cell(name="sphere", region=sphere.below, material=material)
    source = yamc.NeutronSource(
        position=(0, 0, 0), energy=yamc.sources.Discrete([0.0253], [1.0])
    )
    model = yamc.Model(geometry=yamc.Geometry([cell]), source=source, verbose=[])
    if schedule is None:
        schedule = yamc.PulseSchedule(
            [yamc.Pulse(rate=rate, duration=HOUR, source=source)]
        )
    return model.simulate_transmutation(
        method=method, schedule=schedule, total_particles=20_000, seed=seed
    )


@pytest.mark.parametrize("method", ["coupled", "independent"])
def test_edges_are_reported_for_both_methods(method):
    """Same shape from either method, and the dominant edge is in it."""
    rates = _run(method).get_reaction_rates(material_id=MAT_ID, step=0)

    assert rates, "no per-edge rates recorded for the irradiation step"
    (target, rate), = rates["Fe56"]["(n,gamma)"]
    assert target == "Fe57"
    assert rate > 0.0

    for parent, kinds in rates.items():
        assert isinstance(parent, str)
        for kind, edges in kinds.items():
            assert isinstance(kind, str)
            for edge_target, edge_rate in edges:
                assert edge_target is None or isinstance(edge_target, str)
                assert edge_rate > 0.0


@pytest.mark.parametrize("method", ["coupled", "independent"])
def test_the_reported_rate_is_the_one_that_drove_the_step(method):
    """Tie the number to the inventory: dN_Fe57 ~ rate * N_Fe56 * dt.

    A rate reported from anywhere other than the step's own solve would not
    reproduce the step's own burnup. The step is short enough that the
    first-order estimate holds to well under a percent.
    """
    results = _run(method)
    rates = results.get_reaction_rates(material_id=MAT_ID, step=0)
    (_, rate), = rates["Fe56"]["(n,gamma)"]

    n_fe56 = results.get_nuclide_density(MAT_ID, "Fe56", 0)
    fe57 = results.get_nuclide_evolution(MAT_ID, "Fe57")
    produced = fe57[1] - fe57[0]

    assert produced == pytest.approx(rate * n_fe56 * HOUR, rel=1e-3)


def test_independent_rates_scale_with_the_source_rate():
    """In independent mode one transport is scaled per step, so the edges are too."""
    slow = _run("independent", rate=1.0e18)
    fast = _run("independent", rate=2.0e18)

    (_, slow_rate), = slow.get_reaction_rates(MAT_ID, 0)["Fe56"]["(n,gamma)"]
    (_, fast_rate), = fast.get_reaction_rates(MAT_ID, 0)["Fe56"]["(n,gamma)"]

    assert fast_rate == pytest.approx(2.0 * slow_rate, rel=1e-12)


def test_a_cooldown_step_reports_no_edges():
    """Decay drives no reaction edge, so the map is empty rather than absent."""
    source = yamc.NeutronSource(
        position=(0, 0, 0), energy=yamc.sources.Discrete([0.0253], [1.0])
    )
    schedule = yamc.PulseSchedule(
        [
            yamc.Pulse(rate=1.0e18, duration=HOUR, source=source),
            yamc.Cooldown(duration=HOUR),
        ]
    )
    results = _run("independent", schedule=schedule)

    assert results.get_reaction_rates(MAT_ID, 0)
    assert results.get_reaction_rates(MAT_ID, 1) == {}


def test_unknown_material_or_step_is_none():
    """A step index past the schedule is None, distinct from an empty step."""
    results = _run("independent")

    assert results.get_reaction_rates(material_id=999, step=0) is None
    assert results.get_reaction_rates(material_id=MAT_ID, step=1) is None


def test_step_index_follows_timesteps_not_the_composition_index():
    """One rate map per schedule step, one more composition than that."""
    source = yamc.NeutronSource(
        position=(0, 0, 0), energy=yamc.sources.Discrete([0.0253], [1.0])
    )
    schedule = yamc.PulseSchedule(
        [
            yamc.Pulse(rate=1.0e18, duration=HOUR, source=source),
            yamc.Cooldown(duration=HOUR),
        ]
    )
    results = _run("independent", schedule=schedule)

    assert results.num_steps == len(results.timesteps) == 2
    assert results.get_reaction_rates(MAT_ID, len(results.timesteps) - 1) is not None
    assert results.get_reaction_rates(MAT_ID, len(results.timesteps)) is None
    # The composition index has the initial state at 0, so it runs one further.
    assert results.get_material(MAT_ID, len(results.timesteps)) is not None
