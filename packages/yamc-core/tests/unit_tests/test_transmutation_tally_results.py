"""The model's own tallies at every step of simulate_transmutation.

A thin Li6 sphere under a thermal source: its tritium production per source
particle is close to proportional to the Li6 density, so it falls visibly as
the irradiation burns the Li6 away.
"""

import math
import os

import pytest
import yamc

TESTS_DIR = os.path.join("crates", "yamc", "tests")
# The bred products with data. Pinned for the run and the rebuild alike, so
# which products transport sees does not hang on what earlier tests left set.
DATA = {n: os.path.join(TESTS_DIR, f"{n}.arrow") for n in ("Li6", "Li7")}
CHAIN = os.path.join(TESTS_DIR, "transmutation-endf-b8.1-sfr.arrow")
MAT_ID = 1
RADIUS = 5.0
PARTICLES = 4000
SEED = 7


def _model(atoms=None):
    """The sphere, of Li6 at the fixture's gram density or of ``atoms``
    (nuclide -> atom/barn-cm)."""
    composition, density, units = (
        ({"Li6": 1.0}, 0.002, "g/cm3")
        if atoms is None
        else (atoms, sum(atoms.values()), "atom/barn-cm")
    )
    material = yamc.Material(
        composition=composition,
        density=density,
        density_units=units,
        id=MAT_ID,
        transmutable=True,
        volume=4.0 / 3.0 * math.pi * RADIUS**3,
        temperature=294,
    )
    material.read_nuclear_data({n: DATA[n] for n in composition})
    sphere = yamc.Sphere(radius=RADIUS, boundary="vacuum")
    cell = yamc.Cell(name="breeder", region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        position=(0, 0, 0), energy=yamc.sources.Discrete([0.0253], [1.0])
    )
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-RADIUS] * 3, upper_right=[RADIUS] * 3, shape=[2, 3, 1]
    )
    tallies = [
        yamc.Tally(name="tbr", cells=cell, scores=["H3-production"]),
        yamc.Tally(
            name="spectrum",
            cells=cell,
            scores=["flux"],
            energy_bins=[1e-5, 0.1, 1.0, 2e7],
        ),
        yamc.Tally(name="mesh", mesh=mesh, scores=["flux"]),
    ]
    model = yamc.Model(geometry=geometry, source=source, tallies=tallies, verbose=[])
    return model, source, tallies


def _run(method="coupled"):
    # Set here, not by the conftest fixture, which runs after module fixtures.
    yamc.transmutation_decay_data = CHAIN
    yamc.transmutation_reactions = CHAIN
    yamc.transmutation_fission_yields = CHAIN
    yamc.cross_section_data = DATA
    model, source, tallies = _model()
    pulse = yamc.Pulse(rate=2.0e17, duration=1.0e6, source=source)
    schedule = yamc.PulseSchedule([pulse, pulse, pulse, yamc.Cooldown(duration=1.0e5)])
    results = model.simulate_transmutation(
        method=method, schedule=schedule, total_particles=PARTICLES, seed=SEED
    )
    return results, tallies


@pytest.fixture(scope="module")
def coupled():
    return _run("coupled")


def test_tbr_falls_as_li6_burns(coupled):
    results, (tbr, _, _) = coupled
    assert isinstance(results, yamc.TransmutationResults)
    li6 = results.get_nuclide_evolution(MAT_ID, "Li6")
    assert li6[3] < 0.7 * li6[0], "the schedule must burn a good share of the Li6"
    tbrs = [results.tally_results(step)[tbr].mean[0] for step in range(3)]
    assert tbrs[0] > tbrs[1] > tbrs[2] > 0.0
    means, std_devs = results.tally_evolution(tbr)
    assert [m[0] for m in means] == tbrs
    assert len(std_devs) == 3 and all(s[0] > 0.0 for s in std_devs)


def test_decay_only_step_is_none(coupled):
    results, _ = coupled
    assert len(results.timesteps) == 4
    assert results.tally_results(3) is None
    with pytest.raises(IndexError):
        results.tally_results(4)


def test_shapes_of_cell_energy_and_mesh_tallies(coupled):
    results, (tbr, spectrum, mesh) = coupled
    for step in range(3):
        step_results = results.tally_results(step)
        assert len(step_results) == 3, "only the model's own tallies come back"
        assert len(step_results[tbr].mean) == 1
        assert len(step_results[spectrum].mean) == 3
        assert len(step_results[mesh].mean) == 6
        assert step_results["mesh"].to_numpy().size == 6
        assert step_results.runs[0]["n_histories"] == PARTICLES


def test_step_zero_matches_simulate_transport(coupled):
    """The first step's transport is that of the initial model."""
    results, (tbr, _, _) = coupled
    model, _, (plain_tbr, _, _) = _model()
    plain = model.simulate_transport(total_particles=PARTICLES, seed=SEED)
    expected = plain[plain_tbr]
    got = results.tally_results(0)[tbr]
    sigma = math.hypot(expected.standard_deviation[0], got.standard_deviation[0])
    assert abs(got.mean[0] - expected.mean[0]) <= 4.0 * sigma


def test_matches_rebuilding_the_model_at_each_step(coupled):
    """What the workaround of rebuilding from get_material gives, within
    statistics. The rebuild keeps the bred nuclides transport saw (not only
    Li6): by the last step they absorb a visible share of the neutrons."""
    results, (tbr, _, _) = coupled
    for step in (1, 2):
        atoms = results.get_material(MAT_ID, step).get_atoms_per_barn_cm()
        model, _, (rebuilt_tbr, _, _) = _model(
            {n: atoms[n] for n in DATA if n in atoms}
        )
        rebuilt = model.simulate_transport(total_particles=PARTICLES, seed=SEED + step)
        expected = rebuilt[rebuilt_tbr]
        got = results.tally_results(step)[tbr]
        sigma = math.hypot(expected.standard_deviation[0], got.standard_deviation[0])
        assert abs(got.mean[0] - expected.mean[0]) <= 4.0 * sigma


def test_independent_returns_the_single_run():
    results, (tbr, _, _) = _run("independent")
    first = results.tally_results(0)[tbr].mean
    assert results.tally_results(1)[tbr].mean == first
    assert results.tally_results(2)[tbr].mean == first
    assert results.tally_results(3) is None
    means, _ = results.tally_evolution(tbr)
    assert means == [first] * 3


def test_unknown_tally_raises(coupled):
    results, _ = coupled
    with pytest.raises(KeyError):
        results.tally_evolution(yamc.Tally(scores=["flux"]))
