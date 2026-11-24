"""simulate_shutdown_dose: shutdown dose over a machine that changes between steps.

An iron ball (cell "rod") is irradiated by 14 MeV neutrons at the origin. The
dose is a photon flux in a mesh box just above it. Later stages remove the ball,
move it 200 cm away, or put it back.
"""
import numpy as np
import pytest
import yamc

CHAIN = "tests/transmutation-endf-b8.1-sfr.arrow"
PARTICLES = 3000


@pytest.fixture(autouse=True)
def _chain():
    yamc.transmutation_decay_data = CHAIN
    yamc.transmutation_reactions = CHAIN
    yamc.transmutation_fission_yields = CHAIN


def _iron():
    mat = yamc.Material(composition={"Fe56": 1.0}, density=7.874, temperature=294)
    mat.read_nuclear_data({"Fe56": "tests/Fe56.arrow"}, photon_data={"Fe": "tests/Fe.arrow"})
    return mat


def _source(energy=14.06e6):
    return yamc.NeutronSource(position=(0, 0, 0), energy=yamc.sources.Discrete([energy], [1.0]))


def _model(ball_z=0.0, ball=True, ball_name="rod", source=True, energy=14.06e6):
    """A void world with an iron ball of radius 5 at (0, 0, ball_z)."""
    world = yamc.Sphere(radius=400.0, boundary="vacuum")
    cells = []
    if ball:
        s = yamc.Sphere(z0=ball_z, radius=5.0)
        cells.append(yamc.Cell(name=ball_name, region=s.below, material=_iron()))
        cells.append(yamc.Cell(name="space", region=s.above & world.below))
    else:
        cells.append(yamc.Cell(name="space", region=world.below))
    return yamc.Model(geometry=yamc.Geometry(cells), source=_source(energy) if source else None,
                      verbose=[])


# A box just above the ball's starting position.
MESH = yamc.RegularRectangularMesh(lower_left=[-10, -10, 6], upper_right=[10, 10, 26],
                                   shape=[1, 1, 1])


def _tally():
    return yamc.Tally(scores=["flux"], name="dose", mesh=MESH, particle="photon")


def _total(dose):
    return np.array(dose.mean).sum(axis=1)


def _dose(steps):
    return yamc.simulate_shutdown_dose(
        yamc.PulseSchedule(steps), tallies=[_tally()], total_particles=PARTICLES, seed=3,
    )["dose"]


# --- one machine throughout: the staged path reproduces single-run D1S --------

def test_one_model_matches_time_correct_tally():
    model = _model()
    dose = _dose([
        yamc.Pulse(model=model, rate=1e12, duration=(1, "h")),
        yamc.Cooldown(duration=(1, "h")),
        yamc.Cooldown(duration=(1, "d")),
    ])

    # The same problem the established way: one coupled D1S run.
    radionuclides = model.radionuclides()
    mesh_tally = yamc.Tally(scores=["flux"], name="dose", mesh=MESH,
                            particle="photon", parent_nuclides=radionuclides)
    d1s = yamc.Model(geometry=model.geometry, tallies=[mesh_tally], source=_source(),
                     transport_secondary_photons=True, use_decay_photons=True, verbose=[])
    result = d1s.simulate_transport(total_particles=PARTICLES, seed=3)[mesh_tally]
    src = _source()
    ref = yamc.PulseSchedule([
        yamc.Pulse(source=src, rate=1e12, duration=(1, "h")),
        yamc.Cooldown(duration=(1, "h")),
        yamc.Cooldown(duration=(1, "d")),
    ]).time_correct_tally(result)

    staged, single = _total(dose), np.array(ref.mean).sum(axis=1)
    sigma = np.hypot(np.array(dose.std_dev).sum(axis=1), np.array(ref.std_dev).sum(axis=1))
    assert np.all(staged > 0)
    assert np.all(np.abs(staged - single) < 5 * sigma), (staged, single, sigma)
    assert dose.times == [3600.0, 7200.0, 7200.0 + 86400.0]
    # Decays through the cooling, and the breakdown sums to the total.
    assert staged[0] > staged[1] > staged[2]
    by_nuc = sum(np.array(v).sum(axis=1) for v in dose.by_nuclide.values())
    assert np.allclose(by_nuc, staged)


def test_one_model_has_the_single_model_d1s_error():
    # Unchanged machine: the staged estimator is the coupled D1S one, so its
    # standard error is too (the neutron transport's noise included).
    model = _model()
    steps = [yamc.Pulse(model=model, rate=1e12, duration=(1, "h")),
             yamc.Cooldown(duration=(1, "h"))]
    staged = np.array(_dose(steps).std_dev)[:, 0]
    tally = yamc.Tally(scores=["flux"], name="dose", mesh=MESH, particle="photon",
                       parent_nuclides=model.radionuclides())
    d1s = yamc.Model(geometry=model.geometry, tallies=[tally], source=_source(),
                     transport_secondary_photons=True, use_decay_photons=True, verbose=[])
    src = _source()
    ref = yamc.PulseSchedule([yamc.Pulse(source=src, rate=1e12, duration=(1, "h")),
                              yamc.Cooldown(duration=(1, "h"))]).time_correct_tally(
        d1s.simulate_transport(total_particles=PARTICLES, seed=3)[tally])
    single = np.array(ref.std_dev)[:, 0]
    assert np.all(staged > 0)
    assert np.all((staged / single > 0.75) & (staged / single < 1.33)), (staged, single)


def test_reported_error_matches_the_seed_to_seed_spread():
    """The standard error is that of the whole calculation: across independent
    seeds the results scatter by what each run reports. Two irradiations in two
    models (sharing their seeds), a move, and few neutron histories, so the
    neutron transport's noise dominates and would be missed if left out."""
    low = _model(energy=2.45e6)
    high = _model()
    moved = _model(ball_z=200.0, source=False)
    there = yamc.RegularRectangularMesh(lower_left=[-10, -10, 206], upper_right=[10, 10, 226],
                                        shape=[1, 1, 1])
    tally = yamc.Tally(scores=["flux"], name="there", mesh=there, particle="photon")
    sched = yamc.PulseSchedule([
        yamc.Pulse(model=low, rate=1e12, duration=(1, "h")),
        yamc.Pulse(model=high, rate=1e12, duration=(1, "h")),
        yamc.Cooldown(model=moved, duration=(1, "h"), moves={"rod": (0, 0, 200)}),
    ])
    means, stds = [], []
    for seed in range(1, 41):
        d = yamc.simulate_shutdown_dose(sched, tallies=[tally], total_particles=200,
                                        seed=seed)["there"]
        means.append(np.array(d.mean)[:, 0])
        stds.append(np.array(d.std_dev)[:, 0])
    spread = np.array(means).std(axis=0, ddof=1)
    reported = np.array(stds).mean(axis=0)
    # Step 2, the moved ball: 40 seeds pin the spread to about 12%, more with
    # the heavy tail of a few activated histories.
    assert reported[2] > 0
    assert 0.6 < spread[2] / reported[2] < 1.6, (spread, reported)


def test_same_seed_same_answer_whatever_the_threads():
    steps = [yamc.Pulse(model=_model(), rate=1e12, duration=(1, "h")),
             yamc.Cooldown(model=_model(ball_z=200.0, source=False), duration=(1, "h"),
                           moves={"rod": (0, 0, 200)})]

    def run(threads):
        return yamc.simulate_shutdown_dose(yamc.PulseSchedule(steps), tallies=[_tally()],
                                           total_particles=500, seed=4, threads=threads)["dose"]

    one, two = run(1), run(2)
    assert np.allclose(one.mean, two.mean, rtol=1e-12, atol=0)
    assert np.allclose(one.std_dev, two.std_dev, rtol=1e-9, atol=0)


# --- removing, re-adding and moving a component -------------------------------

def test_removed_cell_stops_emitting_and_keeps_decaying_while_away():
    irradiated, opened = _model(), _model(ball=False, source=False)
    common = [yamc.Pulse(model=irradiated, rate=1e12, duration=(1, "h"))]
    away = _total(_dose(common + [
        yamc.Cooldown(model=opened, duration=(1, "h")),
        yamc.Cooldown(model=irradiated, duration=(1, "h")),
    ]))
    stayed = _total(_dose(common + [
        yamc.Cooldown(duration=(1, "h")),
        yamc.Cooldown(duration=(1, "h")),
    ]))
    # Out of the machine it emits nothing there...
    assert away[1] == 0.0
    # ...and back in, it has decayed exactly as if it had never left.
    assert away[2] == pytest.approx(stayed[2], rel=1e-12)


def test_moved_cell_carries_its_activation():
    # Two boxes: one above the ball where it was irradiated, one above where it
    # is moved to.
    there = yamc.RegularRectangularMesh(lower_left=[-10, -10, 206], upper_right=[10, 10, 226],
                                        shape=[1, 1, 1])
    tallies = [_tally(), yamc.Tally(scores=["flux"], name="there", mesh=there, particle="photon")]
    irradiated = _model()
    moved = _model(ball_z=200.0, source=False)

    def run(cooldown):
        sched = yamc.PulseSchedule([yamc.Pulse(model=irradiated, rate=1e12, duration=(1, "h")),
                                    cooldown])
        dose = yamc.simulate_shutdown_dose(sched, tallies=tallies, total_particles=PARTICLES,
                                           seed=3)
        return _total(dose["dose"])[1], _total(dose["there"])[1]

    here_moved, there_moved = run(
        yamc.Cooldown(model=moved, duration=(1, "h"), moves={"rod": (0, 0, 200)}))
    here_stayed, there_stayed = run(yamc.Cooldown(duration=(1, "h")))
    # The same box geometry relative to the ball, so the activation that moved
    # shows up there as it did here when the ball stayed put.
    assert there_moved > 0.0 and here_stayed > 0.0
    assert there_moved == pytest.approx(here_stayed, rel=0.25)
    # And it is gone from where the ball was.
    assert here_moved < 0.05 * here_stayed
    assert there_stayed < 0.05 * there_moved


def test_moves_accept_a_cell_and_a_matrix():
    irradiated = _model()
    moved = _model(ball_z=200.0, source=False)
    ball = next(c for c in moved.geometry.cells if c.name == "rod")
    matrix = np.eye(4)
    matrix[2, 3] = 200.0
    by_cell = yamc.Cooldown(model=moved, duration=1.0, moves={ball: (0, 0, 200)})
    by_matrix = yamc.Cooldown(model=moved, duration=1.0, moves={"rod": matrix.tolist()})
    assert by_cell.moves == by_matrix.moves
    assert by_cell.moves["rod"][2][3] == 200.0
    dose = _dose([yamc.Pulse(model=irradiated, rate=1e12, duration=(1, "h")), by_matrix])
    assert np.all(np.isfinite(dose.mean))


def test_added_cell_activates_only_from_when_it_is_in():
    without = _model(ball=False)
    with_ball = _model()
    dose = _total(_dose([
        yamc.Pulse(model=without, rate=1e12, duration=(1, "h")),
        yamc.Cooldown(duration=(1, "h")),
        yamc.Pulse(model=with_ball, rate=1e12, duration=(1, "h")),
        yamc.Cooldown(duration=(1, "h")),
    ]))
    assert dose[0] == 0.0 and dose[1] == 0.0
    assert dose[2] > 0.0 and dose[3] > 0.0


# --- errors -----------------------------------------------------------------

def test_move_that_does_not_match_the_geometry_raises():
    irradiated, unmoved = _model(), _model(source=False)
    with pytest.raises(ValueError, match="does not line up"):
        _dose([
            yamc.Pulse(model=irradiated, rate=1e12, duration=(1, "h")),
            yamc.Cooldown(model=unmoved, duration=(1, "h"), moves={"rod": (0, 0, 200)}),
        ])


def test_unnamed_activated_cell_raises_when_models_differ():
    irradiated, later = _model(ball_name=None), _model(ball_name=None, source=False)
    with pytest.raises(ValueError, match="unnamed cell"):
        _dose([
            yamc.Pulse(model=irradiated, rate=1e12, duration=(1, "h")),
            yamc.Cooldown(model=later, duration=(1, "h")),
        ])


def test_non_rigid_move_raises():
    irradiated, moved = _model(), _model(source=False)
    scale = np.diag([2.0, 1.0, 1.0, 1.0]).tolist()
    with pytest.raises(ValueError, match="orthonormal"):
        _dose([
            yamc.Pulse(model=irradiated, rate=1e12, duration=(1, "h")),
            yamc.Cooldown(model=moved, duration=(1, "h"), moves={"rod": scale}),
        ])


def test_schedule_shape_errors():
    model = _model()
    with pytest.raises(ValueError, match="source or a model"):
        yamc.Pulse(model=model, source=_source(), rate=1.0, duration=1.0)
    with pytest.raises(ValueError, match="model= as well"):
        yamc.Cooldown(duration=1.0, moves={"rod": (0, 0, 1)})
    with pytest.raises(ValueError, match="translation"):
        yamc.Cooldown(model=model, duration=1.0, moves={"rod": (0, 1)})
    with pytest.raises(ValueError, match="first step"):
        yamc.PulseSchedule([yamc.Cooldown(duration=1.0),
                            yamc.Pulse(model=model, rate=1.0, duration=1.0)])
    with pytest.raises(ValueError, match="leave Pulse.source unset"):
        yamc.PulseSchedule([yamc.Pulse(model=model, rate=1.0, duration=1.0),
                            yamc.Pulse(source=_source(), rate=1.0, duration=1.0)])


def test_staged_and_single_model_entry_points_point_at_each_other():
    model = _model()
    staged = yamc.PulseSchedule([yamc.Pulse(model=model, rate=1.0, duration=1.0)])
    with pytest.raises(ValueError, match="simulate_shutdown_dose"):
        staged.time_correct_tally(None)
    plain = yamc.PulseSchedule([yamc.Pulse(source=_source(), rate=1.0, duration=1.0)])
    with pytest.raises(ValueError, match="time_correct_tally"):
        yamc.simulate_shutdown_dose(plain, tallies=[_tally()], total_particles=10)


def test_tally_arguments_are_checked():
    model = _model()
    sched = yamc.PulseSchedule([yamc.Pulse(model=model, rate=1.0, duration=1.0)])
    unnamed = yamc.Tally(scores=["flux"], particle="photon")
    with pytest.raises(ValueError, match="name"):
        yamc.simulate_shutdown_dose(sched, tallies=[unnamed], total_particles=10)
    binned = yamc.Tally(scores=["flux"], name="b", particle="photon", parent_nuclides=["Mn56"])
    with pytest.raises(ValueError, match="parent_nuclides"):
        yamc.simulate_shutdown_dose(sched, tallies=[binned], total_particles=10)
    with pytest.raises(ValueError, match="positive"):
        yamc.simulate_shutdown_dose(sched, tallies=[_tally()], total_particles=0)


def test_model_getters():
    model = _model()
    pulse = yamc.Pulse(model=model, rate=1.0, duration=1.0)
    assert pulse.model is model and pulse.source is None and pulse.moves == {}
    assert yamc.Cooldown(duration=1.0).model is None
