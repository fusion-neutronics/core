"""simulate_shutdown_dose(method="r2s"): cell-based rigorous two-step.

Iron (cell "rod") is irradiated by 14 MeV neutrons from the origin, and the
dose is a photon flux in a box above it. Fe56(n,p)Mn56 dominates, which D1S
also captures, so where the decay photons are born uniformly through the cell
(a thin shell around the source) the two methods must agree.
"""
import numpy as np
import pytest
import yamc

CHAIN = "tests/transmutation-endf-b8.1-sfr.arrow"
PARTICLES = 4000


@pytest.fixture(autouse=True)
def _chain():
    yamc.transmutation_decay_data = CHAIN
    yamc.transmutation_reactions = CHAIN
    yamc.transmutation_fission_yields = CHAIN


def _iron():
    mat = yamc.Material(composition={"Fe56": 1.0}, density=7.874, temperature=294)
    mat.read_nuclear_data({"Fe56": "tests/Fe56.arrow"}, photon_data={"Fe": "tests/Fe.arrow"})
    return mat


def _source():
    return yamc.NeutronSource(position=(0, 0, 0), energy=yamc.sources.Discrete([14.06e6], [1.0]))


def _model(ball=True, source=True, ball_z=0.0):
    """A void world with an iron ball of radius 2 at (0, 0, ball_z)."""
    world = yamc.Sphere(radius=400.0, boundary="vacuum")
    cells = [yamc.Cell(name="space", region=world.below)]
    if ball:
        s = yamc.Sphere(z0=ball_z, radius=2.0)
        cells = [yamc.Cell(name="rod", region=s.below, material=_iron()),
                 yamc.Cell(name="space", region=s.above & world.below)]
    return yamc.Model(geometry=yamc.Geometry(cells), source=_source() if source else None,
                      verbose=[])


MESH = yamc.RegularRectangularMesh(lower_left=[-10, -10, 10], upper_right=[10, 10, 30],
                                   shape=[1, 1, 1])


def _dose(steps, method="r2s"):
    tally = yamc.Tally(scores=["flux"], name="dose", mesh=MESH, particle="photon")
    return yamc.simulate_shutdown_dose(
        yamc.PulseSchedule(steps), tallies=[tally], total_particles=PARTICLES, seed=3,
        method=method,
    )["dose"]


def _total(dose):
    return np.array(dose.mean).sum(axis=1)


def _shell():
    """A 1 cm iron shell at radius 10 around the source: uniformly irradiated."""
    world = yamc.Sphere(radius=400.0, boundary="vacuum")
    a, b = yamc.Sphere(radius=10.0), yamc.Sphere(radius=11.0)
    cells = [yamc.Cell(name="in", region=a.below),
             yamc.Cell(name="rod", region=a.above & b.below, material=_iron()),
             yamc.Cell(name="out", region=b.above & world.below)]
    return yamc.Model(geometry=yamc.Geometry(cells), source=_source(), verbose=[])


def test_r2s_agrees_with_d1s_on_single_step_activation():
    steps = [
        yamc.Pulse(model=_shell(), rate=1e12, duration=(1, "h")),
        yamc.Cooldown(duration=(1, "h")),
        yamc.Cooldown(duration=(1, "d")),
    ]
    r2s, d1s = _dose(steps), _dose(steps, method="d1s")
    a, b = _total(r2s), _total(d1s)
    sigma = np.hypot(np.array(r2s.std_dev).sum(axis=1), np.array(d1s.std_dev).sum(axis=1))
    assert r2s.times == d1s.times == [3600.0, 7200.0, 7200.0 + 86400.0]
    assert np.all(a > 0) and a[0] > a[1] > a[2]
    assert np.all(np.abs(a - b) < 5 * sigma), (a, b, sigma)
    assert max(r2s.by_nuclide, key=lambda n: np.sum(r2s.by_nuclide[n][1])) == "Mn56"
    # The breakdown sums to the total.
    by_nuc = sum(np.array(v).sum(axis=1) for v in r2s.by_nuclide.values())
    assert np.allclose(by_nuc, a)


def test_removed_cell_emits_nothing_and_moved_cell_emits_where_it_is():
    irradiated = _model()
    away = _total(_dose([
        yamc.Pulse(model=irradiated, rate=1e12, duration=(1, "h")),
        yamc.Cooldown(model=_model(ball=False, source=False), duration=(1, "h")),
        yamc.Cooldown(model=_model(source=False, ball_z=300.0), duration=(1, "s")),
    ]))
    assert away[0] > 0
    assert away[1] == 0.0
    # 300 cm away the box sees almost nothing of it.
    assert 0 < away[2] < 0.01 * away[0]


def test_bad_method_and_two_irradiation_models_are_refused():
    steps = [yamc.Pulse(model=_model(), rate=1e12, duration=(1, "h"))]
    with pytest.raises(ValueError, match='method must be "d1s" or "r2s"'):
        _dose(steps, method="r3s")
    with pytest.raises(ValueError, match="same model"):
        _dose(steps + [yamc.Pulse(model=_model(), rate=1e12, duration=(1, "h"))])
