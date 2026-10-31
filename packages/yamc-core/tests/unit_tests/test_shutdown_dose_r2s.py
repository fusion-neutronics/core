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


# Cell-under-mesh R2S.

SHELL_MESH = yamc.RegularRectangularMesh(lower_left=[-12, -12, -12], upper_right=[12, 12, 12],
                                         shape=[3, 3, 3])


def test_mesh_r2s_agrees_with_cells_on_a_uniformly_irradiated_shell():
    steps = [
        yamc.Pulse(model=_shell(), rate=1e12, duration=(1, "h")),
        yamc.Cooldown(duration=(1, "h")),
    ]
    tally = yamc.Tally(scores=["flux"], name="dose", mesh=MESH, particle="photon")
    mesh = yamc.simulate_shutdown_dose(
        yamc.PulseSchedule(steps), tallies=[tally], total_particles=PARTICLES, seed=3,
        method="r2s", mesh=SHELL_MESH,
    )["dose"]
    cells = _dose(steps)
    a, b = _total(mesh), _total(cells)
    sigma = np.hypot(np.array(mesh.std_dev).sum(axis=1), np.array(cells.std_dev).sum(axis=1))
    # Multigroup against continuous-energy rates: a few percent on a threshold
    # reaction, on top of the noise.
    assert np.all(np.abs(a - b) < 5 * sigma + 0.05 * b), (a, b, sigma)
    assert max(mesh.by_nuclide, key=lambda n: np.sum(mesh.by_nuclide[n][1])) == "Mn56"


def _bar():
    """An iron bar along x from 0 to 40 cm, irradiated from its x=0 end."""
    world = yamc.Sphere(radius=400.0, boundary="vacuum")
    bar = (yamc.Plane("x", 0.0).above & yamc.Plane("x", 40.0).below
           & yamc.Cylinder("x", radius=2.0).below)
    cells = [yamc.Cell(name="rod", region=bar, material=_iron()),
             yamc.Cell(name="space", region=~bar & world.below)]
    src = yamc.NeutronSource(position=(-0.5, 0, 0), energy=yamc.sources.Discrete([14.06e6], [1.0]))
    return yamc.Model(geometry=yamc.Geometry(cells), source=src, verbose=[])


def test_mesh_r2s_puts_the_source_where_the_bar_is_activated():
    near = yamc.RegularRectangularMesh(lower_left=[-5, -5, 5], upper_right=[5, 5, 15],
                                       shape=[1, 1, 1])
    far = yamc.RegularRectangularMesh(lower_left=[35, -5, 5], upper_right=[45, 5, 15],
                                      shape=[1, 1, 1])
    tallies = [yamc.Tally(scores=["flux"], name=n, mesh=m, particle="photon")
               for n, m in (("near", near), ("far", far))]
    steps = [yamc.Pulse(model=_bar(), rate=1e12, duration=(1, "h"))]
    run = lambda **kw: yamc.simulate_shutdown_dose(
        yamc.PulseSchedule(steps), tallies=tallies, total_particles=PARTICLES, seed=3,
        method="r2s", **kw)
    mesh = run(mesh=yamc.RegularRectangularMesh(lower_left=[0, -2, -2], upper_right=[40, 2, 2],
                                                shape=[8, 1, 1]))
    cells = run()
    ratio = lambda d: _total(d["near"])[0] / _total(d["far"])[0]
    # The whole bar emitting uniformly is symmetric end to end; the mesh
    # follows the activation down the bar.
    assert 0.7 < ratio(cells) < 1.4
    assert ratio(mesh) > 5 * ratio(cells)


def test_mesh_r2s_moves_voxel_parts_with_their_cell():
    mesh = yamc.RegularRectangularMesh(lower_left=[-3, -3, -3], upper_right=[3, 3, 3],
                                       shape=[2, 2, 2])

    def run(moves):
        return _total(yamc.simulate_shutdown_dose(
            yamc.PulseSchedule([
                yamc.Pulse(model=_model(), rate=1e12, duration=(1, "h")),
                yamc.Cooldown(model=_model(source=False, ball_z=-30.0), duration=(1, "s"),
                              moves=moves),
                yamc.Cooldown(model=_model(ball=False, source=False), duration=(1, "s")),
            ]),
            tallies=[yamc.Tally(scores=["flux"], name="dose", mesh=MESH, particle="photon")],
            total_particles=PARTICLES, seed=3, method="r2s", mesh=mesh,
        )["dose"])

    # 30 cm further from the box, it sees a fraction of what it did.
    moved = run({"rod": (0, 0, -30)})
    assert moved[0] > 0 and 0 < moved[1] < 0.3 * moved[0] and moved[2] == 0.0
    with pytest.raises(ValueError, match="give every moved cell's move"):
        run({})


def test_mesh_is_refused_for_d1s_and_away_from_the_activated_cells():
    steps = [yamc.Pulse(model=_model(), rate=1e12, duration=(1, "h"))]
    tally = yamc.Tally(scores=["flux"], name="dose", mesh=MESH, particle="photon")
    run = lambda method, mesh: yamc.simulate_shutdown_dose(
        yamc.PulseSchedule(steps), tallies=[tally], total_particles=200, method=method,
        mesh=mesh)
    with pytest.raises(ValueError, match="mesh is for method"):
        run("d1s", SHELL_MESH)
    with pytest.raises(ValueError, match="no material-filled cell under the mesh"):
        run("r2s", yamc.RegularRectangularMesh(lower_left=[100, 100, 100],
                                               upper_right=[101, 101, 101], shape=[1, 1, 1]))
