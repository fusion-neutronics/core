"""Broken meshes lose particles where they break, and sound twins lose none.

Each case is a few hand-made boxes with one defect a mesher or a converter can
leave, and the same boxes without it. Transport is through near-void Li6, so
every particle flies straight along its source direction and the flux tallied
in a volume is just the chord it flew through it. For each defect the test
pins where the particles are lost, not only how many: a tracker can count the
right number of losses after flying the particle through the wrong cells, and
then the flux is wrong too. Its twin must lose nothing and score the chords,
less the 1e-8 cm a crossing steps past each surface.

Defects covered here:

- a hole: a triangle missing from a box's surface;
- a shared surface with a gap behind it: a face recorded as bounding two
  volumes when the second does not start there;
- a facet wound the wrong way round, as at a fold, which is not a defect
  transport should stop on;
- a volume wound inward throughout.

A crack opening onto the volume's own far side, a point no test ray can place,
and overlapping volumes are covered in test_crack_crossing_lost.py,
test_unplaceable_particle_lost.py and test_verify_tracking.py.
"""
import os
import tempfile

import pytest
import yamc
from yamc._core import mesh_to_arrow

LI6 = "tests/Li6.arrow"
PARTICLES = 100
# In the first box, on the line through the half of its +x face that HOLE and
# FLIP name, heading +x.
START = (5.0, 3.0, 7.0)
PLUS_X = (1.0, 0.0, 0.0)
# The half of a box's +x face where z >= y: (1, 6, 5) in _box's numbering.
PLUS_X_UPPER = 11


def _box(x0, x1, y0=0.0, y1=10.0, z0=0.0, z1=10.0, skip_minus_x=False):
    """(vertices, triangles) of an axis-aligned box, wound outward."""
    v = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ]
    quads = [
        (0, 3, 2, 1),  # z = z0, normal -z: triangles 0, 1
        (4, 5, 6, 7),  # z = z1, normal +z: 2, 3
        (0, 1, 5, 4),  # y = y0, normal -y: 4, 5
        (2, 3, 7, 6),  # y = y1, normal +y: 6, 7
        (0, 4, 7, 3),  # x = x0, normal -x: 8, 9
        (1, 2, 6, 5),  # x = x1, normal +x: 10, 11
    ]
    t = []
    for i, (a, b, c, d) in enumerate(quads):
        if skip_minus_x and i == 4:
            t += [None, None]
            continue
        t.append([a, b, c])
        t.append([a, c, d])
    return v, t


def _write(path, surfaces, surface_volumes, names):
    """Write surfaces, each a list of (vertices, triangles) parts, as an
    imprinted mesh. ``surface_volumes`` gives each surface's (forward,
    reverse) volume, 0-based, None for the implicit complement."""
    vertices, triangles, surface_ids, groups = [], [], [], []
    for sid, parts in enumerate(surfaces, start=1):
        fwd, rev = surface_volumes[sid - 1]
        group = (fwd if fwd is not None else rev) + 1
        for v, t in parts:
            offset = len(vertices)
            vertices += v
            for tri in t:
                triangles.append([i + offset for i in tri])
                surface_ids.append(sid)
                groups.append(group)
    physical = ", ".join(
        f'"{i + 1}": {{"name": "mat:{n}", "dim": 3}}' for i, n in enumerate(names)
    )
    pairs = ", ".join(
        "[" + ", ".join("null" if x is None else str(x) for x in pair) + "]"
        for pair in surface_volumes
    )
    mesh_to_arrow(
        path,
        [c for p in vertices for c in p],
        [i for tri in triangles for i in tri],
        surface_ids,
        groups,
        physical_groups_json="{" + physical + "}",
        surface_volumes_json="[" + pairs + "]",
        imprinted=True,
    )


def _keep(parts_triangles, drop=(), flip=()):
    """Triangles with those named by ``drop`` left out and those named by
    ``flip`` wound the other way."""
    out = []
    for i, tri in enumerate(parts_triangles):
        if tri is None or i in drop:
            continue
        out.append([tri[0], tri[2], tri[1]] if i in flip else tri)
    return out


def _run(surfaces, surface_volumes, names, position=START, direction=PLUS_X):
    path = os.path.join(tempfile.mkdtemp(prefix="yamc_broken_"), "mesh.arrow")
    _write(path, surfaces, surface_volumes, names)
    materials = {}
    for n in names:
        m = yamc.Material(composition={"Li6": 1.0}, density=1e-9, name=n, temperature=294)
        m.read_nuclear_data({"Li6": LI6})
        materials[n] = m
    geometry = yamc.MeshGeometry(path, materials)
    source = yamc.NeutronSource(
        position=position,
        direction=yamc.sources.Monodirectional(list(direction)),
        energy=yamc.sources.Discrete([14.06e6], [1.0]),
    )
    tallies = [yamc.Tally(scores=["flux"], materials=m, name=n) for n, m in materials.items()]
    model = yamc.Model(geometry=geometry, tallies=tallies, source=source,
                       max_lost_particles=10 * PARTICLES)
    results = model.simulate_transport(total_particles=PARTICLES, seed=1)
    return model.lost_particles, {t.name: results[t].mean[0] for t in tallies}


def _lost_between(lost, x_lo, x_hi):
    """Every particle was lost, on its line of flight, with x in [x_lo, x_hi]."""
    assert len(lost) == PARTICLES
    for particle in lost:
        x, y, z = particle.position
        assert x_lo - 1e-6 <= x <= x_hi + 1e-6, particle.position
        assert abs(y - START[1]) < 1e-9 and abs(z - START[2]) < 1e-9, particle.position


# -- a hole --------------------------------------------------------------------


@pytest.mark.parametrize("hole", [True, False], ids=["hole", "closed"])
def test_a_hole_loses_the_particles_that_reach_it(hole):
    """A particle heading for a hole in a convex box has no facet ahead: it is
    lost where it stands, before flying any of the box, which cannot be left
    by a way that is not there."""
    v, t = _box(0, 10)
    surfaces = [[(v, _keep(t, drop={PLUS_X_UPPER} if hole else ()))]]
    lost, flux = _run(surfaces, [(0, None)], ["a"])
    if hole:
        _lost_between(lost, START[0], START[0])
        assert flux["a"] == pytest.approx(0.0, abs=1e-9)
    else:
        assert lost == []
        assert flux["a"] == pytest.approx(5.0, rel=1e-6)


# -- a shared surface with a gap behind it ---------------------------------------


@pytest.mark.parametrize("gap", [True, False], ids=["gap", "shared"])
def test_a_shared_surface_with_a_gap_behind_it_loses_the_particles_in_the_gap(gap):
    """Boxes A = [0,10] and B along x, with A's +x face recorded as bounding
    both. When B really starts at x = 10 that is an imprinted interface. When
    it starts at 10.5 the record is wrong: a particle crossing the face is put
    in B while it is in the gap, and next meets B's own -x face from outside.
    It is lost there, at x = 10.5, rather than crossing that face as if it were
    leaving B and flying B as void.
    """
    va, ta = _box(0, 10)
    shared = [ta[10], ta[11]]
    a_rest = _keep(ta, drop={10, 11})
    if gap:
        vb, tb = _box(10.5, 20.5)
        b_rest = _keep(tb)  # B's own surface, -x face included
    else:
        vb, tb = _box(10, 20, skip_minus_x=True)
        b_rest = _keep(tb)  # everything but the face it shares with A
    surfaces = [
        [(va, a_rest)],         # A against the implicit complement
        [(va, shared)],         # recorded as A forward, B reverse
        [(vb, b_rest)],         # B against the implicit complement
    ]
    lost, flux = _run(surfaces, [(0, None), (0, 1), (1, None)], ["a", "b"])
    assert flux["a"] == pytest.approx(5.0, rel=1e-6)
    if gap:
        _lost_between(lost, 10.0, 10.5)
        assert flux["b"] <= 0.5 + 1e-6
    else:
        assert lost == []
        assert flux["b"] == pytest.approx(10.0, rel=1e-6)


# -- a facet wound the wrong way round ----------------------------------------


def test_a_facet_wound_the_wrong_way_round_is_crossed_as_before():
    """At a fold a facet can be wound the wrong way round while the surface
    there is still where it should be. The particle is inside the box short of
    it, so it leaves the box there, loses nothing and scores the full chord."""
    v, t = _box(0, 10)
    lost, flux = _run([[(v, _keep(t, flip={PLUS_X_UPPER}))]], [(0, None)], ["a"])
    assert lost == []
    assert flux["a"] == pytest.approx(5.0, rel=1e-6)


# -- a volume wound inward throughout --------------------------------------------


@pytest.mark.parametrize("inverted", [True, False], ids=["inverted", "outward"])
def test_a_volume_wound_inward_is_not_tracked(inverted):
    """Every facet of the box wound inward: from inside, each ray's parity says
    inside and its nearest facet says outside, so the particle cannot be placed
    and is lost where it is born, instead of being tracked through a volume
    whose orientation transport cannot trust."""
    v, t = _box(0, 10)
    flip = set(range(12)) if inverted else set()
    lost, flux = _run([[(v, _keep(t, flip=flip))]], [(0, None)], ["a"])
    if inverted:
        _lost_between(lost, START[0], START[0])
        assert flux["a"] == pytest.approx(0.0, abs=1e-9)
    else:
        assert lost == []
        assert flux["a"] == pytest.approx(5.0, rel=1e-6)
