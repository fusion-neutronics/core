"""A particle that leaves a mesh volume through a crack is lost where it leaves.

One volume made of two separate boxes, A = [0,10]^3 and B = [20,30] x
[0,10]^2, with a triangle missing from A's +x face. Particles start inside A
heading +x through the hole. The nearest facet of the volume ahead of them is
B's -x face, which they would ENTER the volume through: they are not inside
the volume any more. Taking that facet as a way out put each particle in the
implicit complement, inside B, and the tracking stayed one cell out of step
from there: B was flown as void, the gap and B's far side as the volume, and
the particle was only lost once nothing at all lay ahead, 25 cm on, having
scored the gap as the volume. In a model with more around it nothing need
ever lie ahead: the particle runs on in the wrong cell until it leaks.
"""
import os
import tempfile

import yamc
from yamc._core import mesh_to_arrow

LI6 = "tests/Li6.arrow"
PARTICLES = 200
START = (5.0, 3.0, 7.0)  # in A, and on the line through the hole


def _box(x0, x1, y0, y1, z0, z1):
    """(vertices, triangles) for an axis-aligned box, outward winding. The
    +x face is split into (1, 2, 6), where z <= y, and (1, 6, 5), z >= y."""
    v = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ]
    quads = [
        (0, 3, 2, 1),  # z = z0, normal -z
        (4, 5, 6, 7),  # z = z1, normal +z
        (0, 1, 5, 4),  # y = y0, normal -y
        (2, 3, 7, 6),  # y = y1, normal +y
        (0, 4, 7, 3),  # x = x0, normal -x
        (1, 2, 6, 5),  # x = x1, normal +x
    ]
    t = []
    for a, b, c, d in quads:
        t.append([a, b, c])
        t.append([a, c, d])
    return v, t


def _write_two_lumps(path, hole):
    va, ta = _box(0, 10, 0, 10, 0, 10)
    if hole:
        del ta[11]  # (1, 6, 5): the half of A's +x face the particles aim at
    vb, tb = _box(20, 30, 0, 10, 0, 10)
    vertices = [c for v in va + vb for c in v]
    tris = [i for t in ta for i in t] + [i + 8 for t in tb for i in t]
    n = len(ta) + len(tb)
    mesh_to_arrow(
        path,
        vertices,
        tris,
        [1] * n,
        [1] * n,
        physical_groups_json='{"1": {"name": "mat:a", "dim": 3}}',
        surface_volumes_json="[[0, null]]",
        imprinted=True,
    )


def _run(hole):
    path = os.path.join(tempfile.mkdtemp(prefix="yamc_crack_"), "lumps.arrow")
    _write_two_lumps(path, hole)
    material = yamc.Material(composition={"Li6": 1.0}, density=1e-9, name="a", temperature=294)
    material.read_nuclear_data({"Li6": LI6})
    geometry = yamc.MeshGeometry(path, {"a": material})
    source = yamc.NeutronSource(
        position=START,
        direction=yamc.sources.Monodirectional([1.0, 0.0, 0.0]),
        energy=yamc.sources.Discrete([14.06e6], [1.0]),
    )
    tally = yamc.Tally(scores=["flux"], materials=material, name="flux")
    model = yamc.Model(geometry=geometry, tallies=[tally], source=source,
                       max_lost_particles=10 * PARTICLES)
    results = model.simulate_transport(total_particles=PARTICLES, seed=1)
    return model, results[tally].mean[0]


def test_a_particle_out_through_a_crack_is_lost():
    model, flux = _run(hole=True)
    lost = model.lost_particles
    assert len(lost) == PARTICLES
    # Each is lost where it was when the far side of the volume came into
    # view: still inside A, on its line of flight.
    for particle in lost:
        assert 0.0 <= particle.position[0] <= 10.0
        assert abs(particle.position[1] - START[1]) < 1e-9
        assert abs(particle.position[2] - START[2]) < 1e-9
    # Nothing beyond the crack is scored as the volume.
    assert flux <= 5.0 + 1e-6


def test_the_closed_volume_loses_nothing():
    model, flux = _run(hole=False)
    assert len(model.lost_particles) == 0
    # Each particle crosses A from its start to the +x face, then B: 5 + 10 cm.
    assert abs(flux - 15.0) < 1e-6
