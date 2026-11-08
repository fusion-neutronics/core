"""A particle the mesh cannot place is lost, not put in the implicit complement.

A 10 cm cube with a closed shell of stray triangles inside it: an octahedron
of radius 1 cm about the cube's centre, wound to face outwards, facets with no
volume behind them, as a mesher defect can leave. From the centre every one of
point location's test rays, whichever way it goes, crosses the shell before
the cube's wall, so each counts two crossings, an even number, and parity
alone says "outside". Each particle would be born in the implicit complement,
inside the cube, and tracked as void: lost later, at the cube's wall, far from
where things went wrong, or leaving the geometry as ordinary leakage, with
nothing counted at all.

Each ray's nearest facet is a stray one, which the ray leaves the cube
through, so its facing says "inside" where its parity says "outside". No ray
gives a consistent answer, the point cannot be placed, and the particle is
counted as lost.

A cube without the stray shell is the control.
"""
import os
import tempfile

import yamc
from yamc._core import mesh_to_arrow

LI6 = "tests/Li6.arrow"
PARTICLES = 200
CENTRE = (5.0, 5.0, 5.0)


def _cube(stray):
    v = [
        [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
        [0, 0, 10], [10, 0, 10], [10, 10, 10], [0, 10, 10],
    ]
    t = []
    for a, b, c, d in [
        (0, 3, 2, 1),  # z = 0, normal -z
        (4, 5, 6, 7),  # z = 10, normal +z
        (0, 1, 5, 4),  # y = 0, normal -y
        (2, 3, 7, 6),  # y = 10, normal +y
        (0, 4, 7, 3),  # x = 0, normal -x
        (1, 2, 6, 5),  # x = 10, normal +x
    ]:
        t.append([a, b, c])
        t.append([a, c, d])
    if stray:
        # Corners at the centre plus and minus 1 cm along x, y and z.
        v += [[6, 5, 5], [4, 5, 5], [5, 6, 5], [5, 4, 5], [5, 5, 6], [5, 5, 4]]
        for a, b in [(8, 10), (10, 9), (9, 11), (11, 8)]:
            t.append([a, b, 12])  # upper half, normals outwards
            t.append([b, a, 13])  # lower half
    return v, t


def _run(stray):
    path = os.path.join(tempfile.mkdtemp(prefix="yamc_unplaceable_"), "cube.arrow")
    v, t = _cube(stray)
    mesh_to_arrow(
        path,
        [c for p in v for c in p],
        [i for tri in t for i in tri],
        [1] * len(t),
        [1] * len(t),
        physical_groups_json='{"1": {"name": "mat:a", "dim": 3}}',
        surface_volumes_json="[[0, null]]",
        imprinted=True,
    )
    material = yamc.Material(composition={"Li6": 1.0}, density=1e-9, name="a", temperature=294)
    material.read_nuclear_data({"Li6": LI6})
    geometry = yamc.MeshGeometry(path, {"a": material})
    source = yamc.NeutronSource(position=CENTRE, energy=yamc.sources.Discrete([14.06e6], [1.0]))
    tally = yamc.Tally(scores=["flux"], materials=material, name="flux")
    model = yamc.Model(geometry=geometry, tallies=[tally], source=source,
                       max_lost_particles=10 * PARTICLES)
    results = model.simulate_transport(total_particles=PARTICLES, seed=1)
    return model, results[tally].mean[0]


def test_a_particle_born_where_the_mesh_cannot_say_is_lost():
    model, _ = _run(stray=True)
    lost = model.lost_particles
    assert len(lost) == PARTICLES
    for particle in lost:
        assert max(abs(x - c) for x, c in zip(particle.position, CENTRE)) < 1e-6


def test_the_sound_cube_loses_nothing():
    model, flux = _run(stray=False)
    assert len(model.lost_particles) == 0
    assert flux > 0.0
