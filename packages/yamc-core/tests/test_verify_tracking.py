"""Spatial tracking verification on mesh geometries (issue #254).

Every surface crossing on a mesh geometry is verified spatially: a flight
segment that passes through a surface foreign to the current volume means
the mesh volumes overlap or self-intersect (or the tracking state was
corrupted by an earlier mis-resolved crossing), and the particle is
recorded as lost instead of being transported through a wrong model.

The reproducer is a hand-crafted Arrow mesh of two OVERLAPPING cubes whose
surface adjacency is self-consistent (each shell closed, fwd/rev volumes
coherent), which the adjacency tracker would otherwise follow silently.
"""
import os
import tempfile

import yamc
from yamc._core import mesh_to_arrow

LI6 = "tests/Li6.arrow"


def _box(x0, x1, y0, y1, z0, z1):
    """(vertices, triangles) for an axis-aligned box, outward winding."""
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


def _write_two_cubes(path, overlap):
    """Two cubes: overlapping in x=[5,10] when *overlap*, else separated.

    Each cube is a closed shell bounding its own volume against the
    implicit complement, so the topology is consistent either way; only
    the overlapping variant is geometrically invalid.
    """
    xb = 5.0 if overlap else 30.0
    va, ta = _box(0, 10, 0, 10, 0, 10)
    vb, tb = _box(xb, xb + 10, 0, 10, 0, 10)
    vertices = [c for v in va + vb for c in v]
    tris = []
    for t in ta:
        tris.extend(t)
    for t in tb:
        tris.extend(i + 8 for i in t)
    n = len(ta)
    mesh_to_arrow(
        path,
        vertices,
        tris,
        [1] * n + [2] * n,          # surface ids
        [1] * n + [2] * n,          # physical groups (per material)
        physical_groups_json=(
            '{"1": {"name": "mat:a", "dim": 3}, "2": {"name": "mat:b", "dim": 3}}'
        ),
        surface_volumes_json="[[0, null], [1, null]]",
    )


def _material(name):
    m = yamc.Material(
        composition={"Li6": 1.0}, density=1e-6, name=name, temperature=294
    )
    m.read_nuclear_data({"Li6": LI6})
    return m


def _run(overlap, particles=2000):
    path = os.path.join(tempfile.mkdtemp(prefix="yamc_verify_"), "mesh.arrow")
    _write_two_cubes(path, overlap)
    geom = yamc.MeshGeometry(path, {"a": _material("a"), "b": _material("b")})
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1.0]),
        position=(2.0, 5.0, 5.0),  # inside cube A only
    )
    model = yamc.Model(
        geometry=geom,
        tallies=[],
        source=source,
        max_lost_particles=10 * particles,
    )
    model.simulate_transport(total_particles=particles, seed=42)
    return model


def test_overlapping_volumes_lose_particles():
    """Flight segments through a foreign surface are refused and the
    particle is lost at the start of the invalid segment."""
    model = _run(overlap=True)
    lost = model.lost_particles
    assert len(lost) > 0
    for lp in lost:
        # every loss happens inside the two-cube complex
        assert -0.1 <= lp.position[0] <= 15.1


def test_separated_volumes_no_false_positives():
    """The same two cubes without overlap transport cleanly."""
    model = _run(overlap=False)
    assert len(model.lost_particles) == 0


def test_source_in_implicit_complement_locates_correctly():
    """A source in the gap between two volumes must be located in the
    implicit complement, stream to both cubes, and lose nothing.

    Regression for issue #256: point_in_volume routed its parity count
    through the nearest-hit-pruned BVH traversal, so find_volume claimed
    almost any point for the first tested volume. Every history whose
    source was NOT inside volume 0 was born in the wrong cell, and the
    adjacency walk stayed one volume out of phase forever (2.1x flux
    error against OpenMC on the reactor radial build). All older tests
    sourced inside the first volume, which is why this went unseen.
    """
    path = os.path.join(tempfile.mkdtemp(prefix="yamc_verify_"), "gap.arrow")
    _write_two_cubes(path, overlap=False)  # cubes at x=[0,10] and x=[30,40]
    mats = {"a": _material("a"), "b": _material("b")}
    geom = yamc.MeshGeometry(path, mats)
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1.0]),
        position=(20.0, 5.0, 5.0),  # in the gap: implicit complement
    )
    tally_a = yamc.Tally(scores=["flux"], materials=mats["a"], name="a")
    tally_b = yamc.Tally(scores=["flux"], materials=mats["b"], name="b")
    model = yamc.Model(
        geometry=geom, tallies=[tally_a, tally_b], source=source
    )
    results = model.simulate_transport(total_particles=20000, seed=42)
    assert len(model.lost_particles) == 0
    assert results[tally_a].mean[0] > 0.0
    assert results[tally_b].mean[0] > 0.0


def test_clean_single_cube_no_false_positives():
    """A single well-formed cube loses nothing."""
    path = os.path.join(tempfile.mkdtemp(prefix="yamc_verify_"), "cube.arrow")
    va, ta = _box(0, 10, 0, 10, 0, 10)
    vertices = [c for v in va for c in v]
    tris = [i for t in ta for i in t]
    mesh_to_arrow(
        path,
        vertices,
        tris,
        [1] * len(ta),
        [1] * len(ta),
        physical_groups_json='{"1": {"name": "mat:a", "dim": 3}}',
        surface_volumes_json="[[0, null]]",
    )
    geom = yamc.MeshGeometry(path, {"a": _material("a")})
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1.0]),
        position=(5.0, 5.0, 5.0),
    )
    model = yamc.Model(geometry=geom, tallies=[], source=source)
    model.simulate_transport(total_particles=5000, seed=42)
    assert len(model.lost_particles) == 0
