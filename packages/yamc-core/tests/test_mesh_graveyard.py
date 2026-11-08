"""The vacuum box a Model adds around a whole-model mesh geometry.

A mesh made from CAD has no vacuum surface. Before transport the Model
closes such a mesh with a vacuum box 10 cm outside its bounding box when
the implicit complement is void, and refuses the run with ``ValueError``
when the complement holds a material. A mesh tally reaching outside the
vacuum boundary gets a warning. Inside a real volume, a particle with no
surface ahead is still a lost particle.

The meshes are hand-built boxes written with ``mesh_to_arrow``.
"""
import os
import tempfile

import pytest
import yamc
from yamc._core import mesh_to_arrow

LI6 = "tests/Li6.arrow"

# One quad per box face, outward winding, as vertex indices into _box().
_FACES = {
    "-z": (0, 3, 2, 1),
    "+z": (4, 5, 6, 7),
    "-y": (0, 1, 5, 4),
    "+y": (2, 3, 7, 6),
    "-x": (0, 4, 7, 3),
    "+x": (1, 2, 6, 5),
}


def _write_cube(path, faces=tuple(_FACES)):
    """A 10 cm cube bounding one volume against the implicit complement,
    with only the named *faces* meshed."""
    v = [
        [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
        [0, 0, 10], [10, 0, 10], [10, 10, 10], [0, 10, 10],
    ]
    tris = []
    for face in faces:
        a, b, c, d = _FACES[face]
        tris.extend([a, b, c, a, c, d])
    n = len(tris) // 3
    mesh_to_arrow(
        path,
        [float(c) for p in v for c in p],
        tris,
        [1] * n,
        [1] * n,
        physical_groups_json='{"1": {"name": "mat:a", "dim": 3}}',
        surface_volumes_json="[[0, null]]",
    )


def _material(name="a"):
    m = yamc.Material(
        composition={"Li6": 1.0}, density=1e-3, name=name, temperature=294
    )
    m.read_nuclear_data({"Li6": LI6})
    return m


def _cube_geometry(faces=tuple(_FACES), **kwargs):
    path = os.path.join(tempfile.mkdtemp(prefix="yamc_graveyard_"), "cube.arrow")
    _write_cube(path, faces)
    return yamc.MeshGeometry(path, {"a": _material()}, **kwargs)


def _source(**kwargs):
    return yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1.0]), position=(5, 5, 5), **kwargs
    )


def test_void_complement_gets_a_vacuum_box(capfd):
    """No vacuum boundary and a void complement: the box is added and
    announced, and no particle is lost."""
    model = yamc.Model(geometry=_cube_geometry(), source=_source())
    model.simulate_transport(total_particles=500, seed=1)
    assert len(model.lost_particles) == 0
    out = capfd.readouterr().out
    assert "added a vacuum box 10 cm outside its bounding box" in out
    assert "x [-10.0000, 20.0000]" in out


def test_explicit_graveyard_adds_nothing(capfd):
    model = yamc.Model(geometry=_cube_geometry(graveyard_offset=3.0), source=_source())
    model.simulate_transport(total_particles=200, seed=1)
    assert len(model.lost_particles) == 0
    assert "added a vacuum box" not in capfd.readouterr().out


def test_material_complement_without_vacuum_is_refused():
    geometry = _cube_geometry(implicit_complement_material=_material("air"))
    model = yamc.Model(geometry=geometry, source=_source())
    with pytest.raises(ValueError, match="graveyard_offset"):
        model.simulate_transport(total_particles=10, seed=1)


def test_mesh_tally_beyond_the_vacuum_boundary_warns():
    def tally(lower, upper):
        mesh = yamc.RegularRectangularMesh(
            lower_left=[lower] * 3, upper_right=[upper] * 3, shape=[2, 2, 2]
        )
        return yamc.Tally(scores=["flux"], mesh=mesh, name="voxels")

    model = yamc.Model(
        geometry=_cube_geometry(), source=_source(), tallies=[tally(-15.0, 25.0)]
    )
    with pytest.warns(UserWarning, match="graveyard_offset"):
        model.simulate_transport(total_particles=10, seed=1)

    # The same tally mesh inside an explicit 20 cm box: no warning.
    model = yamc.Model(
        geometry=_cube_geometry(graveyard_offset=20.0),
        source=_source(),
        tallies=[tally(-15.0, 25.0)],
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.simulate_transport(total_particles=10, seed=1)


def test_no_surface_ahead_in_a_real_volume_is_lost():
    """A cube volume missing its -z face: a particle heading -z from inside
    meets no surface of its volume, which is lost even though the
    graveyard closes the complement. (The source still locates in the
    volume: point containment casts its first ray up, 2.5 degrees off +z.)"""
    geometry = _cube_geometry(
        faces=[f for f in _FACES if f != "-z"], graveyard_offset=5.0
    )
    source = _source(direction=yamc.sources.Monodirectional([0.0, 0.0, -1.0]))
    model = yamc.Model(geometry=geometry, source=source, max_lost_particles=1000)
    model.simulate_transport(total_particles=20, seed=1)
    lost = model.lost_particles
    assert len(lost) > 0
    for lp in lost:
        assert lp.last_cell_name.endswith("_a"), lp.last_cell_name
