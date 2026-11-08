"""Surface meshing tolerances set per solid.

``tolerance`` and ``angular_tolerance`` take a dict from solid id or material
tag to a value, so a model can mesh a first wall finely and coil casings
coarsely. A face two solids share is meshed at the finer of their values, and
an edge at the finest of its faces', so touching solids stay watertight. The
assemblies here share CURVED edges between solids at different values
(planar faces have straight edges, which any two discretizations agree on).
"""

import dataclasses
import math

import pytest

cq = pytest.importorskip("cadquery")

import numpy as np  # noqa: E402
from OCP.BRepMesh import BRepMesh_IncrementalMesh  # noqa: E402

from yamc.cad import CadToYamc, OverlappingSolidsError  # noqa: E402
from yamc.cad import mesher as mesher_module  # noqa: E402
from yamc.cad._brepmesh import (  # noqa: E402
    _face_mesh,
    _finest_first_on_shared_edges,
    _follows_its_edges,
)
from yamc.cad._fidelity import _occ_signed_volume, check_mesh_fidelity  # noqa: E402
from yamc.cad._overlap import _solid_volumes  # noqa: E402
from yamc.cad._scene_extract import scene_face_meshes  # noqa: E402
from yamc.cad.mesher import _per_solid, _route_faces  # noqa: E402

FINE = 0.01
COARSE = 0.5
FINE_ANGLE = 0.1
COARSE_ANGLE = 1.0


def _assembly(**parts):
    assembly = cq.Assembly()
    for name, shape in parts.items():
        assembly.add(shape, name=name)
    return assembly


def _mesher(assembly):
    c2y = CadToYamc()
    c2y.add_cadquery_object(assembly, material_tags="assembly_names")
    return c2y


def _sid(mesh, tag):
    return mesh.material_tags.index(tag) + 1


def _solid_triangles(mesh):
    """{solid id: triangles on its boundary}."""
    return {
        sid: [tri for tri, fid in zip(mesh.triangles, mesh.triangle_face_ids) if fid in fids]
        for sid, fids in mesh.solid_faces.items()
    }


def _counts(mesh, tags=("inner", "outer")):
    """{tag: triangles on that solid's boundary}."""
    return {tag: len(_solid_triangles(mesh)[_sid(mesh, tag)]) for tag in tags}


def _open_edges(mesh):
    """{solid id: edges of its boundary not used by exactly two of its
    triangles}, by vertex index: the assembler welds coincident vertices, so
    a watertight boundary is closed at the index level."""
    out = {}
    for sid, tris in _solid_triangles(mesh).items():
        uses = {}
        for a, b, c in tris:
            for edge in ((a, b), (b, c), (c, a)):
                key = (min(edge), max(edge))
                uses[key] = uses.get(key, 0) + 1
        out[sid] = sum(1 for n in uses.values() if n != 2)
    return out


def _volume_error(c2y, mesh, tag):
    """The relative error of a solid's mesh volume against its CAD volume."""
    sid = _sid(mesh, tag)
    return _solid_volumes(mesh)[sid] / _occ_signed_volume(c2y._processed.solid_shapes[sid]) - 1


def _per_solid_settings(fine):
    """Tolerances with *fine* at the fine values and every other solid coarse."""
    other = "outer" if fine == "inner" else "inner"
    return dict(
        tolerance={fine: FINE, other: COARSE},
        angular_tolerance={fine: FINE_ANGLE, other: COARSE_ANGLE},
    )


# Primitive solids, meshed by BRepMesh, sharing curved edges.


def _cylinder_in_tube():
    return _assembly(
        inner=cq.Workplane().cylinder(10, 3),
        outer=cq.Workplane().cylinder(10, 6).cut(cq.Workplane().cylinder(10, 3)),
    )


def _thin_tube_on_cylinder():
    return _assembly(
        inner=cq.Workplane().cylinder(10, 3),
        outer=cq.Workplane().cylinder(10, 3.05).cut(cq.Workplane().cylinder(10, 3)),
    )


def _sphere_in_shell(wall=3.0):
    return _assembly(
        inner=cq.Workplane().sphere(5),
        outer=cq.Workplane().sphere(5 + wall).cut(cq.Workplane().sphere(5)),
    )


def _torus_in_shell(wall):
    core = cq.Solid.makeTorus(10, 2)
    return _assembly(
        inner=cq.Workplane().add(core),
        outer=cq.Workplane().add(cq.Solid.makeTorus(10, 2 + wall).cut(core)),
    )


def _thin_hemispherical_shell():
    upper = cq.Workplane().box(20, 20, 10, centered=(True, True, False))
    return _assembly(
        inner=cq.Workplane().sphere(5).intersect(upper),
        outer=cq.Workplane().sphere(5.05).cut(cq.Workplane().sphere(5)).intersect(upper),
    )


def _revolved_pair():
    """Two profiles with an arc, revolved and stacked: they share an annulus
    bounded by two circles."""
    def revolved(z0):
        return (
            cq.Workplane("XZ").moveTo(5, z0).lineTo(10, z0)
            .threePointArc((11, z0 + 2.5), (10, z0 + 5)).lineTo(5, z0 + 5).close()
            .revolve(360, (0, 0, 0), (0, 1, 0))
        )
    return _assembly(inner=revolved(0), outer=revolved(5))


def _split_torus():
    """A torus cut in two by a plane through its axis: the halves share two
    discs, bounded by circles that the toroidal faces use too."""
    torus = cq.Workplane().add(cq.Solid.makeTorus(10, 3))
    half = cq.Workplane().box(40, 20, 20, centered=(True, False, True))
    return _assembly(inner=torus.intersect(half), outer=torus.cut(half))


def _rod_in_layers(layers=2):
    """A rod of radius 30 in tubes 0.2 thick: each tube's ends are annuli
    0.2 wide at radius 30."""
    parts = {"core": cq.Workplane().cylinder(60, 30)}
    for i in range(layers):
        inner, outer = 30 + 0.2 * i, 30.2 + 0.2 * i
        parts[f"l{i}"] = cq.Workplane().cylinder(60, outer).cut(cq.Workplane().cylinder(60, inner))
    return _assembly(**parts)


# b-spline solids, meshed by the scene CDT, sharing curved edges.


def _spline_prism(z0=0.0, side=-5.0):
    return (
        cq.Workplane("XY").workplane(offset=z0).moveTo(0, side).lineTo(0, 0)
        .spline([(2, 1.5), (4, -1), (6, 0.5), (8, 0)], includeCurrent=True)
        .lineTo(8, side).close().extrude(4)
    )


def _stacked_spline_prisms():
    """One spline-bounded profile extruded twice and stacked: the shared face
    is bounded by the spline, and each solid keeps spline edges of its own."""
    return _assembly(inner=_spline_prism(0.0), outer=_spline_prism(4.0))


def _spline_prisms_side_by_side():
    """Two prisms either side of one spline surface, which they share."""
    return _assembly(inner=_spline_prism(side=-5.0), outer=_spline_prism(side=5.0))


BREPMESH_ASSEMBLIES = [_cylinder_in_tube, _sphere_in_shell, _revolved_pair, _split_torus]
SCENE_ASSEMBLIES = [_stacked_spline_prisms, _spline_prisms_side_by_side]


@pytest.mark.parametrize("fine", ["inner", "outer"])
@pytest.mark.parametrize("build", BREPMESH_ASSEMBLIES, ids=lambda b: b.__name__[1:])
def test_brepmesh_assemblies_at_two_tolerances_are_watertight(build, fine):
    c2y = _mesher(build())
    mesh = c2y.mesh(**_per_solid_settings(fine))
    brep, scene = _route_faces(c2y._processed)
    assert brep and not scene
    assert mesh.shared_faces
    assert _open_edges(mesh) == {1: 0, 2: 0}


@pytest.mark.parametrize("fine", ["inner", "outer"])
@pytest.mark.parametrize(
    "build", [_thin_tube_on_cylinder, _thin_hemispherical_shell], ids=lambda b: b.__name__[1:]
)
def test_a_thin_shell_on_a_solid_at_another_tolerance_is_watertight(build, fine):
    """The shell is 0.05 thick. Coarse on the fine solid, it is too thin to
    be meshed coarser than that, so it takes the fine values (see the thin
    wall tests); fine on the coarse solid, the coarse solid's faces take the
    fine circles they share with it. Watertight, faithful to the CAD (the
    gate is on) and the shell keeps its volume either way."""
    c2y = _mesher(build())
    mesh = c2y.mesh(**_per_solid_settings(fine))
    brep, scene = _route_faces(c2y._processed)
    assert brep and not scene
    assert _open_edges(mesh) == {1: 0, 2: 0}
    assert abs(_volume_error(c2y, mesh, "outer")) < 0.02


@pytest.mark.parametrize("fine", ["inner", "outer"])
@pytest.mark.parametrize("build", SCENE_ASSEMBLIES, ids=lambda b: b.__name__[1:])
def test_scene_assemblies_at_two_tolerances_are_watertight(build, fine):
    c2y = _mesher(build())
    mesh = c2y.mesh(**_per_solid_settings(fine))
    brep, scene = _route_faces(c2y._processed)
    assert scene and not brep
    assert mesh.shared_faces
    assert _open_edges(mesh) == {1: 0, 2: 0}


@pytest.mark.parametrize(
    "build, values",
    [
        # The inner tube's rims, between the rod's fine circle and a coarse
        # one, come out with no triangles at all.
        (_rod_in_layers, {"core": (0.005, 0.05), "l0": (COARSE, 1.0), "l1": (COARSE, 1.0)}),
        # The outer tube's rim, between the inner tube's fine circle and a
        # coarse one, comes out as one stray triangle, with edges that still
        # match its neighbours'.
        (_rod_in_layers, {"core": (0.005, 0.05), "l0": (0.005, 0.05), "l1": (COARSE, 1.0)}),
        # The tube's rim gets the rod's circle discretized again, finer than
        # the rod's faces have it.
        (_thin_tube_on_cylinder, {"inner": (FINE, FINE_ANGLE), "outer": (COARSE, 1.0)}),
    ],
    ids=["rim without triangles", "rim as a stray triangle", "rod circle redone"],
)
def test_a_group_brepmesh_fails_at_mixed_values_is_meshed_again(build, values, monkeypatch):
    """Thin rims between a fine circle and a coarse one, which OCC meshes
    wrongly at mixed values without saying so. The check after meshing finds
    it, and the faces joined to the rim are meshed again at the finest values
    among them: here all of them, exactly as with those values everywhere.
    (Thin solids are meshed finer than they ask anyway, see the thin wall
    tests; that is lifted here to reach the check.)"""
    monkeypatch.setattr(mesher_module, "_THIN_WALL_FRACTION", 1e9)
    mesh = _mesher(build()).mesh(
        tolerance={tag: v[0] for tag, v in values.items()},
        angular_tolerance={tag: v[1] for tag, v in values.items()},
    )
    assert set(_open_edges(mesh).values()) == {0}
    lin, ang = min(v[0] for v in values.values()), min(v[1] for v in values.values())
    finest = _mesher(build()).mesh(tolerance=lin, angular_tolerance=ang)
    assert (mesh.vertices, mesh.triangles) == (finest.vertices, finest.triangles)


def test_a_face_whose_triangles_leave_its_edges_is_caught():
    """Every face's triangles close onto the polygons on its edges (seams
    and poles included), and a face missing one or left with one stray
    triangle does not."""
    for build in (_cylinder_in_tube, _sphere_in_shell, lambda: _torus_in_shell(1.0)):
        c2y = _mesher(build())
        c2y.mesh()
        for face in c2y._processed.face_to_occ.values():
            _, tris = _face_mesh(face)
            assert _follows_its_edges(face, tris)
            assert not _follows_its_edges(face, tris[1:])
            assert not _follows_its_edges(face, tris[:1])


@pytest.mark.parametrize(
    "build, coarse",
    [
        # Before the cap: 36% of the shell's volume lost, all gates passing.
        (lambda: _torus_in_shell(0.1), 0.1),
        # Before the cap: the shell's surfaces crossed, its volume negative.
        (lambda: _sphere_in_shell(0.3), 1.0),
    ],
    ids=["torus shell 0.1 thick", "sphere shell 0.3 thick"],
)
def test_a_thin_coarse_solid_on_a_finer_one_keeps_its_volume(build, coarse):
    """A shell's inner face is the fine core's, while its outer face takes
    its own deflection. Its outer facets sag toward its inner ones by up to
    that, much of a thin wall, where one value everywhere would sag both
    alike. So the shell is meshed no coarser than a fiftieth of 3V/A."""
    c2y = _mesher(build())
    mesh = c2y.mesh(
        tolerance={"inner": FINE, "outer": coarse},
        angular_tolerance={"inner": FINE_ANGLE, "outer": COARSE_ANGLE},
    )
    assert abs(_volume_error(c2y, mesh, "outer")) < 0.02


def test_a_solid_too_thin_for_its_tolerance_takes_its_finer_neighbours_values():
    """The tube is 0.05 thick, too thin to be coarser than the rod in it at
    all: it is meshed exactly as with the rod's values everywhere."""
    mixed = _mesher(_thin_tube_on_cylinder()).mesh(**_per_solid_settings("inner"))
    fine = _mesher(_thin_tube_on_cylinder()).mesh(tolerance=FINE, angular_tolerance=FINE_ANGLE)
    assert _counts(mixed)["outer"] == _counts(fine)["outer"]


def test_a_solid_sharing_only_flat_faces_with_a_finer_one_keeps_its_values():
    """Flat faces do not sag, so the split torus's coarse half, which shares
    only its two discs with the fine half, keeps its own values."""
    c2y = _mesher(_split_torus())
    mesh = c2y.mesh()
    fine, coarse = _sid(mesh, "inner"), _sid(mesh, "outer")
    lin = {fine: FINE, coarse: COARSE}
    ang = {fine: FINE_ANGLE, coarse: COARSE_ANGLE}
    assert mesher_module._cap_thin_solids(c2y._processed, lin, ang) == (lin, ang)


def test_a_thick_solid_next_to_a_finer_one_is_held_to_a_fiftieth_of_its_thickness():
    """The shell round the fine sphere runs from radius 5 to 8, so 3V/A is
    (8^3 - 5^3) / (8^2 + 5^2): it is meshed at its own deflection up to a
    fiftieth of that, as with that deflection everywhere."""
    def outer_face_triangles(mesh):
        (own,) = [f for f in mesh.solid_faces[_sid(mesh, "outer")] if f not in mesh.shared_faces]
        return mesh.triangle_face_ids.count(own)

    cap = 0.02 * (8**3 - 5**3) / (8**2 + 5**2)
    mixed = _mesher(_sphere_in_shell()).mesh(
        tolerance={"inner": FINE, "outer": 0.05}, angular_tolerance=COARSE_ANGLE
    )
    uniform = _mesher(_sphere_in_shell()).mesh(tolerance=0.05, angular_tolerance=COARSE_ANGLE)
    assert 0.05 < cap
    assert outer_face_triangles(mixed) == outer_face_triangles(uniform)

    c2y = _mesher(_sphere_in_shell())
    mesh = c2y.mesh()
    fine, coarse = _sid(mesh, "inner"), _sid(mesh, "outer")
    lin, ang = mesher_module._cap_thin_solids(
        c2y._processed, {fine: FINE, coarse: COARSE}, {fine: FINE_ANGLE, coarse: COARSE_ANGLE}
    )
    assert lin == {fine: FINE, coarse: pytest.approx(cap, rel=1e-6)}
    assert ang == {fine: FINE_ANGLE, coarse: COARSE_ANGLE}


@pytest.mark.parametrize(
    "shapes",
    [
        # BRepMesh: a sphere and a torus apart.
        lambda: (cq.Workplane().sphere(5), cq.Solid.makeTorus(10, 3, pnt=cq.Vector(30, 0, 0))),
        # Scene CDT: two spline prisms apart.
        lambda: (_spline_prism(0.0), _spline_prism(10.0)),
    ],
    ids=["brepmesh", "scene"],
)
def test_each_solid_is_meshed_as_finely_as_it_asks(shapes):
    """Two solids apart in one assembly, one fine and one coarse: each gets
    the triangles it gets meshed alone at its own values."""
    alone = {}
    for name, shape, tol, ang in zip(
        ("fine", "coarse"), shapes(), (FINE, COARSE), (FINE_ANGLE, COARSE_ANGLE)
    ):
        mesh = _mesher(_assembly(**{name: shape})).mesh(tolerance=tol, angular_tolerance=ang)
        alone[name] = len(mesh.triangles)
    fine, coarse = shapes()
    mesh = _mesher(_assembly(fine=fine, coarse=coarse)).mesh(
        tolerance={"fine": FINE, "coarse": COARSE},
        angular_tolerance={"fine": FINE_ANGLE, "coarse": COARSE_ANGLE},
    )
    assert _counts(mesh, alone) == alone
    assert alone["coarse"] * 5 < alone["fine"]


@pytest.mark.parametrize("fine", ["inner", "outer"])
@pytest.mark.parametrize(
    "build", [_split_torus, _revolved_pair, _stacked_spline_prisms], ids=lambda b: b.__name__[1:]
)
def test_touching_solids_keep_their_own_density_away_from_the_shared_faces(build, fine):
    """One solid fine, one coarse, touching: the fine one is meshed exactly as
    with the fine values everywhere (its edges shared with the coarse solid
    included), the coarse one far more coarsely, though it takes the fine
    values on the faces they share."""
    coarse = "outer" if fine == "inner" else "inner"
    everywhere_fine = _counts(_mesher(build()).mesh(tolerance=FINE, angular_tolerance=FINE_ANGLE))
    mixed = _counts(_mesher(build()).mesh(**_per_solid_settings(fine)))
    assert mixed[fine] == everywhere_fine[fine]
    assert mixed[coarse] < 0.6 * everywhere_fine[coarse]


def test_a_shared_face_is_meshed_at_the_finer_values_of_its_solids():
    """The sphere a core and a shell share is meshed at the smaller
    deflection and the smaller angle of the two, each taken on its own."""
    def shared_triangles(tolerance, angular_tolerance):
        mesh = _mesher(_sphere_in_shell()).mesh(
            tolerance=tolerance, angular_tolerance=angular_tolerance
        )
        (fid,) = mesh.shared_faces
        return mesh.triangle_face_ids.count(fid)

    for fine, coarse in (("inner", "outer"), ("outer", "inner")):
        split = {fine: FINE, coarse: COARSE}
        split_angle = {fine: FINE_ANGLE, coarse: COARSE_ANGLE}
        assert shared_triangles(split, COARSE_ANGLE) == shared_triangles(FINE, COARSE_ANGLE)
        assert shared_triangles(COARSE, split_angle) == shared_triangles(COARSE, FINE_ANGLE)
    # The deflection from the core, the angle from the shell.
    assert shared_triangles(
        {"inner": FINE, "outer": COARSE}, {"inner": COARSE_ANGLE, "outer": FINE_ANGLE}
    ) == shared_triangles(FINE, FINE_ANGLE)
    assert shared_triangles(COARSE, COARSE_ANGLE) * 10 < shared_triangles(FINE, COARSE_ANGLE)


def test_scene_edges_take_the_finest_value_whichever_face_comes_first():
    """A scene edge is discretized once for all its faces, at the finest of
    their values, so the faces' order does not matter."""
    c2y = _mesher(_stacked_spline_prisms())
    c2y.mesh()
    pa = c2y._processed
    _, faces = _route_faces(pa)
    (shared,) = pa.shared_faces
    mixed = dict.fromkeys(faces, (COARSE, COARSE_ANGLE)) | {shared: (FINE, FINE_ANGLE)}
    forward = scene_face_meshes(pa, mixed, faces)
    assert scene_face_meshes(pa, mixed, faces[::-1]) == forward
    fine = scene_face_meshes(pa, dict.fromkeys(faces, (FINE, FINE_ANGLE)), faces)
    assert forward[shared] == fine[shared]


def test_the_first_face_on_a_shared_edge_is_the_finest_in_both_tolerances():
    """The first face meshed on an edge discretizes it for all its faces. A
    face finer in deflection but coarser in angle than its neighbour (two
    solids given the two in opposite orders) is meshed at the finer of both;
    the placements of one shape share a triangulation, so all take the
    finest of their values."""
    mesh_at = {1: (0.01, 1.0), 2: (0.5, 0.1), 3: (1.0, 1.0)}
    _finest_first_on_shared_edges(mesh_at, {7: [1, 2], 8: [1, 3], 9: [2, 3]})
    assert mesh_at == {1: (0.01, 0.1), 2: (0.5, 0.1), 3: (1.0, 1.0)}
    mesh_at = {1: (0.01, 1.0), 2: (0.5, 0.1), 3: (1.0, 1.0)}
    _finest_first_on_shared_edges(mesh_at, {}, [[2, 3]])
    assert mesh_at == {1: (0.01, 1.0), 2: (0.5, 0.1), 3: (0.5, 0.1)}


@pytest.mark.parametrize("imprint", [True, False])
def test_placements_of_one_shape_are_meshed_at_the_finest_of_their_values(imprint):
    """A shape added twice shares one triangulation in OCC, so both
    placements are meshed at the smallest deflection and the smallest angle
    asked of either, which meets each one's request."""
    sphere = cq.Workplane().sphere(3)
    assembly = cq.Assembly()
    assembly.add(sphere, name="fuel")
    assembly.add(sphere, name="control", loc=cq.Location((10, 0, 0)))
    mesh = _mesher(assembly).mesh(
        imprint=imprint,
        tolerance={"fuel": FINE, "control": COARSE},
        angular_tolerance={"fuel": COARSE_ANGLE, "control": FINE_ANGLE},
    )
    alone = _mesher(_assembly(fuel=sphere)).mesh(tolerance=FINE, angular_tolerance=FINE_ANGLE)
    finest = len(alone.triangles)
    assert _counts(mesh, ("fuel", "control")) == {"fuel": finest, "control": finest}


def test_solid_ids_and_material_tags_both_set_values():
    """Ids are 1-based solid ids, a tag sets every solid with it, and solids
    the dict does not name take the default."""
    tags = ["first_wall", "steel", "steel", "coil"]
    assert _per_solid({1: 0.01, "steel": 0.5}, "tolerance", 0.1, tags) == {
        1: 0.01, 2: 0.5, 3: 0.5, 4: 0.1,
    }
    assert _per_solid(0.3, "tolerance", 0.1, tags) == {1: 0.3, 2: 0.3, 3: 0.3, 4: 0.3}
    assert _per_solid({}, "tolerance", 0.1, tags) == {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1}
    # One value given twice, by id and by tag, is fine.
    assert _per_solid({2: 0.5, "steel": 0.5}, "tolerance", 0.1, tags)[2] == 0.5


def test_numpy_numbers_are_accepted():
    """Keys and values from numpy work as Python ints and floats do."""
    def triangles(**settings):
        return _mesher(_cylinder_in_tube()).mesh(**settings).triangles

    assert triangles(
        tolerance={np.int64(1): np.float32(0.05)}, angular_tolerance=np.float64(0.3)
    ) == triangles(tolerance={1: float(np.float32(0.05))}, angular_tolerance=0.3)


def test_a_solid_given_two_values_is_refused():
    c2y = _mesher(_cylinder_in_tube())
    with pytest.raises(ValueError, match=r"Solid 1 \('inner'\) is given two tolerance values"):
        c2y.mesh(tolerance={1: 0.01, "inner": 0.5})
    with pytest.raises(ValueError, match="two angular_tolerance values"):
        c2y.mesh(angular_tolerance={"outer": 0.2, 2: 0.3})
    with pytest.raises(ValueError, match=r"values, 0\.3 and 0\.30000000000000004,"):
        c2y.mesh(tolerance={2: 0.3, "outer": 0.1 + 0.2})


def test_an_id_that_is_not_a_solid_is_refused():
    c2y = _mesher(_cylinder_in_tube())
    with pytest.raises(ValueError, match="key 3 is not a solid id: the largest solid id is 2"):
        c2y.mesh(tolerance={3: 0.1})


def test_a_material_tag_on_no_solid_is_refused():
    c2y = CadToYamc()
    c2y.add_cadquery_object(_cylinder_in_tube(), material_tags=["inner", "outer", "spare"])
    with pytest.raises(ValueError, match=r"'spare' in tolerance is on no solid.*\['inner', 'outer'\]"):
        c2y.mesh(tolerance={"spare": 0.01})


@pytest.mark.parametrize("method", ["mesh", "mesh_surfaces"])
@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        pytest.param(
            dict(tolerance={"stee": 0.1}), ValueError,
            r"Material tag 'stee' in tolerance not found\. Available material tags are: "
            r"\['inner', 'outer'\]",
            id="unknown tag",
        ),
        pytest.param(
            dict(angular_tolerance={"x": 0.1}), ValueError, "'x' in angular_tolerance not found",
            id="unknown tag, angular",
        ),
        pytest.param(dict(tolerance={0: 0.1}), ValueError, "key 0 is not a solid id", id="id 0"),
        pytest.param(
            dict(tolerance={-1: 0.1}), ValueError, "key -1 is not a solid id", id="negative id"
        ),
        pytest.param(
            dict(tolerance={True: 0.1}), TypeError, "keys must be int .* got bool", id="bool key"
        ),
        pytest.param(
            dict(tolerance={np.bool_(True): 0.1}), TypeError, "keys must be int .* got bool",
            id="numpy bool key",
        ),
        pytest.param(
            dict(tolerance={1.0: 0.1}), TypeError, "keys must be int .* got float", id="float key"
        ),
        pytest.param(
            dict(angular_tolerance={("inner",): 0.1}), TypeError, "got tuple", id="tuple key"
        ),
        pytest.param(
            dict(tolerance=0.0), ValueError, "tolerance must be positive and finite", id="zero"
        ),
        pytest.param(
            dict(tolerance=-0.1), ValueError, "tolerance must be positive and finite",
            id="negative",
        ),
        pytest.param(
            dict(tolerance=math.nan), ValueError, "tolerance must be positive and finite",
            id="nan",
        ),
        pytest.param(
            dict(tolerance=math.inf), ValueError, "tolerance must be positive and finite",
            id="inf",
        ),
        pytest.param(
            dict(angular_tolerance=0), ValueError, "angular_tolerance must be positive",
            id="zero angle",
        ),
        pytest.param(
            dict(tolerance={"inner": 0.0}), ValueError,
            r"tolerance\['inner'\] must be positive and finite, got 0\.0",
            id="zero in dict",
        ),
        pytest.param(
            dict(tolerance={2: -1}), ValueError, r"tolerance\[2\] must be positive",
            id="negative in dict",
        ),
        pytest.param(
            dict(angular_tolerance={"outer": math.inf}), ValueError,
            r"angular_tolerance\['outer'\] must be positive and finite",
            id="inf angle in dict",
        ),
        pytest.param(
            dict(tolerance="0.1"), TypeError, "a number, or a dict .* got str", id="str"
        ),
        pytest.param(dict(tolerance=None), TypeError, "got NoneType", id="None"),
        pytest.param(dict(tolerance=True), TypeError, "got bool", id="bool"),
        pytest.param(
            dict(tolerance={"inner": "0.1"}), TypeError,
            r"tolerance\['inner'\] must be a number",
            id="str in dict",
        ),
        pytest.param(
            dict(imprint=False, contact_tolerance={"inner": 0.1}), TypeError,
            "contact_tolerance must be a number, one distance for the whole model, got dict",
            id="dict contact tolerance",
        ),
        pytest.param(
            dict(imprint=False, contact_tolerance=True), TypeError,
            "contact_tolerance must be a number", id="bool contact tolerance",
        ),
    ],
)
def test_bad_tolerances_are_refused_before_the_imprint(method, kwargs, error, match, monkeypatch):
    def imprint(*args, **kwargs):
        raise AssertionError("imprinted before checking the tolerances")

    monkeypatch.setattr(mesher_module, "process_assembly", imprint)
    c2y = _mesher(_cylinder_in_tube())
    with pytest.raises(error, match=match):
        getattr(c2y, method)(**kwargs)


def test_float_and_uniform_dicts_mesh_identically():
    """A dict that gives every solid one value meshes exactly as that value
    given as a number; an empty dict as the defaults."""
    def mesh(**settings):
        m = _mesher(_split_torus()).mesh(**settings)
        return m.vertices, m.triangles, m.triangle_face_ids, m.triangle_surface_ids

    as_float = mesh(tolerance=0.2, angular_tolerance=0.4)
    assert mesh(tolerance={"inner": 0.2, "outer": 0.2}, angular_tolerance=0.4) == as_float
    assert mesh(tolerance=0.2, angular_tolerance={1: 0.4, 2: 0.4}) == as_float
    assert mesh(tolerance={}, angular_tolerance={}) == mesh()


@pytest.mark.parametrize("imprint", [True, False])
def test_meshing_again_with_other_values_meshes_afresh(imprint):
    """BRepMesh keeps a triangulation already on a shape when it is finer
    than asked, and a second mesh of the same assembly meshes the same shapes
    (imprinting leaves the faces it does not cut alone). Meshing again at
    coarser values gives what a fresh mesher gives, not the earlier mesh."""
    def triangles(c2y, **settings):
        return c2y.mesh(imprint=imprint, **settings).triangles

    for settings in (
        dict(tolerance=COARSE, angular_tolerance=COARSE_ANGLE),
        _per_solid_settings("inner"),
    ):
        c2y = _mesher(_split_torus())
        triangles(c2y, tolerance=FINE, angular_tolerance=FINE_ANGLE)
        assert triangles(c2y, **settings) == triangles(_mesher(_split_torus()), **settings)


@pytest.mark.parametrize("imprint", [True, False])
def test_shapes_tessellated_before_are_meshed_afresh(imprint):
    """Shapes an STL export or a viewer already tessellated finely mesh as
    fresh ones do, at one value and per solid."""
    def tessellated():
        assembly = _sphere_in_shell()
        for child in assembly.children:
            BRepMesh_IncrementalMesh(child.obj.val().wrapped, 0.001, False, 0.05)
        return assembly

    for settings in (
        dict(tolerance=COARSE, angular_tolerance=COARSE_ANGLE),
        _per_solid_settings("inner"),
    ):
        fresh = _mesher(_sphere_in_shell()).mesh(imprint=imprint, **settings)
        assert _mesher(tessellated()).mesh(imprint=imprint, **settings).triangles == fresh.triangles


def test_without_imprinting_each_solid_takes_its_own_values():
    """Without the imprint nothing is shared, so the coarse solid is meshed
    exactly as on its own, and every boundary is closed."""
    settings = _per_solid_settings("inner")
    mesh = _mesher(_cylinder_in_tube()).mesh(imprint=False, **settings)
    assert not mesh.shared_faces
    assert _open_edges(mesh) == {1: 0, 2: 0}
    alone = _mesher(_assembly(outer=_cylinder_in_tube().objects["outer"].obj)).mesh(
        tolerance=COARSE, angular_tolerance=COARSE_ANGLE
    )
    assert len(_solid_triangles(mesh)[_sid(mesh, "outer")]) == len(alone.triangles)


def _two_spheres():
    return _assembly(
        fine=cq.Workplane().sphere(10),
        coarse=cq.Workplane().sphere(10).translate((40, 0, 0)),
    )


def test_the_fidelity_gate_holds_each_solid_to_its_own_tolerance(monkeypatch):
    """A coarse solid is not refused for missing a finer solid's tolerance,
    and a fine solid is not let off by a coarser one's."""
    # The coarse sphere's shortfall in volume is what the gate is probed
    # with, so its faces are left inscribed rather than meshed to the CAD
    # volume.
    monkeypatch.setattr(mesher_module, "_PRESERVE_FACE_VOLUMES", False)
    c2y = _mesher(_two_spheres())
    tolerance = {"fine": FINE, "coarse": 1.0}
    mesh = c2y.mesh(tolerance=tolerance, angular_tolerance={"fine": FINE_ANGLE, "coarse": 0.8})
    fine, coarse = _sid(mesh, "fine"), _sid(mesh, "coarse")
    per_solid = {fine: FINE, coarse: 1.0}
    assert check_mesh_fidelity(mesh, c2y._processed, per_solid) == []
    # Held to the fine value, the coarse sphere fails: the gate needs its own.
    held_to_fine = " ".join(check_mesh_fidelity(mesh, c2y._processed, FINE))
    assert f"solid {coarse}:" in held_to_fine
    assert f"solid {fine}:" not in held_to_fine

    # The fine sphere shrunk 2%: within the coarse allowance, outside its own.
    fine_vertices = {i for tri in _solid_triangles(mesh)[fine] for i in tri}
    shrunk = dataclasses.replace(mesh, vertices=[
        [0.98 * x for x in v] if i in fine_vertices else v
        for i, v in enumerate(mesh.vertices)
    ])
    flagged = " ".join(check_mesh_fidelity(shrunk, c2y._processed, per_solid))
    assert f"solid {fine}:" in flagged
    assert all(f"face {fid}:" in flagged for fid in mesh.solid_faces[fine])
    assert f"solid {coarse}:" not in flagged
    assert check_mesh_fidelity(shrunk, c2y._processed, 1.0) == []


def test_the_contact_tolerance_defaults_to_twice_the_largest_deflection():
    """Without the imprint, transport takes one contact tolerance for the
    whole model: twice the largest deflection of any solid, those the dict
    does not name counting at 0.1."""
    c2y = _mesher(_two_spheres())
    c2y.mesh(imprint=False, tolerance={"fine": 0.02})
    assert c2y._contact_tolerance == pytest.approx(0.2)
    c2y.mesh(imprint=False, tolerance={"fine": 0.02, "coarse": 0.3})
    assert c2y._contact_tolerance == pytest.approx(0.6)


def _poking_rod_and_far_coil():
    """A rod whose end is 0.3 inside a block, and a coil far from both."""
    return _assembly(
        block=cq.Workplane().box(20, 20, 20),
        rod=cq.Workplane().circle(2).extrude(10).translate((0, 0, 9.7)),
        coil=cq.Workplane().add(cq.Solid.makeTorus(10, 2, pnt=cq.Vector(60, 0, 0))),
    )


@pytest.mark.parametrize("imprint", [True, False])
def test_the_overlap_gate_holds_each_pair_to_its_own_deflections(imprint):
    """Two facetings of one surface each stay within the coarser of their
    deflections of it, so an overlap between two solids deeper than twice
    that is in the CAD. A coarse coil elsewhere does not let the rod's 0.3
    into the block pass."""
    settings = dict(tolerance={"block": 0.02, "rod": 0.02, "coil": 1.0}, imprint=imprint)
    with pytest.raises(
        OverlappingSolidsError,
        match=r"deeper than twice the larger deflection of the two solids:\n  solid \d "
        r"\('block'\) and solid \d \('rod'\) overlap 0\.3 deep, more than the 0\.04 their "
        r"facetings explain",
    ):
        _mesher(_poking_rod_and_far_coil()).mesh(**settings)
    if not imprint:
        # Given a contact tolerance, the caller accepts slack up to it.
        _mesher(_poking_rod_and_far_coil()).mesh(contact_tolerance=0.4, **settings)
