"""Curved faces meshed to the volume the CAD face encloses.

A chordal mesh inscribes a curved surface, so a coarse mesh of a curved solid
is several percent too small. mesh() moves the vertices inside each curved
face and each curved edge off the surface, by no more than the deflection,
until the facets enclose the CAD volume, without adding a triangle. These
check that a coarse sphere and a coarse cylinder (whose curved face has no
vertex of its own, so only its edges can move) come out at their CAD volume,
and that a surface two imprinted solids share stays conformal.
"""

import pytest

cq = pytest.importorskip("cadquery")

from yamc.cad import CadToYamc  # noqa: E402
from yamc.cad import mesher as mesher_module  # noqa: E402


def _assembly(**parts):
    assembly = cq.Assembly()
    for name, shape in parts.items():
        assembly.add(shape, name=name)
    return assembly


def _mesh(assembly, monkeypatch, preserve, **kwargs):
    monkeypatch.setattr(mesher_module, "_PRESERVE_FACE_VOLUMES", preserve)
    c2y = CadToYamc()
    c2y.add_cadquery_object(assembly, material_tags="assembly_names")
    return c2y.mesh(**kwargs), c2y._processed


def _solid_volumes(mesh):
    """{solid id: volume its triangles enclose}."""
    face_volume = {}
    verts = mesh.vertices
    for tri, fid in zip(mesh.triangles, mesh.triangle_face_ids):
        a, b, c = (verts[i] for i in tri)
        face_volume[fid] = face_volume.get(fid, 0.0) + (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            + a[1] * (b[2] * c[0] - b[0] * c[2])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        ) / 6.0
    return {
        sid: sum(
            (-1.0 if mesh.face_solid_reversed[(sid, fid)] else 1.0) * face_volume[fid]
            for fid in fids
        )
        for sid, fids in mesh.solid_faces.items()
    }


def _relative_errors(mesh, pa):
    volumes = _solid_volumes(mesh)
    return {
        sid: volumes[sid] / pa.solid_shapes[sid].Volume(tol=1e-6) - 1.0
        for sid in volumes
    }


@pytest.mark.parametrize(
    "solid",
    [cq.Workplane().sphere(10), cq.Workplane().cylinder(20, 5)],
    ids=["sphere", "cylinder"],
)
def test_coarse_mesh_encloses_cad_volume(solid, monkeypatch):
    """At a coarse 1% of the size and 0.5 rad, the plain mesh of a sphere is
    4.7% and of a cylinder 1% short; preserved, both are within 0.05%, on
    the same triangles."""
    coarse = dict(tolerance=0.3, angular_tolerance=0.5)
    plain, plain_pa = _mesh(_assembly(part=solid), monkeypatch, False, **coarse)
    kept, kept_pa = _mesh(_assembly(part=solid), monkeypatch, True, **coarse)

    before = abs(_relative_errors(plain, plain_pa)[1])
    after = abs(_relative_errors(kept, kept_pa)[1])
    assert before > 0.009
    assert after < 5e-4
    assert after < before / 20.0
    assert kept.triangles == plain.triangles


def test_no_vertex_moves_further_than_the_deflection(monkeypatch):
    """Every vertex stays within the chordal deflection of where it was
    meshed."""
    coarse = dict(tolerance=0.3, angular_tolerance=0.5)
    solid = cq.Workplane().sphere(10).cut(cq.Workplane().cylinder(30, 4))
    plain, _ = _mesh(_assembly(part=solid), monkeypatch, False, **coarse)
    kept, _ = _mesh(_assembly(part=solid), monkeypatch, True, **coarse)
    assert len(kept.vertices) == len(plain.vertices)
    moved = [
        sum((p - q) ** 2 for p, q in zip(a, b)) ** 0.5
        for a, b in zip(kept.vertices, plain.vertices)
    ]
    assert max(moved) > 0.01
    assert max(moved) <= 0.3 * (1.0 + 1e-9)


def _open_edges(mesh, sid):
    """Edges of solid *sid*'s triangles, by vertex index, not used twice."""
    fids = set(mesh.solid_faces[sid])
    uses = {}
    for tri, fid in zip(mesh.triangles, mesh.triangle_face_ids):
        if fid in fids:
            for k in range(3):
                a, b = tri[k], tri[(k + 1) % 3]
                key = (min(a, b), max(a, b))
                uses[key] = uses.get(key, 0) + 1
    return sum(1 for n in uses.values() if n != 2)


def test_shared_curved_face_stays_conformal(monkeypatch):
    """A sphere in a spherical shell, imprinted: the sphere they share is one
    set of vertices, so moving it fixes both solids, and each stays closed
    on the vertex indices of the other."""
    core = cq.Workplane().sphere(5)
    shell = cq.Workplane().sphere(10).cut(core)
    coarse = dict(tolerance=0.3, angular_tolerance=0.5)
    plain, plain_pa = _mesh(
        _assembly(core=core, shell=shell), monkeypatch, False, **coarse
    )
    kept, kept_pa = _mesh(_assembly(core=core, shell=shell), monkeypatch, True, **coarse)

    assert kept.shared_faces, "the imprint should share the inner sphere"
    shared = next(iter(kept.shared_faces))
    shared_verts = {
        v for tri, fid in zip(kept.triangles, kept.triangle_face_ids)
        if fid == shared for v in tri
    }
    assert any(kept.vertices[v] != plain.vertices[v] for v in shared_verts)

    for sid in kept.solid_faces:
        assert _open_edges(kept, sid) == 0
    before = _relative_errors(plain, plain_pa)
    after = _relative_errors(kept, kept_pa)
    for sid in after:
        assert abs(before[sid]) > 0.03
        assert abs(after[sid]) < 1e-3
