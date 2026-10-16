"""Parts placed by a mirroring location.

A mirroring location (one with a negative determinant) is the usual way to
place the left and right copies of a symmetric part, and STEP assemblies use
them for the same reason. OCC reads a solid placed that way inside out: its
face normals point into it and its volume integrates negative. Meshed as it
was, its facets pointed into it. Imprinting it went wrong as well: the result
had an extra solid, and every solid was named after the mirrored part.
"""

import os

import pytest
import yamc

cq = pytest.importorskip("cadquery")
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf  # noqa: E402
from yamc.cad import CadToYamc  # noqa: E402
from yamc.cad._fidelity import check_mesh_fidelity  # noqa: E402
from yamc.cad.assembly_processor import process_assembly  # noqa: E402


def _mirror_in_yz():
    trsf = gp_Trsf()
    trsf.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)))
    return cq.Location(trsf)


def _box_from(x0):
    """A 10 cm box from x0 to x0 + 10, centred on y = z = 0."""
    return cq.Workplane().box(10, 10, 10).translate((x0 + 5, 0, 0))


def _row(mirrored):
    """Three touching 10 cm boxes in a row from x = -10 to 20. The left one
    is either the box at 0..10 placed through a mirror, or built in place."""
    assembly = cq.Assembly()
    if mirrored:
        assembly.add(_box_from(0), name="left", loc=_mirror_in_yz())
    else:
        assembly.add(_box_from(-10), name="left")
    assembly.add(_box_from(0), name="middle")
    assembly.add(_box_from(10), name="right")
    return assembly


def _signed_volumes(mesh):
    """Each solid's volume from its facets, as oriented for that solid:
    negative for a solid whose facets point into it."""
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
            (-1.0 if mesh.face_solid_reversed.get((sid, fid)) else 1.0)
            * face_volume.get(fid, 0.0)
            for fid in fids
        )
        for sid, fids in mesh.solid_faces.items()
    }


def test_a_mirrored_part_imprints_into_its_own_solid():
    result = process_assembly(_row(mirrored=True), ["steel", "lead", "water"])
    placed = {}
    for sid, tag in enumerate(result.material_tags, start=1):
        solid = result.solid_shapes[sid]
        box = solid.BoundingBox()
        placed[tag] = (round(box.xmin, 6), round(box.xmax, 6), round(solid.Volume(), 6))
    assert placed == {
        "steel": (-10.0, 0.0, 1000.0),
        "lead": (0.0, 10.0, 1000.0),
        "water": (10.0, 20.0, 1000.0),
    }
    # Imprinted like the row built in place: both interfaces are shared.
    assert len(result.shared_faces) == 2


def test_a_lone_mirrored_part_is_meshed_facing_out():
    assembly = cq.Assembly()
    assembly.add(_box_from(0), name="left", loc=_mirror_in_yz())
    c2y = CadToYamc()
    c2y.add_cadquery_object(assembly, material_tags=["steel"])
    mesh = c2y.mesh()
    assert _signed_volumes(mesh) == pytest.approx({1: 1000.0})
    xs = [v[0] for v in mesh.vertices]
    assert (min(xs), max(xs)) == pytest.approx((-10.0, 0.0))


def test_the_fidelity_gate_refuses_a_solid_meshed_inside_out():
    assembly = cq.Assembly()
    assembly.add(_box_from(0), name="box")
    c2y = CadToYamc()
    c2y.add_cadquery_object(assembly, material_tags=["steel"])
    mesh = c2y.mesh()
    assert check_mesh_fidelity(mesh, c2y._processed, 0.1) == []
    mesh.triangles = [[t[0], t[2], t[1]] for t in mesh.triangles]
    problems = check_mesh_fidelity(mesh, c2y._processed, 0.1)
    assert len(problems) == 1 and "inside out" in problems[0], problems


def _materials():
    out = {}
    for tag, (nuclide, density) in {
        "steel": ("Fe56", 7.874),
        "lead": ("Pb208", 11.35),
        "beryllium": ("Be9", 1.85),
    }.items():
        material = yamc.Material(
            composition={nuclide: 1.0}, density=density, name=tag, temperature=294
        )
        material.read_nuclear_data({nuclide: f"tests/{nuclide}.arrow"})
        out[tag] = material
    return out


def _fluxes(tmp_path, mirrored):
    c2y = CadToYamc()
    c2y.add_cadquery_object(_row(mirrored), material_tags=["steel", "lead", "beryllium"])
    c2y.mesh()
    path = os.path.join(tmp_path, f"row_{mirrored}.arrow")
    c2y.to_arrow(path)
    materials = _materials()
    geometry = yamc.MeshGeometry(path, materials, graveyard_offset=1.0)
    tallies = [
        yamc.Tally(scores=["flux"], materials=m, name=tag) for tag, m in materials.items()
    ]
    model = yamc.Model(
        geometry=geometry,
        tallies=tallies,
        source=yamc.NeutronSource(
            # In the left box, where the mirrored part goes.
            energy=yamc.sources.Discrete([14.06e6], [1.0]), position=(-4.0, 1.0, 0.5)
        ),
    )
    results = model.simulate_transport(total_particles=5_000, seed=3)
    return {t.name: results[t.name].mean[0] for t in tallies}


def test_a_mirrored_part_transports_like_the_part_built_in_place(tmp_path):
    mirrored = _fluxes(tmp_path, mirrored=True)
    in_place = _fluxes(tmp_path, mirrored=False)
    assert all(flux > 0.0 for flux in in_place.values())
    assert mirrored == pytest.approx(in_place, rel=1e-3)
