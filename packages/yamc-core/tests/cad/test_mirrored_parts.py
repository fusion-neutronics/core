"""Parts placed by a mirroring location.

A mirroring location (one with a negative determinant) is the usual way to
place the left and right copies of a symmetric part. OCC reads a solid placed
that way inside out: its face normals point into it and its volume integrates
negative. Meshed as it was, its facets pointed into it. Imprinting it went
wrong as well: the result had an extra solid, and every solid was named after
the mirrored part.
"""

import math
import os

import pytest
import yamc

cq = pytest.importorskip("cadquery")
from OCP.gp import gp_Ax2, gp_Ax3, gp_Dir, gp_Pnt, gp_Trsf  # noqa: E402
from yamc.cad import CadToYamc  # noqa: E402
from yamc.cad._fidelity import (  # noqa: E402
    MeshFidelityError, _occ_signed_volume, check_mesh_fidelity,
)
from yamc.cad.assembly_processor import process_assembly  # noqa: E402


def _placements():
    """Placements taking the box at x = 0..10 to x = -10..0, each with a
    flag for whether the box goes in reversed."""
    in_yz = gp_Trsf()
    in_yz.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)))
    through_origin = gp_Trsf()
    through_origin.SetMirror(gp_Pnt(0, 0, 0))
    # A frame mirrored in the yz plane: OCC keeps this reflection in the
    # matrix, with a positive scale factor.
    in_mirrored_frame = gp_Trsf()
    in_mirrored_frame.SetTransformation(
        gp_Ax3().Mirrored(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0))), gp_Ax3()
    )
    # Two reflections, (x, y) -> (-x, -y), make a rotation, though OCC
    # keeps one in the scale factor's sign and the other in the matrix.
    in_xz = gp_Trsf()
    in_xz.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)))
    x_reversed = gp_Ax3()
    x_reversed.XReverse()
    flip_x = gp_Trsf()
    flip_x.SetTransformation(x_reversed)
    return {
        "mirror in a plane": (in_yz, False),
        "mirror through a point": (through_origin, False),
        "mirrored frame": (in_mirrored_frame, False),
        "reversed solid through a mirror": (in_yz, True),
        "two reflections": (in_xz.Multiplied(flip_x), False),
    }


PLACEMENTS = list(_placements())


def _box_from(x0):
    """A 10 cm box from x0 to x0 + 10, centred on y = z = 0."""
    return cq.Workplane().box(10, 10, 10).translate((x0 + 5, 0, 0))


def _left(placement):
    """The left box, from -10 to 0: built in place when *placement* is None,
    otherwise the box at 0..10 placed by the named placement."""
    if placement is None:
        return _box_from(-10), cq.Location()
    trsf, reverse = _placements()[placement]
    shape = _box_from(0).val()
    if reverse:
        shape = cq.Solid(shape.wrapped.Reversed())
    return shape, cq.Location(trsf)


def _row(placement):
    """Three touching 10 cm boxes in a row from x = -10 to 20."""
    shape, loc = _left(placement)
    assembly = cq.Assembly()
    assembly.add(shape, name="left", loc=loc)
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


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_a_mirrored_part_imprints_into_its_own_solid(placement):
    result = process_assembly(_row(placement), ["steel", "lead", "water"])
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


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_a_lone_mirrored_part_is_meshed_facing_out(placement):
    shape, loc = _left(placement)
    assembly = cq.Assembly()
    assembly.add(shape, name="left", loc=loc)
    c2y = CadToYamc()
    c2y.add_cadquery_object(assembly, material_tags=["steel"])
    mesh = c2y.mesh()
    assert _signed_volumes(mesh) == pytest.approx({1: 1000.0})
    xs = [v[0] for v in mesh.vertices]
    assert (min(xs), max(xs)) == pytest.approx((-10.0, 0.0))


def test_a_mirrored_spline_solid_is_baked_facing_out():
    """Half a turn of a dense spline profile, placed by a mirror. OCC's
    default volume integration gets the sign of such a solid wrong, and
    baking the mirror into it turned it inside out."""
    angles = [2 * math.pi * i / 150 for i in range(150)]
    radii = [50 + 8 * math.cos(15 * a) for a in angles]
    profile = [(300 + r * math.cos(a), r * math.sin(a)) for r, a in zip(radii, angles)]
    solid = (
        cq.Workplane("XZ").spline(profile, periodic=True).close()
        .revolve(180, (0, 0, 0), (0, 1, 0)).val()
    )
    trsf, _ = _placements()["mirror in a plane"]
    assembly = cq.Assembly()
    assembly.add(solid, name="blanket", loc=cq.Location(trsf))
    baked = process_assembly(assembly, ["steel"], imprint=False).solid_shapes[1]
    assert _occ_signed_volume(baked) == pytest.approx(_occ_signed_volume(solid))


def _box_mesh():
    """A 10 cm box centred on the origin, meshed, with its converter."""
    assembly = cq.Assembly()
    assembly.add(cq.Workplane().box(10, 10, 10), name="box")
    c2y = CadToYamc()
    c2y.add_cadquery_object(assembly, material_tags=["steel"])
    mesh = c2y.mesh()
    assert check_mesh_fidelity(mesh, c2y._processed, 0.1) == []
    return c2y, mesh


def test_the_fidelity_gate_refuses_a_solid_meshed_inside_out():
    c2y, mesh = _box_mesh()
    mesh.triangles = [[t[0], t[2], t[1]] for t in mesh.triangles]
    problems = check_mesh_fidelity(mesh, c2y._processed, 0.1)
    assert len(problems) == 1 and "meshed inside out" in problems[0], problems


def test_the_fidelity_gate_refuses_a_mesh_whose_volume_comes_out_negative():
    # Each face of a box centred on the origin holds a sixth of its volume,
    # so turning four faces round leaves -1/3 of it: not the box inside out,
    # but no box either.
    c2y, mesh = _box_mesh()
    turned = sorted(set(mesh.triangle_face_ids))[:4]
    mesh.triangles = [
        [t[0], t[2], t[1]] if fid in turned else t
        for t, fid in zip(mesh.triangles, mesh.triangle_face_ids)
    ]
    problems = check_mesh_fidelity(mesh, c2y._processed, 0.1)
    assert len(problems) == 1 and "is negative" in problems[0], problems


def test_the_fidelity_gate_names_a_solid_that_is_inside_out_in_the_cad():
    assembly = cq.Assembly()
    assembly.add(cq.Solid(_box_from(0).val().wrapped.Reversed()), name="box")
    c2y = CadToYamc()
    c2y.add_cadquery_object(assembly, material_tags=["steel"])
    with pytest.raises(MeshFidelityError, match="inside out in the CAD"):
        c2y.mesh()


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


def _fluxes(tmp_path, placement):
    c2y = CadToYamc()
    c2y.add_cadquery_object(_row(placement), material_tags=["steel", "lead", "beryllium"])
    c2y.mesh()
    path = os.path.join(tmp_path, f"row_{placement}.arrow")
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


@pytest.mark.parametrize("placement", ["mirror in a plane", "mirrored frame"])
def test_a_mirrored_part_transports_like_the_part_built_in_place(tmp_path, placement):
    mirrored = _fluxes(tmp_path, placement)
    in_place = _fluxes(tmp_path, None)
    assert all(flux > 0.0 for flux in in_place.values())
    assert mirrored == pytest.approx(in_place, rel=1e-3)
