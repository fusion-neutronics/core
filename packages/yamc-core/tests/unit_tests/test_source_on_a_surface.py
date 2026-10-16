"""A point source sitting exactly on a surface.

A particle born on a surface belongs to the cell it is heading into, as in
OpenMC. Its position alone cannot say which: a CSG point on a surface is in
neither half-space, so the particle was lost at birth, and a mesh point on a
facet was put on one side of it whatever the particle's direction, so the
particles heading the other way carried the wrong material until their next
crossing, with no particle lost.
"""

import os

import pytest
import yamc


def _materials():
    out = {}
    for tag, (nuclide, density) in {"light": ("Be9", 1.85), "heavy": ("Pb208", 11.35)}.items():
        material = yamc.Material(
            composition={nuclide: 1.0}, density=density, name=tag, temperature=294
        )
        material.read_nuclear_data({nuclide: f"tests/{nuclide}.arrow"})
        out[tag] = material
    return out


def _fluxes(geometry, materials, x):
    tallies = [yamc.Tally(scores=["flux"], materials=m, name=t) for t, m in materials.items()]
    model = yamc.Model(
        geometry=geometry,
        tallies=tallies,
        source=yamc.NeutronSource(
            energy=yamc.sources.Discrete([14.06e6], [1.0]), position=(x, 0.3, 0.1)
        ),
    )
    results = model.simulate_transport(total_particles=20_000, seed=7)
    return {
        t.name: (results[t.name].mean[0], results[t.name].standard_deviation[0])
        for t in tallies
    }


def _assert_agree(on, beside):
    """Within 4 sigma: a source a hair off the surface draws one more
    distance to collision for the particles heading across it, so the
    histories are not in lockstep."""
    for tag, (a, sa) in on.items():
        b, sb = beside[tag]
        assert abs(a - b) < 4.0 * (sa * sa + sb * sb) ** 0.5, (tag, a, b)


def _csg(materials):
    sphere = yamc.Sphere(radius=10, boundary="vacuum")
    plane = yamc.Plane("x", 0.0)
    return yamc.Geometry(
        [
            yamc.Cell(name="light", region=sphere.below & plane.below, material=materials["light"]),
            yamc.Cell(name="heavy", region=sphere.below & plane.above, material=materials["heavy"]),
        ]
    )


def test_a_source_on_a_csg_surface_starts_where_each_particle_heads():
    # A source on the plane runs (it lost every particle) and agrees with
    # sources a hair either side of it.
    def fluxes(x):
        materials = _materials()
        return _fluxes(_csg(materials), materials, x)

    on = fluxes(0.0)
    for x in (1e-7, -1e-7):
        _assert_agree(on, fluxes(x))


def test_a_source_on_a_shared_mesh_face_starts_where_each_particle_heads(tmp_path):
    cq = pytest.importorskip("cadquery")
    from yamc.cad import CadToYamc

    assembly = cq.Assembly()
    assembly.add(cq.Workplane().box(10, 10, 10).translate((-5, 0, 0)), name="light")
    assembly.add(cq.Workplane().box(10, 10, 10).translate((5, 0, 0)), name="heavy")
    c2y = CadToYamc()
    c2y.add_cadquery_object(assembly, material_tags=["light", "heavy"])
    c2y.mesh()
    path = os.path.join(tmp_path, "boxes.arrow")
    c2y.to_arrow(path)

    def fluxes(x):
        materials = _materials()
        geometry = yamc.MeshGeometry(path, materials, graveyard_offset=1.0)
        return _fluxes(geometry, materials, x)

    on = fluxes(0.0)
    for x in (1e-7, -1e-7):
        _assert_agree(on, fluxes(x))
