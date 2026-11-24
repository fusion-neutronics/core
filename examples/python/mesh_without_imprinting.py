"""
Meshing CAD without imprinting, and transporting the result.

Imprinting makes touching solids share the face where they meet, so the face is
meshed once and the mesh is conformal. It is also the slow, serial step on a
large assembly, and it can go wrong on CAD that is not quite clean. With
``imprint=False`` every solid is meshed on its own (in parallel), and two solids
that touch meet along two separately faceted surfaces that may overlap or leave
thin gaps. ``MeshGeometry`` tracks such a mesh discontinuously: a particle
leaving one solid is placed in whichever touching solid contains it.

Geometry: a cylinder inside two annuli. The annuli are rotated, so each shared
cylindrical face is faceted differently on either side and the two facetings
cross, which is exactly what meshing without imprinting produces.

The same CAD is meshed both ways and transported from the same source with the
same seed; the fluxes agree to a few hundredths of a percent.

Requirements:
    pip install yamc[cad]
"""

import cadquery as cq
import yamc

# ---------------------------------------------------------------------------
# 1. CadQuery: nested cylinders, annuli rotated so their seams differ
# ---------------------------------------------------------------------------

height = 20.0


def nested_cylinders():
    core = cq.Workplane().circle(5).extrude(height).translate((0, 0, -height / 2))
    middle = (
        cq.Workplane().circle(10).circle(5).extrude(height)
        .translate((0, 0, -height / 2)).rotate((0, 0, 0), (0, 0, 1), 7)
    )
    outer = (
        cq.Workplane().circle(15).circle(10).extrude(height)
        .translate((0, 0, -height / 2)).rotate((0, 0, 0), (0, 0, 1), 13)
    )
    assembly = cq.Assembly()
    assembly.add(core, name="beryllium")
    assembly.add(middle, name="iron")
    assembly.add(outer, name="lead")
    return assembly


# ---------------------------------------------------------------------------
# 2. Materials
# ---------------------------------------------------------------------------


def materials():
    out = {}
    for name, nuclide, density in (
        ("beryllium", "Be9", 1.85),
        ("iron", "Fe56", 7.874),
        ("lead", "Pb208", 11.35),
    ):
        m = yamc.Material(
            composition={nuclide: 1.0}, density=density, name=name, temperature=294
        )
        m.read_nuclear_data({nuclide: f"tests/{nuclide}.arrow"})
        out[name] = m
    return out


# ---------------------------------------------------------------------------
# 3. Mesh it twice, imprinted (conformal) and not (discontinuous), and
#    transport both from the same source with the same seed
# ---------------------------------------------------------------------------

fluxes = {}
for imprint in (True, False):
    mats = materials()
    # The non-imprinted mesh has no shared surfaces to tag as an outer
    # boundary, so a graveyard box bounds both models.
    geometry = yamc.MeshGeometry.from_cadquery(
        nested_cylinders(), mats, imprint=imprint, graveyard_offset=5.0
    )
    print(
        f"\nimprint={imprint}: {geometry.num_triangles} triangles, "
        f"tracked discontinuously={geometry.discontinuous}, "
        f"contact tolerance={geometry.contact_tolerance}"
    )
    tallies = {name: yamc.Tally(scores=["flux"], materials=m, name=name)
               for name, m in mats.items()}
    model = yamc.Model(
        geometry=geometry,
        tallies=list(tallies.values()),
        source=yamc.NeutronSource(energy=14.06e6, position=(0, 0, 0)),
    )
    results = model.simulate_transport(total_particles=100_000, seed=1)
    print(f"lost particles: {len(model.lost_particles)}")
    fluxes[imprint] = {name: results[t].mean[0] for name, t in tallies.items()}

print("\nFlux per material (cm per source particle):")
print(f"{'material':>10}  {'imprinted':>12}  {'not imprinted':>14}  {'difference':>10}")
for name in fluxes[True]:
    a, b = fluxes[True][name], fluxes[False][name]
    print(f"{name:>10}  {a:12.5e}  {b:14.5e}  {100 * (b - a) / a:+9.3f}%")
