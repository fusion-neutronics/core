"""Compare coupled transmutation on CSG vs CAD-mesh geometry.

Builds the same 10 cm iron cube two ways:
  1. CSG -- six axis-aligned planes, volume from stochastic sampling
  2. Mesh -- CadQuery box → CadToYamc surface mesh → Arrow IPC → MeshGeometry,
           volume from the divergence theorem over surface triangles

Neither geometry hardcodes the volume -- both compute it automatically.
Transmutation rates should match closely.

Requirements:
    pip install yamc[cad] cadquery

Usage:
    python examples/python/transmutation_csg_vs_mesh.py
"""

import os
import tempfile
import time

import cadquery as cq
import yamc
from yamc.cad import CadToYamc

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------
SIDE = 10.0        # cube side length [cm]
HALF = SIDE / 2.0  # half-width
NUC_DATA = "tests"  # directory containing Fe56.arrow
CHAIN_FILE = "tests/transmutation-endf-b8.1-sfr.arrow"

yamc.cross_section_data = NUC_DATA
yamc.transmutation_decay_data = CHAIN_FILE
yamc.transmutation_reactions = CHAIN_FILE
yamc.transmutation_fission_yields = CHAIN_FILE


# ---------------------------------------------------------------------------
# 1. CSG geometry -- cube from six planes
# ---------------------------------------------------------------------------
def build_csg():
    iron = yamc.Material(
        composition={"Fe56": 1.0},
        density=7.87,
        name="iron",
        transmutable=True,
        temperature=294)

    xn = yamc.Plane(axis="x", offset=-HALF, boundary="vacuum")
    xp = yamc.Plane(axis="x", offset=+HALF, boundary="vacuum")
    yn = yamc.Plane(axis="y", offset=-HALF, boundary="vacuum")
    yp = yamc.Plane(axis="y", offset=+HALF, boundary="vacuum")
    zn = yamc.Plane(axis="z", offset=-HALF, boundary="vacuum")
    zp = yamc.Plane(axis="z", offset=+HALF, boundary="vacuum")

    region = xn.above & xp.below & yn.above & yp.below & zn.above & zp.below
    cell = yamc.Cell(name="iron_cube", region=region, material=iron)
    geom = yamc.Geometry([cell])

    # Stochastic volume -- sets both cell.volume and material.volume
    volumes = geom.calculate_volume(samples=1_000_000)
    return geom, volumes[1].volume


# ---------------------------------------------------------------------------
# 2. Mesh geometry -- CadQuery → CadToYamc → MeshGeometry
# ---------------------------------------------------------------------------
def build_mesh():
    box = cq.Workplane("XY").box(SIDE, SIDE, SIDE)
    assy = cq.Assembly()
    assy.add(box, name="iron")

    tmpdir = tempfile.mkdtemp(prefix="yamc_dep_mesh_")
    arrow_path = os.path.join(tmpdir, "cube.arrow")

    c2y = CadToYamc()
    c2y.add_cadquery_object(assy, material_tags=["iron"])
    c2y.mesh(tolerance=0.1, angular_tolerance=0.1)
    c2y.to_arrow(arrow_path)

    iron = yamc.Material(
        composition={"Fe56": 1.0},
        density=7.87,
        name="iron",
        transmutable=True,
        temperature=294)

    # Volume is set automatically by MeshGeometry from the mesh surface
    mesh_geom = yamc.MeshGeometry(arrow_path, {"iron": iron})
    return mesh_geom


# ---------------------------------------------------------------------------
# Run transmutation on a geometry
# ---------------------------------------------------------------------------
def run_transmutation(geometry, label):
    source = yamc.NeutronSource(
        energy=14.06e6,
        position=(0.0, 0.0, 0.0))
    model = yamc.Model(geometry=geometry, source=source)

    day = 86400.0  # only for converting results.times (seconds) to days below
    schedule = yamc.PulseSchedule([
        yamc.Pulse(rate=1e20, duration=(1, "d"), source=source),   # irradiation
        yamc.Pulse(rate=1e12, duration=(1, "d"), source=source),   # irradiation
        yamc.Cooldown(duration=(1, "d")),                          # cooling
    ])

    t0 = time.time()
    results = model.simulate_transmutation(
        method="coupled",
        schedule=schedule,
        total_particles=2500,
    )
    elapsed = time.time() - t0

    fe56 = results.get_nuclide_evolution(material_id=1, nuclide="Fe56")
    print(f"\n{label} transmutation ({elapsed:.2f}s):")
    for i, density in enumerate(fe56):
        t_day = results.times[i] / day
        print(f"  t={t_day:.1f}d  Fe56 = {density:.12e} atoms/barn-cm")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ANALYTICAL = SIDE**3

print("=" * 60)
print("CSG vs Mesh transmutation comparison")
print(f"Cube: {SIDE} cm side,  analytical volume = {ANALYTICAL:.1f} cm³")
print("=" * 60)

# Build geometries
print("\nBuilding CSG geometry...")
csg_geom, csg_vol = build_csg()
print(f"  CSG cell volume: {csg_vol:.4f} cm³ (stochastic sampling)")

print("\nBuilding mesh geometry...")
mesh_geom = build_mesh()
for i, vol in enumerate(mesh_geom.volume_measures):
    print(f"  Mesh volume {i}: {vol:.4f} cm³ (divergence theorem)")

# Run transmutation
csg_results = run_transmutation(csg_geom, "CSG")
mesh_results = run_transmutation(mesh_geom, "Mesh")

# Compare final Fe56 densities
day = 86400.0
fe56_csg = csg_results.get_nuclide_evolution(material_id=1, nuclide="Fe56")
fe56_mesh = mesh_results.get_nuclide_evolution(material_id=1, nuclide="Fe56")

print("\n" + "=" * 60)
print("Comparison -- Fe56 density [atoms/barn-cm]")
print(f"{'Time':>6s}  {'CSG':>18s}  {'Mesh':>18s}  {'Rel diff':>12s}")
print("-" * 60)
for i in range(len(fe56_csg)):
    t_day = csg_results.times[i] / day
    c = fe56_csg[i]
    m = fe56_mesh[i]
    if c > 0:
        diff = abs(m - c) / c
        print(f"{t_day:5.1f}d  {c:18.12e}  {m:18.12e}  {diff:10.6%}")
    else:
        print(f"{t_day:5.1f}d  {c:18.12e}  {m:18.12e}  {'N/A':>10s}")
