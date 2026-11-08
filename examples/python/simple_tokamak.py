import argparse
import math
import time

import yamc

parser = argparse.ArgumentParser(description="Simple tokamak TBR model.")
parser.add_argument(
    "--tracking-mode",
    choices=["surface", "woodcock", "hybrid"],
    default="surface",
    help="Particle tracking algorithm (default: surface). This model has a "
    "void vacuum vessel and air, so the delta-tracking choice here is "
    "'hybrid' (surface fallback in the voids); pure 'woodcock' would churn "
    "fictitious collisions in those regions and run slower.",
)
parser.add_argument(
    "--no-plots",
    action="store_true",
    help="Skip the (slow) geometry / mesh plots -- useful when comparing "
    "tracking modes for the tally only.",
)
args = parser.parse_args()
MAKE_PLOTS = not args.no_plots

# yamc.cross_section_data = 'fendl-3.2d'
yamc.cross_section_data = 'endf-b8.1'

material1 = yamc.Material(
    composition={"Be9": 1.0},
    density=1.85,
    temperature=294,
    name="beryllium")

material2 = yamc.Material(
    composition={"Li": 1.0},
    density=0.5,
    temperature=294,
    name="lithium")

material3 = yamc.Material(
    composition={"C12": 1.0},
    density=2.26,
    temperature=294,
    name="graphite")

material4 = yamc.Material(
    composition={"Be9": 1.0},
    density=1.85,
    temperature=294,
    name="magnet")

# Concrete, slimmed to its light constituents (O/Si/Al/H/Na). The trace
# Ca and Fe are dropped: as natural elements they expand to full isotope
# sets (Ca40-48, Fe54-58) whose embedded neutron data dominates the
# offline HTML (~460 MB) for a <6% atom-fraction contribution.
material5 = yamc.Material(
    composition={
        "O": 0.532,
        "Si": 0.337,
        "Al": 0.034,
        "H": 0.023,
        "Na": 0.016,
    },
    density=2.3,
    temperature=294,
    name="concrete")

# Air filling the space around the tokamak (atom fractions of dry air).
air = yamc.Material(
    composition={"N": 0.7847, "O": 0.2105,},
    density=0.001225,
    temperature=294,
    name="air")

# Coupled neutron→photon transport needs photon data for every material.
# The library keyword populates both neutron and photon cross sections.
for _mat in (material1, material2, material3, material4, material5, air):
    _mat.read_nuclear_data('endf-b8.1')


# surfaces
central_column_surface = yamc.Cylinder(axis="z", radius=100)
inner_sphere_surface = yamc.Sphere(radius=480)
middle_sphere_surface = yamc.Sphere(radius=500)
outer_sphere_surface = yamc.Sphere(radius=600)

# floor slab surfaces -- 3 m (300 cm) thick concrete slab, top face 5 cm
# below the lowest tokamak component (PF2 coil bottom at z = -645)
floor_top_surface = yamc.Plane(axis="z", offset=-650)
floor_bot_surface = yamc.Plane(axis="z", offset=-950)

# bioshield wall surfaces -- inner faces 10 m back from the outermost PF
# coil (r = 675), walls 2 m thick
wall_in_xmin = yamc.Plane(axis="x", offset=-1675)
wall_in_xmax = yamc.Plane(axis="x", offset=1675)
wall_in_ymin = yamc.Plane(axis="y", offset=-1675)
wall_in_ymax = yamc.Plane(axis="y", offset=1675)
wall_out_xmin = yamc.Plane(axis="x", offset=-1875)
wall_out_xmax = yamc.Plane(axis="x", offset=1875)
wall_out_ymin = yamc.Plane(axis="y", offset=-1875)
wall_out_ymax = yamc.Plane(axis="y", offset=1875)

# bioshield ceiling surfaces -- 5 m above the top of the tokamak (PF1 top
# at z = 675), 1.5 m thick
ceiling_bot_surface = yamc.Plane(axis="z", offset=1175)
ceiling_top_surface = yamc.Plane(axis="z", offset=1325)

# rectangular vacuum bounding box -- 1 m of void beyond the bioshield on
# every side so a mesh tally can image radiation leaking through the shield
bound_xmin = yamc.Plane(axis="x", offset=-1975, boundary='vacuum')
bound_xmax = yamc.Plane(axis="x", offset=1975, boundary='vacuum')
bound_ymin = yamc.Plane(axis="y", offset=-1975, boundary='vacuum')
bound_ymax = yamc.Plane(axis="y", offset=1975, boundary='vacuum')
bound_zmin = yamc.Plane(axis="z", offset=-1050, boundary='vacuum')
bound_zmax = yamc.Plane(axis="z", offset=1425, boundary='vacuum')

# PF coil surfaces -- 4 coils outside the spheres and above the floor
# Each coil is an annulus: inner/outer ZCylinder + top/bottom ZPlane
# (r_center, z_center, r_width, z_height) in cm -- all 50x50
pf_coil_specs = [
    (150, 650, 50, 50),   # PF1 -- upper inboard,  z: 625 → 675
    (150, -620, 50, 50),  # PF2 -- lower inboard,  z: -645 → -595 (above floor at -650)
    (650, 350, 50, 50),   # PF3 -- upper outboard, z: 325 → 375
    (650, -350, 50, 50),  # PF4 -- lower outboard, z: -375 → -325
]

pf_coil_cells = []
pf_coil_surfs = []  # (inner_cyl, outer_cyl, top_plane, bot_plane)
for i, (rc, zc, rw, zh) in enumerate(pf_coil_specs, start=1):
    inner_cyl = yamc.Cylinder(axis="z", radius=rc - rw / 2)
    outer_cyl = yamc.Cylinder(axis="z", radius=rc + rw / 2)
    top_plane = yamc.Plane(axis="z", offset=zc + zh / 2)
    bot_plane = yamc.Plane(axis="z", offset=zc - zh / 2)
    pf_coil_surfs.append((inner_cyl, outer_cyl, top_plane, bot_plane))
    region = inner_cyl.above & outer_cyl.below & top_plane.below & bot_plane.above
    cell = yamc.Cell(region=region, name=f'pf_coil_{i}', material=material4)
    pf_coil_cells.append(cell)

# extract PF coil surfaces for void slicing
pf1_inner, pf1_outer, pf1_top, pf1_bot = pf_coil_surfs[0]  # z: 625 → 675
pf2_inner, pf2_outer, pf2_top, pf2_bot = pf_coil_surfs[1]  # z: -645 → -595
pf3_inner, pf3_outer, pf3_top, pf3_bot = pf_coil_surfs[2]  # z: 325 → 375
pf4_inner, pf4_outer, pf4_top, pf4_bot = pf_coil_surfs[3]  # z: -375 → -325

# xy boxes (reused across slabs): room interior, bioshield footprint, and
# the outer vacuum boundary
room_xy_box = wall_in_xmin.above & wall_in_xmax.below & wall_in_ymin.above & wall_in_ymax.below
shield_xy_box = wall_out_xmin.above & wall_out_xmax.below & wall_out_ymin.above & wall_out_ymax.below
outer_xy_box = bound_xmin.above & bound_xmax.below & bound_ymin.above & bound_ymax.below
outside_room_xy = wall_in_xmin.below | wall_in_xmax.above | wall_in_ymin.below | wall_in_ymax.above
outside_shield_xy = wall_out_xmin.below | wall_out_xmax.above | wall_out_ymin.below | wall_out_ymax.above

# tokamak regions (inside outer sphere)
central_column_region = central_column_surface.below & outer_sphere_surface.below
firstwall_region = middle_sphere_surface.below & inner_sphere_surface.above & central_column_surface.above
blanket_region = middle_sphere_surface.above & outer_sphere_surface.below & central_column_surface.above
inner_vessel_region = central_column_surface.above & inner_sphere_surface.below

# floor slab -- spans the full bioshield footprint (the walls sit on it)
floor_region = shield_xy_box & floor_top_surface.below & floor_bot_surface.above

# bioshield walls (floor top → ceiling bottom) and ceiling (full footprint)
wall_region = shield_xy_box & outside_room_xy & ceiling_bot_surface.below & floor_top_surface.above
ceiling_region = shield_xy_box & ceiling_top_surface.below & ceiling_bot_surface.above

# 1 m void gap between the bioshield and the vacuum boundary
void_gap_side = outer_xy_box & outside_shield_xy & ceiling_top_surface.below & floor_bot_surface.above
void_gap_top = outer_xy_box & bound_zmax.below & ceiling_top_surface.above
void_gap_bottom = outer_xy_box & floor_bot_surface.below & bound_zmin.above

# air slabs -- working top-down inside the room, PF coil planes slice the space
# 1. above PF1 top (z: 675 → 1175, up to the ceiling)
void_a = room_xy_box & ceiling_bot_surface.below & pf1_top.above
# 2. PF1 level, outside PF1 outer cylinder (z: 625 → 675)
void_b = room_xy_box & pf1_top.below & pf1_bot.above & (pf1_outer.above | pf1_inner.below)
# 3. between PF1 and PF3 (z: 375 → 625), outside sphere
void_c = room_xy_box & pf1_bot.below & pf3_top.above & outer_sphere_surface.above
# 4. PF3 level, outside PF3 outer cylinder (z: 325 → 375), outside sphere
void_d = room_xy_box & pf3_top.below & pf3_bot.above & outer_sphere_surface.above & (pf3_outer.above | pf3_inner.below)
# 5. between PF3 and PF4 (z: -325 → 325), outside sphere
void_e = room_xy_box & pf3_bot.below & pf4_top.above & outer_sphere_surface.above
# 6. PF4 level, outside PF4 outer cylinder (z: -375 → -325), outside sphere
void_f = room_xy_box & pf4_top.below & pf4_bot.above & outer_sphere_surface.above & (pf4_outer.above | pf4_inner.below)
# 7. between PF4 and PF2 (z: -595 → -375), outside sphere
void_g = room_xy_box & pf4_bot.below & pf2_top.above & outer_sphere_surface.above
# 8. PF2 level, outside PF2 outer cylinder (z: -645 → -595), outside sphere
void_h = room_xy_box & pf2_top.below & pf2_bot.above & outer_sphere_surface.above & (pf2_outer.above | pf2_inner.below)
# 9. between PF2 and floor (z: -650 → -645)
void_i = room_xy_box & pf2_bot.below & floor_top_surface.above

# cells
firstwall_cell = yamc.Cell(region=firstwall_region, name='firstwall', material=material1)
central_column_cell = yamc.Cell(region=central_column_region, name='central_column', material=material3)
blanket_cell = yamc.Cell(region=blanket_region, name='blanket', material=material2)
inner_vessel_cell = yamc.Cell(region=inner_vessel_region, name='inner_vessel')
floor_cell = yamc.Cell(region=floor_region, name='floor', material=material5)
wall_cell = yamc.Cell(region=wall_region, name='bioshield_wall', material=material5)
ceiling_cell = yamc.Cell(region=ceiling_region, name='bioshield_ceiling', material=material5)

# The space around the tokamak is filled with air. (The inner vacuum
# vessel stays a true void -- no material.)
air_cells = []
for name, region in [('air_a', void_a), ('air_b', void_b), ('air_c', void_c),
                     ('air_d', void_d), ('air_e', void_e), ('air_f', void_f),
                     ('air_g', void_g), ('air_h', void_h), ('air_i', void_i)]:
    air_cells.append(yamc.Cell(region=region, name=name, material=air))

# true-void gap cells (no material) between the bioshield and the vacuum
# boundary -- lets a mesh tally image radiation leaking through the shield
void_gap_cells = [
    yamc.Cell(region=void_gap_side, name='void_gap_side'),
    yamc.Cell(region=void_gap_top, name='void_gap_top'),
    yamc.Cell(region=void_gap_bottom, name='void_gap_bottom'),
]

all_cells = ([central_column_cell, firstwall_cell, blanket_cell, inner_vessel_cell,
              floor_cell, wall_cell, ceiling_cell] + air_cells + void_gap_cells
             + pf_coil_cells)
geometry = yamc.Geometry(all_cells)

# visualization
color_assignment = {
    blanket_cell: 'blue',
    firstwall_cell: 'orange',
    inner_vessel_cell: 'grey',
    central_column_cell: 'purple',
    floor_cell: 'brown',
    wall_cell: 'brown',
    ceiling_cell: 'brown',
}
for vc in air_cells:
    color_assignment[vc] = 'lightblue'
for vg in void_gap_cells:
    color_assignment[vg] = 'white'
for pf_cell in pf_coil_cells:
    color_assignment[pf_cell] = 'red'


if MAKE_PLOTS:
    for what in ["cell"]:
        for basis in ['xy', 'xz', 'yz']:
            plot = geometry.plot(
                # origin=bb.center,  # automatically found
                # width=(bb.width[0], bb.width[1]), # automatically found
                resolution=10000,
                basis=basis,
                color_by=what,
                outline='cell',
                axis_units="m",
                colors=color_assignment
            )
            plot.save(f'{what}_{basis}.html')
            plot.save(f'{what}_{basis}.png')
            print(f'Wrote {what}_{basis}.html and .png')

source = yamc.NeutronSource(
    energy=14.06e6,  # 14 MeV neutrons
    position=yamc.sources.CylindricalRing(
        radius=yamc.sources.Discrete([300], [1]),  # ring at radius=300 cm (midplane of blanket)
        phi=yamc.sources.Uniform(0, 2 * math.pi),  # full ring
        z=yamc.sources.Discrete([0], [1]),  # midplane
    )
)

mesh = yamc.RegularRectangularMesh.from_domain(geometry, shape=(20, 20, 20))

tally1 = yamc.Tally(mesh=mesh, scores=[105], name="tritium production map")
tally2 = yamc.Tally(cells=blanket_cell, scores=[105], name="tritium production blanket")
# Photon heating deposited in the (beryllium) magnets. Coupled
# neutron→photon transport produces these photons from neutron
# interactions; the "heating" score requires the collision estimator.
tally3 = yamc.Tally(cells=pf_coil_cells, scores=["heating"], particle="photon",
                    estimator="collision", name="magnet photon heating")

model = yamc.Model(geometry=geometry, tallies=[tally1, tally2, tally3], source=source,
                   tracking_mode=args.tracking_mode, transport_secondary_photons=True)


# Interactive plots -- pan/zoom/slice in the browser without re-running Python
if MAKE_PLOTS:
    for basis in ['xy', 'xz', 'yz']:
        plot = model.plot(
            resolution=10000, basis=basis, color_by="cell",
            outline="cell", axis_units="m", colors=color_assignment,
            n_samples=100)
        plot.save(f"interactive_tokamak_model_{basis}.html")
        plot.save(f"interactive_tokamak_model_{basis}.png")
        print(f"Wrote interactive_tokamak_model_{basis}.html and .png")


t0 = time.perf_counter()
results = model.simulate_transport(total_particles=1000000, seed=42)
elapsed = time.perf_counter() - t0

tbr = float(results[tally2].mean[0])
tbr_std = float(results[tally2].standard_deviation[0])
heat = float(results[tally3].mean[0])
heat_std = float(results[tally3].standard_deviation[0])
print(f'tracking_mode = {args.tracking_mode}')
print(f'TBR = {tbr} +/- {tbr_std}')
print(f'magnet photon heating = {heat} +/- {heat_std} eV/source-neutron')
print(f'wall time (incl. data load) = {elapsed:.3f} s')

# Seed both HTML exports (below) with this run's scalar results.
result_summary = (
    f"TBR = {tbr} +/- {tbr_std}\n"
    f"magnet photon heating = {heat} +/- {heat_std} eV/source-neutron"
)

# Mesh-tally overlay -- built once (needed for the embedded HTML even with
# --no-plots); the standalone file is only saved when plots are enabled.
mesh_plot = tally1.plot(
    geometry=geometry,
    basis="xz",
    resolution=10000)
if MAKE_PLOTS:
    mesh_plot.save("mesh_with_geometry_plot.html")
    print('Wrote mesh_with_geometry_plot.html')
tally_overlay_html = mesh_plot._repr_html_()

# Browser export model. Same physics as the native run (coupled
# neutron→photon transport, all three tallies); the in-browser re-run
# uses the viewer's default particle count (kept modest so the scalar
# re-run stays quick). The in-browser sim fetches both neutron
# (per-nuclide) and photon (per-element) cross sections from
# yamc-data.xsplot.com on Simulate.
export_tally_map = yamc.Tally(mesh=mesh, scores=[105], name="tritium production map")
export_tally_blanket = yamc.Tally(cells=blanket_cell, scores=[105], name="tritium production blanket")
export_tally_heat = yamc.Tally(cells=pf_coil_cells, scores=["heating"], particle="photon",
                               estimator="collision", name="magnet photon heating")
export_model = yamc.Model(
    geometry=geometry,
    tallies=[export_tally_map, export_tally_blanket, export_tally_heat], source=source,
    tracking_mode=args.tracking_mode, transport_secondary_photons=True)

# Self-contained, browser-runnable exports. The recipient opens the HTML
# and re-runs the simulation in-browser (wasm) -- no Python. Each is
# seeded with this run's numbers and the mesh-tally overlay.
#   - online:  fetches cross sections from yamc-data.xsplot.com on Simulate
#   - offline: embeds every referenced nuclide (large file, but runs with no
#              network connection)
export_model.to_html(
    "simple_tokamak.html",
    embed_cross_sections=False,
    tally_plot_html=tally_overlay_html,
    initial_result_text=result_summary,
)
print("Wrote simple_tokamak.html (online; fetches nuclear data on Simulate)")

offline = export_model.to_html(
    "simple_tokamak_offline.html",
    embed_cross_sections=True,
    tally_plot_html=tally_overlay_html,
    initial_result_text=result_summary,
)
size_mb = offline.stat().st_size / 1e6
print(f"Wrote simple_tokamak_offline.html "
      f"(offline; nuclear data embedded, {size_mb:.0f} MB)")