"""
Virtual-overlay tally example (the ``response=`` argument).

Demonstrates scoring a *virtual* response on a mesh across the whole geometry
(void, iron, ...) without modifying it -- useful for dose/damage maps of a
detector material that isn't physically present. Two flavours:

  * ``response="Fe56"``  -- microscopic XS at unit density (one bin per nuclide).
  * ``response=<Material>`` -- the material's *macroscopic* response directly,
    weighting each nuclide's microscopic XS by its real atom density (one bin).

Uses Fe56 / an iron material here (the concept is identical to a Si28 / SiO2
overlay for silicon dose). Inspired by the neutronics-workshop example
`6_silicon_dose_on_mesh.py`.
"""

import yamc

# ---------------------------------------------------------------------------
# Global config: register nuclear data paths so overlay tallies can find them
# ---------------------------------------------------------------------------
yamc.cross_section_data = {"Fe56": "tests/Fe56.arrow"}

# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
iron = yamc.Material(
    composition={"Fe56": 1.0},
    density=7.874,
    temperature=294)
iron.read_nuclear_data(
    {"Fe56": "tests/Fe56.arrow"},
    photon_data={"Fe": "tests/Fe.arrow"})

# ---------------------------------------------------------------------------
# Geometry: nested spheres -- iron shell surrounding a void plasma region
# ---------------------------------------------------------------------------
# Inner void (plasma): r < 10
sphere_inner = yamc.Sphere(x0=0, y0=0, z0=0, radius=10.0)
# Iron shell: 10 < r < 30
sphere_outer = yamc.Sphere(x0=0, y0=0, z0=0, radius=30.0, boundary="vacuum")

cell_plasma = yamc.Cell(name="plasma_void", region=sphere_inner.below)
cell_shell = yamc.Cell(name="iron_shell", region=sphere_inner.above & sphere_outer.below, material=iron)

geometry = yamc.Geometry([cell_plasma, cell_shell])

# ---------------------------------------------------------------------------
# Source: 14 MeV neutrons at center (simulating DT fusion)
# ---------------------------------------------------------------------------
source = yamc.NeutronSource(
    position=(0, 0, 0),
    energy=14e6)

# ---------------------------------------------------------------------------
# Tallies
# ---------------------------------------------------------------------------
# Mesh covering the full geometry
mesh = yamc.RegularRectangularMesh.from_domain(geometry, shape=(10, 10, 10))

# 1. Fe56 neutron heating overlay -- microscopic heating XS everywhere (barns)
tally_n_overlay = yamc.Tally(
    scores=["heating"],
    response="Fe56",
    name="Fe56_neutron_heating_overlay",
    mesh=mesh,
    particle="neutron")

# 2. Fe56 photon heating overlay -- uses track-length KERMA estimator
tally_p_overlay = yamc.Tally(
    scores=["heating"],
    response="Fe56",
    name="Fe56_photon_heating_overlay",
    mesh=mesh,
    particle="photon")

# 3. Iron *material* response -- the macroscopic neutron heating of iron,
#    weighted by its real atom density, scored everywhere (NEW in #341).
tally_n_material = yamc.Tally(
    scores=["heating"],
    response=iron,
    name="iron_material_neutron_heating",
    mesh=mesh,
    particle="neutron")

# 4. Reference: normal (macroscopic) neutron heating -- only scores in iron
tally_n_normal = yamc.Tally(
    scores=["heating"],
    name="normal_neutron_heating",
    mesh=mesh,
    particle="neutron")

tallies = [tally_n_overlay, tally_p_overlay, tally_n_material, tally_n_normal]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
model = yamc.Model(geometry=geometry, tallies=tallies, source=source, transport_secondary_photons=True)
results = model.simulate_transport(total_particles=25000, seed=42)

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Virtual response (response=) vs Normal tally comparison")
print("=" * 70)

for tally in tallies:
    mean = results[tally].mean
    total = sum(mean)
    nonzero = sum(1 for m in mean if m > 0)
    print(f"\n{tally.name}:")
    print(f"  Total heating across mesh: {total:.4e} eV / source-particle")
    print(f"  Non-zero voxels: {nonzero} / {len(mean)}")

# The material response is the per-nuclide overlay weighted by atom density --
# done natively, no post-processing. Verify it matches the manual workaround.
n_fe56 = iron.get_atoms_per_barn_cm()["Fe56"]
material_total = sum(results[tally_n_material].mean)
workaround_total = sum(results[tally_n_overlay].mean) * n_fe56
print(f"\nMaterial response total:           {material_total:.4e} eV / source-particle")
print(f"Overlay x atom-density (workaround): {workaround_total:.4e} eV / source-particle")
print("(These should match -- #341 moves the density weighting into the tally.)")

# Compare: overlay should score in void region, normal should not
n_overlay_mean = results[tally_n_overlay].mean
n_normal_mean = results[tally_n_normal].mean

overlay_void_score = sum(n_overlay_mean[:500])  # First 500 bins (inner region)
normal_void_score = sum(n_normal_mean[:500])

print(f"\nOverlay neutron heating in inner region (bins 0-499): {overlay_void_score:.4e}")
print(f"Normal neutron heating in inner region  (bins 0-499): {normal_void_score:.4e}")
print("(Overlay should be non-zero in void; normal should be ~zero)")

# Generate interactive HTML plots if mesh tally has data
plot = tally_n_overlay.plot(geometry=geometry, basis="xz")
plot.save("output/overlay_neutron_heating.html")
print("\nPlot saved to output/overlay_neutron_heating.html")

plot = tally_p_overlay.plot(geometry=geometry, basis="xz")
plot.save("output/overlay_photon_heating.html")
print("Plot saved to output/overlay_photon_heating.html")
