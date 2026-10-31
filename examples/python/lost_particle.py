"""
Lost Particle Debugging Example
================================

Demonstrates how YAMC handles "lost" particles -- particles that cross
a surface into a region not covered by any cell (a geometry gap).

Instead of crashing, YAMC:
  1. Prints a detailed diagnostic for each lost particle
  2. Kills the particle and continues the simulation
  3. Collects all lost particles in model.lost_particles
  4. Aborts if more than settings.max_lost_particles are lost

This example intentionally creates a geometry with a gap to show
the diagnostics in action.
"""

import yamc

# ---------------------------------------------------------------------------
# 1. Create geometry WITH a gap
# ---------------------------------------------------------------------------
# Inner sphere (radius=5): transmission boundary -- particles will cross this
# Outer sphere (radius=10): vacuum boundary -- but there's NO cell between radius=5 and radius=10
inner_sphere = yamc.Sphere(x0=0, y0=0, z0=0, radius=5.0)
outer_sphere = yamc.Sphere(x0=0, y0=0, z0=0, radius=10.0,
                         boundary="vacuum")

# Only one cell: inside the inner sphere (void, no material)
# The region between radius=5 and radius=10 is NOT covered by any cell -- this is the gap.
inner_region = inner_sphere.below
inner_cell = yamc.Cell(name="inner_void", region=inner_region)

geometry = yamc.Geometry(cells=[inner_cell])

# ---------------------------------------------------------------------------
# 2. Source: monodirectional in +z, so all particles hit the gap
# ---------------------------------------------------------------------------
source = yamc.NeutronSource(
    position=(0.0, 0.0, 0.0),
    energy=14.06e6,  # 14 MeV
    direction=yamc.sources.Monodirectional([0.0, 0.0, 1.0]))

# max_lost_particles=10 (default) -- up to 10 lost particles are warnings,
# more than 10 would abort the run.
model = yamc.Model(geometry=geometry, source=source, max_lost_particles=10)

# ---------------------------------------------------------------------------
# 3. Run -- diagnostics are printed to stderr as particles get lost
# ---------------------------------------------------------------------------
print("Running simulation with geometry gap...")
print("(Lost particle diagnostics will appear on stderr)\n")
try:
    model.simulate_transport(total_particles=50)
except BaseException as e:
    print(f"\nSimulation aborted as expected: {type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# 4. Inspect lost particles programmatically
# ---------------------------------------------------------------------------
print(f"\nTotal lost particles: {len(model.lost_particles)}")

for i, lp in enumerate(model.lost_particles):
    print(f"\n--- Lost particle {i + 1} ---")
    print(f"  Type:      {lp.particle_type}")
    print(f"  Position:  {lp.position}")
    print(f"  Direction: {lp.direction}")
    print(f"  Energy:    {lp.energy:.6e} eV")
    print(f"  Last cell: {lp.last_cell_name} (ID: {lp.last_cell_id})")
    print(f"  Surface:   ID {lp.surface_id}")

# ---------------------------------------------------------------------------
# 5. Use the replay source to test a geometry fix
# ---------------------------------------------------------------------------
if model.lost_particles:
    lp = model.lost_particles[0]
    print("\n\nTo replay the first lost particle after fixing the geometry:")
    print("  source = yamc.NeutronSource(")
    print(f"      position={tuple(lp.position)},")
    print(f"      energy={lp.energy},")
    print(f"      direction=yamc.sources.Monodirectional({list(lp.direction)}),")
    print("  )")

# ---------------------------------------------------------------------------
# 6. Fix the geometry and verify
# ---------------------------------------------------------------------------
print("\n\n--- Now fixing the geometry by adding the missing cell ---")

# Add the outer annular cell to cover the gap
outer_region = inner_sphere.above & outer_sphere.below
outer_cell = yamc.Cell(name="outer_void", region=outer_region)

fixed_geometry = yamc.Geometry(cells=[inner_cell, outer_cell])

# Replay with the same source
fixed_model = yamc.Model(geometry=fixed_geometry, source=source, max_lost_particles=10)
fixed_model.simulate_transport(total_particles=50)

print(f"\nLost particles after fix: {len(fixed_model.lost_particles)}")
assert len(fixed_model.lost_particles) == 0, "No particles should be lost now!"
print("Geometry is fixed -- no more lost particles.")
