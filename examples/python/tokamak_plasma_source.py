"""Parametric tokamak plasma source.

Builds a neutron source from plasma parameters (shape, confinement mode, ion
density and temperature profiles) rather than from a hand-placed ring, prints
the profiles it is built from, and shows where the neutrons come from.

No nuclear data is needed: this only samples the source. Drop the returned
list into ``yamc.Model(source=...)`` to transport it.

The arguments mirror the openmc-plasma-source package's ``tokamak_source``,
so the same numbers describe the same plasma in both codes.
"""
import math

import yamc

# A EU-DEMO-like plasma. Lengths in cm, densities in m^-3, temperatures in eV.
PLASMA = dict(
    major_radius=906.0,
    minor_radius=292.258,
    elongation=1.557,
    triangularity=0.270,
    mode="H",
    ion_density_centre=1.09e20,
    ion_density_peaking_factor=1,
    ion_density_pedestal=1.09e20,
    ion_density_separatrix=3e19,
    ion_temperature_centre=45.9e3,
    ion_temperature_peaking_factor=8.06,
    ion_temperature_beta=6.0,
    ion_temperature_pedestal=6.09e3,
    ion_temperature_separatrix=0.1e3,
    pedestal_radius=0.8 * 292.258,
    shafranov_factor=0.44789,
    fuel={"D": 0.5, "T": 0.5},
)

# ---------------------------------------------------------------------------
# The profiles the source is built from. Both depend only on the minor radius,
# and both accept a list of radii so they plot directly.
# ---------------------------------------------------------------------------
radii = [i * PLASMA["minor_radius"] / 10 for i in range(11)]

densities = yamc.sources.tokamak_ion_density(
    mode=PLASMA["mode"],
    ion_density_centre=PLASMA["ion_density_centre"],
    ion_density_peaking_factor=PLASMA["ion_density_peaking_factor"],
    ion_density_pedestal=PLASMA["ion_density_pedestal"],
    minor_radius=PLASMA["minor_radius"],
    pedestal_radius=PLASMA["pedestal_radius"],
    ion_density_separatrix=PLASMA["ion_density_separatrix"],
    r=radii,
)

temperatures = yamc.sources.tokamak_ion_temperature(
    r=radii,
    mode=PLASMA["mode"],
    pedestal_radius=PLASMA["pedestal_radius"],
    ion_temperature_pedestal=PLASMA["ion_temperature_pedestal"],
    ion_temperature_centre=PLASMA["ion_temperature_centre"],
    ion_temperature_beta=PLASMA["ion_temperature_beta"],
    ion_temperature_peaking_factor=PLASMA["ion_temperature_peaking_factor"],
    ion_temperature_separatrix=PLASMA["ion_temperature_separatrix"],
    minor_radius=PLASMA["minor_radius"],
)

print("minor radius (cm)   ion density (m^-3)   ion temperature (keV)")
for radius, density, temperature in zip(radii, densities, temperatures):
    print(f"{radius:15.1f}   {density:18.3e}   {temperature / 1e3:20.2f}")

# ---------------------------------------------------------------------------
# The source itself: one ring source per (mesh voxel, reaction).
# ---------------------------------------------------------------------------
sources = yamc.sources.tokamak_source(**PLASMA, mesh_resolution=(50, 50))
print(f"\n{len(sources)} ring sources, strengths summing to "
      f"{sum(source.strength for source in sources):.6f}")

# 50:50 D-T fuel makes mostly 14 MeV D-T neutrons plus a small 2.5 MeV D-D
# component; each source carries the Ballabio spectrum for its own voxel.
dd_yield = sum(source.strength for source in sources if source.energy.mean < 5e6)
print(f"D-D share of the yield: {dd_yield:.3%}")

# Where the neutrons are born, weighted by source strength.
mean_radius = 0.0
mean_energy = 0.0
for source in sources:
    positions, _ = source.sample_n(1)
    x, y, _ = positions[0]
    mean_radius += source.strength * math.hypot(x, y)
    mean_energy += source.strength * source.energy.mean
print(f"strength-weighted mean birth major radius: {mean_radius:.1f} cm")
print(f"strength-weighted mean birth energy: {mean_energy / 1e6:.3f} MeV")

# A wedge model only needs the matching wedge of plasma.
wedge = yamc.sources.tokamak_source(
    **PLASMA, mesh_resolution=(50, 50), start_angle=0.0, rotation_angle=math.pi / 4
)
print(f"\n45 degree sector: {len(wedge)} sources over the same poloidal shape")

# Hand the list straight to a model:
#     model = yamc.Model(geometry=geometry, source=sources, tallies=tallies)
