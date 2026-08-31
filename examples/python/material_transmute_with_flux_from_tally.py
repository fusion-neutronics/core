"""Transmutation using multigroup flux extracted from a transport tally.

Full workflow: run transport with an energy-binned flux tally, extract the
multigroup flux spectrum, then use material.transmute() for standalone transmutation
with a pulsed irradiation schedule.

Usage:
    python examples/python/material_transmute_with_flux_from_tally.py
"""

import math
import time
import numpy as np
import yamc

nuc_data = "/home/jon/nuclear_data/endf-b8.0-hdf5/neutron"

# --- Geometry: concentric spheres ---
def shell_volume(r_inner, r_outer):
    return 4.0 / 3.0 * math.pi * (r_outer**3 - r_inner**3)

s1 = yamc.Sphere(x0=0, y0=0, z0=0, radius=5.0)
s2 = yamc.Sphere(x0=0, y0=0, z0=0, radius=10.0, boundary="vacuum")

# Iron shell material
iron = yamc.Material(
    composition={
        "Fe54": 0.05845,
        "Fe56": 0.91754,
        "Fe57": 0.02119,
        "Fe58": 0.00282,
    },
    density=7.87,
    name="iron_shell",
    temperature=294,
    volume=shell_volume(0, 5.0))
iron.read_nuclear_data({
    "Fe54": f"{nuc_data}/Fe54.h5",
    "Fe55": f"{nuc_data}/Fe55.h5",
    "Fe56": f"{nuc_data}/Fe56.h5",
    "Fe57": f"{nuc_data}/Fe57.h5",
    "Fe58": f"{nuc_data}/Fe58.h5",
})

# Cells
cell_iron = yamc.Cell(region=s1.below, material=iron)
cell_void = yamc.Cell(region=s1.above & s2.below)

geometry = yamc.Geometry(cells=[cell_iron, cell_void])

# --- Tally: flux with VITAMIN-J-175 energy bins ---
flux_tally = yamc.Tally(id=1, scores=["flux"], energy_group_structure="VITAMIN-J-175", cells=cell_iron)

# --- Source and settings ---
source = yamc.NeutronSource()  # 14.06 MeV DT point source at origin

model = yamc.Model(geometry=geometry, tallies=[flux_tally], source=source)

# --- Run transport ---
print("Running transport to extract multigroup flux spectrum...")
t0 = time.time()
sim_results = model.simulate_transport(total_particles=5000)
t_transport = time.time() - t0
print(f"Transport complete in {t_transport:.2f}s\n")

# --- Extract multigroup flux from tally ---
flux_per_source = np.array(sim_results[flux_tally].mean)  # flux per source particle

# Scale to absolute flux: flux [n/cm^2/s] = flux_per_source * source_strength / volume
source_strength = 1e14  # neutrons/s (typical DT source)
multigroup_flux = flux_per_source * source_strength

print(f"Extracted {len(multigroup_flux)} energy group fluxes")
print(f"  Total flux: {sum(multigroup_flux):.4e} n/cm^2/s")
print(f"  Peak group flux: {max(multigroup_flux):.4e} n/cm^2/s\n")

# --- Transmute using extracted flux spectrum ---
# The spectrum rides on the pulses: a NeutronSource whose energy is a Histogram
# over the VITAMIN-J-175 groups (it normalizes the shape, so the tally flux per
# group can be passed directly), and the pulse rate is the total flux magnitude
# [n/cm^2/s]. Cooldown is decay-only. Schedule: 5 hours irradiation + 1h cooling.
spectrum = yamc.NeutronSource(
    energy=yamc.sources.Histogram("VITAMIN-J-175", multigroup_flux.tolist())
)
total_flux = float(sum(multigroup_flux))  # n/cm^2/s
n_irradiation = 5
schedule = yamc.PulseSchedule(
    [yamc.Pulse(rate=total_flux, duration=(1, "h"), source=spectrum) for _ in range(n_irradiation)]
    + [yamc.Cooldown(duration=(1, "h"))]
)

print("Running standalone transmutation with tally-derived flux...")
print("  Schedule: 5h irradiation + 1h cooling")
t0 = time.time()

_CHAIN = "tests/transmutation-endf-b8.1-sfr.arrow"
yamc.transmutation_decay_data = _CHAIN
yamc.transmutation_reactions = _CHAIN
yamc.transmutation_fission_yields = _CHAIN
results = iron.transmute(schedule=schedule)
step_materials = results.step_materials(iron.id or 0)

t_transmute = time.time() - t0
print(f"Transmutation complete in {t_transmute:.2f}s\n")

# --- Print results ---
print(f"Results: {len(step_materials)} timesteps\n")
for i, mat in enumerate(step_materials):
    t_hours = i + 1  # each step is 1 hour
    phase = "irradiation" if i < n_irradiation else "cooling"
    print(f"Step {i+1} (t={t_hours:.0f}h, {phase}):")
    nuclides = sorted(mat.nuclides, key=lambda x: -x[1])
    for name, density in nuclides[:8]:
        print(f"  {name:8s}: {density:.6e} atoms/barn-cm")
    if len(nuclides) > 8:
        print(f"  ... and {len(nuclides) - 8} more nuclides")
    print()
