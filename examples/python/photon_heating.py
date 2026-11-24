"""
Photon heating example: 1 MeV photon source in an iron cube.

Demonstrates:
- Photon source (PhotonSource)
- Cube geometry built from 6 planes
- Photon-specific scores (coherent-scatter, incoherent-scatter, etc.)
- particle kwarg on Tally to restrict tallies to photons
- Flux tally (works for any particle type)
"""
import yamc

# --- Material: natural iron ---
material = yamc.Material(
    composition={"Fe56": 1.0},
    density=7.874,
    temperature=294)
material.read_nuclear_data({"Fe56": "tests/Fe56.arrow"}, photon_data={"Fe": "tests/Fe.arrow"})

# --- Geometry: cube from 6 planes ---
xmin = yamc.Plane(axis="x", offset=-50.0, boundary="vacuum")
xmax = yamc.Plane(axis="x", offset=50.0, boundary="vacuum")
ymin = yamc.Plane(axis="y", offset=-50.0, boundary="vacuum")
ymax = yamc.Plane(axis="y", offset=50.0, boundary="vacuum")
zmin = yamc.Plane(axis="z", offset=-50.0, boundary="vacuum")
zmax = yamc.Plane(axis="z", offset=50.0, boundary="vacuum")

cube_region = xmin.above & xmax.below & ymin.above & ymax.below & zmin.above & zmax.below

cell = yamc.Cell(name="iron_cube", region=cube_region, material=material)
geometry = yamc.Geometry([cell])

# --- Source: 1 MeV photons at center, isotropic ---
source = yamc.PhotonSource(
    position=(0, 0, 0),
    energy=1e6)

# --- Tallies ---
tally_flux = yamc.Tally(name="photon_flux", cells=cell, particle="photon", scores=["flux"])
tally_xs = yamc.Tally(
    name="photon_xs",
    cells=cell,
    particle="photon",
    scores=[
        "coherent-scatter",
        "incoherent-scatter",
        "photoelectric",
        "pair-production",
    ],
)

tallies = [tally_flux, tally_xs]

# --- Run ---
model = yamc.Model(geometry=geometry, tallies=tallies, source=source)
results = model.simulate_transport(total_particles=100000, seed=42)

# --- Results ---
print("=" * 60)
print("Photon Heating Example: 1 MeV photons in Fe cube")
print("=" * 60)

flux_result = results[tally_flux]
flux_mean = flux_result.mean[0]
flux_std = flux_result.standard_deviation[0]
print("\nPhoton flux:")
print(f"  Mean:      {flux_mean:.6e} cm/source")
print(f"  Std Dev:   {flux_std:.6e}")
if flux_mean > 0:
    print(f"  Rel Error: {flux_std / flux_mean:.4f}")

score_names = [
    "Coherent (MT 502)",
    "Incoherent (MT 504)",
    "Photoelectric (MT 522)",
    "Pair production (MT 516)",
]
xs_result = results[tally_xs]
means = xs_result.mean
std_devs = xs_result.standard_deviation

print("\nPhoton interaction scores:")
for i, name in enumerate(score_names):
    m = means[i]
    s = std_devs[i]
    rel = s / m if m > 0 else 0.0
    print(f"  {name}:")
    print(f"    Mean:      {m:.6e}")
    print(f"    Std Dev:   {s:.6e}")
    print(f"    Rel Error: {rel:.4f}")

print("=" * 60)
