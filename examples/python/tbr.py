"""
Tritium Breeding Ratio (TBR) example with per-nuclide breakdown.

Compares total H3-production against per-nuclide contributions from Li6, Li7,
and Be9. The sum of per-nuclide values should match the total within statistical
uncertainty.
"""

import time

import yamc

# --- Geometry: point source inside a Li/Be sphere ---
sphere1 = yamc.Sphere(x0=0, y0=0, z0=0, radius=1.0)
sphere2 = yamc.Sphere(x0=0, y0=0, z0=0, radius=200.0,
                    boundary='vacuum')
region1 = sphere1.below
region2 = sphere1.above & sphere2.below

# --- Material: Li4SiO4 surrogate (Li6 + Li7 + Be9) ---
material1 = yamc.Material(
    composition={
        "Li6": 0.07 / 2,
        "Li7": 0.93 / 2,
        "Be9": 0.5,
    },
    density=2.0,
    temperature=294)
material1.read_nuclear_data({
    "Be9": "tests/Be9.arrow",
    "Li6": "tests/Li6.arrow",
    "Li7": "tests/Li7.arrow",
})

cell1 = yamc.Cell(name="inner_sphere", region=region1)
cell2 = yamc.Cell(name="outer_annular", region=region2,
                material=material1)
geometry = yamc.Geometry([cell1, cell2])

source = yamc.NeutronSource(
    energy=yamc.sources.fusion_neutron_spectrum(20000.0),
    position=(0, 0, 0))

# --- Tally 1: Total H3-production (no nuclide breakdown) ---
tally_total = yamc.Tally(name="tbr_total", cells=cell2, scores=['H3-production'])

# --- Tally 2: Per-nuclide H3-production ---
tally_nuclides = yamc.Tally(
    name="tbr_per_nuclide",
    cells=cell2,
    scores=['H3-production'],
    nuclides=['Li6', 'Li7', 'Be9'],
)

tallies = [tally_total, tally_nuclides]
model = yamc.Model(geometry=geometry, tallies=tallies, source=source)

t0 = time.time()
results = model.simulate_transport(total_particles=250000, seed=1)
elapsed = time.time() - t0

# --- Print results ---
print(f"\n{'='*50}")
print(f"  TBR Results  ({elapsed:.2f}s)")
print(f"{'='*50}")

# Total H3-production
total_mean = results[tally_total].mean
if hasattr(total_mean, 'flatten'):
    total_mean = total_mean.flatten()
print(f"\n  Total H3-production: {total_mean[0]:.6e}")

# Per-nuclide breakdown
nuc_result = results[tally_nuclides]
nuc_mean = nuc_result.mean
if hasattr(nuc_mean, 'flatten'):
    nuc_mean = nuc_mean.flatten()

nuc_std = nuc_result.standard_deviation
if hasattr(nuc_std, 'flatten'):
    nuc_std = nuc_std.flatten()

nuclide_names = ['Li6', 'Li7', 'Be9']
print("\n  Per-nuclide H3-production breakdown:")
per_nuc_sum = 0.0
for i, nuc in enumerate(nuclide_names):
    val = float(nuc_mean[i])
    std = float(nuc_std[i])
    per_nuc_sum += val
    print(f"    {nuc:>4s}: {val:.6e} +/- {std:.6e}")

print(f"\n  Sum of per-nuclide:  {per_nuc_sum:.6e}")
print(f"  Total (from tally):  {float(total_mean[0]):.6e}")
diff = abs(per_nuc_sum - float(total_mean[0]))
print(f"  Absolute difference: {diff:.6e}")
if float(total_mean[0]) > 0:
    print(f"  Relative difference: {diff / float(total_mean[0]):.4e}")
