"""Simplest physics-relevant example: 14 MeV neutrons into a Li-6 sphere,
tally the (n,t) tritium-production rate inside the sphere.

The reaction Li6(n,t)He4 (MT=105) is the breeding reaction that makes
this geometry interesting for fusion -- a tritium-breeding-ratio probe
boiled down to one cell + one tally.

Run from the repo root:

    cd ~/yamc-org/yamc && python examples/python/sphere_li6_tritium.py

Outputs:
- ``sphere_li6_tritium.html`` -- interactive geometry + source plot
  (the same kind the existing examples produce). NOTE: this is *not*
  a "click to simulate" page; the simulation runs in Python here. The
  browser-side simulate button needs more plumbing (wasm-bindgen
  entry points, OPFS-backed Storage impl, web worker, ...) that's
  still in progress.
- Console output with the tally result.
"""

import yamc

# Pull cross-section data from the R2 mirror (the URL switch from PR #168).
# First run downloads + caches per-nuclide tarballs under ~/.cache/yamc/.
yamc.cross_section_data = "endf-b8.1"

# --- Material -------------------------------------------------------------
# Pure Li-6 at natural-lithium-ish density. The composition is intentionally
# just Li6 (not natural Li) so every (n,t) score comes from one isotope and
# the result is easy to reason about.
li6 = yamc.Material(
    composition={"Li6": 1.0},
    density=0.46,  # g/cc, close to natural lithium
    temperature=294,
    name="li6",
)

# --- Geometry: single sphere ---------------------------------------------
outer = yamc.Sphere(radius=10.0, boundary="vacuum")
sphere_cell = yamc.Cell(region=outer.below, material=li6, name="li6_sphere")
geometry = yamc.Geometry(cells=[sphere_cell])

# --- Source: 14 MeV point at the center ----------------------------------
source = yamc.NeutronSource(
    energy=14.06e6,  # 14.06 MeV mono-energetic
    position=(0.0, 0.0, 0.0),
    strength=1.0,
)

# --- Tally: tritium production in the sphere -----------------------------
# Score MT=105 is the ENDF identifier for (n,t). Cell-bound so the tally
# integrates over the whole sphere.
tritium_tally = yamc.Tally(
    cells=sphere_cell,
    scores=[105],
    name="tritium_production",
)

# --- Model: few batches, few particles -----------------------------------
total_particles = 1000
model = yamc.Model(
    geometry=geometry,
    tallies=[tritium_tally],
    source=source,
)

# Interactive HTML -- geometry + source position, pan/zoom in browser.
# (Note: no "simulate" button on this HTML yet -- see module docstring.)
model.plot().save("sphere_li6_tritium.html")
print("Wrote sphere_li6_tritium.html")

# Run the simulation natively.
print(f"\nRunning {total_particles} particles...")
results = model.simulate_transport(total_particles=total_particles, seed=42)

# Print the tally result. `results[tally]` indexes into the per-tally
# TallyResult; `.mean` / `.standard_deviation` are nested lists shaped
# `[score][...bins]`. One cell + one score + no bins → `[[scalar]]`.
def _scalar(x):
    while isinstance(x, list):
        x = x[0]
    return float(x)

score = results[tritium_tally]
print("\nTritium production (Li6(n,t), MT=105):")
print(f"  mean: {_scalar(score.mean):.4e} reactions per source neutron")
print(f"  std:  {_scalar(score.standard_deviation):.4e}")
