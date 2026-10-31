"""Coupled neutron->photon lithium sphere -- proof that coupled transport
runs in the browser.

A 14 MeV isotropic neutron point source sits at the centre of a natural-
lithium sphere with ``transport_secondary_photons=True``. Neutron interactions
produce secondary gammas, which are then transported; we tally the
tritium-production rate (the Li6(n,t) breeding reaction, MT=105) and the
photon heating from those secondary gammas. We run it natively to seed
the numbers, then export a self-contained ``Model.to_html()`` page the
recipient can re-run in-browser (wasm) with no Python.

Run from the repo root:

    cd ~/yamc-org/yamc && python examples/python/sphere_li_coupled_photon.py

Outputs (open either in a browser and click Simulate):
- ``sphere_li_coupled_photon.html``         -- online (fetches Li neutron
  + photon data from yamc-data.xsplot.com on Simulate)
- ``sphere_li_coupled_photon_offline.html`` -- offline (data embedded;
  ~a few MB, runs with no network)
"""

import yamc

# R2 mirror; first run caches under ~/.cache/yamc/. Loads both neutron
# (per-nuclide) and photon (per-element) data for the material.
yamc.cross_section_data = "endf-b8.1"

# --- Material: natural lithium ------------------------------------------
lithium = yamc.Material(
    composition={"Li": 1.0},
    density=0.534,  # g/cc, lithium metal
    temperature=294,
    name="lithium",
)
lithium.read_nuclear_data("endf-b8.1")

# --- Geometry: single solid sphere --------------------------------------
outer = yamc.Sphere(radius=10.0, boundary="vacuum")
sphere_cell = yamc.Cell(region=outer.below, material=lithium, name="lithium_sphere")
geometry = yamc.Geometry(cells=[sphere_cell])

# --- Source: 14 MeV isotropic NEUTRONS at the centre --------------------
source = yamc.NeutronSource(
    position=(0.0, 0.0, 0.0),
    energy=14.06e6,  # 14.06 MeV D-T neutrons
)

# --- Tallies ------------------------------------------------------------
# Tritium production: MT=105 is Li6(n,t)He4, the fusion breeding reaction.
tritium_tally = yamc.Tally(
    cells=sphere_cell,
    scores=[105],
    name="tritium production",
)
# Photon heating from the secondary gammas (photon-only, collision est.).
photon_heating_tally = yamc.Tally(
    cells=sphere_cell,
    particle="photon",
    scores=["heating"],
    estimator="collision",
    name="photon heating",
)

# --- Model: coupled neutron -> photon transport -------------------------
# transport_secondary_photons=True turns on secondary-gamma production *and* makes the
# browser load the per-element Li photon data those gammas need.
total_particles = 20000
model = yamc.Model(
    geometry=geometry,
    tallies=[tritium_tally, photon_heating_tally],
    source=source,
    transport_secondary_photons=True,
)

# --- Native run to seed the exported page -------------------------------
print(f"Running {total_particles} neutrons natively (coupled n->gamma)...")
results = model.simulate_transport(total_particles=total_particles, seed=42)
tritium = results[tritium_tally].mean[0]
heating = results[photon_heating_tally].mean[0]
print(f"  tritium production = {tritium:.6e} reactions/source-neutron")
print(f"  photon heating     = {heating:.6e} eV/source-neutron")

result_text = (
    "14 MeV neutron source in a 10 cm natural-Li sphere "
    "(coupled neutron->photon)\n"
    f"tritium production = {tritium:.6e} reactions/source-neutron\n"
    f"photon heating     = {heating:.6e} eV/source-neutron"
)

# --- Browser-runnable exports -------------------------------------------
model.to_html(
    "sphere_li_coupled_photon.html",
    embed_cross_sections=False,  # online: fetch data on Simulate
    initial_result_text=result_text,
)
print("Wrote sphere_li_coupled_photon.html (online)")

offline = model.to_html(
    "sphere_li_coupled_photon_offline.html",
    embed_cross_sections=True,  # offline: embed Li neutron + photon data
    initial_result_text=result_text,
)
print(
    f"Wrote sphere_li_coupled_photon_offline.html (offline, "
    f"{offline.stat().st_size / 1e6:.1f} MB)"
)
