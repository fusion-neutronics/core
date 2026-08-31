"""Photon-source lithium sphere -- proof that a PHOTON source transports
in the browser (not just neutrons).

A 1 MeV isotropic photon point source sits at the centre of a natural-
lithium sphere. We tally the photon flux and the photon heating in the
sphere, run it natively to seed the numbers, then export a self-contained
``Model.to_html()`` page the recipient can re-run in-browser (wasm) with
no Python.

Note there is **no** ``transport_secondary_photons`` flag here: that flag
only controls secondary gammas produced by *neutron* interactions. A
photon source transports photons on its own, and the photon cross
sections it needs are loaded automatically (native, offline embed, and
in-browser fetch all key off ``Model.has_photons()``, not the flag).

Run from the repo root:

    cd ~/yamc-org/yamc && python examples/python/sphere_li_photon_source.py

Outputs (open either in a browser and click Simulate):
- ``sphere_li_photon_source.html``         -- online (fetches Li photon
  data from yamc-data.xsplot.com on Simulate)
- ``sphere_li_photon_source_offline.html`` -- offline (Li neutron+photon
  data embedded; ~a few MB, runs with no network)
"""

import yamc

# Pull cross-section data from the R2 mirror; first run caches under
# ~/.cache/yamc/. `read_nuclear_data('endf-b8.1')` loads both the neutron
# (per-nuclide) and photon (per-element) data for each material.
yamc.cross_section_data = "endf-b8.1"

# --- Material: natural lithium ------------------------------------------
# `{"Li": 1.0}` expands to natural Li6 + Li7 for neutron data; photon data
# is per element ("Li"). Lithium-metal density.
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

# --- Source: 1 MeV isotropic PHOTONS at the centre ----------------------
# PhotonSource has no default energy (unlike the 14 MeV NeutronSource), so
# it must be given explicitly.
source = yamc.PhotonSource(
    position=(0.0, 0.0, 0.0),
    energy=1.0e6,  # 1 MeV mono-energetic photons
)

# --- Tallies: photon flux + photon heating in the sphere ----------------
flux_tally = yamc.Tally(
    cells=sphere_cell,
    particle="photon",
    scores=["flux"],
    name="photon flux",
)
heating_tally = yamc.Tally(
    cells=sphere_cell,
    particle="photon",
    scores=["heating"],
    estimator="collision",  # required for the heating score
    name="photon heating",
)

# --- Model --------------------------------------------------------------
# No transport_secondary_photons flag: a photon source needs no opt-in to
# transport its own photons (see module docstring). A modest particle
# count keeps the in-browser re-run quick.
total_particles = 20000
model = yamc.Model(
    geometry=geometry,
    tallies=[flux_tally, heating_tally],
    source=source,
)

# --- Native run to seed the exported page -------------------------------
print(f"Running {total_particles} photons natively...")
results = model.simulate_transport(total_particles=total_particles, seed=42)
flux = results[flux_tally].mean[0]
heating = results[heating_tally].mean[0]
print(f"  photon flux    = {flux:.6e} cm/source-photon")
print(f"  photon heating = {heating:.6e} eV/source-photon")

result_text = (
    "1 MeV photon source in a 10 cm natural-Li sphere\n"
    f"photon flux    = {flux:.6e} cm/source-photon\n"
    f"photon heating = {heating:.6e} eV/source-photon"
)

# --- Browser-runnable exports -------------------------------------------
model.to_html(
    "sphere_li_photon_source.html",
    embed_cross_sections=False,  # online: fetch Li data on Simulate
    initial_result_text=result_text,
)
print("Wrote sphere_li_photon_source.html (online)")

offline = model.to_html(
    "sphere_li_photon_source_offline.html",
    embed_cross_sections=True,  # offline: embed Li neutron + photon data
    initial_result_text=result_text,
)
print(
    f"Wrote sphere_li_photon_source_offline.html (offline, "
    f"{offline.stat().st_size / 1e6:.1f} MB)"
)
