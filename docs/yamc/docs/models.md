# Models and simulating

## Model

### Creating a Model

The ``Model`` brings together geometry, sources, tallies, and simulation settings.

<!-- doctest: skip -->
```python
model = yamc.Model(
    geometry=geometry,
    tallies=[tally1, tally2],
    source=source,
)
results = model.simulate_transport()
```

The particle count and random seed are per-run arguments to ``simulate_transport`` (and ``simulate_transmutation``), not ``Model`` constructor kwargs, so one model can be re-run at different counts/seeds without rebuilding it. ``seed`` defaults to 1 and ``total_particles`` to ``None``, no cap, so a run needs it, ``max_runtime`` or convergence targets: one with no stop condition raises ``ValueError``.

Both CSG ``Geometry`` and ``MeshGeometry`` are accepted.

Before a run, inspect which data the assembled model needs:

```python
print(model.required_nuclides())
print(model.required_temperatures())
print(model.required_elements())
```

These are requirements, not a download. Temperatures map each nuclide to the
requested temperature strings, or `None` if a material loads every available
temperature. Elements are needed for photon transport and are empty when the
model will transport no photons.

### Photon transport

Enable coupled neutron-photon transport with ``transport_secondary_photons=True``.
A coupled run reads both ``yamc.neutron_cross_section_data`` and
``yamc.photon_cross_section_data``; set the photon one to a library with
photoatomic data. The model loads the required element data automatically. No explicit material data read is
needed (see [Loading nuclear data](materials.md#loading-nuclear-data)).
A primary ``PhotonSource`` transports its own photons without this coupling flag.

<!-- doctest: skip -->
```python
yamc.neutron_cross_section_data = "endf-b8.1"
yamc.photon_cross_section_data = "endf-b8.1"

model = yamc.Model(
    geometry=geometry,
    source=source,
    transport_secondary_photons=True,
    photon_cutoff_energy=1000.0,     # eV (default)
    electron_treatment="ttb",         # thick-target bremsstrahlung (default); or "local"
)
```

For D1S shutdown dose rate calculations, enable decay photon production. A D1S
run reads both cross-section settings, set above, and also the decay data and
the reactions network, which have no default
(see [Transmutation data](nuclear_data.md#transmutation-data)):

<!-- doctest: skip -->
```python
yamc.transmutation_decay_data = "endf-b8.1"
yamc.transmutation_reactions = "endf-b8.1"

model = yamc.Model(
    geometry=geometry,
    source=source,
    transport_secondary_photons=True,
    use_decay_photons=True,
)
```

### Simulation knobs

These ``Model`` constructor kwargs are changed less often. The defaults are usually fine; adjust them when the simulation aborts or when benchmarking the GPU path.

- ``free_gas_threshold`` (default ``400.0``): the threshold, in multiples of ``k_B·T``, above which elastic scattering treats the target nuclide as a stationary scatterer rather than sampling its thermal velocity from the free-gas Maxwell-Boltzmann distribution. The default of 400 kT is the standard convention; raising it widens the energy range where the thermal motion of the target is sampled (slightly slower); lowering it treats the target as stationary from a lower energy (slightly faster, slightly less accurate near the threshold).
- ``max_lost_particles`` (default ``10``): abort ``simulate_transport`` after this many particles have been lost (no valid cell found at a step, or no surface ahead of a particle in an unbounded cell). A lost particle indicates a geometry problem (overlapping cells, an uncovered region, a numerical-precision edge case); aborting early avoids running a long simulation against a broken model. See [Lost particles and geometry debugging](simulate_transport.md#lost-particles-and-geometry-debugging) for diagnosing them.
- ``gpu_max_steps_per_particle`` (default ``100000``): hard cap on transport steps per particle on the GPU path. CPU transport ignores it, since histories there end through absorption, leakage, and the lost-particle diagnostics; the GPU kernel has no concept of "lost" and needs an explicit cap to bound shader runtime. Setting it and then running on the CPU emits a warning, since the request has no effect there. The default is deliberately high so the GPU matches the CPU on strong scatterers (a ``1000`` cap cost ~10% of the integral flux on a pure-H2 sphere). A cap that binds is an error: if any history in a GPU launch is still transporting at the cap, ``simulate_transport`` raises ``ValueError`` naming the cap rather than returning under-counted tallies, so raise it for a deep-penetration GPU run, or run on the CPU. Lower it only to bound worst-case kernel time on a pathological geometry you expect to leak or absorb well within it.
- ``gpu_fission_bank`` (default ``True``): whether the GPU banks fission progeny on the device. Inert without fissile material and on the CPU. Turn it off to run a fissile model with a mesh tally on the GPU, which the dispatch otherwise refuses: the combination of per-source accumulation across fission generations and direct mesh scoring is not wired up. With it off, fission progeny are not banked on the device and the run takes the non-fissile loop, which is what lets the mesh path work.

<!-- doctest: skip -->
```python
model = yamc.Model(
    geometry=geometry,
    source=source,
    free_gas_threshold=400.0,        # default
    max_lost_particles=10,           # default
    gpu_max_steps_per_particle=100000,   # default; GPU path only
    gpu_fission_bank=True,           # default; GPU path only
)

model.gpu_fission_bank = False       # also settable after construction
```

### Tracking mode (Woodcock / delta tracking)

``tracking_mode`` selects the particle-transport algorithm. ``"surface"`` (default) is standard ray-tracing: at each step it computes the distance to the nearest cell boundary, advances the particle, and samples a collision against the current material's ``Σ_t``. Delta tracking instead samples a free flight against a *global majorant* ``Σ_maj`` (the per-energy max of ``Σ_t`` across every material in the model), moves the particle the full sampled distance ignoring cell boundaries, and runs a rejection loop at the sampled point that decides whether the event is real (``Σ_t / Σ_maj``) or virtual (no physics, continue). The speed-up comes from skipping the per-step ``closest_boundary`` calculation.

There are two delta-tracking modes. For nearly all models, use ``"hybrid"``:

<!-- doctest: skip -->
```python
model = yamc.Model(
    geometry=geometry,
    source=source,
    tracking_mode="hybrid",   # delta tracking, safe in voids
)
```

When to use which:

- ``"surface"``: the default, and always safe. It is a good general choice and the fastest option for typical cell-sparse geometries (few cells per mean free path), where delta tracking has few boundaries to skip.
- ``"hybrid"``: the recommended Woodcock mode. Delta tracking with an automatic per-cell fallback to surface tracking in cells where delta tracking would be inefficient (voids, large low-density regions such as a vacuum vessel or air). It keeps the boundary-skipping gain in dense regions and does not slow down in voids, so it is the safe way to get delta-tracking speed on a realistic fusion model. Use it whenever the geometry contains any void or low-density region.
- ``"woodcock"``: pure delta tracking with no fallback. Use it only for geometrically dense, void-free models (many cells per mean free path with similar ``Σ_t``, e.g. a finely diced or CAD-tessellated geometry). On such models it can be several times faster than surface tracking. Do not use it on a model with large voids: ``Σ_maj`` then far exceeds the local ``Σ_t``, the rejection loop spends most of its time on virtual collisions, and it becomes much slower than ``"surface"``. This is the case ``"hybrid"`` handles. When in doubt, use ``"hybrid"``.

All three give the same answer within statistics; they differ only in speed.

### Survival biasing (implicit capture)

``variance_reduction`` takes a list of technique objects; an empty list (the default) is fully analog transport. Techniques compose in the same list (currently survival biasing, below, and [weight windows](weight_windows.md)), and the list order carries no meaning.

``yamc.SurvivalBiasing`` enables the first technique, implicit capture: instead of killing the particle when absorption is sampled (analog capture), every collision multiplies the particle weight by the scattering probability ``Σ_s / Σ_t`` and the particle always scatters. Histories are never cut short by capture, so they keep contributing track length to deep tallies. Low-weight particles are culled by weight-cutoff Russian roulette: below ``weight_cutoff`` a particle survives with probability ``weight / weight_survive`` (continuing at ``weight_survive``) or is killed, which conserves weight in expectation. The parameters and defaults match the standard ``survival_biasing`` setting.

<!-- doctest: skip -->
```python
model = yamc.Model(
    geometry=geometry,
    source=source,
    variance_reduction=[
        yamc.SurvivalBiasing(
            weight_cutoff=0.25,    # default; roulette trigger
            weight_survive=1.0,    # default; weight of roulette survivors
        ),
    ],
)
```

Notes:

- Works in all three tracking modes; the means are unchanged within statistics, only the variance (and history length) differ.
- Judge the benefit by the per-tally FOM printed at the end of the run rather than by the particles/s line. Histories no longer end at capture, so raw throughput often drops (in strongly absorbing media roulette can instead shorten histories and raise it), while the error bars shrink faster than the throughput changes. On absorption-dominated problems a FOM gain of roughly 2x is typical, which means the same error bars in about half the CPU time.
- Most effective on absorption-dominated problems and deep tallies; for leakage-dominated problems it changes little.
- At most one ``SurvivalBiasing`` entry is allowed in the list (duplicates are rejected). ``model.variance_reduction`` returns copies: configure technique objects and assign a whole list rather than mutating entries in place.
- Composes with everything weight-aware: tallies, secondary-photon production, D1S decay photons, and transmutation reaction rates all score with the particle weight (photons inherit the parent neutron's pre-discount weight). ``compute='gpu'`` refuses ``variance_reduction`` (the kernel runs analog transport) rather than silently ignoring it.

### Weight windows

The second variance-reduction technique, weight windows (mesh-based splitting and roulette), targets deep-penetration and streaming tallies that survival biasing alone cannot reach. It has its own generation workflow and is documented separately in [Weight windows](weight_windows.md).

### Changing materials between runs

For feedback loops, such as a thermal solve that returns temperatures, a model
can change its materials between runs without being rebuilt. The geometry,
loaded nuclear data, tallies and weight windows are all kept.

<!-- doctest: skip -->
```python
model.set_material_temperature(coolant, 650.0)  # K
model.set_material_density(coolant, 0.98)       # in coolant.density_units
results = model.simulate_transport()
```

Only the model's copy of the material changes; ``coolant`` itself keeps its old
values. A density change reuses the microscopic cross sections, so it costs
far less than a new model. A temperature change loads the data at the new
temperature.

On a tet-meshed ``MeshGeometry`` the density can also vary tet by tet, as for a
liquid breeder whose density follows its temperature. Each factor scales every
macroscopic cross section in its tet, so no extra nuclear data is needed:

<!-- doctest: skip -->
```python
import numpy as np

in_breeder = np.array(geometry.tet_volume_names) == "breeder"
geometry.tet_density_factors = np.where(in_breeder, density_at_each_tet / breeder.density, 1.0)

# the model starts from the geometry's factors, as from copies of its materials
model = yamc.Model(geometry=geometry, source=source, tallies=tallies, tracking_mode="hybrid")
results = model.simulate_transport()

# then change the model's copy between runs, as set_material_density does
model.set_tet_density_factors(new_factors)  # one per tet, in the order of geometry.tetrahedra
model.set_tet_density_factors(None)         # back to the materials' own densities
```

Per-tet factors need ``tracking_mode="hybrid"`` or ``"woodcock"``. Heating and
reaction-rate scores follow the scaled density.
[Coupling to a heat solve](examples/thermal_coupling.md) runs this loop with
FEniCSx.

### Saving and loading results

A ``TransportResults`` object can be written to a single Arrow file and read back in any later session. The full per-bin statistics (mean, variance, history counts) and run provenance are preserved, so a loaded result behaves exactly like a fresh one for plotting, inspection, or [combining](#combining-results-from-separate-runs).

<!-- doctest: skip -->
```python
results = model.simulate_transport()
results.to_arrow("run1.arrow")                            # save

# Later, or in another session / on another machine:
loaded = yamc.TransportResults.from_arrow("run1.arrow")  # load
print(loaded["flux"].mean, loaded["flux"].relative_error)

# With no path, to_arrow() returns the file's bytes, and from_arrow takes
# them back: no file system needed, as in a browser or Pyodide session.
data = results.to_arrow()
loaded = yamc.TransportResults.from_arrow(data)
```

### Combining results from separate runs

``yamc.combine_results`` pools the results of independent runs of the same model into one result that is statistically identical to a single longer run. It merges the raw per-bin Welford state (mean, M2, history count) of every tally, the same exact operation the transport loop uses to combine per-thread partials. Means, error bars and figures of merit come out as if all histories had been one simulation (FOM uses the summed wall-clock time).

Every run needs its own seed. yamc derives per-particle random streams from the base seed, so two runs with the same seed re-trace the same histories; ``combine_results`` refuses them with an error rather than silently producing wrong error bars.

<!-- doctest: skip -->
```python
# Two runs of the same model, each with its own seed; particle counts can
# differ. Here a 100k run and a 200k run.
model_a = yamc.Model(geometry=geometry, source=source, tallies=tallies)
model_b = yamc.Model(geometry=geometry, source=source, tallies=tallies)

results_a = model_a.simulate_transport(total_particles=100_000, seed=1)
results_b = model_b.simulate_transport(total_particles=200_000, seed=2)

combined = yamc.combine_results(results_a, results_b)   # pooled, as if 300k histories
print(combined["flux"].relative_error)

# Works across sessions too: save each run with results.to_arrow(...) (see
# "Saving and loading results" above) and combine the Arrow files later.
r1 = yamc.TransportResults.from_arrow("run1.arrow")
r2 = yamc.TransportResults.from_arrow("run2.arrow")
combined = yamc.combine_results(r1, r2)
```

This also allows cluster runs without MPI: launch N jobs with seeds 1..N, save each with ``results.to_arrow(...)``, and combine the files afterwards. The combination is statistically exact.

The following are checked before combining, and any failure is a hard error:

- distinct seeds across all contributing runs;
- same model: geometry, materials, source, physics settings and per-nuclide nuclear-data libraries (a library mismatch is reported per nuclide, e.g. ``n:Co58: "endf-b8.1" vs "fendl-3.2d"``);
- same-name tallies must have identical configurations. A material filter's bins are matched by the material each id names, not by the id, so a model built again in the same process, whose materials get new ids, still combines;
- GPU-produced results are refused (the GPU path does not record the exact merge state).

Results saved by yamc 0.16.2 still load and combine with each other. They do not combine with runs from 0.16.3 if the model is a CSG model with materials (the fingerprint now labels each material by its place in the geometry, not its id), or if a tally both runs have filters on materials (0.16.2 did not record which material each id names).

Tallies present in only some runs are carried through unchanged with a ``UserWarning``. Their statistics come from the runs that scored them, and their FOM uses those runs' time only. ``results.runs`` lists the provenance (seed, history count, elapsed time, fingerprint, data libraries) of every contributing run.


### Plotting

``model.plot()`` renders the geometry with source points overlaid.

<!-- doctest: skip -->
```python
plot = model.plot(
    basis="xz",
    color_by="material",
    outline="cell",
    resolution=400000,
    n_samples=1000,  # number of source points to overlay
)
plot.to_html("model.html")
plot.to_png("model.png")
```

The same plot rendered live (the dots are sampled source points):

```python exec="1" html="1"
import html
import yamc

steel = yamc.Material(composition={"Fe": 1.0}, density=7.874, name="steel")
water = yamc.Material(composition={"H2O": 1.0}, density=1.0, name="water")
inner, outer = yamc.Cylinder(axis="z", radius=2.0), yamc.Cylinder(axis="z", radius=8.0)
bottom, top = yamc.Plane(axis="z", offset=-12.0), yamc.Plane(axis="z", offset=12.0)
box = (
    yamc.Plane(axis="x", offset=-15.0, boundary="vacuum").above
    & yamc.Plane(axis="x", offset=15.0, boundary="vacuum").below
    & yamc.Plane(axis="y", offset=-15.0, boundary="vacuum").above
    & yamc.Plane(axis="y", offset=15.0, boundary="vacuum").below
    & yamc.Plane(axis="z", offset=-15.0, boundary="vacuum").above
    & yamc.Plane(axis="z", offset=15.0, boundary="vacuum").below
)
pipe = yamc.Cell(region=inner.above & outer.below & bottom.above & top.below, material=steel, name="pipe")
coolant = yamc.Cell(region=inner.below & bottom.above & top.below, material=water, name="coolant")
void = yamc.Cell(region=box & ~(outer.below & bottom.above & top.below), name="void")
geometry = yamc.Geometry([pipe, coolant, void])
source = yamc.NeutronSource()
model = yamc.Model(geometry=geometry, source=source)

doc = model.plot(basis="xz", color_by="material", outline="cell", n_samples=400)._repr_html_()
print(f'<iframe srcdoc="{html.escape(doc, quote=True)}" width="100%" height="520" style="border:1px solid #ddd;border-radius:4px"></iframe>')
```


For manual matplotlib plotting:

<!-- doctest: skip -->
```python
slice_data = model.sample_slice(basis="xy", resolution=(400, 400))
```

### VTK-HDF model export

``model.to_vtkhdf`` writes the geometry and the sampled source together (as a VTK-HDF partitioned collection), so a single file shows both in [ParaView](https://www.paraview.org/).

<!-- doctest: skip -->
```python
model.to_vtkhdf("model.vtkhdf", resolution=10000, samples=1000)  # geometry + source
model.to_vtkhdf("source.vtkhdf", include=["source"], samples=1000)
```

The viewer below is rendered live when these docs are built: it writes ``model.vtkhdf``, reads it back, and shows the voxelized geometry (cut in half, coloured by material) together with the sampled source points (drag to rotate, scroll to zoom).

```python exec="1" html="1"
import html

import pyvista as pv
from vtkmodules.vtkIOHDF import vtkHDFReader

import yamc

pv.OFF_SCREEN = True
steel = yamc.Material(composition={"Fe": 1.0}, density=7.874, name="steel")
water = yamc.Material(composition={"H2O": 1.0}, density=1.0, name="water")
inner, outer = yamc.Cylinder(axis="z", radius=2.0), yamc.Cylinder(axis="z", radius=8.0)
bottom, top = yamc.Plane(axis="z", offset=-12.0), yamc.Plane(axis="z", offset=12.0)
box = (
    yamc.Plane(axis="x", offset=-15.0, boundary="vacuum").above
    & yamc.Plane(axis="x", offset=15.0, boundary="vacuum").below
    & yamc.Plane(axis="y", offset=-15.0, boundary="vacuum").above
    & yamc.Plane(axis="y", offset=15.0, boundary="vacuum").below
    & yamc.Plane(axis="z", offset=-15.0, boundary="vacuum").above
    & yamc.Plane(axis="z", offset=15.0, boundary="vacuum").below
)
pipe = yamc.Cell(region=inner.above & outer.below & bottom.above & top.below, material=steel, name="pipe")
coolant = yamc.Cell(region=inner.below & bottom.above & top.below, material=water, name="coolant")
void = yamc.Cell(region=box & ~(outer.below & bottom.above & top.below), name="void")
geometry = yamc.Geometry([pipe, coolant, void])
# Same point source at the origin as the model.plot viewer above.
source = yamc.NeutronSource()
model = yamc.Model(geometry=geometry, source=source)
model.to_vtkhdf("model.vtkhdf", resolution=300000, samples=2000)


def leaves(collection):
    out = []
    for i in range(collection.GetNumberOfPartitionedDataSets()):
        pds = collection.GetPartitionedDataSet(i)
        for j in range(pds.GetNumberOfPartitions()):
            part = pds.GetPartition(j)
            if part is not None:
                out.append(pv.wrap(part))
    return out


reader = vtkHDFReader()
reader.SetFileName("model.vtkhdf")
reader.Update()
geom = next(d for d in leaves(reader.GetOutput()) if "material_id" in d.cell_data)
pts = next(d for d in leaves(reader.GetOutput()) if d.n_points and "energy" in d.point_data)

plotter = pv.Plotter(window_size=(760, 560))
plotter.add_mesh(geom.threshold(-0.5, scalars="material_id").clip(normal="x"),
                 scalars="material_id", cmap="Set2", show_scalar_bar=False, opacity=0.85)
plotter.add_mesh(pts, color="orange", render_points_as_spheres=True, point_size=5)
plotter.add_axes()
plotter.camera_position = "iso"

doc = plotter.export_html(filename=None).read()
print(
    f'<iframe srcdoc="{html.escape(doc, quote=True)}" width="100%" height="580" '
    'style="border:1px solid #ddd;border-radius:4px"></iframe>'
)
```

### Save and load (JSON)

A model can be serialised to and from JSON. The JSON captures the specification: geometry, materials, source, tallies and run parameters. Nuclide cross-section data is not included, so a loaded model still needs to load nuclear data before ``simulate_transport`` (the same way a model built from scratch does).

<!-- doctest: skip -->
```python
# Write a pretty-printed JSON file (git-diff-friendly).
model.to_json("blanket_v3.json")

# Round-trip from disk.
model = yamc.Model.from_json("blanket_v3.json")

# Or without a file: with no path, to_json() returns the JSON as bytes,
# and from_json() takes bytes as well as a path.
spec = model.to_json()                # pretty=False for a single line
model2 = yamc.Model.from_json(spec)
```

Typical uses:

- Check models into git so reviewers can ``diff`` the spec without running anything.
- Build a model in a notebook, serialise it, and re-load it in an HPC job script. The particle count and seed are not part of the saved model, so pass a different ``total_particles`` or ``seed`` to ``simulate_transport`` when you run it there.
- Ship a model spec alongside a publication.

### Export to a runnable HTML page (``to_html``)

``model.to_html(path)`` writes a self-contained HTML file (with no path, ``to_html()`` returns the page as bytes) that opens in any browser and runs the simulation inside the page, using a WASM build of the transport engine. It can be used to share a working example with someone who does not have Python or yamc installed, to embed a runnable demo in a paper or course, or to lock down a model for educational use.

<!-- doctest: skip -->
```python
# Minimal: writes geometry / source / tallies; XS data fetched at
# runtime from yamc-data.xsplot.com (requires network), one temperature
# per nuclide: the byte ranges of the batches each material's
# temperature needs, not the whole file.
model.to_html("demo.html")

# Fully offline: embed every nuclide the current materials use.
# Recipient gets a portable file that runs with no network.
model.to_html("portable.html", embed_cross_sections=True)

# Embed a family of nuclides to anticipate the recipient swapping
# Li6 for Li7 without breaking the offline behaviour.
model.to_html("portable.html", embed_cross_sections=["Li"])

# Lock-down for teaching: the recipient can re-simulate but cannot edit
# geometry or tally definitions.
model.to_html(
    "locked.html",
    show_surfaces=False,
    show_materials=True,    # let students change density / composition
    show_tallies=False,
)

# Bake in a static snapshot of an interactive mesh-tally plot so the
# recipient sees the spatial distribution on first load.
plot = tally.plot(geometry=geometry, basis="xz")
model.to_html(
    "with_flux_map.html",
    embed_cross_sections=True,
    tally_plot_html=plot._repr_html_(),
)
```

Embedded XS data inflates the HTML size by roughly the raw byte count × 1.33 (base64), and embeds every temperature. ENDF/B-VIII.1 nuclides range from ~1.2 MB (Li6) to ~56 MB (Fe56) whole, so embedding everything in a heavy-element material can produce a multi-hundred-MB HTML; pick ``embed_cross_sections`` accordingly. The fetch-at-runtime path is lighter: for a material with a temperature set it fetches that temperature of each nuclide's reactions and energy grids (about 22 MB for Fe56 rather than 56), because the published data is written one Arrow batch per (MT, temperature) with a byte-range index in each nuclide's ``version.json``.

The ``show_*`` flags are presentation-only: the recipient can still inspect (or open devtools and edit) the embedded model JSON. Use them to declutter the UI, not as a security boundary.
