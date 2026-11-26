# Results and post-processing

### Tally results

``model.simulate_transport()`` returns a ``TransportResults`` indexed by tally:

<!-- doctest: skip -->
```python
results = model.simulate_transport()
result = results[tally]
print(result.mean)                # list of mean values
print(result.standard_deviation)  # list of standard deviations
print(result.relative_error)      # list of relative errors
```

### Units

Transport scores per source particle, so a raw result is in units such as
``eV/source-particle`` (heating), ``cm/source-particle`` (flux, a track length)
or ``particles/source-particle`` (a reaction or production count). ``result.units`` gives one string per score.
Set a source strength to get rates, then convert to whatever unit you report
in with ``to()``:

<!-- doctest: skip -->
```python
result = results.with_source_strength(1e20)[tally]   # neutrons per second
print(result.units)                  # e.g. ['eV/s', 'cm/s', 'particles/s']

heating = result.to("W/m3")          # one unit for every score ...
mixed = result.to(["W/cm3", "1/(cm2 s)", "reactions/(m3 s)"])  # ... or one per score
print(heating.mean, heating.units)
```

A per-volume unit divides each bin by its volume: the mesh element, or the
cell when the cell has a ``volume``. For a pulsed source use
``with_source_strength(1e18, per="pulse")`` and units such as ``"J/pulse"``.

| score | example units |
| --- | --- |
| heating | ``"W"``, ``"W/cm3"``, ``"W/m3"``, ``"MW/m3"``, ``"MeV/s"``, ``"J/pulse"`` |
| flux | ``"1/(cm2 s)"``, ``"n/cm2/s"``, ``"1/(m2 s)"`` |
| reactions (e.g. ``"H3-production"``) | ``"reactions/s"``, ``"reactions/(cm3 s)"``, ``"1/(m3 s)"`` |
| dose (flux with dose coefficients) | ``"Sv/h"``, ``"uSv/h"``, ``"mSv/h"`` |

Units take SI prefixes (``p n u m c k M G T``) on ``m``, ``s``, ``eV``,
``J``, ``W``, ``Sv`` and ``Gy``, and time as ``s``, ``min``, ``h``, ``d`` or
``y``. A unit that does not match the score, or that needs a source strength
the result does not have, raises a ``ValueError`` saying why.

The converted result is what ``to_xarray()``, ``to_dataset()`` and
``to_vtkhdf()`` export, each with its units recorded.

### Statistical reliability

Every ``TallyResult`` also carries reliability statistics. They are computed automatically from a small per-history aggregate, so the runtime cost is negligible. Printing a result shows all of them: the mean ± std, the figure of merit, the higher-moment metrics, and a PASS/FAIL verdict:

<!-- doctest: skip -->
```python
result = results["flux"]
print(result)
# TallyResult 'flux'   shape=[1]   particles=1000000   elapsed=1.025 s
#   mean . . . . . . . 1.7667e1 +/- 1.21e-2   (0.07% rel err)
#   figure of merit .. 2.080e6
#   variance of var. . 7.277e-6   skewness 1.875   kurtosis 5.277
#   large-score tail slope .. 7.78
#   statistical checks: PASSED (11/11)
#     mean stable....................... pass
#     relative error < 0.10............. pass
#     per-bin relative error < 0.10..... pass
#     relative error decreasing......... pass
#     relative error falls as 1/sqrt(N). pass
#     variance of variance < 0.10....... pass
#     variance of variance decreasing... pass
#     variance of variance falls as 1/N. pass
#     figure of merit stable............ pass
#     figure of merit has no trend...... pass
#     large-score tail slope > 3........ pass
```

Per-bin array (same shape as ``mean``):

| Attribute | Meaning |
|-----------|---------|
| ``variance`` | variance of the mean (``standard_deviation ** 2``) |

Tally-level reliability scalars, computed from the per-history *total* score:

| Attribute | Meaning |
|-----------|---------|
| ``aggregate_mean`` / ``aggregate_std_dev`` / ``aggregate_relative_error`` | statistics of the integrated tally total |
| ``aggregate_variance_of_variance`` | relative variance of the variance; large values flag an unreliable error estimate (rare large contributions are undersampled) |
| ``aggregate_skewness`` / ``aggregate_kurtosis`` | shape of the history-score distribution |
| ``aggregate_tail_slope`` | slope of the large-score PDF tail; the variance is well-defined only when this exceeds 3 |

The ``statistical_checks`` object gives the pass/fail verdict, and ``convergence_history`` records the statistics versus number of histories:

<!-- doctest: skip -->
```python
checks = result.statistical_checks
checks.passed                 # bool: overall verdict, False if the tail slope could not be estimated
checks.relative_error_ok      # individual checks: True / False / None (not evaluated)
print(checks.summary())       # the pass/fail table and verdict: PASSED, FAILED or INCOMPLETE

for point in result.convergence_history:
    print(point.n_histories, point.relative_error, point.figure_of_merit)
```

The raw empirical history-score PDF is available for inspection or plotting:

<!-- doctest: skip -->
```python
pdf = result.score_pdf
pdf.counts        # histogram counts per log-spaced magnitude bin
pdf.bin_centers   # bin centers (len == counts); bin_edges has one more
pdf.zero          # histories whose score was exactly zero
pdf.tail_slope    # the fitted large-score tail slope
```

#### Stopping on a precision target

A ``ConvergenceTarget`` ends a run early once a tally is precise enough. It works on the CPU (under MPI the decision is made on the rank-combined statistics) and on the GPU, where it is decided between kernel launches, so the run overshoots the target by at most one launch chunk:

<!-- doctest: skip -->
```python
model.convergence_targets = [yamc.ConvergenceTarget("relative_error", 0.02, tally="flux")]
# the run stops once flux relative error <= 2% (or total_particles / max_runtime is reached)
```

The metric is one of ``"relative_error"``, ``"std_dev"``, or ``"variance_of_variance"``, all on the tally's total, or ``"bin_relative_error"``, the relative error that 95% of the tally's scored bins meet (CPU only). Use ``"bin_relative_error"`` for a mesh tally, whose total converges long before its deep voxels do. Omit ``tally=`` to require the threshold for every tally.

### Nuclear-data uncertainty diagnostics

When transport was requested with `data_uncertainty`, a tally exposes the
nuclear-data spread separately from ordinary transport statistics. The
attributes below are `None` on a run without that request:

```python
sigma = result.nuclear_data_standard_deviation
if sigma is not None:
    print(sigma, result.nuclear_data_replica_spread, result.nuclear_data_noise)
    print(result.nuclear_data_noise_dominated)
    print(result.replica_mean, result.replica_standard_error)
    print(result.nuclear_data_linear_variance)
    print(result.nuclear_data_variance_by_nuclide)
```

`mean` remains the nominal calculation; `replica_mean` is the ensemble mean.
Replica spread includes Monte Carlo noise, so it is not interchangeable with
the noise-corrected nuclear-data sigma. The noise flag identifies bins where
that separation is poorly resolved. Per-nuclide contributions describe the
first-order variance, not every nonlinear contribution to the final spread.
See [Nuclear-data uncertainty](materials.md#nuclear-data-uncertainty) for the
sampling controls and coverage limitations.

### Structured results

Each ``TallyResult`` exposes ``shape`` and ``dim_labels`` for reshaping into numpy arrays.

<!-- doctest: skip -->
```python
result = results[tally]
print(result.shape)       # e.g. (1, 2, 20, 20, 20)
print(result.dim_labels)  # e.g. ('score', 'energy', 'mesh_z', 'mesh_y', 'mesh_x')

import numpy as np
data = np.array(result.mean).reshape(result.shape)
```

### xarray integration

With ``xarray`` installed (`pip install "yamc[xarray]"`), the built-in exporters
retain the tally's dimensions and coordinates:

<!-- doctest: skip -->
```python
result = results[tally]
da = result.to_xarray(field="mean")
std_dev = result.to_xarray(field="std_dev")
dataset = result.to_dataset()

# Select a single energy bin
flux_high = da.isel(energy=1)
```

`to_dataset()` exports mean, standard deviation and relative error together.
For manual array handling, `to_numpy()` and the flat arrays remain available.

### Plotting mesh tallies

``tally.plot()`` returns an interactive ``InteractiveTallyPlot`` that renders inline in Jupyter and saves as a self-contained HTML page with ``.to_html("file.html")``. Without a path ``to_html()`` returns the page as bytes.

<!-- doctest: skip -->
```python
plot = tally.plot(geometry=geometry, basis="xz")
plot.to_html("flux_map.html")
```


**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `geometry` | Geometry or MeshGeometry | `None` | Optional outline overlay |
| `basis` | str | `"xy"` | Initial slice plane: ``"xy"``, ``"xz"``, or ``"yz"`` |
| `slice_coord` | float | mesh center | Initial slice coordinate (cm) |
| `slices` | None, list[int], or ``"all"`` | None | Which slices to embed for the in-browser slider. ``None`` = just the initial slice; ``"all"`` = every bin (larger HTML, full slider) |
| `score_index` | int | 0 | Which score (when the tally has multiple) |
| `energy_index` | int or None | None | Energy bin; ``None`` sums over energy |
| `value` | str | `"mean"` | ``"mean"``, ``"standard_deviation"``, or ``"relative_error"`` |
| `colorscale` | str | `"Viridis"` | Plotly colorscale name (``Viridis``, ``Cividis``, ``Hot``, etc.) |
| `log_scale` | bool | `True` | Logarithmic color scale |
| `outline` | str or None | `"material"` | Outline by ``"material"``, ``"cell"``, or ``None`` |
| `resolution` | int | 400000 | Total pixel budget for geometry-outline sampling |
| `axis_units` | str | `"cm"` | Axis units: ``"mm"``, ``"cm"``, ``"m"``, or ``"km"`` |
| `contour_kwargs` | dict | None | Outline appearance: ``{"colors": [...], "linewidths": int}`` |
| `title` | str or None | None | Plot title (auto-generated if None) |
| `colorbar_title` | str or None | None | Colorbar label (auto-generated if None) |
| `scaling_factor` | float | 1.0 | Multiply all tally values by this factor (unit conversion) |
| `font_size` | int | 18 | Base font size in pixels for labels and ticks |
| `show_colorbar` | bool | `True` | Show the colorbar alongside the plot |

<!-- doctest: skip -->
```python
# Per-energy-bin slider over every bin, log scale, source-strength scaled
plot = tally.plot(
    geometry=geometry,
    basis="xz",
    slices="all",
    energy_index=0,
    value="mean",
    colorscale="Cividis",
    log_scale=True,
    axis_units="m",
    scaling_factor=1e14,           # multiply by source strength
    title="Fast neutron flux",
    colorbar_title="Flux [n/cm²/s]",
)
plot.to_html("flux_fast.html")
```

For manual matplotlib plotting, use ``extract_mesh_slice()`` with optional geometry outline overlay.

<!-- doctest: skip -->
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Extract tally slice
mean = tally.extract_mesh_slice(basis="xy", value="mean")
mean_2d = np.array(mean)

# Sample geometry for outline overlay
geom_slice = geometry.sample_slice(basis="xy", resolution=(400, 400))
edge_mask = np.array(geom_slice.edges("material"))

# Plot
fig, ax = plt.subplots()
vmin = mean_2d[mean_2d > 0].min()
im = ax.pcolormesh(mean.h_edges, mean.v_edges, mean_2d,
                   cmap="viridis", norm=LogNorm(vmin=vmin))
outline_rgba = np.zeros((*edge_mask.shape, 4))
outline_rgba[edge_mask] = [1, 1, 1, 0.8]
ax.imshow(outline_rgba, extent=geom_slice.extent, origin="lower",
          aspect="auto", interpolation="nearest", zorder=2)
fig.colorbar(im, ax=ax, label=f"Flux [{mean.h_label}]")
plt.savefig("manual_tally_plot.png")
```

The flux map below is rendered when these docs are built. The model is a small steel pipe carrying water with a 14.06 MeV point source on the axis, sliced in the ``xz`` plane:

```python exec="1" html="1"
import base64
from io import BytesIO

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import yamc

yamc.neutron_cross_section_data = "endf-b8.1"
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
mesh = yamc.RegularRectangularMesh(lower_left=(-15, -15, -15), upper_right=(15, 15, 15), shape=(60, 1, 60))
tally = yamc.Tally(scores=["flux"], mesh=mesh, name="flux_map")
yamc.Model(
    geometry=geometry, source=source, tallies=[tally], verbose=[]
).simulate_transport(total_particles=20000)

mean = tally.extract_mesh_slice(basis="xz", value="mean")
mean_2d = np.array(mean)
fig, ax = plt.subplots(figsize=(6, 5))
vmin = mean_2d[mean_2d > 0].min()
im = ax.pcolormesh(mean.h_edges, mean.v_edges, mean_2d, cmap="viridis", norm=LogNorm(vmin=vmin))
ax.set_xlabel(f"{mean.h_label} [cm]")
ax.set_ylabel(f"{mean.v_label} [cm]")
ax.set_aspect("equal")
fig.colorbar(im, ax=ax, label="flux per source neutron")
buf = BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
print(
    '<img alt="mesh tally flux map" '
    f'src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" '
    'style="max-width:100%">'
)
```


### VTK-HDF export and interactive 3D view

Mesh tally results export to VTK-HDF for [ParaView](https://www.paraview.org/) and, as shown below, can also be viewed in the browser. Export the result in the units you want (see [Units](#units)), and the values are written exactly as the result holds them, with each dataset's unit in its ``units`` attribute:

<!-- doctest: skip -->
```python
result = results.with_source_strength(1e20)[tally]

# Heating in W/m3 on a mesh in metres
result.to("W/m3").to_vtkhdf("heating.vtkhdf", length_units="m")

# Heating in W/cm3, flux in n/cm2/s and tritium production per cm3 per second,
# on a mesh in cm (the default)
result.to(["W/cm3", "1/(cm2 s)", "reactions/(cm3 s)"]).to_vtkhdf("results.vtkhdf")

# Options
result.to("W/m3").to_vtkhdf(
    "heating.vtkhdf",
    scores=["heating"],             # a subset of scores, by name or index
    datasets=("mean", "relative_error"),
    sum_energy=False,               # one array per energy bin
    sum_nuclides=True,              # sum over parent nuclide bins (default)
    length_units="m",
)
```

``length_units`` (``"cm"`` by default, ``"m"`` or ``"mm"``) sets only the coordinates and the unit string sets only the values, so pick them to agree: ``"W/m3"`` with ``"m"``, ``"W/cm3"`` with ``"cm"``. Every ``to_vtkhdf`` (geometry, cells, regions, sources, models, tracks and CAD meshes) takes the same ``length_units``, so files written with the same unit overlay in ParaView:

<!-- doctest: skip -->
```python
model.to_vtkhdf("model.vtkhdf", length_units="m")
tracks.to_vtkhdf("tracks.vtkhdf", length_units="m")
```

``Tally.to_vtkhdf()`` writes the tally itself, per source particle and divided by the element volume in cm³ (``volume_normalization=True``), with a ``scaling_factor`` you work out yourself. It writes no ``units`` attribute, so prefer exporting the result:

<!-- doctest: skip -->
```python
tally.to_vtkhdf("flux.vtkhdf")                          # per source particle per cm3
tally.to_vtkhdf("heating.vtkhdf", scaling_factor=1e20 * 1.602e-19)  # W/cm3, by hand
```

The viewer below is rendered live when these docs are built. It is a neutron radiograph: the word ``YAMC`` is built from boron-10 blocks (a strong thermal-neutron absorber), a thermal point source (orange marker) sits in front of them, and the letters cast a shadow onto a flux screen behind them. Both the flux tally and the geometry are written to VTK-HDF; the grey ``YAMC`` blocks are the geometry (read back from ``geometry.vtkhdf``), and the coloured panel is the flux screen (read back from ``flux.vtkhdf``), where the letters appear as dark shadows. Drag to rotate, scroll to zoom, right-drag to pan. The same files open in [ParaView](https://www.paraview.org/) for further analysis.

```python exec="1" html="1"
import functools
import html
import operator

import numpy as np
import pyvista as pv
from vtkmodules.vtkIOHDF import vtkHDFReader

import yamc

pv.OFF_SCREEN = True
yamc.neutron_cross_section_data = "endf-b8.1"

# 5x7 block font for the word YAMC.
FONT = {
    "Y": ["X...X", "X...X", ".X.X.", "..X..", "..X..", "..X..", "..X.."],
    "A": [".XXX.", "X...X", "X...X", "XXXXX", "X...X", "X...X", "X...X"],
    "M": ["X...X", "XX.XX", "X.X.X", "X...X", "X...X", "X...X", "X...X"],
    "C": [".XXX.", "X...X", "X....", "X....", "X....", "X...X", ".XXX."],
}
PX, GAP, WORD = 1.6, 2.2, "YAMC"


def box(xlo, xhi, ylo, yhi, zlo, zhi, vacuum=False):
    b = {"boundary": "vacuum"} if vacuum else {}
    return (
        yamc.Plane(axis="x", offset=xlo, **b).above & yamc.Plane(axis="x", offset=xhi, **b).below
        & yamc.Plane(axis="y", offset=ylo, **b).above & yamc.Plane(axis="y", offset=yhi, **b).below
        & yamc.Plane(axis="z", offset=zlo, **b).above & yamc.Plane(axis="z", offset=zhi, **b).below
    )


# Build the letters as a union of CSG boxes (run-length-encoded per row).
x0 = -(len(WORD) * 5 * PX + (len(WORD) - 1) * GAP) / 2
z_top = 7 * PX / 2
pixels = []
for li, ch in enumerate(WORD):
    lx = x0 + li * (5 * PX + GAP)
    for r, row in enumerate(FONT[ch]):
        c = 0
        while c < len(row):
            if row[c] == "X":
                c2 = c
                while c2 < len(row) and row[c2] == "X":
                    c2 += 1
                pixels.append(box(lx + c * PX, lx + c2 * PX, -1.5, 1.5, z_top - (r + 1) * PX, z_top - r * PX))
                c = c2
            else:
                c += 1
letters = functools.reduce(operator.or_, pixels)
world = box(-22, 22, -18, 6, -9, 9, vacuum=True)

boron = yamc.Material(composition={"B10": 1.0}, density=2.3, name="boron10")
geometry = yamc.Geometry([
    yamc.Cell(region=letters, material=boron, name="letters"),
    yamc.Cell(region=world & ~letters, name="void"),
])
source = yamc.NeutronSource(energy=0.0253, position=(0, -14, 0))  # thermal point source in front
screen = yamc.RegularRectangularMesh(lower_left=(-22, 2.0, -9), upper_right=(22, 3.0, 9), shape=(150, 1, 60))
tally = yamc.Tally(scores=["flux"], mesh=screen, name="radiograph")
yamc.Model(
    geometry=geometry, source=source, tallies=[tally], verbose=[]
).simulate_transport(total_particles=1_000_000)

# Export both the tally and the geometry to VTK-HDF, then read them back.
tally.to_vtkhdf("flux.vtkhdf")
geometry.to_vtkhdf("geometry.vtkhdf", resolution=400000)


def read_vtkhdf(path):
    reader = vtkHDFReader()
    reader.SetFileName(path)
    reader.Update()
    return pv.wrap(reader.GetOutput())


flux_screen = read_vtkhdf("flux.vtkhdf")
geom = read_vtkhdf("geometry.vtkhdf")

plotter = pv.Plotter(window_size=(820, 560))
plotter.add_mesh(flux_screen, scalars="flux_mean", cmap="viridis", show_scalar_bar=False)
plotter.add_mesh(geom.threshold(0.5, scalars="material_id"), color="dimgray")  # the YAMC blocks
plotter.add_mesh(pv.Sphere(radius=0.8, center=(0, -14, 0)), color="orange")  # the source
plotter.add_mesh(flux_screen.outline(), color="lightgray")
plotter.camera_position = [(35, -45, 28), (0, -4, 0), (0, 0, 1)]

doc = plotter.export_html(filename=None).read()
print(
    f'<iframe srcdoc="{html.escape(doc, quote=True)}" width="100%" height="580" '
    'style="border:1px solid #ddd;border-radius:4px"></iframe>'
)
```

### Performance stats

Timing information is stored on the results returned by ``simulate_transport``:

<!-- doctest: skip -->
```python
results = model.simulate_transport()
print(f"Total time:     {results.elapsed:.2f} s")
print(f"Particles/sec:  {results.particles_per_second}")
print(f"Data load time: {results.data_load_time:.2f} s")
print(f"Transport time: {results.transport_time:.2f} s")
```


