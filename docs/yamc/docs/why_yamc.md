# Why YAMC

YAMC is a Monte Carlo neutron and photon transport code for fusion. These are
the things it does well.

## Easy

[YAMC online](https://fusion-neutronics.github.io/yamc-online/) runs the same
transport engine in your browser, on your own machine, with no account and no
upload, so a first run needs nothing installed.

`pip install yamc` on Linux, macOS or Windows, on x86 or ARM. The solver is a
Rust core shipped inside the wheel behind thin
[PyO3](https://pyo3.rs) bindings that target Python's stable ABI (3.10+), so
nothing is compiled at install time. The dependency-free core also builds to
WebAssembly, which is what YAMC online runs. See [Installation](installation.md).

Nuclear data downloads itself. Pick a library and only the parts your model
needs are fetched and cached, down to just the temperatures you set on your
materials. To work offline, download the full data files and point YAMC at the
directory. See [Nuclear data](nuclear_data.md).

YAMC is MIT or Apache-2.0, and nothing in its dependency stack is copyleft,
including the GPU path.

`model.to_html()` writes one file that runs the simulation in any browser, for
a colleague without Python or a demo in a course. See
[Export to a runnable HTML page](models.md#export-to-a-runnable-html-page-to_html).

## Accurate

Cross sections are used on the evaluation's own energy grid, with no multigroup
collapse. See [Method](method.md).

Statistics are kept per history, with no batches to choose, and each tally
result carries automatic reliability checks on its error, variance of variance
and convergence. See [Statistical reliability](results.md#statistical-reliability).

A transmutation can return every nuclide density, activity, decay heat and dose
with an uncertainty propagated from the evaluation's own covariance data. See
[Nuclear-data uncertainty](materials.md#nuclear-data-uncertainty).

## Fast

`compute="gpu"` runs on AMD, NVIDIA or Intel GPUs in double precision, through
one cubecl/Vulkan code path rather than a proprietary single-vendor toolkit.
See [GPU transport](simulate_transport.md#gpu-transport-computegpu).

Weight windows are generated automatically from the model, and delta tracking
and survival biasing are one setting each. See [Weight windows](weight_windows.md).

Set a target relative error on a tally and the run ends as soon as it is met.
See [Stopping on a precision target](results.md#stopping-on-a-precision-target).

The dose at any cooling time, for any irradiation schedule, comes from a single
transport run. See [Shutdown dose rate](simulate_shutdown_dose.md).

## Flexible

YAMC includes its own CAD mesher and mesh transport, so a CadQuery assembly or
STEP file goes straight into a model, alongside or inside CSG cells. See
[Geometry](geometry.md).

A tokamak plasma source comes from plasma parameters, and tallies cover
heating, DPA, tritium production and dose. See [Sources](sources.md) and
[Tallies](tallies.md).

Materials can be depleted coupled to a transport run, or from a spectrum you
already have, giving activity, decay heat, decay photons and waste clearance.
See [Simulate transmutation](simulate_transmutation.md).

Cross sections, reactions, decay data and fission yields can each come from a
different library, or from your own files. See
[Nuclear data settings](nuclear_data.md#nuclear-data-settings).

Nuclear data, tally results and meshes are stored and exchanged as Apache
Arrow, which is columnar and language-agnostic, so data files are portable and
load quickly. Tallies convert to numpy, pandas, polars or xarray with little or
no copying, and export to VTK-HDF for ParaView. See [Results](results.md).
