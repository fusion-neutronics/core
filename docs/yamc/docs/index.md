# YAMC

YAMC (Yet Another Monte Carlo) is an open-source Monte Carlo particle transport code for
neutron and photon simulations, aimed at fusion neutronics workflows.

It supports constructive solid and mesh-based geometries, coupled neutron-photon transport,
transmutation, shutdown dose rate analysis, CAD geometry import, flexible tallies, and
convenient post-processing, all accessible through a Python API.

YAMC's priorities are:

- First-class support for fusion neutronics workflows
- Runs on Linux, macOS, and Windows, on x86 and ARM CPUs and AMD, NVIDIA, and Intel GPUs, and in the browser via WebAssembly
- Straightforward installation with `pip`, no compilation step
- A fully permissively licensed open-source dependency stack
- Batteries included: CAD geometry meshing, transmutation, convenient cross section data downloading, and visualization built in

What each of those means in practice is on [Why YAMC](why_yamc.md).

## Explore

- [Why YAMC](why_yamc.md): what it does well, and where to read more about each
- [Installation](installation.md): install with `pip`, no compilation step
- [Quick start](quick_start.md): a minimal simulation end to end
- [Usage guide](usage_guide.md): materials, geometry, sources, tallies, transmutation, CAD
- [Nuclear data libraries](nuclear_data_libraries.md): the six you can choose between, and how much is in each
- [Python API](api.md): the type-annotated reference, generated from the package's stubs

To ask questions or get support, try the
[GitHub Discussions](https://github.com/fusion-neutronics/yamc-and-yani/discussions) page.
