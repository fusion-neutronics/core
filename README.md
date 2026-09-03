<!-- The CI badges point at `fusion-neutronics/core` because ci-python, ci-rust
and ci-wasm run here. The Documentation badge points at `fusion-neutronics/yamc`
on purpose: docs.yml lives in that repo, not this one, so retargeting it here
would give a 404. It renders blank until that repo is public. -->
[![Test usage with Python](https://github.com/fusion-neutronics/core/actions/workflows/ci-python.yml/badge.svg)](https://github.com/fusion-neutronics/core/actions/workflows/ci-python.yml)
[![Test usage with Rust](https://github.com/fusion-neutronics/core/actions/workflows/ci-rust.yml/badge.svg)](https://github.com/fusion-neutronics/core/actions/workflows/ci-rust.yml)
[![Documentation](https://github.com/fusion-neutronics/yamc/actions/workflows/docs.yml/badge.svg)](https://github.com/fusion-neutronics/yamc/actions/workflows/docs.yml)
<!-- [![Test usage with WASM](https://github.com/fusion-neutronics/core/actions/workflows/ci-wasm.yml/badge.svg)](https://github.com/fusion-neutronics/core/actions/workflows/ci-wasm.yml) -->

# YAMC

YAMC (Yet Another Monte Carlo) is an open-source Monte Carlo particle transport code for **neutron and photon simulations**, with a focus on **fusion neutronics workflows**.

It supports constructive solid and mesh-based geometries, coupled neutron-photon transport, transmutation, shutdown dose rate analysis, CAD geometry import, flexible tallies, convenient post processing and support for portable simulations all accessible through a Python API.

## Installation

```bash
pip install yamc
```

See the **[documentation](https://fusion-neutronics.github.io/yamc/)** for installation and usage details.

Ask questions or get support on the **[GitHub Discussions forum](https://github.com/fusion-neutronics/yamc/discussions)**.

To build, test, and contribute to YAMC, see the **[developer guide](https://fusion-neutronics.github.io/yamc/developer_info.html)**.

🦀 **Rust core** for high performance across Linux, macOS, and Windows (x86_64 and ARM), with WebAssembly support for browser-based deployment.

🐍 **Python API** where all computation runs in Rust -- Python calls are thin wrappers with no overhead.
