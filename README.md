# yamc-and-yani

Source, documentation and releases for two fusion neutronics codes:

| code | what it does | PyPI | docs |
|---|---|---|---|
| **yamc** | Monte Carlo neutron and photon transport, with transport-coupled transmutation and shutdown dose | [`yamc`](https://pypi.org/project/yamc/) | [docs](https://fusion-neutronics.github.io/yamc-and-yani/yamc/) |
| **yani** | nuclide inventory: transmutation and activation from a material, a schedule and a spectrum | [`yani`](https://pypi.org/project/yani/) | [docs](https://fusion-neutronics.github.io/yamc-and-yani/yani/) |

```bash
pip install yamc   # or: pip install yani
```

Both are compiled wheels built from this Rust workspace. The two are
alternatives rather than layers: `yamc` is a superset that adds transport,
geometry and tallies on top of the transmutation stack, and installing both
puts two extension modules in one process (see
`packages/yani/python/yani/__init__.py`).

## Layout

| path | what |
|---|---|
| `crates/` | the Rust workspace, including the pyo3 bindings |
| `packages/yamc/`, `packages/yani/` | the two Python distributions and their tests |
| `docs/yamc/`, `docs/yani/` | the user documentation for each code |
| `docs/index.html` | the splash page at the root of the docs site |
| `docs/developer/` | how to build and release the workspace |

## Getting started

```bash
cargo test                                   # the Rust workspace
pip install -e "packages/yamc[all,dev]"      # build and install yamc
pytest                                       # the Python tests
```

Full instructions, including the nuclear data the tests need, GPU and MPI
builds, the wasm blobs and the release procedure, are in
**[docs/developer/developer_info.md](docs/developer/developer_info.md)**.

## Continuous integration

| workflow | what it covers |
|---|---|
| [`ci-rust.yml`](.github/workflows/ci-rust.yml) | the Rust workspace across the supported targets |
| [`ci-python.yml`](.github/workflows/ci-python.yml) | the wheels, the Python test suites, the published stubs, both docs sites and the PyPI releases |
| [`ci-wasm.yml`](.github/workflows/ci-wasm.yml) | the wasm32 builds |
| [`licenses.yml`](.github/workflows/licenses.yml) | the third-party license bundle shipped in each wheel |
| [`commit-authors.yml`](.github/workflows/commit-authors.yml) | commit authorship on every pull request |

## Licence

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this work by you, as defined in the Apache-2.0 license, shall
be dual-licensed as above, without any additional terms or conditions.
