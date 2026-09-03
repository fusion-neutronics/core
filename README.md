# core

The Rust workspace behind [`yamc`](https://github.com/fusion-neutronics/yamc)
and [`yani`](https://github.com/fusion-neutronics/yani). Everything that
compiles lives here: 25 crates, the pyo3 bindings, and the two wheels that get
published to PyPI.

**This is not the repository to install from, and not where the user
documentation lives.** Those are the two public repositories above, which are
also where issues, discussions and the released docs belong. What is here is
what only someone with this checkout needs.

## What it publishes

Two compiled wheels, whose distribution names differ from their import names
(as `pillow` provides `PIL`):

| built from | distribution | imported as | front door |
|---|---|---|---|
| `packages/yamc-core/` | `yamc-core` | `yamc` | `yamc`, pinning `yamc-core==<same version>` |
| `packages/yani-core/` | `yani-core` | `yani` | `yani`, pinning `yani-core==<same version>` |

A user installs the front door, not the `-core` wheel. The front door is a
version pin and the documentation; the `-core` wheel is the compiled artefact
this workspace produces.

The two wheels are alternatives rather than layers. `yamc` is a superset that
adds transport, geometry and tallies on top of the transmutation stack, and
installing both puts two extension modules in one process. See
`packages/yani-core/README.md`.

## Getting started

```bash
cargo test                       # the Rust workspace
maturin develop --release        # build and install the yamc extension
pytest                           # the Python tests
```

Full instructions, including the nuclear data the tests need, GPU and MPI
builds, profiling, the wasm blobs and the release procedure, are in
**[docs/developer_info.md](docs/developer_info.md)**, which is also the internal
mkdocs site (`mkdocs build -f mkdocs.yml`). That file has the annotated
crate-by-crate workspace layout.

## Continuous integration

| workflow | what it covers |
|---|---|
| [`ci-rust.yml`](.github/workflows/ci-rust.yml) | the Rust workspace across the supported targets |
| [`ci-python.yml`](.github/workflows/ci-python.yml) | the wheels, the Python test suites and the published stubs |
| [`ci-wasm.yml`](.github/workflows/ci-wasm.yml) | the wasm32 builds |
| [`ci-endf-goldens.yml`](.github/workflows/ci-endf-goldens.yml) | the ENDF and ACE converters against pinned goldens |
| [`licenses.yml`](.github/workflows/licenses.yml) | the third-party license bundle shipped in each wheel |
| [`docs-mkdocs.yml`](.github/workflows/docs-mkdocs.yml) | this repository's internal site |
| [`commit-authors.yml`](.github/workflows/commit-authors.yml) | commit authorship on every pull request |

Status badges are deliberately absent. GitHub serves `badge.svg` through an
image proxy that fetches anonymously, so a badge for a private repository
returns 404 whether or not the reader is signed in, which makes badges here
permanently broken images rather than a signal. The Actions tab works normally.
