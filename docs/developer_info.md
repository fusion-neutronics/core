# Developer Info

This guide shows how to clone, build, test, and generate documentation for the project. Instructions are written for Linux but equivalent commands should be possible in Windows and Mac OS.

## 1. Clone the Repository

```bash
git clone https://github.com/fusion-neutronics/core.git
cd core
```

## 2. Install Rust

Install Rust by following the [official Rust installation guide](https://rust-lang.org/tools/install/)


## 3. Build Rust Package:

The project is a Cargo workspace of 24 crates. See [Workspace Layout](#workspace-layout) below for what lives where. Build everything with:

```bash
cargo build
```

To build only a specific crate:

```bash
cargo build -p yamc
cargo build -p yamt
cargo build -p yamm
cargo build -p yani
```

## 4. Run Rust tests

The test suite reads nuclear data from `crates/yamc/tests/*.arrow`. Those are
published data, not source, so they are **downloaded once** rather than
committed (issue #126):

```bash
python scripts/fetch_test_fixtures.py     # ~530 MB, into ~/.cache/yamc
```

It fetches the endf-b8.1 nuclide, element and transmutation-chain sections into
the same on-disk cache production uses and symlinks them into
`crates/yamc/tests/`. Re-running is cheap: anything already cached is left
alone. `--check` reports what is missing without downloading.

`YAMC_CACHE_DIR` overrides where that cache is, for the script and for yamc
itself. It names the cache root verbatim rather than a home directory to derive
one from, and it is the supported way to point a process at an isolated cache.
Without it the root is `<home>/.cache/yamc`, where home is `USERPROFILE` on
Windows and `HOME` everywhere else.

The tests find it through `yamc_test_cache` (`crates/yamc-test-cache`) rather
than resolving it themselves. Thirty-five of them used to read `$HOME` directly
and fall back to a hardcoded developer path, which on Windows resolved to a
directory that does not exist: every fixture load returned nothing, every test
took its "data absent" skip path, and the suite reported green having read
nothing (issue #544). `YAMC_REQUIRE_FIXTURES=1`, which CI sets after the fetch
step, turns that absence back into a failure.

```bash
cargo test --workspace
```

### GPU tests - run single-threaded

The GPU path shares a **process-global cubecl client/device** (a cached
`GpuContext` singleton). Running GPU tests in parallel launches multiple
kernels concurrently on that one shared client, which intermittently corrupts
results (you'll see `alive=0, mean steps=0, zero tally` - non-deterministic).
So GPU tests **must run single-threaded**. There is a cargo alias for this:

```bash
cargo test-gpu            # = test -p yamc-gpu -p yamc --features yamc/gpu --release -- --test-threads=1
```

or spell it out:

```bash
cargo test -p yamc-gpu --release -- --test-threads=1                 # GPU kernel tests
cargo test -p yamc --features gpu --release -- --test-threads=1      # GPU integration tests
```

(On machines with no f64-capable Vulkan adapter these tests self-skip, so a
plain `cargo test --workspace` stays correct there - the flag only matters on a
real GPU box. The same applies in user code: each `model.simulate_transport(
compute='gpu')` call resets the shared client, so back-to-back calls are safe,
but only one GPU transport runs at a time within a process.)

## 5. Build the Python Extension

(Optional) Create a Python virtual environment:
```bash
sudo apt install python3-venv
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

Build and install the Python module in editable mode. The wheel is defined by
`packages/yamc-core/pyproject.toml`, not by anything at the repo root, so the
package directory is what gets installed:
```bash
pip install -e packages/yamc-core
```

This compiles the Rust extension through the maturin build backend (in release mode) and installs `yamc` as an editable package. Edits to the Python sources under `packages/yamc-core/python/yamc/` are picked up immediately; after changing Rust code, re-run `pip install -e packages/yamc-core` to rebuild. The default features include `mesh` and `cad`.

Add the optional extras you need:
```bash
pip install -e "packages/yamc-core[all]"      # runtime extras: CAD meshing (cadquery), VTK/HDF export (h5py), and numpy
pip install -e "packages/yamc-core[all,dev]"  # also installs the dev and test tooling (pytest, matplotlib, tqdm)
```

For an MPI-enabled build (which requires the `mpi` cargo feature and a local MPI install), see the [installation guide](https://fusion-neutronics.github.io/yamc/installation/).

> You can also build the extension directly with [maturin](https://www.maturin.rs/) from inside the package directory (for example `cd packages/yamc-core && maturin develop`, or `maturin build` to produce a standalone abi3 wheel) if you want finer control over the cargo features or build profile. maturin has no flag that points at a pyproject, its `-m` takes a Cargo manifest, so the package directory has to be the working directory. See the [maturin documentation](https://www.maturin.rs/) for details.

### Building for the machine you will run on

The published wheels target a baseline x86-64, so they run anywhere. If you are
building from source **on the machine that will run it** -- a cluster node, a
workstation -- you can let the compiler use that machine's instruction set:

```bash
RUSTFLAGS="-C target-cpu=native" pip install -e packages/yamc-core
```

Measured at about **3%** on the CRAM solve (`docs/transmute_speed.md` has the
numbers and the two flags that turned out to be worth nothing). Modest, and it
costs nothing but a rebuild.

> **Never put this in a published wheel.** `native` emits whatever instructions
> the build machine happens to have, so the result crashes with an illegal
> instruction on any older CPU that lacks them, and under Rosetta 2. It is a
> from-source option for a known machine, which is why it is not in
> `.cargo/config.toml`.

Note that `[target.<triple>] rustflags` in `.cargo/config.toml` *replaces*
`[build] rustflags` rather than merging with it, so per-target flags there are
easy to get wrong; passing `RUSTFLAGS` on the command line avoids the question.

## 6. Run Python Tests

Install test dependencies:
```bash
pip install pytest numpy h5py cadquery pandas
```

Run tests (the same downloaded fixtures as the Rust suite; pytest stops with the
fetch command if they are missing):
```bash
python scripts/fetch_test_fixtures.py     # once, if not already done
pytest packages/yamc-core/tests/
```

That command covers the CAD tests too, since `packages/yamc-core/tests/cad/`
sits inside it, and they self-skip unless cadquery is installed. To install it
and run just that tier:
```bash
pip install cadquery
pytest packages/yamc-core/tests/cad/
```

A bare `pytest` from the repo root runs all three Python suites: the yamc tests
above, `packages/yani-core/tests/` and `pytests/parity/`. The latter two need
the standalone `yani` wheel installed and self-skip without it. The list lives
in `pytest.ini`, which also explains why each suite sits where it does.

## 7. Build Rust API Docs

```bash
cargo doc --workspace --no-deps
python -m webbrowser target/doc/yamc/index.html
```

## 8. Build Python Docs

The docs are built with [MkDocs](https://www.mkdocs.org/) (Material theme). The
configuration lives in `mkdocs.yml` and the sources in `docs/`.

Install doc requirements:
```bash
pip install -r docs/requirements.txt
```

Building the Python extension is not a prerequisite for this site. The pages
that executed their own code blocks moved to the user-facing repositories, and
the only `mkdocstrings` reference left here is `packages/nuclear_data_to_arrow`,
which is plain Python read straight from source. Nothing here imports `yamc`.

Build the static site (run from the repo root, outputs to `site/`):
```bash
mkdocs build -f mkdocs.yml
```

Open docs:
```bash
python -m webbrowser site/index.html
```

For a live-reloading preview while editing (serves at http://127.0.0.1:8000):
```bash
mkdocs serve -f mkdocs.yml
```

To force a full clean rebuild:
```bash
mkdocs build -f mkdocs.yml --clean
```

## Workspace Layout

```
core/
  crates/
    yamc/           -- Top-level Monte Carlo transport driver (Model,
                       simulate_transport, coupled transmutation, GPU
                       dispatch)
    yamc-particle/  -- Runtime particle state (Particle struct + helpers)
    yamc-source/    -- Particle source specs + spatial/energy/angle samplers
    yamc-geo/       -- CSG geometry primitives (surfaces, regions, BVH);
                       compiles to native and wasm32
    yamc-nuclide/   -- Evaluated neutron nuclear data (Nuclide, Reaction,
                       reaction-product sampling)
    yamc-element/   -- Evaluated photon nuclear data (Element, photon
                       interaction cross-sections, atomic transitions,
                       TTB table build)
    yamc-materials/ -- Material composition + macroscopic cross sections
    yamc-physics/   -- Collision physics (scatter/inelastic kinematics,
                       secondary photons, TTB emission, D1S photon
                       production, particle banking)
    yamc-rng/       -- The shared PCG stream (per-history and per-secondary
                       seeding); mirrored by the GPU #[cube] twin
    yamc-tallies/   -- Tallies (filters, scores, per-history Welford
                       variance, SimulationResults, Arrow IO)
    yamc-plot/      -- Shared interactive-plot HTML/JS templating used by
                       Python's Model.plot() and the browser editor
    yamc-gpu/       -- GPU compute infra (Vulkan via cubecl, f64 required)
    yamc-convert/   -- Converts ENDF and ACE evaluations into the transport
                       Arrow format yamc reads
    yamc-python/    -- pyo3 bindings; built into
                       packages/yamc-core/python/yamc/_core.so
    yamt/           -- Mesh-based geometry runtime (BVH, Arrow IO)
    yamm/           -- CDT surface mesher with DAG scheduler
    endf/           -- Parser for ENDF-6 formatted evaluated nuclear data
                       files
    nuclear-data-schema/
                    -- The Arrow schema of the simulation-ready nuclear
                       data format, declared once
    yani/           -- CRAM matrix-exponential transmutation solver
    yani-transmute/ -- Transmutation driver over yani (irradiation
                       schedules, multigroup spectrum collapse,
                       per-material inventories, reaction-rate tallies)
    yani-decay/     -- Decay-chain observables (chain enumeration, Bateman
                       activity, activity/decay heat, D1S time-correction
                       factors)
    yani-convert/   -- Converts ENDF decay, fission-yield and neutron
                       evaluations into the transmutation Arrow format yani
                       reads
    yani-python/    -- pyo3 bindings for the transmutation stack
                       (materials, nuclides, chains, schedules,
                       inventories), shared by both wheels
    yani-wasm/      -- WebAssembly bindings for yani (transmutation and
                       activation in the browser)
  packages/
    yamc-core/      -- The yamc wheel: pyproject.toml, README.md, LICENSE
      python/
        yamc/       -- Python package (re-exports from yamc._core)
          cad/      -- CAD meshing pipeline (CadToYamc, etc.)
          vtkhdf.py -- VTKHDF mesh-result export
      tests/        -- Python tests for transport (unit_tests/,
                       integration_tests/, regression_tests/, cad/ and
                       typing/)
    yani-core/      -- The yani wheel (python/yani/) plus the tests that
                       prove it ships none of the transport stack
    nuclear_data_to_arrow/
                    -- ENDF/ACE to Arrow converter, pure Python and
                       unpublished
  pytests/
    parity/         -- Tests needing both wheels installed at once, which is
                       why they belong to neither package
  pytest.ini        -- Repo-wide pytest config; testpaths lists all three
                       Python suites above
  examples/
    python/         -- Python example scripts (TBR, transmutation, photon
                       transport, MPI, perf matrix, etc.)
  docs/             -- MkDocs documentation source (built via mkdocs.yml)
```

The split into many small crates is intentional: `yamc-geo`, `yamc-particle`,
and `yamc-source` are wasm-compatible and free of system dependencies so
they can be reused from the browser build, while `yamc-gpu` and `yamc-plot`
are isolated so they don't pull heavy dependencies (cubecl, plot templating)
into builds that don't need them.

The transmutation stack is layered so it can be built without transport:
`yani` (solver) and `yani-decay` (chain observables) are leaves, and
`yani-transmute` reaches only `yani`, `yamc-materials`, `yamc-nuclide` and
`yamc-element`. `Model::transmute` in `yamc` is the sole part that needs
transport, because it runs a transport solve per timestep. `yamc-materials`
deliberately does not depend on `yamc-physics`, so materials and collision
physics can be compiled independently of each other.

### Profile the code

High level profile
```bash
# Build with debug symbols in release mode
cargo build --release -p yamc --example tbr
RUSTFLAGS="-C force-frame-pointers=yes" cargo build --release -p yamc --example tbr

# Profile with full call graph
sudo perf record -g --call-graph dwarf ./target/release/examples/tbr

# View interactive report
sudo perf report --no-children

# Or generate text report with full call stacks
sudo perf report --no-children --stdio > perf_report.txt
```

Detailed profile

```bash
RUSTFLAGS="-C force-frame-pointers=yes" cargo build --release -p yamc --example tbr
sudo perf record -g --call-graph dwarf ./target/release/examples/tbr
sudo perf report --no-children --stdio | head -200
```

### Test Coverage

#### Rust coverage

Install `cargo-llvm-cov` and the LLVM tools component (one-time setup):

```bash
cargo install cargo-llvm-cov
rustup component add llvm-tools-preview
```

Run tests with coverage and print a summary to the terminal:

```bash
cargo llvm-cov --workspace
```

Generate an HTML report:

```bash
cargo llvm-cov --workspace --html
python -m webbrowser target/llvm-cov/html/index.html
```

#### Python coverage

Install `pytest-cov` (one-time setup):

```bash
pip install pytest-cov
```

Run Python tests with coverage:

```bash
pytest packages/yamc-core/tests/ --cov=yamc --cov-report=term-missing
```

Generate an HTML report:

```bash
pytest packages/yamc-core/tests/ --cov=yamc --cov-report=html
python -m webbrowser htmlcov/index.html
```

### Formatting, linting and checking

The CI runs some checks on the Rust code (fmt check and clippy) and also some checks on the Python code (ruff check). The same checks can be run locally with the use of the pre-commit file.

```bash
pip install pre-commit
pre-commit run --all-files
```

If the pre-commit hooks fail (e.g. due to a missing MPI installation) and you still need to commit, you can skip the hooks with `--no-verify`:

```bash
git commit --no-verify -m "commit message"
```

## Releasing

Two distributions are published from this repository, and they carry
independent version numbers: `yamc-core` (the transport wheel) and `yani-core`
(the standalone transmutation wheel).

### Where the version lives

Each wheel's version lives in the crate manifest that its `pyproject.toml`
names as `manifest-path`:

| Distribution | Version lives in | Declared in |
| --- | --- | --- |
| `yamc-core` | `crates/yamc-python/Cargo.toml` | `packages/yamc-core/pyproject.toml` |
| `yani-core` | `crates/yani-python/Cargo.toml` | `packages/yani-core/pyproject.toml` |

Both `pyproject.toml` files declare `dynamic = ["version"]`, which is what tells
maturin to read the number from the manifest. Neither crate is published to
crates.io, so its version field is free to carry the wheel's number — one number
in one place, rather than two that have to be kept in step.

The committed version names the **next** release, not the last one. That is what
makes a bump reviewable: the number that will reach PyPI appears in the diff.

### Bumping a version

```bash
cargo install cargo-edit                      # once
cargo set-version -p yamc-python 0.8.1        # yamc-core
cargo set-version -p yani-python 0.10.1       # yani-core
```

`cargo set-version` updates `Cargo.lock` in the same step, so both files land in
the same commit. Without cargo-edit, `bash scripts/set-wheel-version.sh
yamc-python 0.8.1` does the same two edits (it is what CI uses, so that the wheel
jobs do not have to install cargo-edit on three platforms).

Bump in the PR that earns the bump.

### Tagging a release

A release tag **names every project it publishes and the version each goes out
at**, one project per comma:

```
yamc-core-0.8.1                    yamc-core alone
yani-core-0.9.1                    yani-core alone
yamc-core-0.8.1,yani-core-0.9.1    both, at their own versions
```

A leading `v` on each version is allowed (`yamc-core-v0.8.1`), and whitespace
around a comma is ignored. There is no bare `0.8.1` tag: one tag carries one
number, and the two distributions are versioned apart.

The flow is therefore:

1. Bump the version of each project you are releasing, and merge that PR.
2. Tag with those exact versions and publish the GitHub release.
3. `.github/workflows/ci-python.yml` builds the wheels and publishes to PyPI.

Before anything is uploaded, the `verify-versions` job fails the release if a
tag component names a version that disagrees with its manifest, if the tag names
no project at all, or if the version is already on PyPI. So a tag cannot ship a
number nobody bumped, and a bump cannot ship under a number nobody tagged.

### Publishing without a release

`workflow_dispatch` on the same workflow publishes without a tag, for the two
cases where there should not be a release: a first upload that creates the
project on PyPI, and a re-publish of one wheel after a fix. The `yamc_version` /
`yani_version` inputs override the manifests for that run, since neither case has
a tag to check against.
