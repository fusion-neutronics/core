# Installation

YAMC can be installed and used on Linux, Windows and MacOS.

## Standard Installation (Recommended)

YAMC is a Rust back end package with a Python API. Installation requires no additional Python package dependencies.
YAMC is available on [PyPI](https://pypi.org/project/yamc).

### Install from PyPI

To install, make sure you have [pip installed](https://pip.pypa.io/en/stable/installation/), then run:

```bash
pip install yamc
```

This installs the core YAMC transport solver with multi-threading support (via Rayon). MPI is not required.

### Optional extras

YAMC provides optional extras for additional functionality:

| Extra | Install command | What it adds |
|-------|----------------|--------------|
| cad | `pip install "yamc[cad]"` | CAD meshing pipeline, surface and tetrahedral (requires cadquery) |
| viz | `pip install "yamc[viz]"` | VTKHDF visualization export (h5py, numpy) |
| pandas | `pip install "yamc[pandas]"` | pandas DataFrame export of tally results |
| polars | `pip install "yamc[polars]"` | polars DataFrame export of tally results |
| xarray | `pip install "yamc[xarray]"` | xarray (labelled N-D array) export of tally results |
| all | `pip install "yamc[all]"` | cad and viz (does not include pandas, polars or xarray) |

Extras combine, as in `pip install "yamc[all,pandas]"`. The quotes stop shells such as zsh, the default on macOS, from reading the square brackets as a file name pattern.

To install the CAD-to-mesh-to-transport pipeline and the visualization export together:

```bash
pip install "yamc[all]"
```

### Virtual Environment (Recommended)

Further instructions are in the [Python docs](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install "yamc[all]"
```

### Install the development version

`pip install yamc` gives the latest release. The unreleased changes on the main branch build from source, which needs a [Rust toolchain](https://rustup.rs/):

```bash
pip install "yamc @ git+https://github.com/fusion-neutronics/yamc-and-yani.git#subdirectory=packages/yamc"
```

### Check which build is installed

`yamc.__version__` is the installed `yamc` version. `yamc.__commit__` is the full git hash of the core commit the extension was built from, and `yamc.__commit_dirty__` is `True` if tracked files differed from that commit when it was built. Both are `None` for a build made outside a git checkout.

```bash
python -c "import yamc; print(yamc.__version__, yamc.__commit__, yamc.__commit_dirty__)"
```

A released wheel reports the version of its release. A build from a core source checkout (`pip install -e`, `maturin develop` or a `git+` install, such as the MPI builds below) reports `0.0.0.dev0`, and its commit identifies it instead.


## MPI Installation (Advanced)

YAMC can be built with MPI support for distributed-memory parallelism across multiple nodes, for example on large clusters.

MPI is optional, and the standard installation above is enough for most users. Install MPI only if you need:
- Multi-node parallelism (HPC clusters)
- Scaling beyond a single machine
- Hybrid MPI + threading parallelism

### Prerequisites

First, install an MPI implementation on your system:

#### Ubuntu/Debian

```bash
sudo apt-get update && sudo apt-get install -y \
    libopenmpi-dev openmpi-bin \
    build-essential clang libclang-dev
```

#### Fedora/RHEL/CentOS

```bash
sudo dnf install openmpi-devel
# or
sudo yum install openmpi-devel

# Load MPI module (if using environment modules)
module load mpi/openmpi-x86_64
```

#### macOS (Homebrew)

```bash
brew install open-mpi
```

#### Windows (Microsoft MPI)

1. Download and install [Microsoft MPI](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi)
2. Install both `msmpisetup.exe` (runtime) and `msmpisdk.msi` (SDK)

#### Verify MPI Installation

```bash
# Check MPI compiler is available
mpicc --version

# Test MPI works
mpirun -np 2 hostname
```

### Install YAMC with MPI

Building with MPI requires the Rust toolchain. To install Rust:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

Then build `yamc` from source, with the `mpi` feature switched on. `MATURIN_PEP517_ARGS` passes the flag to maturin, which builds the wheel for pip:

```bash
MATURIN_PEP517_ARGS="--features mpi" pip install "yamc @ git+https://github.com/fusion-neutronics/yamc-and-yani.git#subdirectory=packages/yamc"
```

This builds the main branch. To build a release instead, put its tag after `yamc-and-yani.git@`, as in `git+https://github.com/fusion-neutronics/yamc-and-yani.git@<tag>#subdirectory=packages/yamc`. The repository's [tags](https://github.com/fusion-neutronics/yamc-and-yani/tags) are named after the versions they release. A build from source reports version `0.0.0.dev0`; `yamc.__commit__` gives the commit it was built from (see [Check which build is installed](#check-which-build-is-installed)).

### Verify MPI Installation

```bash
# Check MPI is available in YAMC
python -c "import yamc; print(f'MPI enabled: {yamc.parallel.has_mpi()}')"
# Should print: MPI enabled: True

# Test with multiple ranks
mpirun -np 2 python -c "import yamc; print(f'Rank {yamc.parallel.mpi_rank()} of {yamc.parallel.mpi_size()}')"
# Should print, the two lines in either order:
# Rank 0 of 2
# Rank 1 of 2
```

## Docker Installation

### Standard Docker Image

```dockerfile
FROM python:3.11-slim

# cadquery, which the cad extra brings, needs the system OpenGL library to import
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "yamc[all]"

WORKDIR /workspace
CMD ["python"]
```

### Docker Image with MPI

```dockerfile
FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libopenmpi-dev openmpi-bin \
    build-essential clang libclang-dev curl git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Build yamc from source with MPI
RUN MATURIN_PEP517_ARGS="--features mpi" pip3 install \
    "yamc @ git+https://github.com/fusion-neutronics/yamc-and-yani.git#subdirectory=packages/yamc"

# The container runs as root, and OpenMPI refuses to start as root without these
ENV OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

WORKDIR /workspace
CMD ["python3"]
```

Build and run:

```bash
docker build -t yamc-mpi -f Dockerfile .
docker run --rm yamc-mpi mpirun -np 4 python3 -c "import yamc; print(yamc.parallel.mpi_rank())"
```

## HPC/Cluster Installation

For HPC environments with environment modules:

```bash
# Load required modules
module load python/3.11
module load openmpi/4.1.4

# Install Rust in your home directory: yamc needs the latest stable
# release, which cluster modules rarely carry
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Create virtual environment
python3 -m venv ~/.venvs/yamc-mpi
source ~/.venvs/yamc-mpi/bin/activate

# Build yamc with MPI
MATURIN_PEP517_ARGS="--features mpi" pip install "yamc @ git+https://github.com/fusion-neutronics/yamc-and-yani.git#subdirectory=packages/yamc"

# Test installation
mpirun -np 4 python -c "import yamc; print(f'Rank {yamc.parallel.mpi_rank()}')"
```

See [simulating](simulate_transport.md#running-the-simulation) for MPI usage, hybrid MPI+threading, and Slurm examples.
