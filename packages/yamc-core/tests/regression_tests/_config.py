"""Shared constants for regression tests.

These values MUST match between the test fixtures (conftest.py) and the
reference-data generator (generate_reference_data.py). They live here so the
two stay in sync automatically.

This module is imported two ways:
  * conftest.py imports it as part of the package (relative import).
  * generate_reference_data.py is run as a *script*, so it adds this
    directory to sys.path and does ``import _config``.

Deliberately NOT shared (kept per-file because they legitimately differ):
  * NEUTRON_SCALAR_SCORES -- the generator includes 'heating-local' (a reference-code
    score) while conftest omits it because yamc lacks that score.
  * Chain/data paths -- conftest uses a "tests/" relative path; the generator
    resolves an absolute Arrow path.
  * reference HDF5 paths, REFERENCE_DIR mkdir, tolerance/source-energy helpers.
"""

import os

# Simulation parameters
PARTICLES = 500_000
BATCHES = 10
SEED = 1

# Geometry
CYLINDER_RADIUS = 1.0
CYLINDER_HALF_HEIGHT = 50.0

# Source energy
NEUTRON_SOURCE_ENERGY = 14.06e6

# Photon scalar scores (identical in both files)
PHOTON_SCALAR_SCORES = [
    "flux", "coherent-scatter", "incoherent-scatter",
    "photoelectric", "pair-production", "heating",
]

# Group structures
NEUTRON_GROUP_STRUCTURE = "VITAMIN-J-175"
PHOTON_GROUP_STRUCTURE = "VITAMIN-J-42"

# System-wide neutron data for transmutation daughter cross sections
NEUTRON_DATA_DIR = os.environ.get(
    "YAMC_NEUTRON_DATA_DIR",
    os.path.expanduser("~/nuclear_data/endf-b8.0-arrow/neutron"),
)

# Transmutation parameters
TRANSMUTE_ENERGY_GROUPS = [1e-5, 0.625, 1e5, 2e7]
TRANSMUTE_MULTIGROUP_FLUX = [1e12, 5e12, 1e14]
TRANSMUTE_TIMESTEPS = [3600.0, 3600.0, 3600.0, 15 * 86400.0, 15 * 86400.0]
TRANSMUTE_SOURCE_RATES = [1.0, 1.0, 1.0, 0.0, 0.0]

# D1S parameters
DECAY_PHOTON_TIMESTEPS = [3600.0, 3600.0, 3600.0, 15 * 86400.0, 15 * 86400.0]
DECAY_PHOTON_SOURCE_RATES = [1.0, 1.0, 1.0, 0.0, 0.0]
DECAY_PHOTON_REDUCE_LEVEL = 5

# Nuclide -> element mapping for photon data
NUCLIDE_TO_ELEMENT = {
    "Be9": "Be", "Fe54": "Fe", "Fe56": "Fe", "Fe57": "Fe", "Fe58": "Fe",
    "Li6": "Li", "Li7": "Li",
}

# All neutron isotopes available in tests/
ALL_NEUTRON_NUCLIDES = [
    "Al27", "Be9", "Co58", "Cr52", "Fe54", "Fe56", "Fe57", "Fe58",
    "Li6", "Li7", "Pb208",
]

# Nuclides with photon element data (for coupled and D1S)
COUPLED_NUCLIDES = [n for n in ALL_NEUTRON_NUCLIDES if n in NUCLIDE_TO_ELEMENT]
