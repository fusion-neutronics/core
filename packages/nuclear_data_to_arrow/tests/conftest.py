"""Shared test fixtures for nuclear_data_to_arrow tests."""

import gzip
import lzma
import os
from pathlib import Path

import pytest

# Extra directories to search for bulk nuclear data that is too large to vendor.
# Every test that needs one of these skips when it is absent, so the suite is
# green with none of them set; they only widen coverage on a machine that
# happens to hold a full library.
#
# Set NUCLEAR_DATA_DIRS to a colon-separated list of directories, or point
# OPENMC_CROSS_SECTIONS at a cross_sections.xml and its neutron/ and photon/
# siblings are searched too.
_SEARCH_DIRS = [
    Path(p) for p in os.environ.get("NUCLEAR_DATA_DIRS", "").split(os.pathsep) if p
]


# Nuclear data committed under tests/data so the tests are hermetic: an
# ENDF/B-VIII.1 photoatomic and atomic relaxation pair for iron, and an ACE
# table for Li6. Iron covers 10 subshells, 7 of them with relaxation
# transitions, which exercises the subshell table and the Compton subshell map.
_DATA_DIR = Path(__file__).parent / "data"


def _find_hdf5_file(name):
    """Search common locations for an HDF5 nuclear data file."""
    # Check environment variable first
    if os.environ.get("OPENMC_CROSS_SECTIONS"):
        xs_xml = Path(os.environ["OPENMC_CROSS_SECTIONS"])
        if xs_xml.exists():
            lib_root = xs_xml.parent
            for sub in ["neutron", "photon", ""]:
                candidate = lib_root / sub / name
                if candidate.exists():
                    return candidate

    # Check known directories
    for d in _SEARCH_DIRS:
        candidate = d / name
        if candidate.exists():
            return candidate

    return None


@pytest.fixture(scope="session")
def li6_ace_path(tmp_path_factory):
    """The vendored Li6 ACE table, decompressed to a temporary file."""
    path = tmp_path_factory.mktemp("ace") / "Li6.ace"
    with gzip.open(_DATA_DIR / "Li6.ace.gz", "rb") as src:
        path.write_bytes(src.read())
    return path


@pytest.fixture(scope="session")
def neutron_data(tmp_path_factory):
    """An ``endf.IncidentNeutron`` parsed from a vendored ACE table.

    Li6 is the smallest table that still exercises the interesting parts of the
    format: elastic and level inelastic scattering, Kalbach-Mann and continuous
    tabular distributions, discrete photon production and mixtures with discrete
    lines. It has no unresolved resonance region and no fission, which the
    equivalence runs in issue #19 cover on Fe56 and U235 instead.
    """
    import endf
    path = tmp_path_factory.mktemp("ace") / "Li6.ace"
    with gzip.open(_DATA_DIR / "Li6.ace.gz", "rb") as src:
        path.write_bytes(src.read())
    return endf.IncidentNeutron.from_ace(endf.ace.get_tables(path)[0])


def _find_ace_file(names):
    """Search the known data directories for one of the given ACE filenames."""
    for d in _SEARCH_DIRS + [_DATA_DIR]:
        for name in names:
            candidate = d / name
            if candidate.exists():
                return candidate
    return None


@pytest.fixture(scope="session")
def fissile_data():
    """An ``endf.IncidentNeutron`` for a fissile nuclide, if one is available.

    Fissile ACE tables are far too large to vendor (U235 is 31 MB), so this
    skips unless one is present locally.
    """
    import endf
    p = _find_ace_file(["U235.ace", "U238.ace", "Pu239.ace"])
    if p is None:
        pytest.skip("No fissile ACE table found; place e.g. U235.ace in tests/data")
    return endf.IncidentNeutron.from_ace(endf.ace.get_tables(p)[0])


@pytest.fixture
def photon_h5_path():
    """Path to a photon HDF5 file for testing."""
    for name in ["Fe.h5", "Li.h5", "Be.h5", "H.h5"]:
        p = _find_hdf5_file(name)
        if p is not None:
            return p
    pytest.skip("No photon HDF5 file found; set OPENMC_CROSS_SECTIONS")


@pytest.fixture(scope="session")
def photon_endf_paths(tmp_path_factory):
    """The vendored Fe photoatomic/relaxation ENDF pair, decompressed.

    Vendored as .xz (ENDF's fixed-column text compresses ~4x) so the fixture
    is a binary blob rather than ~9,000 lines of diff noise in the repo.
    """
    tmp = tmp_path_factory.mktemp("photon-endf")
    paths = []
    for name in ("photoat-026_Fe_000.endf", "atom-026_Fe_000.endf"):
        dest = tmp / name
        with lzma.open(_DATA_DIR / f"{name}.xz", "rb") as src:
            dest.write_bytes(src.read())
        paths.append(dest)
    return tuple(paths)


@pytest.fixture(scope="session")
def photon_data(photon_endf_paths):
    """An ``endf.IncidentPhoton`` parsed from vendored ENDF evaluations."""
    import endf
    photoat_path, atom_path = photon_endf_paths
    return endf.IncidentPhoton.from_endf(photoat_path, atom_path)
