"""Shared fixtures and helpers for regression tests."""

import json
from pathlib import Path

import numpy as np
import pytest

# Constants that MUST stay identical to generate_reference_data.py live in
# _config.py and are imported by both files.
from ._config import (  # noqa: F401
    ALL_NEUTRON_NUCLIDES,
    BATCHES,
    COUPLED_NUCLIDES,
    CYLINDER_HALF_HEIGHT,
    CYLINDER_RADIUS,
    DECAY_PHOTON_REDUCE_LEVEL,
    DECAY_PHOTON_SOURCE_RATES,
    DECAY_PHOTON_TIMESTEPS,
    TRANSMUTE_ENERGY_GROUPS,
    TRANSMUTE_MULTIGROUP_FLUX,
    TRANSMUTE_SOURCE_RATES,
    TRANSMUTE_TIMESTEPS,
    NEUTRON_DATA_DIR,
    NEUTRON_GROUP_STRUCTURE,
    NEUTRON_SOURCE_ENERGY,
    NUCLIDE_TO_ELEMENT,
    PARTICLES,
    PHOTON_GROUP_STRUCTURE,
    PHOTON_SCALAR_SCORES,
    SEED,
)

# Directories
REGRESSION_DIR = Path(__file__).parent
REFERENCE_DIR = REGRESSION_DIR / "reference_data"
# Resolved from this file, not from the CWD. `test_transmutation.py` and
# `test_decay_photons.py` build a TransmutationChain at module scope, so a
# CWD-relative path here fails at COLLECTION, before any fixture can run, and
# `cd packages/yamc-core && pytest` dies at import (issue #539). The repo root
# is four levels up: regression_tests -> tests -> yamc-core -> packages.
REPO_ROOT = Path(__file__).resolve().parents[4]
TESTS_DATA_DIR = REPO_ROOT / "tests"  # symlink to crates/yamc/tests

# Source energies (photon source energy is conftest-only)
PHOTON_SOURCE_ENERGY = 1e6

# Scores to test (excluding heating-local which yamc doesn't implement).
# NOTE: intentionally differs from generate_reference_data.py, which includes
# 'heating-local'. Do not share this list.
NEUTRON_SCALAR_SCORES = [
    "flux", "heating", "total", "absorption",
    "H1-production", "H2-production", "H3-production",
    "He3-production", "He4-production",
]

# Scores with known systematic differences (KERMA, etc.)
# These get a wider tolerance
# Scores held to the wide (10 %, 4 sigma) band instead of the tight (5 %,
# 3 sigma) one. heating / heating-local: KERMA convention differences (see
# assert_scalar_agreement). photoelectric: yamc carries a known +5-7 %
# photoionization-rate residual against the OpenMC-derived references
# (data-convention difference, identical on the CPU and GPU backends and
# confirmed against a pre-change build by the V&V suite); under the old
# 32-bit RNG stream the Fe56 coupled realization sat just under the tight
# band by luck, and the 64-bit stream (issue #274) tipped it to 3.4 sigma /
# 6.6 %.
WIDE_TOLERANCE_SCORES = {"heating", "heating-local", "photoelectric"}

# Transmutation chain (through the repo-root "tests/" symlink; absolute for the
# same collection-time reason as TESTS_DATA_DIR above).
CHAIN_FILE = str(TESTS_DATA_DIR / "transmutation-endf-b8.1-sfr.arrow")


def fixture_library(nuclide):
    """Nuclear-data library the yamc fixture was converted from
    (``crates/yamc/tests/<nuclide>.arrow/version.json``), or None if the
    fixture carries no provenance."""
    path = TESTS_DATA_DIR / f"{nuclide}.arrow" / "version.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f).get("library")


def canonical_library(label):
    """Library label with punctuation and case removed, or None.

    Converter versions have spelled the same library both ``endfb-8.1`` and
    ``endf-b8.1``; the release digits are what the guard below actually needs to
    compare, so normalise everything else away (``endf-b8.0`` still differs
    from ``endf-b8.1``).
    """
    if label is None:
        return None
    return label.lower().replace("-", "").replace("_", "").replace(" ", "")


def load_reference(sim_type, nuclide):
    """Load reference data JSON. Returns None if not found.

    Fails loudly when the reference was generated from a different
    nuclear-data library than the yamc fixture (issue #363): evaluations
    shift between releases (Li6 photon production moves ~100x between
    ENDF/B-VIII.0 and VIII.1), so comparing across libraries produces
    failures that look like transport bugs.
    """
    path = REFERENCE_DIR / f"{sim_type}_{nuclide}.json"
    if not path.exists():
        return None
    with open(path) as f:
        ref = json.load(f)
    ref_lib = canonical_library(ref.get("library"))
    fix_lib = canonical_library(fixture_library(nuclide))
    if ref_lib is not None and fix_lib is not None and ref_lib != fix_lib:
        pytest.fail(
            f"{path.name} was generated from '{ref_lib}' but the yamc "
            f"fixture {nuclide}.arrow is '{fix_lib}'. Regenerate the "
            f"reference with matching data "
            f"(using matching {fix_lib} data)."
        )
    return ref


def assert_scalar_agreement(ref_val, yamc_val, ref_std, yamc_std, name,
                            rel_tol=0.05, abs_tol=1e-10, flux_ref=None):
    """Assert two scalar tally values agree within tolerance.

    Default rel_tol=0.05 (5%) targets "Excellent" agreement with reference.

    Parameters
    ----------
    flux_ref : float, optional
        Reference flux value. If provided, scores smaller than 1e-3 * flux_ref
        are compared with relaxed absolute tolerance.
    """
    # Determine if this is a heating score (wider tolerance). Heating is the
    # noisiest score: for a light nuclide that produces almost no photons (e.g.
    # Li6, whose photon flux is ~1e-6 of its neutron flux) the reference photon
    # heating can be an exact 0 +/- 0 while yamc's own estimate carries ~30%
    # relative error. The comparison then reduces to "is yamc consistent with
    # zero", governed entirely by yamc's std, so a 3-sigma band is too tight to
    # be stable across fixed-seed realizations (an RNG stream change like issue
    # #111 reshuffles histories and tips a 3.0-sigma point to 3.1). Use a wider
    # 4-sigma statistical band for heating; a real KERMA bias on a
    # well-sampled nuclide is many sigma and is still caught.
    score_name = name.split("/")[-1] if "/" in name else name
    # Coupled tests label scores with a particle prefix (p_heating, n_flux);
    # strip it so the score-class checks below match (previously "p_heating"
    # silently missed WIDE_TOLERANCE_SCORES and was held to the tight neutron
    # tolerance).
    if score_name.startswith(("p_", "n_")):
        score_name = score_name[2:]
    std_factor = 3.0
    if score_name in WIDE_TOLERANCE_SCORES:
        # 10% for heating (KERMA differences) and photoelectric (known
        # photoionization-rate residual vs the OpenMC references).
        rel_tol = max(rel_tol, 0.10)
        std_factor = 4.0

    # For near-zero production scores, use flux-based absolute tolerance
    if flux_ref is not None and flux_ref > 0:
        score_magnitude = max(abs(ref_val), abs(yamc_val))
        if score_magnitude < 1e-3 * flux_ref:
            abs_tol = max(abs_tol, 1e-3 * flux_ref)

    combined_std = np.sqrt(ref_std**2 + yamc_std**2)
    tolerance = max(abs(ref_val) * rel_tol, std_factor * combined_std, abs_tol)
    diff = abs(yamc_val - ref_val)
    assert diff <= tolerance, (
        f"{name}: yamc={yamc_val:.6e} vs ref={ref_val:.6e}, "
        f"diff={diff:.6e} > tol={tolerance:.6e} "
        f"(rel_tol={rel_tol}, {std_factor}*std={std_factor * combined_std:.6e})"
    )


def assert_spectrum_agreement(ref_mean, yamc_mean, ref_std, yamc_std,
                              max_chi2_dof=1.5):
    """Assert spectral agreement using chi2/dof metric.

    Default threshold is 1.5 targeting "Excellent" agreement with reference.
    """
    ref_mean = np.array(ref_mean)
    yamc_mean = np.array(yamc_mean)
    ref_std = np.array(ref_std)
    yamc_std = np.array(yamc_std)

    # Only compare bins where both codes have significant flux
    max_flux = max(
        np.max(np.abs(ref_mean)) if len(ref_mean) > 0 else 0,
        np.max(np.abs(yamc_mean)) if len(yamc_mean) > 0 else 0,
    )
    threshold = max_flux * 1e-6

    chi2_terms = []
    for i in range(len(ref_mean)):
        if abs(ref_mean[i]) < threshold and abs(yamc_mean[i]) < threshold:
            continue
        diff = yamc_mean[i] - ref_mean[i]
        combined_std = np.sqrt(ref_std[i]**2 + yamc_std[i]**2)
        if combined_std > 0:
            chi2_terms.append((diff / combined_std)**2)

    if not chi2_terms:
        return

    chi2 = np.sum(chi2_terms)
    dof = len(chi2_terms)
    chi2_dof = chi2 / dof

    assert chi2_dof < max_chi2_dof, (
        f"Spectrum chi2/dof = {chi2_dof:.2f} > {max_chi2_dof} "
        f"(dof={dof})"
    )


def skip_if_no_reference(ref, sim_type, nuclide):
    """Skip test if reference data doesn't exist."""
    if ref is None:
        pytest.skip(
            f"No reference data: {sim_type}_{nuclide}.json "
            f"(run generate_reference_data.py)"
        )
