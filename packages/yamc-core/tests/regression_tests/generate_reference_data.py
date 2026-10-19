#!/usr/bin/env python3
"""Regenerate fixed-seed yamc regression references from the test fixtures.

Run from any directory:
    python generate_reference_data.py [--types neutron coupled decay_photons transmutation]
                                      [--nuclides Fe56 Li6]

These values detect changes in yamc, not agreement with another code. A PR
that updates them must explain the physics or nuclear-data change. Independent
transport-code comparisons belong in the verification and validation repos.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yamc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regression_tests import _config  # noqa: E402
from regression_tests.conftest import (  # noqa: E402
    REFERENCE_DIR,
    REPO_ROOT,
    fixture_library,
)


def gen_neutron(nuclide):
    from regression_tests.test_neutron_transport import run_yamc_neutron

    return run_yamc_neutron(nuclide)


def gen_coupled(nuclide):
    from regression_tests.test_coupled_transport import run_yamc_coupled

    return run_yamc_coupled(nuclide)


def gen_decay_photons(nuclide):
    from regression_tests.test_decay_photons import run_yamc_decay_photons

    return run_yamc_decay_photons(nuclide)


def gen_transmutation(nuclide):
    from regression_tests.test_transmutation import (
        FULL_CHAIN_REFUSALS,
        run_yamc_transmute,
    )

    if nuclide in FULL_CHAIN_REFUSALS:
        return {"expected_error": FULL_CHAIN_REFUSALS[nuclide]}
    result = run_yamc_transmute(nuclide)
    return {
        "steps": result["steps"],
        "timesteps": _config.TRANSMUTE_TIMESTEPS,
        "source_rates": _config.TRANSMUTE_SOURCE_RATES,
    }


GENERATORS = {
    "neutron": (gen_neutron, _config.ALL_NEUTRON_NUCLIDES),
    "coupled": (gen_coupled, _config.COUPLED_NUCLIDES),
    "decay_photons": (gen_decay_photons, _config.COUPLED_NUCLIDES),
    "transmutation": (gen_transmutation, _config.ALL_NEUTRON_NUCLIDES),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--types", nargs="+", default=list(GENERATORS), choices=list(GENERATORS))
    parser.add_argument("--nuclides", nargs="+", choices=_config.ALL_NEUTRON_NUCLIDES)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    REFERENCE_DIR.mkdir(exist_ok=True)
    for sim_type in args.types:
        generate, nuclides = GENERATORS[sim_type]
        if args.nuclides:
            nuclides = [nuclide for nuclide in nuclides if nuclide in args.nuclides]
        for nuclide in nuclides:
            print(f"[{sim_type}] {nuclide} ...", flush=True)
            data = generate(nuclide)
            data.update({
                "generator": "yamc",
                "version": yamc.__version__,
                "source_commit": commit,
                "nuclide": nuclide,
                "particles": _config.PARTICLES,
                "batches": _config.BATCHES,
                "seed": _config.SEED,
                "library": fixture_library(nuclide),
            })
            path = REFERENCE_DIR / f"{sim_type}_{nuclide}.json"
            path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
            print(f"[{sim_type}] {nuclide} -> {path.name}", flush=True)


if __name__ == "__main__":
    main()
