"""Regression tests: Type 4 -- Material transmutation vs reference data.

The references are OpenMC depletions, whose chain has no MT=5 (n,X). The
comparison with them runs twice: like for like, on a reactions subsection
without the (n,X) rows, and on the full chain against the reference plus what
MT=5 makes, computed from the data (see ``_mt5``).
"""

import os
import shutil
import tempfile
import warnings

import pytest
import yamc

from . import _mt5
from .conftest import (
    ALL_NEUTRON_NUCLIDES,
    CHAIN_FILE,
    TRANSMUTE_ENERGY_GROUPS,
    TRANSMUTE_MULTIGROUP_FLUX,
    TRANSMUTE_SOURCE_RATES,
    TRANSMUTE_TIMESTEPS,
    NEUTRON_DATA_DIR,
    TESTS_DATA_DIR,
    load_reference,
    skip_if_no_reference,
)

# Load chain once as a yamc Chain object (avoids re-parsing XML per nuclide)
_chain = yamc.TransmutationChain(CHAIN_FILE)

# With the (n,X) rows, a nuclide whose evaluation's MT=5 the solver cannot
# carry is refused, and the refusal is what is checked. ENDF/B-VIII.1's Co58
# and Fe58 give MT=5's light particles but no residual nucleus; its Cr52 gives
# residual "multiplicities" (V51 at 4e4) that restate its (n,np) over a
# vanishing MT=5.
FULL_CHAIN_REFUSALS = {
    "Co58": r"MT=5 \(n,anything\) whose residual nuclei the chain does not model",
    "Fe58": r"MT=5 \(n,anything\) whose residual nuclei the chain does not model",
    "Cr52": r"the branching cannot be applied to this spectrum",
}

DENSITY_THRESHOLD = 1e-14
REL_TOL = 0.5  # percent


def run_yamc_transmute(nuclide, with_anything=True):
    """Run yamc material.transmute() for a single nuclide.

    ``with_anything=False`` takes the MT=5 (n,X) rows out of the reduced
    chain's reactions subsection before the solve, which leaves the reactions
    the reference chain carries. With them, the MT=5 production rates the data
    gives are returned beside the inventories.
    """
    neutron_path = str(TESTS_DATA_DIR / f"{nuclide}.arrow")

    # Load cross sections for daughter nuclides (like verify_all.py)
    if os.path.isdir(NEUTRON_DATA_DIR):
        yamc.cross_section_data = NEUTRON_DATA_DIR

    material = yamc.Material(
        composition={nuclide: 1.0},
        density=1.0,
        name=nuclide,
        volume=1.0,
        temperature=294,
    )
    material.read_nuclear_data({nuclide: neutron_path})

    # The spectrum rides on the pulses via a Histogram-energy NeutronSource; the
    # pulse rate is the flux magnitude. The old per-step multiplier `sr` maps to
    # rate = sr * sum(flux) (Histogram normalizes the shape), and sr == 0 maps to
    # a decay-only Cooldown -- equivalent to the previous (flux, multiplier) form.
    spectrum = yamc.NeutronSource(
        energy=yamc.sources.Histogram(TRANSMUTE_ENERGY_GROUPS, TRANSMUTE_MULTIGROUP_FLUX)
    )
    total_flux = sum(TRANSMUTE_MULTIGROUP_FLUX)
    schedule = yamc.PulseSchedule([
        yamc.Pulse(rate=sr * total_flux, duration=dt, source=spectrum)
        if sr > 0
        else yamc.Cooldown(duration=dt)
        for dt, sr in zip(TRANSMUTE_TIMESTEPS, TRANSMUTE_SOURCE_RATES)
    ])
    # Reduce the chain to the network reachable from this seed before
    # transmuting. Nuclides unreachable from a pure-`nuclide` start stay at
    # zero density throughout the Bateman solve, so the per-step results are
    # unchanged; but reducing avoids loading neutron cross sections for all
    # ~3800 chain nuclides (which needs >14 GB of RAM), keeping the test to the
    # few hundred actually reachable.
    reduced = _chain.reduce([nuclide], level=20)
    # No per-call chain argument any more: export the reduced chain and point
    # the transmutation subsection settings at it.
    tmp_chain = tempfile.mkdtemp(prefix=f"chain_{nuclide}_", suffix=".chain.arrow")
    reduced.export_to_arrow(tmp_chain)
    if not with_anything:
        _mt5.write_without_anything(tmp_chain)
    yamc.transmutation_decay_data = tmp_chain
    yamc.transmutation_reactions = tmp_chain
    yamc.transmutation_fission_yields = tmp_chain
    try:
        results = material.transmute(schedule=schedule)
        steps = []
        for mat in results.step_materials(material.id or 0):
            steps.append(dict(mat.nuclides))
        rates = None
        if with_anything:
            rates = _mt5.mt5_production_rates(
                neutron_path,
                tmp_chain,
                nuclide,
                TRANSMUTE_ENERGY_GROUPS,
                TRANSMUTE_MULTIGROUP_FLUX,
            )
        return {"steps": steps, "mt5_rates": rates, "chain": reduced}
    finally:
        shutil.rmtree(tmp_chain, ignore_errors=True)


def compare_steps(nuclide, yamc_steps, expected_steps, skipped=None):
    """Compare yamc's inventories with the expected ones, step by step, at
    REL_TOL.

    Returns ``(failures, n_compared, max_rel_diff)``. A product in
    ``skipped[step]`` is not compared at that step.
    """
    max_rel_diff = 0.0
    n_compared = 0
    failures = []
    # A product one side has and the other does not is the most severe
    # disagreement there is, so it belongs in `failures` -- but only when yamc
    # had the cross sections needed to produce it at all. Without
    # NEUTRON_DATA_DIR only the seed nuclide's own data is loaded, so chain
    # products legitimately come out absent and calling that a physics failure
    # would just be reporting an uninstalled data set. Track them either way so
    # the reduced coverage is visible instead of silent.
    have_daughter_data = os.path.isdir(NEUTRON_DATA_DIR)
    unproducible = []

    for step_idx in range(len(TRANSMUTE_TIMESTEPS)):
        yamc_step = yamc_steps[step_idx] if step_idx < len(yamc_steps) else {}
        ref_step = expected_steps[step_idx] if step_idx < len(expected_steps) else {}
        skip = skipped[step_idx] if skipped else set()

        all_products = (set(yamc_step.keys()) | set(ref_step.keys())) - skip

        for product in sorted(all_products):
            y = yamc_step.get(product, 0.0)
            o = ref_step.get(product, 0.0)
            max_val = max(abs(y), abs(o))

            if max_val < DENSITY_THRESHOLD:
                continue

            if abs(y) < DENSITY_THRESHOLD or abs(o) < DENSITY_THRESHOLD:
                entry = (
                    f"  step {step_idx+1} {product}: yamc={y:.4e} ref={o:.4e} "
                    f"(present on one side only)"
                )
                if have_daughter_data:
                    n_compared += 1
                    max_rel_diff = 100.0
                    failures.append(entry)
                else:
                    unproducible.append(entry)
                continue

            rd = abs(y - o) / max_val * 100.0
            n_compared += 1
            max_rel_diff = max(max_rel_diff, rd)

            if rd > REL_TOL:
                failures.append(
                    f"  step {step_idx+1} {product}: yamc={y:.4e} ref={o:.4e} "
                    f"rel_diff={rd:.2f}%"
                )

    assert n_compared > 0, f"{nuclide}: no products to compare"

    if unproducible:
        warnings.warn(
            f"{nuclide}: {len(unproducible)} of {n_compared + len(unproducible)} "
            f"product/step pairs not checked -- the reference has them but this "
            f"run had no cross sections to produce them. Set YAMC_NEUTRON_DATA_DIR "
            f"(or install {NEUTRON_DATA_DIR}) to check them:\n"
            + "\n".join(unproducible[:10]),
            stacklevel=3,
        )
    return failures, n_compared, max_rel_diff


@pytest.mark.parametrize("nuclide", ALL_NEUTRON_NUCLIDES)
def test_transmutation(nuclide):
    """Like for like: yamc without MT=5 (n,X) against the reference within 0.5%."""
    ref = load_reference("transmutation", nuclide)
    skip_if_no_reference(ref, "transmutation", nuclide)

    result = run_yamc_transmute(nuclide, with_anything=False)
    failures, n_compared, max_rel_diff = compare_steps(nuclide, result["steps"], ref["steps"])
    assert len(failures) == 0, (
        f"{nuclide}: {len(failures)}/{n_compared} products exceed {REL_TOL}% "
        f"(max={max_rel_diff:.2f}%):\n" + "\n".join(failures[:10])
    )


@pytest.mark.parametrize("nuclide", ALL_NEUTRON_NUCLIDES)
def test_transmutation_with_mt5(nuclide):
    """The full chain against the reference plus what MT=5 makes, within 0.5%.

    What MT=5 adds is computed from the data, not by yamc (see ``_mt5``), so
    this checks yamc's MT=5 residuals and light particles product by product.
    A nuclide whose MT=5 the solver refuses is checked for the refusal.
    """
    ref = load_reference("transmutation", nuclide)
    skip_if_no_reference(ref, "transmutation", nuclide)

    if nuclide in FULL_CHAIN_REFUSALS:
        with pytest.raises(ValueError, match=FULL_CHAIN_REFUSALS[nuclide]):
            run_yamc_transmute(nuclide)
        return

    result = run_yamc_transmute(nuclide)
    chain = result["chain"]
    # The seed's density, taken after the first step, which removes less than
    # 1e-6 of it.
    n0 = result["steps"][0][nuclide]
    steps = [(dt, sr > 0) for dt, sr in zip(TRANSMUTE_TIMESTEPS, TRANSMUTE_SOURCE_RATES)]
    added, unpredicted = _mt5.predicted_mt5_densities(
        result["mt5_rates"], n0, steps, chain.half_lives, chain.decays
    )
    expected = []
    for step_idx, ref_step in enumerate(ref["steps"]):
        step = dict(ref_step)
        for product, n in added[step_idx].items():
            step[product] = step.get(product, 0.0) + n
        expected.append(step)

    failures, n_compared, max_rel_diff = compare_steps(
        nuclide, result["steps"], expected, skipped=unpredicted
    )
    assert len(failures) == 0, (
        f"{nuclide}: {len(failures)}/{n_compared} products exceed {REL_TOL}% of the "
        f"reference plus MT=5 (max={max_rel_diff:.2f}%):\n" + "\n".join(failures[:10])
    )
