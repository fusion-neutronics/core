"""Regression tests: Type 4 -- Material transmutation vs reference data."""

import os
import shutil
import tempfile
import warnings

import pytest
import yamc

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


def run_yamc_transmute(nuclide):
    """Run yamc material.transmute() for a single nuclide."""
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
    yamc.transmutation_decay_data = tmp_chain
    yamc.transmutation_reactions = tmp_chain
    yamc.transmutation_fission_yields = tmp_chain
    try:
        results = material.transmute(schedule=schedule)
        steps = []
        for mat in results.step_materials(material.id or 0):
            steps.append(dict(mat.nuclides))
        return {"steps": steps}
    finally:
        shutil.rmtree(tmp_chain, ignore_errors=True)


@pytest.mark.parametrize("nuclide", ALL_NEUTRON_NUCLIDES)
def test_transmutation(nuclide):
    """Test transmutation products match reference within 10%."""
    ref = load_reference("transmutation", nuclide)
    skip_if_no_reference(ref, "transmutation", nuclide)

    result = run_yamc_transmute(nuclide)

    density_threshold = 1e-14
    rel_tol = 0.5
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
        yamc_step = result["steps"][step_idx] if step_idx < len(result["steps"]) else {}
        ref_step = ref["steps"][step_idx] if step_idx < len(ref["steps"]) else {}

        all_products = set(yamc_step.keys()) | set(ref_step.keys())

        for product in sorted(all_products):
            y = yamc_step.get(product, 0.0)
            o = ref_step.get(product, 0.0)
            max_val = max(abs(y), abs(o))

            if max_val < density_threshold:
                continue

            if abs(y) < density_threshold or abs(o) < density_threshold:
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

            if rd > rel_tol:
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
            stacklevel=2,
        )

    assert len(failures) == 0, (
        f"{nuclide}: {len(failures)}/{n_compared} products exceed {rel_tol}% "
        f"(max={max_rel_diff:.2f}%):\n" + "\n".join(failures[:10])
    )
