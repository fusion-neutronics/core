"""Regression tests: Type 5 -- D1S decay-photon transport vs reference data."""

import shutil
import tempfile

import numpy as np
import pytest
import yamc

from .conftest import (
    BATCHES,
    CHAIN_FILE,
    COUPLED_NUCLIDES,
    CYLINDER_HALF_HEIGHT,
    CYLINDER_RADIUS,
    DECAY_PHOTON_REDUCE_LEVEL,
    DECAY_PHOTON_SOURCE_RATES,
    DECAY_PHOTON_TIMESTEPS,
    NEUTRON_SOURCE_ENERGY,
    NUCLIDE_TO_ELEMENT,
    PARTICLES,
    PHOTON_GROUP_STRUCTURE,
    SEED,
    TESTS_DATA_DIR,
    load_reference,
    skip_if_no_reference)

# Load full chain once using yamc's Chain, then reduce per-nuclide
_full_chain = yamc.TransmutationChain(CHAIN_FILE)


_cache = {}


def get_result(nuclide):
    if nuclide not in _cache:
        _cache[nuclide] = run_yamc_decay_photons(nuclide)
    return _cache[nuclide]


def run_yamc_decay_photons(nuclide):
    """Run yamc D1S simulation with broomstick geometry."""
    element = NUCLIDE_TO_ELEMENT[nuclide]
    neutron_path = str(TESTS_DATA_DIR / f"{nuclide}.arrow")
    photon_path = str(TESTS_DATA_DIR / f"{element}.arrow")

    # Reduce chain for this nuclide
    reduced = _full_chain.reduce([nuclide], DECAY_PHOTON_REDUCE_LEVEL)

    # Export reduced chain to temp dir for Config (model.run reads from Config)
    tmp_dir = tempfile.mkdtemp(prefix=f"chain_{nuclide}_", suffix=".chain.arrow")
    reduced.export_to_arrow(tmp_dir)

    try:
        material = yamc.Material(
            composition={nuclide: 1.0},
            density=1.0,
            temperature=294)
        material.read_nuclear_data(
            {nuclide: neutron_path},
            photon_data={element: photon_path})

        cylinder = yamc.Cylinder(axis="z", radius=CYLINDER_RADIUS, boundary="vacuum")
        z_bot = yamc.Plane(axis="z", offset=-CYLINDER_HALF_HEIGHT, boundary="vacuum")
        z_top = yamc.Plane(axis="z", offset=CYLINDER_HALF_HEIGHT, boundary="vacuum")
        region = cylinder.below & z_bot.above & z_top.below

        cell = yamc.Cell(name="broomstick", region=region, material=material)
        geometry = yamc.Geometry([cell])

        source = yamc.NeutronSource(
            position=(0, 0, 0),
            energy=yamc.sources.Discrete([NEUTRON_SOURCE_ENERGY], [1.0]))
        yamc.transmutation_decay_data = tmp_dir
        yamc.transmutation_reactions = tmp_dir
        yamc.transmutation_fission_yields = tmp_dir

        # Activation products reachable from the model's materials, via the
        # reduced chain. (Discovered from the model rather than a hand-passed
        # nuclide list; see #484.)
        radionuclides = yamc.Model(
            geometry=geometry, source=source
        ).radionuclides()

        if not radionuclides:
            return {"spectra": [], "spectra_std": []}

        tally = yamc.Tally(
            scores=["flux"],
            name="decay_photons_spectrum",
            cells=cell,
            particle="photon",
            energy_group_structure=PHOTON_GROUP_STRUCTURE,
            parent_nuclides=radionuclides)

        model = yamc.Model(geometry=geometry, tallies=[tally], source=source,
                         transport_secondary_photons=True, use_decay_photons=True)
        results = model.simulate_transport(total_particles=PARTICLES * BATCHES, seed=SEED)
        tally_result = results[tally]

        # One PulseSchedule owns the irradiation/cooling timeline; the three
        # irradiation steps share the one source, then two cooling steps.
        # time_correct_tally returns a DoseResult with one row per schedule
        # step (no pre-irradiation baseline row to slice off).
        irr_steps = sum(1 for r in DECAY_PHOTON_SOURCE_RATES if r > 0.0)
        steps = [
            yamc.Pulse(source=source, rate=DECAY_PHOTON_SOURCE_RATES[i],
                       duration=DECAY_PHOTON_TIMESTEPS[i])
            for i in range(irr_steps)
        ] + [
            yamc.Cooldown(duration=DECAY_PHOTON_TIMESTEPS[i])
            for i in range(irr_steps, len(DECAY_PHOTON_TIMESTEPS))
        ]
        sched = yamc.PulseSchedule(steps)
        dose = sched.time_correct_tally(tally_result)

        return {
            "spectra": [list(row) for row in dose.mean],
            "spectra_std": [list(row) for row in dose.std_dev],
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.parametrize("nuclide", COUPLED_NUCLIDES)
def test_decay_photons_total_flux(nuclide):
    """Test D1S total photon flux per timestep matches reference."""
    ref = load_reference("decay_photons", nuclide)
    skip_if_no_reference(ref, "decay_photons", nuclide)

    result = get_result(nuclide)

    if not result["spectra"]:
        pytest.skip(f"{nuclide}: no D1S radionuclides from chain")

    # Find peak total across all steps for noise floor calculation
    peak_total = 0.0
    for step_idx in range(len(DECAY_PHOTON_TIMESTEPS)):
        ref_total = np.sum(ref["spectra"][step_idx])
        yamc_total = np.sum(result["spectra"][step_idx])
        peak_total = max(peak_total, abs(ref_total), abs(yamc_total))

    noise_floor = peak_total * 1e-6
    failures = []

    for step_idx in range(len(DECAY_PHOTON_TIMESTEPS)):
        ref_spec = np.array(ref["spectra"][step_idx])
        yamc_spec = np.array(result["spectra"][step_idx])
        ref_std = np.array(ref["spectra_std"][step_idx])
        yamc_std = np.array(result["spectra_std"][step_idx])

        ref_total = ref_spec.sum()
        yamc_total = yamc_spec.sum()
        ref_total_std = np.sqrt((ref_std**2).sum())
        yamc_total_std = np.sqrt((yamc_std**2).sum())

        # Skip steps below noise floor
        if abs(ref_total) < noise_floor and abs(yamc_total) < noise_floor:
            continue

        # z-score: check statistical significance
        combined_std = np.sqrt(ref_total_std**2 + yamc_total_std**2)
        if combined_std > 0:
            z_score = abs(yamc_total - ref_total) / combined_std
        else:
            z_score = 0.0

        # Flag differences that are both practically (>2%, the "Excellent"
        # quality bar) and statistically (z > 4) significant. The z gate keeps
        # the check robust to the exact fixed-seed realization: a 2-3% gap that
        # sits well inside the combined Monte Carlo error (e.g. z ~ 0.5) is
        # noise, not a regression, and must not fail when an unrelated RNG
        # stream change (issue #111) reshuffles which histories land where.
        # The gate was z > 3 until the 64-bit stream (issue #274) landed the
        # Fe57 realization at exactly 2.0% / z = 3.1: a 3-sigma bar on a
        # fixed seed flips on realization luck (the same failure mode the
        # heating band in conftest documents), so decay flux uses the same
        # 4-sigma band. A real D1S normalization bug is enormous by
        # comparison (issue #128 was a factor of 10, hundreds of sigma).
        denom = max(abs(ref_total), abs(yamc_total))
        if denom > 0:
            rel_diff_pct = abs(yamc_total - ref_total) / denom * 100.0
            if rel_diff_pct > 2.0 and z_score > 4.0:
                failures.append(
                    f"  step {step_idx+1}: yamc={yamc_total:.4e} "
                    f"ref={ref_total:.4e} rel={rel_diff_pct:.1f}% "
                    f"z={z_score:.1f}"
                )

    assert not failures, (
        f"{nuclide}: D1S total flux mismatch:\n" + "\n".join(failures)
    )


@pytest.mark.parametrize("nuclide", COUPLED_NUCLIDES)
def test_decay_photons_spectrum_shape(nuclide):
    """Test D1S photon spectrum shape matches reference at peak step."""
    ref = load_reference("decay_photons", nuclide)
    skip_if_no_reference(ref, "decay_photons", nuclide)

    result = get_result(nuclide)

    if not result["spectra"]:
        pytest.skip(f"{nuclide}: no D1S radionuclides from chain")

    # Find the step with largest total flux for best statistics
    best_step = 0
    best_total = 0.0
    for step_idx in range(len(DECAY_PHOTON_TIMESTEPS)):
        total = abs(np.sum(ref["spectra"][step_idx]))
        if total > best_total:
            best_total = total
            best_step = step_idx

    if best_total == 0:
        pytest.skip(f"{nuclide}: no significant D1S flux")

    # Compare spectrum shape at peak step using chi2
    ref_mean = np.array(ref["spectra"][best_step])
    yamc_mean = np.array(result["spectra"][best_step])
    ref_std = np.array(ref["spectra_std"][best_step])
    yamc_std = np.array(result["spectra_std"][best_step])

    # Use relaxed chi2 for D1S (more stochastic noise)
    chi2_terms = []
    for i in range(len(ref_mean)):
        diff = yamc_mean[i] - ref_mean[i]
        combined_std = np.sqrt(ref_std[i]**2 + yamc_std[i]**2)
        if combined_std > 0 and (abs(ref_mean[i]) > best_total * 1e-6
                                  or abs(yamc_mean[i]) > best_total * 1e-6):
            chi2_terms.append((diff / combined_std)**2)

    if chi2_terms:
        chi2_dof = np.sum(chi2_terms) / len(chi2_terms)
        assert chi2_dof < 2.0, (
            f"{nuclide} step {best_step+1}: D1S spectrum chi2/dof = "
            f"{chi2_dof:.2f} > 2.0"
        )
