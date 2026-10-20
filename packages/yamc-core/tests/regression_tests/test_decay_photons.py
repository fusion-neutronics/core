"""Regression tests: Type 5 -- D1S decay-photon transport vs reference data."""

import shutil
import tempfile

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
    assert_self_reference,
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
        # nuclide list.)
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

    assert_self_reference(
        [sum(spectrum) for spectrum in ref["spectra"]],
        [sum(spectrum) for spectrum in result["spectra"]],
        f"{nuclide}/decay-photon totals",
    )


@pytest.mark.parametrize("nuclide", COUPLED_NUCLIDES)
def test_decay_photons_spectrum_shape(nuclide):
    """Every D1S spectrum bin and uncertainty retains its same-seed answer."""
    ref = load_reference("decay_photons", nuclide)
    skip_if_no_reference(ref, "decay_photons", nuclide)

    result = get_result(nuclide)

    for field in ("spectra", "spectra_std"):
        assert_self_reference(ref[field], result[field], f"{nuclide}/{field}")
