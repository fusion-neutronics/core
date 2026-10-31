"""Regression tests: Type 2 -- Coupled neutron-photon transport vs reference data."""

import pytest
import yamc

from .conftest import (
    BATCHES,
    COUPLED_NUCLIDES,
    CYLINDER_HALF_HEIGHT,
    CYLINDER_RADIUS,
    NEUTRON_GROUP_STRUCTURE,
    NEUTRON_SCALAR_SCORES,
    NEUTRON_SOURCE_ENERGY,
    NUCLIDE_TO_ELEMENT,
    PARTICLES,
    PHOTON_GROUP_STRUCTURE,
    PHOTON_SCALAR_SCORES,
    SEED,
    TESTS_DATA_DIR,
    assert_self_reference,
    load_reference,
    skip_if_no_reference,
)


_cache = {}


def get_result(nuclide):
    if nuclide not in _cache:
        _cache[nuclide] = run_yamc_coupled(nuclide)
    return _cache[nuclide]


def run_yamc_coupled(nuclide):
    """Run yamc coupled neutron-photon broomstick simulation."""
    element = NUCLIDE_TO_ELEMENT[nuclide]
    neutron_path = str(TESTS_DATA_DIR / f"{nuclide}.arrow")
    photon_path = str(TESTS_DATA_DIR / f"{element}.arrow")

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
    tallies = []

    for score in NEUTRON_SCALAR_SCORES:
        t = yamc.Tally(
            scores=[score],
            name=f"n_{score}",
            cells=cell,
            particle="neutron")
        tallies.append(t)

    for score in PHOTON_SCALAR_SCORES:
        t = yamc.Tally(
            scores=[score],
            name=f"p_{score}",
            cells=cell,
            particle="photon")
        tallies.append(t)

    t_n_spec = yamc.Tally(
        scores=["flux"],
        name="neutron_spectrum",
        cells=cell,
        particle="neutron",
        energy_group_structure=NEUTRON_GROUP_STRUCTURE)
    tallies.append(t_n_spec)

    t_p_spec = yamc.Tally(
        scores=["flux"],
        name="photon_spectrum",
        cells=cell,
        particle="photon",
        energy_group_structure=PHOTON_GROUP_STRUCTURE)
    tallies.append(t_p_spec)

    model = yamc.Model(geometry=geometry, tallies=tallies, source=source,
                     transport_secondary_photons=True)
    results = model.simulate_transport(total_particles=PARTICLES * BATCHES, seed=SEED)

    n_scalars, n_stds = {}, {}
    p_scalars, p_stds = {}, {}
    for t in tallies:
        if t.name in ("neutron_spectrum", "photon_spectrum"):
            continue
        r = results[t]
        val = r.mean[0]
        std = r.standard_deviation[0]
        if t.name.startswith("n_"):
            n_scalars[t.name[2:]] = val
            n_stds[t.name[2:]] = std
        elif t.name.startswith("p_"):
            p_scalars[t.name[2:]] = val
            p_stds[t.name[2:]] = std

    n_spec = results[t_n_spec]
    p_spec = results[t_p_spec]
    return {
        "n_scalars": n_scalars,
        "n_stds": n_stds,
        "p_scalars": p_scalars,
        "p_stds": p_stds,
        "n_spectrum_mean": list(n_spec.mean),
        "n_spectrum_std": list(n_spec.standard_deviation),
        "p_spectrum_mean": list(p_spec.mean),
        "p_spectrum_std": list(p_spec.standard_deviation),
    }


@pytest.mark.parametrize("nuclide", COUPLED_NUCLIDES)
def test_coupled_neutron_flux(nuclide):
    """Test coupled neutron flux matches reference (primary check)."""
    ref = load_reference("coupled", nuclide)
    skip_if_no_reference(ref, "coupled", nuclide)

    result = get_result(nuclide)

    assert_self_reference(
        ref["n_scalars"]["flux"], result["n_scalars"]["flux"], f"{nuclide}/n_flux")


@pytest.mark.parametrize("nuclide", COUPLED_NUCLIDES)
def test_coupled_photon_flux(nuclide):
    """Test coupled photon flux matches reference (primary check)."""
    ref = load_reference("coupled", nuclide)
    skip_if_no_reference(ref, "coupled", nuclide)

    result = get_result(nuclide)
    assert_self_reference(
        ref["p_scalars"]["flux"], result["p_scalars"]["flux"], f"{nuclide}/p_flux")


@pytest.mark.parametrize("nuclide", COUPLED_NUCLIDES)
def test_coupled_neutron_scalars(nuclide):
    """Test coupled neutron scores broadly agree with reference."""
    ref = load_reference("coupled", nuclide)
    skip_if_no_reference(ref, "coupled", nuclide)

    result = get_result(nuclide)
    for field in ("n_scalars", "n_stds"):
        assert_self_reference(ref[field], result[field], f"{nuclide}/{field}")


@pytest.mark.parametrize("nuclide", COUPLED_NUCLIDES)
def test_coupled_photon_scalars(nuclide):
    """Test coupled photon scores broadly agree with reference."""
    ref = load_reference("coupled", nuclide)
    skip_if_no_reference(ref, "coupled", nuclide)

    result = get_result(nuclide)
    for field in ("p_scalars", "p_stds"):
        assert_self_reference(ref[field], result[field], f"{nuclide}/{field}")


@pytest.mark.parametrize("nuclide", COUPLED_NUCLIDES)
def test_coupled_neutron_spectrum(nuclide):
    """Test coupled neutron flux spectrum matches reference."""
    ref = load_reference("coupled", nuclide)
    skip_if_no_reference(ref, "coupled", nuclide)

    result = get_result(nuclide)

    for field in ("n_spectrum_mean", "n_spectrum_std"):
        assert_self_reference(ref[field], result[field], f"{nuclide}/{field}")


@pytest.mark.parametrize("nuclide", COUPLED_NUCLIDES)
def test_coupled_photon_spectrum(nuclide):
    """Every photon spectrum bin retains its same-seed answer."""
    ref = load_reference("coupled", nuclide)
    skip_if_no_reference(ref, "coupled", nuclide)

    result = get_result(nuclide)
    for field in ("p_spectrum_mean", "p_spectrum_std"):
        assert_self_reference(ref[field], result[field], f"{nuclide}/{field}")
