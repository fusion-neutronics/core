"""Photon and neutron cross sections follow independent explicit selections.

Photon keywords resolve elements without an explicit material read.
Missing photoatomic data must fail, even after another library was cached.
"""

import pytest

import yamc


def _build_model(
    material, total_particles=4000, energy=1.0e6, secondary_photons=True, source=None,
):
    cyl = yamc.Cylinder(axis="z", radius=1.0, boundary="vacuum")
    z_bot = yamc.Plane(axis="z", offset=-50.0, boundary="vacuum")
    z_top = yamc.Plane(axis="z", offset=50.0, boundary="vacuum")
    cell = yamc.Cell(
        name="iron",
        region=cyl.below & z_bot.above & z_top.below,
        material=material,
    )
    geom = yamc.Geometry([cell])
    src = source or yamc.PhotonSource(
        position=(0, 0, 0),
        direction=yamc.sources.Isotropic(),
        energy=yamc.sources.Discrete([energy], [1.0]),
    )
    flux = yamc.Tally(scores=["flux"], cells=cell, particle="photon", name="flux")
    photoelectric = yamc.Tally(
        scores=["photoelectric"], cells=cell, particle="photon", name="pe"
    )
    incoherent = yamc.Tally(
        scores=["incoherent-scatter"], cells=cell, particle="photon", name="in"
    )
    model = yamc.Model(
        geometry=geom,
        tallies=[flux, photoelectric, incoherent],
        source=src,
        transport_secondary_photons=secondary_photons,
    )
    return (model, flux, photoelectric, incoherent,
            {"total_particles": total_particles, "seed": 1})


def test_transport_secondary_photons_without_data_raises():
    """transport_secondary_photons=True must refuse to run if no photon data was loaded."""
    yamc.neutron_cross_section_data = None
    yamc.photon_cross_section_data = None
    mat = yamc.Material(composition={"Fe56": 1.0}, density=7.874, temperature=294)
    yamc.neutron_cross_section_data = "endf-b8.1"
    model, *_, run_kwargs = _build_model(mat)
    with pytest.raises(ValueError) as exc:
        model.simulate_transport(**run_kwargs)
    assert "photon" in str(exc.value).lower()
    assert "photon_cross_section_data" in str(exc.value)


@pytest.mark.parametrize("keyword", ["endf-b8.1"])
@pytest.mark.parametrize("secondary_photons", [False, True])
@pytest.mark.parametrize("composition,density,energy", [
    ({"Fe56": 1.0}, 7.874, 1.0e6),
    ({"H1": 2.0, "O16": 1.0}, 1.0, 1.0e5),
])
def test_global_keyword_loads_element_photon_data(
    keyword, composition, density, energy, secondary_photons,
):
    yamc.neutron_cross_section_data = None
    yamc.photon_cross_section_data = keyword
    mat = yamc.Material(composition=composition, density=density, temperature=294)
    model, flux, photoelectric, incoherent, run_kwargs = _build_model(
        mat, energy=energy, secondary_photons=secondary_photons,
    )
    results = model.simulate_transport(**run_kwargs)
    assert results[flux].mean[0] > 0.0
    assert results[photoelectric].mean[0] > 0.0
    assert results[incoherent].mean[0] > 0.0


def test_global_keyword_loads_coupled_photon_data():
    yamc.neutron_cross_section_data = "endf-b8.1"
    yamc.photon_cross_section_data = "endf-b8.1"
    material = yamc.Material(composition={"Fe56": 1.0}, density=7.874, temperature=294)
    source = yamc.NeutronSource(position=(0, 0, 0), energy=14.0e6)
    model, flux, _, incoherent, run_kwargs = _build_model(material, source=source)
    results = model.simulate_transport(**run_kwargs)
    assert results[flux].mean[0] > 0.0
    assert results[incoherent].mean[0] > 0.0


def test_photon_only_does_not_read_an_unneeded_neutron_source(tmp_path):
    neutron = tmp_path / "unused-neutron-library"
    neutron.mkdir()
    yamc.neutron_cross_section_data = str(neutron)
    yamc.photon_cross_section_data = "endf-b8.1"
    material = yamc.Material(composition={"Fe56": 1.0}, density=7.874, temperature=294)
    model, flux, _, _, run_kwargs = _build_model(material, total_particles=32, secondary_photons=False)
    results = model.simulate_transport(**run_kwargs)
    assert results[flux].mean[0] > 0.0


def test_global_keyword_without_photoatomic_data_raises_after_another_library_loads():
    yamc.neutron_cross_section_data = None
    yamc.photon_cross_section_data = "endf-b8.1"
    material = yamc.Material(composition={"Fe56": 1.0}, density=7.874, temperature=294)
    model, *_, run_kwargs = _build_model(material, total_particles=10)
    model.simulate_transport(**run_kwargs)

    yamc.photon_cross_section_data = "tendl-2025"
    with pytest.raises(ValueError, match="(?i)photon") as error:
        model.simulate_transport(**run_kwargs)
    assert "tendl-2025" in str(error.value)


@pytest.mark.parametrize("keyword", ["endf-b8.1"])
def test_keyword_form_loads_element_photon_data(keyword):
    """An explicit neutron read must coexist with the chosen photon library.

    Requires network access or a pre-populated nuclear-data cache.
    """
    yamc.photon_cross_section_data = keyword
    mat = yamc.Material(composition={"Fe56": 1.0}, density=7.874, temperature=294)
    mat.read_nuclear_data(keyword)

    model, flux, photoelectric, incoherent, run_kwargs = _build_model(mat)
    results = model.simulate_transport(**run_kwargs)

    # If photon physics were not initialized, every interaction tally would
    # be exactly zero (photons stream uncollided to vacuum). The fix
    # ensures non-trivial reaction rates for an iron column at 1 MeV, where
    # incoherent scatter and photoelectric absorption are the dominant
    # interactions.
    assert results[flux].mean[0] > 0.0
    assert results[photoelectric].mean[0] > 0.0, (
        "photoelectric reaction rate is zero -- photon transport ran without "
        "element-level cross sections"
    )
    assert results[incoherent].mean[0] > 0.0, (
        "incoherent scatter reaction rate is zero -- photon transport ran "
        "without element-level cross sections"
    )
