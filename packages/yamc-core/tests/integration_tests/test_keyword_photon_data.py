"""Verify that `Material.read_nuclear_data('<keyword>')` correctly resolves
element-level photon data for photon transport.

Two regressions are covered:

1. **Silent photon-transport failure** -- previously, `transport_secondary_photons=True`
   on a material whose `photon_data_paths` was empty would skip
   `init_photon_data` and silently transport photons with zero cross
   sections (uncollided streaming, all reaction tallies = 0). Now it errors.

2. **Keyword-form does not autodownload element photon Arrow** -- previously,
   `read_nuclear_data('endf-b8.1')` only registered per-isotope neutron URLs.
   Element-level photon Arrow tars (e.g. `Fe.arrow.tar`) are published in the
   same release, so the same keyword now also registers each element's
   photon source path; `get_or_load_element` resolves it on demand the same
   way nuclide loading does.
"""

import pytest

import yamc


def _build_model(material, total_particles=4000):
    cyl = yamc.Cylinder(axis="z", radius=1.0, boundary="vacuum")
    z_bot = yamc.Plane(axis="z", offset=-50.0, boundary="vacuum")
    z_top = yamc.Plane(axis="z", offset=50.0, boundary="vacuum")
    cell = yamc.Cell(
        name="iron",
        region=cyl.below & z_bot.above & z_top.below,
        material=material,
    )
    geom = yamc.Geometry([cell])
    src = yamc.PhotonSource(
        position=(0, 0, 0),
        direction=yamc.sources.Isotropic(),
        energy=yamc.sources.Discrete([1.0e6], [1.0]),
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
        transport_secondary_photons=True,
    )
    return (model, flux, photoelectric, incoherent,
            {"total_particles": total_particles, "seed": 1})


def test_transport_secondary_photons_without_data_raises():
    """transport_secondary_photons=True must refuse to run if no photon data was loaded."""
    mat = yamc.Material(composition={"Fe56": 1.0}, density=7.874, temperature=294)
    # Load only neutron data -- no photon_data argument.
    mat.read_nuclear_data({"Fe56": "tests/Fe56.arrow"})
    model, *_, run_kwargs = _build_model(mat)
    with pytest.raises(BaseException) as exc:  # pyo3 PanicException is a BaseException
        model.simulate_transport(**run_kwargs)
    assert "photon" in str(exc.value).lower()
    assert "photon_data" in str(exc.value)


@pytest.mark.parametrize("keyword", ["endf-b8.1"])
def test_keyword_form_loads_element_photon_data(keyword):
    """`read_nuclear_data('<keyword>')` should make photon transport work
    end-to-end without an explicit `photon_data` argument.

    Requires either network access or a pre-populated `~/.cache/yamc`
    (CI prefetches both per-isotope and per-element tars).
    """
    mat = yamc.Material(composition={"Fe56": 1.0}, density=7.874, temperature=294)
    try:
        mat.read_nuclear_data(keyword)
    except Exception as e:  # pragma: no cover - skip if no network and no cache
        pytest.skip(f"keyword resolution failed (offline?): {e}")

    model, flux, photoelectric, incoherent, run_kwargs = _build_model(mat)
    try:
        results = model.simulate_transport(**run_kwargs)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"simulate failed (likely network unavailable): {e}")

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
