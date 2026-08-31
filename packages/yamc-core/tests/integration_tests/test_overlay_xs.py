"""Integration tests for virtual-overlay tallies (the ``response=`` argument).

``response=`` replaced the old ``mode="microscopic"`` flag (issue #341):

  * ``response="Fe56"`` / ``response=["Fe56", ...]`` -- microscopic cross
    sections (barns) at unit density; one bin per nuclide (the old overlay).
  * ``response=<Material>`` -- the material's *macroscopic* response, weighting
    each nuclide's microscopic XS by its real atom density; one combined bin.
"""

import yamc
import pytest


@pytest.fixture(autouse=True)
def setup_config():
    """Set up global cross-section data with test nuclear data paths."""
    yamc.cross_section_data = {"Fe56": "tests/Fe56.arrow"}


def _make_void_sphere_model(tallies, batches=5, particles=5000):
    """Create a model with a void sphere and a 14 MeV point source at center."""
    sphere = yamc.Sphere(
        x0=0.0, y0=0.0, z0=0.0, radius=5.0, boundary="vacuum"
    )
    cell = yamc.Cell(name="void", region=sphere.below)
    geometry = yamc.Geometry(cells=[cell])
    source = yamc.NeutronSource(
        position=[0.0, 0.0, 0.0],
        energy=yamc.sources.Discrete([14.0e6], [1.0]))
    if not isinstance(tallies, list):
        tallies = [tallies]
    model = yamc.Model(geometry=geometry, tallies=tallies, source=source)
    return model, {"total_particles": particles * batches}


def _make_fe_sphere_model(tallies, batches=5, particles=5000, transport_secondary_photons=False):
    """Create a model with an iron sphere and a 14 MeV point source at center."""
    sphere = yamc.Sphere(
        x0=0.0, y0=0.0, z0=0.0, radius=5.0, boundary="vacuum"
    )
    mat = yamc.Material(
        composition={"Fe56": 1.0},
        density=7.874,
        temperature=294)
    photon_data = {"Fe": "tests/Fe.arrow"} if transport_secondary_photons else None
    mat.read_nuclear_data({"Fe56": "tests/Fe56.arrow"}, photon_data=photon_data)
    cell = yamc.Cell(name="iron", region=sphere.below, material=mat)
    geometry = yamc.Geometry(cells=[cell])
    source = yamc.NeutronSource(
        position=[0.0, 0.0, 0.0],
        energy=yamc.sources.Discrete([14.0e6], [1.0]))
    if not isinstance(tallies, list):
        tallies = [tallies]
    model = yamc.Model(geometry=geometry, tallies=tallies, source=source,
                    transport_secondary_photons=transport_secondary_photons)
    return model, {"total_particles": particles * batches}


def test_response_default_none():
    """A normal (macroscopic) tally has no response."""
    assert yamc.Tally().response is None


def test_response_str_getter():
    """response='Fe56' -> one-nuclide unit-density overlay; getter lists it."""
    tally = yamc.Tally(scores=["heating"], response="Fe56")
    assert tally.response == ["Fe56"]


def test_response_list_getter():
    """response=[...] preserves the nuclide list."""
    tally = yamc.Tally(scores=["heating"], response=["Fe56", "Fe57"])
    assert tally.response == ["Fe56", "Fe57"]


def test_response_material_getter():
    """response=<Material> -> getter returns the per-nuclide density dict."""
    fe = yamc.Material(composition={"Fe56": 1.0}, density=7.874, temperature=294)
    tally = yamc.Tally(scores=["heating"], response=fe)
    resp = tally.response
    assert isinstance(resp, dict)
    assert "Fe56" in resp and resp["Fe56"] > 0.0


def test_response_invalid_type():
    """response must be a str, list[str], or Material."""
    with pytest.raises((TypeError, ValueError)):
        yamc.Tally(scores=["heating"], response=123)


def test_response_nuclides_mutually_exclusive():
    """response and nuclides cannot both be set."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        yamc.Tally(scores=["heating"], response="Fe56", nuclides=["Fe56"])


def test_response_empty_list_errors():
    """response=[] has no target."""
    with pytest.raises(ValueError, match="must not be empty"):
        yamc.Tally(scores=["heating"], response=[])


def test_overlay_neutron_heating_void():
    """Void cell + response='Fe56' + neutron heating -> non-zero."""
    tally = yamc.Tally(
        scores=["heating"],
        response="Fe56",
        particle="neutron")

    model, run_kwargs = _make_void_sphere_model(tally, batches=3, particles=2000)
    results = model.simulate_transport(**run_kwargs)

    mean = results[tally].mean
    assert len(mean) > 0, "Tally should have results"
    assert mean[0] > 0.0, (
        f"Overlay neutron heating in void should be non-zero, got {mean[0]}"
    )


def test_overlay_neutron_reaction_rate_void():
    """Void cell + response='Fe56' + Fe56 total XS -> non-zero."""
    tally = yamc.Tally(
        scores=[1],  # MT 1 = total
        response="Fe56")

    model, run_kwargs = _make_void_sphere_model(tally, batches=3, particles=2000)
    results = model.simulate_transport(**run_kwargs)

    mean = results[tally].mean
    assert len(mean) > 0
    assert mean[0] > 0.0, (
        f"Overlay total reaction rate in void should be non-zero, got {mean[0]}"
    )


def test_overlay_flux_is_refused():
    """`response=` on a flux score is refused at construction (issue #305).

    A response is applied by folding the overlay's microscopic cross section
    into the score, and flux has none, so the tally used to return plain flux
    with the response silently dropped. Score flux in its own tally instead.
    """
    with pytest.raises(ValueError, match="cannot apply to the 'flux' score"):
        yamc.Tally(scores=["flux"], response="Fe56")


def test_overlay_refused_when_mixed_with_flux():
    """The whole tally is refused, not partly applied.

    With `scores=['flux', 'heating']` the heating bins were response-weighted
    (11.8x for a nuclide response) while the flux bins were plain flux, in one
    result array with nothing marking which was which.
    """
    with pytest.raises(ValueError, match="cannot apply to the 'flux' score"):
        yamc.Tally(scores=["flux", "heating"], response="Fe56")


def test_overlay_photon_heating_material():
    """Material cell + response='Fe56' photon heating -> non-zero.

    Uses photon transport with an iron sphere. The overlay tally scores
    microscopic photon heating KERMA for Fe56 using a track-length estimator.
    """
    tally = yamc.Tally(
        scores=["heating"],
        response="Fe56",
        particle="photon")

    model, run_kwargs = _make_fe_sphere_model(
        tally, batches=3, particles=2000, transport_secondary_photons=True
    )
    results = model.simulate_transport(**run_kwargs)

    mean = results[tally].mean
    assert len(mean) > 0, "Tally should have results"
    assert mean[0] > 0.0, (
        f"Overlay photon heating in iron should be non-zero, got {mean[0]}"
    )


def test_overlay_vs_normal_density_comparison():
    """microscopic result x atom_density ~ macroscopic result.

    For a single-nuclide material:
      overlay_result * N_atom_density ~ normal_result
    """
    tally_overlay = yamc.Tally(
        scores=["heating"],
        response="Fe56",
        particle="neutron")

    tally_normal = yamc.Tally(scores=["heating"], particle="neutron")

    model, run_kwargs = _make_fe_sphere_model(
        [tally_overlay, tally_normal], batches=5, particles=10000
    )
    results = model.simulate_transport(**run_kwargs)

    overlay_mean = results[tally_overlay].mean[0]
    normal_mean = results[tally_normal].mean[0]

    # Fe56 atom density: N = rho * Na / M
    fe56_atom_density = 7.874 * 6.022e23 / (55.845 * 1e24)  # atoms/barn-cm

    reconstructed = overlay_mean * fe56_atom_density
    if normal_mean > 0.0 and overlay_mean > 0.0:
        ratio = reconstructed / normal_mean
        assert 0.8 < ratio < 1.2, (
            f"overlay*N / normal = {ratio:.4f} (overlay={overlay_mean:.6e}, "
            f"normal={normal_mean:.6e}, N={fe56_atom_density:.6e})"
        )


def test_material_response_matches_nuclide_sum_workaround():
    """Issue #341: response=<Material> equals the manual nuclide-sum workaround.

    Scores three tallies in the SAME run (identical tracks):
      * material  -- response=<Fe material> (one combined macroscopic bin)
      * overlay   -- response='Fe56' (unit-density microscopic, one bin)
      * normal    -- ordinary macroscopic tally (ground truth)

    The material response must equal overlay x atom-density exactly, and must
    agree with the ordinary macroscopic tally to within statistics.
    """
    fe = yamc.Material(composition={"Fe56": 1.0}, density=7.874, temperature=294)
    n_fe56 = fe.get_atoms_per_barn_cm()["Fe56"]

    tally_material = yamc.Tally(scores=["heating"], response=fe, particle="neutron")
    tally_overlay = yamc.Tally(scores=["heating"], response="Fe56", particle="neutron")
    tally_normal = yamc.Tally(scores=["heating"], particle="neutron")

    model, run_kwargs = _make_fe_sphere_model(
        [tally_material, tally_overlay, tally_normal], batches=5, particles=10000
    )
    results = model.simulate_transport(**run_kwargs)

    material_mean = results[tally_material].mean
    overlay_mean = results[tally_overlay].mean[0]
    normal_mean = results[tally_normal].mean[0]

    # A material response collapses to a single combined bin.
    assert len(material_mean) == 1, "material response must be one combined bin"
    assert material_mean[0] > 0.0

    # Exact: same tracks, material = overlay scaled by atom density.
    reconstructed = overlay_mean * n_fe56
    ratio = material_mean[0] / reconstructed
    assert 0.999 < ratio < 1.001, (
        f"material response {material_mean[0]:.6e} != overlay*N {reconstructed:.6e} "
        f"(ratio {ratio:.6f})"
    )

    # Physical: matches the ordinary macroscopic tally to within statistics.
    phys_ratio = material_mean[0] / normal_mean
    assert 0.8 < phys_ratio < 1.2, (
        f"material response / macroscopic = {phys_ratio:.4f}"
    )


def test_material_response_in_void():
    """A material response works across void -- a virtual detector with no
    physical presence in the geometry (the silicon-dose-on-mesh use case)."""
    fe = yamc.Material(composition={"Fe56": 1.0}, density=7.874, temperature=294)
    tally = yamc.Tally(scores=["heating"], response=fe, particle="neutron")

    model, run_kwargs = _make_void_sphere_model(tally, batches=3, particles=2000)
    results = model.simulate_transport(**run_kwargs)

    mean = results[tally].mean
    assert len(mean) == 1
    assert mean[0] > 0.0, (
        f"Material response in void should be non-zero, got {mean[0]}"
    )


def test_overlay_rejected_on_gpu():
    """``response=`` must be refused on the GPU, not silently dropped (#288).

    The kernel scores the CELL material's macroscopic cross section, so an
    overlay tally that reached it came back as the plain cell-material score
    (11.8x off for a nuclide response) with no diagnostic.
    """
    if not yamc.parallel.gpu_available():
        pytest.skip("no f64 GPU available")

    tally = yamc.Tally(scores=["heating"], response="Fe56")
    model, run_kwargs = _make_fe_sphere_model(tally, batches=1, particles=200)
    run_kwargs["compute"] = "gpu"
    with pytest.raises(ValueError, match="virtual-overlay tallies"):
        model.simulate_transport(**run_kwargs)


def test_overlay_without_resolvable_data_raises():
    """Unresolvable overlay data must raise, not warn and score plain values.

    The overlay nuclides' data comes from the global configuration, not from the
    cell materials, so a run whose global config cannot resolve them used to
    print a ``[WARNING]`` and then score the un-responded quantity (#288).
    """
    saved = yamc.cross_section_data
    yamc.cross_section_data = {}
    try:
        tally = yamc.Tally(scores=["heating"], response="Fe56")
        model, run_kwargs = _make_fe_sphere_model(tally, batches=1, particles=200)
        with pytest.raises(ValueError, match="response"):
            model.simulate_transport(**run_kwargs)
    finally:
        yamc.cross_section_data = saved
