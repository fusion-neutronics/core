"""End-to-end Python tests for Woodcock (delta) tracking.

The Rust API is exposed to Python via the ``tracking_mode`` keyword
argument on ``yamc.Model``.

These tests verify:
1. The default is ``"surface"`` so existing Python scripts keep working.
2. Passing ``tracking_mode="woodcock"`` produces a non-zero MT 105
   reaction rate on a Li6 sphere -- the algorithm runs end-to-end from
   Python.
3. Woodcock and Surface agree on the MT 105 reaction rate within
   statistical tolerance.
4. ``"surface"``, ``"woodcock"`` and ``"hybrid"`` are accepted; other
   strings (aliases like ``"delta"``) raise ``ValueError``.
5. The ``tracking_mode`` getter / setter round-trips.
"""

import tempfile

import numpy as np
import yamc


def _build_li6_sphere_model(*, tracking_mode: str = "surface", seed: int = 42,
                            total_particles: int = 2000) -> yamc.Model:
    """Build a Li6 sphere model with a MT 105 collision-estimator tally.

    Mirrors the Rust ``test_woodcock_validation`` test rig so the
    Woodcock-vs-Surface match can be observed across both binding paths.
    """
    sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=200.0,
                         boundary="vacuum")
    region = sphere.below

    material = yamc.Material(composition={"Li6": 1.0}, density=0.534,
                             temperature=294)
    material.read_nuclear_data({"Li6": "tests/Li6.arrow"})

    cell = yamc.Cell(name="li6_sphere", region=region, material=material)
    geometry = yamc.Geometry(cells=[cell])

    source = yamc.NeutronSource(position=[0.0, 0.0, 0.0],
                                energy=yamc.sources.Discrete([1e6], [1.0]))

    # Collision estimator -- the only one supported by Phase 1 Woodcock.
    tally = yamc.Tally(scores=[105], name="mt_105_rxrate",
                       cells=cell, estimator="collision")

    model = yamc.Model(geometry=geometry, tallies=[tally], source=source,
                       tracking_mode=tracking_mode, verbose=[])
    return model, {"total_particles": total_particles, "seed": seed}


def test_default_tracking_mode_is_surface():
    """Existing scripts must not see a behaviour change."""
    model, _ = _build_li6_sphere_model()  # no tracking_mode arg
    assert model.tracking_mode == "surface"


def test_woodcock_runs_and_returns_nonzero():
    """Smoke test: Woodcock from Python produces a non-zero MT 105 rate."""
    model, run_kwargs = _build_li6_sphere_model(tracking_mode="woodcock")
    assert model.tracking_mode == "woodcock"
    results = model.simulate_transport(**run_kwargs)
    tally_result = results[model.tallies[0]]
    mean = tally_result.mean[0]
    assert mean > 0.0, (
        f"Woodcock + collision MT 105 returned zero mean ({mean})"
    )


def test_woodcock_matches_surface_within_statistics():
    """Same seed, same geometry: Woodcock and Surface agree on MT 105
    within 3σ.

    Demonstrates Woodcock is statistically equivalent to surface tracking
    from the Python API.
    """
    # Use a slightly larger N to keep the test reliable.
    surf_model, surf_run = _build_li6_sphere_model(
        tracking_mode="surface", total_particles=5000)
    surf_results = surf_model.simulate_transport(**surf_run)
    surf_t = surf_results[surf_model.tallies[0]]
    mean_surf = surf_t.mean[0]
    std_surf = surf_t.standard_deviation[0]

    wood_model, wood_run = _build_li6_sphere_model(
        tracking_mode="woodcock", total_particles=5000)
    wood_results = wood_model.simulate_transport(**wood_run)
    wood_t = wood_results[wood_model.tallies[0]]
    mean_wood = wood_t.mean[0]

    assert mean_surf > 0.0, "Surface mean was zero -- test rig broken"
    assert mean_wood > 0.0, "Woodcock mean was zero"

    diff = abs(mean_surf - mean_wood)
    tol = 3.0 * std_surf
    assert diff < tol, (
        f"Woodcock mean {mean_wood:.6e} differs from Surface mean "
        f"{mean_surf:.6e} by {diff:.2e}, exceeding 3σ tolerance "
        f"{tol:.2e}"
    )


def test_tracking_mode_accepts_hybrid():
    """``"hybrid"`` (delta tracking with a surface fallback in voids /
    low-density cells) is accepted and round-trips."""
    model, _ = _build_li6_sphere_model(tracking_mode="hybrid")
    assert model.tracking_mode == "hybrid"


def test_tracking_mode_rejects_aliases():
    """``"surface"``, ``"woodcock"`` and ``"hybrid"`` are accepted;
    aliases like ``"delta"`` raise ``ValueError`` to keep the surface
    small."""
    import pytest
    for alias in ("delta", "delta-tracking"):
        with pytest.raises(ValueError, match="tracking_mode"):
            _build_li6_sphere_model(tracking_mode=alias)


def test_tracking_mode_setter_round_trips():
    """Setting via attribute access works the same way."""
    model, _ = _build_li6_sphere_model()
    assert model.tracking_mode == "surface"
    for mode in ("woodcock", "hybrid", "surface"):
        model.tracking_mode = mode
        assert model.tracking_mode == mode


def test_tracking_mode_rejects_unknown_string():
    """An unknown value raises a clear ValueError."""
    import pytest
    with pytest.raises(ValueError, match="tracking_mode"):
        _build_li6_sphere_model(tracking_mode="bogus")


# ---------------------------------------------------------------------------
# Photon transport under Woodcock tracking
# ---------------------------------------------------------------------------
#
# Woodcock now transports photons too: photon sources, coupled
# neutron->photon production, and D1S decay photons (all share the
# banked-secondary path). These mirror the Rust photon Woodcock tests.


def _build_fe_photon_model(*, source, tracking_mode="surface", photon_only,
                           estimator="collision", radius=10.0, seed=7,
                           total_particles=20_000):
    """Iron sphere with neutron (Fe56) + photon (Fe) data. Photon flux
    tally (collision estimator is free of the Phase 4.0 track-length
    leakage bias, so it gives a clean Woodcock-vs-Surface match)."""
    sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=radius,
                         boundary="vacuum")
    material = yamc.Material(composition={"Fe56": 1.0}, density=7.874,
                             temperature=294)
    material.read_nuclear_data({"Fe56": "tests/Fe56.arrow"},
                               photon_data={"Fe": "tests/Fe.arrow"})
    cell = yamc.Cell(name="fe", region=sphere.below, material=material)
    geometry = yamc.Geometry(cells=[cell])
    tally = yamc.Tally(scores=["flux"], name="photon_flux", cells=cell,
                       estimator=estimator,
                       particle="photon" if photon_only else None)
    model = yamc.Model(geometry=geometry, tallies=[tally], source=source,
                       transport_secondary_photons=True, tracking_mode=tracking_mode,
                       verbose=[])
    return model, {"total_particles": total_particles, "seed": seed}


def test_woodcock_photon_source_runs_nonzero():
    """A photon source under Woodcock produces non-zero photon flux."""
    source = yamc.PhotonSource(position=[0.0, 0.0, 0.0],
                               energy=yamc.sources.Discrete([1.0e6], [1.0]))
    model, run_kwargs = _build_fe_photon_model(
        source=source, tracking_mode="woodcock",
        photon_only=False, estimator="track-length",
        seed=42, total_particles=5000)
    results = model.simulate_transport(**run_kwargs)
    mean = results[model.tallies[0]].mean[0]
    assert mean > 0.0, f"Woodcock photon flux was zero ({mean})"


def test_woodcock_photon_source_matches_surface():
    """Woodcock and Surface agree on photon flux within 3 sigma."""
    def run(mode):
        source = yamc.PhotonSource(position=[0.0, 0.0, 0.0],
                                   energy=yamc.sources.Discrete([1.0e6], [1.0]))
        model, run_kwargs = _build_fe_photon_model(
            source=source, tracking_mode=mode,
            photon_only=False, estimator="collision")
        res = model.simulate_transport(**run_kwargs)[model.tallies[0]]
        return res.mean[0], res.standard_deviation[0]

    mean_surf, std_surf = run("surface")
    mean_wood, _ = run("woodcock")
    assert mean_surf > 0.0 and mean_wood > 0.0
    diff = abs(mean_surf - mean_wood)
    tol = 3.0 * std_surf
    assert diff < tol, (
        f"Woodcock photon flux {mean_wood:.6e} differs from Surface "
        f"{mean_surf:.6e} by {diff:.2e}, exceeding 3 sigma {tol:.2e}"
    )


def test_woodcock_coupled_neutron_photon():
    """Neutron source + transport_secondary_photons under Woodcock: secondary
    photons are produced, transported, and the photon-filtered flux
    matches surface tracking within 3 sigma."""
    def run(mode):
        source = yamc.NeutronSource(position=[0.0, 0.0, 0.0],
                                    energy=yamc.sources.Discrete([14.06e6], [1.0]))
        model, run_kwargs = _build_fe_photon_model(
            source=source, tracking_mode=mode,
            photon_only=True, estimator="collision",
            radius=20.0, seed=123)
        res = model.simulate_transport(**run_kwargs)[model.tallies[0]]
        return res.mean[0], res.standard_deviation[0]

    mean_surf, std_surf = run("surface")
    mean_wood, _ = run("woodcock")
    assert mean_wood > 0.0, "no coupled photons transported under Woodcock"
    assert mean_surf > 0.0
    diff = abs(mean_surf - mean_wood)
    tol = 3.0 * std_surf
    assert diff < tol, (
        f"Woodcock coupled photon flux {mean_wood:.6e} differs from Surface "
        f"{mean_surf:.6e} by {diff:.2e}, exceeding 3 sigma {tol:.2e}"
    )


# ---------------------------------------------------------------------------
# Free-gas thermal scattering, decay photons, and URR under Woodcock
# ---------------------------------------------------------------------------
#
# These close the remaining "smaller gap" interactions from the Woodcock
# roadmap (issue #231): free-gas thermal scattering and D1S decay photons
# composing with delta tracking, plus an xfail regression that pins the
# known URR + Woodcock self-shielding bias.


def _collision_flux_match(build_model):
    """Run build_model('surface') and build_model('woodcock'); return
    (surface_mean, surface_std, woodcock_mean). Uses the summed tally
    total so it works for scalar and group tallies alike."""

    def total(mode):
        model, run_kwargs = build_model(mode)
        res = model.simulate_transport(**run_kwargs)[model.tallies[0]]
        return float(np.sum(res.mean)), float(
            np.sqrt(np.sum(np.asarray(res.standard_deviation) ** 2))
        )

    ms, ss = total("surface")
    mw, _ = total("woodcock")
    return ms, ss, mw


def test_woodcock_free_gas_matches_surface():
    """Free-gas thermal elastic scattering must compose with Woodcock.

    Light moderator (H2) at 294 K with a 1 eV source -- well below the
    free-gas threshold (400 kT ~ 10 eV at 294 K), so the free-gas target-
    velocity sampling is active. Free gas changes only the scatter
    kinematics (not Σ_t), so unlike URR it carries no flight-correlation
    subtlety; Woodcock and surface must agree within 3σ.
    """

    def build(mode):
        sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=20.0,
                             boundary="vacuum")
        material = yamc.Material(composition={"H2": 1.0}, density=0.1,
                                 temperature=294)
        material.read_nuclear_data({"H2": "tests/H2.arrow"})
        cell = yamc.Cell(name="h2", region=sphere.below, material=material, id=1)
        source = yamc.NeutronSource(position=[0.0, 0.0, 0.0],
                                    energy=yamc.sources.Discrete([1.0], [1.0]))
        tally = yamc.Tally(scores=["flux"], name="flux", cells=cell,
                           estimator="collision")
        model = yamc.Model(geometry=yamc.Geometry(cells=[cell]),
                           tallies=[tally], source=source,
                           tracking_mode=mode, verbose=[])
        return model, {"total_particles": 40_000, "seed": 3}

    ms, ss, mw = _collision_flux_match(build)
    assert ms > 0.0 and mw > 0.0
    assert abs(ms - mw) < 3.0 * ss, (
        f"free-gas Woodcock flux {mw:.6e} vs surface {ms:.6e} "
        f"exceeds 3σ ({3.0 * ss:.2e})"
    )


def test_woodcock_decay_photons_match_surface():
    """D1S decay photons must transport correctly under Woodcock.

    Neutron source on an Fe sphere with the SFR transmutation chain and
    photon data; secondary D1S decay photons are produced at neutron
    collisions and transported as photons. Fat dense sphere + collision
    estimator keeps the delta-tracking estimator's variance low, so the
    decay-photon flux matches surface tracking within 3σ.
    """
    chain = yamc.TransmutationChain("tests/transmutation-endf-b8.1-sfr.arrow")
    reduced = chain.reduce(["Fe56"], 5)
    tmp_dir = tempfile.mkdtemp(suffix=".chain.arrow")
    reduced.export_to_arrow(tmp_dir)

    def build(mode):
        material = yamc.Material(composition={"Fe56": 1.0}, density=7.874,
                                 temperature=294)
        material.read_nuclear_data({"Fe56": "tests/Fe56.arrow"},
                                   photon_data={"Fe": "tests/Fe.arrow"})
        sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=20.0,
                             boundary="vacuum")
        cell = yamc.Cell(name="fe", region=sphere.below, material=material, id=1)
        source = yamc.NeutronSource(position=[0.0, 0.0, 0.0],
                                    energy=yamc.sources.Discrete([14.06e6], [1.0]))
        yamc.transmutation_decay_data = tmp_dir
        yamc.transmutation_reactions = tmp_dir
        yamc.transmutation_fission_yields = tmp_dir
        rn = yamc.Model(geometry=yamc.Geometry(cells=[cell]),
                        source=source).radionuclides()
        tally = yamc.Tally(scores=["flux"], name="decay_photons", cells=cell,
                           particle="photon", estimator="collision",
                           parent_nuclides=rn)
        model = yamc.Model(geometry=yamc.Geometry(cells=[cell]),
                           tallies=[tally], source=source,
                           transport_secondary_photons=True, use_decay_photons=True,
                           tracking_mode=mode, verbose=[])
        return model, {"total_particles": 200_000, "seed": 1}

    ms, ss, mw = _collision_flux_match(build)
    assert mw > 0.0, "no D1S decay photons transported under Woodcock"
    assert ms > 0.0
    assert abs(ms - mw) < 3.0 * ss, (
        f"D1S decay-photon Woodcock flux {mw:.6e} vs surface {ms:.6e} "
        f"exceeds 3σ ({3.0 * ss:.2e})"
    )


def test_woodcock_urr_matches_surface():
    """URR-bearing material (Co58) under Woodcock matches surface within
    3σ once the URR band is held across delta-collisions at a given
    energy (preserving resonance self-shielding). Without that, the flux
    was badly under-estimated (Co58: ~0.94x at 500 keV, ~0.25x at 5 keV)
    because re-sampling the band per delta-collision makes the
    free-flight distance track the mean cross section instead of the
    sampled band."""

    def build(mode):
        sphere = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=30.0,
                             boundary="vacuum")
        material = yamc.Material(composition={"Co58": 1.0}, density=8.9,
                                 temperature=294)
        material.read_nuclear_data({"Co58": "tests/Co58.arrow"})
        cell = yamc.Cell(name="co", region=sphere.below, material=material, id=1)
        # 50 keV source -- inside Co58's unresolved resonance range.
        source = yamc.NeutronSource(position=[0.0, 0.0, 0.0],
                                    energy=yamc.sources.Discrete([5.0e4], [1.0]))
        tally = yamc.Tally(scores=["flux"], name="flux", cells=cell,
                           estimator="collision")
        model = yamc.Model(geometry=yamc.Geometry(cells=[cell]),
                           tallies=[tally], source=source,
                           tracking_mode=mode, verbose=[])
        return model, {"total_particles": 40_000, "seed": 5}

    ms, ss, mw = _collision_flux_match(build)
    assert ms > 0.0 and mw > 0.0
    assert abs(ms - mw) < 3.0 * ss, (
        f"URR Woodcock flux {mw:.6e} vs surface {ms:.6e} "
        f"exceeds 3σ ({3.0 * ss:.2e})"
    )


def test_woodcock_disjoint_bodies_gap_is_zero():
    """Woodcock flights terminate at true vacuum exits (issue #360).

    Two disjoint vacuum-bounded Be9 spheres with a +x beam from the
    centre of the first: every mesh bin past body A's boundary (x=3)
    must be exactly zero under woodcock, exactly as under surface
    tracking. Before the fix, flights tunneled across the gap into
    body B and the track-length scorer deposited into gap voxels."""

    def build(mode):
        s_a = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=3.0,
                          boundary="vacuum")
        s_b = yamc.Sphere(x0=20.0, y0=0.0, z0=0.0, radius=3.0,
                          boundary="vacuum")
        mat_a = yamc.Material(composition={"Be9": 1.0}, density=1.85,
                              temperature=294)
        mat_a.read_nuclear_data({"Be9": "tests/Be9.arrow"})
        mat_b = yamc.Material(composition={"Be9": 1.0}, density=1.85,
                              temperature=294)
        mat_b.read_nuclear_data({"Be9": "tests/Be9.arrow"})
        cell_a = yamc.Cell(name="a", region=s_a.below, material=mat_a, id=1)
        cell_b = yamc.Cell(name="b", region=s_b.below, material=mat_b, id=2)
        source = yamc.NeutronSource(
            position=[0.0, 0.0, 0.0],
            direction=yamc.sources.Monodirectional([1.0, 0.0, 0.0]),
            energy=yamc.sources.Discrete([14.06e6], [1.0]))
        mesh = yamc.RegularRectangularMesh(
            lower_left=[-2.0, -0.5, -0.5],
            upper_right=[22.0, 0.5, 0.5],
            shape=[24, 1, 1])
        tally = yamc.Tally(scores=["flux"], name="beam", mesh=mesh)
        model = yamc.Model(geometry=yamc.Geometry(cells=[cell_a, cell_b]),
                           tallies=[tally], source=source,
                           tracking_mode=mode, verbose=[])
        return model, {"total_particles": 20_000, "seed": 42}

    for mode in ("surface", "woodcock"):
        model, run_kwargs = build(mode)
        results = model.simulate_transport(**run_kwargs)
        mean = results["beam"].mean
        # Bins fully past body A's boundary: x >= 4 -> bins 6..24.
        past_boundary = mean[6:]
        assert all(v == 0.0 for v in past_boundary), (
            f"{mode}: nonzero flux past the body-A vacuum boundary: "
            f"{[(i + 6, v) for i, v in enumerate(past_boundary) if v != 0.0][:4]}"
        )
        assert sum(mean[:5]) > 0.0, f"{mode}: no flux inside body A"
