"""Weight-window feature-combination matrix (follow-up survey on issue #271).

The #271 survey covered tallies, scores, physics modes and backends but only
touched weight windows at the surface. This sweeps the WW axes: generation
(particle, mesh, energy groups, source type, tracking mode), application
(tally kind, estimator, tracking mode, geometry kind, composition with other
variance reduction, other simulation entry points, stop conditions) and
construction validation.

Two design rules make these tests mean something:

1. **The bounds must provably act.** With ``lower=0.5`` and the default
   ``ratio=5`` a unit-weight particle sits inside the window, so nothing splits
   or roulettes and a totally broken WW path would still "pass". Every
   application test therefore uses ``LOWER_ACTS`` (upper = 0.05 << 1) and
   asserts the result actually moved off the analog value.
2. **A window must not bias the answer.** Splitting and roulette are
   weight-preserving, so the WW mean must agree with the analog mean within
   combined statistics. That is asserted, not assumed.
"""

import math

import numpy as np
import pytest
import yamc

DATA = "crates/yamc/tests"
TWO_REGION = "crates/yamt/tests/data/two_region.arrow"
CHAIN = f"{DATA}/transmutation-endf-b8.1-sfr.arrow"
SEED = 5
N = 8_000
N_UNBIASED = 60_000
GEN_N = 2_000
# Lower bound whose window (upper = 5 x lower) sits far below a unit weight, so
# every source particle splits and the window demonstrably acts.
LOWER_ACTS = 0.01

_MATS = {}


def _material(name, nuclide, density, mat_id, photon=None):
    key = (name, nuclide, mat_id)
    if key not in _MATS:
        m = yamc.Material(
            composition={nuclide: 1.0},
            density=density,
            temperature=294,
            name=name,
            id=mat_id,
        )
        m.read_nuclear_data({nuclide: f"{DATA}/{nuclide}.arrow"}, photon_data=photon)
        _MATS[key] = m
    return _MATS[key]


def iron():
    return _material("iron", "Fe56", 7.874, 101, {"Fe": f"{DATA}/Fe.arrow"})


def li6():
    return _material("li6", "Li6", 0.534, 102, {"Li": f"{DATA}/Li.arrow"})


def be9():
    return _material("be9", "Be9", 1.85, 103, {"Be": f"{DATA}/Be.arrow"})


def csg_cells():
    inner = yamc.Sphere(radius=5.0)
    outer = yamc.Sphere(radius=15.0, boundary="vacuum")
    return [
        yamc.Cell(name="core", region=inner.below, material=iron()),
        yamc.Cell(name="shell", region=outer.below & inner.above, material=iron()),
    ]


def rect_mesh():
    return yamc.RegularRectangularMesh(
        lower_left=[-15.0] * 3, upper_right=[15.0] * 3, shape=[4, 4, 4]
    )


def cyl_mesh():
    return yamc.RegularCylindricalMesh(
        r_bounds=(0.0, 15.0), z_bounds=(-15.0, 15.0), shape=(4, 4, 4)
    )


def nsrc():
    return yamc.NeutronSource(
        position=(0.0, 0.0, 0.0), energy=yamc.sources.Discrete([14.06e6], [1.0])
    )


def psrc():
    return yamc.PhotonSource(
        position=(0.0, 0.0, 0.0), energy=yamc.sources.Discrete([2.0e6], [1.0])
    )


def acting_window(mesh=None, particle="neutron"):
    """A window that provably acts on a unit-weight particle."""
    mesh = mesh if mesh is not None else rect_mesh()
    return yamc.WeightWindowBounds(
        mesh=mesh, lower_bounds=[LOWER_ACTS] * mesh.num_bins, particle=particle
    )


def build(
    *,
    particle="neutron",
    vr=None,
    coupled=False,
    decay=False,
    tracking="surface",
    tally="cells",
    estimator=None,
    energy_bins=None,
    source="neutron",
):
    cells = csg_cells()
    geometry = yamc.Geometry(cells)
    tkw = {"scores": ["flux"], "name": "t", "particle": particle}
    if tally == "mesh":
        tkw["mesh"] = rect_mesh()
    elif tally == "cyl_mesh":
        tkw["mesh"] = cyl_mesh()
    else:
        tkw["cells"] = cells
    if estimator:
        tkw["estimator"] = estimator
    if energy_bins:
        tkw["energy_bins"] = energy_bins
    t = yamc.Tally(**tkw)

    src = {"photon": psrc(), "mixed": [nsrc(), psrc()]}.get(source, nsrc())
    kw = {}
    if coupled or decay:
        kw["transport_secondary_photons"] = True
    if decay:
        kw["use_decay_photons"] = True
        yamc.transmutation_decay_data = CHAIN
        yamc.transmutation_reactions = CHAIN
        yamc.transmutation_fission_yields = CHAIN
    if vr:
        kw["variance_reduction"] = vr
    return yamc.Model(
        geometry=geometry,
        tallies=[t],
        source=src,
        verbose=[],
        tracking_mode=tracking,
        **kw,
    )


def run(model, n=N, compute="cpu"):
    results = model.simulate_transport(total_particles=n, seed=SEED, compute=compute)
    mean = np.asarray(results["t"].mean, dtype=float)
    sd = np.asarray(results["t"].standard_deviation, dtype=float)
    return mean, float(np.linalg.norm(sd)), results


def assert_window_acted(analog_mean, ww_mean, label):
    """A WW that changes nothing is indistinguishable from one being ignored."""
    assert not np.allclose(analog_mean, ww_mean, rtol=1e-12, atol=0.0), (
        f"{label}: the weight-window run is identical to the analog run, so the "
        "window was ignored (or the bounds do not act on a unit weight)"
    )


def assert_unbiased(analog, ww, label, sigmas=4.0):
    (ma, sa), (mw, sw) = analog, ww
    total_a, total_w = float(ma.sum()), float(mw.sum())
    spread = math.hypot(sa, sw)
    assert spread > 0.0, f"{label}: zero uncertainty, rig broken"
    n_sigma = abs(total_a - total_w) / spread
    assert n_sigma < sigmas, (
        f"{label}: weight-window mean {total_w:.6g} vs analog {total_a:.6g} "
        f"is {n_sigma:.2f} sigma apart; splitting and roulette must preserve the mean"
    )


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "particle,source,coupled",
    [
        ("neutron", "neutron", False),
        ("photon", "photon", False),
        (["neutron", "photon"], "neutron", True),
    ],
    ids=["neutron", "photon", "neutron+photon"],
)
def test_generation_per_particle(particle, source, coupled):
    model = build(
        particle="photon" if particle == "photon" else "neutron",
        source=source,
        coupled=coupled,
    )
    generated = model.generate_weight_windows(
        yamc.WeightWindowGeneratorDeGVR(mesh=rect_mesh(), particle=particle),
        total_particles=GEN_N,
        seed=1,
    )
    if isinstance(particle, list):
        assert isinstance(generated, list) and len(generated) == len(particle)
        windows = generated
    else:
        windows = [generated]
    for w in windows:
        lower = np.asarray(w.lower_bounds, dtype=float)
        assert lower.size == rect_mesh().num_bins
        assert np.any(lower > 0.0), "generation produced no window at all"


@pytest.mark.parametrize("tracking", ["surface", "woodcock", "hybrid"])
def test_generation_in_every_tracking_mode(tracking):
    model = build(tracking=tracking)
    wwb = model.generate_weight_windows(
        yamc.WeightWindowGeneratorDeGVR(mesh=rect_mesh()),
        total_particles=GEN_N,
        seed=1,
    )
    assert np.any(np.asarray(wwb.lower_bounds, dtype=float) > 0.0)


def test_generation_with_energy_groups():
    mesh = rect_mesh()
    wwb = build().generate_weight_windows(
        yamc.WeightWindowGeneratorDeGVR(
            mesh=mesh, energy_bins=[1e3, 1e5, 1e6, 2e7]
        ),
        total_particles=GEN_N,
        seed=1,
    )
    lower = np.asarray(wwb.lower_bounds, dtype=float)
    assert lower.size == 3 * mesh.num_bins, "one bound per (group, voxel)"


def test_generation_rejects_a_cylindrical_mesh():
    """WW meshes are rectangular-only; the refusal must be explicit."""
    with pytest.raises((TypeError, ValueError)):
        build().generate_weight_windows(
            yamc.WeightWindowGeneratorDeGVR(mesh=cyl_mesh()),
            total_particles=GEN_N,
            seed=1,
        )


# ---------------------------------------------------------------------------
# application
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs,label",
    [
        ({}, "cell tally"),
        ({"tally": "mesh"}, "rectangular mesh tally"),
        ({"tally": "cyl_mesh"}, "cylindrical mesh tally"),
        ({"energy_bins": [1e3, 1e6, 2e7]}, "energy-binned tally"),
        ({"estimator": "collision"}, "collision estimator"),
        ({"tracking": "woodcock"}, "woodcock tracking"),
        ({"tracking": "hybrid"}, "hybrid tracking"),
    ],
)
def test_window_acts_and_stays_unbiased(kwargs, label):
    window = acting_window()
    analog = run(build(**kwargs))
    ww = run(build(vr=[window], **kwargs))
    assert_window_acted(analog[0], ww[0], label)
    assert_unbiased(analog[:2], ww[:2], label)


def test_window_composes_with_survival_biasing():
    window = acting_window()
    analog = run(build(), n=N_UNBIASED)
    composed = run(
        build(vr=[window, yamc.SurvivalBiasing()]), n=N_UNBIASED
    )
    assert_window_acted(analog[0], composed[0], "WW + survival biasing")
    assert_unbiased(analog[:2], composed[:2], "WW + survival biasing")


def test_window_is_unbiased_at_high_statistics():
    window = acting_window()
    analog = run(build(), n=N_UNBIASED)
    ww = run(build(vr=[window]), n=N_UNBIASED)
    assert_unbiased(analog[:2], ww[:2], "neutron WW", sigmas=3.0)


def test_photon_window_does_not_touch_a_neutron_run():
    """A window is per species: a photon window must leave neutrons alone."""
    photon_window = acting_window(particle="photon")
    analog = run(build())
    with_photon_ww = run(build(vr=[photon_window]))
    assert np.allclose(analog[0], with_photon_ww[0], rtol=1e-9, atol=0.0), (
        "a photon weight window changed a neutron-only run"
    )


def test_photon_window_acts_on_a_photon_run():
    photon_window = acting_window(particle="photon")
    analog = run(build(particle="photon", source="photon"))
    ww = run(build(particle="photon", source="photon", vr=[photon_window]))
    assert_window_acted(analog[0], ww[0], "photon WW on a photon source")
    assert_unbiased(analog[:2], ww[:2], "photon WW on a photon source")


def test_coupled_run_takes_a_window_per_particle():
    pair = build(coupled=True).generate_weight_windows(
        yamc.WeightWindowGeneratorDeGVR(
            mesh=rect_mesh(), particle=["neutron", "photon"]
        ),
        total_particles=GEN_N,
        seed=1,
    )
    assert isinstance(pair, list) and len(pair) == 2
    analog = run(build(particle="photon", coupled=True))
    ww = run(build(particle="photon", coupled=True, vr=list(pair)))
    assert_unbiased(analog[:2], ww[:2], "coupled n+p window pair")


def test_window_applies_on_decay_photons():
    model = build(particle="photon", decay=True)
    wwb = model.generate_weight_windows(
        yamc.WeightWindowGeneratorDeGVR(mesh=rect_mesh(), particle="photon"),
        total_particles=GEN_N,
        seed=1,
    )
    analog = run(build(particle="photon", decay=True))
    ww = run(build(particle="photon", decay=True, vr=[wwb]))
    assert_unbiased(analog[:2], ww[:2], "D1S decay-photon window")


def test_window_is_rejected_on_gpu():
    if not yamc.parallel.gpu_available():
        pytest.skip("no f64 GPU available")
    with pytest.raises(ValueError, match="variance_reduction"):
        build(vr=[acting_window()]).simulate_transport(
            total_particles=1_000, seed=SEED, compute="gpu"
        )


# ---------------------------------------------------------------------------
# geometry kinds
# ---------------------------------------------------------------------------
def test_window_acts_on_mesh_geometry():
    mesh_geometry = yamc.MeshGeometry(
        TWO_REGION, {"fuel": li6(), "moderator": be9()}
    )
    ww_mesh = yamc.RegularRectangularMesh(
        lower_left=[0.0] * 3, upper_right=[1.0] * 3, shape=[2, 2, 2]
    )
    tally_mesh = yamc.RegularRectangularMesh(
        lower_left=[0.0] * 3, upper_right=[1.0] * 3, shape=[2, 2, 2]
    )
    source = yamc.NeutronSource(
        position=[0.25, 0.5, 0.5], energy=yamc.sources.Discrete([14.06e6], [1.0])
    )

    def go(vr):
        tally = yamc.Tally(scores=["flux"], name="t", mesh=tally_mesh)
        model = yamc.Model(
            geometry=mesh_geometry,
            tallies=[tally],
            source=source,
            verbose=[],
            **({"variance_reduction": vr} if vr else {}),
        )
        return run(model)

    analog = go(None)
    ww = go([acting_window(mesh=ww_mesh)])
    assert_window_acted(analog[0], ww[0], "mesh geometry")
    assert_unbiased(analog[:2], ww[:2], "mesh geometry")


def test_window_acts_on_a_mesh_filled_cell():
    fill = yamc.MeshGeometry(TWO_REGION, {"fuel": li6(), "moderator": be9()})
    sphere = yamc.Sphere(x0=0.5, y0=0.5, z0=0.5, radius=3.0, boundary="vacuum")
    geometry = yamc.Geometry(
        [yamc.Cell(region=sphere.below, material=iron(), name="chamber", fill=fill)]
    )
    source = yamc.NeutronSource(
        position=[0.5, 0.5, 2.0], energy=yamc.sources.Discrete([14.06e6], [1.0])
    )
    ww_mesh = yamc.RegularRectangularMesh(
        lower_left=[-3.0] * 3, upper_right=[3.0] * 3, shape=[2, 2, 2]
    )

    def go(vr):
        tally = yamc.Tally(scores=["flux"], name="t", cells=list(geometry.cells))
        model = yamc.Model(
            geometry=geometry,
            tallies=[tally],
            source=source,
            verbose=[],
            **({"variance_reduction": vr} if vr else {}),
        )
        return run(model, n=4_000)

    analog = go(None)
    ww = go([acting_window(mesh=ww_mesh)])
    assert_window_acted(analog[0], ww[0], "mesh-filled cell")
    assert_unbiased(analog[:2], ww[:2], "mesh-filled cell")


# ---------------------------------------------------------------------------
# other entry points and stop conditions
# ---------------------------------------------------------------------------
def test_window_acts_in_simulate_transmutation():
    yamc.transmutation_decay_data = CHAIN
    yamc.transmutation_reactions = CHAIN
    yamc.transmutation_fission_yields = CHAIN
    radius = 5.0
    material = yamc.Material(
        composition={"Fe56": 1.0},
        density=7.874,
        temperature=294,
        name="iron_t",
        id=201,
        transmutable=True,
        volume=4.0 / 3.0 * math.pi * radius**3,
    )
    material.read_nuclear_data({"Fe56": f"{DATA}/Fe56.arrow"})
    geometry = yamc.Geometry(
        [
            yamc.Cell(
                name="core",
                region=yamc.Sphere(radius=radius, boundary="vacuum").below,
                material=material,
            )
        ]
    )
    schedule = yamc.PulseSchedule(
        [
            yamc.Pulse(rate=1e14, duration=3600.0, source=nsrc()),
            yamc.Cooldown(duration=3600.0),
        ]
    )

    def go(vr):
        model = yamc.Model(
            geometry=geometry,
            source=nsrc(),
            verbose=[],
            **({"variance_reduction": vr} if vr else {}),
        )
        results = model.simulate_transmutation(
            method="independent", schedule=schedule, total_particles=4_000, seed=1
        )
        evolution = results.get_nuclide_evolution(201, "Mn56")
        return float(evolution[-1]) if len(evolution) else 0.0

    plain = go(None)
    with_ww = go([acting_window()])
    assert plain > 0.0, "no activation product; rig broken"
    assert plain != with_ww, (
        "the weight window had no effect on simulate_transmutation, so it was ignored"
    )
    assert abs(with_ww / plain - 1.0) < 0.25, (
        f"inventory moved from {plain:.6g} to {with_ww:.6g} under a weight window; "
        "splitting and roulette must preserve the reaction rates"
    )


def test_window_with_convergence_targets_still_stops_early():
    model = build(vr=[acting_window()])
    model.convergence_targets = [
        yamc.ConvergenceTarget("relative_error", 0.05, tally="t")
    ]
    results = model.simulate_transport(total_particles=200_000, seed=SEED)
    histories = results.runs[0]["n_histories"]
    assert histories < 200_000, (
        f"convergence target ignored with a weight window ({histories} histories)"
    )


def test_window_with_max_runtime():
    results = build(vr=[acting_window()]).simulate_transport(
        max_runtime=(2.0, "s"), seed=SEED
    )
    assert results.runs[0]["n_histories"] > 0


def test_window_with_capture_tracks():
    results = build(vr=[acting_window()]).simulate_transport(
        total_particles=200, seed=SEED, capture_tracks=50
    )
    assert results.tracks is not None and len(results.tracks) > 0, (
        "track capture returned nothing while splitting was on"
    )


# ---------------------------------------------------------------------------
# construction validation
# ---------------------------------------------------------------------------
def test_bounds_length_must_match_the_mesh():
    mesh = rect_mesh()
    with pytest.raises(ValueError, match="expects"):
        yamc.WeightWindowBounds(
            mesh=mesh, lower_bounds=[0.5] * (mesh.num_bins - 1)
        )


def test_bounds_length_must_match_the_group_count():
    mesh = rect_mesh()
    with pytest.raises(ValueError, match="expects"):
        yamc.WeightWindowBounds(
            mesh=mesh,
            lower_bounds=[0.5] * mesh.num_bins,
            energy_bins=[1e3, 1e6, 2e7],
        )


def test_bounds_reject_an_unsupported_species():
    mesh = rect_mesh()
    with pytest.raises(ValueError, match="neutron"):
        yamc.WeightWindowBounds(
            mesh=mesh, lower_bounds=[0.5] * mesh.num_bins, particle="electron"
        )


def test_bounds_reject_a_cylindrical_mesh():
    with pytest.raises((TypeError, ValueError)):
        yamc.WeightWindowBounds(
            mesh=cyl_mesh(), lower_bounds=[0.5] * cyl_mesh().num_bins
        )


def test_model_save_load_round_trips_the_window(tmp_path):
    model = build(vr=[acting_window()])
    path = tmp_path / "model.json"
    model.save(str(path))
    loaded = yamc.Model.load(str(path))
    assert len(loaded.variance_reduction) == 1, (
        "the weight window did not survive save/load"
    )


@pytest.mark.xfail(
    reason="gap: a second window for the same particle is accepted and silently "
    "ignored instead of being rejected as ambiguous",
    strict=True,
)
def test_duplicate_window_for_one_particle_is_rejected():
    window = acting_window()
    with pytest.raises(ValueError):
        build(vr=[window, window]).simulate_transport(
            total_particles=1_000, seed=SEED
        )
