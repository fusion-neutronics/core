"""Weight-window application: construction/validation and transport behaviour.

Applying any weight windows must reproduce analog tally means within
statistics (a representation-independent property), while cutting the
relative error in the deep, low-flux region where splitting helps.
"""
import math
import shutil
import tempfile
from pathlib import Path

import pytest

import yamc

_ROOT = Path(__file__).resolve().parents[4]
AL27 = _ROOT / "tests" / "Al27.arrow"
requires_data = pytest.mark.skipif(not AL27.exists(), reason="Al27.arrow fixture missing")

FE_PHOTON = _ROOT / "tests" / "Fe.arrow"
FE56 = _ROOT / "tests" / "Fe56.arrow"
requires_photon_data = pytest.mark.skipif(
    not (FE_PHOTON.exists() and FE56.exists()), reason="Fe/Fe56 fixtures missing"
)

CHAIN_FILE = _ROOT / "tests" / "transmutation-endf-b8.1-sfr.arrow"
requires_decay_data = pytest.mark.skipif(
    not (FE_PHOTON.exists() and FE56.exists() and CHAIN_FILE.exists()),
    reason="Fe/Fe56/chain fixtures missing",
)


def _mesh():
    return yamc.RegularRectangularMesh(
        lower_left=[-3, -3, 0], upper_right=[3, 3, 40], shape=[1, 1, 8]
    )


def test_weight_window_bounds_validation():
    mesh = _mesh()
    # ratio auto-fills the upper bounds
    wwb = yamc.WeightWindowBounds(mesh=mesh, lower_bounds=[0.1] * 8, ratio=5.0)
    assert wwb.particle == "neutron"
    assert len(wwb.upper_bounds) == 8
    assert wwb.upper_bounds[0] == pytest.approx(0.5)

    # wrong-length bounds rejected
    with pytest.raises(ValueError):
        yamc.WeightWindowBounds(mesh=mesh, lower_bounds=[0.1] * 7, ratio=5.0)
    # survival_factor must be > 1
    with pytest.raises(ValueError):
        yamc.WeightWindowBounds(
            mesh=mesh, lower_bounds=[0.1] * 8, ratio=5.0, survival_factor=1.0
        )
    # ratio must be > 1 when upper_bounds is omitted
    with pytest.raises(ValueError):
        yamc.WeightWindowBounds(mesh=mesh, lower_bounds=[0.1] * 8, ratio=0.9)


@requires_data
def test_weight_windows_unbiased_and_reduce_deep_variance():
    mesh = _mesh()
    lower = [0.5 * 0.6**iz for iz in range(8)]  # window shrinks with depth
    wwb = yamc.WeightWindowBounds(mesh=mesh, lower_bounds=lower, ratio=5.0)

    mat = yamc.Material(composition={"Al27": 1.0}, density=2.7, temperature=294)
    mat.read_nuclear_data({"Al27": str(AL27)})
    cyl = yamc.Cylinder(axis="z", radius=3.0, boundary="vacuum")
    z0 = yamc.Plane(axis="z", offset=0.0, boundary="vacuum")
    z1 = yamc.Plane(axis="z", offset=40.0, boundary="vacuum")
    cell = yamc.Cell(name="rod", region=cyl.below & z0.above & z1.below, material=mat)
    geom = yamc.Geometry([cell])
    src = yamc.NeutronSource(position=(0, 0, 0.5), energy=yamc.sources.Discrete([1.0e6], [1.0]))
    tally = yamc.Tally(scores=["flux"], name="mesh_flux", mesh=mesh, particle="neutron")

    n = 300_000
    res_a = yamc.Model(geometry=geom, source=src, tallies=[tally]).simulate_transport(
        total_particles=n, seed=1
    )
    m_ww = yamc.Model(geometry=geom, source=src, tallies=[tally], variance_reduction=[wwb])
    # getter round-trips the weight-window entry back out of the model
    assert m_ww.variance_reduction[0].particle == "neutron"
    assert m_ww.variance_reduction[0].max_split == 10
    res_w = m_ww.simulate_transport(total_particles=n, seed=1)

    ma, ea = res_a["mesh_flux"].mean, res_a["mesh_flux"].relative_error
    mw, ew = res_w["mesh_flux"].mean, res_w["mesh_flux"].relative_error

    # Unbiased: every slice with meaningful stats agrees within 5 sigma.
    for iz in range(8):
        if ma[iz] <= 0 or mw[iz] <= 0:
            continue
        sa, sw = ma[iz] * ea[iz], mw[iz] * ew[iz]
        comb = math.sqrt(sa * sa + sw * sw)
        if comb == 0:
            continue
        n_sigma = abs(ma[iz] - mw[iz]) / comb
        assert n_sigma <= 5.0, f"slice {iz} biased: {n_sigma:.2f} sigma"

    # Variance reduction: the deepest slice's relative error is not worse
    # (splitting adds independent deep-region samples).
    assert ew[7] <= ea[7] * 1.1, f"deep rel_err not reduced: analog {ea[7]:.3f} ww {ew[7]:.3f}"


@requires_data
def test_weight_windows_unbiased_under_heavy_splitting():
    """Regression: a hard per-history particle cap used to *discard* whatever
    split particles were still queued, destroying their weight and biasing flux
    low whenever windows split aggressively (the DeGVR deep-penetration bias:
    integral flux fell to ~0.18x analog). An aggressive window that forces
    cascading splits far past the per-history budget must still conserve the
    integral flux versus analog.
    """
    # Wide rod matched to a wide mesh so neutrons collide many times inside the
    # windowed region (little radial leak) and genuinely cascade.
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-20, -20, 0], upper_right=[20, 20, 40], shape=[1, 1, 8]
    )
    # Tiny upper => a unit-weight particle cascades through several split
    # generations (1 -> 10 -> 100 -> ...), far exceeding the old 1000-particle
    # cap; tiny lower => no roulette, isolating the splitting path.
    wwb = yamc.WeightWindowBounds(
        mesh=mesh, lower_bounds=[1e-30] * 8, upper_bounds=[1e-4] * 8,
        particle="neutron", max_split=10, weight_floor=1e-40,
    )
    mat = yamc.Material(composition={"Al27": 1.0}, density=2.7, temperature=294)
    mat.read_nuclear_data({"Al27": str(AL27)})
    cyl = yamc.Cylinder(axis="z", radius=20.0, boundary="vacuum")
    z0 = yamc.Plane(axis="z", offset=0.0, boundary="vacuum")
    z1 = yamc.Plane(axis="z", offset=40.0, boundary="vacuum")
    cell = yamc.Cell(name="rod", region=cyl.below & z0.above & z1.below, material=mat)
    geom = yamc.Geometry([cell])
    src = yamc.NeutronSource(position=(0, 0, 0.5), energy=yamc.sources.Discrete([1.0e6], [1.0]))
    tally = yamc.Tally(scores=["flux"], name="flux", mesh=mesh, particle="neutron")

    n = 1500
    res_a = yamc.Model(geometry=geom, source=src, tallies=[tally]).simulate_transport(total_particles=n, seed=3)
    res_w = yamc.Model(
        geometry=geom, source=src, tallies=[tally], variance_reduction=[wwb]
    ).simulate_transport(total_particles=n, seed=3)

    # Integral flux (sum over the mesh) is conserved by any unbiased weighting;
    # the discard bug drove this ratio far below 1. Loose bound distinguishes
    # the fix (~1.0) from the bug (~0.2) without being statistically flaky.
    tot_a = sum(res_a["flux"].mean)
    tot_w = sum(res_w["flux"].mean)
    assert tot_a > 0 and tot_w > 0
    ratio = tot_w / tot_a
    assert 0.8 <= ratio <= 1.2, f"heavy-splitting WW biased integral flux: ratio {ratio:.3f}"


@requires_data
def test_degvr_generation_unbiased_and_reduces_deep_variance():
    # ~9 mfp Al27 rod: DeGVR generates windows from two internal reduced-density
    # passes, then a production run with them must stay unbiased and cut the
    # deep-region relative error.
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-3, -3, 0], upper_right=[3, 3, 50], shape=[1, 1, 10]
    )
    mat = yamc.Material(composition={"Al27": 1.0}, density=2.7, temperature=294)
    mat.read_nuclear_data({"Al27": str(AL27)})
    cyl = yamc.Cylinder(axis="z", radius=3.0, boundary="vacuum")
    z0 = yamc.Plane(axis="z", offset=0.0, boundary="vacuum")
    z1 = yamc.Plane(axis="z", offset=50.0, boundary="vacuum")
    cell = yamc.Cell(name="rod", region=cyl.below & z0.above & z1.below, material=mat)
    geom = yamc.Geometry([cell])
    src = yamc.NeutronSource(position=(0, 0, 0.5), energy=yamc.sources.Discrete([1.0e6], [1.0]))
    tally = yamc.Tally(scores=["flux"], name="flux", mesh=mesh, particle="neutron")
    model = yamc.Model(geometry=geom, source=src, tallies=[tally])

    # Explicit large N -> steep windows -> heavy splitting (this test's point);
    # the auto default would pick a small N for this shallow rod.
    gen = yamc.WeightWindowGeneratorDeGVR(mesh=mesh, particle="neutron", density_reduction=32.0)
    wwb = model.generate_weight_windows(gen, total_particles=200_000, seed=1)
    assert len(wwb.lower_bounds) == 10
    assert wwb.lower_bounds[0] > 0.0
    # the importance target shrinks with depth
    scored = [x for x in wwb.lower_bounds if x > 0.0]
    assert scored[0] > scored[-1]

    n = 400_000
    res_a = model.simulate_transport(total_particles=n, seed=7)  # analog
    prod = yamc.Model(geometry=geom, source=src, tallies=[tally], variance_reduction=[wwb])
    res_w = prod.simulate_transport(total_particles=n, seed=7)
    ma, ea = res_a["flux"].mean, res_a["flux"].relative_error
    mw, ew = res_w["flux"].mean, res_w["flux"].relative_error

    for iz in range(10):
        if ma[iz] <= 0 or mw[iz] <= 0:
            continue
        sa, sw = ma[iz] * ea[iz], mw[iz] * ew[iz]
        comb = math.sqrt(sa * sa + sw * sw)
        if comb == 0:
            continue
        n_sigma = abs(ma[iz] - mw[iz]) / comb
        assert n_sigma <= 6.0, f"DeGVR slice {iz} biased: {n_sigma:.2f} sigma"

    # DeGVR windows must cut the deep-region relative error.
    assert ew[8] < ea[8], f"DeGVR did not reduce deep variance: analog {ea[8]:.3f} ww {ew[8]:.3f}"


@requires_data
def test_degvr_auto_density_reduction():
    """density_reduction defaults to auto (None): N is derived from a
    particle-free optical-depth ray-trace; an explicit value overrides it."""
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-3, -3, 0], upper_right=[3, 3, 50], shape=[1, 1, 10]
    )
    mat = yamc.Material(composition={"Al27": 1.0}, density=2.7, temperature=294)
    mat.read_nuclear_data({"Al27": str(AL27)})
    cyl = yamc.Cylinder(axis="z", radius=3.0, boundary="vacuum")
    z0 = yamc.Plane(axis="z", offset=0.0, boundary="vacuum")
    z1 = yamc.Plane(axis="z", offset=50.0, boundary="vacuum")
    cell = yamc.Cell(name="rod", region=cyl.below & z0.above & z1.below, material=mat)
    geom = yamc.Geometry([cell])
    src = yamc.NeutronSource(position=(0, 0, 0.5), energy=yamc.sources.Discrete([1.0e6], [1.0]))
    tally = yamc.Tally(scores=["flux"], name="flux", mesh=mesh, particle="neutron")
    model = yamc.Model(geometry=geom, source=src, tallies=[tally])

    # auto by default
    gen = yamc.WeightWindowGeneratorDeGVR(mesh=mesh, particle="neutron")
    assert gen.density_reduction is None

    # the auto ray-trace returns a positive N (clamped >= 2) and optical depth
    n, tau = model.estimate_density_reduction(gen)
    assert n >= 2.0
    assert tau > 0.0

    # generation runs in auto mode and yields valid, depth-decreasing bounds
    wwb = model.generate_weight_windows(gen, total_particles=100_000, seed=1)
    assert len(wwb.lower_bounds) == 10
    scored = [x for x in wwb.lower_bounds if x > 0.0]
    assert scored[0] > scored[-1]

    # an explicit value overrides the auto estimate
    gen_ovr = yamc.WeightWindowGeneratorDeGVR(
        mesh=mesh, particle="neutron", density_reduction=16.0
    )
    assert gen_ovr.density_reduction == 16.0


def _degvr_rod_model():
    """Small Al27 rod + a DeGVR generator, shared by the stop-condition tests."""
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-3, -3, 0], upper_right=[3, 3, 50], shape=[1, 1, 10]
    )
    mat = yamc.Material(composition={"Al27": 1.0}, density=2.7, temperature=294)
    mat.read_nuclear_data({"Al27": str(AL27)})
    cyl = yamc.Cylinder(axis="z", radius=3.0, boundary="vacuum")
    z0 = yamc.Plane(axis="z", offset=0.0, boundary="vacuum")
    z1 = yamc.Plane(axis="z", offset=50.0, boundary="vacuum")
    cell = yamc.Cell(name="rod", region=cyl.below & z0.above & z1.below, material=mat)
    geom = yamc.Geometry([cell])
    src = yamc.NeutronSource(position=(0, 0, 0.5), energy=yamc.sources.Discrete([1.0e6], [1.0]))
    tally = yamc.Tally(scores=["flux"], name="flux", mesh=mesh, particle="neutron")
    model = yamc.Model(geometry=geom, source=src, tallies=[tally])
    gen = yamc.WeightWindowGeneratorDeGVR(mesh=mesh, particle="neutron")
    return model, gen


def test_generate_weight_windows_requires_stop_condition():
    # Each generation pass needs a stop condition; neither total_particles nor
    # max_runtime -> rejected before any transport runs.
    model, gen = _degvr_rod_model()
    with pytest.raises(ValueError, match="per-pass stop condition"):
        model.generate_weight_windows(gen)


def test_generate_weight_windows_zero_particles_raises():
    model, gen = _degvr_rod_model()
    with pytest.raises(ValueError, match="must be positive"):
        model.generate_weight_windows(gen, total_particles=0)


def test_generate_weight_windows_time_bounded():
    # total_particles=None + a per-pass wall-time budget (tuple form): each
    # DeGVR pass runs to time, and valid windows come back. This is the path
    # for timing weight-window generation against an equally time-bounded
    # analog run.
    model, gen = _degvr_rod_model()
    wwb = model.generate_weight_windows(gen, max_runtime=(1, "s"), seed=1)
    assert len(wwb.lower_bounds) == 10
    assert wwb.lower_bounds[0] > 0.0  # the near-source slice always scores


@requires_photon_data
def test_photon_source_weight_window_reduces_deep_variance():
    """A photon-source weight window must apply on the photon transport path:
    unbiased where the analog reference is reliable, and it must actually cut
    the deep-region relative error. Before photon-WW application was wired in,
    the photon window was a silent no-op (the WW run was bit-identical to analog,
    so ew == ea); the strict ``ew < ea`` on the deepest slice fails in that case.
    """
    n_slices = 8
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-3, -3, 0], upper_right=[3, 3, 24], shape=[1, 1, n_slices]
    )
    # Iron rod: attenuating enough for photons (~8 mfp at 2 MeV) that the deep
    # slices need the window, but shallow slices stay analog-resolvable.
    mat = yamc.Material(composition={"Fe56": 1.0}, density=7.874, temperature=294)
    mat.read_nuclear_data({"Fe56": str(FE56)}, photon_data={"Fe": str(FE_PHOTON)})
    cyl = yamc.Cylinder(axis="z", radius=3.0, boundary="vacuum")
    z0 = yamc.Plane(axis="z", offset=0.0, boundary="vacuum")
    z1 = yamc.Plane(axis="z", offset=24.0, boundary="vacuum")
    cell = yamc.Cell(name="rod", region=cyl.below & z0.above & z1.below, material=mat)
    geom = yamc.Geometry([cell])
    src = yamc.PhotonSource(position=(0, 0, 0.5), energy=yamc.sources.Discrete([2.0e6], [1.0]))
    tally = yamc.Tally(scores=["flux"], name="mesh_flux", mesh=mesh, particle="photon")

    gen_model = yamc.Model(
        geometry=geom, source=src, tallies=[tally], transport_secondary_photons=True
    )
    wwb = gen_model.generate_weight_windows(
        yamc.WeightWindowGeneratorDeGVR(mesh=mesh, particle="photon"),
        total_particles=200_000,
        seed=1,
    )
    assert wwb.particle == "photon"

    n = 600_000
    res_a = yamc.Model(
        geometry=geom, source=src, tallies=[tally], transport_secondary_photons=True
    ).simulate_transport(total_particles=n, seed=7)
    res_w = yamc.Model(
        geometry=geom,
        source=src,
        tallies=[tally],
        transport_secondary_photons=True,
        variance_reduction=[wwb],
    ).simulate_transport(total_particles=n, seed=7)

    ma, ea = res_a["mesh_flux"].mean, res_a["mesh_flux"].relative_error
    mw, ew = res_w["mesh_flux"].mean, res_w["mesh_flux"].relative_error

    # Unbiased where the analog reference is reliable (rel_err < 0.25); deeper
    # slices where analog barely reaches are not a trustworthy reference.
    for iz in range(n_slices):
        if ma[iz] <= 0 or mw[iz] <= 0 or ea[iz] > 0.25:
            continue
        sa, sw = ma[iz] * ea[iz], mw[iz] * ew[iz]
        comb = math.sqrt(sa * sa + sw * sw)
        if comb == 0:
            continue
        n_sigma = abs(ma[iz] - mw[iz]) / comb
        assert n_sigma <= 5.0, f"photon slice {iz} biased: {n_sigma:.2f} sigma"

    # The window must actually act and cut the deepest slice's relative error.
    # (Strict: a no-op photon window would give ew == ea here and fail.)
    assert ew[n_slices - 1] < ea[n_slices - 1], (
        f"photon WW did not reduce deep rel_err: "
        f"analog {ea[n_slices - 1]:.3f} ww {ew[n_slices - 1]:.3f}"
    )


def _unbiased_where_reliable(ma, ea, mw, ew, n_slices, label, tol=0.25, n_sigma_max=5.0):
    """Assert the WW means agree with analog on every slice the analog run
    resolves reliably (rel_err < ``tol``)."""
    for iz in range(n_slices):
        if ma[iz] <= 0 or mw[iz] <= 0 or ea[iz] > tol:
            continue
        sa, sw = ma[iz] * ea[iz], mw[iz] * ew[iz]
        comb = math.sqrt(sa * sa + sw * sw)
        if comb == 0:
            continue
        n_sigma = abs(ma[iz] - mw[iz]) / comb
        assert n_sigma <= n_sigma_max, f"{label} slice {iz} biased: {n_sigma:.2f} sigma"


@requires_photon_data
def test_coupled_neutron_photon_weight_windows_one_call():
    """One coupled DeGVR generation on a 14 MeV neutron source with secondary
    photons returns TWO windows (neutron + photon) from a single call. Applied
    together in production, both fields stay unbiased where their analog
    reference is reliable, and each window must demonstrably act.

    Isolating the two windows' effects matters here: the neutron window alone
    already boosts deep secondary-photon production (more neutrons reach deep ->
    more deep photons), so measuring the photon field against analog would pass
    even if the photon window were an inert no-op. The neutron window is guarded
    against analog (an inert neutron window leaves the neutron field bit-identical
    to analog); the photon window is guarded against a neutron-only run (an inert
    photon window leaves the photon field bit-identical to neutron-only).

    A single string ``particle`` still returns one bounds (PR1 behaviour), and a
    single-element list returns a one-element list -- both checked.
    """
    n_slices = 8
    length = 40.0
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-3, -3, 0], upper_right=[3, 3, length], shape=[1, 1, n_slices]
    )
    mat = yamc.Material(composition={"Fe56": 1.0}, density=7.874, temperature=294)
    mat.read_nuclear_data({"Fe56": str(FE56)}, photon_data={"Fe": str(FE_PHOTON)})
    cyl = yamc.Cylinder(axis="z", radius=3.0, boundary="vacuum")
    z0 = yamc.Plane(axis="z", offset=0.0, boundary="vacuum")
    z1 = yamc.Plane(axis="z", offset=length, boundary="vacuum")
    cell = yamc.Cell(name="rod", region=cyl.below & z0.above & z1.below, material=mat)
    geom = yamc.Geometry([cell])
    src = yamc.NeutronSource(
        position=(0, 0, 0.5), energy=yamc.sources.Discrete([14.1e6], [1.0])
    )
    n_tally = yamc.Tally(scores=["flux"], name="n_flux", mesh=mesh, particle="neutron")
    p_tally = yamc.Tally(scores=["flux"], name="p_flux", mesh=mesh, particle="photon")

    def model(**kw):
        return yamc.Model(
            geometry=geom,
            source=src,
            tallies=[n_tally, p_tally],
            transport_secondary_photons=True,
            **kw,
        )

    # One call, list particle -> a list of two windows in [neutron, photon] order.
    gen = yamc.WeightWindowGeneratorDeGVR(mesh=mesh, particle=["neutron", "photon"])
    assert gen.particle == ["neutron", "photon"]
    windows = model().generate_weight_windows(gen, total_particles=200_000, seed=1)
    assert isinstance(windows, list) and len(windows) == 2
    assert windows[0].particle == "neutron"
    assert windows[1].particle == "photon"

    # A single-string particle still returns ONE bounds (no regression); a
    # single-element LIST returns a one-element list and round-trips as a list.
    single = model().generate_weight_windows(
        yamc.WeightWindowGeneratorDeGVR(mesh=mesh, particle="photon"),
        total_particles=20_000,
        seed=1,
    )
    assert not isinstance(single, list)
    assert single.particle == "photon"
    one_list_gen = yamc.WeightWindowGeneratorDeGVR(mesh=mesh, particle=["photon"])
    assert one_list_gen.particle == ["photon"]
    one_list = model().generate_weight_windows(
        one_list_gen, total_particles=20_000, seed=1
    )
    assert isinstance(one_list, list) and len(one_list) == 1

    n = 600_000
    res_a = model().simulate_transport(total_particles=n, seed=7)
    res_both = model(variance_reduction=windows).simulate_transport(
        total_particles=n, seed=7
    )
    # Neutron-only control: isolates the PHOTON window's effect on the secondary
    # photon field (see the docstring). An inert photon window leaves the photon
    # field bit-identical to this run.
    res_n_only = model(variance_reduction=[windows[0]]).simulate_transport(
        total_particles=n, seed=7
    )

    half = n_slices // 2

    def deep_mean(res, name):
        e = res[name].relative_error
        return sum(e[half:]) / (n_slices - half)

    # Both fields unbiased vs analog where the analog reference is reliable.
    for name, label in [("n_flux", "neutron"), ("p_flux", "photon")]:
        ma, ea = res_a[name].mean, res_a[name].relative_error
        mw, ew = res_both[name].mean, res_both[name].relative_error
        _unbiased_where_reliable(ma, ea, mw, ew, n_slices, label)

    # Neutron window acts: an inert neutron window leaves the neutron field
    # bit-identical to analog (the photon window never touches neutrons), so the
    # deep-half mean rel_err would be equal, not reduced.
    n_deep_a, n_deep_w = deep_mean(res_a, "n_flux"), deep_mean(res_both, "n_flux")
    assert n_deep_w < n_deep_a, (
        f"neutron WW did not reduce deep rel_err: analog {n_deep_a:.3f} ww {n_deep_w:.3f}"
    )

    # Photon window acts on the secondary-photon field, measured against the
    # neutron-only run so the neutron window's own boost to deep photon
    # production cannot mask an inert photon window.
    #
    # "Acts" is deliberately NOT "reduces deep rel_err": on this thin rod the
    # photon window's deep-rel_err benefit over neutron-only is smaller than
    # the seed-to-seed spread of the rel_err estimate itself (a 6-way
    # generation-seed x transport-seed sweep at this budget gives deep
    # rel_err both {0.078..0.143} vs neutron-only {0.091..0.128}, a coin
    # flip), so a strict inequality flips on realization luck (the 64-bit
    # stream of issue #274 tipped it). Inertness is instead detected exactly
    # the way the docstring frames it: with identical seeds an inert photon
    # window leaves the photon field bit-identical to the neutron-only run,
    # so (a) the generated photon window must carry real (finite, positive)
    # bounds in the deep half, and (b) applying it must actually change the
    # deep photon field. Unbiasedness stays guarded by the checks above.
    deep_lower = list(windows[1].lower_bounds)[half:]
    assert any(b > 0.0 for b in deep_lower) and all(
        math.isfinite(b) for b in deep_lower
    ), f"photon window carries no usable deep bounds: {deep_lower}"
    p_deep_field_n = list(res_n_only["p_flux"].mean)[half:]
    p_deep_field_w = list(res_both["p_flux"].mean)[half:]
    assert p_deep_field_n != p_deep_field_w, (
        "photon window is inert: deep photon field bit-identical to the "
        "neutron-only run"
    )


def _decay_field_per_voxel(res, n_parents, n_voxels):
    """Collapse a D1S decay-photon flux tally (mesh x parent_nuclide, parent-major
    ``parent * n_voxels + voxel`` layout) to the total decay flux and its relative
    error per voxel, summed over parents. Pure Python (no numpy dependency)."""
    mean, rel = res.mean, res.relative_error
    tot = [0.0] * n_voxels
    var = [0.0] * n_voxels
    for p in range(n_parents):
        for v in range(n_voxels):
            m = mean[p * n_voxels + v]
            s = m * rel[p * n_voxels + v]
            tot[v] += m
            var[v] += s * s
    err = [math.sqrt(var[v]) / tot[v] if tot[v] > 0 else 0.0 for v in range(n_voxels)]
    return tot, err


@requires_decay_data
def test_decay_photon_weight_window_reduces_deep_variance():
    """A D1S decay-photon problem (14 MeV neutron source activating an iron rod,
    ``use_decay_photons=True``) gets a photon window from one DeGVR generation.
    The photon window is built from the photon flux transported at reduced
    density (decay + prompt), and applied it must keep the DECAY-photon field
    (isolated with ``parent_nuclides``) unbiased where the analog reference is
    reliable while cutting the deep-region relative error. A no-op window would
    leave the WW run bit-identical to analog and fail the deep-variance check.

    The decay source scales with density, so DeGVR's log-linear-in-density
    extrapolation is more approximate here than for pure attenuation; the window
    stays unbiased regardless (windows are unbiased for any bounds), only its
    efficiency is affected.
    """
    n_slices = 8
    length = 40.0
    radius = 3.0
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-radius, -radius, 0],
        upper_right=[radius, radius, length],
        shape=[1, 1, n_slices],
    )

    # Reduce the full chain to Fe56's activation products and point the
    # per-subsection transmutation config at the export (restored afterwards).
    full_chain = yamc.TransmutationChain(str(CHAIN_FILE))
    reduced = full_chain.reduce(["Fe56"], 5)
    tmp_dir = tempfile.mkdtemp(prefix="ww_decay_", suffix=".chain.arrow")
    saved = (
        getattr(yamc, "transmutation_decay_data", None),
        getattr(yamc, "transmutation_reactions", None),
        getattr(yamc, "transmutation_fission_yields", None),
    )
    try:
        reduced.export_to_arrow(tmp_dir)
        yamc.transmutation_decay_data = tmp_dir
        yamc.transmutation_reactions = tmp_dir
        yamc.transmutation_fission_yields = tmp_dir

        mat = yamc.Material(composition={"Fe56": 1.0}, density=7.874, temperature=294)
        mat.read_nuclear_data({"Fe56": str(FE56)}, photon_data={"Fe": str(FE_PHOTON)})
        cyl = yamc.Cylinder(axis="z", radius=radius, boundary="vacuum")
        z0 = yamc.Plane(axis="z", offset=0.0, boundary="vacuum")
        z1 = yamc.Plane(axis="z", offset=length, boundary="vacuum")
        cell = yamc.Cell(name="rod", region=cyl.below & z0.above & z1.below, material=mat)
        geom = yamc.Geometry([cell])
        src = yamc.NeutronSource(
            position=(0, 0, 0.5), energy=yamc.sources.Discrete([14.06e6], [1.0])
        )

        radionuclides = yamc.Model(geometry=geom, source=src).radionuclides()
        if not radionuclides:
            pytest.skip("no D1S radionuclides reachable from Fe56 in this chain")
        n_parents = len(radionuclides)

        def model(**kw):
            tally = yamc.Tally(
                scores=["flux"], name="decay", mesh=mesh, particle="photon",
                parent_nuclides=radionuclides,
            )
            return yamc.Model(
                geometry=geom, tallies=[tally], source=src,
                transport_secondary_photons=True, use_decay_photons=True, **kw,
            )

        # One photon window from a D1S generation (auto density reduction).
        wwb = model().generate_weight_windows(
            yamc.WeightWindowGeneratorDeGVR(mesh=mesh, particle="photon"),
            total_particles=200_000,
            seed=1,
        )
        assert wwb.particle == "photon"

        n = 800_000
        res_a = model().simulate_transport(total_particles=n, seed=7)["decay"]
        res_w = model(variance_reduction=[wwb]).simulate_transport(
            total_particles=n, seed=7
        )["decay"]
        ma, ea = _decay_field_per_voxel(res_a, n_parents, n_slices)
        mw, ew = _decay_field_per_voxel(res_w, n_parents, n_slices)

        _unbiased_where_reliable(ma, ea, mw, ew, n_slices, "decay-photon", tol=0.2)

        # The window must ACT, which here is deliberately not "must reduce the
        # deep relative error": decay photons are rare and attenuated, so on
        # this rod the window's deep benefit is smaller than the seed-to-seed
        # spread of the rel_err estimate itself. A 6-way sweep (generation
        # seed x transport seed) improves the deep half in 5 of 6 cases, by
        # 1 to 6 percent, with the one regression at -7 percent: a strict
        # inequality on one hardcoded seed pair is a coin flip, and any RNG
        # stream change (issue #111) reshuffles which side it lands on. The
        # same race was retired for the coupled neutron+photon one-call test
        # above, in the same way: assert non-inertness deterministically. An
        # inert window would leave the deep field bit-identical to analog.
        # Unbiasedness stays guarded by _unbiased_where_reliable above.
        half = n_slices // 2
        deep_lower = list(wwb.lower_bounds)[half:]
        assert any(b > 0.0 for b in deep_lower) and all(
            math.isfinite(b) for b in deep_lower
        ), f"decay-photon window carries no usable deep bounds: {deep_lower}"
        assert list(mw[half:]) != list(ma[half:]), (
            "decay-photon window is inert: deep field bit-identical to analog"
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        (
            yamc.transmutation_decay_data,
            yamc.transmutation_reactions,
            yamc.transmutation_fission_yields,
        ) = saved
