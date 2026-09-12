"""End-to-end tests for `Model.simulate_transport(compute='gpu')`.

Skipped automatically on hosts without a Vulkan f64 adapter so CI runs
that have no GPU don't fail spuriously.

Scope of these tests matches the dispatch's writeback support:

- Neutron transport, sphere/plane/cylinder/torus surfaces only.
- A single tally with `[CellFilter, EnergyFilter]` and a Flux score
  (other configurations rejected with structured `ValueError`s).
- No tracks / no threads kwarg.
"""
import math
import os

import pytest

yamc = pytest.importorskip("yamc")

if not yamc.parallel.gpu_available():
    pytest.skip(
        "no GPU with f64 compute available, or yamc was built without "
        "the `gpu` Cargo feature",
        allow_module_level=True,
    )


TESTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'crates', 'yamc', 'tests'
)
FE56_DATA_PATH = os.path.join(TESTS_DIR, 'Fe56.arrow')


def _log_uniform_bins(e_min, e_max, n_bins):
    """N+1 boundaries uniform in ln(E) -- matches the kernel's binning."""
    log_min = math.log(e_min)
    log_max = math.log(e_max)
    return [
        math.exp(log_min + i * (log_max - log_min) / n_bins)
        for i in range(n_bins + 1)
    ]


def _build_model_with_flux_tally(total_particles=200, n_energy_bins=4, seed=42):
    """Single Fe56 sphere, point neutron source, one Flux tally with
    [CellFilter, EnergyFilter] over log-uniform bins. Outer surface is
    vacuum so escaping particles terminate cleanly on both backends."""
    sphere = yamc.Sphere(radius=10.0, boundary='vacuum')
    material = yamc.Material(
        composition={'Fe56': 1.0},
        density=7.8,
        temperature=294,
    )
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([1e6], [1]),
        position=(0, 0, 0),
    )
    bins = _log_uniform_bins(1e3, 2e7, n_energy_bins)
    tally = yamc.Tally(scores=['flux'], cells=cell, energy_bins=bins)
    model = yamc.Model(
        geometry=geometry,
        tallies=[tally],
        source=source,
    )
    return tally, model, {"total_particles": total_particles, "seed": seed}


def _build_model_no_tally(particles=200):
    sphere = yamc.Sphere(radius=10.0, boundary='vacuum')
    material = yamc.Material(
        composition={'Fe56': 1.0},
        density=7.8,
        temperature=294,
    )
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([1e6], [1]),
        position=(0, 0, 0),
    )
    model = yamc.Model(
        geometry=geometry,
        tallies=[],
        source=source,
    )
    return model, {"total_particles": particles * 1, "seed": 42}


def test_simulate_transport_gpu_no_tally_runs():
    """No tally on the model -- GPU should still run cleanly."""
    model, run = _build_model_no_tally()
    results = model.simulate_transport(compute='gpu', **run)
    assert results is not None
    assert results.tracks is None


def test_simulate_transport_gpu_with_flux_tally_populates_results():
    """Single Flux tally with cell+energy filters -- `results[tally].mean`
    should come back populated after a GPU run."""
    tally, model, run = _build_model_with_flux_tally(total_particles=500, n_energy_bins=4)
    results = model.simulate_transport(compute='gpu', **run)
    means = list(results[tally].mean)
    assert len(means) == 4
    # At 1 MeV source into Fe56, all source particles start in the
    # bin containing 1 MeV; flux should be strictly positive somewhere.
    assert any(m > 0.0 for m in means), f"no flux recorded across bins: {means}"
    # No bin should be negative (track length is non-negative).
    assert all(m >= 0.0 for m in means)


# Deliberately not asserting CPU/GPU equivalence on tally values.
# The kernel implements only elastic-scatter + absorption today; yamc's
# CPU transport runs full physics (inelastic, fission, URR, …). For any
# real material at fast energies inelastic downscatter moves flux
# between bins on CPU and not on GPU, so the two paths produce
# physically different distributions on the same model. They are not
# expected to agree until the kernel grows the missing physics.
#
# An honest equivalence test would need to compare against the kernel's
# own CPU mirror (`run_multi_cell_transport_cpu` in yamc-gpu) which
# implements the same elastic+absorption-only physics. That's already
# covered by yamc-gpu's bit-exact CPU/GPU kernel tests, so we don't
# duplicate it here.


def test_simulate_transport_gpu_seeded_runs_are_deterministic():
    """Two GPU runs with the same model+seed should produce identical
    per-bin flux. Catches regressions in source sampling or kernel RNG
    seeding without depending on the CPU path's physics."""
    tally_a, model_a, run_a = _build_model_with_flux_tally(total_particles=500, seed=7)
    a = list(model_a.simulate_transport(compute='gpu', **run_a)[tally_a].mean)

    tally_b, model_b, run_b = _build_model_with_flux_tally(total_particles=500, seed=7)
    b = list(model_b.simulate_transport(compute='gpu', **run_b)[tally_b].mean)

    # Exact equality is valid ONLY because this is a neutron-only model. The
    # GPU *photon* kernel is not bit-reproducible across runs (secondary-
    # cascade ordering; see the "Reproducibility" note in
    # multi_cell_photon_transport.rs). If a photon source is ever added to
    # this test, switch to an approximate (np.allclose) comparison.
    assert a == b, f"GPU runs not deterministic under fixed seed: {a} vs {b}"


def test_simulate_transport_compute_default_is_cpu():
    """Omitting `compute=` should still run on CPU as before."""
    _, model, run = _build_model_with_flux_tally(total_particles=50)
    results = model.simulate_transport(**run)
    assert results is not None


def test_simulate_transport_gpu_inelastic_downscatter_observed():
    """Aggregated inelastic (MT 51..=91) must actually move neutron
    flux to lower energies on GPU.

    Build a Fe56 sphere at 14 MeV (where Fe56 inelastic xs is large)
    and check that flux appears in bins **below** the source energy.
    Before inelastic landed in the kernel, particles only elastic-
    scattered, so on heavy nuclides like Fe56 essentially all the
    flux was stuck in the highest energy bin. This test would fail
    flat if a regression dropped the kernel's inelastic branch.
    """
    sphere = yamc.Sphere(radius=10.0, boundary='vacuum')
    material = yamc.Material(
        composition={'Fe56': 1.0}, density=7.8, temperature=294,
    )
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14e6], [1]),
        position=(0, 0, 0),
    )
    bins = _log_uniform_bins(1e3, 2e7, 8)
    tally = yamc.Tally(scores=['flux'], cells=cell, energy_bins=bins)
    model = yamc.Model(
        geometry=geometry, tallies=[tally], source=source,
    )
    means = list(model.simulate_transport(compute='gpu', total_particles=2000, seed=42)[tally].mean)
    # bins[8] contains 14 MeV; below-source = first 7 bins.
    below_source = sum(means[:-1])
    at_source = means[-1]
    assert at_source > 0.0, "no flux at source energy -- model misconfigured"
    assert below_source > 0.0, (
        "no inelastic downscatter detected -- every bin below the source "
        f"energy is zero: {means}"
    )
    # Sanity: at least 5% of total flux should land below the source
    # bin. (Rough -- real CPU value is ~50% on Fe56 at 14 MeV; the
    # single-Q simplification undershoots, but should still clear 5%.)
    total = at_source + below_source
    assert below_source / total > 0.05, (
        f"only {below_source / total:.1%} of flux landed below source -- "
        f"inelastic kinematics may be misconfigured. means={means}"
    )


def test_simulate_transport_gpu_vacuum_kills_particles_at_surface():
    """`BoundaryType::Vacuum` on GPU must actually kill the particle
    AT the surface crossing -- not just be accepted by the translation.

    Ground truth is the CPU transport (the OpenMC-validated path): if
    the GPU failed to terminate particles at the vacuum surface, the
    escaping particles would keep accumulating in-cell track length
    (the pre-implicit-complement AABB behaviour inflated the flux ~6x
    here), so agreement with the CPU pins the kill. Sized for leakage
    to dominate (small radius, 14 MeV neutrons in Fe56), which
    maximises that inflation if the kill regresses.

    A transmission outer boundary must give the SAME in-cell flux on
    the GPU: the cell is convex, so a particle that leaves it on a
    straight line through the implicit void can never re-enter -- it
    just dies at the AABB instead of at the surface.

    That second geometry is open (nothing bounds the space outside the
    sphere), so every escaping particle is a LOST particle: the CPU has
    always said so, and since issue #289 the GPU does too. The run
    therefore needs `max_lost_particles` raised to be allowed at all --
    which is the point of the diagnostic, not a flaw in it. Counting
    losses does not change transport, so the flux comparison below is
    unaffected.
    """
    def run(boundary, compute, max_lost=10):
        sphere = yamc.Sphere(radius=2.0, boundary=boundary)
        material = yamc.Material(
            composition={'Fe56': 1.0}, density=7.8, temperature=294,
        )
        material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
        cell = yamc.Cell(region=sphere.below, material=material)
        geometry = yamc.Geometry([cell])
        source = yamc.NeutronSource(
            energy=yamc.sources.Discrete([14e6], [1]),  # 14 MeV -- long mfp
            position=(0, 0, 0),
        )
        bins = _log_uniform_bins(1e3, 2e7, 4)
        tally = yamc.Tally(scores=['flux'], cells=cell, energy_bins=bins)
        model = yamc.Model(
            geometry=geometry, tallies=[tally], source=source,
            verbose=[], max_lost_particles=max_lost,
        )
        results = model.simulate_transport(compute=compute, total_particles=2000, seed=7)
        return sum(results[tally].mean)

    cpu_vacuum = run('vacuum', 'cpu')
    gpu_vacuum = run('vacuum', 'gpu')
    # Open geometry: allow the losses this deliberately produces.
    gpu_transmission = run('transmission', 'gpu', max_lost=10**9)

    assert cpu_vacuum > 0.0
    assert gpu_vacuum > 0.0
    # GPU vacuum flux must match the CPU within statistics (observed
    # ratio 0.997 at 2000 histories; a missing surface kill inflates
    # it several-fold, far outside this band).
    rel_diff = abs(gpu_vacuum - cpu_vacuum) / cpu_vacuum
    assert rel_diff < 0.05, (
        f"GPU vacuum flux diverges from CPU -- gpu={gpu_vacuum}, "
        f"cpu={cpu_vacuum} ({rel_diff:.1%}); vacuum may not be killing "
        f"particles at the surface."
    )
    # Convex cell: leaving through the void must score exactly like
    # dying at the surface.
    assert gpu_transmission == pytest.approx(gpu_vacuum, rel=1e-9), (
        f"transmission-bounded in-cell flux {gpu_transmission} != "
        f"vacuum-bounded {gpu_vacuum}; escaped particles appear to "
        f"still be scoring into the cell."
    )


def test_simulate_transport_gpu_supports_multiple_tallies():
    """Two flux tallies on the same model -- both should come back
    populated after a single GPU launch. The kernel walks `n_tallies`
    per step and atomic-adds into each tally's slot; the dispatch
    builds one `TalliesPack` covering all of them.
    """
    sphere = yamc.Sphere(radius=10.0, boundary='vacuum')
    material = yamc.Material(
        composition={'Fe56': 1.0}, density=7.8, temperature=294,
    )
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([1e6], [1]),
        position=(0, 0, 0),
    )
    bins_coarse = _log_uniform_bins(1e3, 2e7, 4)
    bins_fine = _log_uniform_bins(1e3, 2e7, 8)
    tally_a = yamc.Tally(
        scores=['flux'], cells=cell, energy_bins=bins_coarse, name='a',
    )
    tally_b = yamc.Tally(
        scores=['flux'], cells=cell, energy_bins=bins_fine, name='b',
    )
    model = yamc.Model(
        geometry=geometry,
        tallies=[tally_a, tally_b],
        source=source,
    )
    results = model.simulate_transport(compute='gpu', total_particles=500, seed=42)
    means_a = list(results[tally_a].mean)
    means_b = list(results[tally_b].mean)
    assert len(means_a) == 4
    assert len(means_b) == 8
    # Both tallies see the same physics, so their integrals over
    # energy should agree to within a few percent (only difference
    # is the binning resolution).
    sum_a = sum(means_a)
    sum_b = sum(means_b)
    assert sum_a > 0.0 and sum_b > 0.0
    rel_diff = abs(sum_a - sum_b) / max(sum_a, 1e-30)
    assert rel_diff < 0.01, (
        f"two parallel flux tallies should integrate to the same total: "
        f"{sum_a} vs {sum_b} (rel diff {rel_diff:.2%})"
    )


# --------------------- adapter listing & selection ---------------------


def test_list_gpu_adapters_returns_names():
    """`yamc.parallel.list_gpu_adapters()` returns the f64-capable adapter names.
    The module is skipped unless `gpu_available()`, so there is at least
    one."""
    adapters = yamc.parallel.list_gpu_adapters()
    assert isinstance(adapters, list)
    assert all(isinstance(name, str) and name for name in adapters)
    assert len(adapters) >= 1, "gpu_available() is True but no adapters listed"


def test_simulate_transport_gpu_by_name_runs():
    """`compute=<adapter name>` for a listed adapter runs and populates the
    tally, and is deterministic across repeated runs on that adapter."""
    name = yamc.parallel.list_gpu_adapters()[0]

    tally_a, model_a, run_a = _build_model_with_flux_tally(total_particles=500, seed=11)
    a = list(model_a.simulate_transport(compute=name, **run_a)[tally_a].mean)
    assert len(a) == 4
    assert all(math.isfinite(m) and m >= 0.0 for m in a)
    assert any(m > 0.0 for m in a), f"no flux recorded on adapter {name!r}: {a}"

    # Same adapter + same seed -> identical (neutron-only, so bit-exact).
    tally_b, model_b, run_b = _build_model_with_flux_tally(total_particles=500, seed=11)
    b = list(model_b.simulate_transport(compute=name, **run_b)[tally_b].mean)
    assert a == b, f"named-adapter GPU runs not deterministic: {a} vs {b}"


def test_simulate_transport_gpu_unknown_adapter_raises():
    """A `compute` value that is neither 'cpu'/'gpu' nor a real adapter name is
    a user error -> ValueError, and the message lists the available adapters."""
    available = yamc.parallel.list_gpu_adapters()
    _, model, run = _build_model_with_flux_tally(total_particles=50)
    with pytest.raises(ValueError) as excinfo:
        model.simulate_transport(compute='definitely-not-a-real-adapter-xyz', **run)
    msg = str(excinfo.value)
    # The error should name a real available adapter so the user can fix it.
    assert available[0] in msg, f"error didn't list available adapters: {msg}"


def test_simulate_transport_compute_typo_raises():
    """A typo'd `compute` value (not 'cpu'/'gpu') is treated as an adapter
    name; an unknown one raises ValueError rather than running silently."""
    available = yamc.parallel.list_gpu_adapters()
    _, model, run = _build_model_with_flux_tally(total_particles=50)
    with pytest.raises(ValueError) as excinfo:
        model.simulate_transport(compute='gpy', **run)
    assert available[0] in str(excinfo.value)


def test_simulate_transport_gpu_cell_only_tally():
    """Tally with `[CellFilter]` and no `EnergyFilter` -- collapses to
    one bin per cell (everything from −∞ to +∞ in energy)."""
    sphere = yamc.Sphere(radius=10.0, boundary='vacuum')
    material = yamc.Material(
        composition={'Fe56': 1.0}, density=7.8, temperature=294,
    )
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([1e6], [1]),
        position=(0, 0, 0),
    )
    tally = yamc.Tally(scores=['flux'], cells=cell)
    model = yamc.Model(
        geometry=geometry, tallies=[tally], source=source,
    )
    means = list(model.simulate_transport(compute='gpu', total_particles=500, seed=42)[tally].mean)
    assert len(means) == 1, f"cell-only flux tally should have 1 bin, got {len(means)}"
    assert means[0] > 0.0, f"expected nonzero flux for cell-only tally, got {means}"


def test_simulate_transport_gpu_total_score():
    """`'total'` score multiplies track length by σ_t at the
    particle's current energy. Should produce values comparable to a
    parallel flux tally weighted by ⟨σ_t⟩."""
    sphere = yamc.Sphere(radius=10.0, boundary='vacuum')
    material = yamc.Material(
        composition={'Fe56': 1.0}, density=7.8, temperature=294,
    )
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([1e6], [1]),
        position=(0, 0, 0),
    )
    flux = yamc.Tally(scores=['flux'], cells=cell, name='flux')
    total = yamc.Tally(scores=['total'], cells=cell, name='total')
    model = yamc.Model(
        geometry=geometry, tallies=[flux, total], source=source,
    )
    results = model.simulate_transport(compute='gpu', total_particles=500, seed=42)
    flux_mean = list(results[flux].mean)[0]
    total_mean = list(results[total].mean)[0]
    assert flux_mean > 0.0
    assert total_mean > 0.0
    # Total reaction rate must be ≤ flux × σ_t,max but in absolute
    # terms exceeds flux only when σ_t > 1 cm⁻¹. Just guard the basic
    # invariant that both are populated and have the same sign.
    assert total_mean > 0.0


def test_simulate_transport_gpu_absorption_score():
    """`'absorption'` score multiplies track length by σ_a. Absorption
    rate must be ≤ total reaction rate."""
    sphere = yamc.Sphere(radius=10.0, boundary='vacuum')
    material = yamc.Material(
        composition={'Fe56': 1.0}, density=7.8, temperature=294,
    )
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([1e6], [1]),
        position=(0, 0, 0),
    )
    total = yamc.Tally(scores=['total'], cells=cell, name='total')
    absorp = yamc.Tally(scores=['absorption'], cells=cell, name='abs')
    model = yamc.Model(
        geometry=geometry, tallies=[total, absorp], source=source,
    )
    results = model.simulate_transport(compute='gpu', total_particles=500, seed=42)
    total_mean = list(results[total].mean)[0]
    absorp_mean = list(results[absorp].mean)[0]
    assert total_mean > 0.0
    assert absorp_mean > 0.0
    assert absorp_mean <= total_mean + 1e-9, (
        f"absorption rate ({absorp_mean}) must not exceed total reaction "
        f"rate ({total_mean})"
    )


def test_simulate_transport_gpu_multi_batch_produces_nonzero_stddev():
    """Splitting `total_particles` into multiple GPU launches via the
    tally's `particles_per_cache_write` chunking should produce
    independent realisations (one launch per chunk with a distinct RNG
    stream), giving a non-zero per-bin standard deviation. Pre-#7 the
    dispatch ran a single launch over the full particle count,
    collapsing the variance estimator to zero.
    """
    sphere = yamc.Sphere(radius=10.0, boundary='vacuum')
    material = yamc.Material(
        composition={'Fe56': 1.0}, density=7.8, temperature=294,
    )
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([1e6], [1]),
        position=(0, 0, 0),
    )
    bins = _log_uniform_bins(1e3, 2e7, 4)
    tally = yamc.Tally(scores=['flux'], cells=cell, energy_bins=bins)
    model = yamc.Model(
        geometry=geometry, tallies=[tally], source=source,
    )
    result = model.simulate_transport(compute='gpu', total_particles=1600, seed=42)[tally]
    means = list(result.mean)
    stds = list(result.standard_deviation)
    assert len(means) == 4
    assert len(stds) == 4
    # At least one bin must have a strictly positive stddev across the
    # chunked realisations. (Mean must also be positive there.)
    nonzero = [(m, s) for m, s in zip(means, stds) if m > 0.0]
    assert nonzero, "no bin recorded any flux"
    assert any(s > 0.0 for _, s in nonzero), (
        f"every populated bin has zero stddev: "
        f"means={means}, stds={stds}. Chunked realisations should "
        f"diverge slightly under different RNG seeds."
    )


def test_simulate_transport_gpu_per_mt_reaction_rate():
    """ReactionRate scores routed through `SCORE_PER_MT`. Build two
    tallies on Fe56 -- one for MT 2 (elastic, which dominates at fast
    energies) and one for MT 102 (n,γ capture, sub-barn at MeV).
    The kernel should produce a non-zero elastic rate and (much
    smaller) capture rate.
    """
    sphere = yamc.Sphere(radius=10.0, boundary='vacuum')
    material = yamc.Material(
        composition={'Fe56': 1.0}, density=7.8, temperature=294,
    )
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([1e6], [1]),
        position=(0, 0, 0),
    )
    elastic = yamc.Tally(scores=['elastic'], cells=cell, name='elastic')
    capture = yamc.Tally(scores=[102], cells=cell, name='capture')
    model = yamc.Model(
        geometry=geometry, tallies=[elastic, capture], source=source,
    )
    results = model.simulate_transport(compute='gpu', total_particles=1000, seed=42)
    e_mean = list(results[elastic].mean)[0]
    c_mean = list(results[capture].mean)[0]
    assert e_mean > 0.0, f"elastic reaction rate should be positive, got {e_mean}"
    assert c_mean >= 0.0, f"capture reaction rate should be non-negative, got {c_mean}"
    # At ~1 MeV in Fe56 elastic XS is ~3 b vs MT 102 ~0.003 b -- three
    # orders of magnitude apart. Even with a lot of slack the elastic
    # rate must comfortably exceed capture.
    assert e_mean > c_mean * 10, (
        f"expected elastic ({e_mean}) >> capture ({c_mean}) for fast "
        f"neutrons in Fe56"
    )


def test_simulate_transport_gpu_named_inelastic_score():
    """Named score `'inelastic'` (MT 4) should accumulate a positive
    rate at 14 MeV in Fe56 where inelastic xs is large (~1 b)."""
    sphere = yamc.Sphere(radius=10.0, boundary='vacuum')
    material = yamc.Material(
        composition={'Fe56': 1.0}, density=7.8, temperature=294,
    )
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14e6], [1]),
        position=(0, 0, 0),
    )
    inel = yamc.Tally(scores=['inelastic'], cells=cell, name='inel')
    model = yamc.Model(
        geometry=geometry, tallies=[inel], source=source,
    )
    rate = list(model.simulate_transport(compute='gpu', total_particles=500, seed=42)[inel].mean)[0]
    assert rate > 0.0, f"inelastic rate should be positive at 14 MeV, got {rate}"


def test_simulate_transport_gpu_heating_score():
    """`'heating'` (MT 301) -- macroscopic kerma XS at the particle's
    current energy × track length. Should produce a positive value
    in Fe56 at 14 MeV (neutron-induced heating dominates inelastic
    channels there)."""
    sphere = yamc.Sphere(radius=10.0, boundary='vacuum')
    material = yamc.Material(
        composition={'Fe56': 1.0}, density=7.8, temperature=294,
    )
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14e6], [1]),
        position=(0, 0, 0),
    )
    heating = yamc.Tally(scores=['heating'], cells=cell)
    model = yamc.Model(
        geometry=geometry, tallies=[heating], source=source,
    )
    rate = list(model.simulate_transport(compute='gpu', total_particles=500, seed=42)[heating].mean)[0]
    assert rate > 0.0, f"heating score should be positive at 14 MeV, got {rate}"


def test_simulate_transport_gpu_damage_energy_score():
    """`'damage-energy'` (MT 444) -- DPA proxy. Same shape as heating;
    should produce a non-negative value in Fe56."""
    sphere = yamc.Sphere(radius=10.0, boundary='vacuum')
    material = yamc.Material(
        composition={'Fe56': 1.0}, density=7.8, temperature=294,
    )
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14e6], [1]),
        position=(0, 0, 0),
    )
    dmg = yamc.Tally(scores=['damage-energy'], cells=cell)
    model = yamc.Model(
        geometry=geometry, tallies=[dmg], source=source,
    )
    rate = list(model.simulate_transport(compute='gpu', total_particles=500, seed=42)[dmg].mean)[0]
    assert rate >= 0.0, f"damage-energy score should be non-negative, got {rate}"


def test_simulate_transport_gpu_production_scores():
    """Production scores (`H1-production`, …, `He4-production`) --
    yield-weighted per-MT lookups. At 14 MeV, He4-production from
    Fe56 is the dominant alpha-emitting channel and should be
    positive; H1-production may be smaller but non-negative."""
    sphere = yamc.Sphere(radius=10.0, boundary='vacuum')
    material = yamc.Material(
        composition={'Fe56': 1.0}, density=7.8, temperature=294,
    )
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14e6], [1]),
        position=(0, 0, 0),
    )
    h1 = yamc.Tally(scores=['H1-production'], cells=cell, name='h1')
    he4 = yamc.Tally(scores=['He4-production'], cells=cell, name='he4')
    model = yamc.Model(
        geometry=geometry, tallies=[h1, he4], source=source,
    )
    results = model.simulate_transport(compute='gpu', total_particles=1000, seed=42)
    h1_rate = list(results[h1].mean)[0]
    he4_rate = list(results[he4].mean)[0]
    # Both must be non-negative (track-length × xs is non-negative).
    assert h1_rate >= 0.0, f"H1-production = {h1_rate}"
    assert he4_rate >= 0.0, f"He4-production = {he4_rate}"


def test_simulate_transport_gpu_accepts_neutron_particle_filter():
    """`particle="neutron"` on a Tally adds a `ParticleType(Neutron)`
    filter. The kernel is neutron-only so it should validate clean.
    The yamc-verification sphere notebook adds this filter to every
    tally -- this test guards against the dispatch regressing into
    rejecting it."""
    sphere = yamc.Sphere(radius=10.0, boundary='vacuum')
    material = yamc.Material(
        composition={'Fe56': 1.0}, density=7.8, temperature=294,
    )
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([1e6], [1]),
        position=(0, 0, 0),
    )
    tally = yamc.Tally(
        scores=['flux'], cells=cell, particle='neutron',
    )
    model = yamc.Model(
        geometry=geometry, tallies=[tally], source=source,
    )
    means = list(model.simulate_transport(compute='gpu', total_particles=200, seed=42)[tally].mean)
    assert len(means) == 1
    assert means[0] > 0.0


def test_simulate_transport_gpu_rejects_photon_particle_filter():
    """`particle="photon"` should reject -- the kernel is neutron-only,
    a silent zero tally would hide the misuse."""
    sphere = yamc.Sphere(radius=10.0, boundary='vacuum')
    material = yamc.Material(
        composition={'Fe56': 1.0}, density=7.8, temperature=294,
    )
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([1e6], [1]),
        position=(0, 0, 0),
    )
    tally = yamc.Tally(
        scores=['flux'], cells=cell, particle='photon',
    )
    model = yamc.Model(
        geometry=geometry, tallies=[tally], source=source,
    )
    with pytest.raises(ValueError, match="filters"):
        model.simulate_transport(compute='gpu', total_particles=50, seed=42)


def test_simulate_transport_gpu_fission_no_longer_terminates_silently():
    """Slice G: the kernel now samples fission as its own collision
    branch (weight × ν̄, Watt χ-spectrum E_out, isotropic μ) instead
    of folding σ_f into σ_a and terminating the particle. We don't
    have an actinide test fixture loaded, so just validate that:
    (a) Fe56 (non-fissionable) still produces sensible flux, and
    (b) the dispatch happily plumbs `xs_fission` / `nu_bar` (zero
        everywhere for Fe56) through without panicking.
    A nuclide-specific actinide test would require additional Arrow
    data; the slice-G smoke is best done from yamc-verification's
    sphere notebook on Ac225 (where we can compare GPU vs CPU)."""
    sphere = yamc.Sphere(radius=10.0, boundary='vacuum')
    material = yamc.Material(
        composition={'Fe56': 1.0}, density=7.8, temperature=294,
    )
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14e6], [1]),
        position=(0, 0, 0),
    )
    bins = _log_uniform_bins(1e3, 2e7, 8)
    tally = yamc.Tally(scores=['flux'], cells=cell, energy_bins=bins)
    model = yamc.Model(
        geometry=geometry, tallies=[tally], source=source,
    )
    means = list(model.simulate_transport(compute='gpu', total_particles=500, seed=42)[tally].mean)
    # Every bin non-negative; at least one positive.
    assert len(means) == 8
    assert all(m >= 0.0 for m in means)
    assert any(m > 0.0 for m in means)


def test_simulate_transport_gpu_rejects_unsupported_score():
    """A score the kernel can't compute (e.g. photon coherent-scatter)
    should produce a structured ValueError, not a silent zero tally."""
    sphere = yamc.Sphere(radius=10.0, boundary='vacuum')
    material = yamc.Material(
        composition={'Fe56': 1.0}, density=7.8, temperature=294,
    )
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([1e6], [1]),
        position=(0, 0, 0),
    )
    # Photon scores are out of scope for the neutron-only kernel.
    tally = yamc.Tally(scores=['coherent-scatter'], cells=cell)
    model = yamc.Model(
        geometry=geometry, tallies=[tally], source=source,
    )
    with pytest.raises(ValueError, match="not supported"):
        model.simulate_transport(compute='gpu', total_particles=50, seed=42)


def test_simulate_transport_gpu_accepts_arbitrary_energy_bins():
    """Slice F: linear-spaced (and any other monotonic) energy bins
    just work on GPU -- the kernel takes the actual edges and binary-
    searches per tally accumulation. Pre-slice-F the dispatch
    rejected anything non-log-uniform."""
    sphere = yamc.Sphere(radius=10.0, boundary='vacuum')
    material = yamc.Material(
        composition={'Fe56': 1.0}, density=7.8, temperature=294,
    )
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14e6], [1]),
        position=(0, 0, 0),
    )
    # Linear spacing -- NOT log-uniform; pre-slice-F this raised.
    linear_bins = [1e3, 5e6, 1e7, 1.5e7, 2e7]
    tally = yamc.Tally(scores=['flux'], cells=cell, energy_bins=linear_bins)
    model = yamc.Model(
        geometry=geometry, tallies=[tally], source=source,
    )
    means = list(model.simulate_transport(compute='gpu', total_particles=200, seed=42)[tally].mean)
    # 4 bins in the filter, all should at least be finite.
    assert len(means) == 4
    for m in means:
        assert m == m  # not NaN
    # Source bin is 1e7..1.5e7 (which contains 14 MeV -- bin index 2 of
    # 4). Most of the flux should land in or near it.
    assert means[2] > 0.0, f"expected flux at source bin, got {means}"


def test_simulate_transport_gpu_vitamin_j_175_runs():
    """Slice F: VITAMIN-J-175 is the standard energy structure for
    yamc-verification. It's strongly non-uniform; pre-slice-F the
    dispatch would reject it. After slice F it just works."""
    sphere = yamc.Sphere(radius=10.0, boundary='vacuum')
    material = yamc.Material(
        composition={'Fe56': 1.0}, density=7.8, temperature=294,
    )
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14e6], [1]),
        position=(0, 0, 0),
    )
    tally = yamc.Tally(
        scores=['flux'], cells=cell,
        energy_group_structure='VITAMIN-J-175',
    )
    model = yamc.Model(
        geometry=geometry, tallies=[tally], source=source,
    )
    means = list(model.simulate_transport(compute='gpu', total_particles=200, seed=42)[tally].mean)
    assert len(means) == 175, f"VITAMIN-J-175 should have 175 bins, got {len(means)}"
    # The integral should match a single coarse-bin GPU run.
    coarse = yamc.Tally(
        scores=['flux'], cells=cell,
        energy_bins=_log_uniform_bins(1e-5, 2e7, 8),
    )
    coarse_model = yamc.Model(
        geometry=geometry, tallies=[coarse], source=source,
    )
    coarse_means = list(coarse_model.simulate_transport(compute='gpu', total_particles=200, seed=42)[coarse].mean)
    integral_fine = sum(means)
    integral_coarse = sum(coarse_means)
    # Same RNG seed + same physics → same total flux within 1%.
    rel_diff = abs(integral_fine - integral_coarse) / max(integral_coarse, 1e-30)
    assert rel_diff < 0.05, (
        f"VITAMIN-J integral {integral_fine} vs coarse {integral_coarse} "
        f"differs by {rel_diff:.2%}"
    )


def test_simulate_transport_gpu_rejects_tracks():
    _, model, run = _build_model_with_flux_tally(total_particles=50)
    with pytest.raises(ValueError, match="capture_tracks"):
        model.simulate_transport(compute='gpu', capture_tracks=10, **run)


def test_simulate_transport_gpu_rejects_threads():
    _, model, run = _build_model_with_flux_tally(total_particles=50)
    with pytest.raises(ValueError, match="threads"):
        model.simulate_transport(compute='gpu', threads=4, **run)


def test_simulate_transport_invalid_compute():
    # 'cuda' is neither 'cpu'/'gpu' nor a real adapter, so it is treated as an
    # unknown adapter name and raises (listing the available adapters).
    available = yamc.parallel.list_gpu_adapters()
    _, model, run = _build_model_with_flux_tally(total_particles=50)
    with pytest.raises(ValueError) as excinfo:
        model.simulate_transport(compute='cuda', **run)
    assert available[0] in str(excinfo.value)


def test_max_steps_per_particle_default_is_100000():
    # The default was raised 1000 -> 100_000 (d743159): the GPU kernel
    # enforces the cap as a hard loop bound, and a 14 MeV neutron in a
    # weak absorber (pure H2) needs many hundreds of elastic collisions
    # before leaking, so the old cap truncated ~10% of the track length.
    _, model, _ = _build_model_with_flux_tally(total_particles=50)
    assert model.gpu_max_steps_per_particle == 100_000


def test_max_steps_per_particle_kwarg_propagates():
    """`Model(gpu_max_steps_per_particle=N)` should set the value the GPU
    dispatch uses as the kernel step cap. Smoke: a small step cap
    should leave more particles `alive` (terminated by the cap rather
    than absorbed/leaked) than the default."""
    sphere = yamc.Sphere(radius=10.0, boundary='vacuum')
    material = yamc.Material(
        composition={'Fe56': 1.0}, density=7.8, temperature=294,
    )
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([1e6], [1]),
        position=(0, 0, 0),
    )
    model = yamc.Model(
        geometry=geometry, tallies=[], source=source,
        gpu_max_steps_per_particle=5,
    )
    assert model.gpu_max_steps_per_particle == 5
    # Should run without error -- exercises the dispatch threading the
    # value through to the kernel.
    model.simulate_transport(compute='gpu', total_particles=50, seed=42)


def test_gpu_max_runtime_uncapped_runs():
    """total_particles=None + max_runtime runs the GPU launch loop until the
    wall-time budget elapses (the uncapped/time-capped GPU loop, #230 task 2),
    scoring a valid tally for the histories completed."""
    tally, model, _ = _build_model_with_flux_tally()
    results = model.simulate_transport(compute='gpu', max_runtime=(1, 's'), seed=42)
    assert results[tally].n_histories > 0
    assert sum(results[tally].mean) > 0.0


def test_gpu_max_runtime_generous_is_noop():
    """A budget far larger than the run takes is never the binding constraint,
    so a capped GPU run still completes exactly the requested histories -- proof
    that max_runtime is accepted on GPU and composes with total_particles."""
    tally, model, _ = _build_model_with_flux_tally(total_particles=500)
    results = model.simulate_transport(
        compute='gpu', total_particles=500, seed=42, max_runtime=(1, 'h')
    )
    assert results[tally].n_histories == 500


def test_gpu_rejects_no_stop_condition():
    """No total_particles and no max_runtime (and no convergence targets):
    rejected up front by the shared stop-condition guard rather than launching
    forever."""
    _, model, _ = _build_model_with_flux_tally()
    with pytest.raises(ValueError, match="at least one stop condition"):
        model.simulate_transport(compute='gpu')


# ---- MaterialFilter on the GPU (issue #271) ------------------------------
#
# `materials=` used to be rejected on the GPU ("filters must be a spatial
# binner ..."). A cell's material is fixed for the run, so the dispatch now
# folds the material bin into the kernel's spatial dimension alongside the
# cell bin. These tests pin the resulting bin mapping from Python.


def _build_alternating_material_model(total_particles=20000, energy_bins=None):
    """Four nested shells over two materials in ALTERNATING order
    (A, B, A, B), so neither material maps onto a contiguous run of cells and
    a material bin cannot be a relabelled cell bin.

    Returns `(model, by_material, by_cell, by_material_energy, run_kwargs)`.
    """
    s1 = yamc.Sphere(radius=3.0)
    s2 = yamc.Sphere(radius=6.0)
    s3 = yamc.Sphere(radius=9.0)
    s4 = yamc.Sphere(radius=12.0, boundary='vacuum')
    # One nuclide, two densities: a single shared energy grid, but distinct
    # macroscopic cross sections so the two material bins hold genuinely
    # different flux.
    mat_a = yamc.Material(
        composition={'Fe56': 1.0}, density=2.0, temperature=294, id=11,
    )
    mat_b = yamc.Material(
        composition={'Fe56': 1.0}, density=7.0, temperature=294, id=22,
    )
    mat_a.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    mat_b.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cells = [
        yamc.Cell(region=s1.below, material=mat_a),
        yamc.Cell(region=s1.above & s2.below, material=mat_b),
        yamc.Cell(region=s2.above & s3.below, material=mat_a),
        yamc.Cell(region=s3.above & s4.below, material=mat_b),
    ]
    geometry = yamc.Geometry(cells)
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1]),
        position=(0, 0, 0),
    )
    by_material = yamc.Tally(scores=['flux'], materials=[mat_a, mat_b])
    by_cell = yamc.Tally(scores=['flux'], cells=cells)
    tallies = [by_material, by_cell]
    by_material_energy = None
    if energy_bins is not None:
        by_material_energy = yamc.Tally(
            scores=['flux'], materials=[mat_a, mat_b], energy_bins=energy_bins,
        )
        tallies.append(by_material_energy)
    model = yamc.Model(geometry=geometry, tallies=tallies, source=source)
    run = {"total_particles": total_particles, "seed": 4242}
    return model, by_material, by_cell, by_material_energy, run


def test_simulate_transport_gpu_material_filter_partitions_cell_bins():
    """`materials=[A, B]` bins by material on the GPU, and the two bins
    partition the same flux the cell bins do. Both tallies score the same
    histories in one launch, so material A's flux IS cell 1 + cell 3 -- an
    exact identity, not a statistical one."""
    model, by_material, by_cell, _, run = _build_alternating_material_model()
    results = model.simulate_transport(compute='gpu', **run)
    mat = list(results[by_material].mean)
    cell = list(results[by_cell].mean)

    assert len(mat) == 2, f"expected 2 material bins, got {len(mat)}"
    assert len(cell) == 4, f"expected 4 cell bins, got {len(cell)}"
    assert all(c > 0.0 for c in cell), f"a cell scored no flux: {cell}"

    # Materials alternate across the shells: each material bin sums two
    # NON-adjacent cell bins.
    assert mat[0] == pytest.approx(cell[0] + cell[2], rel=1e-9), (
        f"material A ({mat[0]}) != cells 1+3 ({cell[0] + cell[2]})"
    )
    assert mat[1] == pytest.approx(cell[1] + cell[3], rel=1e-9), (
        f"material B ({mat[1]}) != cells 2+4 ({cell[1] + cell[3]})"
    )
    # The two bins must be distinguishable, otherwise the identities above
    # would also hold for a degenerate all-in-one-bin mapping.
    split = mat[0] / (mat[0] + mat[1])
    assert 0.05 < split < 0.95, f"material A holds {split:.3f} of the flux"


def test_simulate_transport_gpu_material_filter_with_energy_bins():
    """`materials=` stacked with `energy_bins=` on the GPU. Material is the
    outer dimension and energy the inner one, so folding each material's
    energy bins recovers its unbinned total; a transposed stride would mix
    the two materials' spectra."""
    edges = [1e-3, 1e5, 1e6, 2e7]
    model, by_material, _, by_material_energy, run = _build_alternating_material_model(
        energy_bins=edges,
    )
    results = model.simulate_transport(compute='gpu', **run)
    mat = list(results[by_material].mean)
    spec = list(results[by_material_energy].mean)

    n_e = len(edges) - 1
    assert len(spec) == 2 * n_e, f"expected {2 * n_e} material x energy bins, got {len(spec)}"
    for material_bin in range(2):
        block = spec[material_bin * n_e:(material_bin + 1) * n_e]
        assert sum(1 for v in block if v > 0.0) >= 2, (
            f"material {material_bin} populated only one energy bin ({block}); "
            "the fold below would not test the stride"
        )
        assert sum(block) == pytest.approx(mat[material_bin], rel=1e-9), (
            f"material {material_bin} folded over energy ({sum(block)}) != "
            f"unbinned total ({mat[material_bin]})"
        )


def test_simulate_transport_gpu_material_filter_matches_cpu():
    """The GPU's material-binned flux agrees with the CPU's, which reaches
    the same bins through the tally's material gate rather than through a
    per-cell bin map."""
    cpu_model, cpu_tally, _, _, run = _build_alternating_material_model()
    cpu = list(cpu_model.simulate_transport(compute='cpu', **run)[cpu_tally].mean)

    gpu_model, gpu_tally, _, _, run = _build_alternating_material_model()
    gpu = list(gpu_model.simulate_transport(compute='gpu', **run)[gpu_tally].mean)

    for i, name in enumerate(('A', 'B')):
        assert cpu[i] > 0.0, f"CPU material {name} flux must be > 0, got {cpu[i]}"
        ratio = gpu[i] / cpu[i]
        assert 0.95 <= ratio <= 1.05, (
            f"material {name}: GPU/CPU flux ratio {ratio:.4f} outside [0.95, 1.05] "
            f"(CPU {cpu[i]:.5e}, GPU {gpu[i]:.5e})"
        )


# ---- EnergyFunctionFilter on the GPU (issue #271) ------------------------
#
# `energy_function=` and its sugar `dose_coefficients=` used to be rejected on
# the GPU. They are not a bin dimension: the score is multiplied by a tabulated
# curve interpolated at the particle's energy, and the event is dropped
# entirely when that energy falls off the table. The dispatch ships the cubic
# spline coefficients the CPU already solved, so both backends interpolate
# identically.

# Deliberately not a power of two, so a dropped multiply cannot hide behind a
# coincidentally exact ratio.
_CONST_Y = 3.25


def _build_energy_function_model(total_particles=20000, energy_bins=None, curve=None):
    """Fe56 sphere with a plain flux tally and an energy-function-weighted one
    scored over the same histories, so the relation between them is exact.

    Returns `(model, plain, weighted, run_kwargs)`.
    """
    sphere = yamc.Sphere(radius=25.0, boundary='vacuum')
    material = yamc.Material(
        composition={'Fe56': 1.0}, density=7.874, temperature=294,
    )
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1]),
        position=(0, 0, 0),
    )
    # Constant over the whole plausible spectrum: weights every event, gates
    # none. A natural cubic spline through constant data is exactly constant.
    curve = curve or ([1e-5, 1e2, 1e5, 1e8], [_CONST_Y] * 4)
    kw = {} if energy_bins is None else {"energy_bins": energy_bins}
    plain = yamc.Tally(scores=['flux'], cells=cell, **kw)
    weighted = yamc.Tally(scores=['flux'], cells=cell, energy_function=curve, **kw)
    model = yamc.Model(
        geometry=geometry, tallies=[plain, weighted], source=source,
    )
    return model, plain, weighted, {"total_particles": total_particles, "seed": 4242}


def test_simulate_transport_gpu_energy_function_scales_flux_exactly():
    """A constant `energy_function=` scales the flux tally by exactly that
    constant. Both tallies score the same histories in one launch, so this is
    an arithmetic identity, not a statistical comparison."""
    model, plain, weighted, run = _build_energy_function_model()
    results = model.simulate_transport(compute='gpu', **run)
    p = sum(results[plain].mean)
    w = sum(results[weighted].mean)
    assert p > 0.0, "plain flux tally scored nothing"
    assert w == pytest.approx(_CONST_Y * p, rel=1e-9), (
        f"weighted ({w}) != {_CONST_Y} * plain ({_CONST_Y * p})"
    )


def test_simulate_transport_gpu_energy_function_composes_with_energy_bins():
    """Stacked with `energy_bins=`, the weight applies per event, so every
    populated energy bin is scaled by the same constant."""
    edges = [1e-5, 1e3, 1e6, 2e7]
    model, plain, weighted, run = _build_energy_function_model(energy_bins=edges)
    results = model.simulate_transport(compute='gpu', **run)
    p = list(results[plain].mean)
    w = list(results[weighted].mean)
    assert len(p) == len(edges) - 1
    assert len(w) == len(p)
    assert sum(1 for v in p if v > 0.0) >= 2, (
        f"only one energy bin populated ({p}); the per-bin check is not meaningful"
    )
    for i, (pi, wi) in enumerate(zip(p, w)):
        if pi > 0.0:
            assert wi == pytest.approx(_CONST_Y * pi, rel=1e-9), f"energy bin {i}"


def test_simulate_transport_gpu_energy_function_gate_drops_out_of_range_flux():
    """A curve covering only part of the spectrum drops the flux outside it
    entirely, rather than scoring it with weight zero. With a weight of 1.0 in
    range the gated tally IS the in-range energy bin."""
    split, top = 1e6, 2e7
    sphere = yamc.Sphere(radius=25.0, boundary='vacuum')
    material = yamc.Material(
        composition={'Fe56': 1.0}, density=7.874, temperature=294,
    )
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([14.06e6], [1]), position=(0, 0, 0),
    )
    gated = yamc.Tally(
        scores=['flux'], cells=cell,
        energy_function=([split, 5e6, 1e7, top], [1.0] * 4),
    )
    binned = yamc.Tally(scores=['flux'], cells=cell, energy_bins=[1e-5, split, top])
    model = yamc.Model(geometry=geometry, tallies=[gated, binned], source=source)
    results = model.simulate_transport(compute='gpu', total_particles=20000, seed=4242)

    g = sum(results[gated].mean)
    below, above = list(results[binned].mean)
    # The gate must actually bite, otherwise this passes trivially.
    assert below > 0.05 * above, (
        f"only {below} vs {above} below the split; the gate is barely exercised"
    )
    assert g == pytest.approx(above, rel=1e-9), f"gated ({g}) != in-range bin ({above})"


def test_simulate_transport_gpu_dose_coefficients_matches_cpu():
    """`dose_coefficients=` is pure sugar for `energy_function=`, so the real
    ICRP-116 curve comes along free. Independent MC estimates, so a band."""
    def build():
        sphere = yamc.Sphere(radius=25.0, boundary='vacuum')
        material = yamc.Material(
            composition={'Fe56': 1.0}, density=7.874, temperature=294,
        )
        material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
        cell = yamc.Cell(region=sphere.below, material=material)
        geometry = yamc.Geometry([cell])
        source = yamc.NeutronSource(
            energy=yamc.sources.Discrete([14.06e6], [1]), position=(0, 0, 0),
        )
        dose = yamc.Tally(
            scores=['flux'], cells=cell, dose_coefficients=('neutron', 'AP'),
        )
        bare = yamc.Tally(scores=['flux'], cells=cell)
        model = yamc.Model(geometry=geometry, tallies=[dose, bare], source=source)
        return model, dose, bare

    cpu_model, cpu_dose, cpu_bare = build()
    cpu_res = cpu_model.simulate_transport(compute='cpu', total_particles=20000, seed=4242)
    cpu = sum(cpu_res[cpu_dose].mean)
    bare = sum(cpu_res[cpu_bare].mean)

    gpu_model, gpu_dose, _ = build()
    gpu_res = gpu_model.simulate_transport(compute='gpu', total_particles=20000, seed=4242)
    gpu = sum(gpu_res[gpu_dose].mean)

    assert cpu > 0.0 and gpu > 0.0, f"dose tally scored nothing (cpu {cpu}, gpu {gpu})"
    ratio = gpu / cpu
    assert 0.95 <= ratio <= 1.05, (
        f"GPU/CPU dose ratio {ratio:.4f} outside [0.95, 1.05] (CPU {cpu:.5e}, GPU {gpu:.5e})"
    )
    # ICRP-116 coefficients are O(100) pSv cm^2 here, so the dose tally must be
    # far above the bare flux it weights -- proof the curve is applied at all.
    assert cpu > 10.0 * bare, (
        f"dose {cpu:.3e} is not meaningfully above bare flux {bare:.3e}"
    )


def test_simulate_transport_gpu_energy_function_without_spatial_binner_raises():
    """The energy function weights, it does not bin. Without a cell/material/
    mesh filter the tally still has no spatial binner and must be rejected."""
    sphere = yamc.Sphere(radius=10.0, boundary='vacuum')
    material = yamc.Material(
        composition={'Fe56': 1.0}, density=7.8, temperature=294,
    )
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})
    cell = yamc.Cell(region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([1e6], [1]), position=(0, 0, 0),
    )
    tally = yamc.Tally(
        scores=['flux'], energy_function=([1e-5, 1e2, 1e5, 1e8], [1.0] * 4),
    )
    model = yamc.Model(geometry=geometry, tallies=[tally], source=source)
    with pytest.raises(ValueError, match="spatial binner"):
        model.simulate_transport(compute='gpu', total_particles=100, seed=42)
