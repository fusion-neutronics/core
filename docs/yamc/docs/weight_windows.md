# Weight windows

Weight windows are a variance-reduction technique for deep-penetration and
streaming problems: places where an analog simulation almost never sends a
particle, so the tally there is dominated by a handful of lucky histories (or
stays zero). A weight window defines, per mesh voxel and energy group, a target
weight band. A particle above the band is split into several lower-weight
copies (more histories where the tally needs them); a particle below the band is
rouletted (killed or promoted); a particle inside the band is left alone.
Weight is conserved in expectation, so the tally means are unchanged and only
the variance is reduced.

Weight windows are applied at collisions and run on the CPU only
(``compute='gpu'`` refuses a model carrying them). They compose with the rest
of the ``variance_reduction`` list; see [Models](models.md) for that list and
for survival biasing.

## Generating windows automatically (DeGVR)

The usual workflow is to generate the windows from a cheap pre-pass rather
than write bounds by hand. ``model.generate_weight_windows`` runs the
density-extrapolation method (DeGVR): two short transport passes at reduced
material density, then a per-(voxel, group) extrapolation to full density. It
returns a [``WeightWindowBounds``](#applying-windows-directly) you pass into a
production run.

<!-- doctest: skip -->
```python
import yamc

mesh = yamc.RegularRectangularMesh(
    lower_left=[-1100, -1100, -1100], upper_right=[1100, 1100, 1100], shape=[50, 50, 50]
)
tally = yamc.Tally(scores=["flux"], name="flux", mesh=mesh)
model = yamc.Model(geometry=geometry, source=source, tallies=[tally])

# 1. Generate windows on the same mesh you want to resolve. Each pass stops
#    once the window mesh is converged, so no particle budget is needed.
windows = model.generate_weight_windows(
    yamc.WeightWindowGeneratorDeGVR(mesh=mesh)
)

# 2. Run production transport with the windows applied.
prod = yamc.Model(
    geometry=geometry, source=source, tallies=[tally], variance_reduction=[windows]
)
results = prod.simulate_transport(total_particles=2_000_000)
```

Each pass stops at the first of three conditions: the window mesh converged
(``relative_error``, default 0.1), ``total_particles`` or ``max_runtime``. The
mesh counts as converged once 95% of the voxels that have scored, in every
energy group and particle field, have a relative error at or below
``relative_error``. A voxel reached for the first time has an error near 1, so
the pass keeps running while the scored region is still growing into the
shield; voxels that never score (outside the geometry) do not count. The same
target applies to both passes. Pass ``relative_error=None`` to stop on the
budgets alone. The model's own ``convergence_targets`` do not apply: they watch
the model's tallies, which the generation passes do not carry.

The same per-voxel metric is available to any run as a convergence target,
``yamc.ConvergenceTarget("bin_relative_error", 0.05, tally="flux")``, which ends
a mesh-tally run once 95% of its scored voxels reach 5% (CPU only).

Generation has a cost, so for a fair efficiency comparison charge its
wall-clock time against the weight-window budget: time the generation and run
the production pass for the remaining time (see
[``max_runtime``](simulate_transport.md)).

<!-- doctest: skip -->
```python
import time

t0 = time.time()
windows = model.generate_weight_windows(
    yamc.WeightWindowGeneratorDeGVR(mesh=mesh)
)
t_gen = time.time() - t0

prod = yamc.Model(geometry=geometry, source=source, tallies=[tally], variance_reduction=[windows])
results = prod.simulate_transport(total_particles=10_000_000_000, max_runtime=(20, "min") )
# analog baseline for the same total wall-clock, including generation:
analog = model.simulate_transport(total_particles=10_000_000_000,
                                  max_runtime=t_gen + results.elapsed)
```

``WeightWindowGeneratorDeGVR`` parameters (all optional except ``mesh``):

| parameter | default | meaning |
|---|---|---|
| ``mesh`` | -- | the regular mesh the windows are defined on |
| ``particle`` | ``"neutron"`` | particle type the windows act on |
| ``energy_bins`` | ``None`` | energy group edges (eV); ``None`` is a single group |
| ``density_reduction`` | ``None`` (auto) | reduced-density factor ``N`` for the extrapolation; ``None`` computes it from a particle-free optical-depth ray-trace (source to deepest voxel), ``N = tau / 7`` (continuous, clamped ``>= 2``). Pass a float to override. |
| ``ratio`` | ``5`` | upper/lower bound ratio (window width) |
| ``survival_factor`` | ``3`` | sets the roulette survival weight |
| ``max_split`` | ``10`` | most copies one split may create |
| ``weight_floor`` | ``1e-38`` | absolute weight below which a particle is killed |

By default ``density_reduction`` is auto: ``N`` is derived, before the two
generation passes, from the problem's optical thickness ``tau`` (a particle-free
ray-trace from the source to the deepest mesh voxel at the source energy) as
``N = tau / 7`` (continuous, clamped to ``>= 2``; ``tau`` is taken as a high
percentile of the per-voxel optical depth so a single oblique corner voxel does
not skew it). The empirical sweet spot
``tau / N ~ 7`` mean free paths balances fiducial penetration against
extrapolation error: a smaller ``N`` leaves the fiducial too thick to resolve
the deep region, a larger ``N`` over-splits on a noise-amplified
``G = F*(H/F)^(N-1)`` extrapolation. ``model.estimate_density_reduction(gen)``
returns the ``(N, tau)`` it would pick without running generation. Pass an
explicit float to ``density_reduction`` to override.

## Applying windows directly

If you already have bounds (e.g. from a previous generation, via
``windows.lower_bounds`` / ``windows.upper_bounds``, or an analytic importance
map) you can build a ``WeightWindowBounds`` yourself. Bounds are flat, indexed
``group * num_voxels + voxel``; pass ``lower_bounds`` and either ``upper_bounds``
or a ``ratio`` that fills them in. A negative lower bound marks a "no window"
voxel that is skipped.

<!-- doctest: skip -->
```python
windows = yamc.WeightWindowBounds(
    mesh=mesh,
  lower_bounds=[1.0] * mesh.num_bins,  # one energy group, one bound per voxel
    upper_bounds=None,         # or omit and pass ratio=
    ratio=5.0,
    particle="neutron",
    survival_factor=3.0,       # > 1
    max_split=10,
    weight_floor=1e-38,
)
```

## What the windows did

A run with windows reports what they did: how often they split, rouletted and
killed, and whether they acted at all. ``results.weight_window_stats`` is a
dict of counts, or ``None`` for a run without windows and for combined,
reloaded or transmutation-step results:

<!-- doctest: skip -->
```python
s = results.weight_window_stats
s["split_copies"] / s["histories"]  # split copies per source history
s["histories_at_split_budget"]  # histories whose split budget ran out
s["no_window"] / (s["checks"] + s["no_window"])  # collisions with no window
```

The keys are ``histories``, ``particles`` and ``max_particles_per_history``;
``checks`` (collisions at which a window was checked) and ``no_window``
(collisions outside the window mesh or energy bins, or in a voxel without a
window, where the windows do nothing); ``split_events``, ``split_copies``,
``max_split_copies_per_history`` and ``splits_capped_at_max_split`` (splits
that needed more than ``max_split`` copies); ``split_budget``,
``histories_at_split_budget`` and ``splits_cut_short``; ``roulette_kills`` and
``roulette_survivals``; ``weight_floor_kills`` and ``weight_floor_weight``.
The run summary prints most of them on one line.

Three things also print a warning, whatever the ``verbose`` setting:

- **The split budget.** A history stops splitting once it has made 10,000
  split copies. The tallies stay unbiased, but what the history reaches last
  gets fewer, heavier particles. A few such histories is normal and there is
  nothing to change: with DeGVR windows on 3D concrete, steel and air models,
  up to 1.5% of histories reached the budget, and a budget a hundred times
  larger cut the deepest tally's error per history by up to a quarter but
  transported up to about 80% more particles, so the run was no more
  efficient. The run warns when 5% of histories reach it, which usually means
  the windows split far more than the flux falls: they drop faster than the
  flux from voxel to voxel, or sit far below the source weight.
- **The weight floor.** A particle below ``weight_floor`` is killed, not
  rouletted, so any kill biases low the tallies it would have reached. Keep
  ``weight_floor`` below every lower bound, as the default (1e-38) is.
- **No window acting at any collision**, for example energy bins that miss
  the source: the windows then do nothing.

The split budget is not a setting: the warnings point at the windows instead.

## Photon sources

Weight windows are per particle type: set ``particle=`` on both the
generator and the bounds, and the collision handler applies each window only to
matching particles. For a photon source ([``PhotonSource``](sources.md)),
build the window with ``particle="photon"`` and run as above. The photon
transport path splits and roulettes photons at their collisions and the tally
means stay unbiased.

<!-- doctest: skip -->
```python
src = yamc.PhotonSource(energy=2e6)
model = yamc.Model(geometry=geom, source=src, tallies=[flux],
                   transport_secondary_photons=True)
photon_ww = model.generate_weight_windows(
    yamc.WeightWindowGeneratorDeGVR(mesh=mesh, particle="photon"))
prod = yamc.Model(geometry=geom, source=src, tallies=[flux],
                  transport_secondary_photons=True, variance_reduction=[photon_ww])
results = prod.simulate_transport(total_particles=200_000)
```

### Secondary and decay photons (coupled)

Photons produced inside the run (secondary photons from neutron collisions with
``transport_secondary_photons=True``, and D1S decay photons with
``use_decay_photons=True``) travel the same photon transport path, so a
``particle="photon"`` window acts on them at their collisions too.

To generate a neutron window and a photon window for a coupled
neutron-gamma problem in a single call, pass a list of particle types.
One coupled pair of reduced-density passes tallies both fields and returns one
window per particle, in the requested order; apply them together:

<!-- doctest: skip -->
```python
mesh = yamc.RegularRectangularMesh(lower_left=[-3, -3, 0], upper_right=[3, 3, 40],
                                   shape=[1, 1, 8])
src = yamc.NeutronSource(position=(0, 0, 0.5), energy=14.1e6)
model = yamc.Model(geometry=geom, source=src, tallies=[n_flux, p_flux],
                   transport_secondary_photons=True)

neutron_ww, photon_ww = model.generate_weight_windows(
    yamc.WeightWindowGeneratorDeGVR(mesh=mesh, particle=["neutron", "photon"]))

prod = yamc.Model(geometry=geom, source=src, tallies=[n_flux, p_flux],
                  transport_secondary_photons=True,
                  variance_reduction=[neutron_ww, photon_ww])
results = prod.simulate_transport(total_particles=200_000)
```

A single-string ``particle`` still returns one ``WeightWindowBounds`` (unchanged);
a list returns a ``list`` in the same order. See
``examples/python/coupled_np_degvr.py``.

For a photon window, the auto ``density_reduction`` ray-trace uses a photon
energy rather than the driving 14 MeV neutron energy
(photon attenuation at 14 MeV is nothing like at ~1 MeV). For neutron-driven
secondary / decay photons it defaults to ~1 MeV, a typical gamma energy; override
with ``photon_energy=`` on the generator if your spectrum is harder or softer.
This only affects generation efficiency, never bias.

### D1S decay photons

A shutdown-dose (D1S, ``use_decay_photons=True``) run is a coupled run whose
photon field is the decay photons emitted at activation sites, so the same
``particle="photon"`` (or coupled ``["neutron", "photon"]``) generation produces a
window that resolves the deep decay-photon dose. Tally the decay dose as usual
with ``parent_nuclides=model.radionuclides()`` (which, like the D1S run itself,
needs ``yamc.transmutation_decay_data`` and ``yamc.transmutation_reactions`` set,
see [Transmutation data](nuclear_data.md#transmutation-data)); the window shapes transport, and
the pulse-schedule time-correction is applied afterwards unchanged (it is a linear
post-scaling and does not interact with the window). See
``examples/python/decay_photon_degvr.py``.

One caveat: reducing material density for the DeGVR passes lowers the activation
rate as well as the attenuation, so the decay-photon source scales with density
and the log-linear-in-density extrapolation is more approximate than for pure
attenuation. The window stays unbiased regardless (windows are unbiased for any
bounds); only its efficiency is affected. Use a modest generation budget (the D1S
activation path runs twice at reduced density).

## Notes

- With or without windows the tally means agree within statistics;
  only the variance (and the run cost) differ. Judge the benefit by coverage and
  the per-tally relative error in the deep region, or the per-tally FOM.
- Windows help most under exponential attenuation (thick shields, many decades
  of flux drop), where analog cannot reach at all. For pure void streaming (an air duct),
  analog already fills the domain and the gain is smaller.
- A weight-window history can be expensive (it can process thousands of split
  copies), so a timed run stops on ``max_runtime`` at the end of a chunk and may
  overrun the budget by up to one chunk; read the run's measured ``elapsed``.
- Both applying and generating windows are CPU only. ``compute='gpu'``
  rejects a model carrying weight windows, and ``generate_weight_windows`` takes
  a ``compute`` argument that accepts only ``'cpu'`` (the default). Generation is
  ordinary transport under a window, so it cannot move to the GPU before the
  kernels honour windows at all; asking for it raises ``ValueError`` saying so.
