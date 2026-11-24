
## Running the Simulation

### model.simulate_transport()

<!-- doctest: skip -->
```python
results = model.simulate_transport()
```

``simulate_transport()`` returns a ``TransportResults`` object containing a ``TallyResult`` for every tally attached to the model, plus run metadata (``elapsed``, ``transport_time``, ``particles_per_second``, and ``runs``: per run the seed, ``n_histories``, the data libraries and the published release of each, ``data_releases``).

A run stops on a particle budget (``total_particles``), a wall-time budget (``max_runtime``), or convergence targets on the model. They are OR-combined, so the run ends at the first one satisfied. ``total_particles`` defaults to ``None`` (no particle cap).

With none of them set, the run goes on until it has converged (see [Running until converged](#running-until-converged)):

<!-- doctest: skip -->
```python
results = model.simulate_transport()  # stops once every bin passes the statistical checks
```

#### Accessing tally results

A ``TallyResult`` can be looked up four ways: by the ``Tally`` object, by name, by id, or by iteration. Names and ids are optional; set them on the ``Tally`` constructor to look results up that way.

<!-- doctest: skip -->
```python
flux_tally = yamc.Tally(scores=["flux"], cells=cell, name="flux", id=1)
heating_tally = yamc.Tally(scores=["heating"], cells=cell)

model = yamc.Model(
    geometry=geometry,
    source=source,
    tallies=[flux_tally, heating_tally],
)
results = model.simulate_transport()

# By Tally object (always works)
flux_result = results[flux_tally]

# By name (requires name= on the Tally)
flux_result = results["flux"]

# By id (requires id= on the Tally)
flux_result = results[1]

# Iterate over (tally, result) pairs
for tally, result in results.items():
    print(tally.name, result.mean)

# Membership test
assert flux_tally in results
assert "flux" in results
```

A ``TallyResult`` exposes ``mean``, ``standard_deviation``, ``relative_error``, ``variance``, ``shape``, ``dim_labels``, and ``to_numpy()``, plus a set of statistical-reliability metrics and a PASS/FAIL verdict that print whenever you display a result (see [Statistical reliability](results.md#statistical-reliability) and [Results](results.md) below).

### Running until converged

A run with no ``total_particles``, ``max_runtime`` or convergence targets stops once **every scored bin of every tally** passes the [statistical checks](results.md#statistical-reliability) at two checkpoints in a row. Each bin is judged as if it were a single-bin tally: from its own mean, error and history, so a mesh or spectrum tally converges bin by bin, not just in its total. Bins that never scored (a region no particle reaches) are not waited for.

``convergence=`` chooses how many checks must pass. Each level includes every check of the looser levels:

| Check | ``"loose"`` | ``"standard"`` (default) | ``"strict"`` |
|-------|:-----------:|:------------------------:|:------------:|
| relative error < 0.10 | ✓ | ✓ | ✓ |
| mean stable (no drift) | ✓ | ✓ | ✓ |
| relative error decreasing | ✓ | ✓ | ✓ |
| relative error falls as 1/√N | ✓ | ✓ | ✓ |
| figure of merit stable | ✓ | ✓ | ✓ |
| figure of merit has no trend | ✓ | ✓ | ✓ |
| variance of variance < 0.10 |  | ✓ | ✓ |
| variance of variance decreasing |  | ✓ | ✓ |
| variance of variance falls as 1/N |  | ✓ | ✓ |
| large-score tail slope > 3 |  |  | ✓ |
| **checks** | **6** | **9** | **10** |

<!-- doctest: skip -->
```python
results = model.simulate_transport()                       # "standard"
results = model.simulate_transport(convergence="loose")    # quickest; can stop early on heavy-tailed scores
results = model.simulate_transport(convergence="strict")   # every check, every bin
```

Choosing a level:

- ``"loose"`` uses only each bin's mean and error. It is the quickest, but a rarely sampled large score can leave the relative error looking settled while it is not.
- ``"standard"`` adds the variance of the variance, which catches exactly that: an error estimate that is itself unreliable. It keeps two more numbers per bin per thread (16 bytes).
- ``"strict"`` adds the tail slope of each bin's score distribution, which needs each bin's 201 largest scores: about 1.6 KiB per scored bin per thread. That is fine for cell and spectrum tallies; for a mesh of a million voxels on 16 threads it is about 26 GB.

``convergence=`` only applies when no other stop condition is set, and combining it with ``total_particles``, ``max_runtime`` or convergence targets raises ``ValueError``.

The checks are made at checkpoints every ~25% more histories, and the progress output names the level and how many checks the slowest bin passes, with an ETA projected from the slowest bin:

```text
No stop condition set: running until every scored bin passes the standard statistical checks (9).
Progress: 12500 particles  elapsed 1.2s, ETA ~3.4s (statistical checks 7/9 passing)
Every scored bin passed the standard statistical checks (9) after 51046 particles; stopping.
```

Converging every bin can take a long time, for example deep in a shield without variance reduction. When, after a minute, the ETA is more than ten times the time already spent, a warning names the slowest tally bin and its relative error. Cancel the run, add [weight windows](weight_windows.md), or set ``total_particles``, ``max_runtime`` or a ``ConvergenceTarget`` to stop sooner.

Running until converged is CPU only and single process. On the GPU or under MPI, set a stop condition.

### Progress output (``verbose``)

The ``verbose`` argument on ``Model`` is a list of independent flags selecting which categories of progress are printed during transport. Progress is reported in source particles (the run is chunked internally, but you never deal with batches). The flags are:

- ``"progress"``: periodic ``Progress: N/total particles (P%)`` lines.
- ``"eta"``: progress lines that also show elapsed wall-clock and ETA.
- ``"tally"``: each tally's running mean ± std (rel. err) as it converges.
- ``"summary"``: the end-of-run completion line and per-tally final statistics.
- ``"nuclear_data"``: one line per nuclear-data file as it loads (nuclide and path), plus the transmutation chain file.

Any combination is allowed. The default (pass nothing, or ``None``) is ``["eta", "tally", "summary"]``. An empty list ``[]`` prints nothing, not even the end-of-run summary.

<!-- doctest: skip -->
```python
# Default output
model = yamc.Model(geometry=geometry, source=source, tallies=[tally])

# Fully silent, for scripts that post-process results themselves
model = yamc.Model(geometry=geometry, source=source, tallies=[tally], verbose=[])

# Just the final summary, no per-particle progress
model = yamc.Model(geometry=geometry, source=source, tallies=[tally], verbose=["summary"])

# Read or change after construction
print(model.verbose)                       # ['eta', 'tally', 'summary']
model.verbose = ["progress", "summary"]
```

### Threading

By default all available CPU cores are used. Specify a thread count explicitly:

<!-- doctest: skip -->
```python
model.simulate_transport(threads=8)
```

### GPU transport (``compute="gpu"``)

The transport loop has a GPU implementation that runs on Vulkan via cubecl. Select it with ``compute="gpu"`` on ``simulate_transport``:

<!-- doctest: skip -->
```python
# Check that a usable GPU is present at module level.
if yamc.parallel.gpu_available():
    results = model.simulate_transport(total_particles=1_000_000, compute="gpu")
else:
    results = model.simulate_transport(total_particles=1_000_000)
```

On a machine with more than one GPU (e.g. an integrated GPU plus a dedicated one), ``compute="gpu"`` auto-selects an adapter (preferring a discrete GPU via the high-performance power preference). To pin a specific adapter, list the available ones with ``yamc.parallel.list_gpu_adapters()`` and pass its name directly as ``compute=<name>``:

<!-- doctest: skip -->
```python
adapters = yamc.parallel.list_gpu_adapters()
print(adapters)
# e.g. ['NVIDIA GeForce RTX 4070', 'Intel(R) UHD Graphics']

# Pin a specific GPU by name (e.g. the dedicated one):
results = model.simulate_transport(total_particles=1_000_000, compute=adapters[0])
```

``yamc.parallel.gpu_available()`` returns ``True`` only when a GPU with f64 compute support is reachable. The kernel uses double-precision throughout (cross-section interpolation, sampling, and the variance accumulator), so single-precision-only consumer GPUs are rejected.

The GPU path supports a defined subset of the options, verified to match the CPU path on real discrete hardware (NVIDIA RTX A500): `cpu_gpu_equivalence_two_materials` is bit-exact and the `gpu_*.rs` GPU-vs-CPU suite passes. It currently supports:

- Neutron transport, primary-photon transport (a `PhotonSource`), coupled neutron-to-photon secondary production (`transport_secondary_photons=True`) and D1S decay photons (`use_decay_photons=True`). D1S on a mixed neutron and photon source is rejected.
- Tallies with either estimator: flux, heating / heating-local, damage-energy, reaction rates (total / elastic / absorption / (n,γ) / inelastic / fission), gas-production, and photon cross-section scores. Tally covariance (`covariance=True`) and virtual-overlay tallies (`response=`) are rejected.
- Tally filters: `cells=`, `materials=`, `mesh=`, `energy_bins=`, `particle=`, `nuclides=`, `parent_nuclides=`, and `energy_function=` / `dose_coefficients=`. The energy function weights rather than bins, so it still needs a spatial binner alongside it; its interpolation table (cubic, linear or log-log) is built on the host and evaluated in the kernel, so cubic and linear match the CPU bit for bit and log-log to about an ulp.
- Full neutron physics: elastic (tabulated angular), inelastic (Kalbach-Mann, n-body, evaporation), free-gas thermal, URR, and the photon interactions on the photon path.
- CSG geometry only (mesh is rejected), with every CSG surface type.
- Survival biasing is supported. Weight windows are rejected, not silently ignored.
- The kernels always surface-track: a `tracking_mode` of `"woodcock"` or `"hybrid"` gives a warning and runs with surface tracking.

If any unsupported option is used, ``compute="gpu"`` raises a ``ValueError`` at simulate-transport start with a specific message. If no GPU with f64 compute is available, it raises ``RuntimeError``. Passing ``threads``, ``capture_tracks`` or ``data_uncertainty`` with ``compute="gpu"`` raises ``ValueError`` (the kernel parallelises across all particles in one launch, and tracking and nuclear-data uncertainty are CPU-only). Repeated `compute="gpu"` calls in one process are safe; each resets the shared GPU client.

For the full, always-current per-option breakdown of which CPU options the GPU path supports (and which it rejects), see the CPU-vs-GPU option matrix harness `crates/yamc/tests/gpu_cpu_comparison_matrix.rs`, which is the source of truth.

### MPI (distributed memory)

For multi-node parallelism, build YAMC with MPI support (see [installation](installation.md)).

<!-- doctest: skip -->
```python
# Check MPI support
if yamc.parallel.has_mpi():
    print(f"Rank {yamc.parallel.mpi_rank()} of {yamc.parallel.mpi_size()}")
```

Run with ``mpirun``:

```bash
mpirun -np 4 python simulation.py
```

Hybrid MPI + threading:

<!-- doctest: skip -->
```python
model.simulate_transport(total_particles=10000, threads=8)  # 8 threads per MPI rank
```

```bash
# 4 ranks x 8 threads = 32 cores
mpirun -np 4 python simulation.py
```

yamc also runs inside an MPI job that another code started, such as a
coupled solve with mpi4py, FEniCSx or preCICE. If MPI is already initialised
when yamc needs it, yamc joins it on the world communicator and leaves
finalising it to the host:

<!-- doctest: skip -->
```python
from mpi4py import MPI  # the host initialises MPI

import yamc

results = model.simulate_transport(total_particles=1_000_000)  # every rank of MPI.COMM_WORLD
```

Tallies are combined across ranks once, at the end of the run: rank 0 folds every rank's statistics in rank order and broadcasts them, so every rank returns the same result. Read and save results on rank 0 (check ``yamc.parallel.mpi_rank() == 0``); ``yamc.combine_results`` refuses a result from any other rank, since combining two ranks' copies would count the run twice. The tally-level reliability statistics (the ``aggregate_*`` values, the score PDF and the checks built on them) are not gathered across ranks yet, so they are reported for single-process runs only.

Each history's random numbers depend only on the base seed and its global index, not on the thread, chunk or rank that runs it, so the same histories run whatever the rank count; merged statistics can differ in the last bits, since the fold order changes with it.

Slurm example:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
srun python simulation.py  # In script: model.simulate_transport(total_particles=1_000_000, threads=32)
```

### Particle tracking

Track particle histories for debugging or visualization. Tracks are off by default (zero overhead) and are only captured when ``capture_tracks=`` is set on ``simulate_transport()``. Histories are addressed by their global 0-based index, so the captured set is identical regardless of thread count:

- ``capture_tracks=N``: the first ``N`` histories (recommended for debugging)
- ``capture_tracks=range(20, 30)``: histories 20..29 (half-open); strides work too, e.g. ``range(0, 1000, 10)`` for every 10th of the first 1000
- ``capture_tracks=[20, 30, 45]``: exactly those history indices
- ``capture_tracks='all'``: every history (warns; high memory on large runs)
- ``capture_tracks=None`` / omitted: no tracking

When tracking is enabled the captured ``Tracks`` are attached to the returned ``TransportResults`` as ``results.tracks``; otherwise ``results.tracks`` is ``None``.

A captured history holds one track per particle transported in it: the source particle and every secondary it produced (neutrons from (n,2n) and fission, and photons with ``transport_secondary_photons=True``), each with a ``particle_type`` of ``"neutron"`` or ``"photon"``. A track is identified by its ``history`` and its ``particle_id``, which is 0 for the source particle and numbers the secondaries within the history the same way on any thread count. ``parent_id`` and ``generation`` give each secondary's lineage. ``len(tracks)`` is the number of particle tracks, which is more than the number of histories captured once secondaries are made: capturing 5 histories of 14 MeV neutrons in an iron sphere with secondary photons on gives 71 tracks.

<!-- doctest: skip -->
```python
results = model.simulate_transport(total_particles=10000, capture_tracks=100)
tracks = results.tracks  # None if capture_tracks= was not set
assert tracks is not None

print(len(tracks))            # number of particle tracks (source and secondaries)
print(tracks.total_events())  # total events across all tracks

# Iterate over individual particle tracks
for track in tracks.tracks[:3]:  # first 3 particles
    print(f"History {track.history}, {track.particle_type} {track.particle_id} "
          f"(parent {track.parent_id}, generation {track.generation}):")
    for event in track.events[:5]:  # first 5 events
        print(f"  {event.event_type:12s}  E={event.energy_in:.4e} -> {event.energy_out:.4e} eV"
              f"  pos=({event.x:.2f}, {event.y:.2f}, {event.z:.2f})"
              f"  cell={event.cell_id}  nuclide={event.nuclide}  MT={event.reaction_mt}")

# Convert to records for pandas
import pandas as pd
df = pd.DataFrame(tracks.to_dataframe_records())
print(df[["particle_type", "event_type", "energy_in", "energy_out", "x", "y", "z", "nuclide"]].head(10))
photons = df[df.particle_type == "photon"]
```

Each ``TrackEvent`` includes: ``particle_type``, ``event_type``, ``position``, ``direction``, ``energy_in``, ``energy_out``, ``weight``, ``cell_id``, ``reaction_mt``, ``nuclide``, and ``birth_reaction``. A track starts with a ``source_birth`` or a ``secondary_birth`` event, and the other event types are ``collision``, ``surface_crossing``, ``absorption``, ``russian_roulette`` and ``leak``.

#### Viewing tracks in ParaView

``tracks.to_vtkhdf()`` writes one polyline per particle track to a VTK-HDF file for [ParaView](https://www.paraview.org/) (needs ``h5py``, from ``pip install "yamc[viz]"``). The coordinates are in cm, as in ``model.to_vtkhdf()``, so the tracks overlay the geometry when both files are opened together.

<!-- doctest: skip -->
```python
tracks.to_vtkhdf("tracks.vtkhdf")
model.to_vtkhdf("model.vtkhdf")  # the geometry and source to show them in
```

Each point carries ``energy`` (eV, after the event), ``weight``, ``event_type`` and ``cell_id`` (-1 where unknown), and each line carries ``particle_type``, ``particle_id``, ``parent_id`` (-1 for a source particle), ``generation`` and ``history``, so the tracks can be coloured by any of them. ``event_type`` and ``particle_type`` are stored as integer codes, which ParaView can colour by; the names they stand for are in the file's root attributes ``event_type_names`` and ``particle_type_names``. ``tracks.polylines()`` returns the same arrays as a dict, and ``yamc.vtkhdf.polylines_to_vtkhdf(filename, points, offsets, point_data=..., cell_data=...)`` writes any set of polylines in the same format.

### Lost particles and geometry debugging

If a particle crosses into a region not covered by any cell, or reaches an unbounded cell with no surface ahead of it (a missing vacuum boundary), it is "lost". YAMC prints diagnostics, kills the particle and counts it towards ``max_lost_particles``.

On a triangulated ``MeshGeometry`` a particle is also lost at a point the mesh cannot place (a stray or flipped facet makes the test rays disagree), and where it leaves a volume through a crack in the volume's surface (see [Mesh geometry](method.md#mesh-geometry)). A defective mesh therefore shows up as particles lost at the defect, rather than as particles carried on as void and leaking out uncounted.

<!-- doctest: skip -->
```python
model = yamc.Model(
    geometry=geometry,
    source=source,
    max_lost_particles=10,  # abort after this many (default 10)
)
model.simulate_transport(total_particles=1000)

# Inspect lost particles
for lp in model.lost_particles:
    print(f"Position: {lp.position}, Surface: {lp.surface_id}")
```


