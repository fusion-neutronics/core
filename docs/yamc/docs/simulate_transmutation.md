# Transmutation

### Model.simulate_transmutation()

Coupled transport-transmutation runs transport at each timestep with updated material compositions. Materials must have ``transmutable=True`` and a ``volume`` set.

<!-- doctest: skip -->
```python
yamc.neutron_cross_section_data = "endf-b8.1"

_chain = "transmutation-endf-b8.1-sfr.arrow"
yamc.transmutation_decay_data = _chain
yamc.transmutation_reactions = _chain
yamc.transmutation_fission_yields = _chain

model = yamc.Model(geometry=geometry, source=source)

schedule = yamc.PulseSchedule([
    yamc.Pulse(rate=1e14, duration=(1, "d"), source=source),   # irradiation, rate in n/s
    yamc.Pulse(rate=1e14, duration=(1, "d"), source=source),   # irradiation
    yamc.Cooldown(duration=(1, "d")),                          # cooling
])
results = model.simulate_transmutation(
    method="coupled",                    # or "independent"
    schedule=schedule,
    total_particles=5000,
)
```

The irradiation/cooling timeline is one ``PulseSchedule`` of ``Pulse`` / ``Cooldown`` steps, the same type used for the shutdown-dose-rate post-processing below. Each ``Pulse`` carries a duration (seconds, or a ``(value, unit)`` tuple) and a source ``rate`` in n/s; each ``Cooldown`` is decay-only. Transport uses the source configured on the ``Model``, so a schedule with more than one distinct ``Pulse`` source is rejected.

Methods:
- ``"coupled"``: transport runs every step with updated compositions
- ``"independent"``: transport runs once; reaction rates scaled by source rate per step (faster for low-burnup / fusion)

``simulate_transmutation`` accepts a ``compute`` argument for symmetry with
``simulate_transport``, but ``"cpu"`` (the default) is the only value it takes.
The transmutation solve has no GPU backend, and the per-step transport would
have to hand its rates back to a CPU solve every step even if it ran on the
device. Any other value raises a ``ValueError`` explaining this (not a
``TypeError`` about an unexpected keyword).

### Continuous feeds, extraction and transfer

Pass a `Processing` object to `simulate_transmutation()` to change inventories
continuously during irradiation and cooling. Every material it names must be a
transmutable material in the model, with its own volume. This example transfers
tritium from a lithium region to a surrounding collection region and feeds new
lithium during the first schedule step:

```python
import math

feed_material = yamc.Material(
    composition={"Li": 1.0}, density=0.534,
    volume=4 * math.pi * 10**3 / 3, transmutable=True,
)
collection_material = yamc.Material(
    composition={"Fe": 1.0}, density=7.87,
    volume=4 * math.pi * (20**3 - 10**3) / 3, transmutable=True,
)
inner_surface = yamc.Sphere(radius=10)
outer_surface = yamc.Sphere(radius=20, boundary="vacuum")
processing_geometry = yamc.Geometry([
    yamc.Cell(region=inner_surface.below, material=feed_material),
    yamc.Cell(region=inner_surface.above & outer_surface.below,
              material=collection_material),
])
processing_model = yamc.Model(geometry=processing_geometry, source=source)
processing = yamc.Processing()
processing.add_transfer(
    material=feed_material, component="H3", rate=1e-4,
    rate_units="1/s", destination=collection_material,
)
processing.add_feed(
    material=feed_material, composition={"Li6": 0.6, "Li7": 0.4},
    rate=1e-3, rate_units="g/s", steps=[0],
)
processed = processing_model.simulate_transmutation(
    method="independent", schedule=schedule, total_particles=5000,
    processing=processing,
)
```

A transfer rate is a first-order removal rate, not a mass flow. Omit
`destination` to extract material from the system, omit `component` to transfer
the whole composition, or name an element to select all its isotopes. Feed
compositions are normalized atom fractions, with absolute units such as `g/s`,
`kg/d`, `mol/h` or `atoms/s`. `steps` selects zero-based schedule steps; without
it, a term also applies during cooldowns. The same object works with
`method="coupled"` and with the standalone batch function `yamc.transmute()`.

### Cooling curves

A decay curve needs points at cumulative times, but ``Cooldown`` takes the
duration of its own step, so writing one by hand means differencing the times
first. ``cooldown_steps`` does that:

<!-- doctest: skip -->
```python
schedule = yamc.PulseSchedule(
    [yamc.Pulse(rate=1e14, duration=(1, "d"), source=source)]
    + yamc.cooldown_steps((1, "h"), (10, "a"), 12)
)
```

Twelve cooling points from one hour to ten years after shutdown. The step
durations sum to ``stop`` exactly: the first one runs from the end of the
irradiation to ``start``, rather than the steps spanning ``stop - start``.

Log spacing is the default because decay is exponential. A linear sweep wastes
steps in the flat tail and misses the early fall. Activity and decay heat are
plotted on log-log axes, so points evenly spaced on the plot are log-spaced in
time. ``spacing="linear"`` gives evenly spaced times, as in a fixed-interval
measurement campaign.

``start`` and ``stop`` take the same forms as a ``Cooldown`` duration: seconds,
or a ``(value, unit)`` tuple. ``n`` is the number of points and must be at least
2. Log spacing needs a positive ``start``, since the first point is a ratio from
it and zero has no logarithm.

### TransmutationResults

Results are keyed by each transmutable material's ``id``. Below, ``material`` is
the transmutable material in ``geometry``.

<!-- doctest: skip -->
```python
# Evolution of a nuclide over all timesteps
evolution = results.get_nuclide_evolution(material_id=material.id, nuclide="Co60")
print(results.times)  # cumulative times including t=0

# Density at a specific step
density = results.get_nuclide_density(material_id=material.id, nuclide="Co60", step=2)

# Full material at a step. The getters answer None for a material_id or step
# the run does not have, so a real script checks before using the result.
mat = results.get_material(material_id=material.id, step=2)
if mat is not None:
    print(mat.nuclides)

# All nuclides at a step
nuclides = results.get_material_nuclides(material_id=material.id, step=2)
```

Gas production and data-coverage diagnostics are available on the same results:

```python
gas = results.get_gas_production(material_id=material.id)
if gas is not None:
    print(list(zip(results.times, gas["H"], gas["He"])))
screening = results.get_product_screening_info(material_id=material.id)
if screening is not None:
    print(screening["loaded"], screening["below_threshold"], screening["unavailable"])
print(results.data_gaps)
```

Gas is in appm relative to the initial number of atoms, with the initial gas
subtracted by default; `produced=False` instead reports all gas present.
Unknown gas contributions are not silently zero-filled when the chain lacks a
gas nuclide. Screening leaves a small product's production and decay in the
inventory, but may omit its neutron reactions; inspect `unavailable` and
`data_gaps` before treating a small result as a complete prediction.

### Saving results

``to_arrow()`` writes ``TransmutationResults`` to an Arrow IPC file, and
``yamc.TransmutationResults.from_arrow()`` reads it back with every getter
answering as before, to the bit. Without a path ``to_arrow()`` returns the
file's bytes, for a browser or Pyodide session with no file system to write
to.

<!-- doctest: skip -->
```python
results.to_arrow("steel.arrow")
results = yamc.TransmutationResults.from_arrow("steel.arrow")

data = results.to_arrow()                      # bytes
results = yamc.TransmutationResults.from_arrow(data)
```

A calculation too big to hold at once can be split into parts, each saved and
dropped before the next. Every file has the same columns, one row per stored
number (``material_id``, ``material_name``, ``step``, ``quantity``,
``nuclide``, ``reaction``, ``product``, ``value``), so the parts read as one
table without yamc, with ``polars.scan_ipc("part_*.arrow")`` for example.

The file holds the results, not the nuclear data behind them. Three getters
read that data again: ``get_production_routes`` and ``get_isomeric_branching``
walk the transmutation chain, and ``get_reaction_rate_spectrum`` reads the
initial composition's cross sections. The file pins both, the chain by a hash
of its contents and each nuclide by the file it was read from, and
``from_arrow()`` reattaches them from the configured ``yamc.transmutation_*``
sources and those files. Where that data is not the same or is no longer
there, it warns once, and those three getters raise rather than answer from
other data.

### Tallies at each step

The model's own tallies (``Model(tallies=[...])``) are scored by every step's
transport, so anything that changes as the materials burn (TBR as Li6 depletes,
heating, flux, spectra, mesh tallies) comes out of the one run, with no need to
rebuild the model from ``get_material`` and run ``simulate_transport`` again:

<!-- doctest: skip -->
```python
tbr = yamc.Tally(name="tbr", cells=breeder, scores=["H3-production"])
model = yamc.Model(geometry=geometry, source=source, tallies=[tbr])
results = model.simulate_transmutation(method="coupled", schedule=schedule, total_particles=5000)

for step, t in enumerate(results.timesteps):
    step_results = results.tally_results(step)   # a TransportResults, or None
    if step_results is not None:
        print(t, step_results[tbr].mean, step_results[tbr].standard_deviation)

means, std_devs = results.tally_evolution(tbr)   # one list of bins per transport step
```

``tally_results(step)`` returns the whole ``TransportResults`` of that step's
transport, read exactly as ``simulate_transport``'s are (by tally or name,
``to_numpy``, mesh tallies included). ``step`` indexes ``timesteps``, as
``get_source_rates`` and ``get_reaction_rates`` do: step ``i`` is the transport
that drove the materials from composition ``i`` to composition ``i + 1``. Each
step's tallies are reset before its transport, so they reflect that step's
compositions only.

- Values are per source particle, as ``simulate_transport`` returns them;
  multiply by ``get_source_rates(material_id)[step]`` for absolute rates.
- A decay-only step (a ``Cooldown``, or a ``Pulse`` with rate 0) runs no
  transport and returns ``None``; ``tally_evolution`` skips it.
- Only the model's own tallies come back. The transmutation's internal reaction
  rate tallies and its scouting transport are not among them.
- With ``method="independent"`` transport runs once, on the initial
  compositions, and every irradiation step returns that one run's results: they
  do not follow the burnup. Use ``method="coupled"`` for a TBR that falls as the
  Li6 burns.
- Every step keeps its own results, so a large mesh tally costs its size once
  per transport step.

### Production routes from the chain

``TransmutationChain`` carries the topology a solve steps through: which nuclide becomes which, and how. ``reactions`` holds the neutron-induced edges, ``decays`` the radioactive ones, both keyed by parent and both giving ``(kind, target, branching)``:

<!-- doctest: skip -->
```python
chain = yamc.TransmutationChain.from_arrow("transmutation-endf-b8.1-sfr.arrow")

chain.reactions["W186"]   # [('(n,2n)', 'W185', 1.0), ..., ('(n,a)', 'Hf183', 1.0)]
chain.decays["Hf183"]     # [('beta-', 'Ta183', 1.0)]
chain.decays["W185_m1"]   # [('IT', 'W185', 1.0)]
```

Stable nuclides have no decay modes and are absent from ``decays``, as they are from ``half_lives``. A ``target`` of ``None`` means the channel names no single product: fission on the reaction side, a product outside the chain on the decay side. Decay modes carry the evaluation's own spellings (``"beta-"``, ``"ec/beta+"``, ``"alpha"``, ``"IT"``, ``"sf"``, and emissions written as ``"beta-,n"``), and the branchings for one parent sum to 1.

Because the two share a shape they can go in one edge list, which is what a route walk needs. With reactions alone a walk stops at the first product, one step short of most real routes:

<!-- doctest: skip -->
```python
edges = {}
for parent, es in chain.reactions.items():
    edges.setdefault(parent, []).extend(es)
for parent, es in chain.decays.items():
    edges.setdefault(parent, []).extend(es)

for kind1, mid, _ in edges["W186"]:              # two-step routes from W186 to Ta183
    for kind2, end, _ in edges.get(mid, []):
        if end == "Ta183":
            print(f"W186{kind1}{mid}({kind2}){end}")

# W186(n,3n)W184((n,np))Ta183
# W186(n,4n)W183((n,p))Ta183
# W186(n,a)Hf183(beta-)Ta183      <- the one that needs the decay edge
```

The branching on an edge is a share of its channel. It says how a channel splits and nothing about how often the channel fires, so these numbers alone cannot weight two routes into the same product against each other.

### Per-edge reaction rates

The solve computes the rate of every individual production edge to build its burnup matrix, whose entries are sums over the edges into each product. ``get_reaction_rates`` returns those edges, so routes can be weighted as well as listed:

<!-- doctest: skip -->
```python
# None for a material_id or step the run does not have; `or {}` keeps the rest
# of the page working on a plain dict.
rates = results.get_reaction_rates(material_id=material.id, step=0) or {}   # step indexes timesteps
rates["Fe58"]["(n,2n)"]        # [('Fe57', 1.205e-09)]            # rate in 1/s
```

``parent -> reaction kind -> [(target, rate [1/s])]``. A ``target`` of ``None`` is a channel naming no single product, which in practice means fission, whose products come from the yields rather than from an edge. A decay-only step drove no reaction and reports an empty map.

Here ``step`` is the schedule step, the same index as ``timesteps`` and ``get_source_rates(material_id)``, which is one less than the ``step`` the composition getters take (where 0 is the initial composition).

Both methods report the same quantity in the same shape. ``method="coupled"`` takes the rates from that step's own transport tallies and ``method="independent"`` from the single transport's tallies, scaled by the step's source rate. Either way an edge rate is that rate times the branching of the chain the step was solved with, isomeric overlay included, so a consumer does not have to know which method produced the result.

The share of a product arriving down each route is the main column of a decay heat pathway analysis, and it is what makes a C/E useful for diagnosis. A foil reading 2.5x the measurement says something is wrong; knowing that 73% of the dominant product arrives down one channel says which evaluated quantity to check.

<!-- doctest: skip -->
```python
from collections import defaultdict

into = defaultdict(list)
for parent, kinds in rates.items():
    for kind, edges in kinds.items():
        for target, rate in edges:
            if target is not None:
                into[target].append((rate, parent, kind))

routes = sorted(into["Fe57"], reverse=True)
total = sum(rate for rate, _, _ in routes)
for rate, parent, kind in routes:
    print(f"{rate / total:6.1%}  {parent}{kind}Fe57   {rate:.4g} /s")
```

For a 14.06 MeV point source in a 10 cm iron sphere, one hour at 1e18 n/s:

```text
 73.3%  Fe58(n,2n)Fe57       1.205e-09 /s
 18.3%  Co58(n,np)Fe57       3.016e-10 /s
  6.1%  Ni60(n,a)Fe57        1.009e-10 /s
  1.3%  Co58(n,d)Fe57        2.176e-11 /s
  0.6%  Fe56(n,gamma)Fe57    9.747e-12 /s
  0.3%  Ni61(n,na)Fe57       4.809e-12 /s
  0.0%  Co59(n,t)Fe57        6.359e-13 /s
  0.0%  Ni59(n,3He)Fe57      8.149e-15 /s
```

A rate is per atom of the parent, so weighting a route by inventory means multiplying by the parent's density at that step (``get_nuclide_density``). Over one short step the two together reproduce the step's own burnup: the Fe57 produced is the summed edge rate times the parent densities times the step length.

### Uncertainty with the coupled method

``data_uncertainty`` works with both methods, and the inventories, activity,
decay heat and dose come back with a standard deviation as they do from
[Material.transmute()](materials.md#nuclear-data-uncertainty). With
``method="independent"`` the one transport's rates are resampled and each
replica's schedule is solved again. With ``method="coupled"`` that would not
describe the spread: each step's rates were tallied in compositions that the
earlier steps' noise and data had already moved. Each replica therefore runs
the whole coupled calculation again, every step's transport included, on its own
draw:

- ``"statistical"``: the replica has its own transport seed. Without this source
  every replica keeps the nominal seed, and the spread is the nuclear data's.
- ``"cross_sections"``: every transport of the replica, in every material, reads
  nuclear data with the replica's MF=33 and resonance-parameter draw applied, so
  the draw moves the flux, the rates, the compositions and every later step's
  flux. A product that enters transport part way through the run keeps its draw.
- ``"isomer_branching"``: every step of the replica tallies its isomeric partials
  on production curves that carry the replica's MF=40 draw, and splits them at
  each collision energy by the nominal rule.
- ``"half_life"``, ``"decay_branching"`` and ``"fission_yield"`` are used in each
  of the replica's solves, and ``"decay_energy"`` and ``"decay_photon_lines"``
  move what is evaluated from its inventories.

Replica ``k`` draws the same data as replica ``k`` of the independent method, so
the two can be compared replica by replica. Nothing is linearised, and the
nominal results are bit-identical to a run without ``data_uncertainty``. The
default request, every source, runs. ``"cross_sections"`` is carried only beside
``"statistical"``: a replica on its own cross sections follows its own
histories whatever its seed, so its spread holds the transport's noise as well,
and naming ``"cross_sections"`` alone raises. The independent method's replica
weights do separate the two.

<!-- doctest: skip -->
```python
iron = yamc.Material(composition={"Fe": 1.0}, density=7.87, volume=4188.79,
                     transmutable=True)
dt = yamc.NeutronSource(energy=14.1e6)
sphere = yamc.Sphere(radius=10.0, boundary="vacuum")
model = yamc.Model(
    geometry=yamc.Geometry([yamc.Cell(region=sphere.below, material=iron)]),
    source=dt,
)
schedule = yamc.PulseSchedule([
    yamc.Pulse(rate=1e18, duration=(1, "h"), source=dt),
    yamc.Pulse(rate=1e18, duration=(1, "h"), source=dt),
    yamc.Cooldown(duration=(1, "h")),
])
results = model.simulate_transmutation(
    method="coupled",
    schedule=schedule,
    total_particles=2000,
    data_uncertainty=yamc.DataUncertainty(samples=16,
                                          sources=["statistical", "half_life"]),
)
mean = results.get_nuclide_density(material_id=iron.id, nuclide="Mn56", step=3)
sigma = results.get_nuclide_uncertainty(material_id=iron.id, nuclide="Mn56", step=3)
if mean is not None and sigma is not None:
    print(f"Mn56: {mean:.3e} +/- {sigma:.3e}  ({100 * sigma / mean:.1f}%)")
# Mn56: 5.681e-08 +/- 7.772e-10  (1.4%)
info = results.get_data_uncertainty_info(material_id=iron.id)
if info is not None:
    info["flux_response"]   # "coupled_replicas"
```

The cost is the transport of every irradiation step once per replica, the 16
replicas above taking about 7 seconds on 32 cores. Leave ``samples`` unset and
replicas are added 64 at a time by the [convergence
rule](materials.md#nuclear-data-uncertainty), at least 128 of them, so ten
irradiation steps of ten minutes' transport each come to about 210 hours.
Each nuclide whose resonance parameters are sampled adds a few seconds per
replica to rebuild its resonance range. ``samples`` bounds the cost.

``DataUncertainty(attribution=True)`` reruns the ensemble once per applied
source, each source drawn alone on the same streams, so it costs one more
coupled ensemble per source. The ``"cross_sections"`` entry of ``by_source`` is
the variance with ``"statistical"`` and ``"cross_sections"`` less the variance
with ``"statistical"`` alone, the two sharing their transport seeds, and comes
out negative where the cross sections move a nuclide by less than the
transport's noise. There are no first-order ``contributors``, and every
``linearity`` entry is ``None``. On the sphere above:

<!-- doctest: skip -->
```python
attributed = model.simulate_transmutation(
    method="coupled",
    schedule=schedule,
    total_particles=2000,
    data_uncertainty=yamc.DataUncertainty(
        samples=16, sources=["statistical", "cross_sections"], attribution=True,
    ),
)
breakdown = attributed.get_uncertainty_breakdown(material_id=iron.id, nuclide="Mn56", step=3)
if breakdown is not None:
    breakdown["variance"]    # 3.44e-18, Mn56 at 3.3%
    breakdown["by_source"]   # {"cross_sections": 2.84e-18, "statistical": 6.04e-19}
```

The statistical entry is the square of the first example's Mn56 sigma
(7.77e-10), whose replicas ran on the same transport seeds, and the iron cross
sections carry the rest. With the cross sections each replica rebuilds the
resonance ranges of the iron isotopes for its draw, so the two ensembles of 16
replicas took about 200 seconds.

### Statistical uncertainty of the tallied rates

Run ``method="independent"`` with ``data_uncertainty`` including the
``"statistical"`` source and ``get_reaction_rate_uncertainty`` gives each
tallied reaction rate with its Monte Carlo standard deviation at one step:

<!-- doctest: skip -->
```python
uncertain = model.simulate_transmutation(
    method="independent",
    schedule=schedule,
    total_particles=5000,
    data_uncertainty=yamc.DataUncertainty(sources=["statistical"]),
)
for nuclide, reaction, target, rate, std_dev in (
    uncertain.get_reaction_rate_uncertainty(material_id=material.id, step=0) or []
):
    print(nuclide, reaction, target, rate, std_dev)   # 1/s per atom
```

A reaction total has a ``target`` of ``None`` and an isomeric partial names its
final state. The rates are the tally's, scaled by the step's source rate,
as the step's solve used them. They were scored by the same histories
and so are correlated: the inventory sigmas carry those correlations, and these
standard deviations alone do not.

### Isomeric branching report

``get_branching_report(material_id, step)`` says, per channel, how the
isomeric-branching evaluation's split was applied over that step's spectrum:
whether its list was read as shares of the total, as absolute productions or,
for the light particles of ``(n,X)`` (MT=5), as multiplicities, and how much of
the parent's removal rate rests on anything the evaluation does not give.

<!-- doctest: skip -->
```python
report = results.get_branching_report(material_id=material.id, step=0)
if report is not None:
    report["channels"]         # one dict per channel: parent, reaction, mt, file,
                               # representation ("share", "absolute" or
                               # "multiplicity"), states, ...
    report["dropped"]          # channels that were not folded, with the reason
    report["unmodelled_mt5"]   # [(nuclide, share, reason)]: parents whose MT=5
                               # residuals the chain does not model
```

A run refuses when the clipped or held production of its channels, each parent
weighted by its density, is more than 0.1% of the material's neutron removal
rate, and likewise when the material's own nuclides lose more than that to MT=5
residuals the chain does not model. Each channel's own shares are of its
parent's removal rate, so a trace parent can read above 0.1% in a run that was
not refused.
The step is the schedule step, as ``get_reaction_rates`` takes it, and a
decay-only step gives empty lists.

### Material.transmute()

For standalone transmutation with a pre-computed flux spectrum (no transport), see [Material transmutation](materials.md#transmutation) in the Materials section.
