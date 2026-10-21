# yamc-core

Yet Another Monte Carlo: neutron and photon transport for fusion neutronics.
Constructive solid and mesh geometry, coupled neutron-photon transport, CAD
import, flexible tallies and transport-coupled transmutation, with the whole
solver in Rust.

This is the compiled distribution, and it provides the `yamc` module itself, so
the import name and the distribution name differ (as `pillow` provides `PIL`).
Install [`yamc`](https://pypi.org/project/yamc/) rather than this: it pins the
matching version of this wheel and is where the documentation lives.

```python
import yamc

# A point source inside a Li/Be sphere.
inner = yamc.Sphere(x0=0, y0=0, z0=0, radius=1.0)
outer = yamc.Sphere(x0=0, y0=0, z0=0, radius=200.0, boundary="vacuum")

breeder = yamc.Material(
    composition={"Li6": 0.035, "Li7": 0.465, "Be9": 0.5},
    density=2.0,
    temperature=294,
)
breeder.read_nuclear_data({
    "Be9": "tests/Be9.arrow",
    "Li6": "tests/Li6.arrow",
    "Li7": "tests/Li7.arrow",
})

void = yamc.Cell(name="void", region=inner.below)
blanket = yamc.Cell(name="blanket", region=inner.above & outer.below, material=breeder)
geometry = yamc.Geometry([void, blanket])

source = yamc.NeutronSource(
    energy=yamc.sources.fusion_neutron_spectrum(20000.0),
    position=(0, 0, 0),
)
tally = yamc.Tally(name="tbr", cells=blanket, scores=["H3-production"])

model = yamc.Model(geometry=geometry, tallies=[tally], source=source)
results = model.simulate_transport(total_particles=250_000, seed=1)
print(results[tally].mean)
```

## CAD imprint threads

The CAD extra uses stable CadQuery from PyPI. Limit imprinting to fewer
threads to reduce its peak RAM usage, at the cost of speed:

```python
import cadquery
from yamc.cad import CadToYamc

assembly = cadquery.Assembly()
assembly.add(cadquery.Workplane().box(1, 1, 1), name="left")
assembly.add(cadquery.Workplane().box(1, 1, 1).translate((1, 0, 0)), name="right")
mesher = CadToYamc()
mesher.add_cadquery_object(assembly, ["steel", "aluminum"])
mesh = mesher.mesh_surfaces(imprint_threads=2)
```

Both `mesh()` and `mesh_surfaces()` accept `imprint_threads="all"` (the
default) or a positive integer. Only imprinting uses this setting: the prior
OCC thread-pool size and default launch count are restored before meshing,
including when imprinting fails. Skipped and single-solid imprints do not
change the thread pool. Glue behavior is unchanged.

## Relationship to yani

`yamc` is a superset: it ships everything
[`yani`](https://pypi.org/project/yani/) has (materials, decay data,
irradiation schedules, inventories) plus neutron and photon transport,
geometry, tallies and transport-coupled transmutation. The two are
**alternatives**. Installing both puts two extension modules in one process, so
`yamc.Material` and `yani.Material` are distinct types and the nuclear-data
configuration (`cross_section_data`, the four `transmutation_*` sources) exists
twice, once per package.

Pick `yamc` when the neutron spectrum should come from a transport solve; pick
`yani` when you have a spectrum already and want none of the transport stack.

## What it does not do

- Tally reliability checks include a relative-error check of every populated
  bin. Higher moments and the 201-sample tail fit remain aggregate checks,
  not per-bin tail guarantees. An unestimable tail gives an incomplete
  verdict, not a pass. Both figure-of-merit checks need at least 0.1 s of
  last-half history; mean stability uses a 95% Student-t slope test on
  disjoint block means rather than monotonic cumulative snapshots.
- Fixed-source transport only. No criticality or eigenvalue calculations.
- One transmutation stepper (`ForwardEulerStepper`, beginning-of-step rates).
  No predictor-corrector.
- Nuclear-data uncertainty covers MF=33 cross sections, plus the decay data
  for transmutation.
  - `simulate_transport(data_uncertainty=...)` gives every tally a nuclear-data
    standard deviation and a replica mean beside its unchanged nominal mean,
    from one run whose histories carry a weight per cross-section replica. It
    supports CPU transport with track-length tallies in every tracking mode,
    with survival biasing, weight windows, coupled and D1S photons (photon
    tallies carry the neutron data's uncertainty) and MPI, and refuses, with
    the reason, GPU, collision-estimator tallies, overlay tallies, mesh tallies
    and per-nuclide bins.
  - `simulate_transmutation(method="independent", data_uncertainty=...)` gives
    a standard deviation on inventories, activity and decay heat. Its one
    transport carries the same replica weights and tallies every reaction rate
    once per replica, so each replica's rates carry the flux's response to its
    own cross sections (self-shielding, shield transmission). The transport
    configurations the weights refuse for `simulate_transport` are refused here
    too, and the coupled method does not take `data_uncertainty`.
- Pathways are reported per product (`get_production_routes`), but there is no
  automatic pathway search across the whole inventory.
