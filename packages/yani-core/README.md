# yani-core

Yet Another Nuclide Inventory: transmutation and activation without transport. A
material, an irradiation schedule and a neutron spectrum in; inventories,
activities and decay heat out.

This is the compiled distribution, and it provides the `yani` module itself, so
the import name and the distribution name differ (as `pillow` provides `PIL`).
Install [`yani`](https://pypi.org/project/yani/) rather than this: it pins the
matching version of this wheel and is where the documentation lives.

```python
import yani

yani.cross_section_data = "endf-b8.1"

steel = yani.materials.pnnl.material("Steel, Stainless 316", volume=1000.0)
spectrum = yani.NeutronSource(
    energy=yani.sources.Histogram([1e-5, 1e5, 1e6, 1.5e7], [1e12, 1e13, 1e14])
)
schedule = yani.PulseSchedule([
    yani.Pulse(rate=1.11e14, duration=(1, "a"), source=spectrum),
    yani.Cooldown(duration=(1, "d")),
])

results = steel.transmute(schedule=schedule)
final = results.get_final_material(steel.id or 0)
print(final.activity(), "Bq")
print(final.decay_heat(), "W")
print(final.contact_dose(), "Gy/h")
```

## Relationship to yamc

`yamc` is a superset: it ships
everything here plus neutron and photon transport, geometry, tallies and
transport-coupled transmutation. The two are **alternatives**. Installing both
puts two extension modules in one process, so `yamc.Material` and
`yani.Material` are distinct types and the nuclear-data configuration
(`cross_section_data`, the four `transmutation_*` sources) exists twice, once
per package.

Pick `yani` when you have a spectrum already and want none of the transport
stack; pick `yamc` when the spectrum should come from a transport solve.

## What it does not do

- One stepper (`ForwardEulerStepper`, beginning-of-step rates). No
  predictor-corrector.
- No pathway analysis, no sensitivity or uncertainty propagation, no clearance
  indices or ingestion/inhalation dose.
- Cross sections are read from the continuous-energy library and collapsed
  against your spectrum, so this reads transport-format data files even though
  it runs no transport. Only the sections activation needs are read (issue
  #389), which is a small fraction of a library.
