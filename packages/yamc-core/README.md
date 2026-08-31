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

- Fixed-source transport only. No criticality or eigenvalue calculations.
- One transmutation stepper (`ForwardEulerStepper`, beginning-of-step rates).
  No predictor-corrector.
- No sensitivity or uncertainty propagation, and no pathway analysis.
