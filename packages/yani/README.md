# YANI

**Y**et **A**nother **N**uclide **I**nventory. A material, an irradiation
schedule and a neutron spectrum in; inventories, activities, decay heat,
contact dose, decay photon spectra and displacement damage (dpa) out. One call
can solve many materials at once, such as the cells of a mesh, and the result
can be checked against regulatory clearance and disposal limits.

[Try it online](https://fusion-neutronics.github.io/yani-online/) 🌐 |
[Documentation](https://fusion-neutronics.github.io/yamc-and-yani/yani/) 📝 |
[Validation](https://fusion-neutronics.github.io/yani-verification-and-validation/) 📊 |
[PyPI](https://pypi.org/project/yani/) 🐍

The browser version needs no install and no account. It is the same Rust solver
compiled to WebAssembly, against the same continuous-energy data, so it answers
what the wheel answers rather than a reduced version of it. The solve runs on
your machine; nothing is uploaded.

```bash
pip install yani
```

```python
import yani

# flexible use of different libraries
yani.neutron_cross_section_data = "tendl-2025"
yani.transmutation_reactions = "tendl-2025"
yani.transmutation_branch_ratios = "tendl-2025"
yani.transmutation_decay_data = "endf-b8.1"
yani.transmutation_fission_yields = "endf-b8.1"

# make your own or use inbuilt material definitions
steel = yani.materials.pnnl.material(key="Steel, Stainless 316", volume=1000.0)

# define the neutron spectrum
spectrum = yani.NeutronSource(
    energy=yani.sources.Histogram(
        boundaries=[1e-5, 1e5, 1e6, 1.5e7], probabilities=[1e12, 1e13, 1e14]
    )
)

# define the irradiation and cooling schedule
schedule = yani.PulseSchedule(steps=[
    yani.Pulse(rate=1.11e14, duration=(1, "a"), source=spectrum),
    yani.Cooldown(duration=(1, "d")),
])

# perform the simulation
results = steel.transmute(schedule=schedule)

# access the last material in the time steps (results are keyed by material id)
final = results.step_materials(material_id=steel.id)[-1]

# convenient access to useful properties of the material
print(final.activity(), "Bq")
print(final.decay_heat(), "W")
print(final.contact_dose(), "Gy/h")
```

## What it solves

The nuclide inventory under irradiation is the Bateman equations with reaction
terms, `dN/dt = A N`, solved as a matrix exponential `N(t) = exp(A t) N(0)` by
CRAM at order 48. The solver is Rust; the wheel needs no compiler and no
external toolchain.

## Getting the spectrum

If you do not have a neutron spectrum for the material yet,
[YAMC](https://fusion-neutronics.github.io/yamc-and-yani/yamc/) computes one by Monte Carlo
transport through your geometry. It uses the same materials, schedules and
nuclear data, and can run the transmutation coupled to the transport.

## Licence

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
