# YAMC

**Y**et **A**nother **M**onte **C**arlo. Geometry, materials and a neutron or
photon source in; fluxes, heating, tritium production, transmuted materials and
shutdown dose rates out.

[Try it online](https://fusion-neutronics.github.io/yamc-online/) 🌐 |
[Documentation](https://fusion-neutronics.github.io/yamc-and-yani/yamc/) 📝 |
[PyPI](https://pypi.org/project/yamc/) 🐍

The browser version needs no install and no account. It is the same Rust
transport engine compiled to WebAssembly, against the same continuous-energy
data. The simulation runs on your machine; nothing is uploaded.

```bash
pip install yamc
```

```python
import yamc

# pick a neutron data library, downloaded on demand
yamc.neutron_cross_section_data = "endf-b8.1"

# make your own or use inbuilt material definitions
steel = yamc.materials.pnnl.material("Steel, Stainless 316")
breeder = yamc.Material(
    composition={"Li": yamc.enriched(2.0, target="Li6", percent=60.0), "Si": 1.0, "O": 4.0},
    density=2.4,
)

# build the geometry from surfaces and cells
first_wall = yamc.Sphere(radius=100.0)
blanket = yamc.Sphere(radius=101.0)
outer = yamc.Sphere(radius=150.0, boundary="vacuum")
plasma_cell = yamc.Cell(region=first_wall.below)
wall_cell = yamc.Cell(region=first_wall.above & blanket.below, material=steel)
blanket_cell = yamc.Cell(region=blanket.above & outer.below, material=breeder)
geometry = yamc.Geometry([plasma_cell, wall_cell, blanket_cell])

# a 14.06 MeV D-T fusion neutron point source
source = yamc.NeutronSource(energy=14.06e6)

# score tritium production in the blanket and heating in each cell
tbr = yamc.Tally(scores=["H3-production"], cells=blanket_cell)
heating = yamc.Tally(scores=["heating"], cells=[wall_cell, blanket_cell])

# run on every CPU core until every tally bin passes its statistical checks
model = yamc.Model(geometry=geometry, source=source, tallies=[tbr, heating])
results = model.simulate_transport()

# every result carries its statistical uncertainty
print("TBR", results[tbr].mean, "+/-", results[tbr].standard_deviation)
print("heating", results[heating].mean, "eV per source neutron")
```

## What it solves

Continuous-energy neutron and photon transport over CSG, CAD or mesh geometry,
on CPU or GPU, with tallies, variance reduction, transport-coupled transmutation
and shutdown dose rates. The solver is Rust; the wheel needs no compiler and no
external toolchain.

## Transmutation without transport

If you already have a spectrum and do not need transport, install
[yani](https://fusion-neutronics.github.io/yamc-and-yani/yani/) instead. It uses the same
materials, schedules and nuclear data, without the geometry.

## Licence

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
