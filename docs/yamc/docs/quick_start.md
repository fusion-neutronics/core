# Quick start guide

A minimal simulation end to end: configure data, build a material, geometry and source, add a
tally, run, and read the result. The example computes tritium production (the basis of the
tritium breeding ratio, TBR) in a lithium sphere.

## Import YAMC

```python
import yamc
```

## Configure the nuclear data

This example transports neutrons only, so it needs neutron data. A single
library keyword downloads the required data automatically:

```python
yamc.neutron_cross_section_data = "endf-b8.1"
```

A run that transports photons also sets ``yamc.photon_cross_section_data``; see
[Neutron and photon cross sections](nuclear_data.md#neutron-and-photon-cross-sections).

Six libraries are available, and they differ in content as well as name; see
[Nuclear data libraries](nuclear_data_libraries.md).

## Create a material

Lithium breeds tritium, so it serves as the breeder material (`"Li"` expands to natural Li6 + Li7):

```python
mat_lithium = yamc.Material(
    composition={"Li": 1.0},
    density=0.534,
    temperature=294,
)
```

## Build the geometry

A lithium sphere bounded by a vacuum surface:

```python
surface_sphere = yamc.Sphere(radius=100, boundary="vacuum")
cell = yamc.Cell(region=surface_sphere.below, material=mat_lithium)
geometry = yamc.Geometry(cells=[cell])
```

## Build the source

A single float is a monoenergetic source (here the 14.06 MeV D-T neutron):

```python
source = yamc.NeutronSource(energy=14.06e6)
```

## Add a tally

Score H3 (tritium) production in the lithium sphere. With no spatial filter the tally scores
over the whole geometry, which here is one cell. Per source neutron, this is the TBR:

```python
tally = yamc.Tally(scores=["H3-production"])
```

With more than one cell, `cells=cell` bins the tally per cell instead; see
[Spatial binning](tallies.md#spatial-binning).

## Run the simulation

Pass the tally to the model and run the transport:

```python
model = yamc.Model(
    geometry=geometry,
    source=source,
    tallies=[tally],
)
results = model.simulate_transport()  # runs until the statistical checks pass
```

## Read the result

`results[tally]` is a labelled result. Its `.mean` is the H3-production rate per source neutron
(the TBR for this geometry) and `.standard_deviation` is its statistical uncertainty. Both are
lists with one entry per score:

```python
result = results[tally]
print(f"TBR (H3 production per source neutron): {result.mean[0]} +/- {result.standard_deviation[0]}")
```

For a pure-lithium sphere this prints a TBR of roughly:

```text
TBR (H3 production per source neutron): 1.2169485390409323 +/- 0.002059927598397041
```
