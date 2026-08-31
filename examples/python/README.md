# Python examples

Runnable scripts that exercise the `yamc` Python API: building geometries and
materials, defining sources and tallies, running transport, transmutation, photon
transport, and plotting/exporting the results.

Each script is self-contained. Run one with, for example:

```bash
python examples/python/tbr.py
```

Scripts that read nuclear data will download the required Arrow data files into
`~/.cache/yamc/` on first use (or read from the `tests/` paths some scripts
configure explicitly). Scripts marked **(mesh)** below build geometry through
the CadQuery -> `cad_to_yamc` -> Arrow pipeline and need those optional
dependencies installed.

A small subset is run automatically in CI and by the pre-commit hook as a
smoke test (see [Exercised by CI / pre-commit](#exercised-by-ci--pre-commit)).

## Getting started

- [`usage_in_python.py`](usage_in_python.py) -- minimal end-to-end script: build a CSG geometry, run transport.
- [`usage_in_python_cube.py`](usage_in_python_cube.py) -- same, for a simple cube geometry.
- [`transport_example.py`](transport_example.py) -- basic transport through a multi-region sphere geometry.

## Materials & nuclear data

- [`nuclide_example.py`](nuclide_example.py) -- inspect `Nuclide` properties and reaction product data.

## Geometry (CSG / mesh)

- [`geometry_cube.py`](geometry_cube.py) -- build a cube from planes (CSG).
- [`two_cubes_with_gap.py`](two_cubes_with_gap.py) -- two separated cubes; transport through the implicit complement / graveyard.
- [`lost_particle.py`](lost_particle.py) -- how lost particles (geometry gaps) are reported and debugged.
- [`surface_mesh_simulation.py`](surface_mesh_simulation.py) -- **(mesh)** surface-mesh geometry from CadQuery for transport.
- [`yamt_mixed_mesh_adjacent_cubes.py`](yamt_mixed_mesh_adjacent_cubes.py) -- **(mesh)** mixed tet + surface-only adjacent cubes.
- [`yamt_mixed_mesh_hollow_spheres.py`](yamt_mixed_mesh_hollow_spheres.py) -- **(mesh)** mixed tet + surface-only nested hollow spheres.

## Sources

- [`nwl_torus_ring_source.py`](nwl_torus_ring_source.py) -- toroidal ring source for neutron wall loading on a first wall.

## Tallies & results

- [`sphere_li6_tritium.py`](sphere_li6_tritium.py) -- Li-6 (n,t) tritium-production rate; the simplest physics-relevant tally.
- [`tbr.py`](tbr.py) -- tritium breeding ratio with a per-nuclide breakdown.
- [`flux.py`](flux.py) -- energy-resolved neutron flux spectra across elements.
- [`mesh_tally_overlayed_response.py`](mesh_tally_overlayed_response.py) -- virtual `response=` overlay tally (a dose map for a nuclide or material not in the geometry).
- [`tet_mesh_tally.py`](tet_mesh_tally.py) -- **(mesh)** per-tetrahedron flux scoring on a tet mesh.
- [`get_particle_tracks.py`](get_particle_tracks.py) -- enable particle tracking and read back track data.

## Transmutation / activation

- [`transmutation_coupled_csg_vs_mesh.py`](transmutation_coupled_csg_vs_mesh.py) -- **(mesh)** coupled transmutation compared on CSG vs CAD-mesh geometry.
- [`transmutation_independent_csg_vs_mesh.py`](transmutation_independent_csg_vs_mesh.py) -- **(mesh)** independent (transport-once) transmutation on CSG vs CAD-mesh geometry.
- [`material_transmute_with_flux_from_tally.py`](material_transmute_with_flux_from_tally.py) -- standalone transmutation driven by a multigroup flux extracted from a tally.

## Photon transport & heating

- [`flux_photon.py`](flux_photon.py) -- photon flux spectra for a photon source across elements.
- [`flux_coupled_photon.py`](flux_coupled_photon.py) -- coupled neutron-photon flux spectra (14 MeV neutron source).
- [`photon_heating.py`](photon_heating.py) -- photon heating from a 1 MeV photon source in an iron cube.

## Plotting / visualization

- [`manual_slice_plot_model.py`](manual_slice_plot_model.py) -- sample a model on a 2D grid and plot cells/materials/source with matplotlib.
- [`manual_slice_outline_geometry_with_mesh_tally.py`](manual_slice_outline_geometry_with_mesh_tally.py) -- mesh-tally heatmap with a geometry-outline overlay.
- [`plot_threads_auto_sweep.py`](plot_threads_auto_sweep.py) -- plot a particles/sec vs thread-count sweep (consumes data from the Rust `threads_auto_sweep` example).

## Performance / benchmarks

- [`speed_comparison.py`](speed_comparison.py) -- particles/sec comparison against a reference code (no tallies).
- [`tally_speed_comparision.py`](tally_speed_comparision.py) -- particles/sec comparison with tallies enabled.
- [`benchmark_csg_vs_mesh.py`](benchmark_csg_vs_mesh.py) -- **(mesh)** CSG vs mesh geometry transport benchmark.
- [`simple_tokamak.py`](simple_tokamak.py) -- larger tokamak model (configurable particle count and tracking mode).

## Exercised by CI / pre-commit

These scripts are run automatically as a smoke test, so they are kept fast and
data-light. Keep them runnable when changing the public API.

- Python CI (`.github/workflows/ci-python.yml`, "Run example scripts"):
  [`nuclide_example.py`](nuclide_example.py), [`tbr.py`](tbr.py),
  [`usage_in_python.py`](usage_in_python.py),
  [`usage_in_python_cube.py`](usage_in_python_cube.py),
  [`get_particle_tracks.py`](get_particle_tracks.py).
  The MPI test job additionally runs [`tbr.py`](tbr.py) under `mpirun`.
  ([`simple_tokamak.py`](simple_tokamak.py) is listed but commented out -- it
  runs 1M particles and OOMs the CI runner.)
- Pre-commit hook (`.pre-commit-config.yaml`, `python-examples`):
  [`nuclide_example.py`](nuclide_example.py), [`tbr.py`](tbr.py),
  [`usage_in_python.py`](usage_in_python.py),
  [`usage_in_python_cube.py`](usage_in_python_cube.py),
  [`simple_tokamak.py`](simple_tokamak.py),
  [`get_particle_tracks.py`](get_particle_tracks.py).
