# Usage guide

After [installing](installation.md) and importing the package, the Python API lets you
build models, run transport, and analyse the results.

```python
import yamc
```

This guide is organised by topic:

- [Nuclear data](nuclear_data.md): configuring and inspecting cross sections
- [Materials](materials.md): composition, density, properties, mixing
- [Geometry](geometry.md): CSG surfaces and cells, and CAD import
- [Sources](sources.md): spatial, energy, and angular distributions
- [Tallies](tallies.md): scores, estimators, filters, and meshes
- [Models](models.md): assembling a model and model function
- [Weight windows](weight_windows.md): variance reduction for deep-penetration and streaming
- [Simulating transport](simulate_transport.md): run a particle transport simulation
- [Simulating transmutation](simulate_transmutation.md): run transmutation simulations
- [Simulating shutdown dose](simulate_shutdown_dose.md): run shutdown (residual) dose simulations
- [Results and post-processing](results.md): statistics, plotting, and export
