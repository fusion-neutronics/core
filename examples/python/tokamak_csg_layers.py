"""Tokamak first wall and blanket as CSG layers around the plasma.

The plasma boundary is the flux surface rho = 1 of the same TokamakShape that
drives the tokamak source, and each layer is the surface a fixed (true 3-D)
distance outside it. The result is ordinary CSG: surfaces, half-spaces
(``.below`` / ``.above``) and Boolean regions, no CAD and no mesh.
"""

import yamc

yamc.neutron_cross_section_data = "endf-b8.1"

# A EU-DEMO-like plasma (as in tokamak_plasma_source.py). Lengths in cm.
shape = yamc.sources.TokamakShape(
    major_radius=906.0,
    minor_radius=292.258,
    elongation=1.557,
    triangularity=0.270,
    shafranov_factor=0.44789,
)

plasma = yamc.FluxSurface(shape, rho=1.0, name="plasma")
first_wall_front = yamc.OffsetSurface(plasma, 15.0, name="first_wall_front")
first_wall_back = yamc.OffsetSurface(plasma, 17.0, name="first_wall_back")
blanket_back = yamc.OffsetSurface(plasma, 67.0, name="blanket_back")
world = yamc.Sphere(radius=1400.0, boundary="vacuum", name="world")

tungsten = yamc.Material(composition={"W": 1.0}, density=19.3, name="tungsten")
breeder = yamc.Material(composition={"Li6": 0.9, "Li7": 0.1}, density=0.45, name="lithium")

blanket = yamc.Cell(
    region=first_wall_back.above & blanket_back.below, material=breeder, name="blanket"
)
cells = [
    yamc.Cell(region=plasma.below, name="plasma"),
    yamc.Cell(region=plasma.above & first_wall_front.below, name="scrape_off_layer"),
    yamc.Cell(
        region=first_wall_front.above & first_wall_back.below,
        material=tungsten,
        name="first_wall",
    ),
    blanket,
    yamc.Cell(region=blanket_back.above & world.below, name="outside"),
]
geometry = yamc.Geometry(cells)

volume = cells[0].calculate_volume(samples=200_000, seed=1)
print(f"plasma cell volume {volume.volume / 1e6:.1f} +/- {volume.std_dev / 1e6:.1f} m^3, "
      f"shape volume {shape.volume() / 1e6:.1f} m^3")

sources = yamc.sources.tokamak_source(
    shape=shape,
    mode="H",
    ion_density_centre=1.09e20,
    ion_density_peaking_factor=1,
    ion_density_pedestal=1.09e20,
    ion_density_separatrix=3e19,
    ion_temperature_centre=45.9e3,
    ion_temperature_peaking_factor=8.06,
    ion_temperature_beta=6.0,
    ion_temperature_pedestal=6.09e3,
    ion_temperature_separatrix=0.1e3,
    pedestal_radius=0.8 * shape.minor_radius,
)

tbr = yamc.Tally(cells=blanket, scores=[105], name="TBR")
model = yamc.Model(geometry=geometry, source=sources, tallies=[tbr])
geometry.plot(basis="xz", origin=(0.0, 0.0, 0.0), width=(2800.0, 1200.0)).save(
    "tokamak_layers_xz.png"
)

results = model.simulate_transport(total_particles=2000, seed=1)
print(f"TBR {results[tbr].mean[0]:.3f} +/- {results[tbr].standard_deviation[0]:.3f}")
