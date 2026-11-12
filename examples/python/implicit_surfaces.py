"""A blanket module whose steel structure is a gyroid lattice, as CSG.

The lattice is one ImplicitSurface: the zero set of a formula in x, y and z
(cm). Here the formula is |gyroid| - 0.3, so the steel (``.below``) is a
sheet about the gyroid's minimal surface and the breeder (``.above``) fills
the two channels on either side of it. A periodic formula has no bounding
box of its own, so the module's planes bound it. Behind a tungsten first
wall, the module's corners are rounded by a superellipsoid; the rest is
ordinary CSG, with no CAD and no mesh.
"""

import yamc

yamc.neutron_cross_section_data = "endf-b8.1"

# A sheet gyroid of period 4 cm.
lattice = yamc.ImplicitSurface(
    "abs(sin(pi / 2 * x) * cos(pi / 2 * y) + sin(pi / 2 * y) * cos(pi / 2 * z)"
    " + sin(pi / 2 * z) * cos(pi / 2 * x)) - 0.3",
    name="lattice",
)
# The module, 40 cm deep and 40 cm across with rounded edges, from x = 1 to
# x = 41; it has a proved bounding box.
module = yamc.ImplicitSurface(
    "((x - 21) / 20)^8 + (y / 20)^8 + (z / 20)^8 - 1", name="module"
)
wall_front = yamc.Plane("x", 0.0, name="wall_front")
wall_back = yamc.Plane("x", 1.0, name="wall_back")
world = yamc.Sphere(radius=100.0, boundary="vacuum", name="world")
print(f"the module's proved box: {module.bounding_box_inside()}")

tungsten = yamc.Material(composition={"W": 1.0}, density=19.3, name="tungsten")
steel = yamc.Material(composition={"Fe": 1.0}, density=7.8, name="steel")
breeder = yamc.Material(
    composition={"Li6": 0.9, "Li7": 0.1}, density=0.45, name="lithium"
)

span = yamc.Plane("y", -20.0).above & yamc.Plane("y", 20.0).below
span = span & yamc.Plane("z", -20.0).above & yamc.Plane("z", 20.0).below
first_wall = wall_front.above & wall_back.below & span
inside = module.below & wall_back.above
structure = yamc.Cell(region=inside & lattice.below, material=steel, name="structure")
channels = yamc.Cell(region=inside & lattice.above, material=breeder, name="channels")
cells = [
    yamc.Cell(region=first_wall, material=tungsten, name="first_wall"),
    structure,
    channels,
    yamc.Cell(region=world.below & ~first_wall & ~inside, name="outside"),
]
geometry = yamc.Geometry(cells)

steel_volume = structure.calculate_volume(samples=200_000, seed=1)
print(f"steel {steel_volume.volume:.0f} +/- {steel_volume.std_dev:.0f} cm^3")

source = yamc.NeutronSource(
    position=(-20.0, 0.0, 0.0), energy=yamc.sources.Discrete([14.06e6], [1.0])
)
tbr = yamc.Tally(cells=channels, scores=[105], name="TBR")
model = yamc.Model(geometry=geometry, source=source, tallies=[tbr])
geometry.plot(basis="xy", origin=(20.0, 0.0, 0.0), width=(50.0, 50.0)).save(
    "gyroid_blanket_xy.png"
)

results = model.simulate_transport(total_particles=2000, seed=1)
print(f"TBR {results[tbr].mean[0]:.4f} +/- {results[tbr].standard_deviation[0]:.4f}")
