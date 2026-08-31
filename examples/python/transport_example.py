import yamc

# yamc.cross_section_data = "tendl-21"
# Create sphere surface with transmission boundary 
sphere1 = yamc.Sphere(
    x0=0.0,
    y0=0.0,
    z0=0.0,
    radius=1.0)
sphere2 = yamc.Sphere(
    x0=0.0,
    y0=0.0,
    z0=0.0,
    radius=2.0,
    boundary='vacuum')
region1 = sphere1.below
region2 = sphere1.above & sphere2.below

material1 = yamc.Material(
    composition={"Li6": 1.0},
    density=10.0,  # Higher density for more absorption
    temperature=294)
material1.read_nuclear_data({"Li6": "tests/Li6.arrow"})

material2 = yamc.Material(
    composition={"Li7": 1.0},
    density=20.0,  # Higher density for more absorption
    temperature=294)
material2.read_nuclear_data({"Li7": "tests/Li7.arrow"})

cell1 = yamc.Cell(
    # name="sphere_cell",
    region=region1,
    material=material1)
cell2 = yamc.Cell(
    name="annular_cell",
    region=region2,
    material=material2)
geometry = yamc.Geometry(cells=[cell1, cell2])

source = yamc.NeutronSource(
    energy=yamc.sources.fusion_neutron_spectrum(20000.0))  # D-T Ballabio spectrum at 20 keV, space = 0,0,0, direction = isotropic
tally1 = yamc.Tally(scores=[101], name="absorption in cell 1", cells=cell1)
tally2 = yamc.Tally(scores=[101], name="absorption in cell 2", cells=cell2)
tally3 = yamc.Tally(scores=[101], name="absorption in whole model")
tally4 = yamc.Tally(scores=[4], name="inelastic in whole model")
tally5 = yamc.Tally(scores=[55], name="inelastic (55) in whole model")

tallies = [tally1, tally2, tally3, tally4, tally5]

# Add fission tally (MT=18)
abs_constituent_mts = [102, 103, 104, 105, 106, 107, 108, 109]
for mt in abs_constituent_mts:
    tallies.append(yamc.Tally(scores=[mt], name=f"absorption constituent MT={mt} in whole model"))

fission_tally = yamc.Tally(scores=[18], name="fission in whole model")
tallies.append(fission_tally)


model = yamc.Model(geometry=geometry, tallies=tallies, source=source)

results = model.simulate_transport(total_particles=1000)

assert results[tally1].mean != results[tally2].mean
assert abs((results[tally1].mean[0] + results[tally2].mean[0]) - results[tally3].mean[0]) < 1e-10 # checking sum of absorptions in cells equals total absorption
assert results[fission_tally].mean[0] == 0.0  # No fission should occur in Li6 or Li7