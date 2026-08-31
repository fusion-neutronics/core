import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Path to HDF5 nuclear data files
NUCLEAR_DATA_DIR = "/home/jon/nuclear_data/endf-b8.0-hdf5/neutron/"
# Find all HDF5 files in the neutron data directory and sort them alphabetically

h5_files = sorted(glob.glob(os.path.join(NUCLEAR_DATA_DIR, "*.h5")))

# Prepare to collect average relative differences for all isotopes
isotopes = [os.path.splitext(os.path.basename(f))[0] for f in h5_files]
avg_rel_diff_list = []
isotopes = ['C12']
for isotope in isotopes:
    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1], sharex=True)

    # Energy bins: logarithmically spaced from 0.01 eV to 20 MeV
    energy_bins = np.logspace(np.log10(0.01), np.log10(20e6), 20)

    import yamc

    # Create two-cell geometry: inner sphere and outer annular region
    sphere1 = yamc.Sphere(
        x0=0.0,
        y0=0.0,
        z0=0.0,
        radius=1.0)
    sphere2 = yamc.Sphere(
        x0=0.0,
        y0=0.0,
        z0=0.0,
        radius=200.0,
        boundary='vacuum')
    region1 = sphere1.below
    region2 = sphere1.above & sphere2.below

    # Create material
    material1 = yamc.Material(
        composition={isotope: 1.0},
        density=1.0,
        temperature=294)
    print(f"Reading nuclide data for {isotope}...")
    material1.read_nuclear_data({isotope: f"{NUCLEAR_DATA_DIR}{isotope}.h5"})

    # Create cells
    cell1 = yamc.Cell(
        name="inner_sphere",
        region=region1)
    cell2 = yamc.Cell(
        name="outer_annular",
        region=region2,
        material=material1)
    geometry = yamc.Geometry([cell1, cell2])

    particles = 50000
    batches = 20

    source = yamc.NeutronSource(
        energy=14060000.0,
        position=(0, 0, 0)
    )

    model = yamc.Model(geometry=geometry, source=source)

    results = model.simulate_transport(total_particles=particles * batches, seed=1)
    yamc_particles_per_second = results.particles_per_second

print('yamc particles per second', yamc_particles_per_second)
