import time
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import tqdm

# Path to HDF5 nuclear data files
NUCLEAR_DATA_DIR = "/home/jon/nuclear_data/endf-b8.0-hdf5/"
NUCLEAR_DATA_DIR_FULL = "/home/jon/nuclear_data/endf-b8.0-hdf5/neutron/"

# Find all HDF5 files in the neutron data directory and sort them alphabetically
h5_files = sorted(glob.glob(os.path.join(NUCLEAR_DATA_DIR_FULL, "*.h5")))

# Prepare to collect average relative differences for all isotopes
isotopes = [os.path.splitext(os.path.basename(f))[0] for f in h5_files]

for isotope in tqdm.tqdm(isotopes):
    fig, ax = plt.subplots(figsize=(10, 6))
    if '_' not in isotope:

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
            radius=35.0,
            boundary='vacuum')
        region1 = sphere1.below
        region2 = sphere1.above & sphere2.below

        # Create material
        material1 = yamc.Material(
            composition={isotope: 1.0},
            density=1.0,
            temperature=294)
        print(f"Reading nuclide data for {isotope}...")
        material1.read_nuclear_data({isotope: f"{NUCLEAR_DATA_DIR_FULL}/{isotope}.h5"})

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
            energy=14.06e6,
            position=(0, 0, 0)
        )

        tally1 = yamc.Tally(cells=cell2, scores=['flux'], name="flux")
        tally2 = yamc.Tally(
            cells=cell2,
            energy_group_structure='VITAMIN-J-175',
            scores=['flux'],
            name="flux_energy_binned",
        )
        energy_bins = np.array(tally2.energy_bins)

        tallies = [tally1, tally2]

        model = yamc.Model(geometry=geometry, tallies=tallies, source=source)

        start_time = time.time()
        results = model.simulate_transport(total_particles=particles * batches, seed=1)

        print(f"Simulation completed in {time.time() - start_time:.2f} seconds.")
        result1 = results[tally1]
        result2 = results[tally2]
        print(f"Total Flux: {result1.mean}")
        print(f"Energy-binned flux has {len(result2.mean)} bins")

        mean_flux = np.array(result2.mean).squeeze()
        std_flux = np.array(result2.standard_deviation).squeeze()

        # Plot lethargy-normalized flux spectrum
        bin_centers = np.sqrt(energy_bins[:-1] * energy_bins[1:])
        lethargy_width = np.log(energy_bins[1:] / energy_bins[:-1])
        norm_flux = mean_flux / lethargy_width
        norm_std = std_flux / lethargy_width
        ax.step(energy_bins[:-1], norm_flux, where='post', label='yamc',
            linestyle='--', linewidth=2.5, alpha=0.9, color='C1')
        ax.errorbar(bin_centers, norm_flux, yerr=norm_std, fmt='s',
                markersize=2, capsize=2, alpha=0.5, color='C1')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(ymin= 1e-5, ymax= None)
    ax.set_ylabel('Flux / lethargy (particles/cm^2/source particle)')
    ax.set_title(f'Energy-dependent Flux Spectrum {isotope} {NUCLEAR_DATA_DIR}')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'flux_spectrum_{isotope}.png', dpi=150)
    print(f"Flux spectrum plot saved as 'flux_spectrum_{isotope}.png'")
    plt.close()

    # Save flux spectrum to CSV file
    csv_filename = f'flux_spectrum_{isotope}.txt'
    bin_centers = np.sqrt(energy_bins[:-1] * energy_bins[1:])
    with open(csv_filename, 'w') as f:
        f.write('yamc,energy\n')
        for i in range(len(bin_centers)):
            f.write(f"{mean_flux[i]:.8e},{bin_centers[i]:.8e}\n")
    print(f"Flux spectrum data saved as '{csv_filename}'")
