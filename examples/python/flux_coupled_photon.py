"""
Coupled neutron-photon flux spectra.

Loops over all elements with available photon and neutron data,
running a 14 MeV neutron source simulation in a sphere of each element
with photon transport enabled. Tallies the energy-resolved photon flux
from secondary gammas produced by neutron interactions.
"""
import glob
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import tqdm

import yamc

# --- Configuration ---
NUCLEAR_DATA_DIR = "/home/jon/nuclear_data/endf-b8.0-hdf5/"
NEUTRON_DATA_DIR = os.path.join(NUCLEAR_DATA_DIR, "neutron")
PHOTON_DATA_DIR = os.path.join(NUCLEAR_DATA_DIR, "photon")

# Most common (highest natural abundance) isotope per element.
COMMON_ISOTOPES = {
    'H': 'H1', 'He': 'He4', 'Li': 'Li7', 'Be': 'Be9', 'B': 'B11',
    'C': 'C12', 'N': 'N14', 'O': 'O16', 'F': 'F19', 'Ne': 'Ne20',
    'Na': 'Na23', 'Mg': 'Mg24', 'Al': 'Al27', 'Si': 'Si28', 'P': 'P31',
    'S': 'S32', 'Cl': 'Cl35', 'Ar': 'Ar40', 'K': 'K39', 'Ca': 'Ca40',
    'Sc': 'Sc45', 'Ti': 'Ti48', 'V': 'V51', 'Cr': 'Cr52', 'Mn': 'Mn55',
    'Fe': 'Fe56', 'Co': 'Co59', 'Ni': 'Ni58', 'Cu': 'Cu63', 'Zn': 'Zn64',
    'Ga': 'Ga69', 'Ge': 'Ge74', 'As': 'As75', 'Se': 'Se80', 'Br': 'Br79',
    'Kr': 'Kr84', 'Rb': 'Rb85', 'Sr': 'Sr88', 'Y': 'Y89', 'Zr': 'Zr90',
    'Nb': 'Nb93', 'Mo': 'Mo98', 'Ru': 'Ru102', 'Rh': 'Rh103',
    'Pd': 'Pd106', 'Ag': 'Ag107', 'Cd': 'Cd114', 'In': 'In115', 'Sn': 'Sn120',
    'Sb': 'Sb121', 'Te': 'Te130', 'I': 'I127', 'Xe': 'Xe132', 'Cs': 'Cs133',
    'Ba': 'Ba138', 'La': 'La139', 'Ce': 'Ce140', 'Pr': 'Pr141', 'Nd': 'Nd144',
    'Sm': 'Sm152', 'Eu': 'Eu153', 'Gd': 'Gd158', 'Tb': 'Tb159',
    'Dy': 'Dy164', 'Ho': 'Ho165', 'Er': 'Er166', 'Tm': 'Tm169', 'Yb': 'Yb174',
    'Lu': 'Lu175', 'Hf': 'Hf180', 'Ta': 'Ta181', 'W': 'W184', 'Re': 'Re187',
    'Os': 'Os192', 'Ir': 'Ir193', 'Pt': 'Pt195', 'Au': 'Au197', 'Hg': 'Hg202',
    'Tl': 'Tl205', 'Pb': 'Pb208', 'Bi': 'Bi209', 'Th': 'Th232', 'U': 'U238',
}


def find_isotope_for_element(element, neutron_dir):
    """Find a neutron data file for the given element symbol."""
    if element in COMMON_ISOTOPES:
        isotope = COMMON_ISOTOPES[element]
        path = os.path.join(neutron_dir, f"{isotope}.h5")
        if os.path.exists(path):
            return isotope, path

    pattern = re.compile(rf'^{re.escape(element)}\d+\.h5$')
    candidates = []
    for fname in os.listdir(neutron_dir):
        if pattern.match(fname):
            candidates.append(fname)
    if candidates:
        candidates.sort()
        isotope = candidates[0].replace('.h5', '')
        return isotope, os.path.join(neutron_dir, candidates[0])

    return None, None


def find_elements():
    """Find all elements with both photon and neutron data available."""
    photon_files = sorted(glob.glob(os.path.join(PHOTON_DATA_DIR, "*.h5")))
    elements = []
    for pf in photon_files:
        element = os.path.splitext(os.path.basename(pf))[0]
        isotope, neutron_path = find_isotope_for_element(element, NEUTRON_DATA_DIR)
        if isotope is not None:
            elements.append((element, isotope, neutron_path, pf))
    return elements


# --- Main loop ---
elements = find_elements()
print(f"Found {len(elements)} elements with photon + neutron data")

for element, isotope, neutron_path, photon_path in tqdm.tqdm(elements):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Geometry: point source in void center, material shell
    sphere1 = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=1.0)
    sphere2 = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=35.0,
                         boundary='vacuum')

    material = yamc.Material(
        composition={isotope: 1.0},
        density=1.0,
        temperature=294)
    material.read_nuclear_data(
        {isotope: neutron_path},
        photon_data={element: photon_path})

    cell1 = yamc.Cell(name="void_center", region=sphere1.below)
    cell2 = yamc.Cell(name="material_shell",
                    region=sphere1.above & sphere2.below, material=material)
    geometry = yamc.Geometry([cell1, cell2])

    # Source: 14 MeV isotropic neutrons at origin
    source = yamc.NeutronSource(
        position=(0, 0, 0),
        energy=14.06e6)

    # Tallies: energy-resolved photon flux in the material cell
    tally = yamc.Tally(
        name="photon_spectrum",
        cells=cell2,
        particle="photon",
        energy_group_structure='CCFE-24-PHOTON',
        scores=["flux"],
    )
    energy_bins = np.array(tally.energy_bins)

    model = yamc.Model(geometry=geometry, tallies=[tally], source=source, transport_secondary_photons=True)

    t_start = time.time()
    results = model.simulate_transport(total_particles=1000000, seed=1)
    elapsed = time.time() - t_start

    tally_result = results[tally]
    mean_flux = np.array(tally_result.mean).squeeze()
    std_flux = np.array(tally_result.standard_deviation).squeeze()

    print(f"({elapsed:.2f}s): {element} ({isotope}) "
          f"total photon flux = {np.sum(mean_flux):.6e}")

    # --- Plot ---
    bin_centers = np.sqrt(energy_bins[:-1] * energy_bins[1:])
    ax.step(energy_bins[:-1], mean_flux, where='post',
            label='yamc', linestyle='--', linewidth=2, color='C1')
    ax.errorbar(bin_centers, mean_flux, yerr=std_flux,
                fmt='s', markersize=3, capsize=2, alpha=0.5, color='C1')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(ymin=1e-5)
    ax.set_ylabel('Photon flux (cm/source)')
    ax.set_title(f'Coupled n-gamma: 14 MeV neutrons in {element} ({isotope})')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'flux_coupled_photon_{element}.png', dpi=150)
    plt.close()
