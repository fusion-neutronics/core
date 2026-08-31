#!/usr/bin/env python3
"""Regenerate the OpenMC reference data for the yamc regression suite.

Reconstructs the four reference types consumed by the regression tests
(``reference_data/<type>_<nuclide>.json``) against ENDF/B-VIII.1:

* ``neutron``       -- neutron-only broomstick: scalar scores + VITAMIN-J-175 flux spectrum.
* ``coupled``       -- coupled n->photon broomstick: neutron + photon scalars + spectra.
                       Photon heating is summed over the photon, electron and positron
                       particle filters with ``electron_treatment='led'`` so it matches
                       yamc's local-deposition photon-heating convention
                       (``E_in - E_out - E_secondary_photons``).
* ``decay_photons`` -- D1S (Direct-1-Step) decay-photon spectra per cooling step, via
                       OpenMC's native ``openmc.deplete.d1s``.
* ``transmutation`` -- 0-D point depletion via ``openmc.Material.deplete``.

All constants come from ``_config.py`` (the single source of truth shared with
``conftest.py``). The model geometry/source mirror the yamc regression run
helpers (broomstick: ZCylinder r=CYLINDER_RADIUS capped at +/-CYLINDER_HALF_HEIGHT,
14.06 MeV isotropic point neutron source, pure nuclide at 1 g/cc, 294 K).

OpenMC ``tally.mean`` is already normalised per source particle, so values are
stored verbatim (no division by the particle count).

Usage:
    OPENMC_CROSS_SECTIONS=.../cross_sections.xml \
      python generate_reference_data.py --types neutron coupled decay_photons transmutation \
                                         [--nuclides Fe54 Li6 ...]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config as C  # noqa: E402

import openmc  # noqa: E402
import openmc.deplete  # noqa: E402
from openmc.deplete import d1s  # noqa: E402

LIBRARY = "endfb-8.1"
NUCLEAR_DATA_DIR = "/home/jon/nuclear_data/endf-b8.1-hdf5/endfb-viii.1-hdf5"
XS = os.path.join(NUCLEAR_DATA_DIR, "cross_sections.xml")
REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_data")

# Generator neutron scalar scores = conftest list + 'heating-local' (KERMA, a
# reference-code-only score yamc does not expose; written but not compared).
NEUTRON_SCALAR_SCORES = [
    "flux", "heating", "heating-local", "total", "absorption",
    "H1-production", "H2-production", "H3-production", "He3-production", "He4-production",
]

# D1S / transmutation schedules (seconds). yamc source_rates are multipliers on
# the absolute flux; OpenMC takes absolute n/s.
D1S_TIMESTEPS = C.DECAY_PHOTON_TIMESTEPS
D1S_SOURCE_RATES = C.DECAY_PHOTON_SOURCE_RATES
TRANSMUTE_TOTAL_FLUX = float(sum(C.TRANSMUTE_MULTIGROUP_FLUX))
TRANSMUTE_OPENMC_SOURCE_RATES = [sr * TRANSMUTE_TOTAL_FLUX for sr in C.TRANSMUTE_SOURCE_RATES]


def find_chain():
    cands = [
        "/home/jon/yamc-org/openmc_data/src/openmc_data/depletion/chain_endf_b8.1.xml",
    ]
    cands += sorted(glob.glob(
        "/home/jon/*/.venv/lib/python*/site-packages/openmc_data/depletion/chain_endf_b8.1.xml"))
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("chain_endf_b8.1.xml not found")


CHAIN = find_chain()
openmc.config["cross_sections"] = XS


# ---------------------------------------------------------------- model pieces
def _broomstick_cell(nuclide):
    m = openmc.Material()
    m.add_nuclide(nuclide, 1.0, percent_type="ao")
    m.set_density("g/cm3", 1.0)
    m.temperature = 294.0
    cyl = openmc.ZCylinder(r=C.CYLINDER_RADIUS, boundary_type="vacuum")
    zb = openmc.ZPlane(z0=-C.CYLINDER_HALF_HEIGHT, boundary_type="vacuum")
    zt = openmc.ZPlane(z0=C.CYLINDER_HALF_HEIGHT, boundary_type="vacuum")
    cell = openmc.Cell(name="broomstick", region=-cyl & +zb & -zt, fill=m)
    return m, cell


def _neutron_source():
    return openmc.IndependentSource(
        space=openmc.stats.Point((0, 0, 0)),
        angle=openmc.stats.Isotropic(),
        energy=openmc.stats.Discrete([C.NEUTRON_SOURCE_ENERGY], [1.0]),
        particle="neutron",
    )


def _edges(group_structure):
    ef = openmc.EnergyFilter.from_group_structure(group_structure)
    b = ef.bins
    return ef, np.concatenate([[b[0, 0]], b[:, 1]])


def _scalar(tally):
    return float(tally.mean.ravel()[0]), float(tally.std_dev.ravel()[0])


def _tally(name, filters, scores):
    t = openmc.Tally(name=name)
    t.filters = filters
    t.scores = scores
    return t


# ----------------------------------------------------------------- generators
def gen_neutron(nuclide, element=None):
    openmc.reset_auto_ids()
    mat, cell = _broomstick_cell(nuclide)
    geom = openmc.Geometry([cell])
    s = openmc.Settings()
    s.run_mode = "fixed source"
    s.particles = C.PARTICLES
    s.batches = C.BATCHES
    s.seed = C.SEED
    s.source = _neutron_source()
    s.photon_transport = False
    s.output = {"tallies": False, "summary": False}

    cf = openmc.CellFilter([cell])
    pf = openmc.ParticleFilter(["neutron"])
    ef, edges = _edges(C.NEUTRON_GROUP_STRUCTURE)
    tals, scal = [], {}
    for sc in NEUTRON_SCALAR_SCORES:
        t = openmc.Tally(name=f"n_{sc}")
        t.filters = [cf, pf]
        t.scores = [sc]
        tals.append(t)
        scal[sc] = t
    spec = openmc.Tally(name="n_spec")
    spec.filters = [cf, ef, pf]
    spec.scores = ["flux"]
    tals.append(spec)

    model = openmc.Model(geom, openmc.Materials([mat]), s, openmc.Tallies(tals))
    sp_path = model.run(output=False)
    out = {"nuclide": nuclide, "particles": C.PARTICLES, "batches": C.BATCHES, "seed": C.SEED,
           "scalar_results": {}, "scalar_stds": {}}
    with openmc.StatePoint(sp_path) as sp:
        for sc, t in scal.items():
            tt = sp.get_tally(name=t.name)
            m, e = _scalar(tt)
            out["scalar_results"][sc] = m
            out["scalar_stds"][sc] = e
        sp_t = sp.get_tally(name="n_spec")
        out["spectrum_mean"] = [float(x) for x in sp_t.mean.ravel()]
        out["spectrum_std"] = [float(x) for x in sp_t.std_dev.ravel()]
    out["energy_bins"] = [float(x) for x in edges]
    out["library"] = LIBRARY
    return out


# Photon energy deposition depends on OpenMC's electron treatment. yamc tallies
# both conventions, so reference each. The regression test compares yamc against
# PRIMARY_ELECTRON_TREATMENT; the other variant is stored (labelled) so the
# convention is explicit and the "with vs without TTB" sensitivity is visible.
#   * "ttb" (thick-target bremsstrahlung, OpenMC default) -- secondary electron
#     energy partly re-radiated as transported bremsstrahlung photons. Matches
#     yamc's coupled photon physics across all photon scores (within ~3%).
#   * "led" (local energy deposition) -- electron energy deposited locally, no
#     bremsstrahlung. Drops the low-energy photon population, so photoelectric /
#     coherent reaction rates fall well below yamc (kept here for reference).
ELECTRON_TREATMENTS = ("ttb", "led")
PRIMARY_ELECTRON_TREATMENT = "ttb"


def _run_coupled(nuclide, electron_treatment):
    """One coupled n->photon broomstick run; returns (neutron dict, photon dict)."""
    openmc.reset_auto_ids()
    mat, cell = _broomstick_cell(nuclide)
    geom = openmc.Geometry([cell])
    s = openmc.Settings()
    s.run_mode = "fixed source"
    s.particles = C.PARTICLES
    s.batches = C.BATCHES
    s.seed = C.SEED
    s.source = _neutron_source()
    s.photon_transport = True
    s.electron_treatment = electron_treatment
    s.output = {"tallies": False, "summary": False}

    cf = openmc.CellFilter([cell])
    pf_n = openmc.ParticleFilter(["neutron"])
    pf_p = openmc.ParticleFilter(["photon"])
    pf_e = openmc.ParticleFilter(["electron"])
    pf_pos = openmc.ParticleFilter(["positron"])
    ef_n, edges_n = _edges(C.NEUTRON_GROUP_STRUCTURE)
    ef_p, edges_p = _edges(C.PHOTON_GROUP_STRUCTURE)

    tals, nscal, pscal = [], {}, {}
    for sc in NEUTRON_SCALAR_SCORES:
        t = _tally(f"n_{sc}", [cf, pf_n], [sc])
        tals.append(t)
        nscal[sc] = t
    for sc in C.PHOTON_SCALAR_SCORES:
        if sc == "heating":
            continue  # summed below over photon+electron+positron
        t = _tally(f"p_{sc}", [cf, pf_p], [sc])
        tals.append(t)
        pscal[sc] = t
    # Photon heating = energy deposited by the photon cascade (photon + the
    # electrons/positrons it liberates). OpenMC scores heating by depositing
    # particle, and a combined ParticleFilter returns only the photon piece, so
    # tally the three separately and sum.
    tals.append(_tally("p_heating_photon", [cf, pf_p], ["heating"]))
    tals.append(_tally("p_heating_electron", [cf, pf_e], ["heating"]))
    tals.append(_tally("p_heating_positron", [cf, pf_pos], ["heating"]))
    tals.append(_tally("n_spec", [cf, ef_n, pf_n], ["flux"]))
    tals.append(_tally("p_spec", [cf, ef_p, pf_p], ["flux"]))

    model = openmc.Model(geom, openmc.Materials([mat]), s, openmc.Tallies(tals))
    sp_path = model.run(output=False)
    neutron = {"scalars": {}, "stds": {}}
    photon = {"scalars": {}, "stds": {}}
    with openmc.StatePoint(sp_path) as sp:
        for sc, t in nscal.items():
            m, e = _scalar(sp.get_tally(name=t.name))
            neutron["scalars"][sc] = m
            neutron["stds"][sc] = e
        for sc, t in pscal.items():
            m, e = _scalar(sp.get_tally(name=t.name))
            photon["scalars"][sc] = m
            photon["stds"][sc] = e
        hm_p, hs_p = _scalar(sp.get_tally(name="p_heating_photon"))
        hm_e, hs_e = _scalar(sp.get_tally(name="p_heating_electron"))
        hm_pos, hs_pos = _scalar(sp.get_tally(name="p_heating_positron"))
        photon["scalars"]["heating"] = hm_p + hm_e + hm_pos
        photon["stds"]["heating"] = float(np.sqrt(hs_p**2 + hs_e**2 + hs_pos**2))
        nt = sp.get_tally(name="n_spec")
        pt = sp.get_tally(name="p_spec")
        neutron["spectrum_mean"] = [float(x) for x in nt.mean.ravel()]
        neutron["spectrum_std"] = [float(x) for x in nt.std_dev.ravel()]
        neutron["energy_bins"] = [float(x) for x in edges_n]
        photon["spectrum_mean"] = [float(x) for x in pt.mean.ravel()]
        photon["spectrum_std"] = [float(x) for x in pt.std_dev.ravel()]
        photon["energy_bins"] = [float(x) for x in edges_p]
    return neutron, photon


def gen_coupled(nuclide, element):
    out = {"nuclide": nuclide, "element": element, "particles": C.PARTICLES,
           "batches": C.BATCHES, "seed": C.SEED,
           "electron_treatment": PRIMARY_ELECTRON_TREATMENT,
           "photon": {}}
    for treat in ELECTRON_TREATMENTS:
        neutron, photon = _run_coupled(nuclide, treat)
        out["photon"][treat] = photon
        if treat == PRIMARY_ELECTRON_TREATMENT:
            # Neutron transport is electron-treatment independent; take it from
            # the primary run.
            out["n_scalars"] = neutron["scalars"]
            out["n_stds"] = neutron["stds"]
            out["n_spectrum_mean"] = neutron["spectrum_mean"]
            out["n_spectrum_std"] = neutron["spectrum_std"]
            out["n_energy_bins"] = neutron["energy_bins"]
            out["p_energy_bins"] = photon["energy_bins"]
    out["library"] = LIBRARY
    return out


def gen_decay_photons(nuclide, element):
    openmc.reset_auto_ids()
    openmc.config["chain_file"] = CHAIN
    mat, cell = _broomstick_cell(nuclide)
    geom = openmc.Geometry([cell])
    s = openmc.Settings()
    s.run_mode = "fixed source"
    s.particles = C.PARTICLES
    s.batches = C.BATCHES
    s.seed = C.SEED
    s.source = _neutron_source()
    s.photon_transport = True
    s.use_decay_photons = True
    s.output = {"tallies": False, "summary": False}

    cf = openmc.CellFilter([cell])
    pf_p = openmc.ParticleFilter(["photon"])
    ef, edges = _edges(C.PHOTON_GROUP_STRUCTURE)
    tally = openmc.Tally(name="d1s_spectrum")
    tally.filters = [cf, pf_p, ef]
    tally.scores = ["flux"]
    model = openmc.Model(geom, openmc.Materials([mat]), s, openmc.Tallies([tally]))
    d1s.prepare_tallies(model=model)
    radionuclides = d1s.get_radionuclides(model)

    sp_path = model.run(output=False)
    with openmc.StatePoint(sp_path) as sp:
        tally_result = sp.get_tally(name="d1s_spectrum")
        # Load data into memory so the tally is usable after the file closes.
        tally_result.mean
        tally_result.std_dev
    # Native D1S time-correction: per-step TCF applied + summed over parents.
    tcf = d1s.time_correction_factors(
        nuclides=radionuclides, timesteps=D1S_TIMESTEPS,
        source_rates=D1S_SOURCE_RATES, timestep_units="s",
    )
    spectra, spectra_std = [], []
    for step_idx in range(len(D1S_TIMESTEPS)):
        corrected = d1s.apply_time_correction(
            tally=tally_result, time_correction_factors=tcf,
            index=step_idx + 1, sum_nuclides=True,
        )
        spectra.append([float(x) for x in corrected.mean.flatten()])
        spectra_std.append([float(x) for x in corrected.std_dev.flatten()])
    return {
        "nuclide": nuclide, "element": element,
        "radionuclides": list(radionuclides),
        "spectra": spectra, "spectra_std": spectra_std,
        "particles": C.PARTICLES, "batches": C.BATCHES, "seed": C.SEED,
        "library": LIBRARY,
    }


def gen_transmutation(nuclide, element=None):
    openmc.reset_auto_ids()
    full = openmc.deplete.Chain.from_xml(CHAIN)
    reduced = full.reduce([nuclide], level=20)
    mat = openmc.Material(name=nuclide)
    mat.add_nuclide(nuclide, 1.0, percent_type="ao")
    mat.set_density("g/cm3", 1.0)
    mat.volume = 1.0
    mat.temperature = 294.0
    mat.depletable = True
    results = mat.deplete(
        multigroup_flux=C.TRANSMUTE_MULTIGROUP_FLUX,
        energy_group_structure=C.TRANSMUTE_ENERGY_GROUPS,
        timesteps=C.TRANSMUTE_TIMESTEPS,
        source_rates=TRANSMUTE_OPENMC_SOURCE_RATES,
        timestep_units="s",
        chain_file=reduced,
    )
    steps = []
    for m in results[1:]:  # drop initial state
        dens = m.get_nuclide_atom_densities()
        steps.append({k: float(v) for k, v in dens.items()})
    return {
        "nuclide": nuclide,
        "timesteps": [float(x) for x in C.TRANSMUTE_TIMESTEPS],
        "source_rates": [float(x) for x in C.TRANSMUTE_SOURCE_RATES],
        "steps": steps,
        "library": LIBRARY,
    }


GENERATORS = {
    "neutron": (gen_neutron, C.ALL_NEUTRON_NUCLIDES),
    "coupled": (gen_coupled, C.COUPLED_NUCLIDES),
    "decay_photons": (gen_decay_photons, C.COUPLED_NUCLIDES),
    "transmutation": (gen_transmutation, C.ALL_NEUTRON_NUCLIDES),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="+", default=list(GENERATORS), choices=list(GENERATORS))
    ap.add_argument("--nuclides", nargs="+", default=None)
    args = ap.parse_args()
    os.makedirs(REF_DIR, exist_ok=True)
    for typ in args.types:
        gen, nuclides = GENERATORS[typ]
        if args.nuclides:
            nuclides = [n for n in nuclides if n in args.nuclides]
        for nuc in nuclides:
            element = C.NUCLIDE_TO_ELEMENT.get(nuc)
            print(f"[{typ}] {nuc} ...", flush=True)
            try:
                data = gen(nuc, element)
            except Exception as e:  # noqa: BLE001
                print(f"[{typ}] {nuc} FAILED: {e}", flush=True)
                continue
            path = os.path.join(REF_DIR, f"{typ}_{nuc}.json")
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[{typ}] {nuc} -> {os.path.basename(path)}", flush=True)


if __name__ == "__main__":
    main()
