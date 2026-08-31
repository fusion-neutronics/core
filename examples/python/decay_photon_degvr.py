"""DeGVR weight windows for a D1S shutdown-dose (decay-photon) problem.

A 14 MeV neutron source activates an iron rod; the D1S method
(``use_decay_photons=True``) emits the decay photons of the activation products
at their production sites. Deep in the rod those decay photons are both rare and
attenuated, so analog transport resolves the deep decay-dose poorly. A DeGVR
``particle="photon"`` window, generated from one pair of reduced-density passes,
splits photons as they penetrate and cuts the deep-region variance -- unbiased,
at the same particle count. The pulse-schedule time-correction (not shown) is a
linear post-scaling applied afterwards and does not interact with the window.

Caveat: reducing density for the DeGVR passes lowers the activation rate as well
as the attenuation, so the decay-photon source scales with density and the
density extrapolation is more approximate than for pure attenuation. The window
stays unbiased regardless; only its efficiency is affected. Keep the generation
budget modest (the D1S activation path runs twice at reduced density).

Uses the ``tests/`` fixtures (Fe56 neutron + Fe element photon data + the
SFR transmutation chain).

Run (from the repo root):  python examples/python/decay_photon_degvr.py
"""
import tempfile

import numpy as np

import yamc

# --- knobs -----------------------------------------------------------------
DATA_DIR = "tests"
CHAIN = f"{DATA_DIR}/transmutation-endf-b8.1-sfr.arrow"
LENGTH = 40.0
RADIUS = 3.0
N_SLICES = 10
GEN_PARTICLES = 200_000
PROD_PARTICLES = 800_000
SEED = 7

# --- reduce the chain to Fe56's activation products ------------------------
reduced = yamc.TransmutationChain(CHAIN).reduce(["Fe56"], 5)
chain_dir = tempfile.mkdtemp(suffix=".chain.arrow")
reduced.export_to_arrow(chain_dir)
yamc.transmutation_decay_data = chain_dir
yamc.transmutation_reactions = chain_dir
yamc.transmutation_fission_yields = chain_dir

# --- material + geometry ---------------------------------------------------
iron = yamc.Material(composition={"Fe56": 1.0}, density=7.874, temperature=294)
iron.read_nuclear_data({"Fe56": f"{DATA_DIR}/Fe56.arrow"},
                       photon_data={"Fe": f"{DATA_DIR}/Fe.arrow"})
cyl = yamc.Cylinder(axis="z", radius=RADIUS, boundary="vacuum")
z0 = yamc.Plane(axis="z", offset=0.0, boundary="vacuum")
z1 = yamc.Plane(axis="z", offset=LENGTH, boundary="vacuum")
geom = yamc.Geometry(
    [yamc.Cell(name="rod", region=cyl.below & z0.above & z1.below, material=iron)]
)
src = yamc.NeutronSource(position=(0.0, 0.0, 0.5),
                         energy=yamc.sources.Discrete([14.06e6], [1.0]))

# activation products reachable from the model (D1S decay-photon parents)
radionuclides = yamc.Model(geometry=geom, source=src).radionuclides()
print(f"decay-photon parents: {radionuclides}")

mesh = yamc.RegularRectangularMesh(
    lower_left=[-RADIUS, -RADIUS, 0.0], upper_right=[RADIUS, RADIUS, LENGTH],
    shape=[1, 1, N_SLICES])


def build(**kw):
    tally = yamc.Tally(scores=["flux"], name="decay", mesh=mesh, particle="photon",
                       parent_nuclides=radionuclides)
    return yamc.Model(geometry=geom, tallies=[tally], source=src,
                      transport_secondary_photons=True, use_decay_photons=True, **kw)


# --- one DeGVR photon window from the D1S run ------------------------------
wwb = build().generate_weight_windows(
    yamc.WeightWindowGeneratorDeGVR(mesh=mesh, particle="photon"),
    total_particles=GEN_PARTICLES, seed=1)

# --- analog vs weight-window production ------------------------------------
res_a = build().simulate_transport(total_particles=PROD_PARTICLES, seed=SEED)["decay"]
res_w = build(variance_reduction=[wwb]).simulate_transport(
    total_particles=PROD_PARTICLES, seed=SEED)["decay"]

# total decay flux per voxel, summed over the parent-nuclide bins
n_par = len(radionuclides)
ma = np.array(res_a.mean).reshape(n_par, N_SLICES)
ea = np.array(res_a.relative_error).reshape(n_par, N_SLICES)
mw = np.array(res_w.mean).reshape(n_par, N_SLICES)
ew = np.array(res_w.relative_error).reshape(n_par, N_SLICES)
tot_a = ma.sum(0)
tot_w = mw.sum(0)
re_a = np.sqrt(((ma * ea) ** 2).sum(0)) / np.where(tot_a > 0, tot_a, 1)
re_w = np.sqrt(((mw * ew) ** 2).sum(0)) / np.where(tot_w > 0, tot_w, 1)

half = N_SLICES // 2
print(f"deep-half decay flux, mean rel_err   analog {re_a[half:].mean():.3f}"
      f"   WW {re_w[half:].mean():.3f}")
print(f"deepest slice {N_SLICES - 1}: rel_err   analog {re_a[-1]:.3f}   WW {re_w[-1]:.3f}")
