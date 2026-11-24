"""Coupled neutron -> secondary-photon DeGVR weight windows in one call.

A 14 MeV neutron point source at one end of a thick iron rod. Neutrons
attenuate with depth, and the secondary photons they produce
(``transport_secondary_photons=True``) attenuate too, so both the deep neutron
flux and the deep photon flux are hard for analog transport to reach. One
``generate_weight_windows`` call with ``particle=["neutron", "photon"]`` runs a
single coupled pair of reduced-density passes and returns BOTH a neutron window
and a photon window (built from the secondary-photon field). Applied together,
they cut the deep-region variance of both fields at the same particle count,
unbiased.

Uses the ``tests/`` fixtures (Fe56 neutron + Fe element photon data); the
attenuation physics is the point, not the exact material.

Run (from the repo root):  python examples/python/coupled_np_degvr.py
"""
import numpy as np

import yamc

# --- knobs -----------------------------------------------------------------
DATA_DIR = "tests"
LENGTH = 40.0          # cm of iron rod
RADIUS = 3.0
N_SLICES = 10
GEN_PARTICLES = 200_000
PROD_PARTICLES = 600_000
SEED = 7

# --- material (iron) -------------------------------------------------------
iron = yamc.Material(composition={"Fe56": 1.0}, density=7.874, temperature=294)
iron.read_nuclear_data({"Fe56": f"{DATA_DIR}/Fe56.arrow"},
                       photon_data={"Fe": f"{DATA_DIR}/Fe.arrow"})

cyl = yamc.Cylinder(axis="z", radius=RADIUS, boundary="vacuum")
z0 = yamc.Plane(axis="z", offset=0.0, boundary="vacuum")
z1 = yamc.Plane(axis="z", offset=LENGTH, boundary="vacuum")
geom = yamc.Geometry(
    [yamc.Cell(name="rod", region=cyl.below & z0.above & z1.below, material=iron)]
)

# 14 MeV neutron point source near the near end
src = yamc.NeutronSource(
    position=(0.0, 0.0, 0.5),
    energy=yamc.sources.Discrete([14.1e6], [1.0]),
)

mesh = yamc.RegularRectangularMesh(
    lower_left=[-RADIUS, -RADIUS, 0.0], upper_right=[RADIUS, RADIUS, LENGTH],
    shape=[1, 1, N_SLICES])
n_flux = yamc.Tally(scores=["flux"], name="n_flux", mesh=mesh, particle="neutron")
p_flux = yamc.Tally(scores=["flux"], name="p_flux", mesh=mesh, particle="photon")


def build(**kw):
    return yamc.Model(geometry=geom, source=src, tallies=[n_flux, p_flux],
                      transport_secondary_photons=True, **kw)


# --- ONE coupled DeGVR generation -> [neutron_ww, photon_ww] ---------------
neutron_ww, photon_ww = build().generate_weight_windows(
    yamc.WeightWindowGeneratorDeGVR(mesh=mesh, particle=["neutron", "photon"]),
    total_particles=GEN_PARTICLES, seed=1)
print(f"neutron window: {neutron_ww}")
print(f"photon window:  {photon_ww}")

# --- analog vs weight-window production, same particle count ---------------
res_a = build().simulate_transport(total_particles=PROD_PARTICLES, seed=SEED)
res_w = build(variance_reduction=[neutron_ww, photon_ww]).simulate_transport(
    total_particles=PROD_PARTICLES, seed=SEED)

for name, label in [("n_flux", "neutron"), ("p_flux", "photon")]:
    ea = np.array(res_a[name].relative_error)
    ew = np.array(res_w[name].relative_error)
    ma = np.array(res_a[name].mean)
    deep = N_SLICES - 1
    print(f"\n{label} flux, deepest slice {deep} (mean {ma[deep]:.2e}):")
    print(f"  rel_err   analog {ea[deep]:.3f}   WW {ew[deep]:.3f}")
    print(f"  deep-half mean rel_err   analog {ea[N_SLICES // 2:].mean():.3f}"
          f"   WW {ew[N_SLICES // 2:].mean():.3f}")
