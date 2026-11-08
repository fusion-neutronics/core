"""DeGVR weight windows for a photon source in a thick gamma shield.

The point: a 2 MeV photon point source at the centre of a solid iron sphere
(~13 mean free paths to the surface). Analog transport almost never reaches the
outer shell, so its deep-region flux map is essentially empty. A DeGVR photon
weight window, generated from two cheap reduced-density passes, splits photons
as they penetrate so the outer shell is resolved -- at the same particle count,
unbiased. This is the photon-source analogue of ``bioshield_degvr.py``.

Uses the ``tests/`` fixtures (Fe56 neutron + Fe element photon data); the
attenuation physics is the point, not the exact material.

Run (from the repo root):  python examples/python/photon_shield_degvr.py
"""
import numpy as np

import yamc

# --- knobs -----------------------------------------------------------------
DATA_DIR = "tests"
RADIUS = 30.0          # cm of iron (~10 mfp at 2 MeV)
MESH_N = 20
GEN_PARTICLES = 200_000
PROD_PARTICLES = 500_000
SEED = 7

# --- material (iron gamma shield) ------------------------------------------
iron = yamc.Material(composition={"Fe56": 1.0}, density=7.874, temperature=294)
iron.read_nuclear_data({"Fe56": f"{DATA_DIR}/Fe56.arrow"},
                       photon_data={"Fe": f"{DATA_DIR}/Fe.arrow"})

sphere = yamc.Sphere(radius=RADIUS, boundary="vacuum")
geom = yamc.Geometry([yamc.Cell(name="iron", region=sphere.below, material=iron)])

# 2 MeV photon point source at the centre
src = yamc.PhotonSource(
    position=(0.0, 0.0, 0.0),
    energy=yamc.sources.Discrete([2.0e6], [1.0]),
    direction=yamc.sources.Isotropic(),
)

mesh = yamc.RegularRectangularMesh(
    lower_left=[-RADIUS] * 3, upper_right=[RADIUS] * 3, shape=[MESH_N] * 3)
tally = yamc.Tally(scores=["flux"], name="flux", mesh=mesh, particle="photon")
model = yamc.Model(geometry=geom, source=src, tallies=[tally],
                   transport_secondary_photons=True)

# --- DeGVR photon window (density_reduction auto by default) ---------------
wwb = model.generate_weight_windows(
    yamc.WeightWindowGeneratorDeGVR(mesh=mesh, particle="photon"),
    total_particles=GEN_PARTICLES, seed=1)

# --- analog vs weight-window production, same particle count ---------------
res_a = model.simulate_transport(total_particles=PROD_PARTICLES, seed=SEED)
prod = yamc.Model(geometry=geom, source=src, tallies=[tally],
                  transport_secondary_photons=True, variance_reduction=[wwb])
res_w = prod.simulate_transport(total_particles=PROD_PARTICLES, seed=SEED)

fa = np.array(res_a["flux"].mean).reshape(MESH_N, MESH_N, MESH_N)
fw = np.array(res_w["flux"].mean).reshape(MESH_N, MESH_N, MESH_N)
ea = np.array(res_a["flux"].relative_error).reshape(MESH_N, MESH_N, MESH_N)
ew = np.array(res_w["flux"].relative_error).reshape(MESH_N, MESH_N, MESH_N)

# deep shell: the outer 20% of the sphere, where analog struggles
c = -RADIUS + (np.arange(MESH_N) + 0.5) * (2 * RADIUS / MESH_N)
cz, cy, cx = np.meshgrid(c, c, c, indexing="ij")
r = np.sqrt(cx**2 + cy**2 + cz**2)
deep = (r > 0.8 * RADIUS) & (r <= RADIUS)

print(f"deep-shell voxels: {int(deep.sum())}")
print(f"  coverage (flux > 0):   analog {np.mean((fa > 0) & deep):.1%}"
      f"   WW {np.mean((fw > 0) & deep):.1%}")
aa, aw = deep & (fa > 0), deep & (fw > 0)
print(f"  mean rel_err (resolved): analog "
      f"{ea[aa].mean() if aa.any() else float('nan'):.3f}"
      f"   WW {ew[aw].mean() if aw.any() else float('nan'):.3f}")
# where both resolve, the means agree (weight windows are unbiased)
both = deep & (fa > 0) & (fw > 0)
if both.any():
    print(f"  WW/analog flux ratio (overlap): {fw[both].sum() / fa[both].sum():.3f}")
