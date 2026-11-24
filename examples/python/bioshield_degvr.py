"""Deep-penetration bioshield demo: DeGVR weight windows vs analog.

A cubic bioshield: a void interior holding a 600 cm ring source of 14 MeV
neutrons, surrounded by a 3 m thick concrete wall and a 1 m air pad, with a
vacuum boundary. Neutron flux is scored on a regular mesh over the whole model.

The point: an analog run cannot push neutrons through ~3 m of concrete (many
mean free paths), so its flux map is essentially empty outside the wall. DeGVR
weight windows, generated from two cheap reduced-density passes, split particles
inward and resolve the flux map right through the wall and into the air pad --
for the *same* wall-clock budget.

Concrete and air here are surrogates built from locally-available nuclide
fixtures (O16 / Al27 / C12 / Fe56) at concrete / air density: the deep-
attenuation physics is the point, not the exact composition. Point the material
data at your own library for a quantitative study.

Figure-of-merit note: FOM = 1 / (rel_err^2 * wall_clock) is time-normalized, so
comparing the analog and WW runs at a fixed particle count with each run's
measured wall-clock is a fair comparison (the WW budget also folds in the
generation cost). Run on an otherwise-idle machine so the timings are clean.

Usage:
    python bioshield_degvr.py
Writes bioshield_flux_maps.png (analog vs WW midplane flux) to the CWD.
"""
import math
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

import yamc  # noqa: E402

# --- knobs -----------------------------------------------------------------
MESH_N = 30            # mesh voxels per axis over the 2200 cm domain
GEN_PARTICLES = 300_000  # histories per DeGVR generation pass (x2 passes)
PROD_PARTICLES = 200_000  # source histories per production run (analog & WW)
THREADS = None         # CPU threads (None = all cores); keep equal for a fair FOM
DATA_DIR = "tests"     # directory of <nuclide>.arrow fixtures

# We compare at a fixed particle count and read each run's measured wall-clock
# into FOM = 1 / (rel_err^2 * time). FOM is time-normalized, so this is the fair
# comparison and it does NOT require equal wall-clock. (yamc's max_runtime is a
# coarse stop: the run is split into only ~10 chunks and the budget is checked
# between chunks, so it cannot bound a run to a precise wall-clock -- avoid it
# for matched-time studies.)

# --- materials (surrogates from available fixtures) ------------------------
concrete = yamc.Material(
    composition={"O16": 0.50, "Al27": 0.20, "C12": 0.15, "Fe56": 0.15},
    density=2.3,
    temperature=294,
)
concrete.read_nuclear_data({n: f"{DATA_DIR}/{n}.arrow" for n in ("O16", "Al27", "C12", "Fe56")})
air = yamc.Material(composition={"O16": 1.0}, density=1.2e-3, temperature=294)
air.read_nuclear_data({"O16": f"{DATA_DIR}/O16.arrow"})

# --- geometry: nested cubes (void 700 / 3 m concrete / 1 m air / vacuum) ----
def planes(half_width, boundary):
    return {
        ax: (
            yamc.Plane(axis=ax, offset=-half_width, boundary=boundary),
            yamc.Plane(axis=ax, offset=half_width, boundary=boundary),
        )
        for ax in "xyz"
    }


def box(p):
    region = None
    for ax in "xyz":
        lo, hi = p[ax]
        slab = lo.above & hi.below
        region = slab if region is None else region & slab
    return region


inner = box(planes(700.0, "transmission"))    # void cavity
mid = box(planes(1000.0, "transmission"))      # + 300 cm concrete
outer = box(planes(1100.0, "vacuum"))          # + 100 cm air, then vacuum

geom = yamc.Geometry(
    [
        yamc.Cell(name="void", region=inner),                               # void (no material)
        yamc.Cell(name="concrete", region=mid & ~inner, material=concrete),
        yamc.Cell(name="air", region=outer & ~mid, material=air),
    ]
)

# --- 600 cm ring source of 14 MeV neutrons in the midplane -----------------
ring = yamc.sources.CylindricalRing(
    radius=yamc.sources.Discrete([600.0], [1.0]),
    phi=yamc.sources.Uniform(0.0, 2 * math.pi),
    z=yamc.sources.Discrete([0.0], [1.0]),
)
src = yamc.NeutronSource(
    position=ring,
    energy=yamc.sources.Discrete([14.06e6], [1.0]),
    direction=yamc.sources.Isotropic(),
)

# --- mesh + flux tally over the whole model --------------------------------
mesh = yamc.RegularRectangularMesh(
    lower_left=[-1100, -1100, -1100], upper_right=[1100, 1100, 1100],
    shape=[MESH_N, MESH_N, MESH_N],
)
tally = yamc.Tally(scores=["flux"], name="flux", mesh=mesh, particle="neutron")
model = yamc.Model(geometry=geom, source=src, tallies=[tally])

# --- DeGVR generation (two reduced-density passes) -------------------------
# density_reduction is left unset -> auto: N is derived from a particle-free
# optical-depth ray-trace of the problem (pass a float to override).
t0 = time.time()
wwb = model.generate_weight_windows(
    yamc.WeightWindowGeneratorDeGVR(mesh=mesh, particle="neutron"),
    total_particles=GEN_PARTICLES, seed=1, threads=THREADS,
)
t_gen = time.time() - t0
print(f"DeGVR generation: {t_gen:.1f} s")

# --- analog vs WW production, same particle count; FOM uses measured time ---
res_a = model.simulate_transport(total_particles=PROD_PARTICLES, seed=7, threads=THREADS)
prod = yamc.Model(geometry=geom, source=src, tallies=[tally], variance_reduction=[wwb])
res_w = prod.simulate_transport(total_particles=PROD_PARTICLES, seed=7, threads=THREADS)
print(f"production wall-clock:  analog {res_a.elapsed:.1f} s   WW {res_w.elapsed:.1f} s   (+ {t_gen:.1f} s generation)")

fa = np.array(res_a["flux"].mean).reshape(MESH_N, MESH_N, MESH_N)  # [iz, iy, ix]
fw = np.array(res_w["flux"].mean).reshape(MESH_N, MESH_N, MESH_N)

# region masks from voxel-center max-norm
c = -1100 + (np.arange(MESH_N) + 0.5) * (2200.0 / MESH_N)
cz, cy, cx = np.meshgrid(c, c, c, indexing="ij")
rmax = np.maximum(np.maximum(np.abs(cx), np.abs(cy)), np.abs(cz))
air_mask = (rmax > 1000) & (rmax < 1100)  # the 1 m air pad, outside the wall

print(f"mesh coverage (voxels with flux > 0):  analog {np.mean(fa > 0):.1%}   WW {np.mean(fw > 0):.1%}")
print(
    f"air-pad voxels resolved (of {int(air_mask.sum())}):  "
    f"analog {int(np.sum((fa > 0) & air_mask))}   WW {int(np.sum((fw > 0) & air_mask))}"
)
print(f"mean flux in the air pad:  analog {fa[air_mask].mean():.3e}   WW {fw[air_mask].mean():.3e}")

# Air-pad (deep-region) relative error -- the statistic the whole-mesh FOM
# misses. Analog resolves ~no air-pad voxels, so it has no meaningful deep
# statistics; WW converges the field there. This, not the aggregate FOM (which
# is dominated by the bright near-source region), is the deep-penetration win.
re_a = np.array(res_a["flux"].relative_error).reshape(MESH_N, MESH_N, MESH_N)
re_w = np.array(res_w["flux"].relative_error).reshape(MESH_N, MESH_N, MESH_N)
aa, aw = air_mask & (fa > 0), air_mask & (fw > 0)
a_re = f"{re_a[aa].mean():.2f} (n={int(aa.sum())})" if aa.any() else "n/a (0 voxels)"
w_re = f"{re_w[aw].mean():.2f} (n={int(aw.sum())})" if aw.any() else "n/a (0 voxels)"
print(f"air-pad rel_err (resolved voxels):  analog {a_re}   WW {w_re}")

# figure of merit (fold generation cost into the WW budget); only meaningful on a quiet CPU
r_a = res_a["flux"].aggregate_relative_error
r_w = res_w["flux"].aggregate_relative_error
fom_a = 1.0 / (r_a**2 * res_a.elapsed) if r_a > 0 else 0.0
fom_w = 1.0 / (r_w**2 * (t_gen + res_w.elapsed)) if r_w > 0 else 0.0
print(f"FOM (generation-inclusive):  analog {fom_a:.3e}   WW {fom_w:.3e}")

# --- flux-map figure: midplane z-slice, log scale --------------------------
iz = MESH_N // 2
extent = [-1100, 1100, -1100, 1100]
vmax = max(fa.max(), fw.max())
vmin = vmax * 1e-12
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
for ax, data, title in ((axes[0], fa[iz], "Analog"), (axes[1], fw[iz], "DeGVR weight windows")):
    im = ax.imshow(
        np.where(data > 0, data, np.nan),
        origin="lower", extent=extent, norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis",
    )
    for hw in (700, 1000, 1100):
        ax.add_patch(plt.Rectangle((-hw, -hw), 2 * hw, 2 * hw, fill=False, ec="white", lw=0.8, alpha=0.6))
    ax.set_title(f"{title}\n(slice coverage {np.mean(data > 0):.0%})")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    fig.colorbar(im, ax=ax, label="neutron flux (a.u.)", fraction=0.046)
fig.suptitle(
    f"Bioshield: 3 m concrete shell, 600 cm 14 MeV ring source, {PROD_PARTICLES:,} histories each"
)
fig.tight_layout()
fig.savefig("bioshield_flux_maps.png", dpi=110)
print("saved bioshield_flux_maps.png")
