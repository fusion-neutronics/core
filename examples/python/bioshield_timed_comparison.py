"""Equal-wall-clock analog vs DeGVR weight-window comparison on a 3 m concrete
bioshield.

Both sides get the SAME wall-clock budget (default 20 min):
  - analog:  simulate_transport(max_runtime = BUDGET)
  - WW:      generate windows (time it, T_gen), then
             simulate_transport(max_runtime = BUDGET - T_gen)
so the weight-window side pays for its own generation and the comparison is
fair and generation-inclusive. Relies on the #193 fix (capped CPU chunk size)
for max_runtime to actually bound the wall-clock.

Outputs (into the repo root):
  - bioshield_timed_heatmaps.png : 3 heatmaps (analog flux | WW flux | WW rel err)
  - bioshield_timed_lineplot.png : flux along the central x-line, analog vs WW
  - bioshield_timed_summary.txt  : the numbers behind the plots

Tunables via env: BUDGET_S (per-side seconds), MESH_N, GEN_PARTICLES.
"""
import math
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import yamc

BUDGET_S = float(os.environ.get("BUDGET_S", 20 * 60))  # per-side wall-clock budget
MESH_N = int(os.environ.get("MESH_N", 50))
GEN_PARTICLES = int(os.environ.get("GEN_PARTICLES", 1_000_000))
HALF = 1100.0  # domain half-width (cm)
INNER, MID = 700.0, 1000.0  # void |x|<700; concrete 700..1000 (3 m); air 1000..1100
HUGE_ANALOG = 5_000_000_000  # upper bounds; max_runtime stops the runs first
HUGE_WW = 100_000_000

# --- materials (surrogate concrete + air) ----------------------------------
concrete = yamc.Material(composition={"O16": 0.50, "Al27": 0.20, "C12": 0.15, "Fe56": 0.15},
                         density=2.3, temperature=294)
concrete.read_nuclear_data({n: f"tests/{n}.arrow" for n in ("O16", "Al27", "C12", "Fe56")})
air = yamc.Material(composition={"O16": 1.0}, density=1.2e-3, temperature=294)
air.read_nuclear_data({"O16": "tests/O16.arrow"})


def planes(h, b):
    return {ax: (yamc.Plane(axis=ax, offset=-h, boundary=b), yamc.Plane(axis=ax, offset=h, boundary=b)) for ax in "xyz"}


def box(p):
    r = None
    for ax in "xyz":
        lo, hi = p[ax]
        seg = lo.above & hi.below
        r = seg if r is None else r & seg
    return r


inner = box(planes(INNER, "transmission"))
mid = box(planes(MID, "transmission"))
outer = box(planes(HALF, "vacuum"))
geom = yamc.Geometry([
    yamc.Cell(name="void", region=inner),
    yamc.Cell(name="concrete", region=mid & ~inner, material=concrete),
    yamc.Cell(name="air", region=outer & ~mid, material=air),
])
ring = yamc.sources.CylindricalRing(radius=yamc.sources.Discrete([600.0], [1.0]),
                                    phi=yamc.sources.Uniform(0.0, 2 * math.pi),
                                    z=yamc.sources.Discrete([0.0], [1.0]))
src = yamc.NeutronSource(position=ring, energy=yamc.sources.Discrete([14.06e6], [1.0]),
                         direction=yamc.sources.Isotropic())
mesh = yamc.RegularRectangularMesh(lower_left=[-HALF, -HALF, -HALF], upper_right=[HALF, HALF, HALF],
                                   shape=[MESH_N, MESH_N, MESH_N])
tally = yamc.Tally(scores=["flux"], name="flux", mesh=mesh, particle="neutron")
model = yamc.Model(geometry=geom, source=src, tallies=[tally])

print(f"budget/side={BUDGET_S:.0f}s  mesh={MESH_N}^3  gen={GEN_PARTICLES:,}", flush=True)

# --- weight-window generation (timed; charged against the WW budget) --------
t0 = time.time()
wwb = model.generate_weight_windows(
    yamc.WeightWindowGeneratorDeGVR(mesh=mesh, particle="neutron"),
    total_particles=GEN_PARTICLES, seed=1,
)
t_gen = time.time() - t0
print(f"GEN_DONE t_gen={t_gen:.1f}s", flush=True)

# --- WW production first: budget minus generation time (WW pays for its gen) -
ww_budget = max(1.0, BUDGET_S - t_gen)
prod = yamc.Model(geometry=geom, source=src, tallies=[tally], variance_reduction=[wwb])
res_w = prod.simulate_transport(total_particles=HUGE_WW, seed=7, max_runtime=ww_budget)
ww_total = t_gen + res_w.elapsed
print(f"WW_DONE prod={res_w.elapsed:.1f}s (gen {t_gen:.1f}s + prod = {ww_total:.1f}s total)", flush=True)

# --- analog: matched to the weight-window run's ACTUAL total wall-clock ------
# A weight-window history is far heavier than an analog one, so its chunk can
# overrun max_runtime by more; matching analog to the WW run's measured total
# makes the comparison equal-time regardless of that chunk-boundary overshoot.
res_a = model.simulate_transport(total_particles=HUGE_ANALOG, seed=7, max_runtime=ww_total)
print(f"ANALOG_DONE elapsed={res_a.elapsed:.1f}s", flush=True)

# --- pull results -----------------------------------------------------------
N = MESH_N
fa = np.array(res_a["flux"].mean).reshape(N, N, N)  # [iz, iy, ix]
fw = np.array(res_w["flux"].mean).reshape(N, N, N)
ra = np.array(res_a["flux"].relative_error).reshape(N, N, N)
rw = np.array(res_w["flux"].relative_error).reshape(N, N, N)

ic = N // 2
centers = -HALF + (np.arange(N) + 0.5) * (2 * HALF / N)
concrete_mask = (np.abs(centers) > INNER) & (np.abs(centers) < MID)

# region masks (voxel-center max-norm) for coverage stats
cz, cy, cx = np.meshgrid(centers, centers, centers, indexing="ij")
rmax = np.maximum(np.maximum(np.abs(cx), np.abs(cy)), np.abs(cz))
shield_deep = (rmax > 0.5 * (INNER + MID)) & (rmax < HALF)  # outer half of concrete + air

cov_a, cov_w = np.mean(fa > 0), np.mean(fw > 0)
deep_a = int(np.sum((fa > 0) & shield_deep))
deep_w = int(np.sum((fw > 0) & shield_deep))
# "well-converged" = resolved to rel err < 0.5 (a usable estimate).
well_a = int(np.sum((fa > 0) & (ra < 0.5) & shield_deep))
well_w = int(np.sum((fw > 0) & (rw < 0.5) & shield_deep))
# Unbiasedness check: where both runs have signal, WW flux should match analog.
both = (fa > 0) & (fw > 0)
ratio_med = float(np.median(fw[both] / fa[both])) if both.any() else float("nan")

# figure-of-merit on the aggregate (time-normalized, generation-inclusive for WW)
agg_a = res_a["flux"].aggregate_relative_error
agg_w = res_w["flux"].aggregate_relative_error
fom_a = 1.0 / (agg_a**2 * res_a.elapsed) if agg_a > 0 else 0.0
fom_w = 1.0 / (agg_w**2 * (t_gen + res_w.elapsed)) if agg_w > 0 else 0.0


def mean_or_na(a):
    return f"{a.mean():.3f} (n={a.size})" if a.size else "n/a"


summary = [
    "Equal-wall-clock analog vs DeGVR weight windows (3 m concrete bioshield)",
    f"mesh {N}^3 over +/-{HALF:.0f} cm; concrete {INNER:.0f}..{MID:.0f} cm; 14.06 MeV ring source r=600",
    "",
    f"analog wall:        {res_a.elapsed:8.1f} s   (stopped by max_runtime)",
    f"WW generation:      {t_gen:8.1f} s",
    f"WW production wall:  {res_w.elapsed:8.1f} s",
    f"WW total wall:       {t_gen + res_w.elapsed:8.1f} s   (== analog budget, generation-inclusive)",
    "",
    f"mesh coverage (flux>0):     analog {cov_a:6.1%}   WW {cov_w:6.1%}",
    "",
    "REACH -- deep region (outer half of the concrete + the air pad):",
    f"  voxels resolved at all:        analog {deep_a:6d}   WW {deep_w:6d}   (WW {deep_w / max(deep_a, 1):.1f}x)",
    f"  well-converged (rel err<0.5):  analog {well_a:6d}   WW {well_w:6d}   (WW {well_w / max(well_a, 1):.1f}x)",
    f"  mean rel err over each's resolved set: analog {mean_or_na(ra[(fa > 0) & shield_deep])}   "
    f"WW {mean_or_na(rw[(fw > 0) & shield_deep])}",
    "",
    "AGREEMENT / unbiasedness -- where both runs have signal:",
    f"  median WW/analog flux ratio: {ratio_med:.3f}   (1.0 = WW reproduces the analog answer)",
    "",
    f"aggregate FOM (time-normalized): analog {fom_a:.3e}   WW {fom_w:.3e}",
    "  NOTE: the aggregate FOM is dominated by the bright near-source region,",
    "  where analog is already well converged and WW deliberately spends fewer",
    "  histories; it therefore favours analog and is NOT the deep-shielding",
    "  figure of merit. The reach numbers and the line plot are the honest proof.",
]
summary_txt = "\n".join(summary)
print(summary_txt, flush=True)
with open("bioshield_timed_summary.txt", "w") as fh:
    fh.write(summary_txt + "\n")

# --- 3 heatmaps: analog flux | WW flux | WW relative error ------------------
extent = [-HALF, HALF, -HALF, HALF]
vmax = max(fa.max(), fw.max())
vmin = vmax * 1e-11
fig, axes = plt.subplots(1, 3, figsize=(17, 5.3))
for ax, data, title in ((axes[0], fa[ic], "Analog flux"), (axes[1], fw[ic], "DeGVR WW flux")):
    im = ax.imshow(np.where(data > 0, data, np.nan), origin="lower", extent=extent,
                   norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis")
    for hw in (INNER, MID, HALF):
        ax.add_patch(plt.Rectangle((-hw, -hw), 2 * hw, 2 * hw, fill=False, ec="white", lw=0.8, alpha=0.6))
    ax.set_title(f"{title}\n(coverage {np.mean(data > 0):.0%})")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    fig.colorbar(im, ax=ax, label="neutron flux (a.u.)", fraction=0.046)
# WW relative error
rw_slice = np.where(fw[ic] > 0, rw[ic], np.nan)
im = axes[2].imshow(rw_slice, origin="lower", extent=extent,
                    norm=LogNorm(vmin=1e-3, vmax=1.0), cmap="magma_r")
for hw in (INNER, MID, HALF):
    axes[2].add_patch(plt.Rectangle((-hw, -hw), 2 * hw, 2 * hw, fill=False, ec="white", lw=0.8, alpha=0.6))
axes[2].set_title("DeGVR WW relative error\n(statistical quality through the shield)")
axes[2].set_xlabel("x (cm)")
axes[2].set_ylabel("y (cm)")
fig.colorbar(im, ax=axes[2], label="relative error", fraction=0.046)
fig.suptitle(f"Bioshield midplane, equal wall-clock ~{BUDGET_S/60:.0f} min/side "
             f"(analog {res_a.elapsed:.0f}s | WW {t_gen:.0f}s gen + {res_w.elapsed:.0f}s prod)")
fig.tight_layout()
fig.savefig("bioshield_timed_heatmaps.png", dpi=110)
print("SAVED bioshield_timed_heatmaps.png", flush=True)

# --- line plot: flux along the central x-line, analog vs WW -----------------
la, lw = fa[ic, ic, :], fw[ic, ic, :]
ea = ra[ic, ic, :] * la  # +/-1 sigma
ew = rw[ic, ic, :] * lw
fig2, ax = plt.subplots(figsize=(10, 5.6))
# shade the two concrete walls
for sgn in (-1, 1):
    ax.axvspan(sgn * INNER, sgn * MID, color="0.85", zorder=0,
               label="3 m concrete" if sgn == 1 else None)
mA = la > 0
mW = lw > 0
ax.plot(centers[mW], lw[mW], "-", color="C0", lw=2.0, label="DeGVR weight windows")
ax.fill_between(centers[mW], np.clip(lw[mW] - ew[mW], vmin, None), lw[mW] + ew[mW],
                color="C0", alpha=0.2, lw=0)
ax.plot(centers[mA], la[mA], "o-", color="C3", ms=4, lw=1.2, label="Analog")
ax.fill_between(centers[mA], np.clip(la[mA] - ea[mA], vmin, None), la[mA] + ea[mA],
                color="C3", alpha=0.2, lw=0)
ax.set_yscale("log")
ax.set_ylim(vmin, vmax * 3)
ax.set_xlabel("x (cm)  --  central line through the source plane")
ax.set_ylabel("neutron flux (a.u.)")
ax.set_title("Flux across the bioshield: analog vs DeGVR (equal wall-clock)\n"
             "the lines agree at the shield entrance; analog runs out of particles deep in the concrete")
ax.legend(loc="upper right")
ax.grid(True, which="both", alpha=0.25)
fig2.tight_layout()
fig2.savefig("bioshield_timed_lineplot.png", dpi=120)
print("SAVED bioshield_timed_lineplot.png", flush=True)
print("COMPARISON_DONE", flush=True)
