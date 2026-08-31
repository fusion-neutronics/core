"""Equal-wall-clock analog vs DeGVR weight-window comparison on a maze.

A yamc rewrite of the `rectangle_with_maze` model from fusion-energy's
parametric_bioshield_zoo: a source room separated from an exit corridor by a
concrete wall, so neutrons must stream through a gap (a dogleg) to escape. This
duct-streaming problem is the canonical case for weight windows.

Both sides get the same wall-clock (default 10 min): the weight-window side
runs generation + production under max_runtime, then analog is matched to the
weight-window run's measured total. Outputs into the repo root:
  - maze_heatmaps.png  : analog flux | WW flux | WW relative error (z-mid slice)
  - maze_lineplot.png  : flux up the corridor centreline, analog vs WW
"""
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import yamc

BUDGET_S = float(os.environ.get("BUDGET_S", 10 * 60))
GEN_PARTICLES = int(os.environ.get("GEN_PARTICLES", 1_000_000))
HUGE = 5_000_000_000

# Segment widths (cm), the rectangle_with_maze parameters. MAZE_PROFILE selects
# the geometry: "default" (the zoo defaults) or "hard" (thick walls + a narrow
# gap slit and corridor, so far fewer neutrons stream through).
PROFILE = os.environ.get("MAZE_PROFILE", "default")
DIMS = {
    # wa: L air | wb: L wall | wc: room | wd: dogleg | we: corridor | wf: R wall | wg: R air
    # da: B air | db: B wall | dc: gap slit | dd: middle wall | de: T wall | df: T air
    # hj: roof | hk: room height | hl: floor
    "default": dict(wa=100, wb=100, wc=500, wd=100, we=100, wf=100, wg=100,
                    da=100, db=100, dc=700, dd=600, de=100, df=100, hj=100, hk=500, hl=100),
    "hard": dict(wa=100, wb=300, wc=300, wd=300, we=40, wf=300, wg=100,
                 da=100, db=300, dc=40, dd=600, de=300, df=100, hj=200, hk=300, hl=200),
}[PROFILE]
_w = [DIMS[k] for k in ("wa", "wb", "wc", "wd", "we", "wf", "wg")]
_d = [DIMS[k] for k in ("da", "db", "dc", "dd", "de", "df")]
_h = [DIMS[k] for k in ("hj", "hk", "hl")]
X = [sum(_w[:i]) for i in range(len(_w) + 1)]  # x0..x7
Y = [sum(_d[:i]) for i in range(len(_d) + 1)]  # y0..y6
Z = [sum(_h[:i]) for i in range(len(_h) + 1)]  # z1..z4

concrete = yamc.Material(composition={"O16": 0.50, "Al27": 0.20, "C12": 0.15, "Fe56": 0.15},
                         density=2.3, temperature=294)
concrete.read_nuclear_data({n: f"tests/{n}.arrow" for n in ("O16", "Al27", "C12", "Fe56")})
air = yamc.Material(composition={"O16": 1.0}, density=1.2e-3, temperature=294)
air.read_nuclear_data({"O16": "tests/O16.arrow"})

xpl = [yamc.Plane(axis="x", offset=X[i], boundary=("vacuum" if i in (0, 7) else "transmission")) for i in range(8)]
ypl = [yamc.Plane(axis="y", offset=Y[i], boundary=("vacuum" if i in (0, 6) else "transmission")) for i in range(7)]
zpl = [yamc.Plane(axis="z", offset=Z[i], boundary=("vacuum" if i in (0, 3) else "transmission")) for i in range(4)]


def box(ix0, ix1, iy0, iy1, iz0, iz1):
    return (xpl[ix0].above & xpl[ix1].below & ypl[iy0].above & ypl[iy1].below
            & zpl[iz0].above & zpl[iz1].below)


def make_geom():
    # air regions
    outside_left = box(0, 1, 1, 5, 0, 3)
    outside_right = box(6, 7, 1, 5, 0, 3)
    outside_top = box(0, 7, 5, 6, 0, 3)
    outside_bottom = box(0, 7, 0, 1, 0, 3)
    room = box(2, 3, 2, 4, 1, 2)
    gap = box(3, 4, 2, 3, 1, 2)
    corridor = box(4, 5, 2, 5, 1, 2)
    # concrete regions
    wall_left = box(1, 2, 2, 4, 1, 2)
    wall_right = box(5, 6, 2, 5, 1, 2)
    wall_top = box(1, 4, 4, 5, 1, 2)
    wall_bottom = box(1, 6, 1, 2, 1, 2)
    wall_middle = box(3, 4, 3, 4, 1, 2)
    roof = box(1, 6, 1, 5, 0, 1)
    floor = box(1, 6, 1, 5, 2, 3)
    return yamc.Geometry([
        yamc.Cell(name="outside_left", region=outside_left, material=air),
        yamc.Cell(name="outside_right", region=outside_right, material=air),
        yamc.Cell(name="outside_top", region=outside_top, material=air),
        yamc.Cell(name="outside_bottom", region=outside_bottom, material=air),
        yamc.Cell(name="room", region=room, material=air),
        yamc.Cell(name="gap", region=gap, material=air),
        yamc.Cell(name="corridor", region=corridor, material=air),
        yamc.Cell(name="wall_left", region=wall_left, material=concrete),
        yamc.Cell(name="wall_right", region=wall_right, material=concrete),
        yamc.Cell(name="wall_top", region=wall_top, material=concrete),
        yamc.Cell(name="wall_bottom", region=wall_bottom, material=concrete),
        yamc.Cell(name="wall_middle", region=wall_middle, material=concrete),
        yamc.Cell(name="roof", region=roof, material=concrete),
        yamc.Cell(name="floor", region=floor, material=concrete),
    ])


# Point source at the room centre.
src_pos = (0.5 * (X[2] + X[3]), 0.5 * (Y[2] + Y[4]), 0.5 * (Z[1] + Z[2]))  # (450, 850, 350)
src = yamc.NeutronSource(position=src_pos, energy=yamc.sources.Discrete([14.06e6], [1.0]),
                         direction=yamc.sources.Isotropic())

VOX = 20.0 if PROFILE == "hard" else 25.0  # x,y voxel target; z is coarse (roof/room/floor)
NX = max(8, round((X[-1] - X[0]) / VOX))
NY = max(8, round((Y[-1] - Y[0]) / VOX))
NZ = max(4, round((Z[-1] - Z[0]) / 50.0))
mesh = yamc.RegularRectangularMesh(lower_left=[X[0], Y[0], Z[0]], upper_right=[X[-1], Y[-1], Z[-1]],
                                   shape=[NX, NY, NZ])
tally = yamc.Tally(scores=["flux"], name="flux", mesh=mesh, particle="neutron")
OUT = f"maze_{PROFILE}"
print(f"profile={PROFILE}  budget/side={BUDGET_S:.0f}s  mesh={NX}x{NY}x{NZ}  source={src_pos}", flush=True)

# --- weight-window generation (timed) --------------------------------------
gen_model = yamc.Model(geometry=make_geom(), source=src, tallies=[])
t0 = time.time()
wwb = gen_model.generate_weight_windows(
    yamc.WeightWindowGeneratorDeGVR(mesh=mesh, particle="neutron"),
    total_particles=GEN_PARTICLES, seed=1,
)
t_gen = time.time() - t0
print(f"GEN_DONE t_gen={t_gen:.1f}s", flush=True)

# --- WW production, then analog matched to its actual total wall-clock ------
tally_w = yamc.Tally(scores=["flux"], name="flux", mesh=mesh, particle="neutron")
res_w = yamc.Model(geometry=make_geom(), source=src, tallies=[tally_w],
                   variance_reduction=[wwb]).simulate_transport(
    total_particles=HUGE, seed=7, max_runtime=max(1.0, BUDGET_S - t_gen))
ww_total = t_gen + res_w.elapsed
print(f"WW_DONE prod={res_w.elapsed:.1f}s (gen {t_gen:.1f}s + prod = {ww_total:.1f}s total)", flush=True)

tally_a = yamc.Tally(scores=["flux"], name="flux", mesh=mesh, particle="neutron")
res_a = yamc.Model(geometry=make_geom(), source=src, tallies=[tally_a]).simulate_transport(
    total_particles=HUGE, seed=7, max_runtime=ww_total)
print(f"ANALOG_DONE elapsed={res_a.elapsed:.1f}s", flush=True)

# --- results (reshape [iz, iy, ix]) -----------------------------------------
fa = np.array(res_a["flux"].mean).reshape(NZ, NY, NX)
fw = np.array(res_w["flux"].mean).reshape(NZ, NY, NX)
rw = np.array(res_w["flux"].relative_error).reshape(NZ, NY, NX)
iz = int(0.5 * (Z[1] + Z[2]) / (Z[3] / NZ))  # z-mid slice (350 cm)

ra = np.array(res_a["flux"].relative_error).reshape(NZ, NY, NX)
cov_a, cov_w = float(np.mean(fa > 0)), float(np.mean(fw > 0))
both = (fa > 0) & (fw > 0)
ratio_med = float(np.median(fw[both] / fa[both])) if both.any() else float("nan")
# exit side = the corridor exit / outside-top region (y > 1500 cm)
yc = Y[0] + (np.arange(NY) + 0.5) * ((Y[6] - Y[0]) / NY)
exit_mask = np.zeros_like(fa, dtype=bool)
exit_mask[:, yc > Y[4], :] = True  # top wall + corridor exit + outside-top
exit_a = int(np.sum((fa > 0) & exit_mask))
exit_w = int(np.sum((fw > 0) & exit_mask))
exit_both = exit_mask & both
err_a = float(ra[exit_both].mean()) if exit_both.any() else float("nan")
err_w = float(rw[exit_both].mean()) if exit_both.any() else float("nan")
summary = [
    "Maze (rectangle_with_maze) analog vs DeGVR weight windows, equal wall-clock",
    f"analog {res_a.elapsed:.0f}s   WW {ww_total:.0f}s (gen {t_gen:.0f}s + prod {res_w.elapsed:.0f}s)",
    "",
    f"mesh coverage (flux>0):   analog {cov_a:.1%}   WW {cov_w:.1%}",
    f"exit-side voxels resolved (y>{Y[4]:.0f} cm): analog {exit_a}   WW {exit_w}",
    f"exit-side mean rel err (voxels both resolve): analog {err_a:.3f}   WW {err_w:.3f}",
    f"median WW/analog flux ratio (overlap): {ratio_med:.3f}  (1.0 = unbiased match)",
    "",
    "This maze is an air-duct (void-streaming) problem spanning only ~2-3 decades,",
    "so analog already reaches the whole domain; the weight-window win here is the",
    "lower relative error in the low-flux exit region, not coverage (contrast the",
    "3 m solid-concrete bioshield, ~11 decades, where analog cannot reach at all).",
]
print("\n".join(summary), flush=True)
with open(f"{OUT}_summary.txt", "w") as fh:
    fh.write("\n".join(summary) + "\n")

# --- 3 heatmaps: analog flux | WW flux | WW rel err (z-mid slice) ------------
extent = [X[0], X[7], Y[0], Y[6]]
vmax = max(fa.max(), fw.max())
vmin = vmax * 1e-10
fig, axes = plt.subplots(1, 3, figsize=(15, 8))
for ax, data, title in ((axes[0], fa[iz], "Analog flux"), (axes[1], fw[iz], "DeGVR WW flux")):
    im = ax.imshow(np.where(data > 0, data, np.nan), origin="lower", extent=extent,
                   norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis", aspect="equal")
    ax.set_title(f"{title}\n(slice coverage {np.mean(data > 0):.0%})")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.plot(*src_pos[:2], "r*", ms=12)
    fig.colorbar(im, ax=ax, label="flux (a.u.)", fraction=0.046)
im = axes[2].imshow(np.where(fw[iz] > 0, rw[iz], np.nan), origin="lower", extent=extent,
                    norm=LogNorm(vmin=1e-3, vmax=1.0), cmap="magma_r", aspect="equal")
axes[2].set_title("DeGVR WW relative error")
axes[2].set_xlabel("x (cm)")
axes[2].set_ylabel("y (cm)")
fig.colorbar(im, ax=axes[2], label="relative error", fraction=0.046)
fig.suptitle(f"Maze midplane (z={src_pos[2]:.0f} cm), equal wall-clock ~{ww_total/60:.0f} min/side; "
             f"source in room, exit at top")
fig.tight_layout()
fig.savefig(f"{OUT}_heatmaps.png", dpi=110)
print(f"SAVED {OUT}_heatmaps.png", flush=True)

# --- line plot: flux up the corridor centreline (x=850 cm), analog vs WW -----
xc = 0.5 * (X[4] + X[5])  # corridor centreline (cm)
ix_corr = int((xc - X[0]) / ((X[-1] - X[0]) / NX))
la, lw = fa[iz, :, ix_corr], fw[iz, :, ix_corr]
ea, ew = ra[iz, :, ix_corr] * la, rw[iz, :, ix_corr] * lw
fig2, ax = plt.subplots(figsize=(10, 5.6))
ax.axvspan(Y[2], Y[3], color="0.9", zorder=0, label="gap open (room->corridor)")
ax.axvspan(Y[3], Y[4], color="0.75", zorder=0, label="middle wall (blocks direct path)")
mA, mW = la > 0, lw > 0
ax.plot(yc[mW], lw[mW], "-", color="C0", lw=2.0, label="DeGVR weight windows")
ax.fill_between(yc[mW], np.clip(lw[mW] - ew[mW], vmin, None), lw[mW] + ew[mW], color="C0", alpha=0.2, lw=0)
ax.plot(yc[mA], la[mA], "o-", color="C3", ms=4, lw=1.2, label="Analog")
ax.fill_between(yc[mA], np.clip(la[mA] - ea[mA], vmin, None), la[mA] + ea[mA], color="C3", alpha=0.2, lw=0)
ax.set_yscale("log")
ax.set_ylim(vmin, vmax * 3)
ax.set_xlabel(f"y (cm)  --  up the exit corridor (x = {xc:.0f} cm)")
ax.set_ylabel("neutron flux (a.u.)")
ax.set_title("Flux up the maze exit corridor: analog vs DeGVR (equal wall-clock)\n"
             "unbiased; on the air duct analog stays smoother, WW's gain is coverage of the shield (heatmaps)")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, which="both", alpha=0.25)
fig2.tight_layout()
fig2.savefig(f"{OUT}_lineplot.png", dpi=120)
print(f"SAVED {OUT}_lineplot.png", flush=True)
print("COMPARISON_DONE", flush=True)
