"""Where do weight windows start to pay off? A shield-thickness sweep.

A 50 cm void cavity with a 14 MeV point source at its centre, wrapped in a
concrete shell of thickness t. A thin detector sits in the far (+x) face of the
shell, so the tally we care about is attenuated by ~t of concrete. For each
thickness we run analog and DeGVR weight windows for the SAME production
wall-clock (generation excluded), and compare the detector figure of merit
FOM = 1 / (rel_err^2 * time). We report:

  - the measured attenuation (decades) from cavity to detector,
  - FOM_WW / FOM_analog with generation EXCLUDED (do the windows themselves help?),
  - FOM_WW / FOM_analog with generation INCLUDED (is the whole method worth it?),

so the crossover thickness / decade count is explicit.
"""
import math
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yamc

T_PROD = float(os.environ.get("T_PROD", 60.0))       # production wall-clock per method
GEN_PARTICLES = int(os.environ.get("GEN_PARTICLES", 150_000))
CAV = 50.0                                            # cavity half-width (cm)
THICKNESSES = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0]
HUGE = 5_000_000_000

concrete = yamc.Material(composition={"O16": 0.50, "Al27": 0.20, "C12": 0.15, "Fe56": 0.15},
                         density=2.3, temperature=294)
concrete.read_nuclear_data({n: f"tests/{n}.arrow" for n in ("O16", "Al27", "C12", "Fe56")})
src = yamc.NeutronSource(position=(0.0, 0.0, 0.0), energy=yamc.sources.Discrete([14.06e6], [1.0]),
                         direction=yamc.sources.Isotropic())


def cube(half, boundary):
    p = {ax: (yamc.Plane(axis=ax, offset=-half, boundary=boundary),
              yamc.Plane(axis=ax, offset=half, boundary=boundary)) for ax in "xyz"}
    r = None
    for ax in "xyz":
        lo, hi = p[ax]
        seg = lo.above & hi.below
        r = seg if r is None else r & seg
    return r


def make_geom(t):
    inner = cube(CAV, "transmission")
    outer = cube(CAV + t, "vacuum")
    return yamc.Geometry([
        yamc.Cell(name="cavity", region=inner),                          # void
        yamc.Cell(name="shield", region=outer & ~inner, material=concrete),
    ])


def detector_tally(t, name):
    # thin slab in the far (+x) face of the shell: penetration ~ t
    mesh = yamc.RegularRectangularMesh(
        lower_left=[CAV + t - 30.0, -100.0, -100.0], upper_right=[CAV + t, 100.0, 100.0],
        shape=[1, 6, 6])
    return yamc.Tally(scores=["flux"], name=name, mesh=mesh, particle="neutron")


def cavity_tally(name):
    mesh = yamc.RegularRectangularMesh(lower_left=[-CAV, -CAV, -CAV], upper_right=[CAV, CAV, CAV],
                                       shape=[2, 2, 2])
    return yamc.Tally(scores=["flux"], name=name, mesh=mesh, particle="neutron")


def agg(res, name):
    """(aggregate flux, aggregate rel err) for a small detector tally."""
    flux = float(np.sum(res[name].mean))
    re = float(res[name].aggregate_relative_error)
    return flux, re


rows = []
for t in THICKNESSES:
    gen_mesh = yamc.RegularRectangularMesh(
        lower_left=[-(CAV + t)] * 3, upper_right=[CAV + t] * 3,
        shape=[max(8, round(2 * (CAV + t) / 25.0))] * 3)
    # generate windows (timed, but its cost is reported separately)
    t0 = time.time()
    wwb = yamc.Model(geometry=make_geom(t), source=src, tallies=[]).generate_weight_windows(
        yamc.WeightWindowGeneratorDeGVR(mesh=gen_mesh, particle="neutron"),
        total_particles=GEN_PARTICLES, seed=1)
    t_gen = time.time() - t0

    # analog production (T_PROD)
    ra_model = yamc.Model(geometry=make_geom(t), source=src,
                          tallies=[detector_tally(t, "det"), cavity_tally("cav")])
    res_a = ra_model.simulate_transport(total_particles=HUGE, seed=7, max_runtime=T_PROD)
    fa_det, re_a = agg(res_a, "det")
    fa_cav, _ = agg(res_a, "cav")

    # WW production (same T_PROD, generation excluded)
    rw_model = yamc.Model(geometry=make_geom(t), source=src,
                          tallies=[detector_tally(t, "det"), cavity_tally("cav")],
                          variance_reduction=[wwb])
    res_w = rw_model.simulate_transport(total_particles=HUGE, seed=7, max_runtime=T_PROD)
    fw_det, re_w = agg(res_w, "det")
    fw_cav, _ = agg(res_w, "cav")

    # attenuation (decades) from the WW run, which resolves both ends
    cav_ref = fw_cav if fw_cav > 0 else fa_cav
    decades = math.log10(cav_ref / fw_det) if (cav_ref > 0 and fw_det > 0) else float("nan")

    def fom(re, tsec, flux):
        return 1.0 / (re * re * tsec) if (flux > 0 and re > 0) else 0.0

    fom_a = fom(re_a, T_PROD, fa_det)
    fom_w_prod = fom(re_w, T_PROD, fw_det)
    fom_w_incl = fom(re_w, T_PROD + t_gen, fw_det)
    ratio_prod = (fom_w_prod / fom_a) if fom_a > 0 else float("inf")
    ratio_incl = (fom_w_incl / fom_a) if fom_a > 0 else float("inf")
    rows.append(dict(t=t, decades=decades, re_a=re_a, re_w=re_w, fa=fa_det, fw=fw_det,
                     t_gen=t_gen, ratio_prod=ratio_prod, ratio_incl=ratio_incl))
    print(f"t={t:4.0f}cm  ~{decades:4.1f} dec  analog re={re_a:.3f} WW re={re_w:.3f}  "
          f"FOM WW/analog: prod-only {ratio_prod:6.2f}x  gen-incl {ratio_incl:6.2f}x  "
          f"(det flux a={fa_det:.2e} w={fw_det:.2e}, t_gen={t_gen:.0f}s)", flush=True)

# --- plot: FOM ratio vs attenuation depth -----------------------------------
tt = [r["t"] for r in rows]
dd = [r["decades"] for r in rows]
rp = [min(r["ratio_prod"], 1e3) for r in rows]
ri = [min(r["ratio_incl"], 1e3) for r in rows]
fig, ax = plt.subplots(figsize=(9, 5.6))
ax.axhline(1.0, color="k", lw=1, ls="--", label="break-even (FOM equal)")
ax.plot(tt, rp, "o-", color="C0", lw=2, label="windows only (generation excluded)")
ax.plot(tt, ri, "s--", color="C1", lw=2, label="whole method (generation included)")
for r, x in zip(rows, tt):
    ax.annotate(f"{r['decades']:.0f} dec", (x, min(r['ratio_prod'], 1e3)),
                textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
ax.set_yscale("log")
ax.set_xlabel("concrete shield thickness (cm)")
ax.set_ylabel("FOM ratio  (weight windows / analog)  at the deep detector")
ax.set_title(f"When do weight windows pay off? Concrete box sweep, {T_PROD:.0f}s production/side\n"
             "above the dashed line WW wins; annotations are decades of attenuation")
ax.legend()
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
fig.savefig("ww_crossover.png", dpi=120)
print("SAVED ww_crossover.png", flush=True)
print("SWEEP_DONE", flush=True)
