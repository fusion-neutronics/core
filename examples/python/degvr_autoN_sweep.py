"""DeGVR density_reduction (N) sweep over bulk concrete spheres of varying
optical depth, at a FIXED WALL-CLOCK budget (max_runtime) so that heavy weight-
window splitting is costed fairly. Finds the empirical best-N per model, lets us
fit tau_target, and reports how much better WW is than an equal-time analog run.

Why wall-clock, not particle count: an over-aggressive N produces windows whose
(H/F)^(N-1) extrapolation amplifies statistical noise, so every source particle
splits to the budget cap (pure waste). At equal source-history count that config
would run ~10000x longer; only an equal-TIME budget costs it fairly.

Metric `deep_score` = mean over ALL deep-shell voxels (r in [0.8R, R]) of
min(rel_err, 1.0), counting unreached voxels (flux==0) as 1.0. Lower is better
(captures reach AND statistics). Deep FOM = 1/deep_score^2 at equal time;
improvement vs analog = (deep_score_analog / deep_score_ww)^2.

Run:  python examples/python/degvr_autoN_sweep.py [--smoke]
"""
import sys
import json
import numpy as np
import yamc

DATA = "tests"
E_SRC = 14.06e6
MFP = 10.043            # cm, Sigma_t(14.06 MeV) for this concrete surrogate
MESH_N = 30
RESULTS = "degvr_sweep_results.json"   # durable (untracked)

SMOKE = "--smoke" in sys.argv
FINE = "--fine" in sys.argv
if SMOKE:
    TAUS = [8, 24]
    N_VALUES = [3, 8, 20]
    SEEDS = [1]
    PROD_T, GEN_T, AN_T = 4.0, 1.0, 4.0
elif FINE:
    # non-integer N grid, dense near the optima (N ~ 2-6), to locate each
    # model's minimum precisely now that N is continuous.
    TAUS = [8, 14, 20, 26, 32]
    # 0.25 step through N=2-5 (where every model's optimum and auto-N pick live),
    # coarse tail to show the over-split rise.
    N_VALUES = [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75,
                5.0, 6.0, 8.0]
    SEEDS = [1, 2]
    PROD_T, GEN_T, AN_T = 8.0, 4.0, 8.0
    RESULTS = "degvr_finesweep_results.json"
else:
    TAUS = [8, 14, 20, 26, 32]
    N_VALUES = [2, 3, 4, 6, 9, 14, 20, 30]
    SEEDS = [1, 2]
    PROD_T, GEN_T, AN_T = 8.0, 4.0, 8.0


def build(R):
    c = yamc.Material(composition={"O16": 0.50, "Al27": 0.20, "C12": 0.15, "Fe56": 0.15},
                      density=2.3, temperature=294)
    c.read_nuclear_data({n: f"{DATA}/{n}.arrow" for n in ("O16", "Al27", "C12", "Fe56")})
    sph = yamc.Sphere(radius=R, boundary="vacuum")
    geom = yamc.Geometry([yamc.Cell(region=sph.below, material=c)])
    src = yamc.NeutronSource(position=(0, 0, 0),
                             energy=yamc.sources.Discrete([E_SRC], [1.0]),
                             direction=yamc.sources.Isotropic())
    mesh = yamc.RegularRectangularMesh(lower_left=[-R, -R, -R], upper_right=[R, R, R],
                                       shape=[MESH_N] * 3)
    tally = yamc.Tally(scores=["flux"], name="flux", mesh=mesh, particle="neutron")
    return geom, src, mesh, tally


def deep_mask(R):
    cc = -R + (np.arange(MESH_N) + 0.5) * (2 * R / MESH_N)
    z, y, x = np.meshgrid(cc, cc, cc, indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)
    return (r > 0.8 * R) & (r <= R)


def deep_score(res, dmask):
    f = np.array(res["flux"].mean).reshape(MESH_N, MESH_N, MESH_N)
    re = np.array(res["flux"].relative_error).reshape(MESH_N, MESH_N, MESH_N)
    capped = np.where((f > 0), np.minimum(re, 1.0), 1.0)
    s = float(capped[dmask].mean())
    cov = float(np.sum((f > 0) & dmask)) / dmask.sum()
    return s, cov


def main():
    out = []
    for tau in TAUS:
        R = tau * MFP
        geom, src, mesh, tally = build(R)
        dmask = deep_mask(R)
        an_s, an_cov = [], []
        for s in SEEDS:
            m = yamc.Model(geometry=geom, source=src, tallies=[tally])
            res = m.simulate_transport(max_runtime=AN_T, seed=100 + s)
            sc, cov = deep_score(res, dmask)
            an_s.append(sc)
            an_cov.append(cov)
        an_s = float(np.mean(an_s))
        an_cov = float(np.mean(an_cov))
        rows = []
        for N in N_VALUES:
            ss, cc = [], []
            for s in SEEDS:
                m = yamc.Model(geometry=geom, source=src, tallies=[tally])
                gen = yamc.WeightWindowGeneratorDeGVR(mesh=mesh, particle="neutron",
                                                      density_reduction=float(N))
                wwb = m.generate_weight_windows(gen, max_runtime=GEN_T, seed=s)
                prod = yamc.Model(geometry=geom, source=src, tallies=[tally],
                                  variance_reduction=[wwb])
                res = prod.simulate_transport(max_runtime=PROD_T, seed=200 + s)
                sc, cov = deep_score(res, dmask)
                ss.append(sc)
                cc.append(cov)
            row = dict(N=N, score=float(np.mean(ss)), cov=float(np.mean(cc)))
            rows.append(row)
            print(f"  tau={tau:2d} N={N:5.1f}: deep_score={row['score']:.3f} cov={row['cov']:.3f}",
                  flush=True)
        best = min(rows, key=lambda r: r["score"])
        fom = (an_s / best["score"]) ** 2 if best["score"] > 0 else float("inf")
        out.append(dict(tau=tau, R=R, best_N=best["N"], best_score=best["score"],
                        best_cov=best["cov"], an_score=an_s, an_cov=an_cov,
                        fom_gain_vs_analog=fom, rows=rows))
        print(f"tau={tau:2d}: best_N={best['N']:5.1f} score={best['score']:.3f} cov={best['cov']:.3f}"
              f" | analog score={an_s:.3f} cov={an_cov:.3f} | FOM gain x{fom:.1f}", flush=True)
        with open(RESULTS, "w") as fh:      # incremental, durable
            json.dump(out, fh, indent=2)
    print("\n=== SUMMARY: best-N vs tau ===")
    print(f"{'tau':>4} {'R(cm)':>7} {'bestN':>5} {'N/tau':>6} {'WWscore':>8} {'AnScore':>8} "
          f"{'WWcov':>6} {'AnCov':>6} {'FOMx':>8}")
    for o in out:
        print(f"{o['tau']:>4} {o['R']:>7.0f} {o['best_N']:>5} {o['best_N']/o['tau']:>6.2f} "
              f"{o['best_score']:>8.3f} {o['an_score']:>8.3f} {o['best_cov']:>6.2f} "
              f"{o['an_cov']:>6.2f} {o['fom_gain_vs_analog']:>8.1f}")


def validate():
    """Phase C: compare the auto-N pick (density_reduction=None) to the swept
    best-N, per model, and report auto-N's FOM vs analog. Reads the sweep JSON."""
    sweep = {o["tau"]: o for o in json.load(open(RESULTS))}
    rows = []
    for tau in sorted(sweep):
        R = tau * MFP
        geom, src, mesh, tally = build(R)
        dmask = deep_mask(R)
        gen_auto = yamc.WeightWindowGeneratorDeGVR(mesh=mesh, particle="neutron")  # density_reduction=None
        m = yamc.Model(geometry=geom, source=src, tallies=[tally])
        auto_N, tau_est = m.estimate_density_reduction(gen_auto)
        # auto-N production (generation-inclusive time budget, same as sweep)
        ss = []
        for s in SEEDS:
            mm = yamc.Model(geometry=geom, source=src, tallies=[tally])
            wwb = mm.generate_weight_windows(gen_auto, max_runtime=GEN_T, seed=s)
            prod = yamc.Model(geometry=geom, source=src, tallies=[tally], variance_reduction=[wwb])
            res = prod.simulate_transport(max_runtime=PROD_T, seed=300 + s)
            ss.append(deep_score(res, dmask)[0])
        auto_score = float(np.mean(ss))
        o = sweep[tau]
        auto_fom = (o["an_score"] / auto_score) ** 2 if auto_score > 0 else float("inf")
        rows.append(dict(tau=tau, R=R, best_N=o["best_N"], auto_N=auto_N, tau_est=tau_est,
                         auto_score=auto_score, best_score=o["best_score"],
                         an_score=o["an_score"], auto_fom=auto_fom,
                         best_fom=o["fom_gain_vs_analog"]))
        print(f"tau={tau:2d}: best_N(sweep)={o['best_N']:5.1f}  auto_N={auto_N:4.0f} (tau_est={tau_est:5.1f})"
              f"  auto_score={auto_score:.3f} best_score={o['best_score']:.3f}"
              f"  auto FOMx{auto_fom:.1f}", flush=True)
    print("\n=== VALIDATION: auto-N vs swept best-N ===")
    print(f"{'tau':>4} {'R(cm)':>7} {'bestN':>5} {'autoN':>5} {'tauEst':>7} {'match':>6} "
          f"{'autoScore':>9} {'bestScore':>9} {'anScore':>8} {'autoFOMx':>9} {'bestFOMx':>9}")
    for r in rows:
        match = "yes" if abs(r["auto_N"] - r["best_N"]) <= max(1, 0.5 * r["best_N"]) else "NO"
        print(f"{r['tau']:>4} {r['R']:>7.0f} {r['best_N']:>5} {r['auto_N']:>5.0f} {r['tau_est']:>7.1f} "
              f"{match:>6} {r['auto_score']:>9.3f} {r['best_score']:>9.3f} {r['an_score']:>8.3f} "
              f"{r['auto_fom']:>9.1f} {r['best_fom']:>9.1f}")
    with open("degvr_validate_results.json", "w") as fh:
        json.dump(rows, fh, indent=2)


def genscan():
    """Hypothesis test: does the ideal N grow with generation effort? More gen
    particles -> cleaner (H/F)^(N-1) extrapolation -> a larger N stops being
    noise-dominated. Fix a deep tau, sweep N x generation wall-clock budget at a
    fixed production budget, and read best-N per gen budget.
    """
    tau = 26
    R = tau * MFP
    geom, src, mesh, tally = build(R)
    dmask = deep_mask(R)
    GEN_BUDGETS = [2.0, 8.0, 32.0]
    NS = [2, 3, 5, 8, 14, 22]
    PROD = 8.0
    m = yamc.Model(geometry=geom, source=src, tallies=[tally])
    an = deep_score(m.simulate_transport(max_runtime=PROD, seed=1), dmask)[0]
    print(f"tau={tau} (R={R:.0f} cm), analog deep_score={an:.3f}, prod budget={PROD}s", flush=True)
    out = {}
    for gt in GEN_BUDGETS:
        row = []
        for N in NS:
            mm = yamc.Model(geometry=geom, source=src, tallies=[tally])
            gen = yamc.WeightWindowGeneratorDeGVR(mesh=mesh, particle="neutron",
                                                  density_reduction=float(N))
            wwb = mm.generate_weight_windows(gen, max_runtime=gt, seed=1)
            prod = yamc.Model(geometry=geom, source=src, tallies=[tally], variance_reduction=[wwb])
            sc = deep_score(prod.simulate_transport(max_runtime=PROD, seed=2), dmask)[0]
            row.append(dict(N=N, score=sc))
            print(f"  gen={gt:5.0f}s N={N:5.1f}: deep_score={sc:.3f}", flush=True)
        best = min(row, key=lambda r: r["score"])
        out[gt] = dict(best_N=best["N"], best_score=best["score"], row=row)
        print(f"gen={gt:5.0f}s: best_N={best['N']:5.1f} score={best['score']:.3f}", flush=True)
        with open("degvr_genscan_results.json", "w") as fh:
            json.dump({str(k): v for k, v in out.items()}, fh, indent=2)
    print("\n=== best-N vs generation budget (tau=26) ===")
    print(f"{'gen(s)':>8} {'bestN':>6} {'bestScore':>10}")
    for gt in GEN_BUDGETS:
        print(f"{gt:>8.0f} {out[gt]['best_N']:>6} {out[gt]['best_score']:>10.3f}")


def plot():
    """Plot deep-region score vs N, one line per model (reads the fine sweep).
    Star = auto-N pick (max(2, tau/7)); open circle = empirical best-N."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = json.load(open("degvr_finesweep_results.json"))
    # validated categorical palette (dataviz skill), fixed order
    COLORS = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834"]
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for o, col in zip(data, COLORS):
        Ns = np.array([r["N"] for r in o["rows"]], dtype=float)
        sc = np.array([r["score"] for r in o["rows"]], dtype=float)
        order = np.argsort(Ns)
        Ns, sc = Ns[order], sc[order]
        ax.plot(Ns, sc, "-o", color=col, lw=2.0, ms=6, label=f"τ={o['tau']}")
        bi = int(np.argmin(sc))
        ax.plot(Ns[bi], sc[bi], "o", color=col, ms=12, mfc="none", mew=2.0)  # empirical best-N
        aN = max(2.0, o["tau"] / 7.0)
        aY = float(np.interp(aN, Ns, sc))
        ax.plot(aN, aY, "*", color=col, ms=17, mec="#0b0b0b", mew=0.6)         # auto-N
        ax.annotate(f"τ={o['tau']}", (Ns[-1], sc[-1]), color=col, fontsize=9,
                    xytext=(5, 0), textcoords="offset points", va="center")
    ax.set_yscale("log")
    ax.set_xlabel("density_reduction  N")
    ax.set_ylabel("deep-region score   (lower is better)")
    ax.set_title("DeGVR deep-region score vs N   (★ = auto-N = max(2, τ/7);  "
                 "○ = empirical best-N)", fontsize=11)
    ax.grid(True, which="both", color="#e5e5e2", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(title="optical depth", frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    out = "degvr_N_sweep.png"
    fig.savefig(out, dpi=140, facecolor="#fcfcfb")
    print(f"wrote {out}")


if __name__ == "__main__":
    if "--validate" in sys.argv:
        validate()
    elif "--genscan" in sys.argv:
        genscan()
    elif "--plot" in sys.argv:
        plot()
    else:
        main()
