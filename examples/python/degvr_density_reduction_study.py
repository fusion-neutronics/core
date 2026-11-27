"""How should DeGVR choose its density reduction N?

A machine-independent study of the cost of DeGVR weight-window generation and
of the production run it feeds, as functions of N, over many shields:

* 1D slabs: concrete, polyethylene, stainless steel, tungsten and Pb-17Li
  with a 14.06 MeV neutron source, concrete and steel at 2.45 MeV, and
  concrete and lead with photon sources, each at three thicknesses with the
  same source-energy optical depth;
* 3D models: concrete and steel spheres, a layered steel + concrete sphere
  around an air cavity, an air room inside concrete walls, and a concrete
  sphere with a photon source.

For each (model, N, seed) the two DeGVR passes are run as
``Model.generate_weight_windows`` runs them (``validate`` checks that the
windows agree on a test slab), then a production run with the windows. R is
the geometric-mean relative error over the lowest-flux quarter of the mesh,
and the cost of reaching R_target = 5% with each N is

    C(N) = generation cost + production cost * (R / R_target)^2.

Each rule for choosing N (the automatic ``max(5, tau / 7)``, the earlier
``max(2, tau / 7)``, other constants, the measured attenuation, a pilot pass)
is scored by its cost against the best N of each model, interpolating
log-log between the sampled N. An unscored voxel counts as R = 1, which the
(R / R_target)^2 scaling cannot extrapolate, so only the N whose production
scored at least 99% of that quarter in every seed are costed (the rest then
move R by a few percent at most). A rule that lands outside them is charged
30 times the best cost. Models where no N was costed, or where an N between
costed ones was not, are listed, not scored.

Machine independence:

* the passes stop on the default convergence target, not on a time budget,
  and production runs a fixed number of histories. A time guard (``--guard``)
  cuts off runs that take too long; such runs are flagged and their costs are
  lower bounds. A pass that has used more than about 2/7 of the guard gets
  smaller chunks from yamc and can stop at a different checkpoint than
  ``generate_weight_windows`` would; the report counts these;
* every job is single threaded and pinned to its own physical core (Linux);
* every cost is divided by the cost of a reference workload (full-density
  analog histories of the same model) measured in the same process, so costs
  are in analog-history equivalents and do not depend on the core type, the
  clock or the machine. On a hybrid laptop CPU the efficiency cores took 2 to
  3.7 times longer than the performance cores for the same work, so raw times
  are not comparable between runs. Loading nuclear data, which does not depend
  on N, is left out of the costs.

Usage (results are written under ``degvr_density_reduction_study/``; ``run``
resumes, skipping finished jobs):

    python degvr_density_reduction_study.py validate
    python degvr_density_reduction_study.py run --phase slabs --workers 14
    python degvr_density_reduction_study.py run --phase 3d --workers 14 \\
        --seeds 2 --prod-hist 20000 --guard 1800
    python degvr_density_reduction_study.py report

The slab phase is about 660 jobs and the 3D phase about 80; on 14 cores each
takes a few hours. Nuclear data (ENDF/B-VIII.1) is downloaded on first use.
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import random
import time
from collections import defaultdict

import numpy as np
import yamc

yamc.neutron_cross_section_data = "endf-b8.1"
yamc.photon_cross_section_data = "endf-b8.1"

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "degvr_density_reduction_study")

REL_ERR = 0.1  # the default per-pass convergence target of generate_weight_windows
RATIO, SURVIVAL, MAX_SPLIT = 5.0, 3.0, 10  # the generator's default window shape

PNNL = {
    "concrete": "Concrete, Ordinary (NIST)",
    "steel": "Steel, Stainless 316",
    "tungsten": "Tungsten",
    "polyethylene": "Polyethylene, Non-borated",
    "lead": "Lead",
    "air": "Air (dry, near sea level)",
}
LIPB = ({"Pb": 0.83, "Li": 0.17}, 9.4)  # Pb-17Li eutectic, atom fractions, g/cm3

_REF_DENSITY = {}


def make_material(key, factor):
    """Material `key` at `factor` times its reference density."""
    if key == "lipb":
        comp, rho = LIPB
        return yamc.Material(composition=comp, density=rho * factor)
    name = PNNL[key]
    if name not in _REF_DENSITY:
        _REF_DENSITY[name] = yamc.materials.pnnl.material(name).density
    return yamc.materials.pnnl.material(name, density=_REF_DENSITY[name] * factor)


def plane(axis, offset, vacuum=False):
    return yamc.Plane(axis=axis, offset=offset, boundary="vacuum" if vacuum else None)


def make_source(particle, energy, position):
    cls = yamc.NeutronSource if particle == "neutron" else yamc.PhotonSource
    return cls(position=position, energy=energy, direction=yamc.sources.Isotropic())


# --------------------------------------------------------------------------
# Models. Each builder returns (geometry, source, window mesh, mask, target
# cell) at `factor` times every density; mask marks the window-mesh voxels
# inside the geometry, the voxels the global objective is judged on.
# --------------------------------------------------------------------------


def build_slab(spec, factor):
    """Laterally wide slab with a planar isotropic source on its front face and
    a 1D mesh along depth: the depth physics of one material and energy."""
    T, L, nz = spec["T"], spec["L"], spec["nz"]
    dz = T / nz
    mat = make_material(spec["mat"], factor)
    lat = (plane("x", -L, True).above & plane("x", L, True).below
           & plane("y", -L, True).above & plane("y", L, True).below)
    z0, zt, z1 = plane("z", 0.0, True), plane("z", T - dz), plane("z", T, True)
    body = yamc.Cell(region=lat & z0.above & zt.below, material=mat, name="body")
    target = yamc.Cell(region=lat & zt.above & z1.below, material=mat, name="target")
    if spec.get("_estimate"):
        # The auto-N ray trace starts from one sampled source point; on a wide
        # planar source that point is off axis and every ray to the on-axis
        # plate centres is oblique. The face centre gives the true depth.
        position = (0.0, 0.0, 1e-3)
    else:
        face = yamc.RegularRectangularMesh(
            lower_left=[-L, -L, 0.0], upper_right=[L, L, 1e-3], shape=[1, 1, 1])
        position = yamc.sources.MeshDistribution(face)
    mesh = yamc.RegularRectangularMesh(
        lower_left=[-L, -L, 0.0], upper_right=[L, L, T], shape=[1, 1, nz])
    return (yamc.Geometry([body, target]), make_source(spec["particle"], spec["energy"], position),
            mesh, np.ones(nz, dtype=bool), target)


def build_sphere(spec, factor):
    """Point source at the centre of concentric shells (material, outer
    radius); the target is the outermost `target_dr` of the last shell."""
    layers, n, dr = spec["layers"], spec["n"], spec["target_dr"]
    R = layers[-1][1]
    cells, inner, target = [], None, None
    for i, (key, r) in enumerate(layers):
        mat = make_material(key, factor)
        if i == len(layers) - 1:
            s_t, s_o = yamc.Sphere(radius=r - dr), yamc.Sphere(radius=r, boundary="vacuum")
            reg = s_t.below if inner is None else inner.above & s_t.below
            cells.append(yamc.Cell(region=reg, material=mat, name="body"))
            target = yamc.Cell(region=s_t.above & s_o.below, material=mat, name="target")
            cells.append(target)
        else:
            s = yamc.Sphere(radius=r)
            reg = s.below if inner is None else inner.above & s.below
            cells.append(yamc.Cell(region=reg, material=mat, name="layer%d" % i))
            inner = s
    mesh = yamc.RegularRectangularMesh(lower_left=[-R] * 3, upper_right=[R] * 3, shape=[n] * 3)
    c = -R + (np.arange(n) + 0.5) * (2 * R / n)
    z, y, x = np.meshgrid(c, c, c, indexing="ij")
    mask = (np.sqrt(x * x + y * y + z * z) < R).ravel()
    return (yamc.Geometry(cells), make_source(spec["particle"], spec["energy"], (0.0, 0.0, 0.0)),
            mesh, mask, target)


def build_room(spec, factor):
    """Air room (half size a) inside walls of thickness w, point source at the
    centre, target an air layer of thickness g outside the +x wall. The air
    does not shield, so the flux falls partly by geometry, which density does
    not change."""
    a, w, g, n = spec["a"], spec["w"], spec["g"], spec["n"]
    wall, air = make_material(spec["mat"], factor), make_material("air", factor)
    b = a + w
    x_lo, x_wall, x_hi = plane("x", -b, True), plane("x", b), plane("x", b + g, True)
    yz = (plane("y", -b, True).above & plane("y", b, True).below
          & plane("z", -b, True).above & plane("z", b, True).below)
    cav = [plane(ax, v) for ax in "xyz" for v in (-a, a)]
    inner_yz = cav[2].above & cav[3].below & cav[4].above & cav[5].below
    cavity = cav[0].above & cav[1].below & inner_yz
    target = yamc.Cell(region=x_wall.above & x_hi.below & inner_yz, material=air, name="target")
    cells = [
        yamc.Cell(region=cavity, material=air, name="cavity"),
        yamc.Cell(region=x_lo.above & x_wall.below & yz & ~cavity, material=wall, name="walls"),
        target,
        yamc.Cell(region=x_wall.above & x_hi.below & yz & ~inner_yz, material=air, name="outside"),
    ]
    mesh = yamc.RegularRectangularMesh(lower_left=[-b, -b, -b], upper_right=[b + g, b, b], shape=n)
    return (yamc.Geometry(cells), make_source(spec["particle"], spec["energy"], (0.0, 0.0, 0.0)),
            mesh, np.ones(int(np.prod(n)), dtype=bool), target)


BUILDERS = {"slab": build_slab, "sphere": build_sphere, "room": build_room}


def build(spec, factor, windows=None, target_tally=True):
    geom, src, mesh, mask, target = BUILDERS[spec["kind"]](spec, factor)
    p = spec["particle"]
    tallies = [yamc.Tally(scores=["flux"], name="flux", mesh=mesh, particle=p)]
    if target_tally:
        tallies.append(yamc.Tally(scores=["flux"], name="target", cells=target, particle=p))
    model = yamc.Model(geometry=geom, source=src, tallies=tallies,
                       transport_secondary_photons=(p == "photon"),
                       variance_reduction=windows or [], verbose=[])
    return model, mesh, mask


# Slab cases: (material, particle, source energy in eV, Sigma_t at the source
# energy in 1/cm from the code's own ray trace). The slab thickness is set from
# Sigma_t so every case has the same optical depth: the automatic rule then
# picks the same N for all of them.
SLAB_CASES = [
    ("concrete", "neutron", 14.06e6, 0.131656),
    ("steel", "neutron", 14.06e6, 0.223086),
    ("tungsten", "neutron", 14.06e6, 0.339153),
    ("polyethylene", "neutron", 14.06e6, 0.107285),
    ("lipb", "neutron", 14.06e6, 0.152409),
    ("concrete", "neutron", 2.45e6, 0.173444),
    ("steel", "neutron", 2.45e6, 0.385789),
    ("concrete", "photon", 1.25e6, 0.133402),
    ("lead", "photon", 1.25e6, 0.664453),
    ("concrete", "photon", 6.0e6, 0.0618803),
]
SLAB_TAUS = [12, 24, 36]  # nominal optical depths (one mesh plate per mean free path)


def case_label(mat, particle, energy):
    return "%s-%s%g" % (mat, "n" if particle == "neutron" else "g", energy / 1e6)


def all_specs():
    specs = []
    for mat, particle, energy, sigma in SLAB_CASES:
        lab = case_label(mat, particle, energy)
        for tau in SLAB_TAUS:
            T = round(tau / sigma, 1)
            specs.append(dict(name="slab-%s-t%d" % (lab, tau), kind="slab", mat=mat,
                              particle=particle, energy=energy, T=T, L=max(T, 100.0),
                              nz=tau, case=lab))
    n14 = dict(particle="neutron", energy=14.06e6, n=14)
    specs += [
        dict(name="sphere-concrete-n14", kind="sphere", layers=[("concrete", 130.0)],
             target_dr=10.0, case="concrete-n14.06", **n14),
        dict(name="sphere-steel-n14", kind="sphere", layers=[("steel", 100.0)],
             target_dr=8.0, case="steel-n14.06", **n14),
        dict(name="layered-steel-concrete-n14", kind="sphere",
             layers=[("air", 60.0), ("steel", 90.0), ("concrete", 190.0)],
             target_dr=10.0, case="steel+concrete-n14.06", **n14),
        dict(name="room-concrete-n14", kind="room", particle="neutron", energy=14.06e6,
             mat="concrete", a=100.0, w=120.0, g=20.0, n=[15, 14, 14], case="concrete-n14.06"),
        dict(name="sphere-concrete-g1.25", kind="sphere", particle="photon", energy=1.25e6,
             layers=[("concrete", 120.0)], n=14, target_dr=10.0, case="concrete-g1.25"),
    ]
    return {s["name"]: s for s in specs}


N_GRID = [2.0, 2.83, 4.0, 5.66, 8.0, 11.3, 16.0, 22.6, 32.0]


def auto_n(spec):
    """The automatic (N, tau) the code would choose for this model."""
    model, mesh, _ = build(dict(spec, _estimate=True), 1.0, target_tally=False)
    gen = yamc.WeightWindowGeneratorDeGVR(mesh=mesh, particle=spec["particle"])
    return model.estimate_density_reduction(gen)


def n_values(n_auto, tau):
    """The N grid up to tau / 1.5 (thick concrete already loses the deep region
    near tau / 3), plus the automatic N and the earlier automatic N,
    max(2, tau / 7)."""
    grid = {n for n in N_GRID if n <= max(tau / 1.5, 2.0) + 1e-9}
    return sorted(grid | {round(n_auto, 2), round(max(2.0, tau / 7.0), 2)})


# --------------------------------------------------------------------------
# DeGVR passes, as Model.generate_weight_windows runs them
# --------------------------------------------------------------------------


def bounds_from_field(field, mesh, particle):
    """generate_weight_windows' window from a flux field."""
    f = np.asarray(field, dtype=float)
    ok = np.isfinite(f) & (f > 0)
    lower = np.full(f.shape, -1.0)
    if ok.any():
        lower[ok] = f[ok] * (1.0 / (SURVIVAL * f[ok].max()))  # yamc's arithmetic exactly
    upper = np.where(lower > 0, RATIO * lower, -1.0)
    return yamc.WeightWindowBounds(mesh=mesh, lower_bounds=lower.tolist(),
                                   upper_bounds=upper.tolist(), particle=particle,
                                   survival_factor=SURVIVAL, max_split=MAX_SPLIT)


def extrapolate(F, H, n):
    """G = F (H / F)^(N - 1), 0 where either field is empty."""
    F, H = np.asarray(F, dtype=float), np.asarray(H, dtype=float)
    G = np.zeros_like(F)
    ok = (F > 0) & (H > 0)
    with np.errstate(over="ignore"):
        g = np.exp(np.log(F[ok]) + (n - 1.0) * (np.log(H[ok]) - np.log(F[ok])))
    G[ok] = np.where(np.isfinite(g), g, 0.0)
    return G


def transport_cpu(res, t0):
    """CPU seconds since t0, less the run's nuclear-data loading (which does not
    depend on N). The load time is wall time, equal to its CPU time on a pinned
    core with the data cached; floored at zero in case it was not."""
    return max(time.process_time() - t0 - (res.data_load_time or 0.0), 0.0)


def on_schedule(hist):
    """Whether a pass stopped at a checkpoint of yamc's default convergence
    schedule (chunks of a quarter of the histories so far, 100 to 100,000). A
    max_runtime guard shrinks the chunks once time runs short."""
    done = 0
    while done < hist:
        done += min(max(100, done // 4), 100_000)
    return done == hist


def run_pass(spec, factor, seed, guard_s, windows=None):
    """One pass with the default convergence stop and no history budget, so
    the convergence checks fall where generate_weight_windows puts them until
    the guard shrinks the chunks. A pass cut off by the guard is flagged."""
    model, mesh, mask = build(spec, factor, windows=windows, target_tally=False)
    model.convergence_targets = [
        yamc.ConvergenceTarget("bin_relative_error", REL_ERR, tally="flux")]
    t0 = time.process_time()
    res = model.simulate_transport(seed=seed, threads=1, max_runtime=guard_s)
    cpu = transport_cpu(res, t0)
    mean = np.asarray(res["flux"].mean)
    return dict(mean=mean, rel=np.asarray(res["flux"].relative_error),
                hist=int(res.runs[0]["n_histories"]), cpu=cpu,
                load_s=res.data_load_time, capped=res.transport_time >= guard_s,
                reached=float(np.mean((mean > 0)[mask]))), mesh


def generate(spec, n, seed, guard_s):
    fid, mesh = run_pass(spec, 1.0 / n, seed, guard_s)
    fid_ww = bounds_from_field(fid["mean"], mesh, spec["particle"])
    asy, _ = run_pass(spec, 2.0 / n, seed, guard_s, windows=[fid_ww])
    return fid, asy, extrapolate(fid["mean"], asy["mean"], n), mesh


def reference_cost(spec, n_hist=4000):
    """CPU seconds per full-density analog history of this model on the core
    this process runs on: the unit every cost is expressed in."""
    model, _, _ = build(spec, 1.0, target_tally=False)
    model.simulate_transport(total_particles=200, seed=999, threads=1)  # warm caches
    t0 = time.process_time()
    model.simulate_transport(total_particles=n_hist, seed=998, threads=1)
    return (time.process_time() - t0) / n_hist


def quantile_err(rel, mean, mask, q):
    return float(np.quantile(np.where(mean > 0, np.minimum(rel, 1.0), 1.0)[mask], q))


def one_cpu_per_core():
    """One logical CPU per physical core among those this process may use
    (Linux), so no two jobs share a core; None elsewhere."""
    if not hasattr(os, "sched_getaffinity"):
        return None
    cores = {}
    for c in sorted(os.sched_getaffinity(0)):
        p = "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list" % c
        if os.path.exists(p):
            cores.setdefault(open(p).read().strip(), c)
    return sorted(cores.values()) or None


def run_job(job):
    """One (model, N, seed): generate windows, then a production run."""
    spec, n, seed, out_json = job["spec"], job["N"], job["seed"], job["out"]
    cpu_q = job.pop("cpu_queue")
    cpu_id = cpu_q.get() if cpu_q is not None else None
    if cpu_id is not None:
        os.sched_setaffinity(0, {cpu_id})
    # Generation cost grows steeply as N falls: once a larger N of this model and
    # seed needed half the guard for a pass, a smaller N cannot finish inside it.
    for other in glob.glob(os.path.join(RESULTS, "%s__N*__s%d.json" % (spec["name"], seed))):
        r = json.load(open(other))
        if r["N"] > n and (r.get("skipped") or r["fid"]["capped"] or r["asy"]["capped"]
                           or max(r["fid"]["cpu"], r["asy"]["cpu"]) > job["gen_guard_s"] / 2):
            json.dump(dict(model=spec["name"], N=n, seed=seed, skipped=True,
                           because=os.path.basename(other)), open(out_json, "w"))
            if cpu_q is not None:
                cpu_q.put(cpu_id)
            return out_json
    ref0 = reference_cost(spec)
    fid, asy, G, mesh = generate(spec, n, seed, job["gen_guard_s"])
    model, _, mask = build(spec, 1.0, windows=[bounds_from_field(G, mesh, spec["particle"])])
    t0 = time.process_time()
    res = model.simulate_transport(total_particles=job["prod_hist"], seed=10_000 + seed,
                                   threads=1, max_runtime=job["prod_guard_s"])
    prod_cpu = transport_cpu(res, t0)
    ref = 0.5 * (ref0 + reference_cost(spec))
    fm = np.asarray(res["flux"].mean)
    fr = np.asarray(res["flux"].relative_error)
    keys = ("hist", "cpu", "load_s", "capped", "reached")
    rec = dict(
        model=spec["name"], case=spec["case"], kind=spec["kind"], N=n, seed=seed,
        N_auto=job["N_auto"], tau=job["tau"], cpu_id=cpu_id, ref_cpu_per_hist=ref,
        fid={k: fid[k] for k in keys}, asy={k: asy[k] for k in keys},
        gen_ahe=(fid["cpu"] + asy["cpu"]) / ref, prod_ahe=prod_cpu / ref,
        prod=dict(hist=int(res.runs[0]["n_histories"]), cpu=prod_cpu,
                  load_s=res.data_load_time,
                  capped=int(res.runs[0]["n_histories"]) < job["prod_hist"],
                  R90=quantile_err(fr, fm, mask, 0.9), cover=float(np.mean((fm > 0)[mask])),
                  target_rel=float(np.asarray(res["target"].relative_error)[0])),
    )
    np.savez_compressed(out_json.replace(".json", ".npz"), F=fid["mean"], Frel=fid["rel"],
                        H=asy["mean"], Hrel=asy["rel"], G=G, prod_mean=fm, prod_rel=fr, mask=mask)
    json.dump(rec, open(out_json + ".tmp", "w"), indent=1)
    os.replace(out_json + ".tmp", out_json)
    if cpu_q is not None:
        cpu_q.put(cpu_id)
    return out_json


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

R_TARGET = 0.05  # precision the production run is costed to
COVER_MIN = 0.99  # share of the lowest-flux quarter production must score to be costed
REGRET_CAP = 30.0  # charged where a rule lands outside the costed N
FLOORS = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0)  # floors f of max(f, tau / 7) the report compares


def load():
    by_model = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        r = json.load(open(f))
        if r.get("skipped"):
            continue
        z = np.load(f.replace(".json", ".npz"))
        r["_npz"] = {k: z[k] for k in z.files}
        by_model[r["model"]].append(r)
    return by_model


def summarise(runs, r_target=R_TARGET):
    """Reference field (inverse-variance combination of every production run,
    unbiased whatever the windows), Lambda90 = e-fold drop from its peak to its
    90th-percentile voxel, and the cost curve over the costed N: those whose
    production scored at least COVER_MIN of the lowest-flux quarter in every
    seed. The best N is taken over those whose generation finished inside the
    guard, where any did."""
    mask = runs[0]["_npz"]["mask"].astype(bool)
    num = den = 0.0
    for r in runs:
        m, rel = r["_npz"]["prod_mean"], r["_npz"]["prod_rel"]
        ok = (m > 0) & (rel > 0)
        w = np.where(ok, 1.0 / np.where(ok, rel * m, 1.0) ** 2, 0.0)
        num, den = num + w * np.where(ok, m, 0.0), den + w
    ref = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    v = ref[mask & (ref > 0)]
    deep = mask & (ref > 0) & (ref <= np.quantile(v, 0.25))
    by_n = defaultdict(list)
    for r in runs:
        pm, pr = r["_npz"]["prod_mean"], r["_npz"]["prod_rel"]
        rr = np.where(pm > 0, np.minimum(pr, 1.0), 1.0)[deep]
        r["Rdeep"] = float(np.exp(np.mean(np.log(np.maximum(rr, 1e-6)))))
        r["cover"] = float(np.mean(pm[deep] > 0))
        by_n[round(r["N"], 2)].append(r)
    ns = sorted(by_n)
    rows = []
    for n in ns:
        rs = by_n[n]
        if not all(r["cover"] >= COVER_MIN for r in rs):
            continue
        gen = np.mean([r["gen_ahe"] for r in rs])
        prod = np.exp(np.mean([np.log(r["prod_ahe"] * (r["Rdeep"] / r_target) ** 2) for r in rs]))
        rows.append(dict(N=n, gen=gen, cost=gen + prod,
                         capped=any(r["fid"]["capped"] or r["asy"]["capped"] for r in rs)))
    # The cost curve must not be interpolated across an N that was not costed.
    if rows and len(rows) != ns.index(rows[-1]["N"]) - ns.index(rows[0]["N"]) + 1:
        rows = []
    best = min([r for r in rows if not r["capped"]] or rows, key=lambda r: r["cost"],
               default=None)
    return dict(model=runs[0]["model"], kind=runs[0]["kind"], tau=runs[0]["tau"],
                N_auto=runs[0]["N_auto"], L90=float(np.log(v.max() / np.quantile(v, 0.10))),
                rows=rows, best=best)


def sampled(m, n):
    """Whether N lies within the costed N of this model (rounded to 0.01)."""
    ns = [r["N"] for r in m["rows"]]
    return min(ns) * 0.99 <= n <= max(ns) * 1.01


def cost_at(m, n):
    """Cost at N, log-log interpolated between costed N; the cap outside."""
    ns = np.array([r["N"] for r in m["rows"]])
    cs = np.array([r["cost"] for r in m["rows"]])
    if not sampled(m, n):
        return REGRET_CAP * m["best"]["cost"]
    return float(np.exp(np.interp(np.log(n), np.log(ns), np.log(cs))))


def regret(models, rule):
    """Geometric mean and worst of cost(rule) / best cost, over models. A ratio
    below 1, from a rule landing on a guarded N whose cost is a lower bound, is
    counted as 1."""
    v = np.array([np.clip(cost_at(m, max(2.0, rule(m))) / m["best"]["cost"], 1.0, REGRET_CAP)
                  for m in models])
    return float(np.exp(np.mean(np.log(v)))), float(v.max())


def pilot_regret(data, models, k, pilot_ratio=3.0):
    """A pilot DeGVR pair at N_p ~ tau / pilot_ratio, Lambda from its
    extrapolated field, then N = Lambda / k; the pilot's cost is added."""
    v = []
    for m in models:
        runs = data[m["model"]]
        ns = sorted({round(r["N"], 2) for r in runs})
        n_p = min(ns, key=lambda n: abs(np.log(n / max(2.0, m["tau"] / pilot_ratio))))
        pil = [r for r in runs if round(r["N"], 2) == n_p]
        lam = []
        for r in pil:
            G = r["_npz"]["G"]
            g = G[r["_npz"]["mask"].astype(bool) & (G > 0)]
            lam.append(np.log(g.max() / np.quantile(g, 0.10)))
        c = cost_at(m, max(2.0, np.mean(lam) / k)) + np.mean([r["gen_ahe"] for r in pil])
        v.append(np.clip(c / m["best"]["cost"], 1.0, REGRET_CAP))
    v = np.array(v)
    return float(np.exp(np.mean(np.log(v)))), float(v.max())


def floor_scan(models):
    """max(f, tau / 7) for each floor in FLOORS, over the models whose costed N
    span every floor (the thinnest slabs stop at N = 5.66), so no floor is
    charged the cap for an N that was not run."""
    ms = [m for m in models if all(sampled(m, max(f, m["tau"] / 7.0)) for f in FLOORS)]
    if not ms:
        return
    cells = ["%g: %.2f (%.1f)" % ((f,) + regret(ms, lambda m, f=f: max(f, m["tau"] / 7.0)))
             for f in FLOORS]
    print("  floor f of max(f, tau / 7), over the %d models whose costed N span every f:"
          % len(ms))
    print("    " + ", ".join(cells))


def cost_law(data):
    """Per-model slope of ln(histories) of each pass against its span: the
    e-folds the first pass's flux F falls across the mesh (to its 5th-percentile
    voxel), and for the second pass the e-fold range of H / F, the part of the
    fall that the first pass's windows leave to it."""
    out = {"fid": [], "asy": []}
    for runs in data.values():
        xs = {"fid": [], "asy": []}
        ys = {"fid": [], "asy": []}
        for r in runs:
            if r["fid"]["capped"] or r["asy"]["capped"]:
                continue
            z = r["_npz"]
            ok = z["mask"].astype(bool) & (z["F"] > 0) & (z["H"] > 0)
            F, q = z["F"][ok], z["H"][ok] / z["F"][ok]
            xs["fid"].append(np.log(F.max() / np.quantile(F, 0.05)))
            xs["asy"].append(np.log(q.max() / np.quantile(q, 0.05)))
            for k in ("fid", "asy"):
                ys[k].append(np.log(r[k]["hist"]))
        for k in ("fid", "asy"):
            if len(xs[k]) >= 4 and np.ptp(xs[k]) > 1.0:
                out[k].append(np.polyfit(xs[k], ys[k], 1)[0])
    return {k: (float(np.median(v)), float(min(v)), float(max(v)), len(v))
            for k, v in out.items() if v}


def noise_check(data):
    """Seed-to-seed variance of ln G in each voxel against the variance
    (N-2)^2 eF^2 + (N-1)^2 eH^2 predicted from the passes' relative errors eF
    and eH, as if the two passes' errors were independent. In each (model, N)
    cell both are averaged over voxels, so their ratio is unbiased with two or
    three seeds; the root of its mean over cells is returned."""
    obs, pred = [], []
    for runs in data.values():
        by_n = defaultdict(list)
        for r in runs:
            by_n[round(r["N"], 2)].append(r)
        for n, rs in by_n.items():
            if len(rs) < 2:
                continue
            Gs = np.array([r["_npz"]["G"] for r in rs])
            ok = rs[0]["_npz"]["mask"].astype(bool) & np.all(Gs > 0, axis=0)
            if ok.sum() < 3:
                continue
            L = np.log(Gs[:, ok])
            eF = np.mean([r["_npz"]["Frel"][ok] for r in rs], axis=0)
            eH = np.mean([r["_npz"]["Hrel"][ok] for r in rs], axis=0)
            obs.append(np.mean(np.var(L, axis=0, ddof=1)))
            pred.append(np.mean((n - 2) ** 2 * eF ** 2 + (n - 1) ** 2 * eH ** 2))
    o, p = np.array(obs), np.array(pred)
    return (float(np.sqrt(np.mean(o / p))), float(np.corrcoef(np.log(o), np.log(p))[0, 1]),
            len(o))


def cmd_report(_args):
    data = load()
    models = [summarise(v) for _, v in sorted(data.items())]
    scored = [m for m in models if m["best"] is not None]
    new, old = (lambda m: max(5.0, m["tau"] / 7.0)), (lambda m: max(2.0, m["tau"] / 7.0))
    print("%-32s %5s %5s %6s %6s  cost / best: %s, %s" % (
        "model", "tau", "L90", "tau/L", "N_best", "max(5, tau / 7)", "max(2, tau / 7)"))
    for m in scored:
        print("%-32s %5.1f %5.1f %6.2f %6.2f  %15.2f  %15.2f" % (
            m["model"], m["tau"], m["L90"], m["tau"] / m["L90"], m["best"]["N"],
            max(1.0, cost_at(m, new(m)) / m["best"]["cost"]),
            max(1.0, cost_at(m, old(m)) / m["best"]["cost"])))
    unscored = [m["model"] for m in models if m["best"] is None]
    if unscored:
        print("Not scored (no N, or an N between costed ones, scored %g%% of the lowest-flux "
              "quarter in every seed): %s" % (100 * COVER_MIN, ", ".join(unscored)))
    runs = [r for v in data.values() for r in v]
    capped = sum(1 for r in runs if r["fid"]["capped"] or r["asy"]["capped"])
    off = sum(1 for r in runs for p in (r["fid"], r["asy"])
              if not p["capped"] and not on_schedule(p["hist"]))
    cut = sum(1 for v in data.values() for r in v
              if r["prod"]["hist"] < max(x["prod"]["hist"] for x in v))
    print("%d runs: generation stopped short of convergence in %d (their costs are lower "
          "bounds); %d other passes stopped off the default checkpoint schedule; %d production "
          "runs cut short by the guard" % (len(runs), capped, off, cut))
    for kind, sel in (("1D slabs", lambda m: m["kind"] == "slab"),
                      ("3D models", lambda m: m["kind"] != "slab"),
                      ("All models", lambda m: True)):
        ms = [m for m in scored if sel(m)]
        if not ms:
            continue
        print("\n%s (%d models): cost of each rule / best cost, geometric mean (worst)" % (
            kind, len(ms)))
        rows = [("max(5, tau / 7) (automatic)", new), ("max(2, tau / 7) (before)", old),
                ("max(4, tau / 7)", lambda m: max(4.0, m["tau"] / 7.0))]
        rows += [("tau / %g" % k, lambda m, k=k: m["tau"] / k) for k in (4.0, 5.0)]
        rows += [("Lambda90 / %g (oracle)" % k, lambda m, k=k: m["L90"] / k) for k in (3.0, 4.0)]
        for name, rule in rows:
            print("  %-28s %.2f (%.1f)" % ((name,) + regret(ms, rule)))
        for k in (2.0, 3.0):
            print("  %-28s %.2f (%.1f)" % (("pilot, Lambda / %g" % k,) + pilot_regret(data, ms, k)))
        floor_scan(ms)
    print("\nAll models, production costed to other targets: geometric mean (worst)")
    for rt in (0.10, 0.02):
        ms = [m for m in (summarise(v, rt) for _, v in sorted(data.items())) if m["best"]]
        print("  %2.0f%%: max(5, tau / 7) %.2f (%.1f), max(2, tau / 7) %.2f (%.1f)"
              % ((100 * rt,) + regret(ms, new) + regret(ms, old)))
    law = cost_law(data)
    print("\nPass histories against span: slope of ln(histories) per e-fold, median (range)")
    names = {"fid": "first pass, span of F", "asy": "second pass, span of H / F"}
    for k, (s, lo, hi, n) in law.items():
        print("  %-28s %.2f (%.2f to %.2f) over %d models" % (names[k], s, lo, hi, n))
    ratio, corr, cells = noise_check(data)
    print("Window noise, seed-to-seed scatter of ln G / predicted: %.2f, "
          "log correlation %.2f over %d (model, N) cells" % (ratio, corr, cells))


def cmd_validate(_args):
    """The passes here must give the windows generate_weight_windows gives."""
    spec = dict(kind="slab", mat="concrete", particle="neutron", energy=14.06e6,
                T=120.0, L=120.0, nz=15, case="v")
    n, seed = 3.0, 7
    _, _, G, mesh = generate(spec, n, seed, 1e6)
    mine = np.asarray(bounds_from_field(G, mesh, "neutron").lower_bounds)
    model, mesh2, _ = build(spec, 1.0, target_tally=False)
    gen = yamc.WeightWindowGeneratorDeGVR(mesh=mesh2, particle="neutron", density_reduction=n)
    ref = np.asarray(model.generate_weight_windows(gen, seed=seed, threads=1).lower_bounds)
    diff = float(np.max(np.abs(np.log(mine / ref))))
    print("max |log(window ratio)| against generate_weight_windows: %g" % diff)
    # numpy's exp and log may round differently from the C library's by an ulp
    if not diff <= 1e-12:  # also fails on NaN, where one side has no window
        raise SystemExit("the study's passes do not reproduce generate_weight_windows")


def cmd_run(args):
    os.makedirs(RESULTS, exist_ok=True)
    specs = [s for s in all_specs().values()
             if (s["kind"] == "slab") == (args.phase == "slabs")]
    jobs = []
    for s in specs:
        n_auto, tau = auto_n(s)
        for n in n_values(n_auto, tau):
            for seed in range(1, args.seeds + 1):
                out = os.path.join(RESULTS, "%s__N%.2f__s%d.json" % (s["name"], n, seed))
                if not os.path.exists(out):
                    jobs.append(dict(spec=s, N=n, seed=seed, out=out, N_auto=n_auto, tau=tau,
                                     gen_guard_s=args.guard, prod_hist=args.prod_hist,
                                     prod_guard_s=args.guard))
    # Largest N first within each model and seed (the cheap ones, which the
    # skip rule reads), interleaved across models to mix load conditions.
    rng = random.Random(0)
    groups = defaultdict(list)
    for j in jobs:
        groups[(j["spec"]["name"], j["seed"])].append(j)
    ranked = [(rank, rng.random(), j) for g in groups.values()
              for rank, j in enumerate(sorted(g, key=lambda j: -j["N"]))]
    jobs = [j for _, _, j in sorted(ranked, key=lambda t: (t[0], t[1]))]
    print("%d jobs" % len(jobs), flush=True)
    ctx = mp.get_context("spawn")
    cpus = one_cpu_per_core()
    cpu_q = None
    workers = args.workers
    if cpus is not None:
        workers = min(workers, len(cpus))  # one job per physical core
        cpu_q = ctx.Manager().Queue()
        for c in cpus[:workers]:
            cpu_q.put(c)
    for j in jobs:
        j["cpu_queue"] = cpu_q
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        for i, out in enumerate(pool.imap_unordered(run_job, jobs), 1):
            print("[%d/%d] %s" % (i, len(jobs), os.path.basename(out)), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["validate", "run", "report"])
    ap.add_argument("--phase", choices=["slabs", "3d"], default="slabs")
    ap.add_argument("--workers", type=int,
                    default=len(one_cpu_per_core() or []) or max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--prod-hist", type=int, default=200_000)
    ap.add_argument("--guard", type=float, default=600.0,
                    help="seconds after which a pass or production run is cut off (flagged; "
                         "its cost is then a lower bound)")
    args = ap.parse_args()
    {"validate": cmd_validate, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
