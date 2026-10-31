"""Imprinted against non-imprinted meshes of the same CAD: meshing time,
transport speed and agreement.

Every model is meshed twice, once with ``CadToYamc.mesh(imprint=True)``
(tracked conformally) and once with ``mesh(imprint=False)`` (tracked
discontinuously). Both meshes are then transported from the same 14 MeV point
source, twice each:

- **with the same seed**: where the two meshes coincide the histories run in
  lockstep, so the difference (in standard deviations of each tally) is the
  meshes, not the noise;
- **with independent seeds**: the worst |z| over the per-solid flux tallies.

Models:

==========  ===========================================================
cylinders   ten nested annuli whose seams are rotated, so every shared
            cylindrical face is faceted differently on either side
spheres     twenty nested sphere shells (seams aligned, so the imprint
            meshes them)
cubes       a 6x6x6 grid of touching cubes
==========  ===========================================================

Usage::

    python tools/bench_discontinuous_mesh.py                 # all models
    python tools/bench_discontinuous_mesh.py cylinders --particles 1000000
    python tools/bench_discontinuous_mesh.py --density 0.1   # crossing dominated

Needs cadquery and the nuclear data under ``tests/`` (run from the repo root).
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
import time

import cadquery as cq
import yamc
from yamc.cad import CadToYamc

NUCLIDES = [("Be9", 1.85), ("Fe56", 7.874), ("Pb208", 11.35)]


def cylinders(n=10, dr=3.0, h=60.0):
    asm = cq.Assembly()
    for i in range(1, n + 1):
        wp = cq.Workplane().circle(dr * i)
        if i > 1:
            wp = wp.circle(dr * (i - 1))
        body = wp.extrude(h).translate((0, 0, -h / 2))
        if i > 1:
            body = body.rotate((0, 0, 0), (0, 0, 1), 7.3 * i)
        asm.add(body, name=f"c{i}")
    return asm, (0.0, 0.0, 0.0)


def spheres(n=20, dr=5.0):
    asm = cq.Assembly()
    for i in range(1, n + 1):
        body = cq.Workplane().sphere(dr * i)
        if i > 1:
            body = body.cut(cq.Workplane().sphere(dr * (i - 1)))
        asm.add(body, name=f"s{i}")
    return asm, (0.0, 0.0, 0.0)


def cubes(n=6, a=10.0):
    asm = cq.Assembly()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                centre = ((i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a)
                asm.add(cq.Workplane().box(a, a, a).translate(centre), name=f"g{i}_{j}_{k}")
    return asm, (31.0, 29.0, 33.0)


MODELS = {"cylinders": cylinders, "spheres": spheres, "cubes": cubes}


def materials(names, density, data_dir):
    out = {}
    for i, name in enumerate(names):
        nuclide, rho = NUCLIDES[i % len(NUCLIDES)]
        m = yamc.Material(
            composition={nuclide: 1.0}, density=rho * density, name=name, temperature=294
        )
        m.read_nuclear_data({nuclide: os.path.join(data_dir, f"{nuclide}.arrow")})
        out[name] = m
    return out


def mesh(build, imprint, tolerance, out_dir):
    asm, source = build()
    names = [child.name for child in asm.children]
    c2y = CadToYamc()
    c2y.add_cadquery_object(asm, material_tags=names)
    start = time.perf_counter()
    c2y.mesh(tolerance=tolerance, imprint=imprint)
    seconds = time.perf_counter() - start
    path = os.path.join(out_dir, f"{'imprinted' if imprint else 'separate'}.arrow")
    c2y.to_arrow(path)
    return path, names, source, seconds


def transport(path, names, source, seed, args):
    mats = materials(names, args.density, args.data)
    geometry = yamc.MeshGeometry(path, mats, graveyard_offset=1.0)
    tallies = {n: yamc.Tally(scores=["flux"], materials=m, name=n) for n, m in mats.items()}
    model = yamc.Model(
        geometry=geometry,
        tallies=list(tallies.values()),
        max_lost_particles=10**9,
        source=yamc.NeutronSource(
            energy=yamc.sources.Discrete([14.06e6], [1.0]), position=source
        ),
    )
    start = time.perf_counter()
    results = model.simulate_transport(total_particles=args.particles, seed=seed)
    rate = args.particles / (time.perf_counter() - start)
    flux = {
        n: (float(results[t].mean[0]), float(results[t].standard_deviation[0]))
        for n, t in tallies.items()
    }
    return flux, rate, len(model.lost_particles), geometry.discontinuous


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("models", nargs="*", choices=[*MODELS, []], help="default: all")
    parser.add_argument("--particles", type=int, default=300_000)
    parser.add_argument("--density", type=float, default=1.0, help="density scale")
    parser.add_argument("--tolerance", type=float, default=0.1, help="chordal deflection")
    parser.add_argument("--data", default="tests", help="directory of the nuclide .arrow data")
    args = parser.parse_args()

    rows = []
    for name in args.models or list(MODELS):
        with tempfile.TemporaryDirectory() as tmp:
            imp, names, source, mesh_imp = mesh(MODELS[name], True, args.tolerance, tmp)
            sep, _, _, mesh_sep = mesh(MODELS[name], False, args.tolerance, tmp)
            ref, rate_imp, lost_imp, _ = transport(imp, names, source, 1, args)
            same, rate_sep, lost_sep, disc = transport(sep, names, source, 1, args)
            indep, _, _, _ = transport(sep, names, source, 2, args)
        assert disc, "the non-imprinted mesh should be tracked discontinuously"
        # In standard deviations of the tally, so a low-statistics corner
        # solid does not dominate once the histories drift apart.
        same_diff = max(
            abs(same[n][0] - ref[n][0]) / ref[n][1] for n in names if ref[n][1] > 0
        )
        worst_z = max(
            abs(indep[n][0] - ref[n][0]) / math.hypot(indep[n][1], ref[n][1])
            for n in names
            if indep[n][1] > 0 or ref[n][1] > 0
        )
        rows.append((name, len(names), mesh_imp, mesh_sep, rate_imp, rate_sep,
                     lost_imp, lost_sep, same_diff, worst_z))

    print()
    print(f"{args.particles} histories, density x{args.density}, deflection {args.tolerance}")
    print("| model | solids | mesh, imprinted | mesh, separate | p/s, conformal | "
          "p/s, discontinuous | lost (conf / disc) | same-seed max diff / sigma | "
          "independent-seed max abs z |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (name, n, mi, ms, ri, rs, li, ls, sd, z) in rows:
        print(f"| {name} | {n} | {mi:.2f} s | {ms:.2f} s | {ri:,.0f} | {rs:,.0f} | "
              f"{li} / {ls} | {sd:.3f} | {z:.1f} |")


if __name__ == "__main__":
    main()
