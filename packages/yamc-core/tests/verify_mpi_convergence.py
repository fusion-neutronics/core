#!/usr/bin/env python3
"""Convergence targets must stop at the same point under MPI as serially (#241).

Run under ``mpirun``; prints one ``RANKS=<n> histories=<n> mean=<x>`` line from
rank 0 plus ``OK``. The wrapper test compares against a serial run.

Each rank transports its own share, so a target tested against a rank's own
aggregate would stop at the wrong precision. Before the fix the check was skipped
entirely for ``mpi_size > 1`` and the run went to the particle cap instead.

With ``--uncapped`` the particle cap is dropped and the convergence target is the
only stop condition. That combination used to be rejected up front under MPI on
the grounds that convergence was single-process only; now that the decision is
collective it terminates, and this is what proves it (#303).
"""
import sys

import numpy as np

import yamc

DATA = "crates/yamc/tests"
CAP = 2_000_000
TARGET = 0.02


def main() -> int:
    material = yamc.Material(
        composition={"Fe56": 1.0},
        density=7.874,
        temperature=294,
        name="iron",
        id=101,
    )
    material.read_nuclear_data({"Fe56": f"{DATA}/Fe56.arrow"})
    sphere = yamc.Sphere(radius=10.0, boundary="vacuum")
    cells = [yamc.Cell(name="core", region=sphere.below, material=material)]
    geometry = yamc.Geometry(cells)
    tally = yamc.Tally(scores=["flux"], name="t", cells=cells)
    source = yamc.NeutronSource(
        position=(0.0, 0.0, 0.0), energy=yamc.sources.Discrete([14.06e6], [1.0])
    )
    model = yamc.Model(geometry=geometry, tallies=[tally], source=source, verbose=[])
    model.convergence_targets = [
        yamc.ConvergenceTarget("relative_error", TARGET, tally="t")
    ]
    cap = None if "--uncapped" in sys.argv else CAP
    results = model.simulate_transport(total_particles=cap, seed=3)

    if yamc.parallel.mpi_rank() == 0:
        mean = float(np.asarray(results["t"].mean, dtype=float).sum())
        print(
            f"RANKS={yamc.parallel.mpi_size()} "
            f"histories={results.runs[0]['n_histories']} mean={mean:.9g}"
        )
        print("OK")
    yamc.parallel.mpi_finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
