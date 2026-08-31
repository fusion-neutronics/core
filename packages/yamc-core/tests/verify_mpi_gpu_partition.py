#!/usr/bin/env python3
"""GPU histories must be partitioned across MPI ranks, not duplicated (#303).

Run under ``mpirun``; prints ``RANKS=<n> histories=<n> flux=<x>`` from rank 0 plus
``OK``. The wrapper compares against a serial GPU run: the pooled flux must be
identical (the per-chunk RNG streams are the same, each rank replaying a disjoint
subset) and the reported history count must be the global total, not one rank's
share and not n times it.
"""
import sys

import numpy as np

import yamc

DATA = "crates/yamc/tests"
TOTAL = 400_000


def main() -> int:
    material = yamc.Material(
        composition={"Fe56": 1.0}, density=7.874, temperature=294, name="iron", id=101
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
    results = model.simulate_transport(total_particles=TOTAL, seed=5, compute="gpu")

    if yamc.parallel.mpi_rank() == 0:
        flux = float(np.asarray(results["t"].mean, dtype=float).sum())
        print(
            f"RANKS={yamc.parallel.mpi_size()} "
            f"histories={results.runs[0]['n_histories']} flux={flux:.12g}"
        )
        print("OK")
    yamc.parallel.mpi_finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
