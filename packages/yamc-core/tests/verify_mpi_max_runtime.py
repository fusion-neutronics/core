#!/usr/bin/env python3
"""MPI collective max_runtime stop verification (run under mpirun).

Runs a wall-time-bounded simulation across the MPI world. The stop decision
is collective (#230): every rank checks its own elapsed time, an OR-reduce
agrees one global stop bit, and all ranks break at the same chunk checkpoint,
so the post-loop gather collectives never desync. If the collective stop were
wrong the ranks would break at different checkpoints and deadlock at the
gather (this script would hang), or rank 0's results would be empty.

Prints ``OK <n_histories>`` on rank 0 and exits 0 on success.

Usage:
    mpirun -n 2 python verify_mpi_max_runtime.py
"""
import sys

import yamc

SEED = 7


def build_model():
    material = yamc.Material(
        composition={"Li6": 1.0}, density=2.0, temperature=294
    )
    material.read_nuclear_data({"Li6": "tests/Li6.arrow"})
    sphere = yamc.Sphere(radius=10.0, boundary="vacuum")
    cell = yamc.Cell(name="sphere", region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        position=(0, 0, 0), energy=yamc.sources.Discrete([1.0e6], [1.0])
    )
    tally = yamc.Tally(scores=["flux"], name="flux", cells=cell, particle="neutron")
    return yamc.Model(geometry=geometry, tallies=[tally], source=source, verbose=[])


def main():
    if not yamc.parallel.has_mpi():
        print("ERROR: yamc built without MPI support")
        sys.exit(2)

    # Uncapped + wall-time budget: the ONLY thing that can end this run is the
    # collective max_runtime stop, so a successful return proves it fired
    # collectively (a per-rank break would deadlock the post-loop gather).
    model = build_model()
    results = model.simulate_transport(max_runtime=(2, "s"), seed=SEED, threads=1)

    if yamc.parallel.mpi_rank() == 0:
        r = results["flux"]
        assert r.n_histories > 0, "rank 0 has no histories after the collective stop"
        assert sum(r.mean) > 0.0, "rank 0 flux is empty"
        print(f"OK {r.n_histories}")


if __name__ == "__main__":
    main()
