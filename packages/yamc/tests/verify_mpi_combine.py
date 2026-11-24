#!/usr/bin/env python3
"""MPI Welford-reduction verification (run under mpirun).

Runs a small Li6 simulation across the MPI world; the per-rank Welford
states are gathered and folded on rank 0 inside simulate_transport, so
rank 0's results must carry the FULL history count and complete
statistics. Rank 0 writes its results to the Arrow path given as
argv[1]; the companion pytest (test_mpi_combine.py) compares them
against a single-process run of the same seed.

Usage:
    mpirun -n 2 python verify_mpi_combine.py /tmp/out.arrow
"""
import sys

import yamc

SEED = 7
PARTICLES = 4_000


def build_model():
    material = yamc.Material(
        composition={"Li6": 1.0},
        density=2.0,
        temperature=294)
    material.read_nuclear_data({"Li6": "tests/Li6.arrow"})
    sphere = yamc.Sphere(radius=10.0, boundary="vacuum")
    cell = yamc.Cell(name="sphere", region=sphere.below, material=material)
    geometry = yamc.Geometry([cell])
    source = yamc.NeutronSource(
        position=(0, 0, 0),
        energy=yamc.sources.Discrete([1.0e6], [1.0]))
    tallies = [
        yamc.Tally(scores=["flux"], name="flux", cells=cell,
                   particle="neutron"),
        yamc.Tally(scores=["absorption"], name="absorption", cells=cell,
                   particle="neutron"),
    ]
    model = yamc.Model(
        geometry=geometry,
        tallies=tallies,
        source=source,
        verbose=[])
    return model, {"total_particles": PARTICLES, "seed": SEED}


def main():
    if len(sys.argv) != 2:
        print("usage: verify_mpi_combine.py <output.arrow>")
        sys.exit(2)
    if not yamc.parallel.has_mpi():
        print("ERROR: yamc built without MPI support")
        sys.exit(2)

    model, run_kwargs = build_model()
    results = model.simulate_transport(**run_kwargs, threads=1)

    if yamc.parallel.mpi_rank() == 0:
        results.to_arrow(sys.argv[1])
        print(f"rank 0 wrote results (world size {yamc.parallel.mpi_size()})")
    yamc.parallel.mpi_finalize()


if __name__ == "__main__":
    main()
