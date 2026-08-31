#!/usr/bin/env python3
"""Transmutation must give the same inventory under MPI as serially (#287).

Run under ``mpirun``; prints one ``RANKS=<n> <nuclide>=<density>`` line from
rank 0 plus ``OK``. The wrapper test compares the value against a serial run.

The transmutation accumulators are rank-local while their normalisation
denominator is global, so before the fix every inventory came out low by exactly
``1/n_ranks``.
"""
import math
import sys

import yamc

DATA = "crates/yamc/tests"
CHAIN = f"{DATA}/transmutation-endf-b8.1-sfr.arrow"
MATERIAL_ID = 201
NUCLIDE = "Mn56"
RADIUS = 5.0


def main() -> int:
    yamc.transmutation_decay_data = CHAIN
    yamc.transmutation_reactions = CHAIN
    yamc.transmutation_fission_yields = CHAIN

    material = yamc.Material(
        composition={"Fe56": 1.0},
        density=7.874,
        temperature=294,
        name="iron",
        id=MATERIAL_ID,
        transmutable=True,
        volume=4.0 / 3.0 * math.pi * RADIUS**3,
    )
    material.read_nuclear_data({"Fe56": f"{DATA}/Fe56.arrow"})

    sphere = yamc.Sphere(radius=RADIUS, boundary="vacuum")
    geometry = yamc.Geometry(
        [yamc.Cell(name="core", region=sphere.below, material=material)]
    )
    source = yamc.NeutronSource(
        position=(0.0, 0.0, 0.0), energy=yamc.sources.Discrete([14.06e6], [1.0])
    )
    model = yamc.Model(geometry=geometry, source=source, verbose=[])
    schedule = yamc.PulseSchedule(
        [
            yamc.Pulse(rate=1e14, duration=3600.0, source=source),
            yamc.Cooldown(duration=3600.0),
        ]
    )
    results = model.simulate_transmutation(
        method="independent", schedule=schedule, total_particles=8000, seed=1
    )
    evolution = results.get_nuclide_evolution(MATERIAL_ID, NUCLIDE)

    if yamc.parallel.mpi_rank() == 0:
        print(f"RANKS={yamc.parallel.mpi_size()} {NUCLIDE}={evolution[-1]:.12e}")
        print("OK")
    yamc.parallel.mpi_finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
