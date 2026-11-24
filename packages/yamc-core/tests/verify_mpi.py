#!/usr/bin/env python3
"""
MPI verification test - compares serial vs MPI results for correctness.

This test runs the same simulation with different MPI rank counts and
verifies that results are identical (within statistical error).

Usage:
    # Run with different rank counts
    python verify_mpi.py --serial
    mpirun -np 2 python verify_mpi.py
    mpirun -np 4 python verify_mpi.py

    # Then compare results
    python verify_mpi.py --compare
"""

import argparse
import json
import math
import os
import yamc


def run_verification_simulation(output_file):
    """Run a test simulation and save results."""
    rank = yamc.parallel.mpi_rank()
    size = yamc.parallel.mpi_size()

    if rank == 0:
        print(f"Running verification with {size} rank(s)...")

    # Create simple geometry with material for meaningful tallies
    outer = yamc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=10.0,
                      boundary='vacuum')
    cell = yamc.Cell(name="void", region=outer.below, material=None)
    geometry = yamc.Geometry(cells=[cell])

    # Create flux tally
    tally = yamc.Tally(
        name="cell_flux",
        scores=["flux"],
        cells=cell,
        energy_bins=[0.0, 1e5, 1e6, 20e6])

    # Source
    source = yamc.NeutronSource(
        position=[0.0, 0.0, 0.0],
        energy=yamc.sources.Discrete([1.5e6], [1.0])
    )

    # Use fixed seed for reproducibility
    total_particles = 500000
    model = yamc.Model(geometry, tallies=[tally], source=source)
    sim_results = model.simulate_transport(total_particles=total_particles,
                                           seed=123456)

    # Only rank 0 saves results
    if rank == 0:
        tally_result = sim_results[tally]
        results = {
            'ranks': size,
            'total_particles': total_particles,
            'elapsed_time': sim_results.elapsed,
            'particles_per_second': sim_results.particles_per_second,
            'tally_mean': list(tally_result.mean),
            'tally_std_dev': list(tally_result.standard_deviation),
            'tally_rel_error': list(tally_result.relative_error),
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Results saved to {output_file}")
        print(f"  Flux mean: {tally_result.mean}")
        print(f"  Flux std_dev: {tally_result.standard_deviation}")


def compare_results(files):
    """Compare results from multiple runs."""
    print("\n" + "="*70)
    print("VERIFICATION: Comparing results across different rank counts")
    print("="*70)

    # Load all result files
    results = []
    for fname in files:
        if not os.path.exists(fname):
            print(f"Warning: {fname} not found, skipping")
            continue
        with open(fname, 'r') as f:
            results.append(json.load(f))

    if len(results) < 2:
        print("ERROR: Need at least 2 result files to compare")
        return False

    # Use first result as reference
    reference = results[0]
    ref_mean = reference['tally_mean']
    ref_std = reference['tally_std_dev']

    print(f"\nReference: {reference['ranks']} rank(s)")
    print(f"  Mean: {ref_mean}")
    print(f"  Std dev: {ref_std}")

    # Compare all other results to reference
    all_passed = True
    for i, result in enumerate(results[1:], start=1):
        ranks = result['ranks']
        mean = result['tally_mean']
        std = result['tally_std_dev']

        # Check if means are within 3 standard deviations
        # Combined uncertainty: sqrt(sigma1^2 + sigma2^2)
        z_score = [
            abs(m - rm) / (math.sqrt(rs**2 + s**2) + 1e-10)
            for m, rm, rs, s in zip(mean, ref_mean, ref_std, std)
        ]
        max_z = max(z_score)

        passed = max_z < 3.0
        status = "✓ PASS" if passed else "✗ FAIL"

        print(f"\n{ranks} rank(s): {status}")
        print(f"  Mean: {mean}")
        print(f"  Max Z-score: {max_z:.2f} (threshold: 3.0)")

        if not passed:
            all_passed = False
            print("  ERROR: Results differ significantly!")
            print(f"  Difference: {[m - r for m, r in zip(mean, ref_mean)]}")

    print("\n" + "="*70)
    if all_passed:
        print("✓ VERIFICATION PASSED: All results agree within statistical error")
    else:
        print("✗ VERIFICATION FAILED: Results differ across rank counts")
    print("="*70)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description='MPI verification test')
    parser.add_argument('--serial', action='store_true',
                        help='Run serial (1 rank) reference simulation')
    parser.add_argument('--compare', action='store_true',
                        help='Compare results from multiple runs')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file (default: verification_N_ranks.json)')

    args = parser.parse_args()

    if args.compare:
        # Compare existing results
        files = [
            'verification_1_ranks.json',
            'verification_2_ranks.json',
            'verification_4_ranks.json',
        ]
        success = compare_results(files)
        return 0 if success else 1

    # Check MPI
    if not yamc.parallel.has_mpi() and not args.serial:
        print("ERROR: MPI not available. Use --serial for serial test.")
        return 1

    # Determine output file
    ranks = 1 if args.serial else yamc.parallel.mpi_size()
    output = args.output or f'verification_{ranks}_ranks.json'

    # Run simulation
    run_verification_simulation(output)

    return 0


if __name__ == '__main__':
    exit(main())
