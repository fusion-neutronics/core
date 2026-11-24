"""Test that yamc produces reproducible results with a fixed seed.

This test verifies that:
1. Running with the same seed produces identical results within a process
2. Results are deterministic regardless of parallel execution order
"""
import subprocess
import sys
import os


# Use nuclear data from the tests folder (symlinked from crates/yamc/tests/)
TESTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'tests')
FE56_DATA_PATH = os.path.join(TESTS_DIR, 'Fe56.arrow')


def run_yamc_simulation(seed: int, particles: int = 1000, batches: int = 2) -> float:
    """Run a yamc simulation and return the total flux."""
    import yamc

    # Simple geometry
    sphere1 = yamc.Sphere(radius=1.0)
    sphere2 = yamc.Sphere(radius=10.0, boundary='vacuum')

    material = yamc.Material(
        composition={'Fe56': 1.0},
        density=7.8,
        temperature=294)
    material.read_nuclear_data({'Fe56': FE56_DATA_PATH})

    cell1 = yamc.Cell(region=sphere1.below)
    cell2 = yamc.Cell(region=sphere1.above & sphere2.below, material=material)
    geometry = yamc.Geometry([cell1, cell2])

    source = yamc.NeutronSource(
        energy=yamc.sources.Discrete([1e6], [1]),
        position=(0, 0, 0)
    )

    tally = yamc.Tally(scores=['flux'], cells=cell2)

    model = yamc.Model(geometry=geometry, tallies=[tally], source=source)
    results = model.simulate_transport(total_particles=particles * batches, seed=seed)

    return float(sum(results[tally].mean))


def test_reproducibility_same_process():
    """Test that running twice with the same seed gives identical results."""
    seed = 42
    result1 = run_yamc_simulation(seed)
    result2 = run_yamc_simulation(seed)

    # Results should be identical within floating point epsilon
    # (parallel reduction can cause ~1e-15 relative differences)
    rel_diff = abs(result1 - result2) / abs(result1) if result1 != 0 else abs(result2)
    assert rel_diff < 1e-14, f"Results differ too much: {result1} vs {result2}, rel_diff={rel_diff}"


def test_reproducibility_different_seeds():
    """Test that different seeds give different results."""
    result1 = run_yamc_simulation(seed=1)
    result2 = run_yamc_simulation(seed=2)

    # Different seeds should give different results
    assert result1 != result2, "Different seeds should give different results"


def test_reproducibility_across_processes():
    """Test that results are reproducible across separate Python processes.

    This is the critical test - it catches HashMap iteration order issues
    that only manifest across process boundaries.
    """
    # Run simulation in a subprocess and capture the result
    # Use absolute path to Fe56.arrow in the tests folder
    # Use repr() to properly escape backslashes on Windows
    script = f'''
import sys
import yamc

FE56_PATH = {repr(FE56_DATA_PATH)}
sphere1 = yamc.Sphere(radius=1.0)
sphere2 = yamc.Sphere(radius=10.0, boundary='vacuum')
material = yamc.Material(
    composition={{"Fe56": 1.0}},
    density=7.8,
    temperature=294)
material.read_nuclear_data({{"Fe56": FE56_PATH}})
cell1 = yamc.Cell(region=sphere1.below)
cell2 = yamc.Cell(region=sphere1.above & sphere2.below, material=material)
geometry = yamc.Geometry([cell1, cell2])
source = yamc.NeutronSource(
    energy=yamc.sources.Discrete([1e6], [1]),
    position=(0, 0, 0)
)
tally = yamc.Tally(scores=['flux'], cells=cell2)
model = yamc.Model(geometry=geometry, tallies=[tally], source=source)
results = model.simulate_transport(total_particles=2000, seed=42)
# Print result to stderr to avoid mixing with yamc stdout
print(f"RESULT:{{float(sum(results[tally].mean)):.15e}}", file=sys.stderr)
'''

    results = []
    for _ in range(3):
        proc = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            text=True
        )
        assert proc.returncode == 0, f"Subprocess failed: {proc.stderr}"
        # Extract result from stderr (format: "RESULT:value")
        for line in proc.stderr.split('\n'):
            if line.startswith('RESULT:'):
                result = float(line.split(':')[1])
                results.append(result)
                break
        else:
            raise ValueError(f"No RESULT found in stderr: {proc.stderr}")

    # All results should be identical (within floating point epsilon)
    for i, r in enumerate(results[1:], 1):
        rel_diff = abs(r - results[0]) / abs(results[0]) if results[0] != 0 else abs(r)
        assert rel_diff < 1e-14, (
            f"Results differ across processes: run 0 = {results[0]:.15e}, "
            f"run {i} = {r:.15e}, relative diff = {rel_diff:.2e}"
        )


if __name__ == "__main__":
    test_reproducibility_same_process()
    print("PASS: Same process reproducibility")

    test_reproducibility_different_seeds()
    print("PASS: Different seeds give different results")

    test_reproducibility_across_processes()
    print("PASS: Cross-process reproducibility")

    print("\nAll reproducibility tests passed!")
