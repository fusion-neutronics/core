"""``simulate_transmutation`` must not depend on the MPI rank count (#287).

Skipped unless yamc was built with the ``mpi`` feature and ``mpirun`` is
available. Runs ``verify_mpi_transmutation.py`` serially and under
``mpirun -n 2``, then compares the reported activation-product density.

The transmutation tallies accumulate rank-local sums but normalise by the
GLOBAL per-chunk particle count, so before the fix the 2-rank inventory was
exactly half the serial one (and a quarter at 4 ranks). Per-history seeds are
keyed to the global particle index, so the reduced result is not merely
statistically equal but reproduces the serial sum, which is what this asserts.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yamc

pytestmark = pytest.mark.skipif(
    not yamc.parallel.has_mpi() or shutil.which("mpirun") is None,
    reason="requires a yamc build with the mpi feature and mpirun",
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SCRIPT = HERE / "verify_mpi_transmutation.py"
VALUE = re.compile(r"^RANKS=(\d+) \w+=([0-9.eE+-]+)$", re.MULTILINE)


def _run(ranks: int) -> float:
    cmd = [sys.executable, str(SCRIPT)]
    if ranks > 1:
        cmd = ["mpirun", "-n", str(ranks)] + cmd
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=900, cwd=str(REPO_ROOT)
    )
    assert proc.returncode == 0, (
        f"{ranks}-rank run failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "OK" in proc.stdout, f"unexpected output: {proc.stdout}"
    match = VALUE.search(proc.stdout)
    assert match, f"no density line in output: {proc.stdout}"
    assert int(match.group(1)) == ranks, f"rank count mismatch: {proc.stdout}"
    return float(match.group(2))


def test_transmutation_inventory_is_rank_count_independent():
    serial = _run(1)
    assert serial > 0.0, "serial run produced no activation product; rig broken"
    parallel = _run(2)
    ratio = parallel / serial
    assert abs(ratio - 1.0) < 1e-9, (
        f"2-rank inventory {parallel:.6e} vs serial {serial:.6e} (ratio {ratio:.6f}); "
        "a ratio near 0.5 means the transmutation accumulators are not being "
        "reduced across ranks"
    )
