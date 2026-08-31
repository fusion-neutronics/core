"""`compute='gpu'` under MPI must partition the run, not duplicate it (#303).

Skipped unless yamc has the `mpi` feature, `mpirun` exists and an f64 GPU is
present. Before the fix the GPU dispatch had no MPI awareness: every rank
transported the full `total_particles` and rank 0 reported its own result, so 2
ranks did twice the work and took longer than serial (7.57 s vs 4.14 s for 2M
histories) while returning a single-rank answer.

The launch chunks are now strided across ranks, which keeps the per-chunk RNG
streams identical to serial, and the accumulators are summed before the mean and
m2 are derived. So the pooled result is not merely statistically equivalent, it
is the same number.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yamc

pytestmark = pytest.mark.skipif(
    not yamc.parallel.has_mpi()
    or shutil.which("mpirun") is None
    or not yamc.parallel.gpu_available(),
    reason="requires the mpi feature, mpirun and an f64 GPU",
)

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "verify_mpi_gpu_partition.py"
LINE = re.compile(r"^RANKS=(\d+) histories=(\d+) flux=([0-9.eE+-]+)$", re.MULTILINE)
TOTAL = 400_000


def _run(ranks: int) -> tuple[int, float]:
    cmd = [sys.executable, str(SCRIPT)]
    if ranks > 1:
        cmd = ["mpirun", "-n", str(ranks)] + cmd
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1800, cwd=str(HERE.parents[2])
    )
    assert proc.returncode == 0, (
        f"{ranks}-rank run failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    match = LINE.search(proc.stdout)
    assert match, f"no result line: {proc.stdout}"
    assert int(match.group(1)) == ranks
    return int(match.group(2)), float(match.group(3))


def test_gpu_run_is_partitioned_across_ranks():
    serial_histories, serial_flux = _run(1)
    assert serial_histories == TOTAL
    assert serial_flux > 0.0

    mpi_histories, mpi_flux = _run(2)
    assert mpi_histories == TOTAL, (
        f"2 ranks reported {mpi_histories} histories for a {TOTAL} request: "
        f"{'each rank ran the whole job' if mpi_histories == 2 * TOTAL else 'the counts were not pooled'}"
    )
    assert mpi_flux == pytest.approx(serial_flux, rel=1e-12), (
        f"2-rank flux {mpi_flux} vs serial {serial_flux}; striding the launch "
        "chunks should reproduce the serial streams exactly"
    )
