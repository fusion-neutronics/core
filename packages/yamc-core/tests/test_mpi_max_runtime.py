"""MPI collective ``max_runtime`` stop test.

Skipped unless yamc was built with the ``mpi`` feature and ``mpirun`` is
available. Launches ``verify_mpi_max_runtime.py`` under ``mpirun -n 2``: the
script runs a wall-time-bounded simulation whose only stop condition is the
collective ``max_runtime`` early-stop (#230). A per-rank (non-collective)
break would deadlock the post-loop gather collectives, so a clean return with
rank 0 carrying histories is the proof the stop bit is agreed collectively.
"""
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


def test_mpi_max_runtime_stops_collectively():
    proc = subprocess.run(
        ["mpirun", "-n", "2", sys.executable, str(HERE / "verify_mpi_max_runtime.py")],
        capture_output=True,
        text=True,
        timeout=120,  # a hang (broken collective stop) trips this instead of running forever
        cwd=str(HERE.parents[2]),  # repo root, for tests/Li6.arrow
    )
    assert proc.returncode == 0, (
        f"mpirun failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "OK" in proc.stdout, f"unexpected output: {proc.stdout}"
