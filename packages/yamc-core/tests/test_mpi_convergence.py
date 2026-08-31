"""Convergence targets must be respected under MPI (#241, #303).

Skipped unless yamc was built with the ``mpi`` feature and ``mpirun`` is
available. Runs ``verify_mpi_convergence.py`` serially and at 2 ranks, then
compares where each stopped.

The check used to be wrapped in ``if mpi_size == 1``, so a multi-rank run
ignored the target and went to the particle cap instead: with a 5% target and a
200k cap, serial stopped at 20k histories and 2 ranks ran all 200k. The targets
are now evaluated on rank-folded aggregate moments, with the decision made once
on root and broadcast, so ranks stop at the same checkpoint.
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
SCRIPT = HERE / "verify_mpi_convergence.py"
LINE = re.compile(r"^RANKS=(\d+) histories=(\d+) mean=([0-9.eE+-]+)$", re.MULTILINE)
CAP = 2_000_000


def _run(ranks: int, uncapped: bool = False) -> tuple[int, float]:
    cmd = [sys.executable, str(SCRIPT)]
    if uncapped:
        cmd.append("--uncapped")
    if ranks > 1:
        cmd = ["mpirun", "-n", str(ranks)] + cmd
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=900, cwd=str(REPO_ROOT)
    )
    assert proc.returncode == 0, (
        f"{ranks}-rank run failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "OK" in proc.stdout, f"unexpected output: {proc.stdout}"
    match = LINE.search(proc.stdout)
    assert match, f"no result line in output: {proc.stdout}"
    assert int(match.group(1)) == ranks, f"rank count mismatch: {proc.stdout}"
    return int(match.group(2)), float(match.group(3))


def test_convergence_target_stops_the_same_way_under_mpi():
    serial_histories, serial_mean = _run(1)
    assert 0 < serial_histories < CAP, (
        f"serial run did not stop early ({serial_histories} histories); the target "
        "is not being reached, so this test cannot detect the MPI case"
    )

    mpi_histories, mpi_mean = _run(2)
    assert mpi_histories < CAP, (
        f"2-rank run went to the cap ({mpi_histories} histories) while serial stopped "
        f"at {serial_histories}: the convergence target is being ignored under MPI"
    )
    assert mpi_histories == serial_histories, (
        f"2 ranks stopped at {mpi_histories} histories, serial at {serial_histories}; "
        "the reduced aggregate should reproduce the serial decision"
    )
    assert mpi_mean == pytest.approx(serial_mean, rel=1e-9), (
        f"2-rank mean {mpi_mean} vs serial {serial_mean}"
    )


def test_an_uncapped_mpi_run_stops_on_its_convergence_target():
    """An uncapped MPI run with targets set is legal, and terminates (#303).

    It used to be refused before the run started: the guard rejected
    ``total_particles=None`` with no ``max_runtime`` under MPI because the only
    stop condition left was convergence, which was then single-process only.
    The collective gather-fold-broadcast made that reason obsolete, but the
    guard outlived it, so the one configuration the fix enabled stayed
    unreachable.
    """
    histories, _ = _run(2, uncapped=True)
    assert histories > 0, "uncapped 2-rank run reported no histories"
    assert histories < CAP, (
        f"uncapped 2-rank run did not stop on its target ({histories} histories)"
    )
