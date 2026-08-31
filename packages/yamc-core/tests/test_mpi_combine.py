"""MPI Welford rank-reduction tests.

Skipped unless yamc was built with the ``mpi`` feature and ``mpirun`` is
available. Launches ``verify_mpi_combine.py`` under ``mpirun -n 2`` and
checks rank 0's results against a single-process run of the same seed
and particle count: the per-particle RNG streams are global-index based
and the particle range is partitioned across ranks, so the *physical
histories are identical* -- the reduced statistics must match to
floating-point fold-order tolerance, and the history count must equal
the full particle count (a missing reduction would halve it).
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


def test_mpi_two_ranks_reduce_to_complete_statistics(tmp_path):
    out = tmp_path / "mpi_rank0.arrow"
    proc = subprocess.run(
        ["mpirun", "-n", "2", sys.executable,
         str(HERE / "verify_mpi_combine.py"), str(out)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(HERE.parents[2]),  # repo root, for tests/Li6.arrow
    )
    assert proc.returncode == 0, (
        f"mpirun failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}")
    assert out.exists(), "rank 0 did not write its results"

    mpi_results = yamc.SimulationResults.from_arrow(str(out))
    (info,) = mpi_results.runs
    assert info["mpi_size"] == 2
    assert info["mpi_rank"] == 0
    # The load-bearing assertion: a missing rank reduction leaves rank 0
    # with only its own half of the histories.
    assert info["n_histories"] == 4_000
    assert mpi_results["flux"].n_histories == 4_000

    # Same seed + same particle count single-process: identical physical
    # histories, so the reduced means agree to fold-order tolerance.
    from verify_mpi_combine import build_model  # noqa: E402

    model, run_kwargs = build_model()
    single = model.simulate_transport(**run_kwargs, threads=1)
    for name in ("flux", "absorption"):
        m = mpi_results[name].mean[0]
        s = single[name].mean[0]
        assert s > 0.0
        assert abs(m - s) / s < 1e-9, (
            f"{name}: mpi-reduced {m!r} vs single-process {s!r}")


def test_mpi_rank0_results_are_combinable(tmp_path):
    out = tmp_path / "mpi_rank0.arrow"
    subprocess.run(
        ["mpirun", "-n", "2", sys.executable,
         str(HERE / "verify_mpi_combine.py"), str(out)],
        check=True,
        capture_output=True,
        timeout=300,
        cwd=str(HERE.parents[2]),
    )
    mpi_results = yamc.SimulationResults.from_arrow(str(out))

    from verify_mpi_combine import build_model  # noqa: E402

    model, run_kwargs = build_model()
    run_kwargs["seed"] = 8  # distinct stream
    other = model.simulate_transport(**run_kwargs, threads=1)

    combined = yamc.combine_results(mpi_results, other)
    assert combined["flux"].n_histories == 8_000
    assert [r["seed"] for r in combined.runs] == [7, 8]
