"""yamc joins MPI started by the host (mpi4py here, as for preCICE or MOOSE).

mpi4py is imported before yamc, so MPI_Init is mpi4py's. Run with:
    mpirun -np 2 pytest packages/yamc/tests/test_mpi_host.py -v
"""

import pytest

MPI = pytest.importorskip("mpi4py.MPI")

import yamc  # noqa: E402

pytestmark = pytest.mark.skipif(not yamc.parallel.has_mpi(), reason="MPI not available")


def test_rank_and_size_match_host():
    comm = MPI.COMM_WORLD
    assert yamc.parallel.mpi_rank() == comm.Get_rank()
    assert yamc.parallel.mpi_size() == comm.Get_size()


def test_finalize_leaves_host_mpi_running():
    yamc.parallel.mpi_finalize()
    assert not MPI.Is_finalized()
    MPI.COMM_WORLD.Barrier()
