//! yamc joins MPI that the host process initialised (as mpi4py, preCICE or
//! MOOSE do) instead of panicking on `MPI_Init`, and `mpi_finalize` leaves that
//! MPI running for the host to finalise.
//!
//! `cargo test --features mpi` runs this binary as a single process, so the
//! test relaunches itself under `mpirun -np 2`, where each rank plays the host.
#![cfg(feature = "mpi")]

use mpi::traits::*;
use std::process::Command;
use yamc::mpi_context::{mpi_finalize, MpiContext};

const TEST_NAME: &str = "host_initialised_mpi_is_joined_and_not_finalized";
const RANKS: i32 = 2;

fn under_mpi_launcher() -> bool {
    std::env::var("OMPI_COMM_WORLD_SIZE").is_ok() || std::env::var("PMI_SIZE").is_ok()
}

#[test]
fn host_initialised_mpi_is_joined_and_not_finalized() {
    if under_mpi_launcher() {
        let universe = mpi::initialize().expect("host initialises MPI");
        let world = universe.world();

        let ctx = MpiContext::init();
        assert_eq!(ctx.size(), world.size());
        assert_eq!(ctx.rank(), world.rank());

        let mut data = [1.0_f64];
        ctx.reduce_sum_f64(&mut data, 0);
        if ctx.is_root() {
            assert_eq!(data[0], f64::from(RANKS));
        }

        mpi_finalize();
        assert!(
            !mpi::environment::is_finalized(),
            "yamc finalised the host's MPI"
        );
        world.barrier();
        return; // dropping `universe` is the host finalising MPI
    }

    let status = Command::new("mpirun")
        .args(["-np", &RANKS.to_string()])
        .arg(std::env::current_exe().unwrap())
        .args(["--exact", TEST_NAME, "--nocapture", "--test-threads=1"])
        .status()
        .expect("run mpirun: the mpi feature needs an MPI launcher on PATH");
    assert!(status.success(), "mpirun -np {RANKS} failed: {status}");
}
