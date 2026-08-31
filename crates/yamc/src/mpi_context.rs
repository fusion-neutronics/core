//! MPI context management for distributed Monte Carlo simulations.
//!
//! This module provides MPI initialization, communicator management, and collective operations
//! for parallel particle transport. The design follows embarrassingly parallel decomposition
//! at the particle level with periodic tally reduction.

#[cfg(feature = "mpi")]
use mpi::environment::Universe;
#[cfg(feature = "mpi")]
use mpi::topology::SimpleCommunicator;
#[cfg(feature = "mpi")]
use mpi::traits::*;
#[cfg(feature = "mpi")]
use std::sync::Mutex;

/// Global MPI universe (initialized once)
/// Using Mutex instead of OnceCell to allow explicit finalization
#[cfg(feature = "mpi")]
static MPI_UNIVERSE: Mutex<Option<Universe>> = Mutex::new(None);

/// True when the runtime environment indicates an MPI launcher
/// (mpirun / mpiexec / srun) is supervising this process -- i.e. we
/// should actually call `mpi::initialize()`. False on a plain `python
/// foo.py` or `./bin` invocation, where calling MPI_Init would trigger
/// minutes of network discovery before falling back to size=1.
#[cfg(feature = "mpi")]
fn running_under_mpi_launcher() -> bool {
    std::env::var("OMPI_COMM_WORLD_SIZE").is_ok()
        || std::env::var("PMI_SIZE").is_ok()
        || std::env::var("SLURM_NTASKS").is_ok()
}

/// MPI context wrapper providing rank, size, and communicator access
#[cfg(feature = "mpi")]
pub struct MpiContext {
    rank: i32,
    size: i32,
}

#[cfg(feature = "mpi")]
impl MpiContext {
    /// Initialize MPI and return context
    ///
    /// This should be called once at program startup. Subsequent calls will
    /// return the existing context without re-initializing MPI.
    ///
    /// When the process is not launched by an MPI launcher (no
    /// `OMPI_COMM_WORLD_SIZE` / `PMI_SIZE` / `SLURM_NTASKS` env var),
    /// `mpi::initialize()` is skipped entirely and a single-process context
    /// (rank=0, size=1) is returned. This avoids the multi-minute hang that
    /// `MPI_Init` can trigger on a workstation while it probes the network
    /// before settling on size=1. Collectives short-circuit when size == 1,
    /// so no real `Universe` is needed in that case.
    pub fn init() -> Self {
        let mut guard = MPI_UNIVERSE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        if guard.is_none() && !running_under_mpi_launcher() {
            return MpiContext { rank: 0, size: 1 };
        }

        if guard.is_none() {
            let universe = mpi::initialize().expect("Failed to initialize MPI");
            let world = universe.world();
            let rank = world.rank();
            let size = world.size();

            // Only rank 0 prints initialization message
            if rank == 0 {
                eprintln!("[MPI] Initialized with {} ranks", size);
            }

            *guard = Some(universe);

            MpiContext { rank, size }
        } else {
            let universe = guard.as_ref().unwrap();
            let world = universe.world();
            let rank = world.rank();
            let size = world.size();

            MpiContext { rank, size }
        }
    }

    /// Finalize MPI (should be called at program exit)
    ///
    /// This drops the MPI universe, which calls MPI_Finalize.
    /// Safe to call multiple times - only the first call has effect.
    pub fn finalize() {
        let mut guard = MPI_UNIVERSE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *guard = None; // Drop the Universe, which calls MPI_Finalize
    }

    /// Get MPI rank (0-indexed process ID)
    #[inline]
    pub fn rank(&self) -> i32 {
        self.rank
    }

    /// Get MPI size (total number of processes)
    #[inline]
    pub fn size(&self) -> i32 {
        self.size
    }

    /// Check if this is the root rank (rank 0)
    #[inline]
    pub fn is_root(&self) -> bool {
        self.rank == 0
    }

    /// Get the MPI world communicator
    pub fn world(&self) -> SimpleCommunicator {
        let guard = MPI_UNIVERSE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let universe = guard.as_ref().expect("MPI not initialized");
        universe.world()
    }

    /// Barrier synchronization across all ranks
    pub fn barrier(&self) {
        if self.size <= 1 {
            return;
        }
        self.world().barrier();
    }

    /// Sum-reduce f64 array across all ranks (in-place on root)
    ///
    /// # Arguments
    /// * `local_data` - Local data to reduce (modified on root to contain sum)
    /// * `root_rank` - Rank that receives the result (typically 0)
    ///
    /// # Example
    /// ```ignore
    /// let mut tallies = vec![1.0, 2.0, 3.0];
    /// ctx.reduce_sum_f64(&mut tallies, 0);
    /// // On rank 0: tallies now contains sum across all ranks
    /// ```
    pub fn reduce_sum_f64(&self, local_data: &mut [f64], root_rank: i32) {
        if self.size <= 1 {
            return;
        }
        let world = self.world();

        if self.rank == root_rank {
            // Root needs a separate receive buffer to avoid aliasing
            let mut recv_buffer = vec![0.0; local_data.len()];
            let root_process = world.process_at_rank(root_rank);
            root_process.reduce_into_root(
                &local_data[..],
                &mut recv_buffer[..],
                mpi::collective::SystemOperation::sum(),
            );
            // Copy result back to local_data
            local_data.copy_from_slice(&recv_buffer);
        } else {
            // Non-root ranks contribute their data
            let root_process = world.process_at_rank(root_rank);
            root_process.reduce_into(&local_data[..], mpi::collective::SystemOperation::sum());
        }
    }

    /// Broadcast f64 array from root to all ranks
    ///
    /// # Arguments
    /// * `data` - Data buffer (modified on non-root ranks to receive broadcast data)
    /// * `root_rank` - Rank broadcasting the data (typically 0)
    pub fn broadcast_f64(&self, data: &mut [f64], root_rank: i32) {
        if self.size <= 1 {
            return;
        }
        let world = self.world();
        let root_process = world.process_at_rank(root_rank);
        root_process.broadcast_into(data);
    }

    /// Gather equal-length f64 slices from every rank to `root_rank`.
    /// Returns `Some(concatenated)` on the root (rank-major order:
    /// rank 0's slice first), `None` on other ranks.
    ///
    /// Used by the per-history Welford rank reduction: unlike
    /// `reduce_sum_f64`, gathering preserves each rank's raw
    /// `(mean, m2)` state so the root can fold them with the numerically
    /// stable Chen combine instead of a cancellation-prone
    /// sum-of-squares reconstruction.
    pub fn gather_f64(&self, local: &[f64], root_rank: i32) -> Option<Vec<f64>> {
        if self.size <= 1 {
            return Some(local.to_vec());
        }
        let world = self.world();
        let root_process = world.process_at_rank(root_rank);
        if self.rank == root_rank {
            let mut recv = vec![0.0_f64; local.len() * self.size as usize];
            root_process.gather_into_root(local, &mut recv[..]);
            Some(recv)
        } else {
            root_process.gather_into(local);
            None
        }
    }

    /// Gather equal-length u64 slices from every rank to `root_rank`.
    /// Returns `Some(concatenated)` on the root, `None` elsewhere.
    pub fn gather_u64(&self, local: &[u64], root_rank: i32) -> Option<Vec<u64>> {
        if self.size <= 1 {
            return Some(local.to_vec());
        }
        let world = self.world();
        let root_process = world.process_at_rank(root_rank);
        if self.rank == root_rank {
            let mut recv = vec![0_u64; local.len() * self.size as usize];
            root_process.gather_into_root(local, &mut recv[..]);
            Some(recv)
        } else {
            root_process.gather_into(local);
            None
        }
    }

    /// Collective OR across ranks: returns `true` on EVERY rank when `local` is
    /// `true` on ANY rank. Built from a reduce-sum (of a 0/1 flag) to root plus
    /// a broadcast of the decision, so every rank agrees on one global stop bit
    /// in lockstep rather than acting on its rank-local view. For `size <= 1`
    /// it returns `local`. Used for the collective `max_runtime` early-stop so
    /// ranks break at the same chunk checkpoint and never desync the post-loop
    /// gather collectives (#230).
    pub fn any_rank_true(&self, local: bool) -> bool {
        let mut flag = [if local { 1.0 } else { 0.0 }];
        self.reduce_sum_f64(&mut flag, 0);
        let mut decision = [0.0_f64];
        if self.is_root() {
            decision[0] = if flag[0] > 0.5 { 1.0 } else { 0.0 };
        }
        self.broadcast_f64(&mut decision, 0);
        decision[0] > 0.5
    }
}

// No-op implementation when MPI is disabled
#[cfg(not(feature = "mpi"))]
pub struct MpiContext;

#[cfg(not(feature = "mpi"))]
impl MpiContext {
    pub fn init() -> Self {
        MpiContext
    }

    #[inline]
    pub fn rank(&self) -> i32 {
        0
    }

    #[inline]
    pub fn size(&self) -> i32 {
        1
    }

    #[inline]
    pub fn is_root(&self) -> bool {
        true
    }

    pub fn barrier(&self) {
        // No-op
    }

    pub fn reduce_sum_f64(&self, _local_data: &mut [f64], _root_rank: i32) {
        // No-op: single process, data is already "reduced"
    }

    pub fn broadcast_f64(&self, _data: &mut [f64], _root_rank: i32) {
        // No-op: single process, data is already "broadcast"
    }

    pub fn gather_f64(&self, local: &[f64], _root_rank: i32) -> Option<Vec<f64>> {
        // Single process: the "gather" is just this rank's data.
        Some(local.to_vec())
    }

    pub fn gather_u64(&self, local: &[u64], _root_rank: i32) -> Option<Vec<u64>> {
        Some(local.to_vec())
    }

    #[inline]
    pub fn any_rank_true(&self, local: bool) -> bool {
        // Single process: the global OR is just this rank's value.
        local
    }
}

/// Lets `TransmutationTallies::reduce_across_ranks` drive the collectives it
/// needs without `yani-transmute` naming the communicator (or depending on
/// `mpi`). Both `MpiContext` builds expose the same three inherent methods, so
/// one impl covers the MPI and non-MPI configurations; without MPI `size()` is
/// 1 and the reduction short-circuits before touching the other two.
impl yani_transmute::CollectiveOps for MpiContext {
    fn size(&self) -> i32 {
        self.size()
    }

    fn reduce_sum_f64(&self, local: &mut [f64], root_rank: i32) {
        self.reduce_sum_f64(local, root_rank)
    }

    fn broadcast_f64(&self, data: &mut [f64], root_rank: i32) {
        self.broadcast_f64(data, root_rank)
    }
}

/// Check if MPI support is compiled in
pub fn has_mpi_support() -> bool {
    cfg!(feature = "mpi")
}

/// Get current MPI rank (0 if MPI disabled)
///
/// # Panics
/// Panics with a helpful message if called when MPI is not enabled and
/// the environment suggests MPI execution (e.g., OMPI_COMM_WORLD_SIZE is set)
pub fn mpi_rank() -> i32 {
    #[cfg(feature = "mpi")]
    {
        let ctx = MpiContext::init();
        ctx.rank()
    }
    #[cfg(not(feature = "mpi"))]
    {
        // Check if we're being run under mpirun/mpiexec but MPI is not compiled
        if std::env::var("OMPI_COMM_WORLD_SIZE").is_ok()
            || std::env::var("PMI_SIZE").is_ok()
            || std::env::var("SLURM_NTASKS").is_ok()
        {
            eprintln!(
                "\n⚠️  WARNING: Detected MPI environment but YAMC was not built with MPI support!"
            );
            eprintln!("   You are running under mpirun/mpiexec, but this YAMC installation cannot use MPI.");
            eprintln!("   To enable MPI support, rebuild YAMC with:");
            eprintln!("     maturin develop --release --features \"pyo3,mpi\"");
            eprintln!("   Or install system packages first:");
            eprintln!("     sudo apt-get install -y libopenmpi-dev openmpi-bin build-essential clang libclang-dev");
            eprintln!("   Continuing in single-process mode...\n");
        }
        0
    }
}

/// Get MPI world size (1 if MPI disabled)
///
/// # Panics
/// Panics with a helpful message if called when MPI is not enabled and
/// the environment suggests MPI execution (e.g., OMPI_COMM_WORLD_SIZE is set)
pub fn mpi_size() -> i32 {
    #[cfg(feature = "mpi")]
    {
        let ctx = MpiContext::init();
        ctx.size()
    }
    #[cfg(not(feature = "mpi"))]
    {
        // Check if we're being run under mpirun/mpiexec but MPI is not compiled
        if std::env::var("OMPI_COMM_WORLD_SIZE").is_ok()
            || std::env::var("PMI_SIZE").is_ok()
            || std::env::var("SLURM_NTASKS").is_ok()
        {
            eprintln!(
                "\n⚠️  WARNING: Detected MPI environment but YAMC was not built with MPI support!"
            );
            eprintln!("   You are running under mpirun/mpiexec, but this YAMC installation cannot use MPI.");
            eprintln!("   To enable MPI support, rebuild YAMC with:");
            eprintln!("     maturin develop --release --features \"pyo3,mpi\"");
            eprintln!("   Or install system packages first:");
            eprintln!("     sudo apt-get install -y libopenmpi-dev openmpi-bin build-essential clang libclang-dev");
            eprintln!("   Continuing in single-process mode...\n");
        }
        1
    }
}

/// Finalize MPI (call at program exit to avoid spurious error messages)
///
/// This function drops the MPI universe, which properly calls MPI_Finalize.
/// It is safe to call multiple times - only the first call has effect.
/// When MPI is not enabled, this is a no-op.
pub fn mpi_finalize() {
    #[cfg(feature = "mpi")]
    {
        MpiContext::finalize();
    }
    #[cfg(not(feature = "mpi"))]
    {
        // No-op when MPI is disabled
    }
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mpi_context_no_crash() {
        // This test ensures the no-op implementation doesn't crash
        let ctx = MpiContext::init();
        assert_eq!(ctx.rank(), 0);
        assert_eq!(ctx.size(), 1);
        assert!(ctx.is_root());

        ctx.barrier();

        let mut data = vec![1.0, 2.0, 3.0];
        ctx.reduce_sum_f64(&mut data, 0);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);

        ctx.broadcast_f64(&mut data, 0);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn any_rank_true_is_identity_single_process() {
        // Single process (or the no-MPI stub): the collective OR is just this
        // rank's own value, so a time-budget stop decision is honoured locally.
        let ctx = MpiContext::init();
        assert!(ctx.any_rank_true(true));
        assert!(!ctx.any_rank_true(false));
    }

    #[test]
    fn test_has_mpi_support() {
        // Should return false in default build
        #[cfg(not(feature = "mpi"))]
        assert!(!has_mpi_support());

        #[cfg(feature = "mpi")]
        assert!(has_mpi_support());
    }
}
