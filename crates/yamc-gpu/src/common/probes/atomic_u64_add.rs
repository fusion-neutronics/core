//! Phase D fallback probe: `Atomic<u64>::fetch_add` for fixed-point
//! tally accumulation.
//!
//! Two earlier probes ruled out the obvious paths on AMD/RADV +
//! cubecl-spirv: native `Atomic<f64>::fetch_add` isn't exposed by the
//! driver (`atomic_f64.rs`), and `Atomic<u64>::compare_exchange_weak`
//! panics inside cubecl-spirv during shader compile (`atomic_u64_cas.rs`,
//! filed as cubecl#1318). The next viable shape is plain u64 atomic
//! add: scale every f64 contribution by a fixed factor (e.g. 2^30),
//! drop the fractional bits, and accumulate as u64. Convert back to
//! f64 on the host side after the kernel.
//!
//! That trades dynamic range for a working atomic. For yamc tally
//! values (track-length-weighted XS contributions, typical magnitudes
//! 1e-3 to 1e2), a 2^30 scaling factor preserves ~22 useful bits per
//! contribution, well under the 64-bit accumulator limit even after
//! summing millions of particles. The host-side conversion is one
//! `(u64 as f64) / (2^30 as f64)` per cell.
//!
//! What this kernel checks: does cubecl-spirv emit working
//! `Atomic<u64>::fetch_add` on RADV, in the simple non-CAS form?
//! If yes, fixed-point tally accumulation is viable. If no, Phase D
//! falls back to per-thread tally buffers + a reduction kernel.

use crate::{GpuContext, WgpuRuntime};
use cubecl::ir::features::AtomicUsage;
use cubecl::ir::{ElemType, Type, UIntKind};
use cubecl::prelude::*;

/// Each thread atomically adds 1 to `accumulator[0]`. With `N` threads,
/// the final value should be `initial + N`. No CAS, no reinterpret --
/// the simplest possible test of u64 atomic add.
#[cube(launch_unchecked)]
fn atomic_u64_add_kernel(accumulator: &mut [Atomic<u64>]) {
    accumulator[0].fetch_add(1u64);
}

/// True if the active client supports u64 atomic add specifically
/// (rather than just LoadStore).
pub fn supports_u64_atomic_add(ctx: &GpuContext) -> bool {
    let client = ctx.client();
    let ty = Type::atomic(Type::scalar(ElemType::UInt(UIntKind::U64)));
    client
        .properties()
        .atomic_type_usage(ty)
        .contains(AtomicUsage::Add)
}

/// Run the u64 atomic-add kernel. Dispatches `n_threads` threads (must
/// be a multiple of the 64-thread workgroup size); each adds 1 to a
/// single accumulator initialized to `initial`. Returns the final u64.
pub fn run_atomic_u64_add_probe(ctx: &GpuContext, n_threads: u32, initial: u64) -> u64 {
    assert!(
        n_threads.is_multiple_of(64),
        "n_threads must be a multiple of the workgroup size (64)"
    );
    let client = ctx.client();
    let initial_slice = [initial];
    let acc_handle = client.create_from_slice(bytemuck::cast_slice(&initial_slice));

    const WORKGROUP_SIZE: u32 = 64;
    let groups = n_threads / WORKGROUP_SIZE;

    unsafe {
        atomic_u64_add_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(acc_handle.clone(), 1),
        );
    }

    let bytes = client.read_one(acc_handle).unwrap();
    bytemuck::cast_slice::<u8, u64>(&bytes)[0]
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// If u64 atomic add is supported, run 1024 threads each adding 1
    /// to a single u64 accumulator; expect 1024 exactly. If unsupported,
    /// print and return -- that signals Phase D needs per-thread buffers
    /// instead of fixed-point u64 accumulation.
    #[test]
    fn gpu_u64_atomic_add_is_correct_or_unsupported() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        if !supports_u64_atomic_add(&ctx) {
            println!(
                "u64 atomic add not supported on this adapter ({}). \
                 Phase D will need per-thread tally buffers + a reduction kernel.",
                ctx.adapter_info()
            );
            return;
        }

        let n: u32 = 1024;
        let result = run_atomic_u64_add_probe(&ctx, n, 0);
        assert_eq!(
            result, n as u64,
            "u64 atomic add returned {result}, expected {n}"
        );
        println!(
            "u64 atomic add works on {}: {n} threads → {result}",
            ctx.adapter_info()
        );
    }
}
