//! Phase D fallback probe: f64 accumulation via u64 atomic
//! compare-exchange. The natural f64 atomic add is unavailable on
//! AMD/RADV (see `atomic_f64.rs`), so the next viable path for tally
//! accumulation is a CAS loop: load the f64 bit pattern as u64, compute
//! the new value, atomically swap if the slot still holds the old bits.
//!
//! # Empirical finding (AMD Strix Halo + RADV/ACO)
//!
//! **Works on cubecl 0.11.0-pre.3** (Mesa 26.0.3): 1024 threads each
//! CAS-adding 1.0 into one f64 slot read back exactly 1024.0, so the
//! test below runs unconditionally as the regression check.
//!
//! On cubecl 0.10.0-pre.4 through 0.10.0 the same kernel never ran:
//! `Atomic<u64>::compare_exchange_weak` panicked inside cubecl-spirv
//! during shader compilation ("Atomic should have a scope registered",
//! `cubecl-spirv/src/atomic.rs:354`), even though feature detection
//! (`atomic_type_usage` for `Atomic<u64>` with `LoadStore`) reported the
//! type as supported. Filed as cubecl#1318; fixed by the pliron rewrite
//! of the SPIR-V backend that shipped in 0.11.0-pre.3.
//!
//! The production tally path does not use this: it was designed while
//! CAS was broken and settled on fixed-point u64 `fetch_add`
//! (`fixed_point_tally.rs`), which is also cheaper than a CAS retry loop
//! under contention. This probe stays as the record that the CAS
//! fallback is available again if a future accumulator needs true f64
//! atomics.

use crate::{GpuContext, WgpuRuntime};
use cubecl::ir::features::AtomicUsage;
use cubecl::ir::{ElemType, Type, UIntKind};
use cubecl::prelude::*;

/// Each thread atomically adds 1.0 to `accumulator[0]` (interpreted as
/// f64) using a u64 CAS retry loop. The retry bound is generous -- under
/// realistic contention each thread succeeds in a handful of attempts.
#[cube(launch_unchecked)]
fn atomic_f64_cas_kernel(accumulator: &mut [Atomic<u64>]) {
    let mut done = false;
    let mut i = 0u32;
    while i < 1024u32 && !done {
        let old_bits = accumulator[0].load();
        let old_f64 = f64::reinterpret(old_bits);
        let new_f64 = old_f64 + 1.0;
        let new_bits = u64::reinterpret(new_f64);
        let prev = accumulator[0].compare_exchange_weak(old_bits, new_bits);
        if prev == old_bits {
            done = true;
        }
        i += 1u32;
    }
}

/// True if the active client exposes basic u64 atomics (load/store at
/// minimum). cubecl-wgpu/Vulkan registers `AtomicUsage::all()` for any
/// atomic type the driver supports, so this is a yes/no gate.
pub fn supports_u64_atomic(ctx: &GpuContext) -> bool {
    let client = ctx.client();
    let ty = Type::atomic(Type::new(ElemType::UInt(UIntKind::U64)));
    client
        .properties()
        .atomic_type_usage(ty)
        .contains(AtomicUsage::LoadStore)
}

/// Run the f64-via-u64-CAS kernel. Dispatches `n_threads` threads (must
/// be a multiple of the 64-thread workgroup size); each adds 1.0 to a
/// single accumulator initialized to `initial`. Returns the final f64.
pub fn run_atomic_f64_cas_probe(ctx: &GpuContext, n_threads: u32, initial: f64) -> f64 {
    assert!(
        n_threads.is_multiple_of(64),
        "n_threads must be a multiple of the workgroup size (64)"
    );
    let client = ctx.client();
    let initial_bits = initial.to_bits();
    let acc_handle = client.create_from_slice(bytemuck::bytes_of(&initial_bits));

    const WORKGROUP_SIZE: u32 = 64;
    let groups = n_threads / WORKGROUP_SIZE;

    unsafe {
        atomic_f64_cas_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(acc_handle.clone(), 1),
        );
    }

    let bytes = client.read_one(acc_handle).unwrap();
    let final_bits: u64 = bytemuck::cast_slice::<u8, u64>(&bytes)[0];
    f64::from_bits(final_bits)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// Passes on cubecl 0.11.0-pre.3 + RADV. It was `#[ignore]`d on
    /// cubecl 0.10, where cubecl-spirv panicked while lowering the CAS
    /// (cubecl#1318); see module docs.
    #[test]
    fn gpu_f64_via_u64_cas_is_correct_or_unsupported() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        if !supports_u64_atomic(&ctx) {
            println!(
                "u64 atomic ops not supported on this adapter ({}). \
                 Phase D will need per-thread tally buffers + a reduction kernel.",
                ctx.adapter_info()
            );
            return;
        }

        let n: u32 = 1024;
        let result = run_atomic_f64_cas_probe(&ctx, n, 0.0);
        assert_eq!(
            result, n as f64,
            "f64 CAS-loop accumulation returned {result}, expected {n}"
        );
        println!(
            "f64 CAS accumulation works on {}: {n} threads → {result}",
            ctx.adapter_info()
        );
    }
}
