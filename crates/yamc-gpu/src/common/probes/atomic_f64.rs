//! f64 atomic-add probe. Phase D of the GPU port (track-length tally
//! accumulation) needs many threads atomically adding f64 contributions
//! to per-cell tally accumulators. This kernel checks whether
//! `Atomic<f64>::fetch_add` actually works on the host's Vulkan path
//! before any tally code depends on it.
//!
//! In SPIR-V terms, f64 atomic add lives behind the
//! `VK_EXT_shader_atomic_float2` extension and the
//! `AtomicFloat64AddEXT` capability. cubecl-wgpu detects both during
//! `init_setup`; cubecl-spirv emits the right `OpAtomicFAddEXT` when
//! they're present. The feature-detection check below uses cubecl's
//! own `client.properties().atomic_type_usage(...)` API, which is the
//! authoritative source.
//!
//! On a host where f64 atomic add is unavailable, the test prints a
//! "skipped -- not supported" message and returns rather than failing.
//! That's the answer Phase D needs: it tells us up front whether tally
//! accumulation can use native f64 atomics or has to fall back to a
//! u64-bit-reinterpret-CAS-loop, atomic u32 with split high/low halves,
//! or per-thread tally buffers reduced after the kernel.

use crate::{GpuContext, WgpuRuntime};
use cubecl::ir::features::AtomicUsage;
use cubecl::ir::{ElemType, FloatKind, Type};
use cubecl::prelude::*;

/// Each thread atomically adds 1.0 to `accumulator[0]`. Caller must
/// dispatch exactly `N` threads so the expected final value is
/// `initial + N`. Order of adds doesn't matter; atomic add is
/// associative for finite f64 values in the representable range here.
#[cube(launch_unchecked)]
fn atomic_f64_add_kernel(accumulator: &mut [Atomic<f64>]) {
    accumulator[0].fetch_add(1.0);
}

/// True if the active client supports f64 atomic add (i.e., the
/// `VK_EXT_shader_atomic_float2` extension is exposed and cubecl-spirv
/// has the corresponding capability registered).
pub fn supports_f64_atomic_add(ctx: &GpuContext) -> bool {
    let client = ctx.client();
    let ty = Type::atomic(Type::scalar(ElemType::Float(FloatKind::F64)));
    client
        .properties()
        .atomic_type_usage(ty)
        .contains(AtomicUsage::Add)
}

/// Run the atomic-add kernel. Dispatches `n_threads` threads (must be
/// a multiple of the 64-thread workgroup size), each adding 1.0 to a
/// single accumulator initialized to `initial`. Returns the final f64
/// read back from the GPU.
pub fn run_atomic_f64_add_probe(ctx: &GpuContext, n_threads: u32, initial: f64) -> f64 {
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
        atomic_f64_add_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(acc_handle.clone(), 1),
        );
    }

    let bytes = client.read_one(acc_handle).unwrap();
    bytemuck::cast_slice::<u8, f64>(&bytes)[0]
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// If supported on this host, run the kernel with 1024 threads each
    /// adding 1.0 to an accumulator starting at 0.0; expect 1024.0
    /// exactly (integers are exact in f64 well beyond 2^53). If
    /// unsupported, print and skip -- that's the answer Phase D needs.
    #[test]
    fn gpu_f64_atomic_add_is_correct_or_unsupported() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        if !supports_f64_atomic_add(&ctx) {
            println!(
                "f64 atomic add not supported on this adapter ({}). \
                 Phase D will need a workaround (CAS loop, atomic u32 halves, \
                 or per-thread tally + reduction).",
                ctx.adapter_info()
            );
            return;
        }

        let n: u32 = 1024;
        let result = run_atomic_f64_add_probe(&ctx, n, 0.0);
        assert_eq!(
            result, n as f64,
            "f64 atomic add returned {result}, expected {n}"
        );
        println!(
            "f64 atomic add works on {}: {n} threads → {result}",
            ctx.adapter_info()
        );
    }
}
