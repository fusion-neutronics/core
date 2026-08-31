//! Validation kernel for the PCG-32 RNG (64-bit state, 32-bit output
//! word, PCG-XSH-RR 64/32; issue #274), expressed as a cubecl
//! `#[cube]` function so it compiles down to SPIR-V on the Vulkan
//! backend (and to other backends unchanged when we add them).
//!
//! Each GPU thread reads its seed, runs `samples_per_thread` iterations
//! of the PCG step, and writes the `u32` outputs back. The Rust test
//! reads the buffer and demands byte-for-byte agreement with the CPU
//! [`crate::common::rng::GpuRng`] reference. A mismatch means cubecl's codegen,
//! the SPIR-V driver, or the Rust impl have drifted.

use crate::common::rng::{
    PCG_INCR, PCG_MULT, SECONDARY_SEED_GOLDEN, SECONDARY_SEED_MIX_A, SECONDARY_SEED_MIX_B,
};
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// Expand a 32-bit per-history seed into a 64-bit PCG state via splitmix64
/// (`#[cube]` twin of `yamc_rng::expand_seed`). The kernel seeds
/// each history's RNG with this so the CPU (which applies the same expansion in
/// `yamc-physics`) and GPU consume the identical 64-bit stream (issue #274).
///
/// KERNEL-ONLY: the body uses plain `+`/`*` (cubecl wraps on the GPU), so
/// calling this on the host overflow-panics in debug builds. Host code must
/// use the wrapping `crate::common::rng::expand_seed` re-export instead
/// (bit-identical in release).
#[cube]
pub fn expand_seed(seed: u32) -> u64 {
    let mut z = (seed as u64) + 0x9E37_79B9_7F4A_7C15;
    z = (z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9;
    z = (z ^ (z >> 27)) * 0x94D0_49BB_1331_11EB;
    z ^ (z >> 31)
}

/// Derive an in-history secondary's 32-bit seed from its parent walk's seed and
/// its ordinal among that walk's secondaries (`#[cube]` twin of
/// `yamc_rng::secondary_seed`; issue #111).
///
/// The kernel's in-thread (n,xn) queue stores this per entry and re-seeds the
/// thread PCG from it on pop, so a secondary's physics is a function of its
/// place in the emission tree rather than of the drain order. The CPU bank
/// derives the identical value from the identical key, which is what lets the
/// CPU's LIFO stack and this queue's FIFO agree history for history.
///
/// A 32-bit finaliser (two `u32` multiplies), not the 64-bit splitmix64 of
/// [`expand_seed`]: 64-bit multiplies are emulated on the GPU and this runs per
/// secondary created.
///
/// KERNEL-ONLY, like [`expand_seed`]: plain `+`/`*` wrap on the GPU but
/// overflow-panic on the host in debug builds. Host code must call the wrapping
/// `crate::common::rng::secondary_seed` re-export.
#[cube]
pub fn secondary_seed(parent_seed: u32, ordinal: u32) -> u32 {
    let mut z = parent_seed + (ordinal + 1u32) * SECONDARY_SEED_GOLDEN;
    z = (z ^ (z >> 16u32)) * SECONDARY_SEED_MIX_A;
    z = (z ^ (z >> 13u32)) * SECONDARY_SEED_MIX_B;
    z ^ (z >> 16u32)
}

/// One PCG-32 step, value-in / struct-out: takes the current `state` by
/// value and returns the output word plus the advanced state. The neutron
/// and photon kernels inline this same block ~90 times; this helper is the
/// single tested source of the PCG arithmetic so call sites collapse to
/// `let d = pcg_next(state); state = d.state; let r = d.rand;`.
///
/// It deliberately does NOT take `&mut u64`: cubecl's `&mut` bindings alias
/// live state inside a kernel (see the note on [`pcg32_kernel`]), so the
/// output permutation would read the *advanced* state. Value-in / struct-out
/// sidesteps that -- the same pattern the eout-sampler extraction uses, and
/// `gpu_pcg_next_matches_cpu` pins it bit-for-bit on hardware.
#[derive(CubeType)]
pub struct Pcg32Draw {
    /// The PCG output word (a uniform `u32`).
    pub rand: u32,
    /// The advanced 64-bit RNG state to thread into the next draw.
    pub state: u64,
}

/// PCG-XSH-RR 64->32 output permutation of a state word (no advance). `#[cube]`
/// twin of `yamc_rng::pcg_xsh_rr`, for probe/sampling kernels
/// that previously inlined the 32-bit output with `OUT_MULT` (issue #274).
#[cube]
pub fn pcg_out(state: u64) -> u32 {
    let xorshifted = (((state >> 18) ^ state) >> 27) as u32;
    let rot = (state >> 59) as u32;
    (xorshifted >> rot) | (xorshifted << ((32u32 - rot) & 31u32))
}

#[cube]
pub fn pcg_next(state: u64) -> Pcg32Draw {
    // PCG-XSH-RR 64->32 output permutation of the current state, then advance
    // the 64-bit LCG (issue #274). Mirrors `yamc_rng::pcg_xsh_rr`
    // + the LCG step, kept bit-identical for the #40 CPU/GPU matched stream.
    let xorshifted = (((state >> 18) ^ state) >> 27) as u32;
    let rot = (state >> 59) as u32;
    let rand = (xorshifted >> rot) | (xorshifted << ((32u32 - rot) & 31u32));
    let new_state = state * PCG_MULT + PCG_INCR;
    Pcg32Draw {
        rand,
        state: new_state,
    }
}

/// A PCG-32 draw mapped to a uniform `xi` in `(0, 1]` via the kernel's
/// canonical `(r + 1) / (2^32 + 1)` conversion, plus the advanced state.
/// This is the dominant draw form across both kernels; call sites that
/// scale (`xi * total`) keep that scaling at the call site.
#[derive(CubeType)]
pub struct UniformDraw {
    pub xi: f64,
    pub state: u64,
}

#[cube]
pub fn draw_uniform(state: u64) -> UniformDraw {
    let d = pcg_next(state);
    UniformDraw {
        xi: (d.rand as f64 + 1.0) * (1.0 / 4_294_967_297.0),
        state: d.state,
    }
}

/// CPU twin of [`pcg_next`] (wrapping 64-bit arithmetic). Returns `(rand, state)`.
pub fn pcg_next_cpu(state: u64) -> (u32, u64) {
    let rand = yamc_rng::pcg_xsh_rr(state);
    let new_state = state.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
    (rand, new_state)
}

/// CPU twin of [`draw_uniform`]. Returns `(xi, state)`.
pub fn draw_uniform_cpu(state: u64) -> (f64, u64) {
    let (rand, new_state) = pcg_next_cpu(state);
    ((rand as f64 + 1.0) * (1.0 / 4_294_967_297.0), new_state)
}

/// Validation kernel for [`draw_uniform`]: one draw per thread from its seed.
#[cube(launch_unchecked)]
fn draw_uniform_kernel(seeds: &[u32], out_xi: &mut [f64], out_state: &mut [u64]) {
    if ABSOLUTE_POS >= seeds.len() {
        terminate!();
    }
    let d = draw_uniform(expand_seed(seeds[ABSOLUTE_POS]));
    out_xi[ABSOLUTE_POS] = d.xi;
    out_state[ABSOLUTE_POS] = d.state;
}

/// Run [`draw_uniform`] on the GPU for each seed; returns `(xi, advanced_state)`
/// per seed. For test/validation use.
// The output buffers are sized by element count (one entry per seed), not by
// the `seeds` slice's byte size, so `size_of_val(seeds)` would be wrong here.
#[allow(clippy::manual_slice_size_calculation)]
pub fn run_draw_uniform(ctx: &GpuContext, seeds: &[u32]) -> (Vec<f64>, Vec<u64>) {
    let client = ctx.client();
    let n = seeds.len();
    let seed_h = client.create_from_slice(bytemuck::cast_slice(seeds));
    let xi_h = client.empty(n * core::mem::size_of::<f64>());
    let st_h = client.empty(n * core::mem::size_of::<u64>());
    const WG: u32 = 64;
    let groups = (n as u32).div_ceil(WG);
    unsafe {
        draw_uniform_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WG),
            BufferArg::from_raw_parts(seed_h, n),
            BufferArg::from_raw_parts(xi_h.clone(), n),
            BufferArg::from_raw_parts(st_h.clone(), n),
        );
    }
    let xi = bytemuck::cast_slice::<u8, f64>(&client.read_one(xi_h).unwrap()).to_vec();
    let st = bytemuck::cast_slice::<u8, u64>(&client.read_one(st_h).unwrap()).to_vec();
    (xi, st)
}

/// Validation kernel for [`secondary_seed`]: one derived seed per thread, from
/// the thread's `(parent_seed, ordinal)` pair.
#[cube(launch_unchecked)]
fn secondary_seed_kernel(parents: &[u32], ordinals: &[u32], out: &mut [u32]) {
    if ABSOLUTE_POS >= parents.len() {
        terminate!();
    }
    out[ABSOLUTE_POS] = secondary_seed(parents[ABSOLUTE_POS], ordinals[ABSOLUTE_POS]);
}

/// Run [`secondary_seed`] on the GPU for each `(parent, ordinal)` pair. For
/// test/validation use.
// The output buffer is sized by element count (one entry per pair), not by the
// `parents` slice's byte size, so `size_of_val` would be wrong here.
#[allow(clippy::manual_slice_size_calculation)]
pub fn run_secondary_seed(ctx: &GpuContext, parents: &[u32], ordinals: &[u32]) -> Vec<u32> {
    assert_eq!(parents.len(), ordinals.len());
    let client = ctx.client();
    let n = parents.len();
    let parent_h = client.create_from_slice(bytemuck::cast_slice(parents));
    let ord_h = client.create_from_slice(bytemuck::cast_slice(ordinals));
    let out_h = client.empty(n * core::mem::size_of::<u32>());
    const WG: u32 = 64;
    let groups = (n as u32).div_ceil(WG);
    unsafe {
        secondary_seed_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WG),
            BufferArg::from_raw_parts(parent_h, n),
            BufferArg::from_raw_parts(ord_h, n),
            BufferArg::from_raw_parts(out_h.clone(), n),
        );
    }
    bytemuck::cast_slice::<u8, u32>(&client.read_one(out_h).unwrap()).to_vec()
}

/// Per-thread driver: read seed, run `samples_per_thread` PCG iterations,
/// store results contiguously in `outputs`. `samples_per_thread` is a
/// comptime parameter so the loop unrolls cleanly; for the validation
/// test we use a fixed 32 samples.
///
/// The PCG step is inlined rather than factored into a helper that takes
/// `&mut u64`. cubecl's `&mut` bindings inside a kernel don't snapshot
/// the read in `let oldstate = *state` before the subsequent `*state = …`
/// the way Rust does -- empirically the reference aliases live state, so
/// the output permutation reads the *advanced* state and every sample is
/// shifted by one step. Inlining sidesteps the issue entirely.
#[cube(launch_unchecked)]
fn pcg32_kernel(seeds: &[u32], outputs: &mut [u32], #[comptime] samples_per_thread: u32) {
    if ABSOLUTE_POS >= seeds.len() {
        terminate!();
    }
    // ABSOLUTE_POS is `usize` in cubecl; keep array indices in usize
    // throughout and use u64 only for the RNG state itself.
    let tid = ABSOLUTE_POS;
    let samples = samples_per_thread as usize;
    let base = tid * samples;
    let mut state = expand_seed(seeds[tid]);
    for i in 0..samples {
        let d = pcg_next(state);
        outputs[base + i] = d.rand;
        state = d.state;
    }
}

/// Run the PCG-32 kernel on the GPU: each of `seeds.len()` threads
/// consumes its seed and emits `samples_per_thread` u32 outputs. The
/// returned `Vec<u32>` is laid out thread-major:
/// `[t0_s0, t0_s1, …, t0_s(M-1), t1_s0, …]`.
pub fn run_pcg32_validation(ctx: &GpuContext, seeds: &[u32], samples_per_thread: u32) -> Vec<u32> {
    let client = ctx.client();
    let n_threads = seeds.len();
    let total_outputs = n_threads * samples_per_thread as usize;

    let seed_handle = client.create_from_slice(bytemuck::cast_slice(seeds));
    let output_handle = client.empty(total_outputs * core::mem::size_of::<u32>());

    // 64 threads per workgroup is a safe default -- meets the SPIR-V
    // 1024-invocation cap with room to spare and matches NVIDIA/AMD
    // warp/wavefront sizes for predictable scheduling.
    const WORKGROUP_SIZE: u32 = 64;
    let groups = n_threads.div_ceil(WORKGROUP_SIZE as usize) as u32;

    unsafe {
        pcg32_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(seed_handle, n_threads),
            BufferArg::from_raw_parts(output_handle.clone(), total_outputs),
            samples_per_thread,
        );
    }

    let bytes = client.read_one(output_handle).unwrap();
    bytemuck::cast_slice(&bytes).to_vec()
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError, GpuRng};

    /// CPU and GPU PCG-32 must produce identical bits. Run the kernel
    /// for a handful of seeds, recompute on the CPU, demand equality.
    /// A single mismatch means the SPIR-V kernel and the Rust reference
    /// have drifted -- most likely cause is a missing/buggy u32 op in
    /// cubecl-spirv on the host's driver.
    #[test]
    fn gpu_pcg32_matches_cpu_byte_for_byte() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        let seeds: Vec<u32> = vec![0, 1, 42, 0xDEADBEEF, 7, 1_000_000];
        let samples_per_thread: u32 = 32;
        let gpu_out = run_pcg32_validation(&ctx, &seeds, samples_per_thread);

        let mut cpu_out: Vec<u32> = Vec::with_capacity(gpu_out.len());
        for &s in &seeds {
            let mut r = GpuRng::new(s);
            for _ in 0..samples_per_thread {
                cpu_out.push(r.next_u32());
            }
        }

        assert_eq!(gpu_out, cpu_out, "GPU PCG-32 output diverges from CPU");
    }

    /// The in-history secondary seeder (issue #111) must give the same 32-bit
    /// answer in the kernel and on the host. The kernel body uses plain
    /// `+` / `*`, which cubecl wraps on the GPU, and the host body uses
    /// `wrapping_*`; if those ever stop agreeing, a secondary would transport
    /// on one stream on the CPU and a different one on the GPU and the whole
    /// per-history bit-identity of #111 would silently break. Exact integer
    /// arithmetic, so this is equality, not a tolerance.
    #[test]
    fn gpu_secondary_seed_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        // Parents that exercise the high bits and the wrap, and the ordinals a
        // real history uses (0..3 for an (n,4n)) plus a few far out.
        let mut parents = Vec::new();
        let mut ordinals = Vec::new();
        for p in [0u32, 1, 42, 0x9E37_79B9, 0xDEAD_BEEF, u32::MAX] {
            for o in [0u32, 1, 2, 3, 17, 1_000_000, u32::MAX] {
                parents.push(p);
                ordinals.push(o);
            }
        }
        let gpu = run_secondary_seed(&ctx, &parents, &ordinals);
        let cpu: Vec<u32> = parents
            .iter()
            .zip(&ordinals)
            .map(|(&p, &o)| crate::common::rng::secondary_seed(p, o))
            .collect();
        assert_eq!(gpu, cpu, "GPU secondary_seed diverges from the CPU twin");
        println!("secondary_seed: {} pairs, GPU==CPU bit-exact", gpu.len());
    }

    /// The reusable `draw_uniform` / `pcg_next` helpers (value-in / struct-out)
    /// must match the CPU twins bit-for-bit: advanced state is exact `u64`
    /// integer math, and `xi` is pure `u32 -> f64` + one mul/add (no
    /// transcendentals), so equality is exact, not within a tolerance. This
    /// is what lets the ~90 inline draw sites be replaced safely.
    #[test]
    fn gpu_draw_uniform_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        let seeds: Vec<u32> = (0..512)
            .map(|i: u32| i.wrapping_mul(2_654_435_761).wrapping_add(0x9E37_79B9))
            .chain([0, 1, 42, 0xDEAD_BEEF, u32::MAX])
            .collect();

        let (gpu_xi, gpu_state) = run_draw_uniform(&ctx, &seeds);
        assert_eq!(gpu_xi.len(), seeds.len());

        for (i, &s) in seeds.iter().enumerate() {
            let (cpu_xi, cpu_state) = draw_uniform_cpu(crate::common::rng::expand_seed(s));
            assert_eq!(
                gpu_state[i], cpu_state,
                "seed {s}: advanced state differs (gpu {} cpu {cpu_state})",
                gpu_state[i]
            );
            assert_eq!(
                gpu_xi[i].to_bits(),
                cpu_xi.to_bits(),
                "seed {s}: xi differs (gpu {} cpu {cpu_xi})",
                gpu_xi[i]
            );
            assert!(
                gpu_xi[i] > 0.0 && gpu_xi[i] <= 1.0,
                "xi out of (0,1]: {}",
                gpu_xi[i]
            );
        }
        println!(
            "draw_uniform: {} seeds, GPU==CPU bit-exact (state + xi)",
            seeds.len()
        );
    }
}
