//! Fixed-point f64 tally accumulation, validated end-to-end.
//!
//! Phase D needs many GPU threads atomically summing f64 contributions
//! into per-cell tally accumulators. Direct `Atomic<f64>::fetch_add`
//! isn't available on AMD/RADV (`atomic_f64.rs`), and the obvious
//! fallback `Atomic<u64>::compare_exchange_weak` panics in cubecl-spirv
//! (`atomic_u64_cas.rs`, cubecl#1318). Plain `Atomic<u64>::fetch_add`
//! does work (`atomic_u64_add.rs`), so the workable shape is:
//!
//! 1. Pick a fixed scale factor `S` (here `2^30` ≈ 1.07e9). Each f64
//!    contribution `x` is converted to `(x * S).round() as u64` and
//!    atomically added to the u64 accumulator.
//! 2. After the kernel, the host divides the final u64 by `S` to
//!    recover the f64 sum.
//!
//! Range and precision trade-offs at `S = 2^30`:
//! - Per-contribution precision: ~22 bits of fraction preserved
//!   (anything below 2^-30 ≈ 1e-9 is lost).
//! - Maximum representable single contribution: `u64::MAX / S` ≈ 1.7e10.
//! - Maximum total accumulated value: `u64::MAX / S` ≈ 1.7e10 (any
//!   single sum that exceeds this overflows the u64; in practice tally
//!   integrals stay much smaller).
//!
//! For typical yamc tally values (track-length-weighted XS contributions,
//! magnitudes 1e-3 to 1e2, summed over ~1e6 to 1e9 particles) those
//! limits are comfortable. If a future tally needs more dynamic range,
//! `S` can drop (e.g. `2^20`) at the cost of precision, or the
//! accumulator can be split into multiple buckets.
//!
//! This kernel takes a slice of f64 contributions, has each thread
//! convert its element to fixed-point and atomic-add into a single u64
//! accumulator, then the host reads back, divides by `S`, and compares
//! against the trivial sequential CPU sum. Pass criterion: relative
//! error below `1e-7` (well above the precision floor at `S = 2^30`,
//! and well below MC tally noise).

use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// Scale factor `2^30`. f64 round-trip accuracy at this scale is well
/// below MC noise for any tally yamc produces; see module docs.
pub const TALLY_SCALE: f64 = 1_073_741_824.0;

/// Each thread reads its contribution from `contributions[ABSOLUTE_POS]`,
/// converts to fixed-point via the shared scale factor, and atomically
/// adds the result to `accumulator[0]`. Negative contributions are
/// supported by going through `i64` then bit-casting to u64 (two's
/// complement adds correctly modulo 2^64, which is how an unsigned
/// accumulator handles signed sums).
#[cube(launch_unchecked)]
fn fixed_point_tally_kernel(contributions: &[f64], accumulator: &mut [Atomic<u64>]) {
    if ABSOLUTE_POS >= contributions.len() {
        terminate!();
    }
    let x = contributions[ABSOLUTE_POS];
    // Round-to-nearest via floor(x*S + 0.5). i64 cast (not i32) because
    // contributions can scale up: x ~ 100 with S = 2^30 gives ~1e11,
    // beyond i32. cubecl-spirv's f64 -> i64 cast was buggy elsewhere
    // (cubecl#1317) but was specifically wrong on a value-not-pattern
    // path; here we emit the cast directly with no further math, which
    // is the common compiler-supported case.
    let scaled = x * TALLY_SCALE;
    let rounded = if scaled >= 0.0 {
        (scaled + 0.5) as i64
    } else {
        -((-scaled + 0.5) as i64)
    };
    // Two's complement reinterpret: signed addition modulo 2^64 == the
    // u64 atomic add we have available.
    let bits = u64::reinterpret(rounded);
    accumulator[0].fetch_add(bits);
}

/// Run fixed-point tally accumulation: GPU sums `contributions` as u64
/// fixed-point, host divides by the scale factor and returns the f64
/// estimate.
pub fn run_fixed_point_tally(ctx: &GpuContext, contributions: &[f64]) -> f64 {
    let client = ctx.client();
    let n = contributions.len();
    let contrib_handle = client.create_from_slice(bytemuck::cast_slice(contributions));
    let initial: u64 = 0;
    let acc_handle = client.create_from_slice(bytemuck::bytes_of(&initial));

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        fixed_point_tally_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(contrib_handle, n),
            BufferArg::from_raw_parts(acc_handle.clone(), 1),
        );
    }

    let bytes = client.read_one(acc_handle).unwrap();
    let final_bits: u64 = bytemuck::cast_slice::<u8, u64>(&bytes)[0];
    // Reinterpret as signed two's complement, then divide by the scale.
    let final_signed = final_bits as i64;
    final_signed as f64 / TALLY_SCALE
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError};

    /// Sum a deterministic mix of tally-shaped f64 contributions on the
    /// GPU via fixed-point u64 atomics, compare to CPU. The test passes
    /// if the relative error is below 1e-7 (about 23 bits), well above
    /// the 2^-30 floor of the scale factor and well below any MC tally
    /// noise floor we'd care about.
    #[test]
    fn gpu_fixed_point_tally_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        // Tally-shaped: many small positive contributions, a few larger
        // ones, a few negative ones (e.g. balance corrections), magnitude
        // mix that reflects realistic transport scoring.
        let n = 1024usize;
        let contributions: Vec<f64> = (0..n)
            .map(|i| {
                let phase = (i as f64) * 0.1;
                if i % 17 == 0 {
                    100.0
                } else if i % 13 == 0 {
                    -1.5
                } else {
                    0.0123 + (phase % 1.0) * 0.456
                }
            })
            .collect();

        let gpu_sum = run_fixed_point_tally(&ctx, &contributions);
        let cpu_sum: f64 = contributions.iter().sum();

        let abs_err = (gpu_sum - cpu_sum).abs();
        let rel_err = abs_err / cpu_sum.abs().max(1.0);
        println!(
            "fixed-point tally: GPU={gpu_sum}, CPU={cpu_sum}, abs_err={abs_err}, rel_err={rel_err}"
        );
        assert!(
            rel_err < 1e-7,
            "fixed-point tally relative error {rel_err} exceeds 1e-7 budget"
        );
    }
}
