//! Photoelectric subshell sampling on the GPU -- given a photoabsorption
//! event, pick which atomic subshell absorbs the photon, weighted by the
//! per-subshell photoelectric cross section at the event energy. Extracted
//! from the inline block in `multi_cell_photon_transport` so the mega kernel
//! calls it (single source of truth) and it can be unit-tested against the
//! CPU `PhotonInteraction::sample_photoelectric_subshell`.
//!
//! Two-pass + one draw: Pass 1 sums the per-subshell XS (the CPU gets this
//! total from the precomputed `micro_xs.photoelectric`); Pass 2 does the
//! cumulative comparison against `ξ · total`. Deterministic given the draw
//! (no rejection loop), so the extraction is byte-identical.

use crate::common::pcg32::{expand_seed, pcg_out};
use crate::common::polyfills::exp_f64;
use crate::common::rng::{PCG_INCR, PCG_MULT};
use crate::photon::transport::AR_MAX_SHELLS;
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// Result of one photoelectric subshell sample.
#[derive(CubeType)]
pub struct PeSubshellSample {
    /// Advanced PCG-32 state after the one draw.
    pub state: u64,
    /// Index of the sampled subshell.
    pub shell: u32,
}

/// Sample which subshell photo-absorbs. `ar_pe_subshell_xs_log` is the flat
/// `[energy_point × AR_MAX_SHELLS]` per-subshell **log** XS table;
/// `mat_pe_off`, `idx_lo`/`idx_hi`, `frac` select and interpolate the energy
/// row; `ar_ns` is the number of valid subshells. Must be called with
/// `ar_ns > 0`. THE single source of truth, called by both the test launcher
/// and the inline photoelectric branch of `multi_cell_photon_transport`.
#[cube]
pub fn sample_pe_subshell(
    state: u64,
    ar_ns: u32,
    mat_pe_off: u32,
    idx_lo: u32,
    idx_hi: u32,
    frac: f64,
    ar_pe_subshell_xs_log: &[f64],
) -> PeSubshellSample {
    let mut st = state;

    // Pass 1: sum subshell XS at this energy. `frac * 0.0` is a runtime f64
    // zero (avoids the comptime-literal ambiguity for a mutable binding).
    let mut total_sub = frac * 0.0_f64;
    let mut s_p1 = 0u32;
    while s_p1 < ar_ns {
        let off_lo = (mat_pe_off + idx_lo) * AR_MAX_SHELLS + s_p1;
        let off_hi = (mat_pe_off + idx_hi) * AR_MAX_SHELLS + s_p1;
        let xs_lo_v = ar_pe_subshell_xs_log[off_lo as usize];
        let xs_hi_v = ar_pe_subshell_xs_log[off_hi as usize];
        if !(xs_lo_v == 0.0_f64 && xs_hi_v == 0.0_f64) {
            let val_log = if xs_lo_v == 0.0_f64 {
                xs_hi_v
            } else if xs_hi_v == 0.0_f64 {
                xs_lo_v
            } else {
                xs_lo_v + (xs_hi_v - xs_lo_v) * frac
            };
            total_sub += exp_f64(val_log);
        }
        s_p1 += 1u32;
    }

    let s_pe = st;
    let r_pe = pcg_out(s_pe);
    st = s_pe * PCG_MULT + PCG_INCR;
    let xi_pe = (r_pe as f64 + 1.0) * (1.0 / 4_294_967_297.0);
    let cutoff_sub = xi_pe * total_sub;

    // Pass 2: cumulative comparison. Default to last valid shell (CPU fallback).
    let mut prob_acc = frac * 0.0_f64;
    let mut found_pe = 0u32;
    let mut sampled_shell = ar_ns - 1u32;
    let mut s_p2 = 0u32;
    while s_p2 < ar_ns && found_pe == 0u32 {
        let off_lo = (mat_pe_off + idx_lo) * AR_MAX_SHELLS + s_p2;
        let off_hi = (mat_pe_off + idx_hi) * AR_MAX_SHELLS + s_p2;
        let xs_lo_v = ar_pe_subshell_xs_log[off_lo as usize];
        let xs_hi_v = ar_pe_subshell_xs_log[off_hi as usize];
        if !(xs_lo_v == 0.0_f64 && xs_hi_v == 0.0_f64) {
            let val_log = if xs_lo_v == 0.0_f64 {
                xs_hi_v
            } else if xs_hi_v == 0.0_f64 {
                xs_lo_v
            } else {
                xs_lo_v + (xs_hi_v - xs_lo_v) * frac
            };
            prob_acc += exp_f64(val_log);
            if prob_acc > cutoff_sub {
                sampled_shell = s_p2;
                found_pe = 1u32;
            }
        }
        s_p2 += 1u32;
    }

    PeSubshellSample {
        state: st,
        shell: sampled_shell,
    }
}

/// Per-thread test launcher: single element table (`mat_pe_off = 0`), two
/// energy rows at `idx_lo = 0` / `idx_hi = 1` interpolated by `frac`. Writes
/// the sampled subshell index to `out_shell[tid]`.
#[cube(launch_unchecked)]
fn pe_subshell_kernel(
    seeds: &[u32],
    ns_buf: &[u32],
    frac_buf: &[f64],
    ar_pe_subshell_xs_log: &[f64],
    out_shell: &mut [u32],
) {
    if ABSOLUTE_POS >= seeds.len() {
        terminate!();
    }
    // ar_ns / frac as single-element buffers (this cubecl has no scalar
    // launch args; the codebase passes scalars this way).
    let ar_ns = ns_buf[0usize];
    let frac = frac_buf[0usize];
    let state = expand_seed(seeds[ABSOLUTE_POS]);
    let s = sample_pe_subshell(state, ar_ns, 0u32, 0u32, 1u32, frac, ar_pe_subshell_xs_log);
    out_shell[ABSOLUTE_POS] = s.shell;
}

/// Run the photoelectric subshell sampler. `xs_log_two_rows` is the flat
/// `[2 × AR_MAX_SHELLS]` log-XS table (row 0 / row 1 = the two energy grid
/// points bracketing the event), `frac` the interpolation factor, `ar_ns`
/// the valid subshell count. Returns the sampled subshell index per seed.
pub fn run_pe_subshell(
    ctx: &GpuContext,
    seeds: &[u32],
    ar_ns: u32,
    frac: f64,
    xs_log_two_rows: &[f64],
) -> Vec<u32> {
    let client = ctx.client();
    let n = seeds.len();
    let seeds_handle = client.create_from_slice(bytemuck::cast_slice(seeds));
    let ns_handle = client.create_from_slice(bytemuck::cast_slice(&[ar_ns]));
    let frac_handle = client.create_from_slice(bytemuck::cast_slice(&[frac]));
    let xs_handle = client.create_from_slice(bytemuck::cast_slice(xs_log_two_rows));
    let out_handle = client.empty(std::mem::size_of_val(seeds));

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        pe_subshell_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(seeds_handle, n),
            BufferArg::from_raw_parts(ns_handle, 1),
            BufferArg::from_raw_parts(frac_handle, 1),
            BufferArg::from_raw_parts(xs_handle, xs_log_two_rows.len()),
            BufferArg::from_raw_parts(out_handle.clone(), n),
        );
    }

    bytemuck::cast_slice(&client.read_one(out_handle).unwrap()).to_vec()
}
