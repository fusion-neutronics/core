//! Incoherent scattering function S(x, Z) evaluation on the GPU -- the
//! edge-clamped, binary-search lin-lin interpolation of the tabulated
//! incoherent form factor that Compton scattering uses to reject
//! free-electron Klein-Nishina samples (rejection prob 1 − S(x)/S(x_max)).
//! Extracted from the inline lookups in `multi_cell_photon_transport` (used
//! both for `S(x_max)` and per-sample `S(x)`) so it's a single source of
//! truth and can be unit-tested against the CPU reference
//! `incoherent_form_factor.evaluate(x)`.
//!
//! Pure lookup (no RNG), so the extraction is byte-identical and the test is
//! deterministic.

use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// Evaluate S at momentum-transfer `x` by lin-lin interpolation of the
/// tabulated incoherent form factor `(iff_x, iff_s)` (per-material block of
/// `iff_n_pts` points starting at `iff_off`), clamping to the end values
/// outside the table. Must be called with `iff_n_pts >= 2`. Matches the CPU
/// `Tabulated1D::evaluate`. THE single source of truth, called by both the
/// test launcher and the inline Compton form-factor rejection.
#[cube]
pub fn incoherent_s_at(x: f64, iff_off: u32, iff_n_pts: u32, iff_x: &[f64], iff_s: &[f64]) -> f64 {
    let x_first = iff_x[iff_off as usize];
    let x_last = iff_x[(iff_off + iff_n_pts - 1u32) as usize];
    let s_first = iff_s[iff_off as usize];
    let s_last = iff_s[(iff_off + iff_n_pts - 1u32) as usize];
    if x <= x_first {
        s_first
    } else if x >= x_last {
        s_last
    } else {
        let mut lo = 0u32;
        let mut hi = iff_n_pts;
        let mut it = 0u32;
        while it < 16u32 && lo + 1u32 < hi {
            let mid = (lo + hi) / 2u32;
            if iff_x[(iff_off + mid) as usize] <= x {
                lo = mid;
            } else {
                hi = mid;
            }
            it += 1u32;
        }
        let xl = iff_x[(iff_off + lo) as usize];
        let xr = iff_x[(iff_off + lo + 1u32) as usize];
        let sl = iff_s[(iff_off + lo) as usize];
        let sr = iff_s[(iff_off + lo + 1u32) as usize];
        let denom = xr - xl;
        let frac = (x - xl) / denom;
        sl + (sr - sl) * frac
    }
}

/// Per-thread test launcher: evaluate S at each query `x_query[tid]` against
/// a single contiguous form-factor table (`iff_off = 0`).
#[cube(launch_unchecked)]
fn incoherent_s_kernel(x_query: &[f64], iff_x: &[f64], iff_s: &[f64], out_s: &mut [f64]) {
    if ABSOLUTE_POS >= x_query.len() {
        terminate!();
    }
    let n_pts = iff_x.len() as u32;
    out_s[ABSOLUTE_POS] = incoherent_s_at(x_query[ABSOLUTE_POS], 0u32, n_pts, iff_x, iff_s);
}

/// Evaluate S(x) on the GPU for each `x_query`, against the tabulated
/// `(iff_x, iff_s)` incoherent form factor (same length). Returns S per query.
pub fn run_incoherent_s(
    ctx: &GpuContext,
    x_query: &[f64],
    iff_x: &[f64],
    iff_s: &[f64],
) -> Vec<f64> {
    assert_eq!(iff_x.len(), iff_s.len());
    let client = ctx.client();
    let n = x_query.len();
    let xq_handle = client.create_from_slice(bytemuck::cast_slice(x_query));
    let ix_handle = client.create_from_slice(bytemuck::cast_slice(iff_x));
    let is_handle = client.create_from_slice(bytemuck::cast_slice(iff_s));
    let out_handle = client.empty(std::mem::size_of_val(x_query));

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        incoherent_s_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(xq_handle, n),
            BufferArg::from_raw_parts(ix_handle, iff_x.len()),
            BufferArg::from_raw_parts(is_handle, iff_s.len()),
            BufferArg::from_raw_parts(out_handle.clone(), n),
        );
    }

    bytemuck::cast_slice(&client.read_one(out_handle).unwrap()).to_vec()
}
