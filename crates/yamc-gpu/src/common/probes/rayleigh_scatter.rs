//! Rayleigh (coherent) photon scattering angle sampling on the GPU --
//! one inverse-CDF proposal on the integrated coherent form factor
//! F(x², Z), with a Klein-Nishina-style angular acceptance 0.5·(1+µ²).
//! Extracted from the inline block in `multi_cell_photon_transport` so the
//! mega kernel calls it (single source of truth) and it can be unit-tested
//! against the CPU reference `PhotonInteraction::rayleigh_scatter`.
//!
//! ONE proposal (2 PCG draws: CDF sample + angular rejection), not a loop,
//! so the mega kernel keeps its exact rejection loop and the change is
//! byte-identical. `accepted == 0` means the angular rejection failed and
//! the caller should propose again. Rayleigh is elastic -- no energy change,
//! only `mu` is produced.

use crate::common::pcg32::{expand_seed, pcg_out};
use crate::common::rng::{PCG_INCR, PCG_MULT};
use crate::photon::transport::MASS_ELECTRON_EV;
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// h·c in eV·angstrom (so the momentum-transfer x has units 1/angstrom,
/// matching the CPU `PLANCK_C` convention).
const PLANCK_C_EVA: f64 = 12_398.419_843_320_026;

/// Result of ONE Rayleigh angle proposal.
#[derive(CubeType)]
pub struct RayleighProposal {
    /// Advanced PCG-32 state after this proposal's 2 draws.
    pub state: u64,
    /// Scattering cosine µ (valid only when `accepted == 1`).
    pub mu: f64,
    /// 1 if the angular rejection accepted, 0 otherwise.
    pub accepted: u32,
}

/// One Rayleigh (coherent) angle proposal. Inverse-CDF samples x² from the
/// integrated coherent form factor (`rayleigh_cdf` = F values, `rayleigh_x2`
/// = x² axis, `n_ff` valid points starting at `ff_off`, `max_ff` stride),
/// converts to µ = 1 − 2x²/x²_max, then applies the 0.5·(1+µ²) acceptance.
/// Must be called with `n_ff >= 2`. THE single source of truth, called by
/// both the test launcher and the inline Rayleigh branch of
/// `multi_cell_photon_transport`.
#[cube]
pub fn rayleigh_propose(
    state: u64,
    energy: f64,
    ff_off: u32,
    n_ff: u32,
    max_ff: u32,
    rayleigh_x2: &[f64],
    rayleigh_cdf: &[f64],
) -> RayleighProposal {
    let mut st = state;
    let mut mu = 1.0_f64;
    let mut accepted = 0u32;

    let alpha_r = energy / MASS_ELECTRON_EV;
    let kappa = MASS_ELECTRON_EV / PLANCK_C_EVA * alpha_r;
    let x2_max = kappa * kappa;
    // f_max = F(x²_max): interpolate the integrated form factor at the
    // kinematic limit, matching the CPU `coherent_int_form_factor.evaluate
    // (x²_max)`. (#421: previously used the full-table integral
    // `rayleigh_cdf[last]`, which sampled x² beyond x²_max -> mu clamped to
    // -1 -> excess backscatter.) Edge-clamp + binary-search lin-interp; when
    // x²_max exceeds the table the clamp recovers the old full integral.
    let x2_first = rayleigh_x2[ff_off as usize];
    let x2_last = rayleigh_x2[(ff_off + n_ff - 1u32) as usize];
    let f_max = if x2_max <= x2_first {
        rayleigh_cdf[ff_off as usize]
    } else if x2_max >= x2_last {
        rayleigh_cdf[(ff_off + n_ff - 1u32) as usize]
    } else {
        let mut lo = 0u32;
        let mut hi = n_ff;
        let mut it = 0u32;
        while it < 16u32 && lo + 1u32 < hi {
            let mid = (lo + hi) / 2u32;
            if rayleigh_x2[(ff_off + mid) as usize] <= x2_max {
                lo = mid;
            } else {
                hi = mid;
            }
            it += 1u32;
        }
        let xl = rayleigh_x2[(ff_off + lo) as usize];
        let xr = rayleigh_x2[(ff_off + lo + 1u32) as usize];
        let yl = rayleigh_cdf[(ff_off + lo) as usize];
        let yr = rayleigh_cdf[(ff_off + lo + 1u32) as usize];
        yl + (yr - yl) * (x2_max - xl) / (xr - xl)
    };

    let s_r1 = st;
    let r_r1 = pcg_out(s_r1);
    st = s_r1 * PCG_MULT + PCG_INCR;
    let r1 = (r_r1 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
    let f_sample = r1 * f_max;

    // Flat scan for the bracket where cdf[i] <= f_sample < cdf[i+1].
    let mut i_b = 0u32;
    let mut kk = 0u32;
    while kk + 1u32 < max_ff {
        let in_range = kk + 1u32 < n_ff;
        let yk = rayleigh_cdf[(ff_off + kk) as usize];
        let yk1 = rayleigh_cdf[(ff_off + kk + 1u32) as usize];
        if in_range && yk <= f_sample && f_sample < yk1 {
            i_b = kk;
        }
        kk += 1u32;
    }
    let y_lo = rayleigh_cdf[(ff_off + i_b) as usize];
    let y_hi = rayleigh_cdf[(ff_off + i_b + 1u32) as usize];
    let denom_y = y_hi - y_lo;
    let mut r = 0.0_f64;
    if denom_y > 1e-30 {
        r = (f_sample - y_lo) / denom_y;
    }
    let x_lo_v = rayleigh_x2[(ff_off + i_b) as usize];
    let x_hi_v = rayleigh_x2[(ff_off + i_b + 1u32) as usize];
    let x2 = x_lo_v + r * (x_hi_v - x_lo_v);
    let mut mu_try = 1.0_f64;
    if x2_max > 0.0 {
        mu_try = 1.0 - 2.0 * x2 / x2_max;
    }
    // clamp to [-1, 1] (byte-identical to the old if-pattern; .clamp is
    // cube-compatible and avoids clippy::manual_clamp under -D warnings).
    mu_try = mu_try.clamp(-1.0, 1.0);

    let s_r2 = st;
    let r_r2 = pcg_out(s_r2);
    st = s_r2 * PCG_MULT + PCG_INCR;
    let r2 = (r_r2 as f64 + 1.0) * (1.0 / 4_294_967_297.0);
    if r2 < 0.5 * (1.0 + mu_try * mu_try) {
        mu = mu_try;
        accepted = 1u32;
    }

    RayleighProposal {
        state: st,
        mu,
        accepted,
    }
}

/// Per-thread test launcher: loops `rayleigh_propose` until accepted (single
/// material, `ff_off = 0`), then writes `out_mu[tid] = µ`. The form-factor
/// table (`x2`, `cdf`) is one element's `coherent_int_form_factor`.
#[cube(launch_unchecked)]
fn rayleigh_kernel(
    seeds: &[u32],
    energies: &[f64],
    rayleigh_x2: &[f64],
    rayleigh_cdf: &[f64],
    out_mu: &mut [f64],
) {
    if ABSOLUTE_POS >= seeds.len() {
        terminate!();
    }
    // Single contiguous element table -> n_ff and stride are the slice len.
    let n_ff = rayleigh_x2.len() as u32;
    let energy = energies[ABSOLUTE_POS];
    let mut state = expand_seed(seeds[ABSOLUTE_POS]);
    let mut mu = 1.0_f64;
    let mut accepted = 0u32;
    let mut k_iter = 0u32;
    while k_iter < 64u32 && accepted == 0u32 {
        let p = rayleigh_propose(state, energy, 0u32, n_ff, n_ff, rayleigh_x2, rayleigh_cdf);
        state = p.state;
        if p.accepted == 1u32 {
            mu = p.mu;
            accepted = 1u32;
        }
        k_iter += 1u32;
    }
    out_mu[ABSOLUTE_POS] = mu;
}

/// Run the Rayleigh angle sampler. `x2` / `cdf` are one element's integrated
/// coherent form factor (x² axis and F values, same length). Returns the
/// sampled `mu` per input seed.
pub fn run_rayleigh(
    ctx: &GpuContext,
    seeds: &[u32],
    energies: &[f64],
    x2: &[f64],
    cdf: &[f64],
) -> Vec<f64> {
    assert_eq!(seeds.len(), energies.len());
    assert_eq!(x2.len(), cdf.len());
    let client = ctx.client();
    let n = seeds.len();
    let seeds_handle = client.create_from_slice(bytemuck::cast_slice(seeds));
    let energies_handle = client.create_from_slice(bytemuck::cast_slice(energies));
    let x2_handle = client.create_from_slice(bytemuck::cast_slice(x2));
    let cdf_handle = client.create_from_slice(bytemuck::cast_slice(cdf));
    let out_mu_handle = client.empty(std::mem::size_of_val(energies));

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        rayleigh_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(seeds_handle, n),
            BufferArg::from_raw_parts(energies_handle, n),
            BufferArg::from_raw_parts(x2_handle, x2.len()),
            BufferArg::from_raw_parts(cdf_handle, cdf.len()),
            BufferArg::from_raw_parts(out_mu_handle.clone(), n),
        );
    }

    bytemuck::cast_slice(&client.read_one(out_mu_handle).unwrap()).to_vec()
}
