//! Compton (incoherent) scatter kinematics on the GPU -- the
//! free-electron Klein-Nishina energy/angle sampler via Kahn's
//! rejection method, extracted from the inline block in
//! `multi_cell_photon_transport` so it can be unit-tested in isolation
//! against the CPU reference `yamc_element::photon::klein_nishina`.
//!
//! This is the per-scatter energy-loss core. The full transport adds the
//! incoherent scattering function S(x,Z) and Doppler broadening on top of
//! this; those are separate samplers (follow-up extractions). Isolating
//! the bare Klein-Nishina here lets a distribution test catch a per-scatter
//! energy-loss bias directly, instead of only via a downstream tally
//! (the gap that hid #415).
//!
//! Kahn's method (valid for `alpha < 3`, i.e. E < ~1.53 MeV; the CPU
//! switches to a different sampler above that, and so should this kernel
//! before it's wired into the main loop for high-E photons):
//! - With probability `(2α+1)/(2α+9)` take the "left" branch:
//!   `x = 1 + 2α·r2` (so `x ∈ [1, 1+2α]`), accept if `r3·x² ≤ 4(x−1)`;
//!   then `α' = α/x`, `µ = 1 − (x−1)/α`.
//! - Else the "right" branch: `x = (2α+1)/(1+2α·r2)`,
//!   `µ = 1 − (x−1)/α`, accept if `r3 ≤ ½(µ² + 1/x)`; `α' = α/x`.

use crate::common::pcg32::{expand_seed, pcg_out};
use crate::common::rng::{PCG_INCR, PCG_MULT};
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// Result of ONE free-electron Klein-Nishina (Kahn) proposal -- a single
/// 3-random-draw attempt (one branch, one accept test). Carries the
/// advanced PCG state back so callers compose it inside their own
/// rejection loop without re-seeding. `accepted == 0` means this proposal
/// was KN-rejected (caller should draw again).
///
/// Deliberately ONE proposal, not a full loop: the inline Compton branch of
/// `multi_cell_photon_transport` runs a single rejection loop that shares
/// its iteration cap across both the Kahn KN-rejection and the incoherent-
/// form-factor rejection. Keeping the helper a single proposal lets the
/// mega kernel preserve that exact loop (and so byte-identical sampling),
/// while the test launcher wraps it in its own loop.
#[derive(CubeType)]
pub struct KahnProposal {
    /// Advanced PCG-32 state after this proposal's 3 draws.
    pub state: u64,
    /// Scattered photon energy in electron-mass units, α' = α/x.
    pub alpha_out: f64,
    /// Scattering cosine µ.
    pub mu: f64,
    /// 1 if this proposal was accepted, 0 if KN-rejected.
    pub accepted: u32,
}

/// One free-electron Klein-Nishina proposal via Kahn's method (3 PCG draws,
/// one branch, one accept test). THE single source of truth for the Kahn
/// step -- called both by the test launcher below and by the inline Compton
/// branch of `multi_cell_photon_transport`. When `accepted == 0` the values
/// `alpha_out`/`mu` are the incident `alpha` / `1.0` and the caller retries.
#[cube]
pub fn compton_kahn_propose(state: u64, alpha: f64) -> KahnProposal {
    let mut st = state;
    let mut alpha_out = alpha;
    let mut mu = 1.0_f64;
    let mut accepted = 0u32;

    let two_alpha_p1 = 2.0_f64 * alpha + 1.0_f64;
    let two_alpha_p9 = 2.0_f64 * alpha + 9.0_f64;

    let s1 = st;
    let r_1 = pcg_out(s1);
    st = s1 * PCG_MULT + PCG_INCR;
    let r1 = (r_1 as f64 + 1.0_f64) * (1.0_f64 / 4_294_967_297.0_f64);

    let s2 = st;
    let r_2 = pcg_out(s2);
    st = s2 * PCG_MULT + PCG_INCR;
    let r2 = (r_2 as f64 + 1.0_f64) * (1.0_f64 / 4_294_967_297.0_f64);

    let s3 = st;
    let r_3 = pcg_out(s3);
    st = s3 * PCG_MULT + PCG_INCR;
    let r3 = (r_3 as f64 + 1.0_f64) * (1.0_f64 / 4_294_967_297.0_f64);

    if two_alpha_p1 / two_alpha_p9 > r1 {
        // Left branch.
        let x = 1.0_f64 + 2.0_f64 * alpha * r2;
        if r3 * x * x <= 4.0_f64 * (x - 1.0_f64) {
            alpha_out = alpha / x;
            mu = 1.0_f64 - (x - 1.0_f64) / alpha;
            accepted = 1u32;
        }
    } else {
        // Right branch.
        let x = two_alpha_p1 / (1.0_f64 + 2.0_f64 * alpha * r2);
        let mu_try = 1.0_f64 - (x - 1.0_f64) / alpha;
        let acc = 0.5_f64 * (mu_try * mu_try + 1.0_f64 / x);
        if r3 <= acc {
            alpha_out = alpha / x;
            mu = mu_try;
            accepted = 1u32;
        }
    }

    KahnProposal {
        state: st,
        alpha_out,
        mu,
        accepted,
    }
}

/// Per-thread test launcher: loops `compton_kahn_propose` until a sample is
/// accepted (free-electron Klein-Nishina, no form factor), then writes
/// `out_eratio[tid] = α'/α` and `out_mu[tid] = µ`.
#[cube(launch_unchecked)]
fn compton_kahn_kernel(seeds: &[u32], alphas: &[f64], out_eratio: &mut [f64], out_mu: &mut [f64]) {
    if ABSOLUTE_POS >= seeds.len() {
        terminate!();
    }
    let alpha = alphas[ABSOLUTE_POS];
    let mut state = expand_seed(seeds[ABSOLUTE_POS]);
    let mut alpha_out = alpha;
    let mut mu = 1.0_f64;
    let mut accepted = 0u32;
    let mut k_iter = 0u32;
    while k_iter < 64u32 && accepted == 0u32 {
        let p = compton_kahn_propose(state, alpha);
        state = p.state;
        if p.accepted == 1u32 {
            alpha_out = p.alpha_out;
            mu = p.mu;
            accepted = 1u32;
        }
        k_iter += 1u32;
    }
    out_eratio[ABSOLUTE_POS] = alpha_out / alpha;
    out_mu[ABSOLUTE_POS] = mu;
}

/// Run the free-electron Klein-Nishina (Kahn) sampler. Each
/// `(seeds[i], alphas[i])` is one independent sample. Returns
/// `(e_ratio, mu)` = (E'/E, cosine) in input order.
pub fn run_compton_kahn(ctx: &GpuContext, seeds: &[u32], alphas: &[f64]) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(seeds.len(), alphas.len());
    let client = ctx.client();
    let n = seeds.len();
    let seeds_handle = client.create_from_slice(bytemuck::cast_slice(seeds));
    let alphas_handle = client.create_from_slice(bytemuck::cast_slice(alphas));
    let out_eratio_handle = client.empty(std::mem::size_of_val(alphas));
    let out_mu_handle = client.empty(std::mem::size_of_val(alphas));

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        compton_kahn_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(seeds_handle, n),
            BufferArg::from_raw_parts(alphas_handle, n),
            BufferArg::from_raw_parts(out_eratio_handle.clone(), n),
            BufferArg::from_raw_parts(out_mu_handle.clone(), n),
        );
    }

    let e_ratio: Vec<f64> =
        bytemuck::cast_slice(&client.read_one(out_eratio_handle).unwrap()).to_vec();
    let mu: Vec<f64> = bytemuck::cast_slice(&client.read_one(out_mu_handle).unwrap()).to_vec();
    (e_ratio, mu)
}
