//! Atomic-relaxation per-hop transition sampling on the GPU -- given a
//! vacancy in subshell `i`, sample one radiative/non-radiative transition by
//! inverse-CDF on that shell's cumulative transition probabilities, returning
//! the (primary vacancy, secondary, transition energy). Extracted from the
//! inline cascade in `multi_cell_photon_transport` so the cascade loop calls
//! it (single source of truth) and the per-hop sampling can be unit-tested.
//!
//! Just the single-transition sampler (one PCG draw + CDF binary search); the
//! multi-hop cascade state machine (hole stack, depth cap, secondary banking)
//! stays inline. Byte-identical to the old inline code.
//!
//! The CPU `PhotonInteraction::atomic_relaxation` only exposes the full
//! cascade, so the unit test compares this sampler's transition-index
//! distribution to the transition CDF it shares with the CPU.

use crate::common::pcg32::{expand_seed, pcg_out};
use crate::common::rng::{PCG_INCR, PCG_MULT};
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// One sampled atomic-relaxation transition.
#[derive(CubeType)]
pub struct RelaxTransition {
    /// Advanced PCG-32 state.
    pub state: u64,
    /// Sampled transition index within the shell's table.
    pub t_idx: u32,
    /// Primary (originating) vacancy subshell for the next cascade hop.
    pub primary: u32,
    /// Secondary subshell (Auger) or sentinel for a radiative transition.
    pub secondary: u32,
    /// Transition energy (eV).
    pub energy: f64,
}

/// Sample one transition for a vacancy. `cum_prob`/`primary_arr`/
/// `secondary_arr`/`energy_arr` are the flat per-(shell) transition tables;
/// `t_off` is the shell's base offset and `n_trans_s` its transition count
/// (>0). THE single source of truth, called by both the test launcher and the
/// inline relaxation cascade of `multi_cell_photon_transport`.
#[cube]
pub fn sample_relax_transition(
    state: u64,
    t_off: u32,
    n_trans_s: u32,
    cum_prob: &[f64],
    primary_arr: &[u32],
    secondary_arr: &[u32],
    energy_arr: &[f64],
) -> RelaxTransition {
    let mut st = state;
    let s_tr = st;
    let r_tr = pcg_out(s_tr);
    st = s_tr * PCG_MULT + PCG_INCR;
    let xi_tr = (r_tr as f64 + 1.0) * (1.0 / 4_294_967_297.0);

    let mut t_lo = 0u32;
    let mut t_hi = n_trans_s;
    let mut t_iter = 0u32;
    while t_iter < 16u32 && t_lo < t_hi {
        let t_mid = (t_lo + t_hi) / 2u32;
        if cum_prob[(t_off + t_mid) as usize] > xi_tr {
            t_hi = t_mid;
        } else {
            t_lo = t_mid + 1u32;
        }
        t_iter += 1u32;
    }
    let mut t_idx = t_lo;
    if t_idx >= n_trans_s {
        t_idx = n_trans_s - 1u32;
    }
    let primary = primary_arr[(t_off + t_idx) as usize];
    let secondary = secondary_arr[(t_off + t_idx) as usize];
    let e_trans = energy_arr[(t_off + t_idx) as usize];

    RelaxTransition {
        state: st,
        t_idx,
        primary,
        secondary,
        energy: e_trans,
    }
}

/// Per-thread test launcher: single shell's transition table (`t_off = 0`).
/// Writes the sampled transition index to `out_tidx[tid]`.
#[cube(launch_unchecked)]
fn relax_transition_kernel(
    seeds: &[u32],
    cum_prob: &[f64],
    primary_arr: &[u32],
    secondary_arr: &[u32],
    energy_arr: &[f64],
    out_tidx: &mut [u32],
) {
    if ABSOLUTE_POS >= seeds.len() {
        terminate!();
    }
    let n_trans = cum_prob.len() as u32;
    let state = expand_seed(seeds[ABSOLUTE_POS]);
    let rt = sample_relax_transition(
        state,
        0u32,
        n_trans,
        cum_prob,
        primary_arr,
        secondary_arr,
        energy_arr,
    );
    out_tidx[ABSOLUTE_POS] = rt.t_idx;
}

/// Run the per-hop transition sampler for one shell's transition table
/// (`cum_prob`/`primary`/`secondary`/`energy`, all the same length). Returns
/// the sampled transition index per seed.
pub fn run_relax_transition(
    ctx: &GpuContext,
    seeds: &[u32],
    cum_prob: &[f64],
    primary: &[u32],
    secondary: &[u32],
    energy: &[f64],
) -> Vec<u32> {
    let client = ctx.client();
    let n = seeds.len();
    let seeds_h = client.create_from_slice(bytemuck::cast_slice(seeds));
    let cp_h = client.create_from_slice(bytemuck::cast_slice(cum_prob));
    let pr_h = client.create_from_slice(bytemuck::cast_slice(primary));
    let se_h = client.create_from_slice(bytemuck::cast_slice(secondary));
    let en_h = client.create_from_slice(bytemuck::cast_slice(energy));
    let out_h = client.empty(std::mem::size_of_val(seeds));

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        relax_transition_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(seeds_h, n),
            BufferArg::from_raw_parts(cp_h, cum_prob.len()),
            BufferArg::from_raw_parts(pr_h, primary.len()),
            BufferArg::from_raw_parts(se_h, secondary.len()),
            BufferArg::from_raw_parts(en_h, energy.len()),
            BufferArg::from_raw_parts(out_h.clone(), n),
        );
    }

    bytemuck::cast_slice(&client.read_one(out_h).unwrap()).to_vec()
}
