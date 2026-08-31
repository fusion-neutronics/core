//! Compton Doppler broadening on the GPU -- given the incident energy, the
//! sampled Klein-Nishina cosine `mu`, and the free-electron KN result, sample
//! a bound-electron shell, invert its Compton-profile CDF for `pz`, and solve
//! the bound-electron Compton kinematics quadratic for the broadened `E_out`.
//! Extracted from the inline block in `multi_cell_photon_transport` so the
//! mega kernel calls it (single source of truth) and it can be unit-tested
//! against the CPU reference `PhotonInteraction::compton_doppler`.
//!
//! Returns `e_out_kn` unchanged when the material has no Doppler data or the
//! shell sample falls below threshold (the CPU fallback). Three PCG draws
//! (shell, CDF inversion, root pick). Byte-identical to the old inline code.

use crate::common::pcg32::{expand_seed, pcg_out};
use crate::common::rng::{PCG_INCR, PCG_MULT};
use crate::photon::transport::{DOP_MAX_PZ, DOP_MAX_SHELLS, FINE_STRUCTURE, MASS_ELECTRON_EV};
use crate::{GpuContext, WgpuRuntime};
use cubecl::prelude::*;

/// Result of one Doppler-broadened Compton energy sample.
#[derive(CubeType)]
pub struct DopplerSample {
    /// Advanced PCG-32 state.
    pub state: u64,
    /// Doppler-broadened outgoing photon energy (eV); equals `e_out_kn` when
    /// Doppler is skipped.
    pub e_out: f64,
    /// Compton-profile shell sampled from `dop_electron_pdf` (the shell the
    /// event ionizes). Set for every code path once a shell is drawn, matching
    /// the CPU `compton_doppler`, which returns `i_shell` regardless of the
    /// Doppler skip / no-root branches. `u32::MAX` (the none-shell sentinel)
    /// means no shell was sampled (no Doppler data), mirroring the CPU
    /// `i_shell = -1` case; the caller then skips atomic relaxation (its
    /// `< DOP_MAX_SHELLS` guard rejects the sentinel).
    pub shell: u32,
}

/// Sample the Doppler-broadened Compton `E_out`. `e_in` is the incident
/// photon energy (eV), `mu` the KN cosine, `e_out_kn` the free-electron KN
/// result (the fallback). The `dop_*` slices are the per-material Compton
/// profile tables (`mat_idx` selects the material). THE single source of
/// truth, called by both the test launcher and the inline Compton branch of
/// `multi_cell_photon_transport`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn compton_doppler_sample(
    state: u64,
    e_in: f64,
    mu: f64,
    e_out_kn: f64,
    mat_idx: u32,
    dop_n_shells: &[u32],
    dop_has_data: &[u32],
    dop_pz_grid: &[f64],
    dop_electron_pdf: &[f64],
    dop_binding_energy: &[f64],
    dop_profile_pdf: &[f64],
    dop_profile_cdf: &[f64],
) -> DopplerSample {
    let mut st = state;
    let mut e_out_final = e_out_kn;
    // Sampled Compton-profile shell, returned to the caller for atomic
    // relaxation. `u32::MAX` = "no shell sampled" (no Doppler data): the
    // kernel's universal none-shell sentinel, and `< DOP_MAX_SHELLS` on the
    // caller side rejects it. Mirrors the CPU `compton_scatter` returning
    // `i_shell = -1` when doppler is off.
    let mut shell_out = 4_294_967_295u32;
    let dop_n_shells_m = dop_n_shells[mat_idx as usize];
    let n_pz_dop: u32 = dop_pz_grid.len() as u32;
    if dop_has_data[mat_idx as usize] == 1u32 && dop_n_shells_m >= 1u32 && n_pz_dop >= 2u32 {
        let e_in_dop = e_in;
        let one_minus_mu = 1.0_f64 - mu;
        let shell_off = mat_idx * DOP_MAX_SHELLS;
        // Sample shell from electron_pdf cumulative.
        let s_xi_shell = st;
        let r_xi_shell = pcg_out(s_xi_shell);
        st = s_xi_shell * PCG_MULT + PCG_INCR;
        let xi_shell = (r_xi_shell as f64 + 1.0_f64) * (1.0_f64 / 4_294_967_297.0_f64);
        let mut cumsum_sh = 0.0_f64;
        let mut i_shell = dop_n_shells_m - 1u32;
        let mut sh_found = 0u32;
        let mut sh_iter = 0u32;
        while sh_iter < dop_n_shells_m && sh_found == 0u32 {
            cumsum_sh += dop_electron_pdf[(shell_off + sh_iter) as usize];
            if cumsum_sh > xi_shell {
                i_shell = sh_iter;
                sh_found = 1u32;
            }
            sh_iter += 1u32;
        }
        // Record the sampled shell for ALL downstream code paths (Doppler skip,
        // no-root fallback, accepted root), matching the CPU `compton_doppler`
        // which returns `i_shell` unconditionally once a shell is drawn.
        shell_out = i_shell;
        let e_b = dop_binding_energy[(shell_off + i_shell) as usize];
        let e_minus_eb = e_in_dop - e_b;
        let denom_max_sq = 2.0_f64 * e_in_dop * e_minus_eb * one_minus_mu + e_b * e_b;
        let denom_max = denom_max_sq.sqrt();
        let alpha_dop = e_in_dop / MASS_ELECTRON_EV;
        let term_pzmax = e_b - e_minus_eb * alpha_dop * one_minus_mu;
        let pz_max = (0.0_f64 - FINE_STRUCTURE) * term_pzmax / denom_max;

        let dop_skip = e_in_dop < e_b || pz_max < 0.0_f64 || denom_max_sq <= 0.0_f64;

        if !dop_skip {
            let table_off = (shell_off + i_shell) * DOP_MAX_PZ;
            let cdf_last_idx = n_pz_dop - 1u32;
            let pz_last = dop_pz_grid[cdf_last_idx as usize];

            let cmax_full = dop_profile_cdf[(table_off + cdf_last_idx) as usize];
            let mut ip_lo = 0u32;
            let mut ip_hi = n_pz_dop;
            let mut ip_it = 0u32;
            while ip_it < 16u32 && ip_lo + 1u32 < ip_hi {
                let mid = (ip_lo + ip_hi) / 2u32;
                if dop_pz_grid[mid as usize] <= pz_max {
                    ip_lo = mid;
                } else {
                    ip_hi = mid;
                }
                ip_it += 1u32;
            }
            let mut i_pz = ip_lo;
            if i_pz >= n_pz_dop - 1u32 {
                i_pz = n_pz_dop - 2u32;
            }
            let pz_l = dop_pz_grid[i_pz as usize];
            let pz_r = dop_pz_grid[(i_pz + 1u32) as usize];
            let p_l_max = dop_profile_pdf[(table_off + i_pz) as usize];
            let p_r_max = dop_profile_pdf[(table_off + i_pz + 1u32) as usize];
            let c_l_max = dop_profile_cdf[(table_off + i_pz) as usize];
            let dpz_bracket_neg = pz_l - pz_r;
            let dp_bracket = p_l_max - p_r_max;
            let c_partial_flat = c_l_max + (pz_max - pz_l) * p_l_max;
            let slope_bracket = dp_bracket / dpz_bracket_neg;
            let term_q = slope_bracket * (pz_max - pz_l) + p_l_max;
            let c_partial_q =
                c_l_max + (term_q * term_q - p_l_max * p_l_max) / (2.0_f64 * slope_bracket);
            let c_partial = if (p_l_max - p_r_max).abs() < 1e-30_f64 {
                c_partial_flat
            } else {
                c_partial_q
            };
            let c_max = if pz_max > pz_last {
                cmax_full
            } else {
                c_partial
            };

            if c_max > 0.0_f64 {
                let s_xi_c = st;
                let r_xi_c = pcg_out(s_xi_c);
                st = s_xi_c * PCG_MULT + PCG_INCR;
                let xi_c = (r_xi_c as f64 + 1.0_f64) * (1.0_f64 / 4_294_967_297.0_f64);
                let c_target = xi_c * c_max;

                let mut ic_lo = 0u32;
                let mut ic_hi = n_pz_dop;
                let mut ic_it = 0u32;
                while ic_it < 16u32 && ic_lo + 1u32 < ic_hi {
                    let mid = (ic_lo + ic_hi) / 2u32;
                    if dop_profile_cdf[(table_off + mid) as usize] <= c_target {
                        ic_lo = mid;
                    } else {
                        ic_hi = mid;
                    }
                    ic_it += 1u32;
                }
                let mut i_c = ic_lo;
                if i_c >= n_pz_dop - 1u32 {
                    i_c = n_pz_dop - 2u32;
                }
                let pz_l_c = dop_pz_grid[i_c as usize];
                let pz_r_c = dop_pz_grid[(i_c + 1u32) as usize];
                let p_l_c = dop_profile_pdf[(table_off + i_c) as usize];
                let p_r_c = dop_profile_pdf[(table_off + i_c + 1u32) as usize];
                let c_l_c = dop_profile_cdf[(table_off + i_c) as usize];

                let dpz_c_neg = pz_l_c - pz_r_c;
                let dp_c = p_l_c - p_r_c;
                let slope_c = dp_c / dpz_c_neg;
                let disc_pz = p_l_c * p_l_c + 2.0_f64 * slope_c * (c_target - c_l_c);
                let pz_sample_flat = pz_l_c + (c_target - c_l_c) / p_l_c;
                let pz_sample_q = pz_l_c + (disc_pz.sqrt() - p_l_c) / slope_c;
                let pz_sample = if (p_l_c - p_r_c).abs() < 1e-30_f64 {
                    pz_sample_flat
                } else if disc_pz >= 0.0_f64 {
                    pz_sample_q
                } else {
                    pz_l_c
                };

                let p_alpha = pz_sample / FINE_STRUCTURE;
                let momentum_sq = p_alpha * p_alpha;
                let alpha_in = e_in_dop / MASS_ELECTRON_EV;
                let f_val = 1.0_f64 + alpha_in * one_minus_mu;
                let a_coeff = momentum_sq - f_val * f_val;
                let b_coeff = 2.0_f64 * e_in_dop * (f_val - momentum_sq * mu);
                let c_coeff = e_in_dop * e_in_dop * (momentum_sq - 1.0_f64);
                let quad = b_coeff * b_coeff - 4.0_f64 * a_coeff * c_coeff;
                if quad >= 0.0_f64 {
                    let sqrt_quad = quad.sqrt();
                    let e_out_1 = (0.0_f64 - (b_coeff + sqrt_quad)) / (2.0_f64 * a_coeff);
                    let e_out_2 = (0.0_f64 - (b_coeff - sqrt_quad)) / (2.0_f64 * a_coeff);

                    let s_xi_p = st;
                    let r_xi_p = pcg_out(s_xi_p);
                    st = s_xi_p * PCG_MULT + PCG_INCR;
                    let xi_p = (r_xi_p as f64 + 1.0_f64) * (1.0_f64 / 4_294_967_297.0_f64);
                    let both_pos = e_out_1 > 0.0_f64 && e_out_2 > 0.0_f64;
                    let e_out_pick = if both_pos {
                        if xi_p < 0.5_f64 {
                            e_out_1
                        } else {
                            e_out_2
                        }
                    } else if e_out_1 > 0.0_f64 {
                        e_out_1
                    } else {
                        e_out_2
                    };
                    let e_limit = e_in_dop - e_b;
                    if e_out_pick > 0.0_f64 && e_out_pick < e_limit {
                        e_out_final = e_out_pick;
                    }
                }
            }
        }
    }

    DopplerSample {
        state: st,
        e_out: e_out_final,
        shell: shell_out,
    }
}

/// Per-thread test launcher: single material (`mat_idx = 0`), fixed incident
/// energy + `mu`. Writes the Doppler-broadened `E_out` to `out_e[tid]`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn compton_doppler_kernel(
    seeds: &[u32],
    e_in_buf: &[f64],
    mu_buf: &[f64],
    dop_n_shells: &[u32],
    dop_has_data: &[u32],
    dop_pz_grid: &[f64],
    dop_electron_pdf: &[f64],
    dop_binding_energy: &[f64],
    dop_profile_pdf: &[f64],
    dop_profile_cdf: &[f64],
    out_e: &mut [f64],
) {
    if ABSOLUTE_POS >= seeds.len() {
        terminate!();
    }
    let e_in = e_in_buf[0usize];
    let mu = mu_buf[0usize];
    // Free-electron KN result as the fallback (e_out_kn).
    let alpha = e_in / MASS_ELECTRON_EV;
    let e_out_kn = alpha / (1.0_f64 + alpha * (1.0_f64 - mu)) * MASS_ELECTRON_EV;
    let state = expand_seed(seeds[ABSOLUTE_POS]);
    let ds = compton_doppler_sample(
        state,
        e_in,
        mu,
        e_out_kn,
        0u32,
        dop_n_shells,
        dop_has_data,
        dop_pz_grid,
        dop_electron_pdf,
        dop_binding_energy,
        dop_profile_pdf,
        dop_profile_cdf,
    );
    out_e[ABSOLUTE_POS] = ds.e_out;
}

/// Run the Doppler sampler for a single material at fixed `(e_in, mu)`.
#[allow(clippy::too_many_arguments)]
pub fn run_compton_doppler(
    ctx: &GpuContext,
    seeds: &[u32],
    e_in: f64,
    mu: f64,
    pz_grid: &[f64],
    n_shells: u32,
    electron_pdf: &[f64],
    binding_energy: &[f64],
    profile_pdf: &[f64],
    profile_cdf: &[f64],
) -> Vec<f64> {
    let client = ctx.client();
    let n = seeds.len();
    let seeds_h = client.create_from_slice(bytemuck::cast_slice(seeds));
    let e_in_h = client.create_from_slice(bytemuck::cast_slice(&[e_in]));
    let mu_h = client.create_from_slice(bytemuck::cast_slice(&[mu]));
    let ns_h = client.create_from_slice(bytemuck::cast_slice(&[n_shells]));
    let has_h = client.create_from_slice(bytemuck::cast_slice(&[1u32]));
    let pz_h = client.create_from_slice(bytemuck::cast_slice(pz_grid));
    let epdf_h = client.create_from_slice(bytemuck::cast_slice(electron_pdf));
    let be_h = client.create_from_slice(bytemuck::cast_slice(binding_energy));
    let ppdf_h = client.create_from_slice(bytemuck::cast_slice(profile_pdf));
    let pcdf_h = client.create_from_slice(bytemuck::cast_slice(profile_cdf));
    let out_h = client.empty(n * std::mem::size_of::<f64>());

    const WORKGROUP_SIZE: u32 = 64;
    let groups = (n as u32).div_ceil(WORKGROUP_SIZE);

    unsafe {
        compton_doppler_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(seeds_h, n),
            BufferArg::from_raw_parts(e_in_h, 1),
            BufferArg::from_raw_parts(mu_h, 1),
            BufferArg::from_raw_parts(ns_h, 1),
            BufferArg::from_raw_parts(has_h, 1),
            BufferArg::from_raw_parts(pz_h, pz_grid.len()),
            BufferArg::from_raw_parts(epdf_h, electron_pdf.len()),
            BufferArg::from_raw_parts(be_h, binding_energy.len()),
            BufferArg::from_raw_parts(ppdf_h, profile_pdf.len()),
            BufferArg::from_raw_parts(pcdf_h, profile_cdf.len()),
            BufferArg::from_raw_parts(out_h.clone(), n),
        );
    }

    bytemuck::cast_slice(&client.read_one(out_h).unwrap()).to_vec()
}
