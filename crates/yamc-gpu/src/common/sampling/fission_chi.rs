//! Shared `#[cube]` fission outgoing-energy (chi) sampler.
//!
//! Samples one outgoing neutron energy from a material's fission spectrum,
//! dispatching on the per-row `fission_eout_kind`:
//!   - `1` ContinuousTabular: stochastic E_in bracket pick + interp-aware CDF
//!     inversion + bracket-bound stretch (mirrors the inelastic ContinuousTabular
//!     eout sampler and `yamc_physics::gpu::flat::fission_eout_continuous`).
//!     The within-bin inversion honours each E_in row's tabulated interpolation
//!     (`fission_eout_interp`: 0 histogram, 1 lin-lin): lin-lin rows use the
//!     quadratic inversion off the stored PDF (`fission_eout_p`), histogram /
//!     degenerate rows keep the legacy linear-in-c form.
//!   - `6` Maxwell (ENDF File 5, Law 7): theta(E_in) interpolation + 3-uniform
//!     rejection (`maxwell_rejection_draw`), mirroring `sample_maxwell`.
//!   - `4` Evaporation (Law 9): theta(E_in) interpolation + 2-uniform rejection
//!     (`evaporation_rejection_draw`), mirroring `sample_evaporation`.
//!   - else / fall-through: Watt (Law 11) rejection (`watt_fission_draw`),
//!     mirroring `sample_watt_spectrum_params`.
//!
//! This is the exact per-progeny chi-energy logic that previously lived inline
//! in the neutron transport kernel's fission branch. Extracting it lets the
//! fission-bank path call it once per fission progeny (the continuing walk plus
//! the `N - 1` banked secondaries), each drawing an independent chi energy from
//! the same incident energy -- the GPU twin of the CPU
//! `sample_fission_neutrons` per-neutron product sampling.
//!
//! The draw ORDER is preserved bit-for-bit from the original inline block, so
//! the RNG `state` advances identically; the sampled energy flows through the
//! `ln`/`exp`/`cos` polyfills (in the rejection branches) so it is parity-, not
//! bit-, exact versus the CPU twin (same convention as `eout_rejection`).

use crate::common::polyfills::exp_f64;
use cubecl::prelude::*;

/// Result of one prompt-fission chi draw: the sampled outgoing energy and the
/// advanced PCG-32 state.
#[derive(CubeType)]
pub struct FissionChiDraw {
    pub e_out: f64,
    pub state: u64,
}

/// Sample one prompt-fission outgoing energy from the per-material chi spectrum.
///
/// `chi_row` selects the row in every per-material chi buffer. Rows come in
/// (prompt, delayed) pairs, so material `m`'s prompt spectrum is row `2m` and its
/// delayed spectrum is row `2m + 1` (issue #364). `e_in` is the incident neutron
/// energy. `watt_a` / `watt_b` are the material's Watt
/// parameters (the fall-through law). The fission chi table is tight CSR
/// (issue #104): `fission_eout_ae_offset[chi_row]` is the row's first
/// ae-row in `fission_eout_n_x_per_material` / `fission_eout_energy_grid_per_material`,
/// and `fission_eout_x_offset[ae_row]` is each ae-row's start in
/// `fission_eout_x_per_material` / `fission_eout_cdf_per_material` /
/// `fission_eout_p_per_material`. `fission_eout_interp_per_material` carries
/// one interpolation code per ae-row (0 histogram, 1 lin-lin). No fixed
/// per-axis stride.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn sample_fission_chi(
    e_in: f64,
    chi_row: u32,
    watt_a: f64,
    watt_b: f64,
    fission_eout_kind_per_material: &[u32],
    fission_eout_n_energies_per_material: &[u32],
    fission_eout_ae_offset: &[u32],
    fission_eout_energy_grid_per_material: &[f64],
    fission_eout_n_x_per_material: &[u32],
    fission_eout_x_offset: &[u32],
    fission_eout_x_per_material: &[f64],
    fission_eout_cdf_per_material: &[f64],
    fission_eout_p_per_material: &[f64],
    fission_eout_interp_per_material: &[u32],
    state_in: u64,
) -> FissionChiDraw {
    let mut state = state_in;
    let mut energy = e_in;
    let fission_kind = fission_eout_kind_per_material[chi_row as usize];
    let mut sampled_from_table = 0u32;
    if fission_kind == 1u32 {
        // ContinuousTabular fission spectrum.
        let n_fae = fission_eout_n_energies_per_material[chi_row as usize];
        if n_fae > 0u32 {
            let eg_off_f = fission_eout_ae_offset[chi_row as usize];
            let mut i_eb = 0u32;
            let e_first = fission_eout_energy_grid_per_material[eg_off_f as usize];
            let e_last = fission_eout_energy_grid_per_material[(eg_off_f + n_fae - 1u32) as usize];
            let mut r_eb = 0.0_f64;
            if energy >= e_last {
                if n_fae > 1u32 {
                    i_eb = n_fae - 2u32;
                }
                r_eb = 1.0;
            } else if energy > e_first {
                let mut k = 0u32;
                while k + 1u32 < n_fae {
                    let e_k = fission_eout_energy_grid_per_material[(eg_off_f + k) as usize];
                    let e_k1 =
                        fission_eout_energy_grid_per_material[(eg_off_f + k + 1u32) as usize];
                    if energy >= e_k && energy < e_k1 {
                        i_eb = k;
                        let de = e_k1 - e_k;
                        if de > 0.0 {
                            r_eb = (energy - e_k) / de;
                        }
                    }
                    k += 1u32;
                }
            }

            // Stochastic E_in bracket pick -- same pattern as the inelastic
            // eout sampler.
            let pick_f = crate::common::sampling::energy_bracket::pick_energy_bracket(
                r_eb, i_eb, n_fae, state,
            );
            state = pick_f.state;
            let bin_e = pick_f.bin;

            let n_x = fission_eout_n_x_per_material[(eg_off_f + bin_e) as usize];
            if n_x >= 2u32 {
                let x_off = fission_eout_x_offset[(eg_off_f + bin_e) as usize];
                let d_xi_fx = crate::common::pcg32::draw_uniform(state);
                state = d_xi_fx.state;
                let xi_fx = d_xi_fx.xi;

                let mut j = 0u32;
                let mut j_found = 0u32;
                let mut k = 0u32;
                while k + 1u32 < n_x {
                    let c_k1 = fission_eout_cdf_per_material[(x_off + k + 1u32) as usize];
                    if j_found == 0u32 && xi_fx <= c_k1 {
                        j = k;
                        j_found = 1u32;
                    }
                    k += 1u32;
                }
                if j_found == 0u32 {
                    j = n_x - 2u32;
                }
                let c_j = fission_eout_cdf_per_material[(x_off + j) as usize];
                let c_j1 = fission_eout_cdf_per_material[(x_off + j + 1u32) as usize];
                let x_j = fission_eout_x_per_material[(x_off + j) as usize];
                let x_j1 = fission_eout_x_per_material[(x_off + j + 1u32) as usize];
                let p_j = fission_eout_p_per_material[(x_off + j) as usize];
                let p_j1 = fission_eout_p_per_material[(x_off + j + 1u32) as usize];
                let interp_row = fission_eout_interp_per_material[(eg_off_f + bin_e) as usize];
                let dx = x_j1 - x_j;
                let dc = c_j1 - c_j;
                let mut e_sampled = x_j;
                if interp_row == 1u32 && dx > 0.0 {
                    // Lin-lin row: quadratic inversion off the stored PDF --
                    // same form as the inelastic ContinuousTabular eout path.
                    let m = (p_j1 - p_j) / dx;
                    let abs_m = if m < 0.0 { -m } else { m };
                    if abs_m < 1e-30 {
                        if p_j > 0.0 {
                            e_sampled = x_j + (xi_fx - c_j) / p_j;
                        }
                    } else {
                        let mut disc = p_j * p_j + 2.0 * m * (xi_fx - c_j);
                        if disc < 0.0 {
                            disc = 0.0;
                        }
                        e_sampled = x_j + (disc.sqrt() - p_j) / m;
                    }
                } else if dc > 0.0 {
                    // Histogram / degenerate row: legacy linear-in-c form
                    // (identical to a histogram inversion when the table is
                    // consistent, p_j == dc/dx).
                    e_sampled = x_j + (xi_fx - c_j) / dc * (x_j1 - x_j);
                }

                // Bracket-bound stretch -- mirrors the inelastic eout sampler.
                if n_fae >= 2u32 {
                    let n_x_i = fission_eout_n_x_per_material[(eg_off_f + i_eb) as usize];
                    let n_x_i1 = fission_eout_n_x_per_material[(eg_off_f + i_eb + 1u32) as usize];
                    if n_x_i >= 2u32 && n_x_i1 >= 2u32 {
                        let x_off_i = fission_eout_x_offset[(eg_off_f + i_eb) as usize];
                        let x_off_i1 = fission_eout_x_offset[(eg_off_f + i_eb + 1u32) as usize];
                        let e_i_1 = fission_eout_x_per_material[x_off_i as usize];
                        let e_i_k = fission_eout_x_per_material[(x_off_i + n_x_i - 1u32) as usize];
                        let e_i1_1 = fission_eout_x_per_material[x_off_i1 as usize];
                        let e_i1_k =
                            fission_eout_x_per_material[(x_off_i1 + n_x_i1 - 1u32) as usize];
                        let e_1 = e_i_1 + r_eb * (e_i1_1 - e_i_1);
                        let e_k = e_i_k + r_eb * (e_i1_k - e_i_k);
                        let e_l_1 = if bin_e == i_eb { e_i_1 } else { e_i1_1 };
                        let e_l_k = if bin_e == i_eb { e_i_k } else { e_i1_k };
                        let denom_l = e_l_k - e_l_1;
                        if denom_l > 0.0 {
                            e_sampled = e_1 + (e_sampled - e_l_1) * (e_k - e_1) / denom_l;
                        }
                    }
                }

                if e_sampled <= 0.0 {
                    e_sampled = 1.0e-6;
                }
                energy = e_sampled;
                sampled_from_table = 1u32;
            }
        }
    } else if fission_kind == 6u32 {
        // Maxwell prompt-fission chi (ENDF File 5, Law 7). Tight CSR
        // (issue #104): each E_in row carries one point, so θ at row k is
        // `x[fission_eout_x_offset[eg_off_f + k]]` and the scalar `u` is in
        // the material's row-0 single slot
        // `cdf[fission_eout_x_offset[eg_off_f]]`.
        let n_fae = fission_eout_n_energies_per_material[chi_row as usize];
        if n_fae > 0u32 {
            let eg_off_f = fission_eout_ae_offset[chi_row as usize];
            let x_base = fission_eout_x_offset[eg_off_f as usize];
            let mf_first = fission_eout_energy_grid_per_material[eg_off_f as usize];
            let mf_last = fission_eout_energy_grid_per_material[(eg_off_f + n_fae - 1u32) as usize];
            let mut theta_val = fission_eout_x_per_material
                [fission_eout_x_offset[(eg_off_f + n_fae - 1u32) as usize] as usize];
            if energy <= mf_first {
                theta_val = fission_eout_x_per_material[x_base as usize];
            } else if energy < mf_last {
                let mut k = 0u32;
                while k + 1u32 < n_fae {
                    let e_k = fission_eout_energy_grid_per_material[(eg_off_f + k) as usize];
                    let e_k1 =
                        fission_eout_energy_grid_per_material[(eg_off_f + k + 1u32) as usize];
                    if energy >= e_k && energy < e_k1 {
                        let de = e_k1 - e_k;
                        let mut f = 0.0_f64;
                        if de > 0.0 {
                            f = (energy - e_k) / de;
                        }
                        let t_k = fission_eout_x_per_material
                            [fission_eout_x_offset[(eg_off_f + k) as usize] as usize];
                        let t_k1 = fission_eout_x_per_material
                            [fission_eout_x_offset[(eg_off_f + k + 1u32) as usize] as usize];
                        theta_val = t_k + f * (t_k1 - t_k);
                    }
                    k += 1u32;
                }
            }
            let u_fis = fission_eout_cdf_per_material[x_base as usize];
            if theta_val > 0.0 && energy > u_fis {
                let cap_e = energy - u_fis;
                let mut accepted = 0u32;
                let mut sampled_e = 0.0;
                let mut iter = 0u32;
                while iter < 32u32 && accepted == 0u32 {
                    let rj = crate::common::sampling::eout_rejection::maxwell_rejection_draw(
                        theta_val, cap_e, state,
                    );
                    state = rj.state;
                    if rj.accepted == 1u32 {
                        sampled_e = rj.e_out;
                        accepted = 1u32;
                    }
                    iter += 1u32;
                }
                if accepted == 1u32 && sampled_e > 0.0 {
                    energy = sampled_e;
                    sampled_from_table = 1u32;
                }
            }
        }
    } else if fission_kind == 4u32 {
        // Evaporation prompt-fission chi (ENDF File 5, Law 9). Same tight
        // CSR θ / u packing as the Maxwell branch (issue #104).
        let n_fae = fission_eout_n_energies_per_material[chi_row as usize];
        if n_fae > 0u32 {
            let eg_off_f = fission_eout_ae_offset[chi_row as usize];
            let x_base = fission_eout_x_offset[eg_off_f as usize];
            let ef_first = fission_eout_energy_grid_per_material[eg_off_f as usize];
            let ef_last = fission_eout_energy_grid_per_material[(eg_off_f + n_fae - 1u32) as usize];
            let mut theta_val = fission_eout_x_per_material
                [fission_eout_x_offset[(eg_off_f + n_fae - 1u32) as usize] as usize];
            if energy <= ef_first {
                theta_val = fission_eout_x_per_material[x_base as usize];
            } else if energy < ef_last {
                let mut k = 0u32;
                while k + 1u32 < n_fae {
                    let e_k = fission_eout_energy_grid_per_material[(eg_off_f + k) as usize];
                    let e_k1 =
                        fission_eout_energy_grid_per_material[(eg_off_f + k + 1u32) as usize];
                    if energy >= e_k && energy < e_k1 {
                        let de = e_k1 - e_k;
                        let mut f = 0.0_f64;
                        if de > 0.0 {
                            f = (energy - e_k) / de;
                        }
                        let t_k = fission_eout_x_per_material
                            [fission_eout_x_offset[(eg_off_f + k) as usize] as usize];
                        let t_k1 = fission_eout_x_per_material
                            [fission_eout_x_offset[(eg_off_f + k + 1u32) as usize] as usize];
                        theta_val = t_k + f * (t_k1 - t_k);
                    }
                    k += 1u32;
                }
            }
            let u_fis = fission_eout_cdf_per_material[x_base as usize];
            if theta_val > 0.0 && energy > u_fis {
                let y = (energy - u_fis) / theta_val;
                let v_e = 1.0 - exp_f64(-y);
                let mut accepted = 0u32;
                let mut sampled_e = 0.0;
                let mut iter = 0u32;
                while iter < 32u32 && accepted == 0u32 {
                    let rj = crate::common::sampling::eout_rejection::evaporation_rejection_draw(
                        v_e, y, theta_val, state,
                    );
                    state = rj.state;
                    if rj.accepted == 1u32 {
                        sampled_e = rj.e_out;
                        accepted = 1u32;
                    }
                    iter += 1u32;
                }
                if accepted == 1u32 && sampled_e > 0.0 {
                    energy = sampled_e;
                    sampled_from_table = 1u32;
                }
            }
        }
    }
    if sampled_from_table == 0u32 {
        // Watt-rejection fallback -- same algorithm as `sample_watt_spectrum_params`.
        let wf = crate::common::sampling::eout_rejection::watt_fission_draw(watt_a, watt_b, state);
        state = wf.state;
        let mut e_out = wf.e_out;
        if e_out <= 0.0 {
            e_out = 1.0e-6;
        }
        energy = e_out;
    }

    FissionChiDraw {
        e_out: energy,
        state,
    }
}

/// Sample one fission progeny's outgoing energy, choosing the prompt or the
/// delayed spectrum first (issue #364).
///
/// `beta` is the material's delayed fraction `nu_d(E) / nu_t(E)` at the incident
/// energy. The count of progeny already comes from nu_TOTAL, so what the delayed
/// groups add is that a `beta` share of them are born from the (much softer)
/// delayed spectrum in row `2*mat_idx + 1` instead of the prompt spectrum in row
/// `2*mat_idx`.
///
/// Mirrors CPU `yamc_physics::neutron::interaction::fission_progeny_energy`: ONE
/// uniform, drawn only when `beta > 0.0`, so a material with no delayed data keeps
/// the previous draw schedule exactly.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn sample_fission_progeny_energy(
    e_in: f64,
    mat_idx: u32,
    beta: f64,
    watt_a: f64,
    watt_b: f64,
    fission_eout_kind_per_material: &[u32],
    fission_eout_n_energies_per_material: &[u32],
    fission_eout_ae_offset: &[u32],
    fission_eout_energy_grid_per_material: &[f64],
    fission_eout_n_x_per_material: &[u32],
    fission_eout_x_offset: &[u32],
    fission_eout_x_per_material: &[f64],
    fission_eout_cdf_per_material: &[f64],
    fission_eout_p_per_material: &[f64],
    fission_eout_interp_per_material: &[u32],
    state_in: u64,
) -> FissionChiDraw {
    let mut state = state_in;
    let mut chi_row = 2u32 * mat_idx;
    if beta > 0.0 {
        let d_del = crate::common::pcg32::draw_uniform(state);
        state = d_del.state;
        if d_del.xi < beta {
            chi_row = 2u32 * mat_idx + 1u32;
        }
    }
    sample_fission_chi(
        e_in,
        chi_row,
        watt_a,
        watt_b,
        fission_eout_kind_per_material,
        fission_eout_n_energies_per_material,
        fission_eout_ae_offset,
        fission_eout_energy_grid_per_material,
        fission_eout_n_x_per_material,
        fission_eout_x_offset,
        fission_eout_x_per_material,
        fission_eout_cdf_per_material,
        fission_eout_p_per_material,
        fission_eout_interp_per_material,
        state,
    )
}
