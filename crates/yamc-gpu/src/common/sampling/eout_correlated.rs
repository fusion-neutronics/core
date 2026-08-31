//! Shared correlated angle-energy outgoing-energy sampler (ENDF File-6 Law-1,
//! `CorrelatedAngleEnergy`).
//!
//! Bracket the incident energy between two `corr_energy_grid` points,
//! stochastically pick one bracketing distribution (via the shared
//! [`pick_energy_bracket`](crate::common::sampling::energy_bracket::pick_energy_bracket)),
//! invert the chosen slice's outgoing-energy CDF, then stretch the result into
//! the incident-energy-interpolated bounds. This is the E_out half of the
//! kernel's `eout_kind == EOUT_KIND_CORRELATED (2)` branch; the matching `mu`
//! sample is a per-`(E_in, E_out)` angular sub-table that the caller draws with
//! the shared [`invert_angle_cdf`](crate::common::sampling::angle_cdf_invert::invert_angle_cdf),
//! keyed by the `mu_idx_off` this sampler returns. Splitting the law here keeps
//! both shared CDF inverters single-sourced while still preserving the exact
//! `(E_out CDF draw, then mu draw)` RNG order at the call site.
//!
//! Faithful to `yamc_nuclide`'s `CorrelatedAngleEnergy::sample` +
//! `TabulatedProbability::sample_with_discrete_info`. The E_out walk draws up
//! to two PCG-32 uniforms (the incident-energy bracket pick, then the CDF
//! inversion -- the latter only when the chosen slice has >= 2 points), so it
//! both consumes and returns the RNG `state`. When the chosen slice has < 2
//! points the result carries `valid == 0`: the caller leaves its current
//! `e_cm` / `mu_sampled` untouched and consumes only the bracket-pick draw,
//! reproducing the kernel's `if n_x >= 2` guard exactly.
//!
//! Pure integer-PCG + f64 arithmetic apart from one `sqrt` in the LinLin
//! quadratic inversion, so the `#[cube]` kernel matches the
//! [`sample_correlated_eout_cpu`] twin to within a few ULPs
//! (`gpu_eout_correlated_matches_cpu`).

use crate::common::pcg32::{expand_seed, pcg_out};
use crate::common::rng::{PCG_INCR, PCG_MULT};
use crate::neutron::xs::constants::ANGLE_INTERP_LINLIN;
use cubecl::prelude::*;

/// Result of [`sample_correlated_eout`]: the sampled outgoing energy, the flat
/// offset into the `corr_mu_*` buffers' index dimension for the matching
/// angular sub-table (`x_off + j`, valid only when `valid == 1`), a `valid`
/// flag (`1` when the chosen E_in slice had >= 2 outgoing points and both
/// `e_out` and `mu_idx_off` are meaningful), and the advanced PCG-32 state.
#[derive(CubeType)]
pub struct CorrelatedEoutSample {
    pub e_out: f64,
    pub mu_idx_off: u32,
    pub valid: u32,
    pub state: u64,
}

/// Sample a correlated (File-6 Law-1) outgoing energy. `eg_off_c` is the slot's
/// global base ae-row into the per-incident-energy buffers
/// (`corr_ae_offset[mat_slot]` with tight CSR storage, issue #104);
/// `corr_x_offset` gives the global x-point start of each ae-row's
/// `(x, cdf, p)` points in the flat `corr_x` / `corr_cdf` / `corr_p` arrays
/// (row `eg_off_c + bin_e` starts at `corr_x_offset[eg_off_c + bin_e]`). The
/// flat buffers (`corr_energy_grid`, `corr_n_x`, `corr_x`, `corr_cdf`,
/// `corr_p`, `corr_interp`, `corr_n_discrete`) are the kernel's corr_*
/// outgoing-energy buffers. Returns the sampled energy, the `mu_idx_off` (the
/// global x-point index of the chosen E_out bin, used by the caller to read
/// the matching mu sub-table via `corr_mu_offset[mu_idx_off]`), the `valid`
/// flag, and the advanced `state`.
#[cube]
pub fn sample_correlated_eout(
    energy: f64,
    eg_off_c: u32,
    n_corr: u32,
    corr_x_offset: &[u32],
    corr_energy_grid: &[f64],
    corr_n_x: &[u32],
    corr_x: &[f64],
    corr_cdf: &[f64],
    corr_p: &[f64],
    corr_interp: &[u32],
    corr_n_discrete: &[u32],
    state_in: u64,
) -> CorrelatedEoutSample {
    let mut state = state_in;
    let mut e_out = 0.0_f64;
    let mut mu_idx_off = 0u32;
    let mut valid = 0u32;

    let mut i_eb = 0u32;
    let e_first = corr_energy_grid[eg_off_c as usize];
    let e_last = corr_energy_grid[(eg_off_c + n_corr - 1u32) as usize];
    let mut r_eb = 0.0_f64;
    if energy >= e_last {
        if n_corr > 1u32 {
            i_eb = n_corr - 2u32;
        }
        r_eb = 1.0;
    } else if energy > e_first {
        let mut k = 0u32;
        while k + 1u32 < n_corr {
            let e_k = corr_energy_grid[(eg_off_c + k) as usize];
            let e_k1 = corr_energy_grid[(eg_off_c + k + 1u32) as usize];
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

    // Stochastic incident-energy-bracket pick (shared helper draws one uniform).
    let pick_c =
        crate::common::sampling::energy_bracket::pick_energy_bracket(r_eb, i_eb, n_corr, state);
    state = pick_c.state;
    let bin_e = pick_c.bin;

    let n_x = corr_n_x[(eg_off_c + bin_e) as usize];
    if n_x >= 2u32 {
        valid = 1u32;
        let x_off = corr_x_offset[(eg_off_c + bin_e) as usize];
        let d_xi_cx = crate::common::pcg32::draw_uniform(state);
        state = d_xi_cx.state;
        let xi_cx = d_xi_cx.xi;

        let n_disc = corr_n_discrete[(eg_off_c + bin_e) as usize];
        let interp_kind = corr_interp[(eg_off_c + bin_e) as usize];
        // Discrete-then-continuous CDF search matching the CPU reference
        // `sample_with_discrete_info` (issue #103): discrete head (first
        // `n_disc` points) searched with `xi < c[k]` (exact line), continuous
        // tail from `n_disc` with `xi <= c[k+1]`, carrying `c_j`. The previous
        // single scan collapsed every discrete line onto index 0. For
        // `n_disc == 0` this is identical to the old scan.
        let mut j = 0u32;
        let mut c_j = corr_cdf[x_off as usize];
        let mut is_discrete = 0u32;
        let mut done = 0u32;
        let mut kd = 0u32;
        while kd < n_disc {
            if done == 0u32 {
                j = kd;
                c_j = corr_cdf[(x_off + kd) as usize];
                if xi_cx < c_j {
                    is_discrete = 1u32;
                    done = 1u32;
                }
            }
            kd += 1u32;
        }
        let mut kc = n_disc;
        while kc + 1u32 < n_x {
            if done == 0u32 {
                j = kc;
                let c_k1 = corr_cdf[(x_off + kc + 1u32) as usize];
                if xi_cx <= c_k1 {
                    done = 1u32;
                } else {
                    j = kc + 1u32;
                    c_j = c_k1;
                }
            }
            kc += 1u32;
        }
        if is_discrete == 0u32 && j >= n_x - 1u32 {
            j = n_x - 2u32;
        }
        let x_j = corr_x[(x_off + j) as usize];
        let x_j1 = corr_x[(x_off + j + 1u32) as usize];
        let p_j = corr_p[(x_off + j) as usize];
        let p_j1 = corr_p[(x_off + j + 1u32) as usize];
        let dx = x_j1 - x_j;
        let mut e_sampled = x_j;
        if is_discrete == 0u32 {
            if interp_kind == ANGLE_INTERP_LINLIN && dx > 0.0 {
                // CPU's LinLin Tabular quadratic inversion:
                //   x_k + (sqrt(p_k^2 + 2*m*(r - c_k)) - p_k) / m
                let m = (p_j1 - p_j) / dx;
                let abs_m = if m < 0.0 { -m } else { m };
                if abs_m < 1e-30 {
                    if p_j > 0.0 {
                        e_sampled = x_j + (xi_cx - c_j) / p_j;
                    }
                } else {
                    let mut disc = p_j * p_j + 2.0 * m * (xi_cx - c_j);
                    if disc < 0.0 {
                        disc = 0.0;
                    }
                    e_sampled = x_j + (disc.sqrt() - p_j) / m;
                }
            } else if p_j > 0.0 {
                // Histogram bracket.
                e_sampled = x_j + (xi_cx - c_j) / p_j;
            }
        }
        // else: discrete delta -- keep e_sampled = x_j.

        // Bracket-bound stretch -- SKIP when sampled bin was discrete (mirrors
        // CPU's early return).
        if is_discrete == 0u32 && n_corr >= 2u32 {
            let n_x_i = corr_n_x[(eg_off_c + i_eb) as usize];
            let n_x_i1 = corr_n_x[(eg_off_c + i_eb + 1u32) as usize];
            if n_x_i >= 2u32 && n_x_i1 >= 2u32 {
                let x_off_i = corr_x_offset[(eg_off_c + i_eb) as usize];
                let x_off_i1 = corr_x_offset[(eg_off_c + i_eb + 1u32) as usize];
                let e_i_1 = corr_x[x_off_i as usize];
                let e_i_k = corr_x[(x_off_i + n_x_i - 1u32) as usize];
                let e_i1_1 = corr_x[x_off_i1 as usize];
                let e_i1_k = corr_x[(x_off_i1 + n_x_i1 - 1u32) as usize];
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
            e_sampled = 0.0;
        }
        e_out = e_sampled;
        // Angular sub-table at the NEARER of the two bracketing E_out points,
        // measured on the CDF (issue #371). Taking `j` unconditionally, as this
        // did, biases μ wherever adjacent sub-tables differ -- which is exactly
        // at a spectrum's falling edge. Reads `corr_cdf` only, so the draw
        // schedule is unchanged.
        let mut mu_bin = j;
        if is_discrete == 0u32 && interp_kind == ANGLE_INTERP_LINLIN {
            let c_j1 = corr_cdf[(x_off + j + 1u32) as usize];
            if xi_cx - c_j >= c_j1 - xi_cx {
                mu_bin = j + 1u32;
            }
        }
        mu_idx_off = x_off + mu_bin;
    }

    CorrelatedEoutSample {
        e_out,
        mu_idx_off,
        valid,
        state,
    }
}

/// CPU twin of [`sample_correlated_eout`]. Same algorithm and same PCG draws
/// (wrapping 64-bit arithmetic), so it matches the `#[cube]` kernel to within a
/// few ULPs. Returns `(e_out, mu_idx_off, valid, state)`.
#[allow(clippy::too_many_arguments)]
pub fn sample_correlated_eout_cpu(
    energy: f64,
    eg_off_c: u32,
    n_corr: u32,
    corr_x_offset: &[u32],
    corr_energy_grid: &[f64],
    corr_n_x: &[u32],
    corr_x: &[f64],
    corr_cdf: &[f64],
    corr_p: &[f64],
    corr_interp: &[u32],
    corr_n_discrete: &[u32],
    state_in: u64,
) -> (f64, u32, u32, u64) {
    let mut state = state_in;
    let mut e_out = 0.0_f64;
    let mut mu_idx_off = 0u32;
    let mut valid = 0u32;

    let mut i_eb = 0u32;
    let e_first = corr_energy_grid[eg_off_c as usize];
    let e_last = corr_energy_grid[(eg_off_c + n_corr - 1) as usize];
    let mut r_eb = 0.0_f64;
    if energy >= e_last {
        if n_corr > 1 {
            i_eb = n_corr - 2;
        }
        r_eb = 1.0;
    } else if energy > e_first {
        let mut k = 0u32;
        while k + 1 < n_corr {
            let e_k = corr_energy_grid[(eg_off_c + k) as usize];
            let e_k1 = corr_energy_grid[(eg_off_c + k + 1) as usize];
            if energy >= e_k && energy < e_k1 {
                i_eb = k;
                let de = e_k1 - e_k;
                if de > 0.0 {
                    r_eb = (energy - e_k) / de;
                }
            }
            k += 1;
        }
    }

    // Bracket pick: one PCG draw, upper bracket when r > xi (bit-for-bit the
    // shared `pick_energy_bracket`).
    let s_eb = state;
    let r_eb_rand = pcg_out(s_eb);
    state = s_eb.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
    let xi_eb = (r_eb_rand as f64 + 1.0) * (1.0 / 4_294_967_297.0);
    let mut bin_e = i_eb;
    if r_eb > xi_eb && bin_e + 1 < n_corr {
        bin_e = i_eb + 1;
    }

    let n_x = corr_n_x[(eg_off_c + bin_e) as usize];
    if n_x >= 2 {
        valid = 1;
        let x_off = corr_x_offset[(eg_off_c + bin_e) as usize];
        let s_x = state;
        let r_x = pcg_out(s_x);
        state = s_x.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
        let xi_cx = (r_x as f64 + 1.0) * (1.0 / 4_294_967_297.0);

        let n_disc = corr_n_discrete[(eg_off_c + bin_e) as usize];
        let interp_kind = corr_interp[(eg_off_c + bin_e) as usize];
        // Discrete-then-continuous CDF search (issue #103); bit-twin of the
        // `#[cube]` scan above. See its comment.
        let mut j = 0u32;
        let mut c_j = corr_cdf[x_off as usize];
        let mut is_discrete = 0u32;
        let mut done = 0u32;
        let mut kd = 0u32;
        while kd < n_disc {
            if done == 0 {
                j = kd;
                c_j = corr_cdf[(x_off + kd) as usize];
                if xi_cx < c_j {
                    is_discrete = 1u32;
                    done = 1;
                }
            }
            kd += 1;
        }
        let mut kc = n_disc;
        while kc + 1 < n_x {
            if done == 0 {
                j = kc;
                let c_k1 = corr_cdf[(x_off + kc + 1) as usize];
                if xi_cx <= c_k1 {
                    done = 1;
                } else {
                    j = kc + 1;
                    c_j = c_k1;
                }
            }
            kc += 1;
        }
        if is_discrete == 0u32 && j >= n_x - 1 {
            j = n_x - 2;
        }
        let x_j = corr_x[(x_off + j) as usize];
        let x_j1 = corr_x[(x_off + j + 1) as usize];
        let p_j = corr_p[(x_off + j) as usize];
        let p_j1 = corr_p[(x_off + j + 1) as usize];
        let dx = x_j1 - x_j;
        let mut e_sampled = x_j;
        if is_discrete == 0 {
            if interp_kind == ANGLE_INTERP_LINLIN && dx > 0.0 {
                let m = (p_j1 - p_j) / dx;
                let abs_m = if m < 0.0 { -m } else { m };
                if abs_m < 1e-30 {
                    if p_j > 0.0 {
                        e_sampled = x_j + (xi_cx - c_j) / p_j;
                    }
                } else {
                    let mut disc = p_j * p_j + 2.0 * m * (xi_cx - c_j);
                    if disc < 0.0 {
                        disc = 0.0;
                    }
                    e_sampled = x_j + (disc.sqrt() - p_j) / m;
                }
            } else if p_j > 0.0 {
                e_sampled = x_j + (xi_cx - c_j) / p_j;
            }
        }

        if is_discrete == 0 && n_corr >= 2 {
            let n_x_i = corr_n_x[(eg_off_c + i_eb) as usize];
            let n_x_i1 = corr_n_x[(eg_off_c + i_eb + 1) as usize];
            if n_x_i >= 2 && n_x_i1 >= 2 {
                let x_off_i = corr_x_offset[(eg_off_c + i_eb) as usize];
                let x_off_i1 = corr_x_offset[(eg_off_c + i_eb + 1) as usize];
                let e_i_1 = corr_x[x_off_i as usize];
                let e_i_k = corr_x[(x_off_i + n_x_i - 1) as usize];
                let e_i1_1 = corr_x[x_off_i1 as usize];
                let e_i1_k = corr_x[(x_off_i1 + n_x_i1 - 1) as usize];
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
            e_sampled = 0.0;
        }
        e_out = e_sampled;
        // Angular sub-table at the NEARER of the two bracketing E_out points,
        // measured on the CDF (issue #371). Taking `j` unconditionally, as this
        // did, biases μ wherever adjacent sub-tables differ -- which is exactly
        // at a spectrum's falling edge. Reads `corr_cdf` only, so the draw
        // schedule is unchanged.
        let mut mu_bin = j;
        if is_discrete == 0u32 && interp_kind == ANGLE_INTERP_LINLIN {
            let c_j1 = corr_cdf[(x_off + j + 1u32) as usize];
            if xi_cx - c_j >= c_j1 - xi_cx {
                mu_bin = j + 1u32;
            }
        }
        mu_idx_off = x_off + mu_bin;
    }

    (e_out, mu_idx_off, valid, state)
}

/// Test/validation kernel: one sample per thread, using a fixed small layout
/// (`max_corr_ae = 4`, `max_corr_x = 8`) so the buffers stay tiny. Each sample
/// carries its own incident energy, slot offset, `n_corr`, and PCG seed
/// (expanded in-kernel via [`expand_seed`]).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn eout_corr_test_kernel(
    energy_in: &[f64],
    eg_off_c: &[u32],
    n_corr: &[u32],
    corr_x_offset: &[u32],
    corr_energy_grid: &[f64],
    corr_n_x: &[u32],
    corr_x: &[f64],
    corr_cdf: &[f64],
    corr_p: &[f64],
    corr_interp: &[u32],
    corr_n_discrete: &[u32],
    seeds: &[u32],
    out_e: &mut [f64],
    out_mu_off: &mut [u32],
    out_valid: &mut [u32],
    out_state: &mut [u64],
) {
    if ABSOLUTE_POS >= out_e.len() {
        terminate!();
    }
    let r = sample_correlated_eout(
        energy_in[ABSOLUTE_POS],
        eg_off_c[ABSOLUTE_POS],
        n_corr[ABSOLUTE_POS],
        corr_x_offset,
        corr_energy_grid,
        corr_n_x,
        corr_x,
        corr_cdf,
        corr_p,
        corr_interp,
        corr_n_discrete,
        expand_seed(seeds[ABSOLUTE_POS]),
    );
    out_e[ABSOLUTE_POS] = r.e_out;
    out_mu_off[ABSOLUTE_POS] = r.mu_idx_off;
    out_valid[ABSOLUTE_POS] = r.valid;
    out_state[ABSOLUTE_POS] = r.state;
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError, WgpuRuntime};

    // Fixed test layout (kept tiny; the kernel hardcodes max_corr_x = 8).
    const AE: usize = 4; // max_corr_ae for the test
    const MX: usize = 8; // max_corr_x for the test

    /// Build a 2-slot corr eout fixture. Slot 0: two incident energies, each a
    /// 4-point LinLin continuous spectrum. Slot 1: a single incident energy
    /// with a 3-point histogram spectrum plus a leading discrete line.
    #[allow(clippy::type_complexity)]
    fn fixture() -> (
        Vec<f64>,
        Vec<u32>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<u32>,
        Vec<u32>,
    ) {
        let n_slots = 2usize;
        let mut energy_grid = vec![0.0_f64; n_slots * AE];
        let mut n_x = vec![0u32; n_slots * AE];
        let mut x = vec![0.0_f64; n_slots * AE * MX];
        let mut cdf = vec![0.0_f64; n_slots * AE * MX];
        let mut p = vec![0.0_f64; n_slots * AE * MX];
        let mut interp = vec![0u32; n_slots * AE];
        let mut n_disc = vec![0u32; n_slots * AE];

        // Slot 0, incident energy index 0 and 1.
        energy_grid[0] = 1.0e6;
        energy_grid[1] = 2.0e6;
        for (ae, scale) in [(0usize, 1.0_f64), (1usize, 1.4_f64)] {
            n_x[ae] = 4;
            interp[ae] = ANGLE_INTERP_LINLIN;
            let off = ae * MX;
            let xs = [0.1e6 * scale, 0.4e6 * scale, 0.8e6 * scale, 1.0e6 * scale];
            let cs = [0.0, 0.35, 0.8, 1.0];
            x[off..off + 4].copy_from_slice(&xs);
            cdf[off..off + 4].copy_from_slice(&cs);
            for k in 0..3 {
                let dxk = xs[k + 1] - xs[k];
                p[off + k] = if dxk > 0.0 {
                    (cs[k + 1] - cs[k]) / dxk
                } else {
                    0.0
                };
            }
            p[off + 3] = p[off + 2];
        }

        // Slot 1, single incident energy: discrete line then histogram.
        let s1 = AE; // slot 1 base in the per-ae arrays
        energy_grid[s1] = 5.0e5;
        n_x[s1] = 3;
        interp[s1] = 0; // histogram
        n_disc[s1] = 1; // first point is a discrete line
        let off = s1 * MX;
        let xs = [3.0e5, 6.0e5, 1.2e6];
        let cs = [0.4, 0.7, 1.0];
        x[off..off + 3].copy_from_slice(&xs);
        cdf[off..off + 3].copy_from_slice(&cs);
        p[off] = 0.0; // discrete
        p[off + 1] = (cs[2] - cs[1]) / (xs[2] - xs[1]);
        p[off + 2] = p[off + 1];

        (energy_grid, n_x, x, cdf, p, interp, n_disc)
    }

    /// GPU correlated eout sampling must match the CPU twin: the PCG state
    /// advances bit-for-bit (pure u64 integer math) and the sampled energy
    /// agrees to a few ULPs (one `sqrt` in the LinLin inversion). The
    /// `mu_idx_off` (the returned E_out bin offset) and `valid` flag are pure
    /// integer outputs and must match exactly. Sweeps incident energies below
    /// / inside / above the grid, both slots, and many seeds.
    #[test]
    fn gpu_eout_correlated_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        let (energy_grid, n_x, x, cdf, p, interp, n_disc) = fixture();
        // CSR x-offsets for the dense (MX-strided) fixture: ae-row `r` starts
        // at `r * MX` in the per-x-point arrays (issue #104 -- the sampler now
        // takes per-row offsets rather than a `max_corr_x` stride).
        let x_offset: Vec<u32> = (0..n_x.len()).map(|r| (r * MX) as u32).collect();

        // (eg_off_c, n_corr, incident energies) per case. Slot 0 (eg_off=0,
        // n_corr=2) swept around [1,2] MeV; slot 1 (eg_off=AE, n_corr=1) around
        // 0.5 MeV. Each (slot, energy) crossed with many seeds.
        let cases: &[(u32, u32, &[f64])] = &[
            (0, 2, &[0.5e6, 1.0e6, 1.5e6, 2.0e6, 3.0e6]),
            (AE as u32, 1, &[0.5e6]),
        ];

        let mut energy_in = Vec::new();
        let mut eg_off = Vec::new();
        let mut n_corr = Vec::new();
        let mut seeds = Vec::new();
        let n_seeds = 120u32;
        for &(off, ne, energies) in cases {
            for &e in energies {
                for s in 0..n_seeds {
                    energy_in.push(e);
                    eg_off.push(off);
                    n_corr.push(ne);
                    seeds.push((off.wrapping_mul(7919) + s).wrapping_mul(2_654_435_761));
                }
            }
        }
        let n = seeds.len();

        // CPU reference.
        let mut cpu_e = vec![0.0_f64; n];
        let mut cpu_mu_off = vec![0u32; n];
        let mut cpu_valid = vec![0u32; n];
        let mut cpu_state = vec![0u64; n];
        for i in 0..n {
            let (e, mu, v, st) = sample_correlated_eout_cpu(
                energy_in[i],
                eg_off[i],
                n_corr[i],
                &x_offset,
                &energy_grid,
                &n_x,
                &x,
                &cdf,
                &p,
                &interp,
                &n_disc,
                crate::common::rng::expand_seed(seeds[i]),
            );
            cpu_e[i] = e;
            cpu_mu_off[i] = mu;
            cpu_valid[i] = v;
            cpu_state[i] = st;
        }

        // GPU run.
        let client = ctx.client();
        let ein_h = client.create_from_slice(bytemuck::cast_slice(&energy_in));
        let off_h = client.create_from_slice(bytemuck::cast_slice(&eg_off));
        let nc_h = client.create_from_slice(bytemuck::cast_slice(&n_corr));
        let xoff_h = client.create_from_slice(bytemuck::cast_slice(&x_offset));
        let eg_h = client.create_from_slice(bytemuck::cast_slice(&energy_grid));
        let nx_h = client.create_from_slice(bytemuck::cast_slice(&n_x));
        let x_h = client.create_from_slice(bytemuck::cast_slice(&x));
        let cdf_h = client.create_from_slice(bytemuck::cast_slice(&cdf));
        let p_h = client.create_from_slice(bytemuck::cast_slice(&p));
        let interp_h = client.create_from_slice(bytemuck::cast_slice(&interp));
        let nd_h = client.create_from_slice(bytemuck::cast_slice(&n_disc));
        let seed_h = client.create_from_slice(bytemuck::cast_slice(&seeds));
        let oute_h = client.empty(n * core::mem::size_of::<f64>());
        let outmu_h = client.empty(n * core::mem::size_of::<u32>());
        let outv_h = client.empty(n * core::mem::size_of::<u32>());
        let outs_h = client.empty(n * core::mem::size_of::<u64>());

        const WG: u32 = 64;
        let groups = (n as u32).div_ceil(WG);
        unsafe {
            eout_corr_test_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(groups, 1, 1),
                CubeDim::new_1d(WG),
                BufferArg::from_raw_parts(ein_h, energy_in.len()),
                BufferArg::from_raw_parts(off_h, eg_off.len()),
                BufferArg::from_raw_parts(nc_h, n_corr.len()),
                BufferArg::from_raw_parts(xoff_h, x_offset.len()),
                BufferArg::from_raw_parts(eg_h, energy_grid.len()),
                BufferArg::from_raw_parts(nx_h, n_x.len()),
                BufferArg::from_raw_parts(x_h, x.len()),
                BufferArg::from_raw_parts(cdf_h, cdf.len()),
                BufferArg::from_raw_parts(p_h, p.len()),
                BufferArg::from_raw_parts(interp_h, interp.len()),
                BufferArg::from_raw_parts(nd_h, n_disc.len()),
                BufferArg::from_raw_parts(seed_h, seeds.len()),
                BufferArg::from_raw_parts(oute_h.clone(), n),
                BufferArg::from_raw_parts(outmu_h.clone(), n),
                BufferArg::from_raw_parts(outv_h.clone(), n),
                BufferArg::from_raw_parts(outs_h.clone(), n),
            );
        }
        let gpu_e = bytemuck::cast_slice::<u8, f64>(&client.read_one(oute_h).unwrap()).to_vec();
        let gpu_mu_off =
            bytemuck::cast_slice::<u8, u32>(&client.read_one(outmu_h).unwrap()).to_vec();
        let gpu_valid = bytemuck::cast_slice::<u8, u32>(&client.read_one(outv_h).unwrap()).to_vec();
        let gpu_state = bytemuck::cast_slice::<u8, u64>(&client.read_one(outs_h).unwrap()).to_vec();

        for i in 0..n {
            assert_eq!(
                gpu_state[i], cpu_state[i],
                "sample {i}: PCG state diverged (gpu {} cpu {}, energy {}, off {}, seed {})",
                gpu_state[i], cpu_state[i], energy_in[i], eg_off[i], seeds[i]
            );
            assert_eq!(
                gpu_valid[i], cpu_valid[i],
                "sample {i}: valid flag diverged (gpu {} cpu {}, off {}, seed {})",
                gpu_valid[i], cpu_valid[i], eg_off[i], seeds[i]
            );
            assert_eq!(
                gpu_mu_off[i], cpu_mu_off[i],
                "sample {i}: mu_idx_off diverged (gpu {} cpu {}, off {}, seed {})",
                gpu_mu_off[i], cpu_mu_off[i], eg_off[i], seeds[i]
            );
            let tol = 1e-9 * cpu_e[i].abs().max(1.0);
            assert!(
                (gpu_e[i] - cpu_e[i]).abs() <= tol,
                "sample {i}: energy diverged (gpu {} cpu {}, tol {tol}, energy {}, off {}, seed {})",
                gpu_e[i], cpu_e[i], energy_in[i], eg_off[i], seeds[i]
            );
            // Slot 0 (n_corr=2, n_x>=2) must be valid.
            if eg_off[i] == 0 {
                assert_eq!(cpu_valid[i], 1, "sample {i}: slot 0 not valid");
            }
        }
        println!(
            "eout correlated: {} samples, GPU state+valid+mu_off bit-exact, energy within 1e-9 rel",
            n
        );
    }
}
