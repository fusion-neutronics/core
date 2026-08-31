//! Shared continuous-tabular outgoing-energy sampler (ENDF `ContinuousTabular`).
//!
//! Bracket the incident energy between two grid points, stochastically pick
//! one bracketing distribution, invert its outgoing-energy CDF, then stretch
//! the result into the incident-energy-interpolated bounds. Extracted verbatim
//! from the neutron transport kernel's `eout_kind == CONTINUOUS_TABULAR`
//! branch so one `#[cube]` sampler serves neutron outgoing energy and (in the
//! coupled-photon work) neutron-induced photon production -- both use
//! `EnergyDistribution::ContinuousTabular`. Faithful to `yamc_nuclide`'s
//! `EnergyDistribution::sample` + `TabulatedProbability::sample_with_discrete_info`.
//!
//! The walk draws up to two PCG-32 uniforms (the incident-energy bracket pick,
//! then the CDF inversion -- the second only when the chosen bin has >= 2
//! points), so it both consumes and returns the RNG `state`; passing the
//! caller's current `e_cm` as `e_default` and returning it unchanged when the
//! slot has no usable distribution reproduces the kernel's in-place behaviour
//! exactly. Pure integer-PCG + f64 arithmetic apart from one `sqrt` in the
//! LinLin quadratic inversion, so the `#[cube]` kernel matches the
//! [`sample_continuous_tabular_eout_cpu`] twin to within a few ULPs
//! (`gpu_eout_continuous_tabular_matches_cpu`).

use crate::common::pcg32::{expand_seed, pcg_out};
use crate::common::rng::{PCG_INCR, PCG_MULT};
use crate::neutron::xs::constants::ANGLE_INTERP_LINLIN;
use cubecl::prelude::*;

/// Result of [`sample_continuous_tabular_eout`]: the sampled outgoing energy
/// (the single tabulated point when the chosen bin has exactly 1 point, or the
/// passed-in `e_default` when it has 0 points) and the advanced 64-bit PCG state.
#[derive(CubeType)]
pub struct EoutSample {
    pub e_cm: f64,
    pub state: u64,
}

/// Sample a `ContinuousTabular` outgoing energy. `eg_off_e` is the slot's
/// base ae-row into the per-incident-energy buffers (`eout_ae_offset[mat_slot]`
/// with tight CSR storage, issue #104); `eout_x_offset` gives the global start
/// of each ae-row's `(x, cdf, p)` points in the flat `eout_x` / `eout_cdf` /
/// `eout_p` arrays (row `eg_off_e + bin` starts at `eout_x_offset[eg_off_e +
/// bin]`). `hist_outer` is the slot's `histogram_interp` flag. The flat buffers
/// (`eout_energy_grid`, `eout_n_x`, `eout_x`, `eout_cdf`, `eout_p`,
/// `eout_interp`, `eout_n_discrete`) are the kernel's eout buffers. Returns
/// the sampled energy and the advanced `state`.
#[cube]
pub fn sample_continuous_tabular_eout(
    e_default: f64,
    energy: f64,
    eg_off_e: u32,
    n_eout: u32,
    hist_outer: u32,
    eout_x_offset: &[u32],
    eout_energy_grid: &[f64],
    eout_n_x: &[u32],
    eout_x: &[f64],
    eout_cdf: &[f64],
    eout_p: &[f64],
    eout_interp: &[u32],
    eout_n_discrete: &[u32],
    state_in: u64,
) -> EoutSample {
    let mut state = state_in;
    let mut e_cm = e_default;

    let mut i_eb = 0u32;
    let e_first = eout_energy_grid[eg_off_e as usize];
    let e_last = eout_energy_grid[(eg_off_e + n_eout - 1u32) as usize];
    let mut r_eb = 0.0_f64;
    if energy >= e_last {
        if n_eout > 1u32 {
            i_eb = n_eout - 2u32;
        }
        r_eb = 1.0;
    } else if energy > e_first {
        let mut k = 0u32;
        while k + 1u32 < n_eout {
            let e_k = eout_energy_grid[(eg_off_e + k) as usize];
            let e_k1 = eout_energy_grid[(eg_off_e + k + 1u32) as usize];
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

    // Stochastic incident-energy-bracket pick. Suppressed when the outer
    // histogram_interp is set (mirrors CPU's `l = i` when histogram).
    let d_xi_eeb = crate::common::pcg32::draw_uniform(state);
    state = d_xi_eeb.state;
    let xi_eeb = d_xi_eeb.xi;
    let mut bin_e = i_eb;
    if hist_outer == 0u32 && r_eb > xi_eeb && bin_e + 1u32 < n_eout {
        bin_e = i_eb + 1u32;
    }

    let n_x = eout_n_x[(eg_off_e + bin_e) as usize];
    if n_x >= 2u32 {
        let x_off = eout_x_offset[(eg_off_e + bin_e) as usize];
        let n_disc = eout_n_discrete[(eg_off_e + bin_e) as usize];
        let interp_kind = eout_interp[(eg_off_e + bin_e) as usize];
        let d_xi_x = crate::common::pcg32::draw_uniform(state);
        state = d_xi_x.state;
        let xi_x = d_xi_x.xi;

        // Discrete-then-continuous CDF search matching the CPU reference
        // `sample_with_discrete_info` (issue #103): the discrete head (first
        // `n_disc` points) is searched with `xi < c[k]` (selects the exact
        // line); the continuous tail from `n_disc` with `xi <= c[k+1]`,
        // carrying `c_j` as the CPU does. The previous single scan collapsed
        // every discrete line onto index 0. For `n_disc == 0` this is identical
        // to the old scan (`c_j == c[j]`), so non-discrete distributions are
        // unchanged.
        let mut j = 0u32;
        let mut c_j = eout_cdf[x_off as usize];
        let mut is_discrete = 0u32;
        let mut done = 0u32;
        let mut kd = 0u32;
        while kd < n_disc {
            if done == 0u32 {
                j = kd;
                c_j = eout_cdf[(x_off + kd) as usize];
                if xi_x < c_j {
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
                let c_k1 = eout_cdf[(x_off + kc + 1u32) as usize];
                if xi_x <= c_k1 {
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
        let x_j = eout_x[(x_off + j) as usize];
        let x_j1 = eout_x[(x_off + j + 1u32) as usize];
        let p_j = eout_p[(x_off + j) as usize];
        let p_j1 = eout_p[(x_off + j + 1u32) as usize];
        let dx = x_j1 - x_j;
        let mut e_sampled = x_j;
        if is_discrete == 0u32 {
            if interp_kind == ANGLE_INTERP_LINLIN && dx > 0.0 {
                // CPU's LinLin Tabular quadratic inversion:
                // x_k + (sqrt(p_k^2 + 2*m*(r - c_k)) - p_k) / m.
                let m = (p_j1 - p_j) / dx;
                let abs_m = if m < 0.0 { -m } else { m };
                if abs_m < 1e-30 {
                    if p_j > 0.0 {
                        e_sampled = x_j + (xi_x - c_j) / p_j;
                    }
                } else {
                    let mut disc = p_j * p_j + 2.0 * m * (xi_x - c_j);
                    if disc < 0.0 {
                        disc = 0.0;
                    }
                    e_sampled = x_j + (disc.sqrt() - p_j) / m;
                }
            } else if p_j > 0.0 {
                // Histogram bracket (continuous part).
                e_sampled = x_j + (xi_x - c_j) / p_j;
            }
        }
        // else: discrete line -- e_sampled stays x_j.

        // Bracket-bound stretch -- skipped when the outer histogram_interp is
        // set or the sampled bin was discrete (mirrors CPU's early-return).
        if hist_outer == 0u32 && is_discrete == 0u32 && n_eout >= 2u32 {
            let n_x_i = eout_n_x[(eg_off_e + i_eb) as usize];
            let n_x_i1 = eout_n_x[(eg_off_e + i_eb + 1u32) as usize];
            if n_x_i >= 2u32 && n_x_i1 >= 2u32 {
                let x_off_i = eout_x_offset[(eg_off_e + i_eb) as usize];
                let x_off_i1 = eout_x_offset[(eg_off_e + i_eb + 1u32) as usize];
                let e_i_1 = eout_x[x_off_i as usize];
                let e_i_k = eout_x[(x_off_i + n_x_i - 1u32) as usize];
                let e_i1_1 = eout_x[x_off_i1 as usize];
                let e_i1_k = eout_x[(x_off_i1 + n_x_i1 - 1u32) as usize];
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
        e_cm = e_sampled;
    } else if n_x == 1u32 {
        // Single tabulated outgoing energy (e.g. a fixed level-inelastic gamma
        // line stored as a one-point ContinuousTabular row): emit that energy.
        // Returning `e_default` here was wrong -- on the coupled-photon path
        // `e_default` was the incident NEUTRON energy, so these one-point rows
        // emitted spurious ~14 MeV photons (issue #175). Matches the CPU
        // single-point `TabulatedProbability` sample. No RNG draw, so the kernel
        // and its CPU twin stay in lock-step.
        let x_off = eout_x_offset[(eg_off_e + bin_e) as usize];
        e_cm = eout_x[x_off as usize];
    }
    // n_x == 0: no sampleable point; `e_cm` stays `e_default` (callers pass 0.0
    // so the empty row drops the secondary rather than emitting a bogus energy).

    EoutSample { e_cm, state }
}

/// CPU twin of [`sample_continuous_tabular_eout`]. Same algorithm and the same
/// PCG draws (64-bit state advanced via `wrapping_*`), so it matches the
/// `#[cube]` kernel to within a few ULPs. Returns `(e_cm, state)`.
#[allow(clippy::too_many_arguments)]
pub fn sample_continuous_tabular_eout_cpu(
    e_default: f64,
    energy: f64,
    eg_off_e: u32,
    n_eout: u32,
    hist_outer: u32,
    eout_x_offset: &[u32],
    eout_energy_grid: &[f64],
    eout_n_x: &[u32],
    eout_x: &[f64],
    eout_cdf: &[f64],
    eout_p: &[f64],
    eout_interp: &[u32],
    eout_n_discrete: &[u32],
    state_in: u64,
) -> (f64, u64) {
    let mut state = state_in;
    let mut e_cm = e_default;

    let mut i_eb = 0u32;
    let e_first = eout_energy_grid[eg_off_e as usize];
    let e_last = eout_energy_grid[(eg_off_e + n_eout - 1) as usize];
    let mut r_eb = 0.0_f64;
    if energy >= e_last {
        if n_eout > 1 {
            i_eb = n_eout - 2;
        }
        r_eb = 1.0;
    } else if energy > e_first {
        let mut k = 0u32;
        while k + 1 < n_eout {
            let e_k = eout_energy_grid[(eg_off_e + k) as usize];
            let e_k1 = eout_energy_grid[(eg_off_e + k + 1) as usize];
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

    let s_xi_eeb = state;
    let r_xi_eeb = pcg_out(s_xi_eeb);
    state = s_xi_eeb.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
    let xi_eeb = (r_xi_eeb as f64 + 1.0) * (1.0 / 4_294_967_297.0);
    let mut bin_e = i_eb;
    if hist_outer == 0 && r_eb > xi_eeb && bin_e + 1 < n_eout {
        bin_e = i_eb + 1;
    }

    let n_x = eout_n_x[(eg_off_e + bin_e) as usize];
    if n_x >= 2 {
        let x_off = eout_x_offset[(eg_off_e + bin_e) as usize];
        let n_disc = eout_n_discrete[(eg_off_e + bin_e) as usize];
        let interp_kind = eout_interp[(eg_off_e + bin_e) as usize];
        let s_xi_x = state;
        let r_xi_x = pcg_out(s_xi_x);
        state = s_xi_x.wrapping_mul(PCG_MULT).wrapping_add(PCG_INCR);
        let xi_x = (r_xi_x as f64 + 1.0) * (1.0 / 4_294_967_297.0);

        // Discrete-then-continuous CDF search (issue #103); bit-twin of the
        // `#[cube]` scan above. See its comment.
        let mut j = 0u32;
        let mut c_j = eout_cdf[x_off as usize];
        let mut is_discrete = 0u32;
        let mut done = 0u32;
        let mut kd = 0u32;
        while kd < n_disc {
            if done == 0 {
                j = kd;
                c_j = eout_cdf[(x_off + kd) as usize];
                if xi_x < c_j {
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
                let c_k1 = eout_cdf[(x_off + kc + 1) as usize];
                if xi_x <= c_k1 {
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
        let x_j = eout_x[(x_off + j) as usize];
        let x_j1 = eout_x[(x_off + j + 1) as usize];
        let p_j = eout_p[(x_off + j) as usize];
        let p_j1 = eout_p[(x_off + j + 1) as usize];
        let dx = x_j1 - x_j;
        let mut e_sampled = x_j;
        if is_discrete == 0 {
            if interp_kind == ANGLE_INTERP_LINLIN && dx > 0.0 {
                let m = (p_j1 - p_j) / dx;
                let abs_m = if m < 0.0 { -m } else { m };
                if abs_m < 1e-30 {
                    if p_j > 0.0 {
                        e_sampled = x_j + (xi_x - c_j) / p_j;
                    }
                } else {
                    let mut disc = p_j * p_j + 2.0 * m * (xi_x - c_j);
                    if disc < 0.0 {
                        disc = 0.0;
                    }
                    e_sampled = x_j + (disc.sqrt() - p_j) / m;
                }
            } else if p_j > 0.0 {
                e_sampled = x_j + (xi_x - c_j) / p_j;
            }
        }

        if hist_outer == 0 && is_discrete == 0 && n_eout >= 2 {
            let n_x_i = eout_n_x[(eg_off_e + i_eb) as usize];
            let n_x_i1 = eout_n_x[(eg_off_e + i_eb + 1) as usize];
            if n_x_i >= 2 && n_x_i1 >= 2 {
                let x_off_i = eout_x_offset[(eg_off_e + i_eb) as usize];
                let x_off_i1 = eout_x_offset[(eg_off_e + i_eb + 1) as usize];
                let e_i_1 = eout_x[x_off_i as usize];
                let e_i_k = eout_x[(x_off_i + n_x_i - 1) as usize];
                let e_i1_1 = eout_x[x_off_i1 as usize];
                let e_i1_k = eout_x[(x_off_i1 + n_x_i1 - 1) as usize];
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
        e_cm = e_sampled;
    } else if n_x == 1 {
        // Single tabulated outgoing energy: emit it (bit-twin of the `#[cube]`
        // n_x == 1 branch above; issue #175).
        let x_off = eout_x_offset[(eg_off_e + bin_e) as usize];
        e_cm = eout_x[x_off as usize];
    }
    // n_x == 0: `e_cm` stays `e_default`.

    (e_cm, state)
}

/// Test/validation kernel: one sample per thread, using a fixed small layout
/// (`max_eout_ae = 4`, `max_eout_x = 8`) so the buffers stay tiny. Each
/// sample carries its own incident energy, slot offset, `n_eout`,
/// histogram flag, default energy, and PCG seed.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn eout_ct_test_kernel(
    e_default: &[f64],
    energy_in: &[f64],
    eg_off_e: &[u32],
    n_eout: &[u32],
    hist_outer: &[u32],
    eout_x_offset: &[u32],
    eout_energy_grid: &[f64],
    eout_n_x: &[u32],
    eout_x: &[f64],
    eout_cdf: &[f64],
    eout_p: &[f64],
    eout_interp: &[u32],
    eout_n_discrete: &[u32],
    seeds: &[u32],
    out_e: &mut [f64],
    out_state: &mut [u64],
) {
    if ABSOLUTE_POS >= out_e.len() {
        terminate!();
    }
    let r = sample_continuous_tabular_eout(
        e_default[ABSOLUTE_POS],
        energy_in[ABSOLUTE_POS],
        eg_off_e[ABSOLUTE_POS],
        n_eout[ABSOLUTE_POS],
        hist_outer[ABSOLUTE_POS],
        eout_x_offset,
        eout_energy_grid,
        eout_n_x,
        eout_x,
        eout_cdf,
        eout_p,
        eout_interp,
        eout_n_discrete,
        expand_seed(seeds[ABSOLUTE_POS]),
    );
    out_e[ABSOLUTE_POS] = r.e_cm;
    out_state[ABSOLUTE_POS] = r.state;
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContext, GpuInitError, WgpuRuntime};

    // Fixed test layout (kept tiny; the kernel hardcodes max_eout_x = 8).
    const AE: usize = 4; // max_eout_ae for the test
    const MX: usize = 8; // max_eout_x for the test

    /// Build a 2-slot eout fixture. Slot 0: two incident energies, each a
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
        let n_slots = 3usize;
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
            // A normalised CDF and a matching (rough) PDF.
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

        // Slot 2, single incident energy with a SINGLE outgoing point
        // (n_x == 1): a fixed level-inelastic-style gamma line. The sampler must
        // emit this point, not e_default (issue #175).
        let s2 = 2 * AE;
        energy_grid[s2] = 1.0e6;
        n_x[s2] = 1;
        interp[s2] = 0;
        x[s2 * MX] = 3.0e4;
        cdf[s2 * MX] = 1.0;
        p[s2 * MX] = 0.0;

        (energy_grid, n_x, x, cdf, p, interp, n_disc)
    }

    /// GPU continuous-tabular eout sampling must match the CPU twin: the PCG
    /// state advances bit-for-bit (pure u64 integer math) and the sampled
    /// energy agrees to a few ULPs (one `sqrt` in the LinLin inversion).
    /// Sweeps incident energies below / inside / above the grid, both slots,
    /// and many seeds.
    #[test]
    fn gpu_eout_continuous_tabular_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        let (energy_grid, n_x, x, cdf, p, interp, n_disc) = fixture();
        // CSR x-offsets for the padded fixture: row `(slot*AE + ae)` starts at
        // `(slot*AE + ae) * MX` (issue #104 -- the sampler now takes per-row
        // offsets rather than a `max_eout_x` stride).
        let x_offset: Vec<u32> = (0..n_x.len()).map(|r| (r * MX) as u32).collect();

        // Build the per-sample arrays: slot 0 (eg_off=0, n_eout=2) swept over
        // energies around [1,2] MeV; slot 1 (eg_off=AE, n_eout=1) around
        // 0.5 MeV. Each (slot, energy) crossed with many seeds.
        let cases: &[(u32, u32, &[f64], u32)] = &[
            // (eg_off_e, n_eout, incident energies, hist_outer)
            (0, 2, &[0.5e6, 1.0e6, 1.5e6, 2.0e6, 3.0e6], 0),
            (AE as u32, 1, &[0.5e6], 0),
            // Single-point (n_x == 1) slot swept below / at / above the grid.
            ((2 * AE) as u32, 1, &[0.5e6, 1.0e6, 1.406e7], 0),
        ];

        let mut e_default = Vec::new();
        let mut energy_in = Vec::new();
        let mut eg_off = Vec::new();
        let mut n_eout = Vec::new();
        let mut hist = Vec::new();
        let mut seeds = Vec::new();
        let n_seeds = 120u32;
        for &(off, ne, energies, ho) in cases {
            for &e in energies {
                for s in 0..n_seeds {
                    e_default.push(-1.0); // sentinel: must be overwritten when n_x >= 2
                    energy_in.push(e);
                    eg_off.push(off);
                    n_eout.push(ne);
                    hist.push(ho);
                    seeds.push((off.wrapping_mul(7919) + s).wrapping_mul(2_654_435_761));
                }
            }
        }
        let n = seeds.len();

        // CPU reference. Expand each 32-bit seed to the 64-bit PCG state
        // exactly as the kernel does (host wrapping splitmix64, issue #274).
        let mut cpu_e = vec![0.0_f64; n];
        let mut cpu_state = vec![0u64; n];
        for i in 0..n {
            let (e, st) = sample_continuous_tabular_eout_cpu(
                e_default[i],
                energy_in[i],
                eg_off[i],
                n_eout[i],
                hist[i],
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
            cpu_state[i] = st;
        }

        // GPU run.
        let client = ctx.client();
        let edef_h = client.create_from_slice(bytemuck::cast_slice(&e_default));
        let ein_h = client.create_from_slice(bytemuck::cast_slice(&energy_in));
        let off_h = client.create_from_slice(bytemuck::cast_slice(&eg_off));
        let ne_h = client.create_from_slice(bytemuck::cast_slice(&n_eout));
        let ho_h = client.create_from_slice(bytemuck::cast_slice(&hist));
        let xoff_h = client.create_from_slice(bytemuck::cast_slice(&x_offset));
        let eg_h = client.create_from_slice(bytemuck::cast_slice(&energy_grid));
        let nx_h = client.create_from_slice(bytemuck::cast_slice(&n_x));
        let x_h = client.create_from_slice(bytemuck::cast_slice(&x));
        let cdf_h = client.create_from_slice(bytemuck::cast_slice(&cdf));
        let interp_h = client.create_from_slice(bytemuck::cast_slice(&interp));
        let nd_h = client.create_from_slice(bytemuck::cast_slice(&n_disc));
        let p_h = client.create_from_slice(bytemuck::cast_slice(&p));
        let seed_h = client.create_from_slice(bytemuck::cast_slice(&seeds));
        let oute_h = client.empty(n * core::mem::size_of::<f64>());
        let outs_h = client.empty(n * core::mem::size_of::<u64>());

        const WG: u32 = 64;
        let groups = (n as u32).div_ceil(WG);
        unsafe {
            eout_ct_test_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(groups, 1, 1),
                CubeDim::new_1d(WG),
                BufferArg::from_raw_parts(edef_h, e_default.len()),
                BufferArg::from_raw_parts(ein_h, energy_in.len()),
                BufferArg::from_raw_parts(off_h, eg_off.len()),
                BufferArg::from_raw_parts(ne_h, n_eout.len()),
                BufferArg::from_raw_parts(ho_h, hist.len()),
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
                BufferArg::from_raw_parts(outs_h.clone(), n),
            );
        }
        let gpu_e = bytemuck::cast_slice::<u8, f64>(&client.read_one(oute_h).unwrap()).to_vec();
        let gpu_state = bytemuck::cast_slice::<u8, u64>(&client.read_one(outs_h).unwrap()).to_vec();

        for i in 0..n {
            assert_eq!(
                gpu_state[i], cpu_state[i],
                "sample {i}: PCG state diverged (gpu {} cpu {}, energy {}, off {}, seed {})",
                gpu_state[i], cpu_state[i], energy_in[i], eg_off[i], seeds[i]
            );
            let tol = 1e-9 * cpu_e[i].abs().max(1.0);
            assert!(
                (gpu_e[i] - cpu_e[i]).abs() <= tol,
                "sample {i}: energy diverged (gpu {} cpu {}, tol {tol}, energy {}, off {}, seed {})",
                gpu_e[i], cpu_e[i], energy_in[i], eg_off[i], seeds[i]
            );
            // Slot 0 (n_eout=2, n_x>=2) must overwrite the -1 sentinel.
            if eg_off[i] == 0 {
                assert!(
                    cpu_e[i] >= 0.0,
                    "sample {i}: slot 0 left default ({})",
                    cpu_e[i]
                );
            }
            // Slot 2 (n_x == 1) must emit the tabulated point, not the -1
            // sentinel default (issue #175).
            if eg_off[i] == (2 * AE) as u32 {
                assert!(
                    (cpu_e[i] - 3.0e4).abs() < 1e-6,
                    "sample {i}: single-point slot returned {} (expected 3.0e4)",
                    cpu_e[i]
                );
            }
        }
        println!(
            "eout continuous-tabular: {} samples, GPU state bit-exact, energy within 1e-9 rel",
            n
        );
    }

    /// Regression for issue #175: a single-point (`n_x == 1`) incident-energy
    /// row (e.g. a fixed level-inelastic gamma line) must emit the tabulated
    /// point, NOT the `e_default` fallback. Returning `e_default` (the incident
    /// neutron energy on the coupled-photon path) produced spurious ~14 MeV
    /// photons. Exercises the CPU twin directly (the GPU kernel is its bit-twin
    /// and is covered by `gpu_eout_continuous_tabular_matches_cpu`).
    #[test]
    fn single_point_row_emits_tabulated_point() {
        let energy_grid = vec![1.0e6_f64];
        let n_x = vec![1u32];
        let x_offset = vec![0u32];
        let x = vec![3.0e4_f64]; // 30 keV line
        let cdf = vec![1.0_f64];
        let p = vec![0.0_f64];
        let interp = vec![0u32];
        let n_disc = vec![0u32];
        // A deliberately wrong default (stand-in for the incident neutron energy).
        const WRONG_DEFAULT: f64 = 1.406e7;
        for &energy in &[1.0e5_f64, 1.0e6, 1.406e7] {
            for seed in [1u32, 7, 12_345, 0xDEAD_BEEF] {
                let (e, _st) = sample_continuous_tabular_eout_cpu(
                    WRONG_DEFAULT,
                    energy,
                    0,
                    1,
                    0,
                    &x_offset,
                    &energy_grid,
                    &n_x,
                    &x,
                    &cdf,
                    &p,
                    &interp,
                    &n_disc,
                    crate::common::rng::expand_seed(seed),
                );
                assert!(
                    (e - 3.0e4).abs() < 1e-9,
                    "n_x==1 returned {e}, expected the tabulated 3.0e4 (energy {energy}, seed {seed})"
                );
            }
        }
    }

    /// An empty (`n_x == 0`) row has nothing to sample and returns `e_default`,
    /// so coupled-photon callers pass 0.0 and drop the secondary (issue #175).
    #[test]
    fn empty_row_returns_default() {
        let energy_grid = vec![1.0e6_f64];
        let n_x = vec![0u32];
        let x_offset = vec![0u32];
        let x = vec![0.0_f64];
        let cdf = vec![0.0_f64];
        let p = vec![0.0_f64];
        let interp = vec![0u32];
        let n_disc = vec![0u32];
        let (e, _st) = sample_continuous_tabular_eout_cpu(
            0.0,
            1.406e7,
            0,
            1,
            0,
            &x_offset,
            &energy_grid,
            &n_x,
            &x,
            &cdf,
            &p,
            &interp,
            &n_disc,
            42,
        );
        assert_eq!(
            e, 0.0,
            "n_x==0 should return the e_default (0.0) so the caller drops it"
        );
    }
}
