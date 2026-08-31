//! Shared Kalbach-Mann (ENDF File-6 Law-4) correlated outgoing-energy + angle
//! sampler.
//!
//! Bracket the incident energy between two `km_energy_grid` points,
//! stochastically pick one bracketing distribution (via the shared
//! [`pick_energy_bracket`](crate::common::sampling::energy_bracket::pick_energy_bracket)),
//! invert the chosen slice's outgoing-energy CDF (histogram / discrete-prefix
//! or LinLin quadratic), stretch the result into the incident-energy
//! interpolated bounds, then sample `mu` from the Kalbach-Mann angular formula
//! using the per-`(E_in, E_out)` `r` (pre-equilibrium fraction) and `a`
//! (slope) parameters. This is the kernel's `eout_kind ==
//! EOUT_KIND_KALBACH_MANN (3)` branch, faithful to `yamc_nuclide`'s
//! `KalbachMann::sample`.
//!
//! RNG draw order (preserved exactly from the inline kernel):
//!   1. incident-energy bracket pick (one uniform, the shared helper),
//!   2. E_out CDF inversion (one uniform), only when the chosen slice has
//!      `>= 2` outgoing points,
//!   3. compound-vs-precompound mixture pick (one uniform),
//!   4. mu sample (one uniform).
//!
//! Draws 2-4 happen only when the chosen slice has `>= 2` points (the
//! `n_kx >= 2` guard); when it does not, the helper consumes only the
//! bracket-pick draw and returns `e_valid == 0` / `mu_valid == 0`, so the
//! caller leaves its current `e_cm` / `mu_sampled` untouched.
//!
//! The mu formula needs `sinh(a)`, `exp(+/-a)`, and `ln`. There is no GPU
//! `sinh` polyfill, so -- exactly as the inline kernel did -- `sinh(a)` is
//! formed from `(exp(a) - exp(-a)) / 2` using the [`exp_f64`] polyfill, and
//! the `arcsinh`/log-form mu uses [`ln_f64`]. Because `e_out` and `mu` flow
//! through the `exp`/`ln` polyfills (~16 ULP each per `common/polyfills.rs`)
//! and one `sqrt`, GPU vs CPU is NOT bit-exact for those: the parity test
//! uses ~1e-9 relative for `e_out` and ~1e-9 absolute for `mu`, while the PCG
//! `state` and the `e_valid` / `mu_valid` flags stay bit-for-bit
//! (`gpu_kalbach_mann_eout_mu_matches_cpu`).

use crate::common::pcg32::expand_seed;
use crate::common::polyfills::{exp_f64, ln_f64};
use cubecl::prelude::*;

/// Result of [`sample_kalbach_mann`]: the sampled outgoing energy (apply only
/// when `e_valid == 1`), the sampled scattering cosine `mu` (apply only when
/// `mu_valid == 1`), and the advanced PCG-32 state.
///
/// `e_valid == 1` iff the chosen E_in slice had `>= 2` outgoing points AND the
/// stretched energy came out `> 0` (mirroring the kernel's `if e_sampled_km >
/// 0.0 { e_cm = e_sampled_km }`). `mu_valid == 1` iff the slice had `>= 2`
/// outgoing points (the kernel always overwrites `mu_sampled` in that case).
#[derive(CubeType)]
pub struct KalbachMannSample {
    pub e_out: f64,
    pub mu: f64,
    pub e_valid: u32,
    pub mu_valid: u32,
    pub state: u64,
}

/// Sample a Kalbach-Mann (File-6 Law-4) outgoing energy and scattering cosine.
/// `eg_off_k` is the slot's base ae-row into the per-incident-energy buffers
/// (`km_ae_offset[mat_slot]` with tight CSR storage, issue #104); `km_x_offset`
/// gives the global start of each ae-row's `(x, p, c, r, a)` points in the flat
/// `km_x` / `km_p` / `km_c` / `km_r` / `km_a` arrays (row `eg_off_k + bin`
/// starts at `km_x_offset[eg_off_k + bin]`). The flat buffers
/// (`km_energy_grid`, `km_n_x`, `km_interp`, `km_n_discrete`, `km_x`, `km_p`,
/// `km_c`, `km_r`, `km_a`) are the kernel's km_* buffers. Returns the sampled
/// energy + mu, their `valid` flags, and the advanced `state`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn sample_kalbach_mann(
    energy: f64,
    eg_off_k: u32,
    n_kae: u32,
    km_x_offset: &[u32],
    km_energy_grid: &[f64],
    km_n_x: &[u32],
    km_interp: &[u32],
    km_n_discrete: &[u32],
    km_x: &[f64],
    km_p: &[f64],
    km_c: &[f64],
    km_r: &[f64],
    km_a: &[f64],
    state_in: u64,
) -> KalbachMannSample {
    let mut state = state_in;
    let mut e_out = 0.0_f64;
    let mut mu = 0.0_f64;
    let mut e_valid = 0u32;
    let mut mu_valid = 0u32;

    let mut i_kb = 0u32;
    let e_kfirst = km_energy_grid[eg_off_k as usize];
    let e_klast = km_energy_grid[(eg_off_k + n_kae - 1u32) as usize];
    let mut r_kb = 0.0_f64;
    if energy >= e_klast {
        if n_kae > 1u32 {
            i_kb = n_kae - 2u32;
        }
        r_kb = 1.0;
    } else if energy > e_kfirst {
        let mut k = 0u32;
        while k + 1u32 < n_kae {
            let e_k = km_energy_grid[(eg_off_k + k) as usize];
            let e_k1 = km_energy_grid[(eg_off_k + k + 1u32) as usize];
            if energy >= e_k && energy < e_k1 {
                i_kb = k;
                let de = e_k1 - e_k;
                if de > 0.0 {
                    r_kb = (energy - e_k) / de;
                }
            }
            k += 1u32;
        }
    }

    // Stochastic incident-energy bracket pick: choose i_kb or i_kb+1 weighted
    // by r_kb (shared helper draws one uniform).
    let pick_k =
        crate::common::sampling::energy_bracket::pick_energy_bracket(r_kb, i_kb, n_kae, state);
    state = pick_k.state;
    let bin_k = pick_k.bin;

    let n_kx = km_n_x[(eg_off_k + bin_k) as usize];
    if n_kx >= 2u32 {
        let interp_k = km_interp[(eg_off_k + bin_k) as usize];
        let n_disc = km_n_discrete[(eg_off_k + bin_k) as usize];
        let x_off_k = km_x_offset[(eg_off_k + bin_k) as usize];
        let d_xkx = crate::common::pcg32::draw_uniform(state);
        state = d_xkx.state;
        let xi_kx = d_xkx.xi;

        // Discrete-then-continuous CDF inversion matching the CPU reference
        // `sample_with_discrete_info` (issue #103): discrete head (first
        // `n_disc` points) searched with `xi < c[k]` (exact line), continuous
        // tail from `n_disc` with `xi <= c[k+1]`, carrying `c_kj`. The previous
        // single scan collapsed every discrete line onto index 0. For
        // `n_disc == 0` this is identical to the old scan.
        let mut kj = 0u32;
        let mut c_kj = km_c[x_off_k as usize];
        let mut is_discrete = 0u32;
        let mut done = 0u32;
        let mut md = 0u32;
        while md < n_disc {
            if done == 0u32 {
                kj = md;
                c_kj = km_c[(x_off_k + md) as usize];
                if xi_kx < c_kj {
                    is_discrete = 1u32;
                    done = 1u32;
                }
            }
            md += 1u32;
        }
        let mut mk = n_disc;
        while mk + 1u32 < n_kx {
            if done == 0u32 {
                kj = mk;
                let c_mk1 = km_c[(x_off_k + mk + 1u32) as usize];
                if xi_kx <= c_mk1 {
                    done = 1u32;
                } else {
                    kj = mk + 1u32;
                    c_kj = c_mk1;
                }
            }
            mk += 1u32;
        }
        if is_discrete == 0u32 && kj >= n_kx - 1u32 {
            kj = n_kx - 2u32;
        }
        let p_kj = km_p[(x_off_k + kj) as usize];
        let x_kj = km_x[(x_off_k + kj) as usize];
        let mut e_sampled_km = x_kj;
        let mut km_r_sampled = km_r[(x_off_k + kj) as usize];
        let mut km_a_sampled = km_a[(x_off_k + kj) as usize];

        if interp_k == 0u32 || kj < n_disc {
            // Histogram (or discrete prefix):
            // E_out = e[kj] + (xi - c[kj]) / p[kj]
            if p_kj > 0.0 && kj >= n_disc {
                e_sampled_km = x_kj + (xi_kx - c_kj) / p_kj;
            }
        } else {
            // LinLin: quadratic formula
            // E_out = e[kj] + (sqrt(p_kj^2 + 2*frac*(xi - c_kj)) - p_kj) / frac
            // where frac = (p[kj+1] - p[kj]) / (e[kj+1] - e[kj])
            let p_kj1 = km_p[(x_off_k + kj + 1u32) as usize];
            let x_kj1 = km_x[(x_off_k + kj + 1u32) as usize];
            let de = x_kj1 - x_kj;
            if de > 0.0 {
                let frac = (p_kj1 - p_kj) / de;
                if frac == 0.0 {
                    if p_kj > 0.0 {
                        e_sampled_km = x_kj + (xi_kx - c_kj) / p_kj;
                    }
                } else {
                    let disc_l = p_kj * p_kj + 2.0 * frac * (xi_kx - c_kj);
                    let mut disc_clamped = disc_l;
                    if disc_l <= 0.0 {
                        disc_clamped = 0.0_f64;
                    }
                    e_sampled_km = x_kj + (disc_clamped.sqrt() - p_kj) / frac;
                }
                // Interpolate r, a using the in-bracket fraction
                // f = (E - e[kj]) / (e[kj+1] - e[kj]).
                let f = ((e_sampled_km - x_kj) / de).clamp(0.0, 1.0);
                let r_kj1 = km_r[(x_off_k + kj + 1u32) as usize];
                let a_kj1 = km_a[(x_off_k + kj + 1u32) as usize];
                km_r_sampled = km_r_sampled + f * (r_kj1 - km_r_sampled);
                km_a_sampled = km_a_sampled + f * (a_kj1 - km_a_sampled);
            }
        }

        // Bracket-bound stretch: map the sampled E_out from the picked
        // bracket's domain into the incident-energy-interpolated bounds (only
        // for continuous bins, kj >= n_disc).
        if kj >= n_disc && n_kae >= 2u32 {
            let n_x_i = km_n_x[(eg_off_k + i_kb) as usize];
            let n_x_i1 = km_n_x[(eg_off_k + i_kb + 1u32) as usize];
            let n_disc_i = km_n_discrete[(eg_off_k + i_kb) as usize];
            let n_disc_i1 = km_n_discrete[(eg_off_k + i_kb + 1u32) as usize];
            if n_x_i > n_disc_i && n_x_i1 > n_disc_i1 {
                let x_off_i = km_x_offset[(eg_off_k + i_kb) as usize];
                let x_off_i1 = km_x_offset[(eg_off_k + i_kb + 1u32) as usize];
                let e_i_1 = km_x[(x_off_i + n_disc_i) as usize];
                let e_i_k = km_x[(x_off_i + n_x_i - 1u32) as usize];
                let e_i1_1 = km_x[(x_off_i1 + n_disc_i1) as usize];
                let e_i1_k = km_x[(x_off_i1 + n_x_i1 - 1u32) as usize];
                let e_1 = e_i_1 + r_kb * (e_i1_1 - e_i_1);
                let e_k_top = e_i_k + r_kb * (e_i1_k - e_i_k);
                let e_l_1 = if bin_k == i_kb { e_i_1 } else { e_i1_1 };
                let e_l_k = if bin_k == i_kb { e_i_k } else { e_i1_k };
                let denom_l = e_l_k - e_l_1;
                if denom_l > 0.0 {
                    e_sampled_km = e_1 + (e_sampled_km - e_l_1) * (e_k_top - e_1) / denom_l;
                }
            }
        }

        if e_sampled_km > 0.0 {
            e_out = e_sampled_km;
            e_valid = 1u32;
        }

        // Mu sampling. Two-path: compound (arcsinh) vs precompound (CDF
        // inversion of exp(a*mu)), chosen by xi vs km_r.
        let d_xkmu = crate::common::pcg32::draw_uniform(state);
        state = d_xkmu.state;
        let xi_kmu = d_xkmu.xi;

        let d_xks = crate::common::pcg32::draw_uniform(state);
        state = d_xks.state;
        let xi_ks = d_xks.xi;

        let abs_a = if km_a_sampled < 0.0 {
            -km_a_sampled
        } else {
            km_a_sampled
        };
        let mut mu_km = 1.0 - 2.0 * xi_ks;
        if abs_a >= 1e-6 {
            if xi_kmu > km_r_sampled {
                // Compound: arcsinh form.
                // T = (2*xi - 1) * sinh(a)
                // mu = ln(T + sqrt(T^2 + 1)) / a
                // sinh(a) computed via (exp(a) - exp(-a)) / 2 (no GPU sinh
                // polyfill).
                let exp_a = exp_f64(km_a_sampled);
                let exp_neg_a = exp_f64(-km_a_sampled);
                let sinh_a = 0.5 * (exp_a - exp_neg_a);
                let t = (2.0 * xi_ks - 1.0) * sinh_a;
                mu_km = ln_f64(t + (t * t + 1.0).sqrt()) / km_a_sampled;
            } else {
                // Precompound: CDF inversion of (a * exp(a*mu)) / (2*sinh(a))
                // on [-1, 1].
                // mu = ln(xi*exp(a) + (1-xi)*exp(-a)) / a
                let exp_a = exp_f64(km_a_sampled);
                let exp_neg_a = exp_f64(-km_a_sampled);
                let arg = xi_ks * exp_a + (1.0 - xi_ks) * exp_neg_a;
                if arg > 0.0 {
                    mu_km = ln_f64(arg) / km_a_sampled;
                }
            }
        }
        mu = mu_km.clamp(-1.0, 1.0);
        mu_valid = 1u32;
    }

    KalbachMannSample {
        e_out,
        mu,
        e_valid,
        mu_valid,
        state,
    }
}

/// CPU twin of [`sample_kalbach_mann`]. Same algorithm and same PCG draws
/// (wrapping 64-bit arithmetic), but libm `exp`/`ln` rather than the polyfills,
/// so it matches the `#[cube]` kernel bit-for-bit on `state` + the valid flags
/// and within the polyfills' ULP drift on `e_out` / `mu`. Returns
/// `(e_out, mu, e_valid, mu_valid, state)`.
#[allow(clippy::too_many_arguments)]
pub fn sample_kalbach_mann_cpu(
    energy: f64,
    eg_off_k: u32,
    n_kae: u32,
    km_x_offset: &[u32],
    km_energy_grid: &[f64],
    km_n_x: &[u32],
    km_interp: &[u32],
    km_n_discrete: &[u32],
    km_x: &[f64],
    km_p: &[f64],
    km_c: &[f64],
    km_r: &[f64],
    km_a: &[f64],
    state_in: u64,
) -> (f64, f64, u32, u32, u64) {
    let mut state = state_in;
    let mut e_out = 0.0_f64;
    let mut mu = 0.0_f64;
    let mut e_valid = 0u32;
    let mut mu_valid = 0u32;

    let mut i_kb = 0u32;
    let e_kfirst = km_energy_grid[eg_off_k as usize];
    let e_klast = km_energy_grid[(eg_off_k + n_kae - 1) as usize];
    let mut r_kb = 0.0_f64;
    if energy >= e_klast {
        if n_kae > 1 {
            i_kb = n_kae - 2;
        }
        r_kb = 1.0;
    } else if energy > e_kfirst {
        let mut k = 0u32;
        while k + 1 < n_kae {
            let e_k = km_energy_grid[(eg_off_k + k) as usize];
            let e_k1 = km_energy_grid[(eg_off_k + k + 1) as usize];
            if energy >= e_k && energy < e_k1 {
                i_kb = k;
                let de = e_k1 - e_k;
                if de > 0.0 {
                    r_kb = (energy - e_k) / de;
                }
            }
            k += 1;
        }
    }

    let (bin_k, st) =
        crate::common::sampling::energy_bracket::pick_energy_bracket_cpu(r_kb, i_kb, n_kae, state);
    state = st;

    let n_kx = km_n_x[(eg_off_k + bin_k) as usize];
    if n_kx >= 2 {
        let interp_k = km_interp[(eg_off_k + bin_k) as usize];
        let n_disc = km_n_discrete[(eg_off_k + bin_k) as usize];
        let x_off_k = km_x_offset[(eg_off_k + bin_k) as usize];
        let (xi_kx, st) = crate::common::pcg32::draw_uniform_cpu(state);
        state = st;

        // Discrete-then-continuous CDF inversion (issue #103); bit-twin of the
        // `#[cube]` scan above. See its comment.
        let mut kj = 0u32;
        let mut c_kj = km_c[x_off_k as usize];
        let mut is_discrete = 0u32;
        let mut done = 0u32;
        let mut md = 0u32;
        while md < n_disc {
            if done == 0 {
                kj = md;
                c_kj = km_c[(x_off_k + md) as usize];
                if xi_kx < c_kj {
                    is_discrete = 1u32;
                    done = 1;
                }
            }
            md += 1;
        }
        let mut mk = n_disc;
        while mk + 1 < n_kx {
            if done == 0 {
                kj = mk;
                let c_mk1 = km_c[(x_off_k + mk + 1) as usize];
                if xi_kx <= c_mk1 {
                    done = 1;
                } else {
                    kj = mk + 1;
                    c_kj = c_mk1;
                }
            }
            mk += 1;
        }
        if is_discrete == 0u32 && kj >= n_kx - 1 {
            kj = n_kx - 2;
        }
        let p_kj = km_p[(x_off_k + kj) as usize];
        let x_kj = km_x[(x_off_k + kj) as usize];
        let mut e_sampled_km = x_kj;
        let mut km_r_sampled = km_r[(x_off_k + kj) as usize];
        let mut km_a_sampled = km_a[(x_off_k + kj) as usize];

        if interp_k == 0 || kj < n_disc {
            if p_kj > 0.0 && kj >= n_disc {
                e_sampled_km = x_kj + (xi_kx - c_kj) / p_kj;
            }
        } else {
            let p_kj1 = km_p[(x_off_k + kj + 1) as usize];
            let x_kj1 = km_x[(x_off_k + kj + 1) as usize];
            let de = x_kj1 - x_kj;
            if de > 0.0 {
                let frac = (p_kj1 - p_kj) / de;
                if frac == 0.0 {
                    if p_kj > 0.0 {
                        e_sampled_km = x_kj + (xi_kx - c_kj) / p_kj;
                    }
                } else {
                    let disc_l = p_kj * p_kj + 2.0 * frac * (xi_kx - c_kj);
                    let mut disc_clamped = disc_l;
                    if disc_l <= 0.0 {
                        disc_clamped = 0.0_f64;
                    }
                    e_sampled_km = x_kj + (disc_clamped.sqrt() - p_kj) / frac;
                }
                let f = ((e_sampled_km - x_kj) / de).clamp(0.0, 1.0);
                let r_kj1 = km_r[(x_off_k + kj + 1) as usize];
                let a_kj1 = km_a[(x_off_k + kj + 1) as usize];
                km_r_sampled += f * (r_kj1 - km_r_sampled);
                km_a_sampled += f * (a_kj1 - km_a_sampled);
            }
        }

        if kj >= n_disc && n_kae >= 2 {
            let n_x_i = km_n_x[(eg_off_k + i_kb) as usize];
            let n_x_i1 = km_n_x[(eg_off_k + i_kb + 1) as usize];
            let n_disc_i = km_n_discrete[(eg_off_k + i_kb) as usize];
            let n_disc_i1 = km_n_discrete[(eg_off_k + i_kb + 1) as usize];
            if n_x_i > n_disc_i && n_x_i1 > n_disc_i1 {
                let x_off_i = km_x_offset[(eg_off_k + i_kb) as usize];
                let x_off_i1 = km_x_offset[(eg_off_k + i_kb + 1) as usize];
                let e_i_1 = km_x[(x_off_i + n_disc_i) as usize];
                let e_i_k = km_x[(x_off_i + n_x_i - 1) as usize];
                let e_i1_1 = km_x[(x_off_i1 + n_disc_i1) as usize];
                let e_i1_k = km_x[(x_off_i1 + n_x_i1 - 1) as usize];
                let e_1 = e_i_1 + r_kb * (e_i1_1 - e_i_1);
                let e_k_top = e_i_k + r_kb * (e_i1_k - e_i_k);
                let e_l_1 = if bin_k == i_kb { e_i_1 } else { e_i1_1 };
                let e_l_k = if bin_k == i_kb { e_i_k } else { e_i1_k };
                let denom_l = e_l_k - e_l_1;
                if denom_l > 0.0 {
                    e_sampled_km = e_1 + (e_sampled_km - e_l_1) * (e_k_top - e_1) / denom_l;
                }
            }
        }

        if e_sampled_km > 0.0 {
            e_out = e_sampled_km;
            e_valid = 1;
        }

        let (xi_kmu, st) = crate::common::pcg32::draw_uniform_cpu(state);
        state = st;
        let (xi_ks, st) = crate::common::pcg32::draw_uniform_cpu(state);
        state = st;

        let abs_a = if km_a_sampled < 0.0 {
            -km_a_sampled
        } else {
            km_a_sampled
        };
        let mut mu_km = 1.0 - 2.0 * xi_ks;
        if abs_a >= 1e-6 {
            if xi_kmu > km_r_sampled {
                let exp_a = km_a_sampled.exp();
                let exp_neg_a = (-km_a_sampled).exp();
                let sinh_a = 0.5 * (exp_a - exp_neg_a);
                let t = (2.0 * xi_ks - 1.0) * sinh_a;
                mu_km = (t + (t * t + 1.0).sqrt()).ln() / km_a_sampled;
            } else {
                let exp_a = km_a_sampled.exp();
                let exp_neg_a = (-km_a_sampled).exp();
                let arg = xi_ks * exp_a + (1.0 - xi_ks) * exp_neg_a;
                if arg > 0.0 {
                    mu_km = arg.ln() / km_a_sampled;
                }
            }
        }
        mu = mu_km.clamp(-1.0, 1.0);
        mu_valid = 1;
    }

    (e_out, mu, e_valid, mu_valid, state)
}

/// Test/validation kernel: one sample per thread, using a fixed small layout
/// (`max_km_ae = 4`, `max_km_x = 8`) so the buffers stay tiny.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn km_test_kernel(
    energy_in: &[f64],
    eg_off_k: &[u32],
    n_kae: &[u32],
    km_x_offset: &[u32],
    km_energy_grid: &[f64],
    km_n_x: &[u32],
    km_interp: &[u32],
    km_n_discrete: &[u32],
    km_x: &[f64],
    km_p: &[f64],
    km_c: &[f64],
    km_r: &[f64],
    km_a: &[f64],
    seeds: &[u32],
    out_e: &mut [f64],
    out_mu: &mut [f64],
    out_evalid: &mut [u32],
    out_muvalid: &mut [u32],
    out_state: &mut [u64],
) {
    if ABSOLUTE_POS >= out_e.len() {
        terminate!();
    }
    let r = sample_kalbach_mann(
        energy_in[ABSOLUTE_POS],
        eg_off_k[ABSOLUTE_POS],
        n_kae[ABSOLUTE_POS],
        km_x_offset,
        km_energy_grid,
        km_n_x,
        km_interp,
        km_n_discrete,
        km_x,
        km_p,
        km_c,
        km_r,
        km_a,
        expand_seed(seeds[ABSOLUTE_POS]),
    );
    out_e[ABSOLUTE_POS] = r.e_out;
    out_mu[ABSOLUTE_POS] = r.mu;
    out_evalid[ABSOLUTE_POS] = r.e_valid;
    out_muvalid[ABSOLUTE_POS] = r.mu_valid;
    out_state[ABSOLUTE_POS] = r.state;
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neutron::xs::constants::ANGLE_INTERP_LINLIN;
    use crate::{GpuContext, GpuInitError, WgpuRuntime};

    // Fixed test layout (kept tiny; the kernel hardcodes max_km_x = 8).
    const AE: usize = 4; // max_km_ae for the test
    const MX: usize = 8; // max_km_x for the test

    /// Build a 2-slot KM fixture. Slot 0: two incident energies, each a
    /// 4-point LinLin continuous spectrum with non-trivial (r, a) params (so
    /// both mu branches are exercised). Slot 1: a single incident energy with
    /// a 3-point histogram spectrum plus a leading discrete line.
    #[allow(clippy::type_complexity)]
    fn fixture() -> (
        Vec<f64>, // energy_grid
        Vec<u32>, // n_x
        Vec<u32>, // interp
        Vec<u32>, // n_disc
        Vec<f64>, // x
        Vec<f64>, // p
        Vec<f64>, // c
        Vec<f64>, // r
        Vec<f64>, // a
    ) {
        let n_slots = 2usize;
        let mut energy_grid = vec![0.0_f64; n_slots * AE];
        let mut n_x = vec![0u32; n_slots * AE];
        let mut interp = vec![0u32; n_slots * AE];
        let mut n_disc = vec![0u32; n_slots * AE];
        let mut x = vec![0.0_f64; n_slots * AE * MX];
        let mut p = vec![0.0_f64; n_slots * AE * MX];
        let mut c = vec![0.0_f64; n_slots * AE * MX];
        let mut r = vec![0.0_f64; n_slots * AE * MX];
        let mut a = vec![0.0_f64; n_slots * AE * MX];

        // Slot 0, incident-energy indices 0 and 1: LinLin continuous.
        energy_grid[0] = 1.0e6;
        energy_grid[1] = 2.0e6;
        for (ae, scale) in [(0usize, 1.0_f64), (1usize, 1.4_f64)] {
            n_x[ae] = 4;
            interp[ae] = ANGLE_INTERP_LINLIN;
            let off = ae * MX;
            let xs = [0.1e6 * scale, 0.4e6 * scale, 0.8e6 * scale, 1.0e6 * scale];
            let cs = [0.0, 0.35, 0.8, 1.0];
            x[off..off + 4].copy_from_slice(&xs);
            c[off..off + 4].copy_from_slice(&cs);
            for k in 0..3 {
                let dxk = xs[k + 1] - xs[k];
                p[off + k] = if dxk > 0.0 {
                    (cs[k + 1] - cs[k]) / dxk
                } else {
                    0.0
                };
            }
            p[off + 3] = p[off + 2];
            // Kalbach (r, a): r in (0,1), a a few-tenths slope so both compound
            // and precompound branches fire and |a| >= 1e-6.
            let rs = [0.2, 0.4, 0.6, 0.8];
            let as_ = [0.5 * scale, 0.8 * scale, 1.2 * scale, 1.5 * scale];
            r[off..off + 4].copy_from_slice(&rs);
            a[off..off + 4].copy_from_slice(&as_);
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
        c[off..off + 3].copy_from_slice(&cs);
        p[off] = 0.0; // discrete
        p[off + 1] = (cs[2] - cs[1]) / (xs[2] - xs[1]);
        p[off + 2] = p[off + 1];
        let rs = [0.3, 0.5, 0.5];
        let as_ = [0.0, 0.9, 0.9]; // first a = 0 (isotropic), rest non-trivial
        r[off..off + 3].copy_from_slice(&rs);
        a[off..off + 3].copy_from_slice(&as_);

        (energy_grid, n_x, interp, n_disc, x, p, c, r, a)
    }

    /// GPU Kalbach-Mann sampling must match the CPU twin: the PCG state
    /// advances bit-for-bit (pure u64 integer math), the `e_valid` / `mu_valid`
    /// flags are pure integer outputs and must match exactly, and the sampled
    /// energy + mu agree to within the polyfills' drift (the mu formula flows
    /// through `exp`/`ln` and a `sqrt`; the E_out LinLin inversion through one
    /// `sqrt`). Sweeps incident energies below / inside / above the grid, both
    /// slots, and many seeds (covering both mu branches via `xi_kmu` vs `r`).
    #[test]
    fn gpu_kalbach_mann_eout_mu_matches_cpu() {
        let ctx = match GpuContext::new() {
            Ok(c) => c,
            Err(GpuInitError::NoF64Adapter) => {
                println!("no Vulkan f64 adapter -- skipping");
                return;
            }
        };

        let (energy_grid, n_x, interp, n_disc, x, p, c, r, a) = fixture();
        // CSR x-offsets for the padded fixture: row `r` (`slot*AE + ae`) starts
        // at `r * MX` (issue #104 -- the sampler now takes per-row offsets
        // rather than a `max_km_x` stride).
        let x_offset: Vec<u32> = (0..n_x.len()).map(|row| (row * MX) as u32).collect();

        // (eg_off_k, n_kae, incident energies) per case. Slot 0 (eg_off=0,
        // n_kae=2) swept around [1,2] MeV; slot 1 (eg_off=AE, n_kae=1) around
        // 0.5 MeV. Each (slot, energy) crossed with many seeds.
        let cases: &[(u32, u32, &[f64])] = &[
            (0, 2, &[0.5e6, 1.0e6, 1.5e6, 2.0e6, 3.0e6]),
            (AE as u32, 1, &[0.5e6]),
        ];

        let mut energy_in = Vec::new();
        let mut eg_off = Vec::new();
        let mut n_kae = Vec::new();
        let mut seeds = Vec::new();
        let n_seeds = 200u32;
        for &(off, ne, energies) in cases {
            for &e in energies {
                for s in 0..n_seeds {
                    energy_in.push(e);
                    eg_off.push(off);
                    n_kae.push(ne);
                    seeds.push((off.wrapping_mul(7919) + s).wrapping_mul(2_654_435_761));
                }
            }
        }
        let n = seeds.len();

        // CPU reference.
        let mut cpu_e = vec![0.0_f64; n];
        let mut cpu_mu = vec![0.0_f64; n];
        let mut cpu_ev = vec![0u32; n];
        let mut cpu_mv = vec![0u32; n];
        let mut cpu_state = vec![0u64; n];
        for i in 0..n {
            let (e, m, ev, mv, st) = sample_kalbach_mann_cpu(
                energy_in[i],
                eg_off[i],
                n_kae[i],
                &x_offset,
                &energy_grid,
                &n_x,
                &interp,
                &n_disc,
                &x,
                &p,
                &c,
                &r,
                &a,
                crate::common::rng::expand_seed(seeds[i]),
            );
            cpu_e[i] = e;
            cpu_mu[i] = m;
            cpu_ev[i] = ev;
            cpu_mv[i] = mv;
            cpu_state[i] = st;
        }

        // GPU run.
        let client = ctx.client();
        let ein_h = client.create_from_slice(bytemuck::cast_slice(&energy_in));
        let off_h = client.create_from_slice(bytemuck::cast_slice(&eg_off));
        let nk_h = client.create_from_slice(bytemuck::cast_slice(&n_kae));
        let xoff_h = client.create_from_slice(bytemuck::cast_slice(&x_offset));
        let eg_h = client.create_from_slice(bytemuck::cast_slice(&energy_grid));
        let nx_h = client.create_from_slice(bytemuck::cast_slice(&n_x));
        let interp_h = client.create_from_slice(bytemuck::cast_slice(&interp));
        let nd_h = client.create_from_slice(bytemuck::cast_slice(&n_disc));
        let x_h = client.create_from_slice(bytemuck::cast_slice(&x));
        let p_h = client.create_from_slice(bytemuck::cast_slice(&p));
        let c_h = client.create_from_slice(bytemuck::cast_slice(&c));
        let r_h = client.create_from_slice(bytemuck::cast_slice(&r));
        let a_h = client.create_from_slice(bytemuck::cast_slice(&a));
        let seed_h = client.create_from_slice(bytemuck::cast_slice(&seeds));
        let oute_h = client.empty(n * core::mem::size_of::<f64>());
        let outmu_h = client.empty(n * core::mem::size_of::<f64>());
        let outev_h = client.empty(n * core::mem::size_of::<u32>());
        let outmv_h = client.empty(n * core::mem::size_of::<u32>());
        let outs_h = client.empty(n * core::mem::size_of::<u64>());

        const WG: u32 = 64;
        let groups = (n as u32).div_ceil(WG);
        unsafe {
            km_test_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(groups, 1, 1),
                CubeDim::new_1d(WG),
                BufferArg::from_raw_parts(ein_h, energy_in.len()),
                BufferArg::from_raw_parts(off_h, eg_off.len()),
                BufferArg::from_raw_parts(nk_h, n_kae.len()),
                BufferArg::from_raw_parts(xoff_h, x_offset.len()),
                BufferArg::from_raw_parts(eg_h, energy_grid.len()),
                BufferArg::from_raw_parts(nx_h, n_x.len()),
                BufferArg::from_raw_parts(interp_h, interp.len()),
                BufferArg::from_raw_parts(nd_h, n_disc.len()),
                BufferArg::from_raw_parts(x_h, x.len()),
                BufferArg::from_raw_parts(p_h, p.len()),
                BufferArg::from_raw_parts(c_h, c.len()),
                BufferArg::from_raw_parts(r_h, r.len()),
                BufferArg::from_raw_parts(a_h, a.len()),
                BufferArg::from_raw_parts(seed_h, seeds.len()),
                BufferArg::from_raw_parts(oute_h.clone(), n),
                BufferArg::from_raw_parts(outmu_h.clone(), n),
                BufferArg::from_raw_parts(outev_h.clone(), n),
                BufferArg::from_raw_parts(outmv_h.clone(), n),
                BufferArg::from_raw_parts(outs_h.clone(), n),
            );
        }
        let gpu_e = bytemuck::cast_slice::<u8, f64>(&client.read_one(oute_h).unwrap()).to_vec();
        let gpu_mu = bytemuck::cast_slice::<u8, f64>(&client.read_one(outmu_h).unwrap()).to_vec();
        let gpu_ev = bytemuck::cast_slice::<u8, u32>(&client.read_one(outev_h).unwrap()).to_vec();
        let gpu_mv = bytemuck::cast_slice::<u8, u32>(&client.read_one(outmv_h).unwrap()).to_vec();
        let gpu_state = bytemuck::cast_slice::<u8, u64>(&client.read_one(outs_h).unwrap()).to_vec();

        for i in 0..n {
            assert_eq!(
                gpu_state[i], cpu_state[i],
                "sample {i}: PCG state diverged (gpu {} cpu {}, energy {}, off {}, seed {})",
                gpu_state[i], cpu_state[i], energy_in[i], eg_off[i], seeds[i]
            );
            assert_eq!(
                gpu_ev[i], cpu_ev[i],
                "sample {i}: e_valid diverged (gpu {} cpu {}, off {}, seed {})",
                gpu_ev[i], cpu_ev[i], eg_off[i], seeds[i]
            );
            assert_eq!(
                gpu_mv[i], cpu_mv[i],
                "sample {i}: mu_valid diverged (gpu {} cpu {}, off {}, seed {})",
                gpu_mv[i], cpu_mv[i], eg_off[i], seeds[i]
            );
            if cpu_ev[i] == 1 {
                let tol = 1e-9 * cpu_e[i].abs().max(1.0);
                assert!(
                    (gpu_e[i] - cpu_e[i]).abs() <= tol,
                    "sample {i}: energy diverged (gpu {} cpu {}, tol {tol}, energy {}, off {}, seed {})",
                    gpu_e[i], cpu_e[i], energy_in[i], eg_off[i], seeds[i]
                );
            }
            if cpu_mv[i] == 1 {
                assert!(
                    (gpu_mu[i] - cpu_mu[i]).abs() <= 1e-9,
                    "sample {i}: mu diverged (gpu {} cpu {}, energy {}, off {}, seed {})",
                    gpu_mu[i],
                    cpu_mu[i],
                    energy_in[i],
                    eg_off[i],
                    seeds[i]
                );
                assert!(
                    gpu_mu[i] >= -1.0 && gpu_mu[i] <= 1.0,
                    "sample {i}: mu out of [-1,1]: {}",
                    gpu_mu[i]
                );
            }
            // Slot 0 (n_kae=2, n_x>=2) must be valid for both e and mu.
            if eg_off[i] == 0 {
                assert_eq!(cpu_mv[i], 1, "sample {i}: slot 0 mu not valid");
            }
        }
        println!(
            "kalbach-mann: {} samples, GPU state+e_valid+mu_valid bit-exact, e within 1e-9 rel, mu within 1e-9 abs",
            n
        );
    }
}
