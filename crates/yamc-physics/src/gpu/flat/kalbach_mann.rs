//! Kalbach-Mann correlated (E_out, μ) sampler.
//!
//! Implements the ENDF Kalbach-Mann angular-energy
//! representation: a tabulated per-incident-energy distribution
//! that couples outgoing energy and angle through the
//! `r(E_out)`, `a(E_out)` parameters. The angle is sampled from
//! either the compound-nucleus (arcsinh) or precompound
//! (`a·exp(a·μ)`) branch, weighted by `r(E_out)`.
//!
//! Bit-identical to the cubecl kernel's KM branch. Touches four
//! delicate code paths that historically caused CPU/GPU drift
//! (see the `cubecl_loop_doubling` memory note):
//!
//! 1. **Stochastic incident-energy bracket pick** -- `bin_k = i_kb` or
//!    `i_kb + 1` with probability `r_kb`.
//! 2. **Discrete + continuous CDF inversion** -- the chosen
//!    bracket's `(x, p, c)` table holds `n_discrete` discrete lines
//!    followed by the continuous tail; the energy-sampling branch
//!    differs between the two regions, and the `(r, a)` parameters
//!    are linearly interpolated using the pre-stretch sampled
//!    energy on the continuous branch.
//! 3. **Bracket-bound stretch** (continuous portion only) -- stretch
//!    the sampled E_out from the chosen bracket's continuous-tail
//!    endpoints to the interpolated endpoints at the actual
//!    incident energy.
//! 4. **Compound vs precompound μ branch** -- `xi_kmu > r(E_out)` →
//!    compound (arcsinh-of-`sinh(a)`), else precompound (CDF
//!    inversion of `a·exp(a·μ)`). Both consume the same `xi_ks`
//!    draw, so the RNG schedule is identical regardless of branch.

use super::grid::locate_bracket;
use yamc_rng::next_xi;

/// Interpolation flag for the per-bracket `(x, p, c)` table -- `0`
/// selects histogram interpolation (linear-in-`c` energy sample),
/// `1` selects linear-linear (quadratic-in-`c` sample with linearly
/// interpolated `r`, `a` parameters). Mirrors the kernel's u32 codes.
pub const KM_INTERP_HISTOGRAM: u32 = 0;
pub const KM_INTERP_LINLIN: u32 = 1;

/// Sample an outgoing CM energy and μ from a Kalbach-Mann
/// distribution.
///
/// # Arguments
/// * `e_in` -- incident energy.
/// * `energy_grid` -- sorted per-slot incident-energy grid, length `n_e`.
/// * `n_x_per_i` -- number of `(x, p, c, r, a)` points at each
///   incident-energy bracket, length `n_e`.
/// * `interp_per_i` -- interpolation flag per incident-energy point,
///   length `n_e` (`KM_INTERP_HISTOGRAM` or `KM_INTERP_LINLIN`).
/// * `n_discrete_per_i` -- number of discrete points at each
///   incident-energy bracket (followed by `n_x - n_discrete`
///   continuous points), length `n_e`.
/// * `x_table`, `p_table`, `c_table`, `r_table`, `a_table` -- flat
///   tables packed variable-length (issue #104): row `i` occupies
///   `x_offset[i] .. x_offset[i] + n_x_per_i[i]`. Rows are stored
///   back-to-back with no padding.
/// * `x_offset` -- start index of each incident-energy row in the flat
///   `x_table` / `p_table` / `c_table` / `r_table` / `a_table`, length
///   `n_e`.
/// * `state` -- inline PCG RNG state, advanced 4× on the happy path:
///   `xi_keb` (bracket pick), `xi_kx` (E_out CDF inversion),
///   `xi_kmu` (compound/precompound branch), `xi_ks` (μ sample).
///
/// # Returns
/// `Some((e_cm, mu_cm))` on success with `e_cm ≥ 0` and `mu_cm`
/// clamped to `[-1, 1]`; `None` when the slot is empty
/// (`n_e == 0`) or the chosen bracket has fewer than 2 (x, c)
/// points.
#[allow(clippy::too_many_arguments)]
pub fn sample_kalbach_mann(
    e_in: f64,
    energy_grid: &[f64],
    n_x_per_i: &[u32],
    interp_per_i: &[u32],
    n_discrete_per_i: &[u32],
    x_table: &[f64],
    p_table: &[f64],
    c_table: &[f64],
    r_table: &[f64],
    a_table: &[f64],
    x_offset: &[u32],
    state: &mut u64,
) -> Option<(f64, f64)> {
    let n_e = energy_grid.len();
    if n_e == 0 {
        return None;
    }
    let (i_kb, r_kb) = locate_bracket(e_in, energy_grid);

    let xi_keb = next_xi(state);
    let mut bin_k = i_kb;
    if r_kb > xi_keb && bin_k + 1 < n_e {
        bin_k = i_kb + 1;
    }

    let n_kx = n_x_per_i[bin_k] as usize;
    if n_kx < 2 {
        return None;
    }
    let interp_k = interp_per_i[bin_k];
    let n_disc = n_discrete_per_i[bin_k] as usize;
    let x_off_k = x_offset[bin_k] as usize;

    let xi_kx = next_xi(state);

    // Discrete-then-continuous CDF search, matching the CPU reference
    // `sample_with_discrete_info` (issue #103). The discrete head (first
    // `n_disc` points) is searched with `xi < c[k]` and selects the exact
    // discrete line; the continuous tail is searched from `n_disc` with
    // `xi <= c[k+1]` ("first match wins, else last"), carrying `c_kj` as the
    // CPU does. The previous single unified scan collapsed every discrete line
    // onto index 0. For `n_disc == 0` this is identical to the old scan.
    let mut kj = 0usize;
    let mut c_kj = c_table[x_off_k];
    let mut is_discrete = false;
    let mut done = false;
    for k in 0..n_disc {
        if !done {
            kj = k;
            c_kj = c_table[x_off_k + k];
            if xi_kx < c_kj {
                is_discrete = true;
                done = true;
            }
        }
    }
    for k in n_disc..n_kx - 1 {
        if !done {
            kj = k;
            let c_k1 = c_table[x_off_k + k + 1];
            if xi_kx <= c_k1 {
                done = true;
            } else {
                kj = k + 1;
                c_kj = c_k1;
            }
        }
    }
    if !is_discrete && kj >= n_kx - 1 {
        kj = n_kx - 2;
    }
    let p_kj = p_table[x_off_k + kj];
    let x_kj = x_table[x_off_k + kj];
    let mut e_sampled = x_kj;
    let mut km_r_sampled = r_table[x_off_k + kj];
    let mut km_a_sampled = a_table[x_off_k + kj];

    if interp_k == KM_INTERP_HISTOGRAM || kj < n_disc {
        // Histogram OR within the discrete region. `r`, `a` stay at
        // their kj-th values; energy uses linear-in-c on continuous,
        // delta on discrete.
        if p_kj > 0.0 && kj >= n_disc {
            e_sampled = x_kj + (xi_kx - c_kj) / p_kj;
        }
    } else {
        // Linear interp on the continuous tail. Quadratic energy
        // sample plus linear interp of `r`, `a` on the actual
        // pre-stretch sampled energy.
        let p_kj1 = p_table[x_off_k + kj + 1];
        let x_kj1 = x_table[x_off_k + kj + 1];
        let de = x_kj1 - x_kj;
        if de > 0.0 {
            let frac = (p_kj1 - p_kj) / de;
            if frac == 0.0 {
                if p_kj > 0.0 {
                    e_sampled = x_kj + (xi_kx - c_kj) / p_kj;
                }
            } else {
                let disc = (p_kj * p_kj + 2.0 * frac * (xi_kx - c_kj)).max(0.0);
                e_sampled = x_kj + (disc.sqrt() - p_kj) / frac;
            }
            let f = ((e_sampled - x_kj) / de).clamp(0.0, 1.0);
            let r_kj1 = r_table[x_off_k + kj + 1];
            let a_kj1 = a_table[x_off_k + kj + 1];
            km_r_sampled += f * (r_kj1 - km_r_sampled);
            km_a_sampled += f * (a_kj1 - km_a_sampled);
        }
    }

    // Bracket-bound stretch (continuous portion only). Need both
    // adjacent brackets to carry a non-empty continuous tail
    // (`n_x > n_discrete`); index `i_kb + 1` is safe because
    // `r_kb == 1.0` is only reached when `n_e > 1` and
    // `i_kb == n_e - 2`.
    if kj >= n_disc && n_e >= 2 {
        let n_x_i = n_x_per_i[i_kb] as usize;
        let n_x_i1 = n_x_per_i[i_kb + 1] as usize;
        let n_disc_i = n_discrete_per_i[i_kb] as usize;
        let n_disc_i1 = n_discrete_per_i[i_kb + 1] as usize;
        if n_x_i > n_disc_i && n_x_i1 > n_disc_i1 {
            let x_off_i = x_offset[i_kb] as usize;
            let x_off_i1 = x_offset[i_kb + 1] as usize;
            let e_i_1 = x_table[x_off_i + n_disc_i];
            let e_i_k = x_table[x_off_i + n_x_i - 1];
            let e_i1_1 = x_table[x_off_i1 + n_disc_i1];
            let e_i1_k = x_table[x_off_i1 + n_x_i1 - 1];
            let e_1 = e_i_1 + r_kb * (e_i1_1 - e_i_1);
            let e_k_top = e_i_k + r_kb * (e_i1_k - e_i_k);
            let e_l_1 = if bin_k == i_kb { e_i_1 } else { e_i1_1 };
            let e_l_k = if bin_k == i_kb { e_i_k } else { e_i1_k };
            let denom_l = e_l_k - e_l_1;
            if denom_l > 0.0 {
                e_sampled = e_1 + (e_sampled - e_l_1) * (e_k_top - e_1) / denom_l;
            }
        }
    }

    // μ sample: compound (arcsinh) vs precompound (CDF inversion).
    // Both consume `xi_ks`, so the RNG schedule is identical across
    // branches -- the `if abs_a >= 1e-6` short-circuit keeps the
    // isotropic-fallback path using the SAME `xi_ks` value, just
    // unmapped through `1 - 2·ξ` instead of through the Kalbach
    // angle distribution.
    let xi_kmu = next_xi(state);
    let xi_ks = next_xi(state);

    let abs_a = km_a_sampled.abs();
    let mut mu_km = 1.0 - 2.0 * xi_ks;
    if abs_a >= 1e-6 {
        if xi_kmu > km_r_sampled {
            // Compound: arcsinh form. μ = asinh((2ξ−1)·sinh(a))/a.
            let sinh_a = km_a_sampled.sinh();
            let t = (2.0 * xi_ks - 1.0) * sinh_a;
            mu_km = (t + (t * t + 1.0).sqrt()).ln() / km_a_sampled;
        } else {
            // Precompound: CDF inversion of (a · exp(a·μ))/(2·sinh(a)).
            let arg = xi_ks * km_a_sampled.exp() + (1.0 - xi_ks) * (-km_a_sampled).exp();
            if arg > 0.0 {
                mu_km = arg.ln() / km_a_sampled;
            }
        }
    }
    Some((e_sampled.max(0.0), mu_km.clamp(-1.0, 1.0)))
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty grid → `None`, no RNG advance.
    #[test]
    fn kalbach_mann_returns_none_for_empty_slot() {
        let empty_f: [f64; 0] = [];
        let empty_u: [u32; 0] = [];
        let mut state: u64 = 0xC0FFEE;
        let before = state;
        let r = sample_kalbach_mann(
            5.0e6, &empty_f, &empty_u, &empty_u, &empty_u, &empty_f, &empty_f, &empty_f, &empty_f,
            &empty_f, &empty_u, &mut state,
        );
        assert!(r.is_none());
        assert_eq!(state, before, "empty slot must not advance RNG state");
    }

    /// Chosen bracket with `n_x < 2` → `None`. The bracket-pick
    /// draw is consumed before the early return, so state advances
    /// exactly once.
    #[test]
    fn kalbach_mann_returns_none_when_chosen_bracket_too_narrow() {
        let energy_grid = [1.0e3, 1.0e7];
        let n_x_per_i = [4u32, 1u32];
        let interp_per_i = [KM_INTERP_LINLIN, KM_INTERP_LINLIN];
        let n_discrete_per_i = [0u32, 0u32];
        let max_x: usize = 8;
        let x_table = vec![0.0; 2 * max_x];
        let p_table = vec![0.0; 2 * max_x];
        let c_table = vec![0.0; 2 * max_x];
        let r_table = vec![0.0; 2 * max_x];
        let a_table = vec![0.0; 2 * max_x];
        // E_in above e_last → i_kb = 0, r_kb = 1.0 → bin_k = 1 (degenerate).
        let x_off = [0u32, max_x as u32];
        let mut state: u64 = 0xDEADBEEF;
        let result = sample_kalbach_mann(
            2.0e7,
            &energy_grid,
            &n_x_per_i,
            &interp_per_i,
            &n_discrete_per_i,
            &x_table,
            &p_table,
            &c_table,
            &r_table,
            &a_table,
            &x_off,
            &mut state,
        );
        assert!(result.is_none());
    }

    /// Return type of [`build_continuous_km`]: (energy_grid, n_x_per_i,
    /// n_discrete_per_i, interp_per_i, x, p, c, r, a tables, max_x).
    type ContinuousKmFixture = (
        [f64; 2],
        [u32; 2],
        [u32; 2],
        [u32; 2],
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        usize,
    );

    /// Build a 2-bracket synthetic KM distribution where each
    /// bracket has 4 continuous (x, p, c) points uniformly
    /// distributed over `[1e5, 2e6]` / `[2e5, 3e6]`. At the
    /// mid-incident-energy the stretched bounds are
    /// `[1.5e5, 2.5e6]`. Drive 4k samples and verify the sampled
    /// `e_cm` always lands within those bounds, with mean near
    /// the midpoint.
    fn build_continuous_km() -> ContinuousKmFixture {
        let energy_grid = [1.0e3, 1.0e7];
        let n_x_per_i = [4u32, 4u32];
        let interp_per_i = [KM_INTERP_LINLIN, KM_INTERP_LINLIN];
        let n_discrete_per_i = [0u32, 0u32];
        let max_x: usize = 8;
        let mut x_table = vec![0.0; 2 * max_x];
        let mut p_table = vec![0.0; 2 * max_x];
        let mut c_table = vec![0.0; 2 * max_x];
        let mut r_table = vec![0.0; 2 * max_x];
        let mut a_table = vec![0.0; 2 * max_x];
        // Bracket 0: x = [1e5, 7e5, 1.4e6, 2e6]; CDF uniform.
        // p_table chosen so the linear-in-c form `e = x + (ξ - c)/p`
        // gives a uniform draw across the bracket.
        let xs0 = [1.0e5, 7.0e5, 1.4e6, 2.0e6];
        let cs0 = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
        // Bracket 1: x = [2e5, 8e5, 2.2e6, 3e6]
        let xs1 = [2.0e5, 8.0e5, 2.2e6, 3.0e6];
        let cs1 = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
        // r = 0.5, a = 0 in both → isotropic μ; we only care about
        // the energy mean and bounds here.
        for j in 0..4 {
            x_table[j] = xs0[j];
            c_table[j] = cs0[j];
            x_table[max_x + j] = xs1[j];
            c_table[max_x + j] = cs1[j];
            r_table[j] = 0.5;
            r_table[max_x + j] = 0.5;
            a_table[j] = 0.0;
            a_table[max_x + j] = 0.0;
        }
        // p[j] = (c[j+1] - c[j]) / (x[j+1] - x[j]); set p[last] = 0
        // (sentinel -- the kernel doesn't sample past kj = n_x - 2).
        for j in 0..3 {
            let dc = cs0[j + 1] - cs0[j];
            let dx = xs0[j + 1] - xs0[j];
            p_table[j] = dc / dx;
            let dc1 = cs1[j + 1] - cs1[j];
            let dx1 = xs1[j + 1] - xs1[j];
            p_table[max_x + j] = dc1 / dx1;
        }
        (
            energy_grid,
            n_x_per_i,
            interp_per_i,
            n_discrete_per_i,
            x_table,
            p_table,
            c_table,
            r_table,
            a_table,
            max_x,
        )
    }

    #[test]
    fn kalbach_mann_samples_within_stretched_bounds() {
        let (
            energy_grid,
            n_x_per_i,
            interp_per_i,
            n_discrete_per_i,
            x_t,
            p_t,
            c_t,
            r_t,
            a_t,
            max_x,
        ) = build_continuous_km();
        // r ≈ 0.5 in both brackets at e_in = 5e6 → stretched bounds
        //   e_1 = 1.5e5, e_k = 2.5e6
        let e_in = 5.0005e6;
        let lo = 1.5e5;
        let hi = 2.5e6;
        let x_off = [0u32, max_x as u32];
        let mut state: u64 = 0x12345678;
        let n = 4_000usize;
        let mut sum = 0.0;
        for _ in 0..n {
            let (e_out, mu) = sample_kalbach_mann(
                e_in,
                &energy_grid,
                &n_x_per_i,
                &interp_per_i,
                &n_discrete_per_i,
                &x_t,
                &p_t,
                &c_t,
                &r_t,
                &a_t,
                &x_off,
                &mut state,
            )
            .expect("happy path should always return Some");
            assert!(
                (lo..=hi).contains(&e_out),
                "e_out {e_out} fell outside stretched bounds [{lo}, {hi}]"
            );
            assert!((-1.0..=1.0).contains(&mu), "μ {mu} out of range");
            sum += e_out;
        }
        let mean = sum / n as f64;
        let mid = 0.5 * (lo + hi);
        let rel_err = ((mean - mid) / mid).abs();
        assert!(
            rel_err < 0.10,
            "mean {mean:.4e} differs from midpoint {mid:.4e} by {rel_err:.3} (>10%)"
        );
    }

    /// With `a = 0` in both brackets the μ sample collapses to the
    /// isotropic fallback `1 − 2·ξ_ks`; the mean over many draws
    /// must sit near 0.
    #[test]
    fn kalbach_mann_isotropic_mu_when_a_is_zero() {
        let (
            energy_grid,
            n_x_per_i,
            interp_per_i,
            n_discrete_per_i,
            x_t,
            p_t,
            c_t,
            r_t,
            a_t,
            max_x,
        ) = build_continuous_km();
        let x_off = [0u32, max_x as u32];
        let mut state: u64 = 0xC1A551C5;
        let mut sum = 0.0;
        let n = 4_000usize;
        for _ in 0..n {
            let (_e, mu) = sample_kalbach_mann(
                5.0e6,
                &energy_grid,
                &n_x_per_i,
                &interp_per_i,
                &n_discrete_per_i,
                &x_t,
                &p_t,
                &c_t,
                &r_t,
                &a_t,
                &x_off,
                &mut state,
            )
            .unwrap();
            sum += mu;
        }
        let mean = sum / n as f64;
        assert!(
            mean.abs() < 0.05,
            "isotropic μ mean {mean} should be ≈ 0, got |μ̄| ≥ 0.05"
        );
    }

    /// With `a` strongly positive and `r = 1.0` (precompound only),
    /// μ is biased forward -- mean should be positive.
    #[test]
    fn kalbach_mann_forward_peaked_when_a_positive_precompound() {
        let (
            energy_grid,
            n_x_per_i,
            interp_per_i,
            n_discrete_per_i,
            x_t,
            p_t,
            c_t,
            mut r_t,
            mut a_t,
            max_x,
        ) = build_continuous_km();
        // r = 1.0 → ξ_kmu ≤ r always → precompound branch.
        // a = 2.0 → exp(2·μ) heavily favours μ ≈ 1.
        for v in r_t.iter_mut() {
            *v = 1.0;
        }
        for v in a_t.iter_mut() {
            *v = 2.0;
        }
        let x_off = [0u32, max_x as u32];
        let mut state: u64 = 0xFADEFADE;
        let mut sum = 0.0;
        let n = 4_000usize;
        for _ in 0..n {
            let (_e, mu) = sample_kalbach_mann(
                5.0e6,
                &energy_grid,
                &n_x_per_i,
                &interp_per_i,
                &n_discrete_per_i,
                &x_t,
                &p_t,
                &c_t,
                &r_t,
                &a_t,
                &x_off,
                &mut state,
            )
            .unwrap();
            sum += mu;
        }
        let mean = sum / n as f64;
        // ⟨μ⟩ = coth(a) − 1/a for the precompound a·exp(a·μ)/(2·sinh(a))
        // distribution. At a = 2: coth(2) ≈ 1.0373, 1/a = 0.5 → ⟨μ⟩ ≈ 0.537.
        assert!(
            mean > 0.45,
            "forward-peaked KM should give ⟨μ⟩ > 0.45 (theory ≈ 0.537), got {mean}"
        );
    }

    /// Flat-vs-production-CPU parity (issue #101/#108). The flat
    /// `sample_kalbach_mann` must reproduce the production
    /// `yamc_nuclide::secondary_kalbach::KalbachMann::sample` distribution
    /// (outgoing energy AND Kalbach angle) over a shared fixture. The two use
    /// different RNGs / draw schedules, so this is a statistical comparison
    /// (like the secondary-sampler parity in PR #110). A 3-bin continuous
    /// Histogram table with constant Kalbach `r`/`a` (so the angular law is a
    /// fixed Kalbach(r, a) regardless of the sampled bin) sampled at the lower
    /// incident-energy knot (`r_interp = 0`, no bracket stretch) isolates the
    /// CDF inversion + the compound/precompound angular branch.
    #[test]
    fn flat_kalbach_matches_production_distribution() {
        use rand::{rngs::StdRng, SeedableRng};
        use yamc_nuclide::secondary_kalbach::{Interpolation, KMTable, KalbachMann};

        // 4 e_out points -> 3 equal-probability continuous bins; constant
        // Kalbach r = 0.3, a = 1.0.
        let e_out = vec![1.0e6, 4.0e6, 7.0e6, 1.0e7];
        let c = vec![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
        let mut p = vec![0.0; 4];
        for k in 0..3 {
            p[k] = (c[k + 1] - c[k]) / (e_out[k + 1] - e_out[k]);
        }
        let r = vec![0.3_f64; 4];
        let a = vec![1.0_f64; 4];
        let table = KMTable {
            interpolation: Interpolation::Histogram,
            n_discrete: 0,
            e_out: e_out.clone(),
            p: p.clone(),
            c: c.clone(),
            r: r.clone(),
            a: a.clone(),
        };
        // Two identical incident-energy points; sampling at the lower knot
        // gives r_interp = 0 (always pick bracket 0, no stretch).
        let km = KalbachMann {
            energy: vec![1.0e3, 1.0e7],
            distributions: vec![table.clone(), table],
        };
        let e_in = 1.0e3_f64;

        const N: usize = 400_000;
        let mut rng = StdRng::seed_from_u64(0x101_0ABC);
        let prod: Vec<(f64, f64)> = (0..N).map(|_| km.sample(e_in, &mut rng)).collect();

        // Flat buffers from the same data.
        let max_x = 8usize;
        let energy_grid = [1.0e3, 1.0e7];
        let n_x_per_i = [4u32, 4u32];
        let interp_per_i = [KM_INTERP_HISTOGRAM, KM_INTERP_HISTOGRAM];
        let n_discrete_per_i = [0u32, 0u32];
        let mut x_t = vec![0.0; 2 * max_x];
        let mut p_t = vec![0.0; 2 * max_x];
        let mut c_t = vec![0.0; 2 * max_x];
        let mut r_t = vec![0.0; 2 * max_x];
        let mut a_t = vec![0.0; 2 * max_x];
        for row in 0..2 {
            let off = row * max_x;
            x_t[off..off + 4].copy_from_slice(&e_out);
            p_t[off..off + 4].copy_from_slice(&p);
            c_t[off..off + 4].copy_from_slice(&c);
            r_t[off..off + 4].copy_from_slice(&r);
            a_t[off..off + 4].copy_from_slice(&a);
        }
        let x_off = [0u32, max_x as u32];
        let mut state: u64 = 0x0101_5EED;
        let flat: Vec<(f64, f64)> = (0..N)
            .map(|_| {
                sample_kalbach_mann(
                    e_in,
                    &energy_grid,
                    &n_x_per_i,
                    &interp_per_i,
                    &n_discrete_per_i,
                    &x_t,
                    &p_t,
                    &c_t,
                    &r_t,
                    &a_t,
                    &x_off,
                    &mut state,
                )
                .unwrap()
            })
            .collect();

        let mean_std = |v: &[f64]| {
            let n = v.len() as f64;
            let m = v.iter().sum::<f64>() / n;
            let s = (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n).sqrt();
            (m, s)
        };
        let (pe, pes) = mean_std(&prod.iter().map(|x| x.0).collect::<Vec<_>>());
        let (fe, fes) = mean_std(&flat.iter().map(|x| x.0).collect::<Vec<_>>());
        let (pm, pms) = mean_std(&prod.iter().map(|x| x.1).collect::<Vec<_>>());
        let (fm, fms) = mean_std(&flat.iter().map(|x| x.1).collect::<Vec<_>>());
        eprintln!("KM e_out: prod ({pe:.4e},{pes:.4e}) flat ({fe:.4e},{fes:.4e}) | mu: prod ({pm:.4},{pms:.4}) flat ({fm:.4},{fms:.4})");
        // Outgoing energy: uniform over [1e6, 1e7] both sides.
        assert!(((pe - fe) / pe).abs() < 0.01, "KM e_out mean differs");
        assert!(((pes - fes) / pes).abs() < 0.03, "KM e_out std differs");
        // Kalbach angle (r=0.3, a=1.0): mean + std must match.
        assert!(
            (pm - fm).abs() < 0.01,
            "KM mu mean differs (prod {pm:.4} flat {fm:.4})"
        );
        assert!(((pms - fms) / pms).abs() < 0.03, "KM mu std differs");
    }

    /// Discrete-line branch: a single discrete point at `x = 1.5e6`
    /// must sample exactly that energy (no continuous tail to
    /// stretch through).
    #[test]
    fn kalbach_mann_samples_discrete_line() {
        let energy_grid = [1.0e3, 1.0e7];
        let n_x_per_i = [2u32, 2u32];
        let interp_per_i = [KM_INTERP_HISTOGRAM, KM_INTERP_HISTOGRAM];
        // 1 discrete point per bracket, no continuous tail.
        let n_discrete_per_i = [1u32, 1u32];
        let max_x: usize = 4;
        let mut x_table = vec![0.0; 2 * max_x];
        let mut p_table = vec![0.0; 2 * max_x];
        let mut c_table = vec![0.0; 2 * max_x];
        let r_table = vec![0.5; 2 * max_x];
        let a_table = vec![0.0; 2 * max_x];
        // Discrete point (j=0): x = 1.5e6, p = 1.0 (impulse), c = 1.0
        // Sentinel (j=1): x = 1.5e6, c = 1.0 (CDF stays at 1).
        for row in 0..2 {
            let off = row * max_x;
            x_table[off] = 1.5e6;
            x_table[off + 1] = 1.5e6;
            p_table[off] = 1.0;
            c_table[off] = 1.0;
            c_table[off + 1] = 1.0;
        }
        let x_off = [0u32, max_x as u32];
        let mut state: u64 = 0xABCDEF;
        for _ in 0..100 {
            let (e_out, _mu) = sample_kalbach_mann(
                5.0e6,
                &energy_grid,
                &n_x_per_i,
                &interp_per_i,
                &n_discrete_per_i,
                &x_table,
                &p_table,
                &c_table,
                &r_table,
                &a_table,
                &x_off,
                &mut state,
            )
            .unwrap();
            assert!(
                (e_out - 1.5e6).abs() < 1.0,
                "discrete line should sample exactly 1.5e6, got {e_out}"
            );
        }
    }
}
