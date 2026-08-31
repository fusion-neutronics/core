//! Fission outgoing-energy sampler for the per-material tabulated
//! ContinuousTabular distribution (`EOUT_KIND_CONTINUOUS_TABULAR`
//! on the fission branch).
//!
//! Implements three coupled steps, bit-identical to the cubecl
//! kernel's fission branch:
//!
//! 1. **Stochastic incident-energy bracket pick.** Locate the bracket
//!    `[E_i, E_{i+1}]` containing `E_in`, then pick `bin_e = i` or
//!    `i+1` with probability `r_eb` (the linear fraction of `E_in`
//!    within the bracket). This avoids the discontinuity at bracket
//!    boundaries that pure interpolation would introduce.
//!
//! 2. **Interp-aware CDF inversion** within the chosen bracket's
//!    `(x, p, c)` table: find `j` such that `ξ ≤ c[j+1]`, then invert
//!    within the bin honouring the row's tabulated interpolation.
//!    Lin-lin rows (`interp_per_i[bin] == 1`) use the quadratic
//!    inversion off the PDF (`x_j + (√(p_j² + 2m(ξ − c_j)) − p_j)/m`
//!    with `m = (p_{j+1} − p_j)/Δx`, same form as the inelastic
//!    ContinuousTabular eout sampler); histogram / degenerate rows keep
//!    the legacy linear-in-c form across `(c_j → c_{j+1}, x_j → x_{j+1})`
//!    (identical for a consistent histogram table where `p_j = Δc/Δx`).
//!
//! 3. **Bracket-bound stretch** (the subtle bit). If both incident-
//!    energy brackets `i` and `i+1` have valid data (`n_x ≥ 2`),
//!    compute the interpolated CM-energy bounds
//!    `e_1 = e_{i,1} + r·(e_{i+1,1} − e_{i,1})` and
//!    `e_k = e_{i,k} + r·(e_{i+1,k} − e_{i,k})`,
//!    then stretch the sample from the chosen bracket's actual
//!    endpoints `(e_l,1, e_l,k)` to the interpolated `(e_1, e_k)`.
//!    Without this, the sampled CM energy would be discontinuous
//!    across the bracket boundary.

use super::grid::locate_bracket;
use yamc_rng::next_xi;

/// Sample an outgoing CM-frame fission energy from a per-material
/// tabulated `(E_in, x, CDF)` distribution.
///
/// # Arguments
/// * `e_in` -- incident energy.
/// * `energy_grid` -- sorted per-material incident-energy grid of
///   length `n_e`.
/// * `n_x_per_i` -- number of `(x, cdf)` points at each incident-energy
///   bracket, length `n_e`.
/// * `x_table`, `cdf_table`, `p_table` -- flat tables packed
///   variable-length (issue #104): row `i` occupies `x_offset[i] ..
///   x_offset[i] + n_x_per_i[i]`. Rows are stored back-to-back with no
///   padding. `p_table` is the PDF normalized like `cdf_table`;
///   zero-filled rows (no usable PDF) must carry `interp == 0` so the
///   inversion falls back to linear-in-c.
/// * `x_offset` -- start index of each incident-energy row in the flat
///   `x_table` / `cdf_table` / `p_table`, length `n_e`.
/// * `interp_per_i` -- per-row interpolation code, length `n_e`:
///   `0` histogram (legacy linear-in-c inversion), `1` lin-lin
///   (quadratic inversion off the PDF).
/// * `state` -- inline PCG RNG state, advanced 2× (one for the
///   bracket pick `ξ_feeb`, one for the CDF inversion `ξ_fx`).
///
/// # Returns
/// `Some(e_out)` when the slot has data and the inversion produces
/// a valid positive energy (clamped to `1e-6` if the table itself
/// is all zeros at this bracket). `None` when the slot is empty
/// (`n_e == 0`) or the chosen bracket has fewer than 2 (x, cdf)
/// points -- callers fall back to the Watt rejection branch.
#[allow(clippy::too_many_arguments)]
pub fn sample_fission_eout_continuous(
    e_in: f64,
    energy_grid: &[f64],
    n_x_per_i: &[u32],
    x_table: &[f64],
    cdf_table: &[f64],
    p_table: &[f64],
    x_offset: &[u32],
    interp_per_i: &[u32],
    state: &mut u64,
) -> Option<f64> {
    let n_e = energy_grid.len();
    if n_e == 0 {
        return None;
    }
    let (i_eb, r_eb) = locate_bracket(e_in, energy_grid);

    let xi_feeb = next_xi(state);
    let mut bin_e = i_eb;
    if r_eb > xi_feeb && bin_e + 1 < n_e {
        bin_e = i_eb + 1;
    }

    let n_x = n_x_per_i[bin_e] as usize;
    if n_x < 2 {
        return None;
    }
    let x_off = x_offset[bin_e] as usize;
    let xi_fx = next_xi(state);

    // Linear scan for the first `j` with `ξ ≤ c[j+1]`. Match the
    // kernel's "first match wins, else last" semantics -- replacing
    // with binary search would change the RNG-correlated bias on
    // platforms where `c` has plateaus.
    let mut j = 0usize;
    let mut j_found = false;
    for k in 0..n_x - 1 {
        let c_k1 = cdf_table[x_off + k + 1];
        if !j_found && xi_fx <= c_k1 {
            j = k;
            j_found = true;
        }
    }
    if !j_found {
        j = n_x - 2;
    }
    let c_j = cdf_table[x_off + j];
    let c_j1 = cdf_table[x_off + j + 1];
    let x_j = x_table[x_off + j];
    let x_j1 = x_table[x_off + j + 1];
    let p_j = p_table[x_off + j];
    let p_j1 = p_table[x_off + j + 1];
    let dx = x_j1 - x_j;
    let dc = c_j1 - c_j;
    let mut e_sampled = x_j;
    if interp_per_i[bin_e] == 1 && dx > 0.0 {
        // Lin-lin row: quadratic inversion off the stored PDF -- same
        // form as the inelastic ContinuousTabular eout sampler.
        let m = (p_j1 - p_j) / dx;
        if m.abs() < 1e-30 {
            if p_j > 0.0 {
                e_sampled = x_j + (xi_fx - c_j) / p_j;
            }
        } else {
            let disc = (p_j * p_j + 2.0 * m * (xi_fx - c_j)).max(0.0);
            e_sampled = x_j + (disc.sqrt() - p_j) / m;
        }
    } else if dc > 0.0 {
        // Histogram / degenerate row: legacy linear-in-c form (identical
        // to a histogram inversion when the table is consistent,
        // p_j == dc/dx).
        e_sampled = x_j + (xi_fx - c_j) / dc * (x_j1 - x_j);
    }

    // Bracket-bound stretch: only when both adjacent brackets carry
    // ≥2 points so the interpolated endpoints `(e_1, e_k)` are
    // well-defined. Index `i_eb + 1` is safe because `r_eb == 1` is
    // only reached when `n_e > 1` and `i_eb == n_e − 2`.
    if n_e >= 2 {
        let n_x_i = n_x_per_i[i_eb] as usize;
        let n_x_i1 = n_x_per_i[i_eb + 1] as usize;
        if n_x_i >= 2 && n_x_i1 >= 2 {
            let x_off_i = x_offset[i_eb] as usize;
            let x_off_i1 = x_offset[i_eb + 1] as usize;
            let e_i_1 = x_table[x_off_i];
            let e_i_k = x_table[x_off_i + n_x_i - 1];
            let e_i1_1 = x_table[x_off_i1];
            let e_i1_k = x_table[x_off_i1 + n_x_i1 - 1];
            let e_1 = e_i_1 + r_eb * (e_i1_1 - e_i_1);
            let e_k = e_i_k + r_eb * (e_i1_k - e_i_k);
            let (e_l_1, e_l_k) = if bin_e == i_eb {
                (e_i_1, e_i_k)
            } else {
                (e_i1_1, e_i1_k)
            };
            let denom_l = e_l_k - e_l_1;
            if denom_l > 0.0 {
                e_sampled = e_1 + (e_sampled - e_l_1) * (e_k - e_1) / denom_l;
            }
        }
    }

    if e_sampled <= 0.0 {
        e_sampled = 1.0e-6;
    }
    Some(e_sampled)
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Histogram-consistent PDF for the strided test tables: within each
    /// row, `p[k] = (c[k+1] - c[k]) / (x[k+1] - x[k])` and the last point
    /// repeats the last bin's value. With `interp == 0` the inversion
    /// never reads it, but keeping the tables self-consistent mirrors
    /// production packing.
    fn dcdx_pdf(
        x_table: &[f64],
        cdf_table: &[f64],
        x_offset: &[u32],
        n_x_per_i: &[u32],
    ) -> Vec<f64> {
        let mut p = vec![0.0; x_table.len()];
        for (i, &off) in x_offset.iter().enumerate() {
            let off = off as usize;
            let n = n_x_per_i[i] as usize;
            if n < 2 {
                continue;
            }
            for k in 0..n - 1 {
                let dx = x_table[off + k + 1] - x_table[off + k];
                if dx > 0.0 {
                    p[off + k] = (cdf_table[off + k + 1] - cdf_table[off + k]) / dx;
                }
            }
            p[off + n - 1] = p[off + n - 2];
        }
        p
    }

    /// Empty grid → `None`, no state advance.
    #[test]
    fn fission_eout_returns_none_for_empty_slot() {
        let empty_f: [f64; 0] = [];
        let empty_u: [u32; 0] = [];
        let mut state: u64 = 0xC0FFEE;
        let before = state;
        let r = sample_fission_eout_continuous(
            5.0e6, &empty_f, &empty_u, &empty_f, &empty_f, &empty_f, &empty_u, &empty_u, &mut state,
        );
        assert!(r.is_none());
        assert_eq!(state, before, "empty slot must not advance RNG state");
    }

    /// Single-point grid where the chosen bracket has `n_x < 2`
    /// → `None`, RNG advances exactly once (the bracket pick draw)
    /// before the early return.
    #[test]
    fn fission_eout_returns_none_when_chosen_bracket_too_narrow() {
        // 2 brackets, second has only 1 point.
        let energy_grid = [1.0e3, 1.0e7];
        let n_x_per_i = [4u32, 1u32];
        let max_x: usize = 8;
        let mut x_table = vec![0.0; 2 * max_x];
        let mut cdf_table = vec![0.0; 2 * max_x];
        // Bracket 0: x = [1e5, 5e5, 1e6, 2e6], cdf = [0, 0.3, 0.7, 1.0]
        for (j, &(x, c)) in [(1.0e5, 0.0), (5.0e5, 0.3), (1.0e6, 0.7), (2.0e6, 1.0)]
            .iter()
            .enumerate()
        {
            x_table[j] = x;
            cdf_table[j] = c;
        }
        // Bracket 1: only x[0] populated -- degenerate.
        x_table[max_x] = 1.0e5;
        // Force E_in into bracket 1 with r_eb = 1.0 so the stochastic
        // pick always lands on bin_e = 1 (`r_eb > xi` always true
        // since xi ∈ (0, 1]).
        let e_in = 2.0e7; // above e_last → r_eb = 1.0, i_eb = 0
        let x_off = [0u32, max_x as u32];
        let p_table = dcdx_pdf(&x_table, &cdf_table, &x_off, &n_x_per_i);
        let interp = [0u32, 0u32];
        let mut state: u64 = 42;
        let r = sample_fission_eout_continuous(
            e_in,
            &energy_grid,
            &n_x_per_i,
            &x_table,
            &cdf_table,
            &p_table,
            &x_off,
            &interp,
            &mut state,
        );
        assert!(
            r.is_none(),
            "expected None on degenerate bracket, got {r:?}"
        );
    }

    /// Happy path: with both adjacent brackets carrying valid data,
    /// the sampled CM energy lies within the interpolated bracket
    /// bounds `[e_1, e_k]` (after the stretch step). Drives 4k
    /// samples through a 2-bracket synthetic distribution where
    /// bracket 0 covers `[1e5, 2e6]` and bracket 1 covers
    /// `[2e5, 3e6]`; at midpoint E_in the stretched bounds are
    /// `[1.5e5, 2.5e6]`.
    #[test]
    fn fission_eout_samples_within_stretched_bounds() {
        let energy_grid = [1.0e3, 1.0e7];
        let n_x_per_i = [4u32, 4u32];
        let max_x: usize = 8;
        let mut x_table = vec![0.0; 2 * max_x];
        let mut cdf_table = vec![0.0; 2 * max_x];
        // Bracket 0: x = [1e5, 6e5, 1.4e6, 2e6], cdf = [0, 0.3, 0.7, 1.0]
        for (j, &(x, c)) in [(1.0e5, 0.0), (6.0e5, 0.3), (1.4e6, 0.7), (2.0e6, 1.0)]
            .iter()
            .enumerate()
        {
            x_table[j] = x;
            cdf_table[j] = c;
        }
        // Bracket 1: x = [2e5, 8e5, 2.2e6, 3e6], cdf = [0, 0.3, 0.7, 1.0]
        for (j, &(x, c)) in [(2.0e5, 0.0), (8.0e5, 0.3), (2.2e6, 0.7), (3.0e6, 1.0)]
            .iter()
            .enumerate()
        {
            x_table[max_x + j] = x;
            cdf_table[max_x + j] = c;
        }

        // E_in = 5.0005e6 → r_eb ≈ 0.5; stretched bounds:
        //   e_1 = 1e5 + 0.5·(2e5 - 1e5) = 1.5e5
        //   e_k = 2e6 + 0.5·(3e6 - 2e6) = 2.5e6
        let e_in = 5.0005e6;
        let lo = 1.5e5;
        let hi = 2.5e6;
        let x_off = [0u32, max_x as u32];
        let p_table = dcdx_pdf(&x_table, &cdf_table, &x_off, &n_x_per_i);
        let interp = [0u32, 0u32];
        let mut state: u64 = 0xBEEFCAFE;
        let n = 4_000usize;
        let mut sum = 0.0;
        for _ in 0..n {
            let e_out = sample_fission_eout_continuous(
                e_in,
                &energy_grid,
                &n_x_per_i,
                &x_table,
                &cdf_table,
                &p_table,
                &x_off,
                &interp,
                &mut state,
            )
            .expect("happy path should always return Some");
            assert!(
                (lo..=hi).contains(&e_out),
                "e_out {e_out} fell outside stretched bounds [{lo}, {hi}]"
            );
            sum += e_out;
        }
        // Mean should lie in the middle of the stretched range
        // (CDF is symmetric in this construction).
        let mean = sum / n as f64;
        let mid = 0.5 * (lo + hi);
        let rel_err = ((mean - mid) / mid).abs();
        assert!(
            rel_err < 0.10,
            "mean {mean:.4e} differs from midpoint {mid:.4e} by {rel_err:.3} (>10%)"
        );
    }

    /// Below-range E_in clamps to the first bracket (`i_eb = 0`,
    /// `r_eb = 0.0`); the stochastic pick always lands on
    /// `bin_e = 0` (no second bracket reachable with `r_eb = 0`).
    #[test]
    fn fission_eout_below_range_clamps_to_first_bracket() {
        let energy_grid = [1.0e6, 1.0e7];
        let n_x_per_i = [3u32, 3u32];
        let max_x: usize = 8;
        let mut x_table = vec![0.0; 2 * max_x];
        let mut cdf_table = vec![0.0; 2 * max_x];
        // Bracket 0: x = [10, 20, 30], cdf = [0, 0.5, 1.0]
        for (j, &(x, c)) in [(10.0, 0.0), (20.0, 0.5), (30.0, 1.0)].iter().enumerate() {
            x_table[j] = x;
            cdf_table[j] = c;
        }
        // Bracket 1: x = [1000, 2000, 3000] -- easy to detect if
        // anything leaks through.
        for (j, &(x, c)) in [(1000.0, 0.0), (2000.0, 0.5), (3000.0, 1.0)]
            .iter()
            .enumerate()
        {
            x_table[max_x + j] = x;
            cdf_table[max_x + j] = c;
        }
        // E_in = 1 eV -- well below grid[0]; with the stretch step
        // off (only `bin_e = 0` reachable and `r_eb = 0`), e_sampled
        // must lie in [10, 30].
        let x_off = [0u32, max_x as u32];
        let p_table = dcdx_pdf(&x_table, &cdf_table, &x_off, &n_x_per_i);
        let interp = [0u32, 0u32];
        let mut state: u64 = 7;
        for _ in 0..200 {
            let e_out = sample_fission_eout_continuous(
                1.0,
                &energy_grid,
                &n_x_per_i,
                &x_table,
                &cdf_table,
                &p_table,
                &x_off,
                &interp,
                &mut state,
            )
            .unwrap();
            assert!(
                (10.0..=30.0).contains(&e_out),
                "below-range E_in should sample bracket 0 only, got {e_out}"
            );
        }
    }

    /// Interp-aware inversion: a single-bracket row whose PDF rises
    /// linearly from 0 (`x = [0, 1e6]`, `p = [0, 2e-6]`, `c = [0, 1]`)
    /// sampled with `interp = 1` (lin-lin) must reproduce the triangular
    /// distribution's mean `2/3 · 1e6`; the SAME table with `interp = 0`
    /// must keep the legacy linear-in-c behaviour (uniform in x, mean
    /// `1/2 · 1e6`). Guards the fix in both directions.
    #[test]
    fn fission_eout_linlin_row_uses_pdf_quadratic_inversion() {
        let energy_grid = [1.0e6];
        let n_x_per_i = [2u32];
        let x_table = [0.0, 1.0e6];
        let p_table = [0.0, 2.0e-6];
        let cdf_table = [0.0, 1.0];
        let x_off = [0u32];
        let n = 200_000usize;

        // Lin-lin: E[x] of p(x) = 2x/(1e6)^2 on [0, 1e6] is 2/3 · 1e6.
        let interp_linlin = [1u32];
        let mut state: u64 = 0xF15510;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += sample_fission_eout_continuous(
                1.0e6,
                &energy_grid,
                &n_x_per_i,
                &x_table,
                &cdf_table,
                &p_table,
                &x_off,
                &interp_linlin,
                &mut state,
            )
            .expect("lin-lin row should always sample");
        }
        let mean = sum / n as f64;
        let expect = 2.0 / 3.0 * 1.0e6;
        let rel = ((mean - expect) / expect).abs();
        assert!(
            rel < 0.01,
            "lin-lin mean {mean:.4e} differs from {expect:.4e} by {rel:.4} (>1%)"
        );

        // Histogram code on the same table: legacy linear-in-c, mean 0.5e6.
        let interp_hist = [0u32];
        let mut state: u64 = 0xF15511;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += sample_fission_eout_continuous(
                1.0e6,
                &energy_grid,
                &n_x_per_i,
                &x_table,
                &cdf_table,
                &p_table,
                &x_off,
                &interp_hist,
                &mut state,
            )
            .expect("histogram row should always sample");
        }
        let mean = sum / n as f64;
        let expect = 0.5e6;
        let rel = ((mean - expect) / expect).abs();
        assert!(
            rel < 0.01,
            "histogram mean {mean:.4e} differs from {expect:.4e} by {rel:.4} (>1%)"
        );
    }
}
