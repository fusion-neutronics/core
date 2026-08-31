//! Correlated angle-energy sampler (slice D in the GPU kernel
//! taxonomy: `EOUT_KIND_CORRELATED`).
//!
//! Joint `(E_out, μ)` distribution where the angular sub-table lives
//! at the chosen `(bin_e, j)` bin of the per-incident-energy E_out
//! table. Conceptually the same as `tabulated_continuous_eout` for
//! the E_out part -- bracket pick + CDF inversion with discrete +
//! continuous + bracket-stretch -- but with an additional second
//! tabulated angular distribution sampled at the chosen bin.
//!
//! No `histogram_outer` flag here (unlike slice C / `tabulated
//! _continuous_eout`); the bracket pick and stretch are always active
//! when both adjacent brackets carry usable data.
//!
//! Bit-identical to the cubecl kernel's `EOUT_KIND_CORRELATED`
//! branch. RNG draw schedule on the happy path: 3 draws -- `xi_ceb`
//! (bracket pick), `xi_cx` (E_out CDF inversion), `xi_cmu` (μ CDF
//! inversion within the chosen `(bin_e, j)` sub-table).

use super::grid::locate_bracket;
use yamc_rng::next_xi;

/// Interpolation flag for both the `(x, p, c)` and `(mu, pdf, cdf)`
/// sub-tables -- `0` selects histogram (linear-in-c), `1` selects
/// linear-linear (quadratic-in-c on the continuous tail).
pub const INTERP_HISTOGRAM: u32 = 0;
pub const INTERP_LINLIN: u32 = 1;

/// Sample a coupled `(E_out, μ)` pair from a correlated angle-energy
/// distribution.
///
/// # Arguments
/// * `e_in` -- incident energy.
/// * `energy_grid` -- sorted per-slot incident-energy grid, length
///   `n_e`.
/// * `n_x_per_i` -- number of `(x, p, c)` E_out points at each
///   incident-energy bracket, length `n_e`.
/// * `interp_per_i` -- E_out interpolation flag per bracket,
///   length `n_e`.
/// * `n_discrete_per_i` -- number of discrete E_out points at each
///   bracket (followed by `n_x − n_discrete` continuous points),
///   length `n_e`.
/// * `x_table`, `p_table`, `c_table` -- tight, full E_out tables
///   (one entry per global x-point, issue #104). Row `i`'s points
///   start at `x_offset[i]`.
/// * `n_mu_per_j` -- number of `(mu, pdf, cdf)` points at each
///   `(bracket, E_out)` bin, one entry per global x-point.
/// * `mu_interp_per_j` -- μ interpolation flag at each
///   `(bracket, E_out)` bin, one entry per global x-point.
/// * `mu_table`, `mu_pdf_table`, `mu_cdf_table` -- tight, full μ
///   sub-tables (one entry per global mu point). The sub-table for
///   global x-point `g` starts at `mu_offset[g]`.
/// * `x_offset` -- per-E_in-row CSR base (issue #104): global x-point
///   index where row `i`'s `(x, p, c)` points start. Length `n_e`.
/// * `mu_offset` -- per-x-point CSR base (issue #104): global mu-point
///   index where x-point `g`'s `(mu, pdf, cdf)` sub-table starts.
///   Indexed by the global x-point index `x_offset[bin_e] + j`.
/// * `state` -- inline PCG RNG state.
///
/// # Returns
/// `Some((e_out, Some(mu)))` when both E_out and μ are sampled.
/// `Some((e_out, None))` when E_out is sampled but the chosen
/// `(bin_e, j)` bin has no μ sub-table -- the caller keeps its
/// existing μ value (typically the slice-B sample or the isotropic
/// fallback). `None` when the slot is empty or the chosen E_out
/// bracket has fewer than 2 `(x, c)` points.
#[allow(clippy::too_many_arguments)]
pub fn sample_correlated_angle_energy(
    e_in: f64,
    energy_grid: &[f64],
    n_x_per_i: &[u32],
    interp_per_i: &[u32],
    n_discrete_per_i: &[u32],
    x_table: &[f64],
    p_table: &[f64],
    c_table: &[f64],
    n_mu_per_j: &[u32],
    mu_interp_per_j: &[u32],
    mu_table: &[f64],
    mu_pdf_table: &[f64],
    mu_cdf_table: &[f64],
    x_offset: &[u32],
    mu_offset: &[u32],
    state: &mut u64,
) -> Option<(f64, Option<f64>)> {
    let n_e = energy_grid.len();
    if n_e == 0 {
        return None;
    }
    let (i_eb, r_eb) = locate_bracket(e_in, energy_grid);

    let xi_ceb = next_xi(state);
    let bin_e = if r_eb > xi_ceb && i_eb + 1 < n_e {
        i_eb + 1
    } else {
        i_eb
    };

    let n_x = n_x_per_i[bin_e] as usize;
    if n_x < 2 {
        return None;
    }
    let x_off = x_offset[bin_e] as usize;
    let n_disc = n_discrete_per_i[bin_e] as usize;
    let interp_kind = interp_per_i[bin_e];
    let xi_cx = next_xi(state);

    // Discrete-then-continuous CDF search, matching the CPU reference
    // `sample_with_discrete_info` (issue #103). Discrete head (first `n_disc`
    // points) searched with `xi < c[k]` (exact discrete line); continuous tail
    // from `n_disc` with `xi <= c[k+1]`, carrying `c_j` as the CPU does. The
    // old single unified scan collapsed every discrete line onto index 0. For
    // `n_disc == 0` this is identical to the old continuous-only scan.
    let mut j = 0usize;
    let mut c_j = c_table[x_off];
    let mut is_discrete = false;
    let mut done = false;
    for k in 0..n_disc {
        if !done {
            j = k;
            c_j = c_table[x_off + k];
            if xi_cx < c_j {
                is_discrete = true;
                done = true;
            }
        }
    }
    for k in n_disc..n_x - 1 {
        if !done {
            j = k;
            let c_k1 = c_table[x_off + k + 1];
            if xi_cx <= c_k1 {
                done = true;
            } else {
                j = k + 1;
                c_j = c_k1;
            }
        }
    }
    if !is_discrete && j >= n_x - 1 {
        j = n_x - 2;
    }
    let x_j = x_table[x_off + j];
    let x_j1 = x_table[x_off + j + 1];
    let p_j = p_table[x_off + j];
    let p_j1 = p_table[x_off + j + 1];
    let dx = x_j1 - x_j;
    let mut e_sampled = if interp_kind == INTERP_LINLIN && !is_discrete && dx > 0.0 {
        let m = (p_j1 - p_j) / dx;
        if m.abs() < 1e-30 {
            if p_j > 0.0 {
                x_j + (xi_cx - c_j) / p_j
            } else {
                x_j
            }
        } else {
            let disc = (p_j * p_j + 2.0 * m * (xi_cx - c_j)).max(0.0);
            x_j + (disc.sqrt() - p_j) / m
        }
    } else if !is_discrete && p_j > 0.0 {
        x_j + (xi_cx - c_j) / p_j
    } else {
        x_j
    };

    // Bracket-bound stretch -- skipped when the sampled point is in
    // the discrete head.
    if !is_discrete && n_e >= 2 {
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

    if e_sampled < 0.0 {
        e_sampled = 0.0;
    }

    // Sample μ from the angular sub-table at the chosen (bin_e, j)
    // bin. The kernel uses a per-`(i, j)` table with the same
    // tabulated CDF inversion shape as the slice-B angular tables.
    // When no sub-table is present at this bin, return `None` for μ
    // -- the caller keeps whatever μ it had before this branch.
    // Which of the bracketing E_out points supplies the angular sub-table.
    // The correlated law tabulates μ AT the E_out grid points, so a sampled
    // energy strictly inside `[x_j, x_j+1]` has two candidates and the law
    // takes the NEARER one, measured on the CDF (issue #371):
    //
    //   xi - c_j < c_j+1 - xi  ->  j, else j+1
    //
    // This flat path used to take `j` unconditionally. That is not a rounding
    // difference: the sub-tables at adjacent E_out points differ most exactly
    // where the spectrum falls off, so on W184 MT91 at 14.06 MeV it doubled
    // the population above 12.84 MeV and cut the bin below it by 4.6%. The
    // pick reads `c_table` only, so the RNG draw schedule is unchanged and the
    // kernel twin stays in lockstep.
    //
    // A point in the discrete head, or a histogram row, takes `j` outright --
    // there is no "inside the bracket" to be nearer one end of.
    let mu_bin = if is_discrete || interp_kind == INTERP_HISTOGRAM {
        j
    } else {
        let c_j1 = c_table[x_off + j + 1];
        if xi_cx - c_j < c_j1 - xi_cx {
            j
        } else {
            j + 1
        }
    };
    let mu_idx = x_off + mu_bin;
    let n_mu = n_mu_per_j[mu_idx] as usize;
    if n_mu < 2 {
        return Some((e_sampled, None));
    }
    let mu_off = mu_offset[mu_idx] as usize;
    let xi_cmu = next_xi(state);
    let mut mj = 0usize;
    let mut mj_found = false;
    for k in 0..n_mu - 1 {
        let c_k1 = mu_cdf_table[mu_off + k + 1];
        if !mj_found && xi_cmu <= c_k1 {
            mj = k;
            mj_found = true;
        }
    }
    if !mj_found {
        mj = n_mu - 2;
    }
    let cm_j = mu_cdf_table[mu_off + mj];
    let xm_j = mu_table[mu_off + mj];
    let xm_j1 = mu_table[mu_off + mj + 1];
    let pm_j = mu_pdf_table[mu_off + mj];
    let pm_j1 = mu_pdf_table[mu_off + mj + 1];
    let dx_m = xm_j1 - xm_j;
    let mu_interp = mu_interp_per_j[mu_idx];
    let mu_raw = if mu_interp == INTERP_LINLIN && dx_m > 0.0 {
        let slope = (pm_j1 - pm_j) / dx_m;
        if slope.abs() < 1e-30 {
            if pm_j > 0.0 {
                xm_j + (xi_cmu - cm_j) / pm_j
            } else {
                xm_j
            }
        } else {
            let disc = (pm_j * pm_j + 2.0 * slope * (xi_cmu - cm_j)).max(0.0);
            xm_j + (disc.sqrt() - pm_j) / slope
        }
    } else if pm_j > 0.0 {
        xm_j + (xi_cmu - cm_j) / pm_j
    } else {
        xm_j
    };
    Some((e_sampled, Some(mu_raw.clamp(-1.0, 1.0))))
}

// ----------------------------- Tests -----------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty grid → `None`, no RNG advance.
    #[test]
    fn empty_slot_returns_none() {
        let empty_f: [f64; 0] = [];
        let empty_u: [u32; 0] = [];
        let mut state: u64 = 0xDEADBEEF;
        let before = state;
        let r = sample_correlated_angle_energy(
            5.0e6, &empty_f, &empty_u, &empty_u, &empty_u, &empty_f, &empty_f, &empty_f, &empty_u,
            &empty_u, &empty_f, &empty_f, &empty_f, &empty_u, &empty_u, &mut state,
        );
        assert!(r.is_none());
        assert_eq!(state, before);
    }

    /// 2-bracket synthetic with uniform E_out CDFs and a forward-
    /// peaked μ sub-table at every (i, j). Mean μ over many draws
    /// must be > 0; mean e_out lies within the stretched bounds.
    #[test]
    fn correlated_samples_forward_peak() {
        let max_x = 4usize;
        let max_mu = 4usize;
        let n_e = 2usize;
        let energy_grid = [1.0e3, 1.0e7];
        let n_x_per_i = [3u32, 3u32];
        let interp_per_i = [INTERP_LINLIN, INTERP_LINLIN];
        let n_discrete_per_i = [0u32, 0u32];

        let mut x_table = vec![0.0; n_e * max_x];
        let mut p_table = vec![0.0; n_e * max_x];
        let mut c_table = vec![0.0; n_e * max_x];
        // Bracket 0: x = [1e5, 1e6, 2e6]; uniform CDF.
        // Bracket 1: x = [2e5, 1.5e6, 3e6]; uniform CDF.
        let xs0 = [1.0e5, 1.0e6, 2.0e6];
        let xs1 = [2.0e5, 1.5e6, 3.0e6];
        let cs = [0.0, 0.5, 1.0];
        x_table[..3].copy_from_slice(&xs0);
        c_table[..3].copy_from_slice(&cs);
        x_table[max_x..max_x + 3].copy_from_slice(&xs1);
        c_table[max_x..max_x + 3].copy_from_slice(&cs);
        for j in 0..2 {
            p_table[j] = (cs[j + 1] - cs[j]) / (xs0[j + 1] - xs0[j]);
            p_table[max_x + j] = (cs[j + 1] - cs[j]) / (xs1[j + 1] - xs1[j]);
        }

        // Forward-peaked μ sub-table at every (i, j) bin: μ uniform
        // on [0.5, 1.0] with CDF [0, 1]. 2 points each.
        let n_mu_per_j = vec![2u32; n_e * max_x];
        let mu_interp_per_j = vec![INTERP_LINLIN; n_e * max_x];
        let mut mu_table = vec![0.0; n_e * max_x * max_mu];
        let mut mu_pdf_table = vec![0.0; n_e * max_x * max_mu];
        let mut mu_cdf_table = vec![0.0; n_e * max_x * max_mu];
        for ij in 0..n_e * max_x {
            let off = ij * max_mu;
            mu_table[off] = 0.5;
            mu_table[off + 1] = 1.0;
            mu_pdf_table[off] = 2.0;
            mu_pdf_table[off + 1] = 2.0;
            mu_cdf_table[off] = 0.0;
            mu_cdf_table[off + 1] = 1.0;
        }
        // CSR offsets for the dense fixture: row `i` starts at `i * max_x`,
        // and x-point `g`'s mu sub-table at `g * max_mu` (issue #104).
        let x_offset: Vec<u32> = (0..n_e).map(|i| (i * max_x) as u32).collect();
        let mu_offset: Vec<u32> = (0..n_e * max_x).map(|g| (g * max_mu) as u32).collect();

        let mut state: u64 = 0xCAFEBABE;
        let n = 4_000usize;
        let mut sum_mu = 0.0;
        let mut sum_e = 0.0;
        for _ in 0..n {
            let (e, mu_opt) = sample_correlated_angle_energy(
                5.0e6,
                &energy_grid,
                &n_x_per_i,
                &interp_per_i,
                &n_discrete_per_i,
                &x_table,
                &p_table,
                &c_table,
                &n_mu_per_j,
                &mu_interp_per_j,
                &mu_table,
                &mu_pdf_table,
                &mu_cdf_table,
                &x_offset,
                &mu_offset,
                &mut state,
            )
            .unwrap();
            let mu = mu_opt.expect("μ sub-table is populated everywhere");
            assert!((0.5..=1.0).contains(&mu), "μ {mu} should be in [0.5, 1]");
            sum_e += e;
            sum_mu += mu;
        }
        let mean_mu = sum_mu / n as f64;
        let mean_e = sum_e / n as f64;
        // Mean μ should be ≈ 0.75 (midpoint of [0.5, 1.0]).
        assert!(
            (0.70..=0.80).contains(&mean_mu),
            "mean μ {mean_mu} should be ≈ 0.75 (forward peak [0.5, 1])"
        );
        // Stretched E_out bounds at midpoint: [1.5e5, 2.5e6].
        assert!(
            (5.0e5..=2.5e6).contains(&mean_e),
            "mean e_out {mean_e} should fall in stretched bounds"
        );
    }

    /// Issue #371: the angular sub-table comes from the NEARER of the two
    /// bracketing E_out points, not always the lower one.
    ///
    /// The law tabulates μ AT the E_out grid points, so a sample landing
    /// strictly inside `[x_j, x_j+1]` has two candidates, and the reference
    /// (`CorrelatedAngleEnergy::sample`, and OpenMC) takes whichever the
    /// sampled CDF value is closer to. This path used to take `j` outright.
    ///
    /// Every other test in this module uses the SAME μ sub-table at every
    /// `(i, j)`, which cannot tell the two rules apart -- that is how the bug
    /// survived. Here the two ends of the single E_out bracket disagree
    /// completely: μ = -1 at point 0, μ = +1 at point 1. The sign of the
    /// sampled μ then reports which end was used, and the CDF is uniform, so
    /// the lower half of the draws must take point 0 and the upper half
    /// point 1.
    #[test]
    fn mu_sub_table_comes_from_the_nearer_e_out_point() {
        let n_e = 2usize;
        let max_x = 2usize;
        let max_mu = 2usize;
        let energy_grid = [1.0e3, 1.0e7];
        let n_x_per_i = [2u32, 2u32];
        let interp_per_i = [INTERP_LINLIN, INTERP_LINLIN];
        let n_discrete_per_i = [0u32, 0u32];

        // One E_out bracket per incident point, uniform CDF 0 -> 1.
        let x_table = vec![1.0e6, 2.0e6, 1.0e6, 2.0e6];
        let p_table = vec![1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6];
        let c_table = vec![0.0, 1.0, 0.0, 1.0];

        // μ sub-tables: point 0 is a hard backward delta-ish table on
        // [-1, -0.9], point 1 a forward one on [0.9, 1].
        let n_mu_per_j = vec![2u32; n_e * max_x];
        let mu_interp_per_j = vec![INTERP_LINLIN; n_e * max_x];
        let mut mu_table = vec![0.0; n_e * max_x * max_mu];
        let mut mu_pdf_table = vec![0.0; n_e * max_x * max_mu];
        let mut mu_cdf_table = vec![0.0; n_e * max_x * max_mu];
        for g in 0..n_e * max_x {
            let off = g * max_mu;
            let backward = g % max_x == 0;
            mu_table[off] = if backward { -1.0 } else { 0.9 };
            mu_table[off + 1] = if backward { -0.9 } else { 1.0 };
            mu_pdf_table[off] = 10.0;
            mu_pdf_table[off + 1] = 10.0;
            mu_cdf_table[off] = 0.0;
            mu_cdf_table[off + 1] = 1.0;
        }
        let x_offset: Vec<u32> = (0..n_e).map(|i| (i * max_x) as u32).collect();
        let mu_offset: Vec<u32> = (0..n_e * max_x).map(|g| (g * max_mu) as u32).collect();

        let mut state: u64 = 0x5EED_1234;
        let n = 20_000usize;
        let mut forward = 0usize;
        for _ in 0..n {
            let (_e, mu) = sample_correlated_angle_energy(
                5.0e6,
                &energy_grid,
                &n_x_per_i,
                &interp_per_i,
                &n_discrete_per_i,
                &x_table,
                &p_table,
                &c_table,
                &n_mu_per_j,
                &mu_interp_per_j,
                &mu_table,
                &mu_pdf_table,
                &mu_cdf_table,
                &x_offset,
                &mu_offset,
                &mut state,
            )
            .unwrap();
            if mu.expect("mu table present") > 0.0 {
                forward += 1;
            }
        }
        let frac = forward as f64 / n as f64;
        // Uniform CDF: half the draws sit nearer the upper E_out point.
        // Taking `j` unconditionally gives 0.0 here.
        assert!(
            (0.47..=0.53).contains(&frac),
            "forward fraction {frac} should be ~0.5; taking the lower E_out \
             point unconditionally gives 0.0 (issue #371)"
        );
    }

    /// When the chosen `(bin_e, j)` bin has no μ table, the sampler
    /// returns `mu = None` so the caller can keep its existing μ
    /// value (e.g. the slice-B sample or the isotropic fallback).
    #[test]
    fn returns_none_mu_when_no_mu_table() {
        let max_x = 4usize;
        let max_mu = 4usize;
        let n_e = 2usize;
        let energy_grid = [1.0e3, 1.0e7];
        let n_x_per_i = [3u32, 3u32];
        let interp_per_i = [INTERP_LINLIN, INTERP_LINLIN];
        let n_discrete_per_i = [0u32, 0u32];

        let mut x_table = vec![0.0; n_e * max_x];
        let mut p_table = vec![0.0; n_e * max_x];
        let mut c_table = vec![0.0; n_e * max_x];
        let xs = [1.0e5, 1.0e6, 2.0e6];
        let cs = [0.0, 0.5, 1.0];
        for row in 0..n_e {
            let off = row * max_x;
            x_table[off..off + 3].copy_from_slice(&xs);
            c_table[off..off + 3].copy_from_slice(&cs);
            for j in 0..2 {
                p_table[off + j] = (cs[j + 1] - cs[j]) / (xs[j + 1] - xs[j]);
            }
        }
        // No μ data anywhere -- n_mu_per_j all zero.
        let n_mu_per_j = vec![0u32; n_e * max_x];
        let mu_interp_per_j = vec![INTERP_LINLIN; n_e * max_x];
        let mu_table = vec![0.0; n_e * max_x * max_mu];
        let mu_pdf_table = vec![0.0; n_e * max_x * max_mu];
        let mu_cdf_table = vec![0.0; n_e * max_x * max_mu];
        // CSR offsets for the dense fixture (issue #104).
        let x_offset: Vec<u32> = (0..n_e).map(|i| (i * max_x) as u32).collect();
        let mu_offset: Vec<u32> = (0..n_e * max_x).map(|g| (g * max_mu) as u32).collect();

        let mut state: u64 = 1;
        let (_e, mu_opt) = sample_correlated_angle_energy(
            5.0e6,
            &energy_grid,
            &n_x_per_i,
            &interp_per_i,
            &n_discrete_per_i,
            &x_table,
            &p_table,
            &c_table,
            &n_mu_per_j,
            &mu_interp_per_j,
            &mu_table,
            &mu_pdf_table,
            &mu_cdf_table,
            &x_offset,
            &mu_offset,
            &mut state,
        )
        .unwrap();
        assert!(
            mu_opt.is_none(),
            "expected None for μ when no sub-table data is present, got {mu_opt:?}"
        );
    }
}
