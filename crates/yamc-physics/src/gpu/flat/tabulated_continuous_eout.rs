//! Tabulated continuous E_out sampler (slice C in the GPU kernel
//! taxonomy: `EOUT_KIND_CONTINUOUS_TABULAR` on the inelastic branch).
//!
//! Generalises the simpler `fission_eout_continuous` sampler:
//!
//! - Per-bracket `(x, p, c)` triplet (PDF column makes the quadratic
//!   linlin inversion possible -- fission uses only linear-in-c).
//! - Per-bracket interpolation flag (`linlin` or `histogram`).
//! - Discrete portion at the start of each bracket (`n_discrete` of the
//!   `n_x` table points are discrete delta lines; the rest are
//!   continuous).
//! - Global `histogram_outer` flag that suppresses both the stochastic
//!   incident-energy bracket pick AND the bracket-bound stretch
//!   (mirrors `ContinuousTabular`'s outer interp kind).
//!
//! Returns `e_cm` in the laboratory units used by the kernel. Bit-
//! identical to the cubecl kernel's `EOUT_KIND_CONTINUOUS_TABULAR`
//! inelastic branch.

use super::grid::locate_bracket;
use yamc_rng::next_xi;

/// Interpolation flag for the per-bracket `(x, p, c)` table --
/// `0` selects histogram (linear-in-c on the continuous tail),
/// `1` selects linear-linear (quadratic-in-c on the continuous
/// tail). Mirrors the kernel's u32 codes.
pub const INTERP_HISTOGRAM: u32 = 0;
pub const INTERP_LINLIN: u32 = 1;

/// Sample an outgoing CM-frame energy from a per-slot tabulated
/// continuous distribution with optional discrete head.
///
/// # Arguments
/// * `e_in` -- incident energy.
/// * `energy_grid` -- sorted per-slot incident-energy grid, length `n_e`.
/// * `n_x_per_i` -- number of `(x, p, c)` points at each
///   incident-energy bracket, length `n_e`.
/// * `interp_per_i` -- interpolation flag per bracket, length `n_e`.
/// * `n_discrete_per_i` -- number of discrete points at each bracket
///   (followed by `n_x − n_discrete` continuous points), length `n_e`.
/// * `histogram_outer` -- global flag (`true` ≙ kernel's `u32 == 1`)
///   that disables the stochastic bracket pick AND the bracket-bound
///   stretch. The kernel always draws the bracket-pick `xi_eeb`
///   regardless of this flag, so callers don't need to special-case
///   it to keep RNG schedules aligned.
/// * `x_table`, `p_table`, `c_table` -- flat tables packed
///   variable-length (issue #104): row `i` occupies `x_offset[i] ..
///   x_offset[i] + n_x_per_i[i]`. Rows are stored back-to-back with no
///   padding.
/// * `x_offset` -- start index of each incident-energy row in the flat
///   `x_table` / `p_table` / `c_table`, length `n_e`.
/// * `state` -- inline PCG RNG state. Advanced 0× when the slot is
///   empty (`n_e == 0`), 1× when the chosen bracket has `n_x < 2`
///   (`xi_eeb` is still drawn), 2× on the happy path
///   (`xi_eeb` + `xi_x`).
///
/// # Returns
/// `Some(e_out)` (clamped to ≥ 0) on success; `None` when the slot
/// is empty or the chosen bracket has fewer than 2 `(x, c)` points.
#[allow(clippy::too_many_arguments)]
pub fn sample_tabulated_continuous_eout(
    e_in: f64,
    energy_grid: &[f64],
    n_x_per_i: &[u32],
    interp_per_i: &[u32],
    n_discrete_per_i: &[u32],
    histogram_outer: bool,
    x_table: &[f64],
    p_table: &[f64],
    c_table: &[f64],
    x_offset: &[u32],
    state: &mut u64,
) -> Option<f64> {
    let n_e = energy_grid.len();
    if n_e == 0 {
        return None;
    }
    let (i_eb, r_eb) = locate_bracket(e_in, energy_grid);

    // Always draw -- the kernel does, so the RNG schedule must too.
    let xi_eeb = next_xi(state);
    let bin_e = if !histogram_outer && r_eb > xi_eeb && i_eb + 1 < n_e {
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

    let xi_x = next_xi(state);

    // Discrete-then-continuous CDF search, matching the CPU reference
    // `sample_with_discrete_info` (issue #103). The discrete head (the first
    // `n_disc` points) is searched with `xi < c[k]` and selects the exact
    // discrete line; the continuous tail is searched from `n_disc` with
    // `xi <= c[k+1]`, carrying `c_j` across iterations as the CPU does. The
    // previous single unified scan collapsed every discrete line onto index 0
    // and misattributed the first continuous bin's mass to the delta at x[0].
    // For `n_disc == 0` (no discrete head -- the common case) this is
    // identical to the old continuous-only scan (`c_j == c[j]`), so
    // non-discrete distributions are unchanged.
    let mut j = 0usize;
    let mut c_j = c_table[x_off];
    let mut is_discrete = false;
    let mut done = false;
    for k in 0..n_disc {
        if !done {
            j = k;
            c_j = c_table[x_off + k];
            if xi_x < c_j {
                is_discrete = true;
                done = true;
            }
        }
    }
    for k in n_disc..n_x - 1 {
        if !done {
            j = k;
            let c_k1 = c_table[x_off + k + 1];
            if xi_x <= c_k1 {
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
                x_j + (xi_x - c_j) / p_j
            } else {
                x_j
            }
        } else {
            let disc = (p_j * p_j + 2.0 * m * (xi_x - c_j)).max(0.0);
            x_j + (disc.sqrt() - p_j) / m
        }
    } else if !is_discrete && p_j > 0.0 {
        x_j + (xi_x - c_j) / p_j
    } else {
        x_j
    };

    // Bracket-bound stretch -- skipped when histogram_outer is set
    // OR the sampled point is in the discrete head.
    if !histogram_outer && !is_discrete && n_e >= 2 {
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
    Some(e_sampled)
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
        let mut state: u64 = 0xC0FFEE;
        let before = state;
        let r = sample_tabulated_continuous_eout(
            5.0e6, &empty_f, &empty_u, &empty_u, &empty_u, false, &empty_f, &empty_f, &empty_f,
            &empty_u, &mut state,
        );
        assert!(r.is_none());
        assert_eq!(state, before);
    }

    /// Return type of [`build_continuous`]: (energy_grid, n_x_per_i,
    /// n_discrete_per_i, interp_per_i, x_table, p_table, c_table, max_x).
    type ContinuousFixture = (
        [f64; 2],
        [u32; 2],
        [u32; 2],
        [u32; 2],
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        usize,
    );

    /// Build a 2-bracket continuous-only synthetic with uniform CDFs
    /// over `[1e5, 2e6]` (bracket 0) and `[2e5, 3e6]` (bracket 1).
    /// At mid-incident-energy the stretched bounds are `[1.5e5, 2.5e6]`.
    fn build_continuous() -> ContinuousFixture {
        let energy_grid = [1.0e3, 1.0e7];
        let n_x_per_i = [4u32, 4u32];
        let interp_per_i = [INTERP_LINLIN, INTERP_LINLIN];
        let n_discrete_per_i = [0u32, 0u32];
        let max_x: usize = 8;
        let mut x_table = vec![0.0; 2 * max_x];
        let mut p_table = vec![0.0; 2 * max_x];
        let mut c_table = vec![0.0; 2 * max_x];
        let xs0 = [1.0e5, 7.0e5, 1.4e6, 2.0e6];
        let cs0 = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
        let xs1 = [2.0e5, 8.0e5, 2.2e6, 3.0e6];
        let cs1 = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
        x_table[..4].copy_from_slice(&xs0);
        c_table[..4].copy_from_slice(&cs0);
        x_table[max_x..max_x + 4].copy_from_slice(&xs1);
        c_table[max_x..max_x + 4].copy_from_slice(&cs1);
        for j in 0..3 {
            p_table[j] = (cs0[j + 1] - cs0[j]) / (xs0[j + 1] - xs0[j]);
            p_table[max_x + j] = (cs1[j + 1] - cs1[j]) / (xs1[j + 1] - xs1[j]);
        }
        (
            energy_grid,
            n_x_per_i,
            interp_per_i,
            n_discrete_per_i,
            x_table,
            p_table,
            c_table,
            max_x,
        )
    }

    /// With `histogram_outer = false`, the stretched bounds hold and
    /// the empirical mean lies near the midpoint.
    #[test]
    fn continuous_samples_within_stretched_bounds() {
        let (e_grid, n_x, interp, n_disc, x_t, p_t, c_t, max_x) = build_continuous();
        let x_off = [0u32, max_x as u32];
        let mut state: u64 = 0x1234_5678;
        let n = 4_000usize;
        let lo = 1.5e5;
        let hi = 2.5e6;
        let mut sum = 0.0;
        for _ in 0..n {
            let e = sample_tabulated_continuous_eout(
                5.0005e6, &e_grid, &n_x, &interp, &n_disc, false, &x_t, &p_t, &c_t, &x_off,
                &mut state,
            )
            .unwrap();
            assert!((lo..=hi).contains(&e), "e_out {e} outside [{lo}, {hi}]");
            sum += e;
        }
        let mean = sum / n as f64;
        let mid = 0.5 * (lo + hi);
        assert!(
            ((mean - mid) / mid).abs() < 0.10,
            "mean {mean:.4e} should be near midpoint {mid:.4e}"
        );
    }

    /// `histogram_outer = true` suppresses the bracket pick and the
    /// stretch -- the samples must stay within the i_eb bracket's own
    /// bounds, not the stretched interpolated bounds.
    #[test]
    fn histogram_outer_suppresses_stretch() {
        let (e_grid, n_x, interp, n_disc, x_t, p_t, c_t, max_x) = build_continuous();
        let x_off = [0u32, max_x as u32];
        let mut state: u64 = 0xC1A551C5;
        // At E_in = 5e6 we land in bracket 0 (i_eb = 0). With
        // hist_outer = true, bin_e stays at i_eb and no stretch
        // happens -- samples must be in bracket 0's own range
        // [1e5, 2e6].
        for _ in 0..200 {
            let e = sample_tabulated_continuous_eout(
                5.0e6, &e_grid, &n_x, &interp, &n_disc, true, &x_t, &p_t, &c_t, &x_off, &mut state,
            )
            .unwrap();
            assert!(
                (1.0e5..=2.0e6).contains(&e),
                "hist_outer should keep e_out in bracket-0 range, got {e}"
            );
        }
    }

    /// Parity vs the CPU reference `TabulatedProbability::sample_with_discrete_info`
    /// over a MULTI-line discrete head (issue #103). The pre-fix unified scan
    /// collapsed every discrete line onto index 0, so line 1+ were never
    /// selected and the first continuous bin's mass was misattributed to the
    /// delta at x[0]. With `histogram_outer = true` the flat sampler reduces to
    /// the within-bracket CDF inversion, so it must reproduce the CPU's
    /// per-discrete-line probabilities and continuous spectrum (statistically;
    /// the two use different RNGs).
    #[test]
    fn discrete_head_matches_cpu_sample_with_discrete_info() {
        use rand::{rngs::StdRng, SeedableRng};
        use yamc_nuclide::reaction_product::{TabulatedInterp, TabulatedProbability};

        // 2 discrete lines (1, 2 MeV) + 3-point continuous tail (3, 5, 10 MeV).
        // CDF: line0 owns [0,0.2), line1 [0.2,0.5); continuous starts at 0.5
        // (c[n_disc] == c[n_disc-1], the well-formed convention).
        let x = vec![1.0e6, 2.0e6, 3.0e6, 5.0e6, 1.0e7];
        let c = vec![0.2, 0.5, 0.5, 0.75, 1.0];
        let n_discrete = 2usize;
        // Continuous histogram PDF on the tail; discrete entries unused for
        // selection (CPU returns the exact x[k]).
        let mut p = vec![0.0; 5];
        for k in n_discrete..4 {
            p[k] = (c[k + 1] - c[k]) / (x[k + 1] - x[k]);
        }

        const N: usize = 400_000;
        let cpu_dist = TabulatedProbability::Tabulated {
            x: x.clone(),
            p: p.clone(),
            c: c.clone(),
            interp: TabulatedInterp::Histogram,
            n_discrete,
        };
        let mut rng = StdRng::seed_from_u64(0x103_0103);
        let cpu: Vec<f64> = (0..N)
            .map(|_| cpu_dist.sample_with_discrete_info(&mut rng).0)
            .collect();

        // Flat buffers: single bracket exercised via histogram_outer = true.
        let max_x = 8usize;
        let energy_grid = [1.0e3, 1.0e7];
        let n_x_per_i = [5u32, 5u32];
        let interp_per_i = [INTERP_HISTOGRAM, INTERP_HISTOGRAM];
        let n_discrete_per_i = [n_discrete as u32, n_discrete as u32];
        let mut x_table = vec![0.0; 2 * max_x];
        let mut p_table = vec![0.0; 2 * max_x];
        let mut c_table = vec![0.0; 2 * max_x];
        for row in 0..2 {
            let off = row * max_x;
            x_table[off..off + 5].copy_from_slice(&x);
            p_table[off..off + 5].copy_from_slice(&p);
            c_table[off..off + 5].copy_from_slice(&c);
        }
        let x_off = [0u32, max_x as u32];
        let mut state: u64 = 0x0103_5EED;
        let flat: Vec<f64> = (0..N)
            .map(|_| {
                sample_tabulated_continuous_eout(
                    5.0e6,
                    &energy_grid,
                    &n_x_per_i,
                    &interp_per_i,
                    &n_discrete_per_i,
                    true, // histogram_outer: suppress bracket pick + stretch
                    &x_table,
                    &p_table,
                    &c_table,
                    &x_off,
                    &mut state,
                )
                .unwrap()
            })
            .collect();

        // Per-discrete-line selection probability, both paths.
        let frac = |v: &[f64], e: f64| {
            v.iter().filter(|&&x| (x - e).abs() < 1.0).count() as f64 / v.len() as f64
        };
        let (c0, f0) = (frac(&cpu, 1.0e6), frac(&flat, 1.0e6));
        let (c1, f1) = (frac(&cpu, 2.0e6), frac(&flat, 2.0e6));
        eprintln!("discrete line0: cpu {c0:.4} flat {f0:.4} | line1: cpu {c1:.4} flat {f1:.4}");
        // Both lines must be selected near their true weights (0.2, 0.3). The
        // pre-fix flat gave line0 ~= 0.5 and line1 ~= 0.0 -- this is the bug.
        assert!(
            (c0 - 0.2).abs() < 0.01 && (f0 - 0.2).abs() < 0.01,
            "line0 prob off"
        );
        assert!(
            (c1 - 0.3).abs() < 0.01 && (f1 - 0.3).abs() < 0.01,
            "line1 prob off"
        );
        assert!((c0 - f0).abs() < 0.01, "line0 cpu-vs-flat differ");
        assert!((c1 - f1).abs() < 0.01, "line1 cpu-vs-flat differ");

        // Continuous-tail mean (samples not on a discrete line) must agree.
        let cont_mean = |v: &[f64]| {
            let s: Vec<f64> = v
                .iter()
                .copied()
                .filter(|&x| (x - 1.0e6).abs() >= 1.0 && (x - 2.0e6).abs() >= 1.0)
                .collect();
            s.iter().sum::<f64>() / s.len() as f64
        };
        let (cm, fm) = (cont_mean(&cpu), cont_mean(&flat));
        eprintln!("continuous mean: cpu {cm:.4e} flat {fm:.4e}");
        assert!(
            ((cm - fm) / cm).abs() < 0.02,
            "continuous-tail mean cpu {cm:.4e} vs flat {fm:.4e} differ >2%"
        );
    }

    /// Discrete head: a single discrete line at `x = 1.5e6` samples
    /// exactly that energy, with no stretch.
    #[test]
    fn discrete_line_samples_exactly() {
        let energy_grid = [1.0e3, 1.0e7];
        let n_x_per_i = [2u32, 2u32];
        let interp_per_i = [INTERP_HISTOGRAM, INTERP_HISTOGRAM];
        let n_discrete_per_i = [1u32, 1u32];
        let max_x: usize = 4;
        let mut x_table = vec![0.0; 2 * max_x];
        let mut p_table = vec![0.0; 2 * max_x];
        let mut c_table = vec![0.0; 2 * max_x];
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
            let e = sample_tabulated_continuous_eout(
                5.0e6,
                &energy_grid,
                &n_x_per_i,
                &interp_per_i,
                &n_discrete_per_i,
                false,
                &x_table,
                &p_table,
                &c_table,
                &x_off,
                &mut state,
            )
            .unwrap();
            assert!(
                (e - 1.5e6).abs() < 1.0,
                "discrete line should sample 1.5e6 exactly, got {e}"
            );
        }
    }
}
